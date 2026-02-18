"""Core semantic diff engine — structural comparison with change classification."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ChangeType(Enum):
    """Classification of a code change."""

    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"
    RENAMED = "renamed"
    MOVED = "moved"


class StructuralElement(Enum):
    """Structural elements that can change in source code."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT = "import"
    VARIABLE = "variable"
    COMMENT = "comment"
    WHITESPACE = "whitespace"
    STRING_LITERAL = "string_literal"
    OTHER = "other"


@dataclass
class Hunk:
    """A contiguous block of changes."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]
    structural_type: StructuralElement = StructuralElement.OTHER


@dataclass
class FileChange:
    """Semantic description of changes to a single file."""

    path: str
    change_type: ChangeType
    hunks: list[Hunk] = field(default_factory=list)
    old_path: Optional[str] = None  # For renames
    additions: int = 0
    deletions: int = 0
    structural_elements: dict[StructuralElement, int] = field(default_factory=dict)

    @property
    def is_api_change(self) -> bool:
        """True if the change affects public API (functions, classes, methods)."""
        api_elements = {
            StructuralElement.FUNCTION,
            StructuralElement.CLASS,
            StructuralElement.METHOD,
        }
        return bool(api_elements & set(self.structural_elements.keys()))


@dataclass
class DiffResult:
    """Complete result of a semantic diff operation."""

    files: list[FileChange]
    total_additions: int = 0
    total_deletions: int = 0
    semantic_score: float = 0.0

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def has_api_changes(self) -> bool:
        return any(f.is_api_change for f in self.files)


# ── Language-aware pattern matching ──

_PATTERNS: dict[str, dict[StructuralElement, re.Pattern]] = {
    ".py": {
        StructuralElement.FUNCTION: re.compile(r"^(\s*)def\s+(\w+)"),
        StructuralElement.CLASS: re.compile(r"^(\s*)class\s+(\w+)"),
        StructuralElement.IMPORT: re.compile(r"^(from\s+\S+\s+)?import\s+"),
        StructuralElement.COMMENT: re.compile(r"^\s*#"),
        StructuralElement.STRING_LITERAL: re.compile(r'^\s*["\']'),
    },
    ".rs": {
        StructuralElement.FUNCTION: re.compile(r"^\s*(pub\s+)?(async\s+)?fn\s+(\w+)"),
        StructuralElement.CLASS: re.compile(r"^\s*(pub\s+)?struct\s+(\w+)"),
        StructuralElement.IMPORT: re.compile(r"^\s*use\s+"),
        StructuralElement.COMMENT: re.compile(r"^\s*//"),
    },
    ".go": {
        StructuralElement.FUNCTION: re.compile(r"^func\s+"),
        StructuralElement.CLASS: re.compile(r"^type\s+\w+\s+struct"),
        StructuralElement.IMPORT: re.compile(r"^\s*import\s+"),
        StructuralElement.COMMENT: re.compile(r"^\s*//"),
    },
    ".ts": {
        StructuralElement.FUNCTION: re.compile(
            r"^\s*(export\s+)?(async\s+)?function\s+(\w+)"
        ),
        StructuralElement.CLASS: re.compile(r"^\s*(export\s+)?class\s+(\w+)"),
        StructuralElement.IMPORT: re.compile(r"^\s*import\s+"),
        StructuralElement.COMMENT: re.compile(r"^\s*//"),
    },
}


def _classify_line(line: str, ext: str) -> StructuralElement:
    """Classify a line of code by its structural role."""
    stripped = line.strip()
    if not stripped:
        return StructuralElement.WHITESPACE

    patterns = _PATTERNS.get(ext, _PATTERNS.get(".py", {}))
    for element, pattern in patterns.items():
        if pattern.match(stripped) or pattern.match(line):
            # Distinguish methods from functions in Python
            if element == StructuralElement.FUNCTION and ext == ".py":
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    return StructuralElement.METHOD
            return element

    return StructuralElement.OTHER


def _detect_rename(old_path: str, new_path: str, old_lines: list[str], new_lines: list[str]) -> bool:
    """Detect if two files are likely a rename (>70% content similarity)."""
    if old_path == new_path:
        return False
    ratio = difflib.SequenceMatcher(None, old_lines, new_lines).ratio()
    return ratio > 0.7


class SemanticDiffer:
    """Semantic diff engine that classifies changes structurally.

    Example:
        >>> differ = SemanticDiffer()
        >>> result = differ.diff_files("old.py", "new.py")
        >>> print(result.semantic_score)
        0.85
    """

    def diff_texts(
        self,
        old_text: str,
        new_text: str,
        path: str = "<unknown>",
        old_path: Optional[str] = None,
    ) -> FileChange:
        """Diff two text strings and return a classified FileChange."""
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        ext = Path(path).suffix

        # Detect rename
        change_type = ChangeType.MODIFIED
        if old_path and old_path != path:
            if _detect_rename(old_path, path, old_lines, new_lines):
                change_type = ChangeType.RENAMED

        if not old_lines and new_lines:
            change_type = ChangeType.ADDED
        elif old_lines and not new_lines:
            change_type = ChangeType.DELETED

        # Generate unified diff
        diff_lines = list(
            difflib.unified_diff(old_lines, new_lines, fromfile=old_path or path, tofile=path, lineterm="")
        )

        hunks = _parse_hunks(diff_lines, ext)

        additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

        # Aggregate structural elements
        structural: dict[StructuralElement, int] = {}
        for hunk in hunks:
            for line in hunk.lines:
                if line.startswith("+") or line.startswith("-"):
                    content = line[1:]
                    elem = _classify_line(content, ext)
                    if elem != StructuralElement.WHITESPACE:
                        structural[elem] = structural.get(elem, 0) + 1

        return FileChange(
            path=path,
            change_type=change_type,
            hunks=hunks,
            old_path=old_path,
            additions=additions,
            deletions=deletions,
            structural_elements=structural,
        )

    def diff_files(self, old_path: str, new_path: str) -> FileChange:
        """Diff two files on disk."""
        old_text = ""
        new_text = ""

        op = Path(old_path)
        np_ = Path(new_path)

        if op.exists():
            old_text = op.read_text(encoding="utf-8", errors="replace")
        if np_.exists():
            new_text = np_.read_text(encoding="utf-8", errors="replace")

        return self.diff_texts(
            old_text, new_text, path=new_path, old_path=old_path if old_path != new_path else None
        )

    def diff_directories(
        self,
        old_dir: str,
        new_dir: str,
        extensions: Optional[set[str]] = None,
    ) -> DiffResult:
        """Diff two directory trees and return a DiffResult."""
        old_root = Path(old_dir)
        new_root = Path(new_dir)

        old_files = _collect_files(old_root, extensions)
        new_files = _collect_files(new_root, extensions)

        old_rel = {str(f.relative_to(old_root)) for f in old_files}
        new_rel = {str(f.relative_to(new_root)) for f in new_files}

        changes: list[FileChange] = []
        total_add = 0
        total_del = 0

        # Added files
        for rel in sorted(new_rel - old_rel):
            new_text = (new_root / rel).read_text(encoding="utf-8", errors="replace")
            fc = self.diff_texts("", new_text, path=rel)
            fc.change_type = ChangeType.ADDED
            changes.append(fc)
            total_add += fc.additions

        # Deleted files
        for rel in sorted(old_rel - new_rel):
            old_text = (old_root / rel).read_text(encoding="utf-8", errors="replace")
            fc = self.diff_texts(old_text, "", path=rel)
            fc.change_type = ChangeType.DELETED
            changes.append(fc)
            total_del += fc.deletions

        # Modified files
        for rel in sorted(old_rel & new_rel):
            old_text = (old_root / rel).read_text(encoding="utf-8", errors="replace")
            new_text = (new_root / rel).read_text(encoding="utf-8", errors="replace")
            if old_text != new_text:
                fc = self.diff_texts(old_text, new_text, path=rel)
                changes.append(fc)
                total_add += fc.additions
                total_del += fc.deletions

        from sigma_diff.scorer import compute_semantic_score

        score = compute_semantic_score(changes)

        return DiffResult(
            files=changes,
            total_additions=total_add,
            total_deletions=total_del,
            semantic_score=score,
        )


def _parse_hunks(diff_lines: list[str], ext: str) -> list[Hunk]:
    """Parse unified diff output into Hunk objects."""
    hunks: list[Hunk] = []
    current_lines: list[str] = []
    old_start = new_start = old_count = new_count = 0

    hunk_header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    for line in diff_lines:
        m = hunk_header.match(line)
        if m:
            if current_lines:
                hunks.append(
                    Hunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        lines=current_lines,
                        structural_type=_dominant_element(current_lines, ext),
                    )
                )
            old_start = int(m.group(1))
            old_count = int(m.group(2) or "1")
            new_start = int(m.group(3))
            new_count = int(m.group(4) or "1")
            current_lines = []
        elif line.startswith(("+++", "---")):
            continue
        else:
            current_lines.append(line)

    if current_lines:
        hunks.append(
            Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=current_lines,
                structural_type=_dominant_element(current_lines, ext),
            )
        )

    return hunks


def _dominant_element(lines: list[str], ext: str) -> StructuralElement:
    """Find the most common structural element in a set of diff lines."""
    counts: dict[StructuralElement, int] = {}
    for line in lines:
        if line.startswith(("+", "-")):
            elem = _classify_line(line[1:], ext)
            if elem != StructuralElement.WHITESPACE:
                counts[elem] = counts.get(elem, 0) + 1
    if not counts:
        return StructuralElement.OTHER
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def _collect_files(root: Path, extensions: Optional[set[str]] = None) -> list[Path]:
    """Recursively collect files, optionally filtering by extension."""
    if not root.exists():
        return []
    files = []
    for f in root.rglob("*"):
        if f.is_file() and not any(p.startswith(".") for p in f.relative_to(root).parts):
            if extensions is None or f.suffix in extensions:
                files.append(f)
    return files
