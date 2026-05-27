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


# ── Code-snippet diff (AST-based + symbolic equivalence) ──

import ast as _ast


@dataclass
class NodeChange:
    change_type: str  # "added" | "removed" | "modified" | "moved"
    node_name: str
    old_value: Optional[str]
    new_value: Optional[str]


@dataclass
class ASTDiffResult:
    additions: list[str]
    deletions: list[str]
    moved_blocks: list[tuple[str, str]]
    structural_similarity: float
    node_changes: list[NodeChange]


def _extract_top_level_names(tree: _ast.Module) -> dict[str, _ast.AST]:
    """Return mapping of name → node for top-level functions and classes."""
    result: dict[str, _ast.AST] = {}
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            result[node.name] = node
    return result


def _node_signature(node: _ast.AST) -> str:
    """Produce a stable string fingerprint for a function/class node."""
    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
        args = [a.arg for a in node.args.args]
        return f"{node.name}({','.join(args)})"
    if isinstance(node, _ast.ClassDef):
        bases = [_ast.unparse(b) for b in node.bases]
        return f"{node.name}({','.join(bases)})"
    return _ast.dump(node)


class ASTDiffer:
    def diff(self, code_a: str, code_b: str, lang: str = "python") -> ASTDiffResult:
        try:
            tree_a = _ast.parse(code_a)
        except SyntaxError:
            tree_a = None
        try:
            tree_b = _ast.parse(code_b)
        except SyntaxError:
            tree_b = None

        if tree_a is None or tree_b is None:
            return ASTDiffResult(
                additions=[], deletions=[], moved_blocks=[],
                structural_similarity=0.0, node_changes=[],
            )

        dump_a = _ast.dump(tree_a)
        dump_b = _ast.dump(tree_b)
        structural_similarity = difflib.SequenceMatcher(None, dump_a, dump_b).ratio()

        names_a = _extract_top_level_names(tree_a)
        names_b = _extract_top_level_names(tree_b)

        set_a = set(names_a)
        set_b = set(names_b)

        additions = sorted(set_b - set_a)
        deletions = sorted(set_a - set_b)

        node_changes: list[NodeChange] = []

        for name in additions:
            node_changes.append(NodeChange("added", name, None, _node_signature(names_b[name])))
        for name in deletions:
            node_changes.append(NodeChange("removed", name, _node_signature(names_a[name]), None))

        for name in set_a & set_b:
            sig_a = _node_signature(names_a[name])
            sig_b = _node_signature(names_b[name])
            if sig_a != sig_b:
                node_changes.append(NodeChange("modified", name, sig_a, sig_b))
            else:
                body_a = _ast.dump(names_a[name])
                body_b = _ast.dump(names_b[name])
                if body_a != body_b:
                    node_changes.append(NodeChange("modified", name, sig_a, sig_b))

        return ASTDiffResult(
            additions=additions,
            deletions=deletions,
            moved_blocks=[],
            structural_similarity=structural_similarity,
            node_changes=node_changes,
        )


# ── Symbolic equivalence checker ──

import random
import types as _types
from dataclasses import dataclass as _dc


@dataclass
class EquivalenceResult:
    is_equivalent: bool
    confidence: float
    counterexample: Optional[dict]


_SAFE_BUILTINS = {
    "range": range, "len": len, "abs": abs, "min": min, "max": max,
    "sum": sum, "sorted": sorted, "enumerate": enumerate, "zip": zip,
    "int": int, "float": float, "str": str, "bool": bool, "list": list,
    "tuple": tuple, "dict": dict, "set": set, "None": None,
    "True": True, "False": False,
}


def _extract_func_name(source: str) -> Optional[str]:
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return None
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            return node.name
    return None


def _load_func(source: str) -> Optional[object]:
    name = _extract_func_name(source)
    if name is None:
        return None
    ns: dict = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(source, "<string>", "exec"), ns)  # noqa: S102
    except Exception:
        return None
    return ns.get(name)


def _count_params(source: str) -> int:
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return 0
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            return len(node.args.args)
    return 0


class SymbolicEquivalenceChecker:
    def are_equivalent(self, func_a: str, func_b: str) -> EquivalenceResult:
        fn_a = _load_func(func_a)
        fn_b = _load_func(func_b)

        if fn_a is None or fn_b is None:
            return EquivalenceResult(is_equivalent=False, confidence=0.0, counterexample=None)

        n_params = _count_params(func_a)
        n_params_b = _count_params(func_b)
        if n_params != n_params_b:
            return EquivalenceResult(is_equivalent=False, confidence=1.0, counterexample=None)

        tested = 0
        for _ in range(50):
            inputs = tuple(random.randint(-100, 100) for _ in range(n_params))
            try:
                out_a = fn_a(*inputs)
            except Exception as exc_a:
                try:
                    fn_b(*inputs)
                except Exception as exc_b:
                    if type(exc_a) is type(exc_b):
                        tested += 1
                        continue
                    return EquivalenceResult(
                        is_equivalent=False, confidence=1.0,
                        counterexample={"inputs": inputs, "output_a": repr(exc_a), "output_b": repr(exc_b)},
                    )
                continue
            try:
                out_b = fn_b(*inputs)
            except Exception:
                continue

            if out_a != out_b:
                return EquivalenceResult(
                    is_equivalent=False, confidence=1.0,
                    counterexample={"inputs": inputs, "output_a": out_a, "output_b": out_b},
                )
            tested += 1

        if tested == 0:
            return EquivalenceResult(is_equivalent=False, confidence=0.0, counterexample=None)

        confidence = min(0.9, 0.5 + tested * 0.008)
        return EquivalenceResult(is_equivalent=True, confidence=confidence, counterexample=None)
