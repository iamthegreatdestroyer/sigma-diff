"""Semantic scoring — weight changes by impact level."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sigma_diff.differ import FileChange, StructuralElement


class ImpactLevel(Enum):
    """Impact classification for changes."""

    CRITICAL = "critical"  # API signature changes, breaking changes
    HIGH = "high"  # Function/class body changes
    MEDIUM = "medium"  # Import, variable changes
    LOW = "low"  # Comments, whitespace, string literals
    NONE = "none"


# Weight map for structural elements
_ELEMENT_WEIGHTS: dict[str, float] = {
    "function": 1.0,
    "class": 1.0,
    "method": 0.9,
    "import": 0.5,
    "variable": 0.6,
    "comment": 0.1,
    "whitespace": 0.0,
    "string_literal": 0.2,
    "other": 0.4,
}

# Multipliers for change types
_CHANGE_TYPE_MULTIPLIERS: dict[str, float] = {
    "added": 0.8,
    "deleted": 0.9,
    "modified": 1.0,
    "renamed": 0.3,
    "moved": 0.2,
}


def element_impact(element_name: str) -> ImpactLevel:
    """Classify a structural element by its impact level."""
    weight = _ELEMENT_WEIGHTS.get(element_name, 0.4)
    if weight >= 0.9:
        return ImpactLevel.CRITICAL
    if weight >= 0.6:
        return ImpactLevel.HIGH
    if weight >= 0.4:
        return ImpactLevel.MEDIUM
    return ImpactLevel.LOW


def compute_semantic_score(files: list[FileChange]) -> float:
    """Compute a 0.0-1.0 semantic impact score for a set of file changes.

    Higher scores indicate more impactful changes (API modifications,
    structural changes). Lower scores indicate cosmetic changes.

    The score is computed as a weighted average of:
    - Structural element weights (functions > imports > comments)
    - Change type multipliers (modified > deleted > added > renamed)
    - Line count normalization
    """
    if not files:
        return 0.0

    total_weight = 0.0
    total_lines = 0

    for fc in files:
        change_mult = _CHANGE_TYPE_MULTIPLIERS.get(fc.change_type.value, 0.5)
        file_lines = fc.additions + fc.deletions

        if not fc.structural_elements:
            # No structural classification available — use base weight
            total_weight += file_lines * 0.4 * change_mult
        else:
            for elem, count in fc.structural_elements.items():
                elem_weight = _ELEMENT_WEIGHTS.get(elem.value, 0.4)
                total_weight += count * elem_weight * change_mult

        total_lines += file_lines

    if total_lines == 0:
        return 0.0

    # Normalize to 0-1 range using sigmoid-like function
    raw = total_weight / total_lines
    # Clamp to [0, 1]
    return min(1.0, max(0.0, raw))
