"""Human-readable change summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sigma_diff.differ import ASTDiffResult, DiffResult, FileChange
    from sigma_diff.scorer import DiffScore

from sigma_diff.scorer import ImpactLevel, element_impact


def summarize_diff(result: DiffResult) -> str:
    """Generate a human-readable summary of a DiffResult.

    Example output:
        3 files changed (+42, -18), semantic impact: 0.82 (high)
        - Modified src/engine.py: function changes (critical)
        - Added tests/test_engine.py: 24 lines
        - Deleted old_module.py: class removal (critical)
    """
    lines: list[str] = []

    # Header
    impact_label = _score_label(result.semantic_score)
    lines.append(
        f"{result.file_count} file(s) changed "
        f"(+{result.total_additions}, -{result.total_deletions}), "
        f"semantic impact: {result.semantic_score:.2f} ({impact_label})"
    )

    # Per-file summaries
    for fc in result.files:
        line = f"  - {fc.change_type.value.capitalize()} {fc.path}"
        if fc.old_path and fc.old_path != fc.path:
            line += f" (from {fc.old_path})"

        details = _file_detail(fc)
        if details:
            line += f": {details}"
        else:
            line += f": +{fc.additions}, -{fc.deletions}"

        lines.append(line)

    return "\n".join(lines)


def _file_detail(fc: FileChange) -> str:
    """Generate a short detail string for a file change."""
    if not fc.structural_elements:
        return ""

    parts: list[str] = []
    for elem, count in sorted(fc.structural_elements.items(), key=lambda x: -x[1]):
        impact = element_impact(elem.value)
        tag = f" ({impact.value})" if impact in (ImpactLevel.CRITICAL, ImpactLevel.HIGH) else ""
        parts.append(f"{elem.value} x{count}{tag}")

    return ", ".join(parts[:3])  # Top 3 element types


def _score_label(score: float) -> str:
    """Convert a numeric score to a human label."""
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    if score >= 0.2:
        return "low"
    return "minimal"


class DiffSummarizer:
    def summarize(self, diff: "ASTDiffResult", score: "DiffScore") -> str:
        parts: list[str] = []

        if score.risk_level == "high":
            lead = "⚠️  High-risk diff detected."
        elif score.risk_level == "medium":
            lead = "Moderate changes detected."
        else:
            lead = "No significant behavioral changes."
        parts.append(lead)

        n_modified = len(diff.node_changes)
        if n_modified:
            parts.append(
                f"{n_modified} node change(s) ({score.risk_level} risk). "
                f"Structural similarity: {score.structural:.0%}. "
                f"Semantic distance: {score.semantic_distance if hasattr(score, 'semantic_distance') else 1.0 - score.semantic:.2f}."
            )
        else:
            parts.append(
                f"Structural similarity: {score.structural:.0%}. "
                f"Semantic distance: {(1.0 - score.semantic):.2f}."
            )

        if diff.additions:
            parts.append(f"Added: [{', '.join(diff.additions)}].")
        if diff.deletions:
            parts.append(f"Removed: [{', '.join(diff.deletions)}].")

        return " ".join(parts)
