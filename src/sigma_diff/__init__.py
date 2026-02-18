"""sigma-diff: Semantic diff engine with structural comparison and change classification."""

from sigma_diff.differ import SemanticDiffer, DiffResult, ChangeType, FileChange
from sigma_diff.scorer import compute_semantic_score, ImpactLevel
from sigma_diff.summary import summarize_diff

__all__ = [
    "SemanticDiffer",
    "DiffResult",
    "ChangeType",
    "FileChange",
    "compute_semantic_score",
    "ImpactLevel",
    "summarize_diff",
]
