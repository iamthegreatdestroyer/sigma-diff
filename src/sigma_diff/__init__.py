"""sigma-diff: Semantic diff engine with structural comparison and change classification."""

from sigma_diff.differ import (
    SemanticDiffer,
    DiffResult,
    ChangeType,
    FileChange,
    ASTDiffer,
    ASTDiffResult,
    NodeChange,
    SymbolicEquivalenceChecker,
    EquivalenceResult,
)
from sigma_diff.scorer import compute_semantic_score, ImpactLevel, EmbeddingScorer, DiffScore
from sigma_diff.summary import summarize_diff, DiffSummarizer

__all__ = [
    "SemanticDiffer",
    "DiffResult",
    "ChangeType",
    "FileChange",
    "ASTDiffer",
    "ASTDiffResult",
    "NodeChange",
    "SymbolicEquivalenceChecker",
    "EquivalenceResult",
    "compute_semantic_score",
    "ImpactLevel",
    "EmbeddingScorer",
    "DiffScore",
    "summarize_diff",
    "DiffSummarizer",
    "SigmaDiff",
]


class SigmaDiff:
    """Top-level pipeline: AST diff → score → summary."""

    def __init__(self, ryzanstein_url: str | None = None) -> None:
        self._differ = ASTDiffer()
        self._scorer = EmbeddingScorer(ryzanstein_url)
        self._summarizer = DiffSummarizer()

    def diff(self, code_a: str, code_b: str) -> ASTDiffResult:
        return self._differ.diff(code_a, code_b)

    def score(self, diff_result: ASTDiffResult, code_a: str, code_b: str) -> DiffScore:
        return self._scorer.score(diff_result, code_a, code_b)

    def analyze(self, code_a: str, code_b: str) -> tuple[ASTDiffResult, DiffScore, str]:
        dr = self.diff(code_a, code_b)
        sc = self.score(dr, code_a, code_b)
        summary = self._summarizer.summarize(dr, sc)
        return dr, sc, summary
