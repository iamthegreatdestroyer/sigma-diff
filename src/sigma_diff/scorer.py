"""Semantic scoring — weight changes by impact level + embedding-based similarity."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sigma_diff.differ import ASTDiffResult, FileChange, StructuralElement


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


# ── Embedding-based scorer ──

@dataclass
class DiffScore:
    structural: float
    semantic: float
    combined: float
    risk_level: str  # "low" | "medium" | "high"


def _tfidf_similarity(code_a: str, code_b: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _cosine
        import numpy as np  # noqa: F401 — imported for sklearn internals

        vec = TfidfVectorizer().fit_transform([code_a, code_b])
        return float(_cosine(vec[0], vec[1])[0][0])
    except Exception:
        # Last-resort: simple token overlap (Jaccard)
        toks_a = set(code_a.split())
        toks_b = set(code_b.split())
        if not toks_a and not toks_b:
            return 1.0
        if not toks_a or not toks_b:
            return 0.0
        return len(toks_a & toks_b) / len(toks_a | toks_b)


class EmbeddingScorer:
    def __init__(self, ryzanstein_url: str | None = None) -> None:
        self._url = ryzanstein_url or os.getenv("RYZANSTEIN_URL")

    def _get_embedding(self, code: str) -> list[float]:
        import urllib.request, json  # noqa: E401

        payload = json.dumps({"input": code}).encode()
        req = urllib.request.Request(
            f"{self._url}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return data["data"][0]["embedding"]

    def semantic_similarity(self, code_a: str, code_b: str) -> float:
        if self._url:
            try:
                import numpy as np

                emb_a = self._get_embedding(code_a)
                emb_b = self._get_embedding(code_b)
                va = np.array(emb_a, dtype=float)
                vb = np.array(emb_b, dtype=float)
                norm = np.linalg.norm(va) * np.linalg.norm(vb)
                if norm == 0:
                    return 0.0
                return float(np.dot(va, vb) / norm)
            except Exception:
                pass
        return _tfidf_similarity(code_a, code_b)

    def semantic_distance(self, code_a: str, code_b: str) -> float:
        return 1.0 - self.semantic_similarity(code_a, code_b)

    def score(self, diff_result: "ASTDiffResult", code_a: str, code_b: str) -> DiffScore:
        structural = diff_result.structural_similarity
        semantic = self.semantic_similarity(code_a, code_b)
        combined = 0.7 * structural + 0.3 * semantic
        if combined < 0.5:
            risk = "high"
        elif combined < 0.8:
            risk = "medium"
        else:
            risk = "low"
        return DiffScore(structural=structural, semantic=semantic, combined=combined, risk_level=risk)
