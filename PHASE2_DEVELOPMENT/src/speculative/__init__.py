"""
Speculative decoding for fast generation.
"""

from .decoder import (
    SpeculativeDecoder,
    DraftModel,
    SpeculativeVerifier,
    AdaptiveSpeculation,
    SpeculationConfig,
    SpeculativeOutput,
    create_speculative_decoder,
)

__all__ = [
    "SpeculativeDecoder",
    "DraftModel",
    "SpeculativeVerifier",
    "AdaptiveSpeculation",
    "SpeculationConfig",
    "SpeculativeOutput",
    "create_speculative_decoder",
]
