"""
Ryot LLM Core Type Definitions
==============================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray


class ModelType(Enum):
    BITNET = auto()
    MAMBA = auto()
    RWKV = auto()
    HYBRID = auto()


class InferenceMode(Enum):
    STANDARD = auto()
    SPECULATIVE = auto()
    BATCH = auto()
    COMPRESSED = auto()
    RECYCLED = auto()


class QuantizationType(Enum):
    TERNARY = auto()
    INT4 = auto()
    INT8 = auto()
    FP16 = auto()
    FP32 = auto()


class CacheStrategy(Enum):
    STANDARD = auto()
    SLIDING_WINDOW = auto()
    SIGMA_ANCHORED = auto()
    TIERED = auto()


class StopReason(Enum):
    MAX_TOKENS = auto()
    EOS_TOKEN = auto()
    STOP_SEQUENCE = auto()
    CANCELLED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class TokenSequence:
    tokens: Tuple[int, ...]
    attention_mask: Optional[Tuple[int, ...]] = None
    position_ids: Optional[Tuple[int, ...]] = None
    sigma_encoded: bool = False
    compression_ratio: Optional[float] = None

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_list(cls, tokens: List[int]) -> 'TokenSequence':
        return cls(tokens=tuple(tokens))

    def to_list(self) -> List[int]:
        return list(self.tokens)


@dataclass
class GenerationConfig:
    max_tokens: int = 256
    min_tokens: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    seed: Optional[int] = None
    inference_mode: InferenceMode = InferenceMode.STANDARD
    use_sigma_compression: bool = False
    sigma_compression_level: int = 1
    cache_strategy: CacheStrategy = CacheStrategy.STANDARD
    reuse_cache_id: Optional[str] = None


@dataclass
class GenerationResult:
    generated_tokens: TokenSequence
    generated_text: str
    stop_reason: StopReason
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time_ms: float
    tokens_per_second: float
    compression_ratio: Optional[float] = None
    effective_context_size: Optional[int] = None
    cache_hit: bool = False
    cache_reuse_ratio: Optional[float] = None
    rsu_reference: Optional[str] = None
    cache_id: Optional[str] = None


@dataclass
class StreamChunk:
    token_id: int
    token_text: str
    position: int
    is_first: bool = False
    is_last: bool = False
    final_result: Optional[GenerationResult] = None


@dataclass
class ModelInfo:
    model_id: str
    model_type: ModelType
    parameter_count: int
    context_window: int
    vocab_size: int
    quantization: QuantizationType
    bits_per_weight: float
    model_size_bytes: int
    estimated_memory_mb: float
    supports_streaming: bool = True
    supports_batching: bool = True
    supports_sigma_compression: bool = False
    supports_kv_recycling: bool = False
    estimated_tokens_per_second: float = 0.0


@dataclass
class KVCacheState:
    cache_id: str
    model_id: str
    num_layers: int
    num_heads: int
    head_dim: int
    sequence_length: int
    key_states: List[NDArray[np.float32]]
    value_states: List[NDArray[np.float32]]
    anchor_positions: Optional[List[int]] = None
    anchor_semantic_hashes: Optional[List[int]] = None
    created_timestamp: float = 0.0
    access_count: int = 0

    def size_bytes(self) -> int:
        total = 0
        for k, v in zip(self.key_states, self.value_states):
            total += k.nbytes + v.nbytes
        return total


@dataclass
class SigmaEncodedContext:
    glyph_sequence: bytes
    original_token_count: int
    compressed_glyph_count: int
    decompressed_tokens: Optional[TokenSequence] = None
    compression_ratio: float = 1.0
    semantic_hash: int = 0
    is_delta_encoded: bool = False
    parent_rsu_reference: Optional[str] = None


@dataclass
class RSUReference:
    rsu_id: str
    semantic_hash: int
    storage_tier: str
    storage_path: Optional[str] = None
    chain_depth: int = 0
    parent_reference: Optional['RSUReference'] = None
    has_kv_cache: bool = False
    kv_cache_id: Optional[str] = None


@dataclass
class AgentRequest:
    agent_id: str
    agent_role: str
    prompt: str
    config: GenerationConfig
    conversation_id: str
    task_id: str
    sigma_context: Optional[SigmaEncodedContext] = None
    priority: int = 5
    max_latency_ms: Optional[int] = None


@dataclass
class AgentResponse:
    agent_id: str
    task_id: str
    result: GenerationResult
    should_continue: bool = False
    next_agent_hint: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class InferenceError:
    error_code: str
    error_message: str
    model_id: Optional[str] = None
    request_id: Optional[str] = None
    is_retryable: bool = False
    suggested_action: Optional[str] = None
    stack_trace: Optional[str] = None
