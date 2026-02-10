"""
Ryot LLM Core Interface Protocols
=================================
"""

from __future__ import annotations
from abc import abstractmethod
from typing import AsyncIterator, Iterator, List, Optional, Protocol, runtime_checkable

from .types import (
    AgentRequest, AgentResponse, CacheStrategy, GenerationConfig,
    GenerationResult, KVCacheState, ModelInfo, ModelType,
    RSUReference, SigmaEncodedContext, StreamChunk, TokenSequence,
)


@runtime_checkable
class InferenceEngine(Protocol):
    """Core inference engine protocol - PRIMARY integration point."""

    @abstractmethod
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        ...

    @abstractmethod
    def generate_from_tokens(self, tokens: TokenSequence, config: Optional[GenerationConfig] = None) -> GenerationResult:
        ...

    @abstractmethod
    def stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[StreamChunk]:
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        ...

    @abstractmethod
    def get_context_window(self) -> int:
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Tokenizer protocol for text-to-token conversion."""

    @abstractmethod
    def encode(self, text: str) -> TokenSequence:
        ...

    @abstractmethod
    def decode(self, tokens: TokenSequence) -> str:
        ...

    @abstractmethod
    def decode_single(self, token_id: int) -> str:
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        ...

    @abstractmethod
    def get_special_tokens(self) -> dict[str, int]:
        ...


@runtime_checkable
class CacheManagerProtocol(Protocol):
    """KV cache management protocol."""

    @abstractmethod
    def get_current_length(self) -> int:
        ...

    @abstractmethod
    def get_max_length(self) -> int:
        ...

    @abstractmethod
    def export_state(self) -> KVCacheState:
        ...

    @abstractmethod
    def import_state(self, state: KVCacheState) -> bool:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def register_sigma_anchors(self, positions: List[int], semantic_hashes: List[int]) -> None:
        ...

    @abstractmethod
    def find_recyclable_range(self, semantic_hash: int, tolerance: float = 0.9) -> Optional[tuple[int, int]]:
        ...


@runtime_checkable
class CompressionEngineProtocol(Protocol):
    """ΣLANG compression engine protocol."""

    @abstractmethod
    def encode(self, tokens: TokenSequence, conversation_id: Optional[str] = None) -> SigmaEncodedContext:
        ...

    @abstractmethod
    def decode(self, encoded: SigmaEncodedContext) -> TokenSequence:
        ...

    @abstractmethod
    def get_compression_ratio(self) -> float:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


@runtime_checkable
class RSUManagerProtocol(Protocol):
    """RSU management protocol."""

    @abstractmethod
    def store(self, tokens: TokenSequence, kv_state: Optional[KVCacheState] = None, conversation_id: Optional[str] = None) -> RSUReference:
        ...

    @abstractmethod
    def retrieve(self, reference: RSUReference) -> tuple[TokenSequence, Optional[KVCacheState]]:
        ...

    @abstractmethod
    def find_matching_rsu(self, semantic_hash: int, similarity_threshold: float = 0.85) -> Optional[RSUReference]:
        ...

    @abstractmethod
    def get_chain(self, conversation_id: str) -> List[RSUReference]:
        ...

    @abstractmethod
    def get_statistics(self) -> dict:
        ...


@runtime_checkable
class AgentBridgeProtocol(Protocol):
    """Bridge protocol for Elite Agent Collective integration."""

    @abstractmethod
    async def process_agent_request(self, request: AgentRequest) -> AgentResponse:
        ...

    @abstractmethod
    async def stream_agent_response(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        ...

    @abstractmethod
    def get_agent_statistics(self, agent_id: Optional[str] = None) -> dict:
        ...


@runtime_checkable
class HTTPServerProtocol(Protocol):
    """OpenAI-compatible HTTP server protocol."""

    @abstractmethod
    def start(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def is_running(self) -> bool:
        ...

    @abstractmethod
    def get_endpoint_stats(self) -> dict:
        ...


@runtime_checkable
class EngineFactoryProtocol(Protocol):
    """Factory protocol for creating configured engine instances."""

    @abstractmethod
    def create(
        self,
        model_type: ModelType = ModelType.BITNET,
        model_path: Optional[str] = None,
        enable_sigma: bool = False,
        enable_rsu: bool = False,
        cache_strategy: CacheStrategy = CacheStrategy.STANDARD,
    ) -> InferenceEngine:
        ...

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        ...
