"""
Mock Inference Engine for Integration Testing
==============================================
"""

import time
import random
from typing import Iterator, List, Optional

from ..api.api_types import (
    GenerationConfig, GenerationResult, KVCacheState, ModelInfo,
    ModelType, QuantizationType, StopReason, StreamChunk, TokenSequence,
)
from ..api.interfaces import InferenceEngine, TokenizerProtocol, CacheManagerProtocol


class MockTokenizer(TokenizerProtocol):
    def __init__(self, vocab_size: int = 32000):
        self._vocab_size = vocab_size
        self._special_tokens = {"bos": 1, "eos": 2, "pad": 0, "unk": 3}

    def encode(self, text: str) -> TokenSequence:
        tokens = [hash(w) % self._vocab_size for w in text.split()]
        return TokenSequence.from_list(tokens)

    def decode(self, tokens: TokenSequence) -> str:
        return f"[decoded {len(tokens)} tokens]"

    def decode_single(self, token_id: int) -> str:
        return f"[tok_{token_id}]"

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_special_tokens(self) -> dict[str, int]:
        return self._special_tokens.copy()


class MockCacheManager(CacheManagerProtocol):
    def __init__(self, max_length: int = 4096):
        self._max_length = max_length
        self._current_length = 0
        self._anchors: List[tuple[int, int]] = []

    def get_current_length(self) -> int:
        return self._current_length

    def get_max_length(self) -> int:
        return self._max_length

    def export_state(self) -> KVCacheState:
        import numpy as np
        return KVCacheState(
            cache_id=f"mock_cache_{int(time.time())}",
            model_id="mock-model",
            num_layers=32, num_heads=32, head_dim=128,
            sequence_length=self._current_length,
            key_states=[np.zeros((1, 32, 128), dtype=np.float32) for _ in range(32)],
            value_states=[np.zeros((1, 32, 128), dtype=np.float32) for _ in range(32)],
            anchor_positions=[p for p, _ in self._anchors],
            anchor_semantic_hashes=[h for _, h in self._anchors],
            created_timestamp=time.time(),
        )

    def import_state(self, state: KVCacheState) -> bool:
        self._current_length = state.sequence_length
        if state.anchor_positions and state.anchor_semantic_hashes:
            self._anchors = list(zip(state.anchor_positions, state.anchor_semantic_hashes))
        return True

    def clear(self) -> None:
        self._current_length = 0
        self._anchors = []

    def register_sigma_anchors(self, positions: List[int], semantic_hashes: List[int]) -> None:
        self._anchors = list(zip(positions, semantic_hashes))

    def find_recyclable_range(self, semantic_hash: int, tolerance: float = 0.9) -> Optional[tuple[int, int]]:
        for pos, h in self._anchors:
            if h == semantic_hash:
                return (0, pos)
        return None


class MockInferenceEngine(InferenceEngine):
    def __init__(self, model_type: ModelType = ModelType.BITNET, context_window: int = 4096, tokens_per_second: float = 20.0):
        self._model_type = model_type
        self._context_window = context_window
        self._tokens_per_second = tokens_per_second
        self._tokenizer = MockTokenizer()
        self._cache_manager = MockCacheManager(context_window)
        self._ready = True

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        config = config or GenerationConfig()
        input_tokens = self._tokenizer.encode(prompt)
        num_output_tokens = min(config.max_tokens, 50)
        generation_time = num_output_tokens / self._tokens_per_second
        time.sleep(generation_time * 0.1)

        output_tokens = TokenSequence.from_list([random.randint(10, 31999) for _ in range(num_output_tokens)])

        return GenerationResult(
            generated_tokens=output_tokens,
            generated_text=f"[mock response with {num_output_tokens} tokens for: {prompt[:50]}...]",
            stop_reason=StopReason.MAX_TOKENS,
            prompt_tokens=len(input_tokens),
            completion_tokens=num_output_tokens,
            total_tokens=len(input_tokens) + num_output_tokens,
            generation_time_ms=generation_time * 1000,
            tokens_per_second=self._tokens_per_second,
            compression_ratio=15.0 if config.use_sigma_compression else 1.0,
        )

    def generate_from_tokens(self, tokens: TokenSequence, config: Optional[GenerationConfig] = None) -> GenerationResult:
        return self.generate(f"[{len(tokens)} input tokens]", config)

    def stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[StreamChunk]:
        config = config or GenerationConfig()
        num_tokens = min(config.max_tokens, 50)

        for i in range(num_tokens):
            yield StreamChunk(
                token_id=random.randint(10, 31999),
                token_text=f"[t{i}]",
                position=i,
                is_first=(i == 0),
                is_last=(i == num_tokens - 1),
                final_result=self.generate(prompt, config) if i == num_tokens - 1 else None,
            )
            time.sleep(1.0 / self._tokens_per_second * 0.1)

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id="mock-bitnet-7b",
            model_type=self._model_type,
            parameter_count=7_000_000_000,
            context_window=self._context_window,
            vocab_size=32000,
            quantization=QuantizationType.TERNARY,
            bits_per_weight=1.58,
            model_size_bytes=2_600_000_000,
            estimated_memory_mb=2600,
            supports_streaming=True,
            supports_batching=True,
            supports_sigma_compression=True,
            supports_kv_recycling=True,
            estimated_tokens_per_second=self._tokens_per_second,
        )

    def get_context_window(self) -> int:
        return self._context_window

    def is_ready(self) -> bool:
        return self._ready

    def get_tokenizer(self) -> TokenizerProtocol:
        return self._tokenizer

    def get_cache_manager(self) -> CacheManagerProtocol:
        return self._cache_manager

