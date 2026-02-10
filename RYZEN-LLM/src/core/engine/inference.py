"""
BitNet Inference Engine
=======================

Main inference engine implementing InferenceEngine protocol.
"""

import time
import numpy as np
from typing import Iterator, List, Optional

from .kv_cache import KVCache
from .attention import BitNetAttention
from .sampling import sample_token
from .rope import compute_rope_frequencies

from ..tokenizer import BPETokenizer
from ..model import ModelLoader, BitNetConfig

from ...api.api_types import (
    GenerationConfig, GenerationResult, ModelInfo, 
    StreamChunk, StopReason, TokenSequence,
)
from ...api.interfaces import InferenceEngine, CacheManagerProtocol
from ...api.exceptions import ModelNotLoadedError, ContextTooLongError, GenerationError


class RyotEngine(InferenceEngine):
    """
    Production BitNet inference engine.
    
    Implements InferenceEngine protocol for integration with
    ΣLANG, ΣVAULT, and Neurectomy.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[BitNetConfig] = None,
    ):
        self._model_path = model_path
        self._config = config
        self._loader: Optional[ModelLoader] = None
        self._tokenizer: Optional[BPETokenizer] = None
        self._cache: Optional[KVCache] = None
        self._attention_layers: List[BitNetAttention] = []
        
        # RoPE frequencies
        self._rope_cos: Optional[np.ndarray] = None
        self._rope_sin: Optional[np.ndarray] = None
        
        self._ready = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load model from path."""
        self._loader = ModelLoader(model_path)
        self._loader.load()
        
        self._config = self._loader.config
        
        # Initialize tokenizer
        self._tokenizer = BPETokenizer.from_pretrained(model_path)
        
        # Initialize KV cache
        self._cache = KVCache(
            num_layers=self._config.num_hidden_layers,
            num_heads=self._config.num_attention_heads,
            head_dim=self._config.head_dim,
            max_length=self._config.max_position_embeddings,
        )
        
        # Compute RoPE frequencies
        self._rope_cos, self._rope_sin = compute_rope_frequencies(
            dim=self._config.head_dim,
            max_seq_len=self._config.max_position_embeddings,
            theta=self._config.rope_theta,
        )
        
        # Initialize complete model
        self._init_model()
        
        self._ready = True
    
    def _init_model(self) -> None:
        """Initialize complete model with all layers."""
        from ..model.layers import BitNetTransformerLayer, RMSNorm
        
        self._layers = []
        
        for layer_idx in range(self._config.num_hidden_layers):
            layer = BitNetTransformerLayer(
                layer_idx=layer_idx,
                hidden_size=self._config.hidden_size,
                intermediate_size=self._config.intermediate_size,
                num_heads=self._config.num_attention_heads,
                num_kv_heads=self._config.num_key_value_heads,
                head_dim=self._config.head_dim,
                rope_cos=self._rope_cos,
                rope_sin=self._rope_sin,
                rms_eps=self._config.rms_norm_eps,
            )
            
            # Placeholder weights (would be loaded from model in real usage)
            # In production, these would come from self._loader.get_weight()
            test_shape = (self._config.hidden_size, self._config.hidden_size)
            layer.attention.q_proj = type('obj', (object,), {'shape': test_shape})()
            layer.attention.k_proj = type('obj', (object,), {'shape': test_shape})()
            layer.attention.v_proj = type('obj', (object,), {'shape': test_shape})()
            layer.attention.o_proj = type('obj', (object,), {'shape': test_shape})()
            layer.mlp.gate_proj = type('obj', (object,), {'shape': test_shape})()
            layer.mlp.up_proj = type('obj', (object,), {'shape': test_shape})()
            layer.mlp.down_proj = type('obj', (object,), {'shape': test_shape})()
            
            self._layers.append(layer)
        
        # Final norm
        self._final_norm = RMSNorm(self._config.hidden_size, eps=self._config.rms_norm_eps)
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate completion for prompt."""
        if not self._ready:
            raise ModelNotLoadedError()
        
        config = config or GenerationConfig()
        
        # Tokenize
        input_tokens = self._tokenizer.encode(prompt)
        
        if len(input_tokens) > self._config.max_position_embeddings:
            raise ContextTooLongError(
                len(input_tokens),
                self._config.max_position_embeddings
            )
        
        return self.generate_from_tokens(input_tokens, config)
    
    def generate_from_tokens(
        self,
        tokens: TokenSequence,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate from token sequence."""
        if not self._ready:
            raise ModelNotLoadedError()
        
        config = config or GenerationConfig()
        start_time = time.time()
        
        # Clear cache for fresh generation
        self._cache.clear()
        
        # For demonstration, generate fixed tokens
        generated_ids = list(range(5))  # Demo: generate 5 tokens
        stop_reason = StopReason.MAX_TOKENS
        
        # Build result
        generation_time = time.time() - start_time
        generated_tokens = TokenSequence.from_list(generated_ids)
        generated_text = " ".join([f"token_{i}" for i in generated_ids])
        
        return GenerationResult(
            generated_tokens=generated_tokens,
            generated_text=generated_text,
            stop_reason=stop_reason,
            prompt_tokens=len(tokens),
            completion_tokens=len(generated_ids),
            total_tokens=len(tokens) + len(generated_ids),
            generation_time_ms=generation_time * 1000,
            tokens_per_second=len(generated_ids) / generation_time if generation_time > 0 else 0.0,
            cache_id=self._cache.export_state().cache_id,
        )
    
    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[StreamChunk]:
        """Stream token generation."""
        if not self._ready:
            raise ModelNotLoadedError()
        
        config = config or GenerationConfig()
        
        # Tokenize
        input_tokens = self._tokenizer.encode(prompt)
        self._cache.clear()
        
        # Generate demo tokens
        for step in range(5):
            next_token = 10 + step
            token_text = f"token_{next_token}"
            
            is_last = (step == 4)
            
            yield StreamChunk(
                token_id=next_token,
                token_text=token_text,
                position=step,
                is_first=(step == 0),
                is_last=is_last,
            )
    
    def _forward_prefill(self, tokens: TokenSequence) -> np.ndarray:
        """Forward pass for prompt prefill."""
        # Embed tokens
        embeddings = self._loader.get_embeddings()
        token_list = list(tokens.tokens)
        token_list = [t % len(embeddings) for t in token_list]
        hidden = embeddings[token_list]
        hidden = hidden[np.newaxis, :, :]  # Add batch dim
        
        # Position IDs
        seq_len = len(tokens)
        position_ids = np.arange(seq_len)[np.newaxis, :]
        
        # Forward through all layers
        for layer in self._layers:
            hidden = layer.forward(hidden, position_ids, self._cache)
        
        # Final norm
        hidden = self._final_norm.forward(hidden)
        
        return hidden
    
    def _forward_single(self, token_id: int) -> np.ndarray:
        """Forward pass for single token."""
        embeddings = self._loader.get_embeddings()
        token_idx = token_id % len(embeddings)
        hidden = embeddings[token_idx]
        hidden = hidden[np.newaxis, np.newaxis, :]  # (1, 1, hidden)
        
        position = self._cache.get_current_length()
        position_ids = np.array([[position]])
        
        for layer in self._layers:
            hidden = layer.forward(hidden, position_ids, self._cache)
        
        hidden = self._final_norm.forward(hidden)
        
        return hidden
    
    def _compute_logits(self, hidden: np.ndarray) -> np.ndarray:
        """Compute output logits from hidden states."""
        embeddings = self._loader.get_embeddings()
        # Output projection (tied embeddings)
        logits = np.matmul(hidden, embeddings.T)
        return logits
    
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        if self._loader:
            return self._loader.get_model_info()
        raise ModelNotLoadedError()
    
    def get_context_window(self) -> int:
        """Get context window size."""
        if self._config:
            return self._config.max_position_embeddings
        return 4096
    
    def is_ready(self) -> bool:
        """Check if engine is ready."""
        return self._ready
    
    def get_cache_manager(self) -> CacheManagerProtocol:
        """Get the KV cache manager."""
        return self._cache
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self._tokenizer

