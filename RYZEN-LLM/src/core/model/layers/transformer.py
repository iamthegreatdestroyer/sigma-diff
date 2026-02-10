"""
Complete BitNet Transformer Layer
=================================
"""

import numpy as np
from typing import Optional, List

from .ffn import BitNetMLP
from .rmsnorm import RMSNorm
from ...engine.kv_cache import KVCache


class BitNetTransformerLayer:
    """
    Single transformer layer with attention and MLP.
    """
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
        rms_eps: float = 1e-6,
    ):
        self.layer_idx = layer_idx
        
        # Lazy import to avoid circular dependency
        from ...engine.attention import BitNetAttention
        
        # Attention
        self.attention = BitNetAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        
        # MLP
        self.mlp = BitNetMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        
        # Normalization layers
        self.input_norm = RMSNorm(hidden_size, eps=rms_eps)
        self.post_attn_norm = RMSNorm(hidden_size, eps=rms_eps)
    
    def forward(
        self,
        hidden_states: np.ndarray,
        position_ids: np.ndarray,
        cache: Optional[KVCache] = None,
    ) -> np.ndarray:
        """
        Forward pass through transformer layer.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            position_ids: (batch, seq_len)
            cache: Optional KV cache
        
        Returns:
            Output hidden states
        """
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_norm.forward(hidden_states)
        hidden_states = self.attention.forward(
            hidden_states, position_ids, cache, self.layer_idx
        )
        hidden_states = residual + hidden_states
        
        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attn_norm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class BitNetModel:
    """
    Complete BitNet transformer model.
    """
    
    def __init__(self, config):
        from ..config import BitNetConfig
        self.config: BitNetConfig = config
        
        self.layers: List[BitNetTransformerLayer] = []
        self.final_norm: Optional[RMSNorm] = None
        self.embeddings: Optional[np.ndarray] = None
        
        # RoPE frequencies
        self.rope_cos: Optional[np.ndarray] = None
        self.rope_sin: Optional[np.ndarray] = None
    
    def forward(
        self,
        input_ids: np.ndarray,
        position_ids: np.ndarray,
        cache: Optional[KVCache] = None,
    ) -> np.ndarray:
        """
        Full forward pass through model.
        
        Args:
            input_ids: (batch, seq_len)
            position_ids: (batch, seq_len)
            cache: Optional KV cache
        
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        # Embed tokens
        hidden_states = self.embeddings[input_ids]
        
        # Forward through layers
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, position_ids, cache)
        
        # Final norm
        hidden_states = self.final_norm.forward(hidden_states)
        
        # Output logits (tied embeddings)
        logits = np.matmul(hidden_states, self.embeddings.T)
        
        return logits
