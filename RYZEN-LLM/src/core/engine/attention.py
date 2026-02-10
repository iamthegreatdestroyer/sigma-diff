"""
BitNet Attention Computation
============================
"""

import numpy as np
from typing import Optional, Tuple

from .kv_cache import KVCache
from .rope import apply_rope


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> np.ndarray:
    """
    Compute scaled dot-product attention.
    
    Args:
        query: (batch, heads, seq_len, head_dim)
        key: (batch, heads, seq_len, head_dim)
        value: (batch, heads, seq_len, head_dim)
        mask: Optional attention mask
        scale: Optional scale factor (default: 1/sqrt(head_dim))
    
    Returns:
        Attention output (batch, heads, seq_len, head_dim)
    """
    head_dim = query.shape[-1]
    scale = scale or (1.0 / np.sqrt(head_dim))
    
    # Compute attention scores
    scores = np.matmul(query, key.transpose(0, 1, 3, 2)) * scale
    
    # Apply mask
    if mask is not None:
        scores = scores + mask
    
    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    # Compute output
    output = np.matmul(attention_weights, value)
    
    return output


def create_causal_mask(seq_len: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """Create causal attention mask."""
    mask = np.triu(np.ones((seq_len, seq_len), dtype=dtype), k=1)
    return mask * -1e9


class BitNetAttention:
    """
    BitNet attention layer with ternary weights.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        
        # Weights will be set by model loader
        self.q_proj = None  # QuantizedTensor
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
    
    def forward(
        self,
        hidden_states: np.ndarray,
        position_ids: np.ndarray,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> np.ndarray:
        """
        Forward pass through attention layer.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            position_ids: (batch, seq_len)
            cache: Optional KV cache
            layer_idx: Layer index for cache
        
        Returns:
            Output (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V using ternary weights
        from ..model.quantization import ternary_matmul
        
        q = ternary_matmul(hidden_states, self.q_proj)
        k = ternary_matmul(hidden_states, self.k_proj)
        v = ternary_matmul(hidden_states, self.v_proj)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
        
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(0, 2, 1, 3)
        
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)
        
        # Apply RoPE
        q, k = apply_rope(q, k, self.rope_cos, self.rope_sin, position_ids)
        
        # Handle GQA (grouped query attention)
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = np.repeat(k, repeat_factor, axis=1)
            v = np.repeat(v, repeat_factor, axis=1)
        
        # Update cache
        if cache is not None:
            cache_pos = cache.get_current_length()
            cache.update(layer_idx, k, v, cache_pos)
            k, v = cache.get(layer_idx)
        
        # Create causal mask
        mask = create_causal_mask(k.shape[2])
        if cache is not None and seq_len == 1:
            # Single token generation - only mask future
            mask = mask[-1:, :]
        
        # Compute attention
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = ternary_matmul(attn_output, self.o_proj)
        
        return output
