"""
Tensor Parallel Attention Layer
==============================

Implements head-wise parallel attention for distributed inference.
Supports multi-head attention with tensor parallelism across GPUs.

Key Features:
- Head-wise parallelism for attention layers
- Distributed attention computation
- Optimized memory usage for large models
- NCCL communication for collectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
import logging

from .tensor_parallel import TensorParallelLayer
from ..distributed.communication import NCCLCommunicator

logger = logging.getLogger(__name__)


class ParallelAttention(TensorParallelLayer):
    """
    Head-wise parallel multi-head attention.

    Splits attention heads across GPUs for parallel computation.
    Uses tensor parallelism to scale attention to large models.

    Memory: O(seq_len² * num_heads/TP + hidden_size * num_heads/TP)
    Communication: all_reduce for attention output
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        comm_handler: Optional[NCCLCommunicator] = None
    ):
        """Initialize parallel attention layer.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Total number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Attention dropout probability
            comm_handler: Communication handler for collectives
        """
        super().__init__(comm_handler)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # Get tensor parallel configuration
        self.world_size = self.comm_handler.world_size if self.comm_handler else 1
        self.rank = self.comm_handler.rank if self.comm_handler else 0

        # Heads per GPU (tensor parallelism)
        self.heads_per_gpu = num_heads // self.world_size
        assert num_heads % self.world_size == 0, "num_heads must be divisible by world_size"

        # Local head dimension
        self.local_hidden_size = self.heads_per_gpu * self.head_dim

        # Linear layers for Q, K, V (column parallel)
        self.q_proj = nn.Linear(hidden_size, self.local_hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.local_hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.local_hidden_size, bias=False)

        # Output projection (row parallel)
        self.out_proj = nn.Linear(self.local_hidden_size, hidden_size, bias=False)

        # RoPE frequencies (computed once)
        self._rope_freqs: Optional[torch.Tensor] = None

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize attention weights."""
        # Xavier/Glorot initialization for projections
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def _compute_rope_frequencies(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute RoPE rotation frequencies."""
        if self._rope_freqs is None or self._rope_freqs.size(-1) < seq_len:
            # Compute frequencies for all positions up to max_seq_len
            positions = torch.arange(self.max_seq_len, device=device).float()
            freqs = torch.outer(positions, 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim)))

            # Cache frequencies
            self._rope_freqs = freqs

        return self._rope_freqs[:seq_len]

    def _apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation to input tensor."""
        # x shape: (batch, seq_len, num_heads, head_dim)
        # freqs shape: (seq_len, head_dim//2)

        # Split into even/odd dimensions
        x_even = x[..., ::2]  # (batch, seq_len, num_heads, head_dim//2)
        x_odd = x[..., 1::2]  # (batch, seq_len, num_heads, head_dim//2)

        # Apply rotation
        cos_freq = freqs.cos().unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
        sin_freq = freqs.sin().unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)

        # Rotate: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
        rotated_even = x_even * cos_freq - x_odd * sin_freq
        rotated_odd = x_even * sin_freq + x_odd * cos_freq

        # Interleave back
        rotated = torch.zeros_like(x)
        rotated[..., ::2] = rotated_even
        rotated[..., 1::2] = rotated_odd

        return rotated

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with tensor parallelism.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Attention mask (batch, 1, seq_len, seq_len)
            past_key_value: Cached K,V from previous steps
            output_attentions: Whether to return attention weights

        Returns:
            output: Attention output
            attentions: Attention weights (if output_attentions=True)
            past_key_value: Updated KV cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V (column parallel - each GPU gets subset of heads)
        query = self.q_proj(hidden_states)  # (batch, seq_len, local_hidden_size)
        key = self.k_proj(hidden_states)    # (batch, seq_len, local_hidden_size)
        value = self.v_proj(hidden_states)  # (batch, seq_len, local_hidden_size)

        # Reshape for multi-head attention
        # (batch, seq_len, local_hidden_size) -> (batch, seq_len, heads_per_gpu, head_dim)
        query = query.view(batch_size, seq_len, self.heads_per_gpu, self.head_dim)
        key = key.view(batch_size, seq_len, self.heads_per_gpu, self.head_dim)
        value = value.view(batch_size, seq_len, self.heads_per_gpu, self.head_dim)

        # Transpose for attention: (batch, heads_per_gpu, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        # Apply RoPE to query and key
        rope_freqs = self._compute_rope_frequencies(key.size(2), query.device)
        query = self._apply_rope(query, rope_freqs)
        key = self._apply_rope(key, rope_freqs)

        # Compute attention scores
        # (batch, heads_per_gpu, seq_len, head_dim) @ (batch, heads_per_gpu, head_dim, kv_seq_len)
        # -> (batch, heads_per_gpu, seq_len, kv_seq_len)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # (batch, heads_per_gpu, seq_len, kv_seq_len) @ (batch, heads_per_gpu, kv_seq_len, head_dim)
        # -> (batch, heads_per_gpu, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value)

        # Transpose back: (batch, seq_len, heads_per_gpu, head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Reshape: (batch, seq_len, local_hidden_size)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.local_hidden_size)

        # Output projection (row parallel - all_reduce across GPUs)
        output = self.out_proj(attn_output)

        # All-reduce across tensor parallel ranks
        if self.comm_handler and self.world_size > 1:
            self.comm_handler.all_reduce(output)

        # Prepare outputs
        attentions = attn_weights if output_attentions else None
        past_key_value = (key, value) if past_key_value is not None else None

        return output, attentions, past_key_value

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"ParallelAttention(hidden_size={self.hidden_size}, num_heads={self.num_heads}, heads_per_gpu={self.heads_per_gpu}, rank={self.rank}/{self.world_size})"
