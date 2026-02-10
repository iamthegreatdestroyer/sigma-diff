"""
Rotary Position Embeddings (RoPE)
=================================
"""

import numpy as np
from typing import Tuple


def compute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RoPE frequency matrices.
    
    Returns:
        cos, sin: Frequency matrices of shape (max_seq_len, dim//2)
    """
    # Frequency for each dimension pair
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
    
    # Position indices
    positions = np.arange(max_seq_len).astype(np.float32)
    
    # Outer product: (max_seq_len, dim//2)
    angles = np.outer(positions, freqs)
    
    return np.cos(angles), np.sin(angles)


def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    position_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        cos: Cosine frequencies (max_seq_len, head_dim//2)
        sin: Sine frequencies (max_seq_len, head_dim//2)
        position_ids: Position indices (batch, seq_len)
    
    Returns:
        Rotated query and key tensors
    """
    # Get frequencies for current positions
    cos_pos = cos[position_ids]  # (batch, seq_len, head_dim//2)
    sin_pos = sin[position_ids]
    
    # Expand for heads
    cos_pos = cos_pos[:, np.newaxis, :, :]  # (batch, 1, seq_len, head_dim//2)
    sin_pos = sin_pos[:, np.newaxis, :, :]
    
    # Split into pairs
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    # Apply rotation
    q_rot = np.concatenate([
        q1 * cos_pos - q2 * sin_pos,
        q1 * sin_pos + q2 * cos_pos,
    ], axis=-1)
    
    k_rot = np.concatenate([
        k1 * cos_pos - k2 * sin_pos,
        k1 * sin_pos + k2 * cos_pos,
    ], axis=-1)
    
    return q_rot, k_rot
