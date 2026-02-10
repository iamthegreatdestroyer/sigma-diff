"""
KV Cache Management
===================

Implements CacheManagerProtocol for efficient inference.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time

from ...api.api_types import KVCacheState
from ...api.interfaces import CacheManagerProtocol
from ...api.exceptions import CacheError


class KVCache(CacheManagerProtocol):
    """
    Key-Value cache for transformer inference.
    
    Stores intermediate key/value states to avoid recomputation
    during autoregressive generation.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_length: int,
        dtype: np.dtype = np.float32,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.dtype = dtype
        
        # Pre-allocate cache tensors
        self._keys: List[np.ndarray] = []
        self._values: List[np.ndarray] = []
        
        for _ in range(num_layers):
            # Shape: (batch=1, num_heads, max_length, head_dim)
            self._keys.append(np.zeros(
                (1, num_heads, max_length, head_dim),
                dtype=dtype
            ))
            self._values.append(np.zeros(
                (1, num_heads, max_length, head_dim),
                dtype=dtype
            ))
        
        self._current_length = 0
        self._model_id = ""
        
        # ΣLANG anchor tracking
        self._anchor_positions: List[int] = []
        self._anchor_hashes: List[int] = []
    
    def update(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        position: int,
    ) -> None:
        """
        Update cache with new key/value for a layer.
        
        Args:
            layer_idx: Transformer layer index
            key: New key tensor (batch, heads, seq_len, head_dim)
            value: New value tensor (batch, heads, seq_len, head_dim)
            position: Starting position in sequence
        """
        seq_len = key.shape[2]
        end_pos = position + seq_len
        
        if end_pos > self.max_length:
            raise CacheError(
                f"Cache overflow: {end_pos} > {self.max_length}",
                cache_id=self._model_id
            )
        
        self._keys[layer_idx][:, :, position:end_pos, :] = key
        self._values[layer_idx][:, :, position:end_pos, :] = value
        
        self._current_length = max(self._current_length, end_pos)
    
    def get(
        self,
        layer_idx: int,
        end_position: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cached key/value up to position.
        
        Args:
            layer_idx: Transformer layer index
            end_position: End position (default: current length)
        
        Returns:
            Tuple of (keys, values) up to end_position
        """
        end = end_position or self._current_length
        return (
            self._keys[layer_idx][:, :, :end, :],
            self._values[layer_idx][:, :, :end, :]
        )
    
    def get_current_length(self) -> int:
        """Get current cache length."""
        return self._current_length
    
    def get_max_length(self) -> int:
        """Get maximum cache length."""
        return self.max_length
    
    def clear(self) -> None:
        """Clear all cached values."""
        for layer_idx in range(self.num_layers):
            self._keys[layer_idx].fill(0)
            self._values[layer_idx].fill(0)
        
        self._current_length = 0
        self._anchor_positions = []
        self._anchor_hashes = []
    
    def export_state(self) -> KVCacheState:
        """Export cache state for persistence."""
        return KVCacheState(
            cache_id=f"cache_{int(time.time())}",
            model_id=self._model_id,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            sequence_length=self._current_length,
            key_states=[
                self._keys[i][:, :, :self._current_length, :].copy()
                for i in range(self.num_layers)
            ],
            value_states=[
                self._values[i][:, :, :self._current_length, :].copy()
                for i in range(self.num_layers)
            ],
            anchor_positions=self._anchor_positions.copy(),
            anchor_semantic_hashes=self._anchor_hashes.copy(),
            created_timestamp=time.time(),
        )
    
    def import_state(self, state: KVCacheState) -> bool:
        """Import cache state from persistence."""
        try:
            # Validate compatibility
            if state.num_layers != self.num_layers:
                raise CacheError(f"Layer mismatch: {state.num_layers} vs {self.num_layers}")
            if state.num_heads != self.num_heads:
                raise CacheError(f"Head mismatch: {state.num_heads} vs {self.num_heads}")
            
            # Copy states
            for i in range(self.num_layers):
                seq_len = state.key_states[i].shape[2]
                self._keys[i][:, :, :seq_len, :] = state.key_states[i]
                self._values[i][:, :, :seq_len, :] = state.value_states[i]
            
            self._current_length = state.sequence_length
            self._model_id = state.model_id
            
            if state.anchor_positions:
                self._anchor_positions = list(state.anchor_positions)
            if state.anchor_semantic_hashes:
                self._anchor_hashes = list(state.anchor_semantic_hashes)
            
            return True
        
        except Exception as e:
            raise CacheError(f"Failed to import state: {e}")
    
    def register_sigma_anchors(
        self,
        positions: List[int],
        semantic_hashes: List[int],
    ) -> None:
        """Register ΣLANG semantic anchors for RSU recycling."""
        self._anchor_positions = list(positions)
        self._anchor_hashes = list(semantic_hashes)
    
    def find_recyclable_range(
        self,
        semantic_hash: int,
        tolerance: float = 0.9,
    ) -> Optional[Tuple[int, int]]:
        """
        Find a recyclable cache range matching semantic hash.
        
        Returns:
            Tuple of (start, end) positions, or None if no match
        """
        for pos, h in zip(self._anchor_positions, self._anchor_hashes):
            if h == semantic_hash:
                return (0, pos)
        return None
    
    def truncate(self, length: int) -> None:
        """Truncate cache to specified length."""
        if length < self._current_length:
            self._current_length = length
            
            # Clear truncated positions
            for layer_idx in range(self.num_layers):
                self._keys[layer_idx][:, :, length:, :] = 0
                self._values[layer_idx][:, :, length:, :] = 0
            
            # Remove anchors beyond truncation
            self._anchor_positions = [
                p for p in self._anchor_positions if p < length
            ]
            self._anchor_hashes = self._anchor_hashes[:len(self._anchor_positions)]

