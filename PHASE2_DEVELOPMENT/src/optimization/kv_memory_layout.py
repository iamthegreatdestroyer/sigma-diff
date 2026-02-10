"""
KV Cache Memory Layout Optimization Module
Sprint 4.4 - Task 3: Memory Layout Optimization

Optimizes KV cache memory layout for improved performance:
- Cache-line alignment (64B boundaries)
- Contiguous allocation
- Block-structured layouts (better spatial locality)
- Access pattern optimization
- NUMA-aware allocation hints
- Memory pooling for reuse

Target: >30% cache hit improvement, <10ns per access
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Constants & Enums
# =============================================================================

class MemoryLayout(Enum):
    """Memory layout strategies."""
    NAIVE = auto()           # Row-major, unaligned
    ALIGNED = auto()         # Cache-line aligned
    BLOCKED = auto()         # Block-structured (better locality)
    INTERLEAVED = auto()     # Interleaved K/V for temporal locality
    COLUMNAR = auto()        # Column-major (transpose)


# Standard sizes
CACHE_LINE_SIZE = 64  # Intel/AMD cache line = 64 bytes
BLOCK_SIZE = 256  # Typical SIMD block for 256-bit operations
PAGE_SIZE = 4096  # Page alignment for NUMA


@dataclass
class LayoutStats:
    """Statistics about memory layout performance."""
    access_latency_ns: float = 0.0
    cache_miss_rate: float = 0.0
    memory_fragmentation: float = 0.0
    utilization: float = 0.0
    locality_score: float = 0.0  # 0-1, higher is better
    bandwidth_utilization: float = 0.0


# =============================================================================
# Memory Layout Strategies
# =============================================================================

class MemoryLayoutOptimizer(ABC):
    """Base class for memory layout optimization."""
    
    @abstractmethod
    def layout(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Transform data layout, return transformed data and metadata."""
        pass
    
    @abstractmethod
    def restore(self, data: np.ndarray, metadata: Dict) -> np.ndarray:
        """Restore original layout from optimized layout."""
        pass


class AlignedLayoutOptimizer(MemoryLayoutOptimizer):
    """
    Cache-line aligned memory layout.
    
    Pads each row to 64-byte boundary for cache efficiency.
    """
    
    def __init__(self, alignment: int = CACHE_LINE_SIZE):
        self.alignment = alignment
    
    def layout(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Align data to cache lines."""
        # data shape: (seq_len, hidden_dim)
        seq_len, hidden_dim = data.shape
        
        # Calculate aligned width (next multiple of alignment/element_size)
        element_size = data.dtype.itemsize
        elements_per_line = self.alignment // element_size
        aligned_width = ((hidden_dim + elements_per_line - 1) // elements_per_line) * elements_per_line
        
        # Create aligned array
        aligned = np.zeros((seq_len, aligned_width), dtype=data.dtype)
        aligned[:, :hidden_dim] = data
        
        metadata = {
            "original_shape": data.shape,
            "aligned_shape": aligned.shape,
            "alignment": self.alignment,
        }
        
        return aligned, metadata
    
    def restore(self, data: np.ndarray, metadata: Dict) -> np.ndarray:
        """Restore original shape."""
        original_shape = metadata["original_shape"]
        seq_len, hidden_dim = original_shape
        return data[:seq_len, :hidden_dim]


class BlockedLayoutOptimizer(MemoryLayoutOptimizer):
    """
    Block-structured memory layout.
    
    Reorganizes data into blocks for improved spatial locality.
    Better for SIMD operations and cache behavior.
    """
    
    def __init__(self, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
    
    def layout(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Reorganize into blocks."""
        seq_len, hidden_dim = data.shape
        
        # Calculate block dimensions
        # block_size = elements per block
        element_size = data.dtype.itemsize
        elements_per_block = self.block_size // element_size
        
        # Number of complete blocks
        num_blocks = (hidden_dim + elements_per_block - 1) // elements_per_block
        padded_width = num_blocks * elements_per_block
        
        # Pad and reshape
        padded = np.zeros((seq_len, padded_width), dtype=data.dtype)
        padded[:, :hidden_dim] = data
        
        # Reshape to blocks: (seq_len, num_blocks, elements_per_block)
        blocked = padded.reshape(seq_len, num_blocks, elements_per_block)
        
        metadata = {
            "original_shape": data.shape,
            "blocked_shape": blocked.shape,
            "block_size": self.block_size,
            "padded_width": padded_width,
        }
        
        return blocked, metadata
    
    def restore(self, data: np.ndarray, metadata: Dict) -> np.ndarray:
        """Restore original layout."""
        seq_len, num_blocks, elements_per_block = data.shape
        original_seq, original_hidden = metadata["original_shape"]
        
        # Reshape back
        reshaped = data.reshape(seq_len, -1)
        return reshaped[:original_seq, :original_hidden]


class InterleavedLayoutOptimizer(MemoryLayoutOptimizer):
    """
    Interleaved K/V layout.
    
    Stores K and V side-by-side for better temporal locality
    when processing key-value pairs together.
    """
    
    def layout(self, keys: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Interleave keys and values."""
        seq_len, hidden_dim = keys.shape
        
        # Create interleaved layout: [K0, V0, K1, V1, ...]
        interleaved = np.zeros((seq_len, hidden_dim * 2), dtype=keys.dtype)
        interleaved[:, ::2] = keys    # Even indices: keys
        interleaved[:, 1::2] = values  # Odd indices: values
        
        metadata = {
            "original_k_shape": keys.shape,
            "original_v_shape": values.shape,
            "interleaved": True,
        }
        
        return interleaved, metadata
    
    def restore(self, data: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Separate keys and values."""
        seq_len, _ = metadata["original_k_shape"]
        keys = data[:, ::2]
        values = data[:, 1::2]
        return keys, values


class ColumnarLayoutOptimizer(MemoryLayoutOptimizer):
    """
    Column-major (transpose) layout.
    
    Better for column-wise operations and vectorization.
    """
    
    def layout(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Transpose to column-major layout."""
        # Transpose: (seq_len, hidden_dim) -> (hidden_dim, seq_len)
        columnar = data.T.copy()
        
        metadata = {
            "original_shape": data.shape,
            "columnar_shape": columnar.shape,
            "transposed": True,
        }
        
        return columnar, metadata
    
    def restore(self, data: np.ndarray, metadata: Dict) -> np.ndarray:
        """Transpose back to row-major."""
        return data.T


# =============================================================================
# Access Pattern Analyzer
# =============================================================================

class AccessPatternAnalyzer:
    """
    Analyzes memory access patterns to optimize layout.
    """
    
    def __init__(self):
        self.access_history: List[Tuple[int, int]] = []  # (sequence_idx, hidden_idx)
        self.spatial_locality: float = 0.0
        self.temporal_locality: float = 0.0
    
    def record_access(self, seq_idx: int, hidden_idx: int) -> None:
        """Record a memory access."""
        self.access_history.append((seq_idx, hidden_idx))
    
    def compute_spatial_locality(self) -> float:
        """
        Compute spatial locality score (0-1).
        Higher = accesses are close together in memory.
        """
        if len(self.access_history) < 2:
            return 0.0
        
        distances = []
        for i in range(len(self.access_history) - 1):
            seq1, hidden1 = self.access_history[i]
            seq2, hidden2 = self.access_history[i + 1]
            
            # Manhattan distance in memory
            distance = abs(seq1 - seq2) + abs(hidden1 - hidden2)
            distances.append(distance)
        
        # Normalize: close accesses get higher score
        if not distances:
            return 0.0
        
        avg_distance = np.mean(distances)
        max_possible = 10000  # Arbitrary large distance
        locality = max(0, 1.0 - (avg_distance / max_possible))
        
        return locality
    
    def compute_temporal_locality(self) -> float:
        """
        Compute temporal locality score (0-1).
        Higher = same memory locations accessed frequently.
        """
        if len(self.access_history) < 2:
            return 0.0
        
        # Count unique accesses
        unique_accesses = len(set(self.access_history))
        total_accesses = len(self.access_history)
        
        # Temporal locality = 1 - (unique / total)
        # If all unique = 0 locality, if all same = 1 locality
        locality = 1.0 - (unique_accesses / total_accesses) if total_accesses > 0 else 0.0
        
        return locality
    
    def recommend_layout(self) -> MemoryLayout:
        """Recommend layout based on access patterns."""
        spatial = self.compute_spatial_locality()
        temporal = self.compute_temporal_locality()
        
        if temporal > 0.7:
            # Many repeated accesses: use interleaved
            return MemoryLayout.INTERLEAVED
        elif spatial > 0.6:
            # Spatially local: use blocked
            return MemoryLayout.BLOCKED
        else:
            # Default: aligned
            return MemoryLayout.ALIGNED


# =============================================================================
# Memory Pool for Efficient Reuse
# =============================================================================

class MemoryPool:
    """
    Pre-allocated memory pool for efficient cache buffer reuse.
    Reduces allocation overhead and fragmentation.
    """
    
    def __init__(self, buffer_size: int, num_buffers: int = 8):
        self.buffer_size = buffer_size
        self.num_buffers = num_buffers
        self.available_buffers: List[np.ndarray] = [
            np.zeros(buffer_size, dtype=np.float32)
            for _ in range(num_buffers)
        ]
        self.in_use_buffers: Dict[int, np.ndarray] = {}  # id -> buffer
        self.next_id = 0
        self.allocation_count = 0
        self.reuse_count = 0
    
    def allocate(self) -> Tuple[int, np.ndarray]:
        """Allocate a buffer from pool."""
        if self.available_buffers:
            buffer = self.available_buffers.pop()
            self.reuse_count += 1
        else:
            # Allocate new if pool exhausted
            buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        buffer_id = self.next_id
        self.next_id += 1
        self.in_use_buffers[buffer_id] = buffer
        self.allocation_count += 1
        
        return buffer_id, buffer
    
    def deallocate(self, buffer_id: int) -> bool:
        """Return buffer to pool."""
        if buffer_id not in self.in_use_buffers:
            return False
        
        buffer = self.in_use_buffers.pop(buffer_id)
        
        if len(self.available_buffers) < self.num_buffers:
            self.available_buffers.append(buffer)
        
        return True
    
    @property
    def reuse_rate(self) -> float:
        """Percentage of allocations that reused existing buffers."""
        return (self.reuse_count / self.allocation_count * 100) if self.allocation_count > 0 else 0.0


# =============================================================================
# Integrated Layout Manager
# =============================================================================

class CacheLayoutManager:
    """
    Manages overall cache memory layout strategy.
    """
    
    def __init__(self, hidden_dim: int, max_seq_len: int):
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Initialize optimizers
        self.aligned_opt = AlignedLayoutOptimizer()
        self.blocked_opt = BlockedLayoutOptimizer()
        self.interleaved_opt = InterleavedLayoutOptimizer()
        self.columnar_opt = ColumnarLayoutOptimizer()
        
        # Memory pool
        buffer_size = hidden_dim * max_seq_len
        self.memory_pool = MemoryPool(buffer_size, num_buffers=4)
        
        # Access pattern analyzer
        self.analyzer = AccessPatternAnalyzer()
        self.current_layout = MemoryLayout.ALIGNED
    
    def optimize_layout(self, keys: np.ndarray, values: np.ndarray) -> Tuple[Dict, LayoutStats]:
        """
        Optimize KV cache layout.
        
        Args:
            keys: Key cache (seq_len, hidden_dim)
            values: Value cache (seq_len, hidden_dim)
        
        Returns:
            (optimized_data_dict, stats)
        """
        stats = LayoutStats()
        
        # Recommend layout based on access patterns
        recommended = self.analyzer.recommend_layout()
        if recommended != self.current_layout:
            logger.info(f"Switching layout from {self.current_layout.name} to {recommended.name}")
            self.current_layout = recommended
        
        # Apply layout optimization
        optimized_data = {}
        
        if self.current_layout == MemoryLayout.ALIGNED:
            keys_opt, meta_k = self.aligned_opt.layout(keys)
            values_opt, meta_v = self.aligned_opt.layout(values)
            optimized_data = {
                "keys": (keys_opt, meta_k),
                "values": (values_opt, meta_v),
                "layout": "aligned"
            }
        
        elif self.current_layout == MemoryLayout.BLOCKED:
            keys_opt, meta_k = self.blocked_opt.layout(keys)
            values_opt, meta_v = self.blocked_opt.layout(values)
            optimized_data = {
                "keys": (keys_opt, meta_k),
                "values": (values_opt, meta_v),
                "layout": "blocked"
            }
        
        elif self.current_layout == MemoryLayout.INTERLEAVED:
            interleaved, meta = self.interleaved_opt.layout(keys, values)
            optimized_data = {
                "interleaved": (interleaved, meta),
                "layout": "interleaved"
            }
        
        elif self.current_layout == MemoryLayout.COLUMNAR:
            keys_opt, meta_k = self.columnar_opt.layout(keys)
            values_opt, meta_v = self.columnar_opt.layout(values)
            optimized_data = {
                "keys": (keys_opt, meta_k),
                "values": (values_opt, meta_v),
                "layout": "columnar"
            }
        
        # Compute statistics
        stats.locality_score = self.analyzer.compute_spatial_locality()
        stats.utilization = self.memory_pool.reuse_rate / 100.0
        
        return optimized_data, stats


# =============================================================================
# Convenience Functions
# =============================================================================

def create_layout_manager(hidden_dim: int, max_seq_len: int) -> CacheLayoutManager:
    """Create a cache layout manager."""
    return CacheLayoutManager(hidden_dim, max_seq_len)


def benchmark_layouts(data: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Benchmark different layout strategies.
    
    Args:
        data: Original data (seq_len, hidden_dim)
    
    Returns:
        {layout_name: (optimized_data, overhead_percent)}
    """
    import time
    
    results = {}
    
    # Aligned layout
    start = time.time()
    aligned_opt = AlignedLayoutOptimizer()
    aligned, _ = aligned_opt.layout(data)
    overhead = (time.time() - start) * 1000
    results["aligned"] = (aligned, overhead)
    
    # Blocked layout
    start = time.time()
    blocked_opt = BlockedLayoutOptimizer()
    blocked, _ = blocked_opt.layout(data)
    overhead = (time.time() - start) * 1000
    results["blocked"] = (blocked, overhead)
    
    # Columnar layout
    start = time.time()
    columnar_opt = ColumnarLayoutOptimizer()
    columnar, _ = columnar_opt.layout(data)
    overhead = (time.time() - start) * 1000
    results["columnar"] = (columnar, overhead)
    
    return results
