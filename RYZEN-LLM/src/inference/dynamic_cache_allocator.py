"""
Dynamic Cache Allocation Strategy
=================================

Advanced cache allocation with adaptive sizing, memory pooling,
and intelligent eviction policies for distributed KV-cache.

Key Features:
- Workload-adaptive cache sizing
- Memory pool management with reuse
- Distributed-aware eviction policies
- Cache partitioning based on access patterns
- Performance monitoring and optimization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
import time
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from distributed.communication import NCCLCommunicator

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policies for distributed setting."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    SIZE_AWARE = "size_aware"      # Consider entry size
    DISTRIBUTED_AWARE = "distributed_aware"  # Consider GPU distribution
    ADAPTIVE = "adaptive"          # Learn optimal policy


@dataclass
class CacheEntry:
    """Cache entry metadata for dynamic allocation."""
    layer_idx: int
    seq_pos: int
    size_bytes: int
    access_count: int
    last_access: float
    gpu_rank: int
    compression_ratio: float
    priority_score: float = 0.0


@dataclass
class MemoryPool:
    """Memory pool for efficient cache reuse."""
    total_size: int
    used_size: int
    free_blocks: List[Tuple[int, int]]  # (start_offset, size)
    allocated_blocks: Dict[str, Tuple[int, int]]  # key -> (start_offset, size)


class WorkloadAnalyzer:
    """Analyze workload patterns for cache optimization."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.access_patterns: deque = deque(maxlen=window_size)
        self.layer_access_freq: Dict[int, int] = defaultdict(int)
        self.sequence_length_dist: Dict[int, int] = defaultdict(int)
        self.computation_patterns: Dict[str, int] = defaultdict(int)

    def record_access(self, layer_idx: int, seq_len: int, access_type: str = "read"):
        """Record cache access pattern."""
        timestamp = time.time()
        self.access_patterns.append({
            'layer': layer_idx,
            'seq_len': seq_len,
            'type': access_type,
            'timestamp': timestamp
        })

        self.layer_access_freq[layer_idx] += 1
        self.sequence_length_dist[seq_len] += 1
        self.computation_patterns[access_type] += 1

    def predict_cache_size(self, target_layer: int) -> int:
        """Predict optimal cache size for a layer based on patterns."""
        if not self.access_patterns:
            return 1024  # Default

        # Analyze recent patterns for this layer
        layer_accesses = [p for p in self.access_patterns if p['layer'] == target_layer]

        if not layer_accesses:
            return 512  # Conservative default

        # Calculate average sequence length and access frequency
        avg_seq_len = sum(p['seq_len'] for p in layer_accesses) / len(layer_accesses)
        access_freq = len(layer_accesses) / self.window_size

        # Predict cache size based on patterns
        predicted_size = int(avg_seq_len * (1 + access_freq * 2))
        return max(256, min(predicted_size, 4096))  # Clamp to reasonable range

    def get_hot_layers(self, top_k: int = 3) -> List[int]:
        """Get most frequently accessed layers."""
        return sorted(self.layer_access_freq.keys(),
                     key=lambda x: self.layer_access_freq[x],
                     reverse=True)[:top_k]


class DynamicCacheAllocator:
    """
    Dynamic cache allocator with adaptive sizing and memory management.

    Features:
    - Workload-adaptive cache sizing
    - Memory pool management
    - Intelligent eviction policies
    - Distributed-aware allocation
    """

    def __init__(
        self,
        max_memory_mb: float = 1024.0,  # 1GB default
        world_size: int = 1,
        rank: int = 0,
        comm_handler: Optional[NCCLCommunicator] = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE
    ):
        """Initialize dynamic cache allocator.

        Args:
            max_memory_mb: Maximum memory allocation in MB
            world_size: Number of GPUs
            rank: Current GPU rank
            comm_handler: Communication handler
            eviction_policy: Cache eviction policy
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.world_size = world_size
        self.rank = rank
        self.comm_handler = comm_handler
        self.eviction_policy = eviction_policy

        # Memory management
        self.memory_pool = MemoryPool(
            total_size=self.max_memory_bytes,
            used_size=0,
            free_blocks=[(0, self.max_memory_bytes)],
            allocated_blocks={}
        )

        # Cache metadata
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.layer_allocations: Dict[int, int] = {}  # layer -> allocated_size

        # Workload analysis
        self.workload_analyzer = WorkloadAnalyzer()

        # Performance monitoring
        self.allocation_overhead = 0.0
        self.eviction_count = 0
        self.cache_hit_rate = 0.0

        # Adaptive parameters
        self.target_memory_utilization = 0.8  # 80% target utilization
        self.min_allocation_size = 64 * 1024  # 64KB minimum
        self.allocation_granularity = 4 * 1024  # 4KB granularity

        logger.info(f"Initialized DynamicCacheAllocator: {max_memory_mb}MB max, "
                   f"rank {rank}/{world_size}, policy {eviction_policy.value}")

    def allocate_cache(
        self,
        layer_idx: int,
        requested_size: int,
        priority: float = 1.0
    ) -> Optional[int]:
        """Allocate cache space for a layer.

        Args:
            layer_idx: Transformer layer index
            requested_size: Requested cache size in bytes
            priority: Allocation priority (0.0-1.0)

        Returns:
            Allocated offset or None if allocation failed
        """
        start_time = time.time()

        try:
            # Analyze workload for this layer
            predicted_size = self.workload_analyzer.predict_cache_size(layer_idx)
            adaptive_size = max(requested_size, predicted_size * 1024)  # Convert to bytes

            # Check if we need to evict entries
            available_space = self.memory_pool.total_size - self.memory_pool.used_size
            if adaptive_size > available_space:
                self._evict_entries(adaptive_size - available_space)

            # Try to allocate
            offset = self._allocate_memory_block(f"layer_{layer_idx}", adaptive_size)
            if offset is not None:
                # Record allocation
                self.layer_allocations[layer_idx] = adaptive_size

                # Create cache entry metadata
                entry = CacheEntry(
                    layer_idx=layer_idx,
                    seq_pos=0,  # Will be updated during use
                    size_bytes=adaptive_size,
                    access_count=0,
                    last_access=time.time(),
                    gpu_rank=self.rank,
                    compression_ratio=1.0,  # Will be updated with compression
                    priority_score=priority
                )
                self.cache_entries[f"layer_{layer_idx}"] = entry

                logger.debug(f"Allocated {adaptive_size} bytes for layer {layer_idx} at offset {offset}")
                return offset

            return None

        finally:
            # Track allocation overhead
            self.allocation_overhead += time.time() - start_time

    def deallocate_cache(self, layer_idx: int) -> bool:
        """Deallocate cache space for a layer.

        Args:
            layer_idx: Transformer layer index

        Returns:
            True if deallocation successful
        """
        key = f"layer_{layer_idx}"
        if key in self.memory_pool.allocated_blocks:
            # Free the memory block
            self._free_memory_block(key)

            # Remove metadata
            if key in self.cache_entries:
                del self.cache_entries[key]

            if layer_idx in self.layer_allocations:
                del self.layer_allocations[layer_idx]

            logger.debug(f"Deallocated cache for layer {layer_idx}")
            return True

        return False

    def record_access(self, layer_idx: int, seq_len: int, access_type: str = "read"):
        """Record cache access for workload analysis."""
        self.workload_analyzer.record_access(layer_idx, seq_len, access_type)

        # Update cache entry metadata
        key = f"layer_{layer_idx}"
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            entry.access_count += 1
            entry.last_access = time.time()
            entry.seq_pos = seq_len  # Update current sequence position

    def optimize_allocation(self):
        """Optimize cache allocation based on current workload patterns."""
        # Get hot layers
        hot_layers = self.workload_analyzer.get_hot_layers(top_k=5)

        # Redistribute memory to hot layers
        total_memory = self.memory_pool.total_size
        base_allocation = total_memory // len(hot_layers) if hot_layers else total_memory

        for layer_idx in hot_layers:
            predicted_size = self.workload_analyzer.predict_cache_size(layer_idx) * 1024
            optimal_size = min(predicted_size, base_allocation)

            # Resize allocation if needed
            self._resize_allocation(layer_idx, optimal_size)

    def _allocate_memory_block(self, key: str, size: int) -> Optional[int]:
        """Allocate a memory block from the pool."""
        # Align size to granularity
        aligned_size = math.ceil(size / self.allocation_granularity) * self.allocation_granularity
        aligned_size = max(aligned_size, self.min_allocation_size)

        # Find suitable free block (first fit)
        for i, (offset, block_size) in enumerate(self.memory_pool.free_blocks):
            if block_size >= aligned_size:
                # Allocate from this block
                allocated_offset = offset
                remaining_size = block_size - aligned_size

                # Update free blocks
                del self.memory_pool.free_blocks[i]
                if remaining_size > 0:
                    self.memory_pool.free_blocks.insert(i, (offset + aligned_size, remaining_size))

                # Record allocation
                self.memory_pool.allocated_blocks[key] = (allocated_offset, aligned_size)
                self.memory_pool.used_size += aligned_size

                return allocated_offset

        return None  # No suitable block found

    def _free_memory_block(self, key: str):
        """Free a memory block back to the pool."""
        if key in self.memory_pool.allocated_blocks:
            offset, size = self.memory_pool.allocated_blocks[key]

            # Add back to free blocks
            self.memory_pool.free_blocks.append((offset, size))
            self.memory_pool.used_size -= size

            # Merge adjacent free blocks
            self._merge_free_blocks()

            del self.memory_pool.allocated_blocks[key]

    def _merge_free_blocks(self):
        """Merge adjacent free memory blocks."""
        self.memory_pool.free_blocks.sort(key=lambda x: x[0])  # Sort by offset

        merged = []
        current_offset, current_size = self.memory_pool.free_blocks[0]

        for offset, size in self.memory_pool.free_blocks[1:]:
            if offset == current_offset + current_size:
                # Merge adjacent blocks
                current_size += size
            else:
                # Add current block and start new one
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size

        merged.append((current_offset, current_size))
        self.memory_pool.free_blocks = merged

    def _evict_entries(self, required_space: int):
        """Evict cache entries to free up space."""
        if self.eviction_policy == EvictionPolicy.LRU:
            # Sort by last access time (oldest first)
            victims = sorted(
                self.cache_entries.values(),
                key=lambda x: x.last_access
            )
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Sort by access count (least used first)
            victims = sorted(
                self.cache_entries.values(),
                key=lambda x: x.access_count
            )
        elif self.eviction_policy == EvictionPolicy.SIZE_AWARE:
            # Sort by size (largest first, but consider access patterns)
            victims = sorted(
                self.cache_entries.values(),
                key=lambda x: (x.size_bytes / (x.access_count + 1), -x.last_access)
            )
        elif self.eviction_policy == EvictionPolicy.DISTRIBUTED_AWARE:
            # Consider GPU distribution and cross-GPU access patterns
            victims = self._distributed_aware_eviction()
        else:  # ADAPTIVE
            victims = self._adaptive_eviction()

        # Evict entries until we have enough space
        freed_space = 0
        for victim in victims:
            if freed_space >= required_space:
                break

            # Deallocate this entry
            self.deallocate_cache(victim.layer_idx)
            freed_space += victim.size_bytes
            self.eviction_count += 1

            logger.debug(f"Evicted cache entry for layer {victim.layer_idx}")

    def _distributed_aware_eviction(self) -> List[CacheEntry]:
        """Distributed-aware eviction considering cross-GPU access."""
        # In distributed setting, prefer to keep entries that are accessed
        # frequently across multiple GPUs
        entries = list(self.cache_entries.values())

        # Sort by priority score (considering distributed access patterns)
        # This is a simplified version - in practice, would track cross-GPU access
        return sorted(entries, key=lambda x: x.priority_score)

    def _adaptive_eviction(self) -> List[CacheEntry]:
        """Adaptive eviction based on learned patterns."""
        entries = list(self.cache_entries.values())

        # Use a combination of LRU, LFU, and size factors
        return sorted(entries, key=lambda x: (
            x.last_access,  # LRU factor
            1.0 / (x.access_count + 1),  # LFU factor (inverse)
            x.size_bytes  # Size factor
        ))

    def _resize_allocation(self, layer_idx: int, new_size: int):
        """Resize allocation for a layer."""
        current_size = self.layer_allocations.get(layer_idx, 0)
        if current_size == new_size:
            return

        # Deallocate current and allocate new
        self.deallocate_cache(layer_idx)
        self.allocate_cache(layer_idx, new_size)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'total_memory_mb': self.memory_pool.total_size / (1024 * 1024),
            'used_memory_mb': self.memory_pool.used_size / (1024 * 1024),
            'utilization_percent': (self.memory_pool.used_size / self.memory_pool.total_size) * 100,
            'free_blocks': len(self.memory_pool.free_blocks),
            'allocated_blocks': len(self.memory_pool.allocated_blocks),
            'allocation_overhead_ms': self.allocation_overhead * 1000,
            'eviction_count': self.eviction_count,
            'layer_allocations': dict(self.layer_allocations)
        }

    def get_workload_stats(self) -> Dict[str, Any]:
        """Get workload analysis statistics."""
        return {
            'layer_access_freq': dict(self.workload_analyzer.layer_access_freq),
            'sequence_length_dist': dict(self.workload_analyzer.sequence_length_dist),
            'computation_patterns': dict(self.workload_analyzer.computation_patterns),
            'hot_layers': self.workload_analyzer.get_hot_layers(5)
        }
