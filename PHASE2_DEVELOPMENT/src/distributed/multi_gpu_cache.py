"""
Multi-GPU Distributed KV Cache System

Provides distributed caching across multiple GPUs with:
- Cross-GPU page sharing and migration
- Cache coherency protocols
- Memory-efficient distributed allocation
- Load-balanced page distribution

Integrates with Sprint 2.2 Days 5-9 cache compression work.

Copyright (c) 2025. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import logging

# Type hints for torch (actual import happens at runtime)
try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    dist = None

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================

class CacheCoherencyProtocol(Enum):
    """Cache coherency protocols for multi-GPU synchronization."""
    MSI = auto()        # Modified, Shared, Invalid (simple)
    MESI = auto()       # Modified, Exclusive, Shared, Invalid
    MOESI = auto()      # Modified, Owned, Exclusive, Shared, Invalid
    DIRECTORY = auto()  # Directory-based coherency (scalable)


class PageState(Enum):
    """State of a cache page in coherency protocol."""
    INVALID = auto()
    SHARED = auto()
    EXCLUSIVE = auto()
    MODIFIED = auto()
    OWNED = auto()  # For MOESI only


# Alias for MESI-specific state (subset of PageState)
MESIState = PageState


class MigrationPolicy(Enum):
    """Policy for migrating pages between GPUs."""
    LAZY = auto()        # Migrate only when needed
    EAGER = auto()       # Proactively migrate for locality
    THRESHOLD = auto()   # Migrate when access count exceeds threshold
    PREDICTIVE = auto()  # Use access patterns to predict migration


class AllocationStrategy(Enum):
    """Strategy for allocating new pages across GPUs."""
    ROUND_ROBIN = auto()      # Simple round-robin
    LEAST_LOADED = auto()     # GPU with most free memory
    LOCALITY_AWARE = auto()   # Prefer GPU that will use it most
    HASH_BASED = auto()       # Consistent hashing for distribution


class EvictionPolicy(Enum):
    """Eviction policy for distributed cache."""
    LOCAL_LRU = auto()        # LRU within each GPU
    GLOBAL_LRU = auto()       # Global LRU across all GPUs
    LOCALITY_FIRST = auto()   # Evict remote copies first
    COST_BASED = auto()       # Consider transfer costs


@dataclass
class DistributedCacheConfig:
    """Configuration for multi-GPU distributed cache."""
    # GPU configuration
    num_gpus: int = 4
    gpu_ids: Optional[List[int]] = None
    
    # Memory configuration (per GPU, in MB)
    cache_size_per_gpu_mb: int = 4096  # 4GB per GPU
    page_size_kb: int = 64
    
    # Coherency and migration
    coherency_protocol: CacheCoherencyProtocol = CacheCoherencyProtocol.MESI
    migration_policy: MigrationPolicy = MigrationPolicy.THRESHOLD
    migration_threshold: int = 10  # Access count threshold for migration
    
    # Allocation and eviction
    allocation_strategy: AllocationStrategy = AllocationStrategy.LOCALITY_AWARE
    eviction_policy: EvictionPolicy = EvictionPolicy.LOCALITY_FIRST
    eviction_watermark: float = 0.9  # Start eviction at 90% capacity
    
    # Performance tuning
    prefetch_enabled: bool = True
    prefetch_window: int = 4  # Prefetch next N pages
    async_transfers: bool = True
    transfer_overlap: bool = True  # Overlap computation with transfers
    
    # Compression (integrates with Days 5-9 work)
    compression_enabled: bool = True
    compression_ratio_target: float = 4.0
    
    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = list(range(self.num_gpus))
        else:
            self.num_gpus = len(self.gpu_ids)
    
    @property
    def pages_per_gpu(self) -> int:
        """Number of pages per GPU."""
        return (self.cache_size_per_gpu_mb * 1024) // self.page_size_kb
    
    @property
    def total_pages(self) -> int:
        """Total pages across all GPUs."""
        return self.pages_per_gpu * self.num_gpus


@dataclass
class PageMetadata:
    """Metadata for a cache page."""
    page_id: str
    sequence_id: int
    layer_id: int
    owner_gpu: int
    state: PageState = PageState.INVALID
    sharers: Set[int] = field(default_factory=set)
    access_count: int = 0
    last_access_time: float = 0.0
    creation_time: float = field(default_factory=time.time)
    size_bytes: int = 0
    is_compressed: bool = False
    compression_ratio: float = 1.0
    
    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_access_time = time.time()


@dataclass
class CacheStatistics:
    """Statistics for distributed cache performance."""
    # Hit/miss statistics
    local_hits: int = 0
    remote_hits: int = 0
    misses: int = 0
    
    # Transfer statistics
    pages_transferred: int = 0
    bytes_transferred: int = 0
    transfer_time_ms: float = 0.0
    
    # Memory statistics
    total_pages_allocated: int = 0
    pages_evicted: int = 0
    memory_used_mb: float = 0.0
    
    # Migration statistics
    pages_migrated: int = 0
    migrations_avoided: int = 0
    
    # Coherency statistics
    invalidations_sent: int = 0
    invalidations_received: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total = self.local_hits + self.remote_hits + self.misses
        if total == 0:
            return 0.0
        return (self.local_hits + self.remote_hits) / total
    
    @property
    def local_hit_rate(self) -> float:
        """Calculate local hit rate (excludes remote)."""
        total = self.local_hits + self.remote_hits + self.misses
        if total == 0:
            return 0.0
        return self.local_hits / total
    
    @property
    def avg_transfer_latency_ms(self) -> float:
        """Average latency per transfer."""
        if self.pages_transferred == 0:
            return 0.0
        return self.transfer_time_ms / self.pages_transferred


# =============================================================================
# Coherency Protocol Implementations
# =============================================================================

class CoherencyProtocolBase(ABC):
    """Base class for cache coherency protocols."""
    
    def __init__(self, config: DistributedCacheConfig):
        self.config = config
        self.lock = threading.RLock()
    
    @abstractmethod
    def handle_read(
        self,
        page: PageMetadata,
        requester_gpu: int
    ) -> Tuple[PageState, Set[int]]:
        """Handle a read request and return new state and GPUs to notify."""
        pass
    
    @abstractmethod
    def handle_write(
        self,
        page: PageMetadata,
        requester_gpu: int
    ) -> Tuple[PageState, Set[int]]:
        """Handle a write request and return new state and GPUs to invalidate."""
        pass
    
    @abstractmethod
    def handle_eviction(
        self,
        page: PageMetadata,
        evicting_gpu: int
    ) -> Set[int]:
        """Handle page eviction and return GPUs to notify."""
        pass


class MESIProtocol(CoherencyProtocolBase):
    """
    MESI coherency protocol implementation.
    
    States:
    - Modified: Only one cache has valid copy, dirty
    - Exclusive: Only one cache has valid copy, clean
    - Shared: Multiple caches may have valid copies
    - Invalid: Cache does not have valid copy
    """
    
    def handle_read(
        self,
        page: PageMetadata,
        requester_gpu: int
    ) -> Tuple[PageState, Set[int]]:
        """Handle read request in MESI protocol."""
        with self.lock:
            gpus_to_notify: Set[int] = set()
            
            if page.state == PageState.INVALID:
                # Load from memory, exclusive access
                page.state = PageState.EXCLUSIVE
                page.owner_gpu = requester_gpu
                page.sharers = {requester_gpu}
                
            elif page.state == PageState.EXCLUSIVE:
                if requester_gpu != page.owner_gpu:
                    # Another GPU wants to read - move to Shared
                    page.state = PageState.SHARED
                    page.sharers.add(requester_gpu)
                    gpus_to_notify.add(page.owner_gpu)
                    
            elif page.state == PageState.SHARED:
                # Add to sharers
                page.sharers.add(requester_gpu)
                
            elif page.state == PageState.MODIFIED:
                if requester_gpu != page.owner_gpu:
                    # Write back and move to Shared
                    page.state = PageState.SHARED
                    page.sharers.add(requester_gpu)
                    gpus_to_notify.add(page.owner_gpu)
            
            page.touch()
            return page.state, gpus_to_notify
    
    def handle_write(
        self,
        page: PageMetadata,
        requester_gpu: int
    ) -> Tuple[PageState, Set[int]]:
        """Handle write request in MESI protocol."""
        with self.lock:
            gpus_to_invalidate: Set[int] = set()
            
            if page.state == PageState.INVALID:
                # Load and modify
                page.state = PageState.MODIFIED
                page.owner_gpu = requester_gpu
                page.sharers = {requester_gpu}
                
            elif page.state == PageState.EXCLUSIVE:
                if requester_gpu == page.owner_gpu:
                    # Can modify without notification
                    page.state = PageState.MODIFIED
                else:
                    # Invalidate current owner
                    gpus_to_invalidate.add(page.owner_gpu)
                    page.state = PageState.MODIFIED
                    page.owner_gpu = requester_gpu
                    page.sharers = {requester_gpu}
                    
            elif page.state == PageState.SHARED:
                # Invalidate all sharers except requester
                gpus_to_invalidate = page.sharers - {requester_gpu}
                page.state = PageState.MODIFIED
                page.owner_gpu = requester_gpu
                page.sharers = {requester_gpu}
                
            elif page.state == PageState.MODIFIED:
                if requester_gpu != page.owner_gpu:
                    # Write back and take ownership
                    gpus_to_invalidate.add(page.owner_gpu)
                    page.owner_gpu = requester_gpu
                    page.sharers = {requester_gpu}
            
            page.touch()
            return page.state, gpus_to_invalidate
    
    def handle_eviction(
        self,
        page: PageMetadata,
        evicting_gpu: int
    ) -> Set[int]:
        """Handle eviction in MESI protocol."""
        with self.lock:
            gpus_to_notify: Set[int] = set()
            
            if page.state == PageState.MODIFIED and evicting_gpu == page.owner_gpu:
                # Must write back before eviction
                gpus_to_notify.add(evicting_gpu)  # Signal write-back needed
            
            page.sharers.discard(evicting_gpu)
            
            if not page.sharers:
                page.state = PageState.INVALID
            elif len(page.sharers) == 1:
                # Single sharer becomes exclusive owner
                remaining_gpu = next(iter(page.sharers))
                page.owner_gpu = remaining_gpu
                page.state = PageState.EXCLUSIVE
                gpus_to_notify.add(remaining_gpu)
            
            return gpus_to_notify


class DirectoryProtocol(CoherencyProtocolBase):
    """
    Directory-based coherency protocol for scalability.
    
    Uses a central directory to track page states,
    more scalable than snooping-based protocols.
    """
    
    def __init__(self, config: DistributedCacheConfig):
        super().__init__(config)
        # Directory entries: page_id -> set of sharers
        self.directory: Dict[str, Set[int]] = defaultdict(set)
        self.owners: Dict[str, int] = {}
        self.dirty: Dict[str, bool] = defaultdict(bool)
    
    def handle_read(
        self,
        page: PageMetadata,
        requester_gpu: int
    ) -> Tuple[PageState, Set[int]]:
        """Handle read request with directory lookup."""
        with self.lock:
            gpus_to_notify: Set[int] = set()
            page_id = page.page_id
            
            if page_id not in self.directory or not self.directory[page_id]:
                # First access - exclusive
                self.directory[page_id] = {requester_gpu}
                self.owners[page_id] = requester_gpu
                self.dirty[page_id] = False
                page.state = PageState.EXCLUSIVE
                
            elif self.dirty.get(page_id, False):
                # Current owner has dirty copy - request sharing
                owner = self.owners[page_id]
                gpus_to_notify.add(owner)
                self.directory[page_id].add(requester_gpu)
                self.dirty[page_id] = False
                page.state = PageState.SHARED
                
            else:
                # Clean - just add sharer
                self.directory[page_id].add(requester_gpu)
                page.state = PageState.SHARED
            
            page.sharers = self.directory[page_id].copy()
            page.touch()
            return page.state, gpus_to_notify
    
    def handle_write(
        self,
        page: PageMetadata,
        requester_gpu: int
    ) -> Tuple[PageState, Set[int]]:
        """Handle write request with directory invalidation."""
        with self.lock:
            page_id = page.page_id
            
            # Invalidate all other sharers
            current_sharers = self.directory.get(page_id, set())
            gpus_to_invalidate = current_sharers - {requester_gpu}
            
            # Update directory
            self.directory[page_id] = {requester_gpu}
            self.owners[page_id] = requester_gpu
            self.dirty[page_id] = True
            
            page.state = PageState.MODIFIED
            page.owner_gpu = requester_gpu
            page.sharers = {requester_gpu}
            page.touch()
            
            return page.state, gpus_to_invalidate
    
    def handle_eviction(
        self,
        page: PageMetadata,
        evicting_gpu: int
    ) -> Set[int]:
        """Handle eviction with directory update."""
        with self.lock:
            gpus_to_notify: Set[int] = set()
            page_id = page.page_id
            
            if page_id in self.directory:
                self.directory[page_id].discard(evicting_gpu)
                
                if self.dirty.get(page_id, False) and self.owners.get(page_id) == evicting_gpu:
                    gpus_to_notify.add(evicting_gpu)  # Write-back needed
                
                if not self.directory[page_id]:
                    # No more sharers
                    del self.directory[page_id]
                    self.owners.pop(page_id, None)
                    self.dirty.pop(page_id, None)
                elif len(self.directory[page_id]) == 1:
                    # Single sharer becomes owner
                    new_owner = next(iter(self.directory[page_id]))
                    self.owners[page_id] = new_owner
                    gpus_to_notify.add(new_owner)
            
            return gpus_to_notify


def create_coherency_protocol(
    config: DistributedCacheConfig
) -> CoherencyProtocolBase:
    """Factory function for coherency protocols."""
    protocol_map = {
        CacheCoherencyProtocol.MSI: MESIProtocol,  # Use MESI for MSI
        CacheCoherencyProtocol.MESI: MESIProtocol,
        CacheCoherencyProtocol.MOESI: MESIProtocol,  # Extend later
        CacheCoherencyProtocol.DIRECTORY: DirectoryProtocol,
    }
    
    protocol_class = protocol_map.get(
        config.coherency_protocol, MESIProtocol
    )
    return protocol_class(config)


# =============================================================================
# GPU Memory Allocator
# =============================================================================

class GPUPageAllocator:
    """
    Manages page allocation on a single GPU.
    
    Integrates with compression from Days 5-9 work.
    """
    
    def __init__(
        self,
        gpu_id: int,
        config: DistributedCacheConfig
    ):
        self.gpu_id = gpu_id
        self.config = config
        self.lock = threading.RLock()
        
        # Memory tracking
        self.total_pages = config.pages_per_gpu
        self.allocated_pages: Dict[str, PageMetadata] = {}
        self.free_pages = self.total_pages
        
        # LRU tracking
        self.access_order: List[str] = []
        
        # Statistics
        self.allocations = 0
        self.evictions = 0
    
    def allocate(
        self,
        page_id: str,
        sequence_id: int,
        layer_id: int,
        size_bytes: int,
        compressed: bool = False,
        compression_ratio: float = 1.0
    ) -> Optional[PageMetadata]:
        """Allocate a new page on this GPU."""
        with self.lock:
            if self.free_pages <= 0:
                return None
            
            metadata = PageMetadata(
                page_id=page_id,
                sequence_id=sequence_id,
                layer_id=layer_id,
                owner_gpu=self.gpu_id,
                state=PageState.EXCLUSIVE,
                sharers={self.gpu_id},
                size_bytes=size_bytes,
                is_compressed=compressed,
                compression_ratio=compression_ratio
            )
            
            self.allocated_pages[page_id] = metadata
            self.access_order.append(page_id)
            self.free_pages -= 1
            self.allocations += 1
            
            return metadata
    
    def deallocate(self, page_id: str) -> bool:
        """Deallocate a page from this GPU."""
        with self.lock:
            if page_id not in self.allocated_pages:
                return False
            
            del self.allocated_pages[page_id]
            if page_id in self.access_order:
                self.access_order.remove(page_id)
            self.free_pages += 1
            self.evictions += 1
            
            return True
    
    def touch(self, page_id: str) -> None:
        """Update access order for LRU."""
        with self.lock:
            if page_id in self.access_order:
                self.access_order.remove(page_id)
                self.access_order.append(page_id)
            if page_id in self.allocated_pages:
                self.allocated_pages[page_id].touch()
    
    def get_lru_candidates(self, count: int = 1) -> List[str]:
        """Get least recently used pages for eviction."""
        with self.lock:
            return self.access_order[:count]
    
    @property
    def utilization(self) -> float:
        """Current memory utilization ratio."""
        return (self.total_pages - self.free_pages) / self.total_pages
    
    @property
    def memory_used_mb(self) -> float:
        """Memory used in MB."""
        with self.lock:
            total_bytes = sum(
                p.size_bytes for p in self.allocated_pages.values()
            )
            return total_bytes / (1024 * 1024)


# =============================================================================
# Cross-GPU Page Transfer
# =============================================================================

class PageTransferManager:
    """
    Manages page transfers between GPUs.
    
    Supports both synchronous and asynchronous transfers
    with optional overlap with computation.
    """
    
    def __init__(self, config: DistributedCacheConfig):
        self.config = config
        self.lock = threading.Lock()
        self.pending_transfers: Dict[str, asyncio.Future] = {}
        self.transfer_stats = CacheStatistics()
        
        # Transfer streams per GPU pair (for overlap)
        self.streams: Dict[Tuple[int, int], Any] = {}
    
    def _get_stream(self, src_gpu: int, dst_gpu: int) -> Any:
        """Get or create CUDA stream for GPU pair."""
        key = (src_gpu, dst_gpu)
        if key not in self.streams and TORCH_AVAILABLE:
            with torch.cuda.device(dst_gpu):
                self.streams[key] = torch.cuda.Stream()
        return self.streams.get(key)
    
    def transfer_sync(
        self,
        data: Any,  # torch.Tensor
        src_gpu: int,
        dst_gpu: int,
        page_metadata: PageMetadata
    ) -> Any:
        """Synchronous page transfer between GPUs."""
        start_time = time.time()
        
        if not TORCH_AVAILABLE:
            # Simulate transfer for testing
            time.sleep(0.001)
            return data
        
        with torch.cuda.device(dst_gpu):
            if src_gpu == dst_gpu:
                result = data
            else:
                result = data.to(f"cuda:{dst_gpu}")
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        with self.lock:
            self.transfer_stats.pages_transferred += 1
            self.transfer_stats.bytes_transferred += page_metadata.size_bytes
            self.transfer_stats.transfer_time_ms += elapsed_ms
        
        return result
    
    async def transfer_async(
        self,
        data: Any,  # torch.Tensor
        src_gpu: int,
        dst_gpu: int,
        page_metadata: PageMetadata
    ) -> Any:
        """Asynchronous page transfer using CUDA streams."""
        if not TORCH_AVAILABLE or not self.config.async_transfers:
            return self.transfer_sync(data, src_gpu, dst_gpu, page_metadata)
        
        start_time = time.time()
        stream = self._get_stream(src_gpu, dst_gpu)
        
        with torch.cuda.device(dst_gpu):
            with torch.cuda.stream(stream):
                if src_gpu == dst_gpu:
                    result = data
                else:
                    result = data.to(f"cuda:{dst_gpu}", non_blocking=True)
        
        # Wait for transfer completion
        stream.synchronize()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        with self.lock:
            self.transfer_stats.pages_transferred += 1
            self.transfer_stats.bytes_transferred += page_metadata.size_bytes
            self.transfer_stats.transfer_time_ms += elapsed_ms
        
        return result
    
    def prefetch(
        self,
        page_ids: List[str],
        src_gpu: int,
        dst_gpu: int,
        page_store: Dict[str, Tuple[Any, PageMetadata]]
    ) -> None:
        """Prefetch pages asynchronously."""
        if not self.config.prefetch_enabled:
            return
        
        if not TORCH_AVAILABLE:
            return
        
        stream = self._get_stream(src_gpu, dst_gpu)
        
        with torch.cuda.device(dst_gpu):
            with torch.cuda.stream(stream):
                for page_id in page_ids:
                    if page_id in page_store:
                        data, metadata = page_store[page_id]
                        if hasattr(data, 'to'):
                            # Non-blocking transfer
                            data.to(f"cuda:{dst_gpu}", non_blocking=True)


# =============================================================================
# Distributed Cache Manager
# =============================================================================

class DistributedKVCache:
    """
    Multi-GPU distributed KV cache with coherency and migration.
    
    Main entry point for the distributed cache system.
    """
    
    def __init__(self, config: DistributedCacheConfig):
        self.config = config
        self.lock = threading.RLock()
        
        # Per-GPU allocators
        self.allocators: Dict[int, GPUPageAllocator] = {}
        for gpu_id in config.gpu_ids:
            self.allocators[gpu_id] = GPUPageAllocator(gpu_id, config)
        
        # Coherency protocol
        self.coherency = create_coherency_protocol(config)
        
        # Transfer manager
        self.transfer_manager = PageTransferManager(config)
        
        # Page data storage: page_id -> (data, metadata)
        self.page_store: Dict[str, Tuple[Any, PageMetadata]] = {}
        
        # GPU-local views: gpu_id -> set of page_ids
        self.gpu_pages: Dict[int, Set[str]] = defaultdict(set)
        
        # Global statistics
        self.stats = CacheStatistics()
        
        # Allocation index for round-robin
        self._next_gpu_idx = 0
        
        # Access pattern tracking for locality
        self.access_patterns: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
    
    def _select_allocation_gpu(
        self,
        sequence_id: int,
        layer_id: int,
        preferred_gpu: Optional[int] = None
    ) -> int:
        """Select GPU for new page allocation."""
        strategy = self.config.allocation_strategy
        
        if preferred_gpu is not None and preferred_gpu in self.allocators:
            allocator = self.allocators[preferred_gpu]
            if allocator.utilization < self.config.eviction_watermark:
                return preferred_gpu
        
        if strategy == AllocationStrategy.ROUND_ROBIN:
            gpu = self.config.gpu_ids[self._next_gpu_idx]
            self._next_gpu_idx = (
                (self._next_gpu_idx + 1) % len(self.config.gpu_ids)
            )
            return gpu
        
        elif strategy == AllocationStrategy.LEAST_LOADED:
            min_util = float('inf')
            selected = self.config.gpu_ids[0]
            for gpu_id in self.config.gpu_ids:
                util = self.allocators[gpu_id].utilization
                if util < min_util:
                    min_util = util
                    selected = gpu_id
            return selected
        
        elif strategy == AllocationStrategy.LOCALITY_AWARE:
            # Use access patterns to determine locality
            max_accesses = -1
            selected = self.config.gpu_ids[0]
            
            key = f"{sequence_id}:{layer_id}"
            for gpu_id in self.config.gpu_ids:
                accesses = self.access_patterns[gpu_id].get(key, 0)
                allocator = self.allocators[gpu_id]
                if (accesses > max_accesses and 
                    allocator.utilization < self.config.eviction_watermark):
                    max_accesses = accesses
                    selected = gpu_id
            return selected
        
        elif strategy == AllocationStrategy.HASH_BASED:
            # Consistent hashing based on page attributes
            hash_input = f"{sequence_id}:{layer_id}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            return self.config.gpu_ids[hash_val % len(self.config.gpu_ids)]
        
        return self.config.gpu_ids[0]
    
    def _evict_if_needed(self, gpu_id: int) -> None:
        """Evict pages if GPU memory is above watermark."""
        allocator = self.allocators[gpu_id]
        
        while allocator.utilization >= self.config.eviction_watermark:
            candidates = allocator.get_lru_candidates(1)
            if not candidates:
                break
            
            page_id = candidates[0]
            
            # Handle coherency for eviction
            if page_id in self.page_store:
                _, metadata = self.page_store[page_id]
                gpus_to_notify = self.coherency.handle_eviction(
                    metadata, gpu_id
                )
                
                # If modified, might need write-back (logged for now)
                if gpu_id in gpus_to_notify:
                    logger.debug(
                        f"Write-back needed for page {page_id} on GPU {gpu_id}"
                    )
                
                # Remove from GPU-local view
                self.gpu_pages[gpu_id].discard(page_id)
                
                # If no more sharers, remove from global store
                if not metadata.sharers:
                    del self.page_store[page_id]
            
            allocator.deallocate(page_id)
            self.stats.pages_evicted += 1
    
    def put(
        self,
        key_data: Any,  # torch.Tensor
        value_data: Any,  # torch.Tensor
        sequence_id: int,
        layer_id: int,
        position: int,
        requesting_gpu: int,
        compressed: bool = False,
        compression_ratio: float = 1.0
    ) -> str:
        """
        Store KV data in the distributed cache.
        
        Returns page_id for later retrieval.
        """
        with self.lock:
            # Generate page ID
            page_id = f"kv:{sequence_id}:{layer_id}:{position}"
            
            # Select GPU for allocation
            target_gpu = self._select_allocation_gpu(
                sequence_id, layer_id, requesting_gpu
            )
            
            # Evict if needed
            self._evict_if_needed(target_gpu)
            
            # Calculate size
            size_bytes = 0
            if TORCH_AVAILABLE and hasattr(key_data, 'element_size'):
                size_bytes = (
                    key_data.numel() * key_data.element_size() +
                    value_data.numel() * value_data.element_size()
                )
            else:
                size_bytes = self.config.page_size_kb * 1024
            
            # Allocate page
            metadata = self.allocators[target_gpu].allocate(
                page_id=page_id,
                sequence_id=sequence_id,
                layer_id=layer_id,
                size_bytes=size_bytes,
                compressed=compressed,
                compression_ratio=compression_ratio
            )
            
            if metadata is None:
                logger.warning(
                    f"Failed to allocate page on GPU {target_gpu}"
                )
                return ""
            
            # Transfer data to target GPU if needed
            if TORCH_AVAILABLE and target_gpu != requesting_gpu:
                key_data = self.transfer_manager.transfer_sync(
                    key_data, requesting_gpu, target_gpu, metadata
                )
                value_data = self.transfer_manager.transfer_sync(
                    value_data, requesting_gpu, target_gpu, metadata
                )
            
            # Store in global store
            combined_data = (key_data, value_data)
            self.page_store[page_id] = (combined_data, metadata)
            self.gpu_pages[target_gpu].add(page_id)
            
            # Update statistics
            self.stats.total_pages_allocated += 1
            
            # Track access pattern
            pattern_key = f"{sequence_id}:{layer_id}"
            self.access_patterns[requesting_gpu][pattern_key] += 1
            
            return page_id
    
    def get(
        self,
        page_id: str,
        requesting_gpu: int
    ) -> Optional[Tuple[Any, Any]]:
        """
        Retrieve KV data from the distributed cache.
        
        Returns (key_data, value_data) tuple or None if not found.
        """
        with self.lock:
            if page_id not in self.page_store:
                self.stats.misses += 1
                return None
            
            combined_data, metadata = self.page_store[page_id]
            key_data, value_data = combined_data
            
            # Handle coherency
            new_state, gpus_to_notify = self.coherency.handle_read(
                metadata, requesting_gpu
            )
            
            # Update statistics
            if requesting_gpu == metadata.owner_gpu:
                self.stats.local_hits += 1
            else:
                self.stats.remote_hits += 1
                
                # Transfer data if on different GPU
                if TORCH_AVAILABLE:
                    key_data = self.transfer_manager.transfer_sync(
                        key_data, metadata.owner_gpu, requesting_gpu, metadata
                    )
                    value_data = self.transfer_manager.transfer_sync(
                        value_data, metadata.owner_gpu, requesting_gpu, metadata
                    )
            
            # Add to local view if sharing
            if requesting_gpu != metadata.owner_gpu:
                self.gpu_pages[requesting_gpu].add(page_id)
            
            # Touch for LRU
            self.allocators[metadata.owner_gpu].touch(page_id)
            
            # Track access pattern
            pattern_key = f"{metadata.sequence_id}:{metadata.layer_id}"
            self.access_patterns[requesting_gpu][pattern_key] += 1
            
            # Check migration threshold
            if (self.config.migration_policy == MigrationPolicy.THRESHOLD and
                metadata.access_count >= self.config.migration_threshold and
                requesting_gpu != metadata.owner_gpu):
                self._migrate_page(page_id, requesting_gpu)
            
            # Prefetch next pages
            if self.config.prefetch_enabled:
                next_positions = [
                    f"kv:{metadata.sequence_id}:{metadata.layer_id}:{i}"
                    for i in range(
                        metadata.layer_id + 1,
                        metadata.layer_id + 1 + self.config.prefetch_window
                    )
                ]
                self.transfer_manager.prefetch(
                    next_positions,
                    metadata.owner_gpu,
                    requesting_gpu,
                    self.page_store
                )
            
            return (key_data, value_data)
    
    def _migrate_page(self, page_id: str, target_gpu: int) -> bool:
        """Migrate a page to a different GPU for locality."""
        if page_id not in self.page_store:
            return False
        
        combined_data, metadata = self.page_store[page_id]
        old_gpu = metadata.owner_gpu
        
        if old_gpu == target_gpu:
            self.stats.migrations_avoided += 1
            return True
        
        # Check if target has space
        target_allocator = self.allocators[target_gpu]
        if target_allocator.utilization >= self.config.eviction_watermark:
            self._evict_if_needed(target_gpu)
        
        # Transfer data
        key_data, value_data = combined_data
        if TORCH_AVAILABLE:
            key_data = self.transfer_manager.transfer_sync(
                key_data, old_gpu, target_gpu, metadata
            )
            value_data = self.transfer_manager.transfer_sync(
                value_data, old_gpu, target_gpu, metadata
            )
        
        # Update metadata
        metadata.owner_gpu = target_gpu
        metadata.state = PageState.EXCLUSIVE
        metadata.sharers = {target_gpu}
        
        # Update allocators
        self.allocators[old_gpu].deallocate(page_id)
        self.allocators[target_gpu].allocate(
            page_id=page_id,
            sequence_id=metadata.sequence_id,
            layer_id=metadata.layer_id,
            size_bytes=metadata.size_bytes,
            compressed=metadata.is_compressed,
            compression_ratio=metadata.compression_ratio
        )
        
        # Update GPU views
        self.gpu_pages[old_gpu].discard(page_id)
        self.gpu_pages[target_gpu].add(page_id)
        
        # Store updated data
        self.page_store[page_id] = ((key_data, value_data), metadata)
        
        self.stats.pages_migrated += 1
        logger.debug(
            f"Migrated page {page_id} from GPU {old_gpu} to GPU {target_gpu}"
        )
        
        return True
    
    def invalidate(
        self,
        page_id: str,
        gpu_id: Optional[int] = None
    ) -> bool:
        """Invalidate a cache page."""
        with self.lock:
            if page_id not in self.page_store:
                return False
            
            _, metadata = self.page_store[page_id]
            
            if gpu_id is not None:
                # Invalidate only on specific GPU
                self.gpu_pages[gpu_id].discard(page_id)
                metadata.sharers.discard(gpu_id)
                
                if gpu_id == metadata.owner_gpu:
                    if metadata.sharers:
                        # Transfer ownership
                        new_owner = next(iter(metadata.sharers))
                        metadata.owner_gpu = new_owner
                        metadata.state = PageState.EXCLUSIVE
                    else:
                        # Remove entirely
                        del self.page_store[page_id]
                
                self.allocators[gpu_id].deallocate(page_id)
            else:
                # Invalidate everywhere
                for gid in list(metadata.sharers):
                    self.gpu_pages[gid].discard(page_id)
                    self.allocators[gid].deallocate(page_id)
                
                del self.page_store[page_id]
            
            self.stats.invalidations_sent += 1
            return True
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self.lock:
            self.page_store.clear()
            for gpu_id in self.config.gpu_ids:
                self.gpu_pages[gpu_id].clear()
                self.allocators[gpu_id] = GPUPageAllocator(gpu_id, self.config)
            self.access_patterns.clear()
    
    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics."""
        with self.lock:
            # Aggregate memory usage
            self.stats.memory_used_mb = sum(
                alloc.memory_used_mb for alloc in self.allocators.values()
            )
            return self.stats
    
    def get_gpu_utilization(self) -> Dict[int, float]:
        """Get per-GPU memory utilization."""
        return {
            gpu_id: alloc.utilization
            for gpu_id, alloc in self.allocators.items()
        }


# =============================================================================
# Cache Sharding for Sequences
# =============================================================================

class SequenceAwareCacheSharding:
    """
    Shards cache pages across GPUs based on sequence affinity.
    
    Optimizes for common access patterns in LLM inference:
    - Same sequence accesses are colocated
    - Attention heads can be distributed
    """
    
    def __init__(
        self,
        cache: DistributedKVCache,
        num_attention_heads: int,
        num_layers: int
    ):
        self.cache = cache
        self.num_heads = num_attention_heads
        self.num_layers = num_layers
        
        # Sequence to GPU affinity
        self.sequence_affinity: Dict[int, int] = {}
        self._next_gpu_idx = 0
    
    def get_sequence_gpu(self, sequence_id: int) -> int:
        """Get the primary GPU for a sequence."""
        if sequence_id not in self.sequence_affinity:
            gpu = self.cache.config.gpu_ids[self._next_gpu_idx]
            self.sequence_affinity[sequence_id] = gpu
            self._next_gpu_idx = (
                (self._next_gpu_idx + 1) % len(self.cache.config.gpu_ids)
            )
        return self.sequence_affinity[sequence_id]
    
    def put_sequence_kv(
        self,
        sequence_id: int,
        layer_id: int,
        key_data: Any,
        value_data: Any,
        position: int,
        requesting_gpu: int,
        **kwargs
    ) -> str:
        """Store KV data with sequence-aware placement."""
        # Override requesting GPU with sequence affinity
        preferred_gpu = self.get_sequence_gpu(sequence_id)
        
        return self.cache.put(
            key_data=key_data,
            value_data=value_data,
            sequence_id=sequence_id,
            layer_id=layer_id,
            position=position,
            requesting_gpu=preferred_gpu,
            **kwargs
        )
    
    def rebalance_sequences(self) -> Dict[int, int]:
        """Rebalance sequence assignments for even distribution."""
        # Count pages per GPU
        gpu_page_counts = {
            gpu_id: len(pages)
            for gpu_id, pages in self.cache.gpu_pages.items()
        }
        
        # Find overloaded and underloaded GPUs
        avg_pages = sum(gpu_page_counts.values()) / len(gpu_page_counts)
        
        overloaded = [
            (gpu, count) for gpu, count in gpu_page_counts.items()
            if count > avg_pages * 1.2  # 20% threshold
        ]
        underloaded = [
            (gpu, count) for gpu, count in gpu_page_counts.items()
            if count < avg_pages * 0.8
        ]
        
        migrations = {}
        
        # Reassign sequences from overloaded to underloaded
        for seq_id, gpu_id in list(self.sequence_affinity.items()):
            if any(gpu_id == g for g, _ in overloaded) and underloaded:
                new_gpu, _ = underloaded[0]
                self.sequence_affinity[seq_id] = new_gpu
                migrations[seq_id] = new_gpu
        
        return migrations


# =============================================================================
# Factory Functions
# =============================================================================

def create_distributed_cache(
    num_gpus: int = 4,
    cache_size_per_gpu_mb: int = 4096,
    coherency: CacheCoherencyProtocol = CacheCoherencyProtocol.MESI,
    **kwargs
) -> DistributedKVCache:
    """
    Factory function to create a distributed KV cache.
    
    Args:
        num_gpus: Number of GPUs to use
        cache_size_per_gpu_mb: Cache size per GPU in MB
        coherency: Coherency protocol to use
        **kwargs: Additional configuration options
    
    Returns:
        Configured DistributedKVCache instance
    """
    config = DistributedCacheConfig(
        num_gpus=num_gpus,
        cache_size_per_gpu_mb=cache_size_per_gpu_mb,
        coherency_protocol=coherency,
        **kwargs
    )
    
    return DistributedKVCache(config)


def estimate_distributed_cache_memory(
    num_gpus: int,
    cache_size_per_gpu_mb: int,
    compression_ratio: float = 4.0,
    overhead_ratio: float = 0.1
) -> Dict[str, float]:
    """
    Estimate memory requirements for distributed cache.
    
    Returns:
        Dictionary with memory estimates in MB
    """
    base_memory = num_gpus * cache_size_per_gpu_mb
    compressed_effective = base_memory * compression_ratio
    overhead = base_memory * overhead_ratio
    
    return {
        "raw_cache_mb": base_memory,
        "effective_capacity_mb": compressed_effective,
        "metadata_overhead_mb": overhead,
        "total_required_mb": base_memory + overhead,
        "compression_savings_mb": compressed_effective - base_memory,
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "CacheCoherencyProtocol",
    "PageState",
    "MESIState",
    "MigrationPolicy",
    "AllocationStrategy",
    "EvictionPolicy",
    # Configuration
    "DistributedCacheConfig",
    "PageMetadata",
    "CacheStatistics",
    # Coherency protocols
    "CoherencyProtocolBase",
    "MESIProtocol",
    "DirectoryProtocol",
    "create_coherency_protocol",
    # Core components
    "GPUPageAllocator",
    "PageTransferManager",
    "DistributedKVCache",
    "SequenceAwareCacheSharding",
    # Factory functions
    "create_distributed_cache",
    "estimate_distributed_cache_memory",
]
