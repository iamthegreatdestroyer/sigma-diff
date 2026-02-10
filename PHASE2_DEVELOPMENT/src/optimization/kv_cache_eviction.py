"""
KV Cache Eviction Policies Module
Sprint 4.4 - Task 2: Cache Eviction Policies

Implements intelligent eviction strategies for KV cache memory management:
- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- FIFO (First In, First Out)
- Hybrid (combines multiple strategies)

Target: Cache hit rate >85%, memory efficiency >90%
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, List, Tuple, Set
from collections import defaultdict, OrderedDict
import heapq
import time
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums & Data Classes
# =============================================================================

class EvictionPolicy(Enum):
    """Available eviction policies."""
    LRU = auto()          # Least Recently Used
    LFU = auto()          # Least Frequently Used
    FIFO = auto()         # First In, First Out
    HYBRID = auto()       # Combination


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: object
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    priority: float = 1.0  # Higher = more important to keep


@dataclass
class EvictionStats:
    """Statistics about cache evictions."""
    total_accesses: int = 0
    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    current_size_bytes: int = 0
    max_size_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.hit_rate if total > 0 else 0.0 if self.total_accesses == 0 else self.total_hits / self.total_accesses
    
    @property
    def utilization(self) -> float:
        return self.current_size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0.0


# =============================================================================
# Base Cache Manager
# =============================================================================

class CacheManager(ABC):
    """Base class for cache managers with different eviction policies."""
    
    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = EvictionStats(max_size_bytes=max_size_bytes)
    
    @abstractmethod
    def get(self, key: str) -> Optional[object]:
        """Get value from cache, return None if not found."""
        pass
    
    @abstractmethod
    def put(self, key: str, value: object, size_bytes: int = 0) -> None:
        """Put value in cache, evict if necessary."""
        pass
    
    @abstractmethod
    def evict_one(self) -> Optional[str]:
        """Evict one entry according to policy, return evicted key."""
        pass
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            self.stats.current_size_bytes -= entry.size_bytes
            return True
        return False
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        self.stats.current_size_bytes = 0


# =============================================================================
# LRU (Least Recently Used) Cache
# =============================================================================

class LRUCache(CacheManager):
    """
    LRU Cache: Evicts least recently used items.
    
    Uses OrderedDict to track access order.
    Access time: O(1)
    """
    
    def __init__(self, max_size_bytes: int):
        super().__init__(max_size_bytes)
        self.access_order = OrderedDict()  # key -> None (order tracked by dict)
    
    def get(self, key: str) -> Optional[object]:
        """Get value and mark as recently used."""
        self.stats.total_accesses += 1
        
        if key not in self.cache:
            self.stats.total_misses += 1
            return None
        
        self.stats.total_hits += 1
        entry = self.cache[key]
        entry.last_access_time = time.time()
        entry.access_count += 1
        
        # Move to end (most recent)
        self.access_order.move_to_end(key)
        
        return entry.value
    
    def put(self, key: str, value: object, size_bytes: int = 0) -> None:
        """Put value in cache, evicting LRU if necessary."""
        
        # Update if exists
        if key in self.cache:
            old_entry = self.cache[key]
            self.stats.current_size_bytes -= old_entry.size_bytes
        
        # Evict while full
        while self.stats.current_size_bytes + size_bytes > self.max_size_bytes and len(self.cache) > 0:
            evicted = self.evict_one()
            if evicted is None:
                break
        
        # Add new entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes
        )
        
        self.cache[key] = entry
        self.access_order[key] = None
        self.stats.current_size_bytes += size_bytes
    
    def evict_one(self) -> Optional[str]:
        """Evict least recently used entry."""
        if not self.cache:
            return None
        
        # Get first (oldest) entry
        lru_key = next(iter(self.access_order))
        entry = self.cache[lru_key]
        
        del self.cache[lru_key]
        del self.access_order[lru_key]
        self.stats.current_size_bytes -= entry.size_bytes
        self.stats.total_evictions += 1
        
        logger.debug(f"LRU Evicted: {lru_key}")
        return lru_key


# =============================================================================
# LFU (Least Frequently Used) Cache
# =============================================================================

class LFUCache(CacheManager):
    """
    LFU Cache: Evicts least frequently used items.
    
    Uses frequency counter and min-heap for efficiency.
    Access time: O(log n)
    """
    
    def __init__(self, max_size_bytes: int):
        super().__init__(max_size_bytes)
        self.freq_count: Dict[str, int] = {}
        self.freq_buckets: Dict[int, Set[str]] = defaultdict(set)
        self.min_freq = 0
    
    def get(self, key: str) -> Optional[object]:
        """Get value and increment frequency."""
        self.stats.total_accesses += 1
        
        if key not in self.cache:
            self.stats.total_misses += 1
            return None
        
        self.stats.total_hits += 1
        entry = self.cache[key]
        entry.last_access_time = time.time()
        
        # Update frequency
        old_freq = self.freq_count.get(key, 0)
        new_freq = old_freq + 1
        
        self.freq_count[key] = new_freq
        self.freq_buckets[old_freq].discard(key)
        self.freq_buckets[new_freq].add(key)
        
        # Update min frequency if bucket empty
        if old_freq == self.min_freq and not self.freq_buckets[old_freq]:
            self.min_freq = new_freq
        
        return entry.value
    
    def put(self, key: str, value: object, size_bytes: int = 0) -> None:
        """Put value, evicting LFU if necessary."""
        
        # Update if exists
        if key in self.cache:
            old_entry = self.cache[key]
            self.stats.current_size_bytes -= old_entry.size_bytes
        else:
            self.freq_count[key] = 0
            self.freq_buckets[0].add(key)
            self.min_freq = 0
        
        # Evict while full
        while self.stats.current_size_bytes + size_bytes > self.max_size_bytes and len(self.cache) > 0:
            evicted = self.evict_one()
            if evicted is None:
                break
        
        # Add entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes
        )
        
        self.cache[key] = entry
        self.stats.current_size_bytes += size_bytes
    
    def evict_one(self) -> Optional[str]:
        """Evict least frequently used entry."""
        if not self.cache:
            return None
        
        # Get LFU key from minimum frequency bucket
        if self.min_freq not in self.freq_buckets:
            return None
        
        lfu_key = next(iter(self.freq_buckets[self.min_freq]))
        entry = self.cache[lfu_key]
        
        del self.cache[lfu_key]
        self.freq_buckets[self.min_freq].discard(lfu_key)
        del self.freq_count[lfu_key]
        self.stats.current_size_bytes -= entry.size_bytes
        self.stats.total_evictions += 1
        
        logger.debug(f"LFU Evicted: {lfu_key}")
        return lfu_key


# =============================================================================
# FIFO (First In, First Out) Cache
# =============================================================================

class FIFOCache(CacheManager):
    """
    FIFO Cache: Evicts oldest entries first.
    
    Simple queue-based eviction.
    Access time: O(1)
    """
    
    def __init__(self, max_size_bytes: int):
        super().__init__(max_size_bytes)
        self.insertion_order: List[str] = []
    
    def get(self, key: str) -> Optional[object]:
        """Get value from cache."""
        self.stats.total_accesses += 1
        
        if key not in self.cache:
            self.stats.total_misses += 1
            return None
        
        self.stats.total_hits += 1
        entry = self.cache[key]
        entry.last_access_time = time.time()
        entry.access_count += 1
        
        return entry.value
    
    def put(self, key: str, value: object, size_bytes: int = 0) -> None:
        """Put value, evicting FIFO if necessary."""
        
        # Update if exists
        if key in self.cache:
            old_entry = self.cache[key]
            self.stats.current_size_bytes -= old_entry.size_bytes
        else:
            self.insertion_order.append(key)
        
        # Evict while full
        while self.stats.current_size_bytes + size_bytes > self.max_size_bytes and len(self.cache) > 0:
            evicted = self.evict_one()
            if evicted is None:
                break
        
        # Add entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes
        )
        
        self.cache[key] = entry
        self.stats.current_size_bytes += size_bytes
    
    def evict_one(self) -> Optional[str]:
        """Evict oldest entry."""
        if not self.insertion_order:
            return None
        
        fifo_key = self.insertion_order.pop(0)
        
        if fifo_key not in self.cache:
            return self.evict_one()  # Skip if already removed
        
        entry = self.cache[fifo_key]
        del self.cache[fifo_key]
        self.stats.current_size_bytes -= entry.size_bytes
        self.stats.total_evictions += 1
        
        logger.debug(f"FIFO Evicted: {fifo_key}")
        return fifo_key


# =============================================================================
# Hybrid Adaptive Cache
# =============================================================================

class HybridAdaptiveCache(CacheManager):
    """
    Hybrid Cache: Adapts eviction policy based on workload.
    
    Monitors cache hit rate and switches between LRU/LFU/FIFO.
    """
    
    def __init__(self, max_size_bytes: int):
        super().__init__(max_size_bytes)
        self.lru_cache = LRUCache(max_size_bytes // 3)
        self.lfu_cache = LFUCache(max_size_bytes // 3)
        self.fifo_cache = FIFOCache(max_size_bytes // 3)
        
        self.current_policy = EvictionPolicy.LRU
        self.policy_history: List[Tuple[float, str]] = []  # (timestamp, policy_name)
        self.policy_switch_threshold = 0.75  # Switch if hit rate < this
    
    def get(self, key: str) -> Optional[object]:
        """Try to get from current policy cache."""
        self.stats.total_accesses += 1
        
        # Try to get from appropriate cache
        caches = {
            EvictionPolicy.LRU: self.lru_cache,
            EvictionPolicy.LFU: self.lfu_cache,
            EvictionPolicy.FIFO: self.fifo_cache,
        }
        
        cache = caches[self.current_policy]
        value = cache.get(key)
        
        if value is not None:
            self.stats.total_hits += 1
        else:
            self.stats.total_misses += 1
        
        # Check if should switch policy
        self._check_policy_switch()
        
        return value
    
    def put(self, key: str, value: object, size_bytes: int = 0) -> None:
        """Put into current policy cache."""
        caches = {
            EvictionPolicy.LRU: self.lru_cache,
            EvictionPolicy.LFU: self.lfu_cache,
            EvictionPolicy.FIFO: self.fifo_cache,
        }
        
        cache = caches[self.current_policy]
        cache.put(key, value, size_bytes)
        self.stats.current_size_bytes = cache.stats.current_size_bytes
    
    def evict_one(self) -> Optional[str]:
        """Evict from current policy cache."""
        caches = {
            EvictionPolicy.LRU: self.lru_cache,
            EvictionPolicy.LFU: self.lfu_cache,
            EvictionPolicy.FIFO: self.fifo_cache,
        }
        
        cache = caches[self.current_policy]
        return cache.evict_one()
    
    def _check_policy_switch(self) -> None:
        """Check if should switch to different policy."""
        if self.stats.total_accesses % 100 != 0:  # Check every 100 accesses
            return
        
        hit_rate = self.stats.total_hits / self.stats.total_accesses if self.stats.total_accesses > 0 else 0.0
        
        if hit_rate < self.policy_switch_threshold:
            # Switch to LFU (better for skewed workloads)
            if self.current_policy != EvictionPolicy.LFU:
                logger.info(f"Switching cache policy from {self.current_policy.name} to LFU (hit rate: {hit_rate:.2%})")
                self.current_policy = EvictionPolicy.LFU
                self.policy_history.append((time.time(), self.current_policy.name))


# =============================================================================
# Factory Functions
# =============================================================================

def create_cache(
    policy: str = "lru",
    max_size_bytes: int = 1024 * 1024 * 1024  # 1GB default
) -> CacheManager:
    """
    Create a cache manager with specified policy.
    
    Args:
        policy: "lru", "lfu", "fifo", "hybrid"
        max_size_bytes: Maximum cache size in bytes
    
    Returns:
        CacheManager instance
    """
    policy_map = {
        "lru": LRUCache,
        "lfu": LFUCache,
        "fifo": FIFOCache,
        "hybrid": HybridAdaptiveCache,
    }
    
    cache_class = policy_map.get(policy.lower(), LRUCache)
    return cache_class(max_size_bytes)


def benchmark_eviction_policies() -> Dict[str, EvictionStats]:
    """Benchmark different eviction policies."""
    
    # Create caches
    caches = {
        "LRU": LRUCache(100 * 1024),      # 100KB
        "LFU": LFUCache(100 * 1024),
        "FIFO": FIFOCache(100 * 1024),
    }
    
    # Simulate workload: 1000 accesses with 80/20 rule
    # 20% of keys get 80% of accesses
    hot_keys = [f"key_{i}" for i in range(10)]
    cold_keys = [f"key_{i}" for i in range(10, 100)]
    
    for cache_name, cache in caches.items():
        # Populate cache
        for key in hot_keys + cold_keys:
            cache.put(key, f"value_{key}", size_bytes=1024)
        
        # Simulate accesses (80/20 rule)
        for _ in range(800):
            key = hot_keys[_ % len(hot_keys)]
            cache.get(key)
        
        for _ in range(200):
            key = cold_keys[(_ + 10) % len(cold_keys)]
            cache.get(key)
        
        print(f"\n{cache_name} Cache Statistics:")
        print(f"  Hit Rate: {cache.stats.total_hits / cache.stats.total_accesses * 100:.1f}%")
        print(f"  Evictions: {cache.stats.total_evictions}")
        print(f"  Utilization: {cache.stats.utilization * 100:.1f}%")
    
    return {name: cache.stats for name, cache in caches.items()}
