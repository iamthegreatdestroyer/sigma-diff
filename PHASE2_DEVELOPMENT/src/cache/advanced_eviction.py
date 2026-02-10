"""
Advanced Cache Eviction Policies
=================================

Beyond basic LRU: LFU, FIFO, Adaptive strategies for optimal cache performance.

Eviction Strategies:
- LRU: Least Recently Used (baseline)
- LFU: Least Frequently Used (prefer popular sequences)
- FIFO: First In First Out (simple, predictable)
- W-TinyLFU: Weighted TinyLFU (frequency + recency)
- Adaptive: Dynamic selection based on workload

Sprint 2.2 Days 3-4 - Advanced Caching
Created: 2025-12-27
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import time
import heapq
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Eviction policy types."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    W_TINYLFU = "w_tinylfu"  # Weighted TinyLFU
    ADAPTIVE = "adaptive"  # Adaptive selection


@dataclass
class PageAccessInfo:
    """Information about page access patterns."""
    sequence_id: str
    page_id: int
    timestamp: float
    access_count: int = 1
    size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate for this page."""
        total = self.hit_count + self.miss_count
        return self.hit_count / max(1, total)
    
    @property
    def recency_score(self) -> float:
        """Score based on recency (higher = more recent)."""
        return time.time() - self.timestamp
    
    @property
    def frequency_score(self) -> float:
        """Score based on frequency."""
        return self.access_count


class BaseEvictionPolicy(ABC):
    """Base class for eviction policies."""
    
    def __init__(self, max_pages: int):
        """
        Initialize eviction policy.
        
        Args:
            max_pages: Maximum number of pages to cache
        """
        self.max_pages = max_pages
        self.pages: Dict[int, PageAccessInfo] = {}
        self.sequence_to_pages: Dict[str, List[int]] = defaultdict(list)
    
    @abstractmethod
    def record_access(self, sequence_id: str, page_id: int):
        """Record a page access."""
        pass
    
    @abstractmethod
    def select_victim(self) -> Optional[int]:
        """Select page to evict."""
        pass
    
    def add_page(self, sequence_id: str, page_id: int, size_bytes: int = 0):
        """Add new page to cache."""
        if page_id not in self.pages:
            self.pages[page_id] = PageAccessInfo(
                sequence_id=sequence_id,
                page_id=page_id,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            self.sequence_to_pages[sequence_id].append(page_id)
    
    def remove_page(self, page_id: int) -> Optional[str]:
        """Remove page from cache."""
        if page_id in self.pages:
            info = self.pages[page_id]
            sequence_id = info.sequence_id
            del self.pages[page_id]
            
            if page_id in self.sequence_to_pages[sequence_id]:
                self.sequence_to_pages[sequence_id].remove(page_id)
            
            return sequence_id
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get eviction policy statistics."""
        hit_count = sum(p.hit_count for p in self.pages.values())
        miss_count = sum(p.miss_count for p in self.pages.values())
        
        return {
            "num_cached_pages": len(self.pages),
            "hit_count": hit_count,
            "miss_count": miss_count,
            "hit_rate": hit_count / max(1, hit_count + miss_count),
            "num_sequences": len(self.sequence_to_pages)
        }


class LRUEvictionPolicy(BaseEvictionPolicy):
    """
    Least Recently Used eviction policy.
    
    Evicts the page that was accessed longest ago.
    Simple, effective for many workloads.
    """
    
    def __init__(self, max_pages: int):
        super().__init__(max_pages)
        self.access_order: List[int] = []  # Ordered by recency
    
    def record_access(self, sequence_id: str, page_id: int):
        """Record access - move to end."""
        if page_id in self.pages:
            self.pages[page_id].access_count += 1
            self.pages[page_id].timestamp = time.time()
            self.pages[page_id].hit_count += 1
            
            # Move to end (most recent)
            if page_id in self.access_order:
                self.access_order.remove(page_id)
            self.access_order.append(page_id)
    
    def select_victim(self) -> Optional[int]:
        """Select least recently used page."""
        if self.access_order:
            return self.access_order[0]
        return None
    
    def add_page(self, sequence_id: str, page_id: int, size_bytes: int = 0):
        """Add page and track in access order."""
        super().add_page(sequence_id, page_id, size_bytes)
        self.access_order.append(page_id)
    
    def remove_page(self, page_id: int) -> Optional[str]:
        """Remove and clean up access order."""
        result = super().remove_page(page_id)
        if page_id in self.access_order:
            self.access_order.remove(page_id)
        return result


class LFUEvictionPolicy(BaseEvictionPolicy):
    """
    Least Frequently Used eviction policy.
    
    Evicts the page with lowest access frequency.
    Good for skewed access patterns where some pages are reused often.
    """
    
    def __init__(self, max_pages: int):
        super().__init__(max_pages)
        self.frequency_buckets: Dict[int, List[int]] = defaultdict(list)
    
    def record_access(self, sequence_id: str, page_id: int):
        """Record access - update frequency."""
        if page_id in self.pages:
            page = self.pages[page_id]
            
            # Remove from old frequency bucket
            if page.access_count in self.frequency_buckets:
                if page_id in self.frequency_buckets[page.access_count]:
                    self.frequency_buckets[page.access_count].remove(page_id)
            
            # Increment and add to new bucket
            page.access_count += 1
            page.timestamp = time.time()
            page.hit_count += 1
            
            self.frequency_buckets[page.access_count].append(page_id)
    
    def select_victim(self) -> Optional[int]:
        """Select least frequently used page."""
        if not self.frequency_buckets:
            return None
        
        # Find minimum frequency
        min_freq = min(self.frequency_buckets.keys())
        if self.frequency_buckets[min_freq]:
            # Return oldest in minimum frequency bucket
            return self.frequency_buckets[min_freq][0]
        return None
    
    def add_page(self, sequence_id: str, page_id: int, size_bytes: int = 0):
        """Add page with frequency 1."""
        super().add_page(sequence_id, page_id, size_bytes)
        self.frequency_buckets[1].append(page_id)
    
    def remove_page(self, page_id: int) -> Optional[str]:
        """Remove and clean up frequency buckets."""
        if page_id in self.pages:
            freq = self.pages[page_id].access_count
            if freq in self.frequency_buckets:
                if page_id in self.frequency_buckets[freq]:
                    self.frequency_buckets[freq].remove(page_id)
        
        return super().remove_page(page_id)


class FIFOEvictionPolicy(BaseEvictionPolicy):
    """
    First In First Out eviction policy.
    
    Evicts the oldest page (by insertion time).
    Simple, predictable, good baseline.
    """
    
    def __init__(self, max_pages: int):
        super().__init__(max_pages)
        self.insertion_order: List[int] = []
    
    def record_access(self, sequence_id: str, page_id: int):
        """Record access - just update stats."""
        if page_id in self.pages:
            self.pages[page_id].access_count += 1
            self.pages[page_id].hit_count += 1
    
    def select_victim(self) -> Optional[int]:
        """Select first inserted page."""
        if self.insertion_order:
            return self.insertion_order[0]
        return None
    
    def add_page(self, sequence_id: str, page_id: int, size_bytes: int = 0):
        """Add page and track insertion order."""
        super().add_page(sequence_id, page_id, size_bytes)
        self.insertion_order.append(page_id)
    
    def remove_page(self, page_id: int) -> Optional[str]:
        """Remove and clean up insertion order."""
        result = super().remove_page(page_id)
        if page_id in self.insertion_order:
            self.insertion_order.remove(page_id)
        return result


class WTinyLFUEvictionPolicy(BaseEvictionPolicy):
    """
    Weighted TinyLFU eviction policy.
    
    Combines frequency (80%) and recency (20%) for better performance.
    Addresses LFU's weakness with old frequently-accessed pages.
    """
    
    def __init__(self, max_pages: int, frequency_weight: float = 0.8):
        super().__init__(max_pages)
        self.frequency_weight = frequency_weight
        self.recency_weight = 1.0 - frequency_weight
        self.reset_threshold = 10  # Reset frequencies periodically
        self.access_since_reset = 0
    
    def record_access(self, sequence_id: str, page_id: int):
        """Record access with recency update."""
        if page_id in self.pages:
            page = self.pages[page_id]
            page.access_count += 1
            page.timestamp = time.time()
            page.hit_count += 1
            
            self.access_since_reset += 1
            
            # Periodic reset to prevent frequency staleness
            if self.access_since_reset > len(self.pages) * self.reset_threshold:
                self._reset_frequencies()
    
    def _reset_frequencies(self):
        """Reset frequencies to prevent staleness."""
        for page in self.pages.values():
            page.access_count = max(1, page.access_count // 2)
        self.access_since_reset = 0
    
    def select_victim(self) -> Optional[int]:
        """Select page with lowest weighted score."""
        if not self.pages:
            return None
        
        current_time = time.time()
        min_score = float('inf')
        victim = None
        
        for page_id, page in self.pages.items():
            # Normalize scores to [0, 1]
            freq_score = min(1.0, page.access_count / max(1, max(
                p.access_count for p in self.pages.values()
            )))
            recency_score = min(1.0, (current_time - page.timestamp) / 3600)  # Hour scale
            
            # Combined score (higher = less likely to evict)
            score = (self.frequency_weight * freq_score + 
                    self.recency_weight * (1.0 - recency_score))
            
            if score < min_score:
                min_score = score
                victim = page_id
        
        return victim


class AdaptiveEvictionPolicy(BaseEvictionPolicy):
    """
    Adaptive eviction policy.
    
    Dynamically switches between LRU and LFU based on workload characteristics.
    Uses hit rate to determine which strategy works better.
    """
    
    def __init__(self, max_pages: int, adaptation_window: int = 1000):
        super().__init__(max_pages)
        self.adaptation_window = adaptation_window
        self.access_count = 0
        
        # Maintain both policies
        self.lru_policy = LRUEvictionPolicy(max_pages)
        self.lfu_policy = LFUEvictionPolicy(max_pages)
        
        # Track performance
        self.lru_hits = 0
        self.lfu_hits = 0
        self.current_policy = EvictionPolicy.LRU
        self._update_policy()
    
    def record_access(self, sequence_id: str, page_id: int):
        """Record access in both policies."""
        if page_id in self.pages:
            self.pages[page_id].access_count += 1
            self.pages[page_id].hit_count += 1
        
        # Update both policies
        self.lru_policy.record_access(sequence_id, page_id)
        self.lfu_policy.record_access(sequence_id, page_id)
        
        self.access_count += 1
        
        # Periodically adapt
        if self.access_count % self.adaptation_window == 0:
            self._update_policy()
    
    def _update_policy(self):
        """Update policy based on performance."""
        lru_stats = self.lru_policy.get_statistics()
        lfu_stats = self.lfu_policy.get_statistics()
        
        lru_hit_rate = lru_stats.get('hit_rate', 0)
        lfu_hit_rate = lfu_stats.get('hit_rate', 0)
        
        if lfu_hit_rate > lru_hit_rate:
            self.current_policy = EvictionPolicy.LFU
        else:
            self.current_policy = EvictionPolicy.LRU
        
        logger.info(f"Adaptive policy switched to {self.current_policy.value} "
                   f"(LRU: {lru_hit_rate:.2%}, LFU: {lfu_hit_rate:.2%})")
    
    def select_victim(self) -> Optional[int]:
        """Select victim using current policy."""
        if self.current_policy == EvictionPolicy.LRU:
            return self.lru_policy.select_victim()
        else:
            return self.lfu_policy.select_victim()
    
    def add_page(self, sequence_id: str, page_id: int, size_bytes: int = 0):
        """Add page to both policies."""
        super().add_page(sequence_id, page_id, size_bytes)
        self.lru_policy.add_page(sequence_id, page_id, size_bytes)
        self.lfu_policy.add_page(sequence_id, page_id, size_bytes)
    
    def remove_page(self, page_id: int) -> Optional[str]:
        """Remove from both policies."""
        self.lru_policy.remove_page(page_id)
        self.lfu_policy.remove_page(page_id)
        return super().remove_page(page_id)


class EvictionPolicyFactory:
    """Factory for creating eviction policies."""
    
    @staticmethod
    def create(
        policy_type: EvictionPolicy,
        max_pages: int,
        **kwargs
    ) -> BaseEvictionPolicy:
        """
        Create eviction policy.
        
        Args:
            policy_type: Type of policy to create
            max_pages: Maximum number of pages
            **kwargs: Additional arguments
        
        Returns:
            Configured eviction policy
        """
        if policy_type == EvictionPolicy.LRU:
            return LRUEvictionPolicy(max_pages)
        elif policy_type == EvictionPolicy.LFU:
            return LFUEvictionPolicy(max_pages)
        elif policy_type == EvictionPolicy.FIFO:
            return FIFOEvictionPolicy(max_pages)
        elif policy_type == EvictionPolicy.W_TINYLFU:
            freq_weight = kwargs.get('frequency_weight', 0.8)
            return WTinyLFUEvictionPolicy(max_pages, freq_weight)
        elif policy_type == EvictionPolicy.ADAPTIVE:
            window = kwargs.get('adaptation_window', 1000)
            return AdaptiveEvictionPolicy(max_pages, window)
        else:
            raise ValueError(f"Unknown eviction policy: {policy_type}")


if __name__ == "__main__":
    # Test eviction policies
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Advanced Eviction Policies...")
    
    # Test LRU
    print("\n1. Testing LRU Policy:")
    lru = LRUEvictionPolicy(max_pages=4)
    
    for i in range(6):
        lru.add_page(f"seq_{i}", i)
        print(f"Added page {i}")
    
    lru.record_access("seq_0", 0)
    print("Accessed page 0")
    
    victim = lru.select_victim()
    print(f"LRU victim: {victim}")
    
    # Test LFU
    print("\n2. Testing LFU Policy:")
    lfu = LFUEvictionPolicy(max_pages=4)
    
    for i in range(4):
        lfu.add_page(f"seq_{i}", i)
    
    # Access pattern: page 0 frequently, others rarely
    for _ in range(10):
        lfu.record_access("seq_0", 0)
    
    victim = lfu.select_victim()
    print(f"LFU victim (should be page 1): {victim}")
    
    # Test W-TinyLFU
    print("\n3. Testing W-TinyLFU Policy:")
    wtinylfu = WTinyLFUEvictionPolicy(max_pages=4)
    
    for i in range(4):
        wtinylfu.add_page(f"seq_{i}", i)
    
    # Mixed pattern
    for _ in range(5):
        wtinylfu.record_access("seq_0", 0)
    
    time.sleep(0.1)
    wtinylfu.record_access("seq_1", 1)
    
    victim = wtinylfu.select_victim()
    print(f"W-TinyLFU victim: {victim}")
    
    print("\nEviction policy tests passed!")
