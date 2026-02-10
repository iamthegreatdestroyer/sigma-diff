"""
Adaptive Cache Sizing
====================

Dynamic cache size and threshold optimization based on workload patterns.

Techniques:
- Workload-aware buffer allocation
- Hit rate-based threshold tuning
- Dynamic threshold adjustment
- Feedback-driven learning
- Predictive sizing

Sprint 2.2 Days 5-6 - Cache Optimization
Created: 2025-12-27
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class WorkloadPattern(Enum):
    """Workload pattern types."""
    REPETITIVE = "repetitive"  # Same sequences repeated
    SIMILAR = "similar"        # Similar but not identical
    DIVERSE = "diverse"        # Very different sequences
    MIXED = "mixed"            # Mix of patterns


@dataclass
class WorkloadMetrics:
    """Metrics for workload characterization."""
    hit_rate: float = 0.0
    exact_hit_rate: float = 0.0
    semantic_hit_rate: float = 0.0
    avg_sequence_length: int = 0
    max_sequence_length: int = 0
    num_unique_sequences: int = 0
    num_total_sequences: int = 0
    avg_similarity: float = 0.0
    cache_utilization: float = 0.0
    eviction_rate: float = 0.0


@dataclass
class CacheSizeRecommendation:
    """Recommended cache size configuration."""
    exact_cache_size: int
    semantic_cache_size: int
    exact_hit_threshold: float
    semantic_hit_threshold: float
    compression_enabled: bool
    compression_format: str
    estimated_memory_mb: float


class AdaptiveThresholdController:
    """
    Dynamically adjusts cache thresholds based on hit rate feedback.
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.85,
        min_threshold: float = 0.75,
        max_threshold: float = 0.95
    ):
        """
        Initialize threshold controller.
        
        Args:
            initial_threshold: Initial similarity threshold
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
        self.current_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Tracking
        self.hit_history: deque = deque(maxlen=100)  # Last 100 accesses
        self.adjustment_count = 0
    
    def update(self, hit: bool):
        """
        Record a cache access result.
        
        Args:
            hit: Whether access was a cache hit
        """
        self.hit_history.append(hit)
    
    def compute_hit_rate(self) -> float:
        """Compute current hit rate."""
        if not self.hit_history:
            return 0.0
        return sum(self.hit_history) / len(self.hit_history)
    
    def adjust_threshold(self) -> float:
        """
        Adjust threshold based on recent hit rate.
        
        Returns:
            New threshold value
        """
        if len(self.hit_history) < 20:  # Need sufficient data
            return self.current_threshold
        
        hit_rate = self.compute_hit_rate()
        
        # Target hit rate: 70%
        target_hit_rate = 0.70
        
        if hit_rate < target_hit_rate - 0.05:
            # Too low hit rate: lower threshold to accept more matches
            adjustment = -0.01
        elif hit_rate > target_hit_rate + 0.05:
            # Too high hit rate: raise threshold to be more selective
            adjustment = +0.01
        else:
            # Good hit rate: no adjustment
            return self.current_threshold
        
        # Apply adjustment
        new_threshold = self.current_threshold + adjustment
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
        
        if new_threshold != self.current_threshold:
            logger.info(f"Adjusted threshold: {self.current_threshold:.3f} → {new_threshold:.3f} "
                       f"(hit rate: {hit_rate:.1%})")
            self.adjustment_count += 1
        
        self.current_threshold = new_threshold
        return new_threshold


class WorkloadCharacterizer:
    """
    Characterizes workload patterns to guide caching strategy.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize characterizer.
        
        Args:
            window_size: Sliding window for analysis
        """
        self.window_size = window_size
        self.sequence_history: deque = deque(maxlen=window_size)
        self.unique_sequences: Dict[str, int] = {}
        self.metrics: Optional[WorkloadMetrics] = None
    
    def add_sequence(
        self,
        tokens: torch.Tensor,
        hit: bool = False,
        hit_type: str = "miss"  # "exact", "semantic", "miss"
    ):
        """
        Add sequence to analysis.
        
        Args:
            tokens: Token sequence
            hit: Whether was a cache hit
            hit_type: Type of hit
        """
        token_str = str(tokens.tolist())
        self.sequence_history.append({
            "tokens": tokens,
            "token_str": token_str,
            "hit": hit,
            "hit_type": hit_type
        })
        
        # Track uniqueness
        if token_str not in self.unique_sequences:
            self.unique_sequences[token_str] = 0
        self.unique_sequences[token_str] += 1
    
    def characterize(self) -> WorkloadMetrics:
        """
        Analyze workload and return metrics.
        
        Returns:
            WorkloadMetrics
        """
        if not self.sequence_history:
            return WorkloadMetrics()
        
        # Collect data
        exact_hits = sum(1 for s in self.sequence_history if s["hit_type"] == "exact")
        semantic_hits = sum(1 for s in self.sequence_history if s["hit_type"] == "semantic")
        total_hits = sum(1 for s in self.sequence_history if s["hit"])
        
        lengths = [len(s["tokens"]) for s in self.sequence_history]
        
        metrics = WorkloadMetrics(
            hit_rate=total_hits / max(1, len(self.sequence_history)),
            exact_hit_rate=exact_hits / max(1, len(self.sequence_history)),
            semantic_hit_rate=semantic_hits / max(1, len(self.sequence_history)),
            avg_sequence_length=sum(lengths) // max(1, len(lengths)),
            max_sequence_length=max(lengths) if lengths else 0,
            num_unique_sequences=len(self.unique_sequences),
            num_total_sequences=len(self.sequence_history)
        )
        
        # Calculate similarity (rough estimate)
        if len(self.unique_sequences) > 0:
            avg_frequency = sum(self.unique_sequences.values()) / len(self.unique_sequences)
            metrics.avg_similarity = 1.0 - (len(self.unique_sequences) / len(self.sequence_history))
        
        self.metrics = metrics
        return metrics
    
    def detect_pattern(self) -> WorkloadPattern:
        """
        Detect workload pattern.
        
        Returns:
            WorkloadPattern
        """
        metrics = self.characterize()
        
        # Pattern detection logic
        if metrics.hit_rate > 0.7 and metrics.num_unique_sequences < 10:
            return WorkloadPattern.REPETITIVE
        elif metrics.hit_rate > 0.4 and metrics.avg_similarity > 0.6:
            return WorkloadPattern.SIMILAR
        elif metrics.hit_rate < 0.1:
            return WorkloadPattern.DIVERSE
        else:
            return WorkloadPattern.MIXED


class CacheSizeOptimizer:
    """
    Recommends optimal cache sizes based on workload.
    """
    
    def __init__(self):
        """Initialize optimizer."""
        self.characterizer = WorkloadCharacterizer()
    
    def recommend_size(
        self,
        available_memory_gb: float = 16.0,
        target_hit_rate: float = 0.70
    ) -> CacheSizeRecommendation:
        """
        Recommend cache size configuration.
        
        Args:
            available_memory_gb: Available GPU memory
            target_hit_rate: Target hit rate to achieve
        
        Returns:
            CacheSizeRecommendation
        """
        metrics = self.characterizer.characterize()
        pattern = self.characterizer.detect_pattern()
        
        logger.info(f"Workload pattern: {pattern.value}")
        logger.info(f"Metrics: {metrics}")
        
        # Base allocation (MB)
        available_mb = available_memory_gb * 1024
        cache_memory = available_mb * 0.2  # Reserve 20% for caching
        
        # Allocation based on pattern
        if pattern == WorkloadPattern.REPETITIVE:
            # Small exact cache for repetitive
            exact_ratio = 0.7
            semantic_ratio = 0.3
            compression = False
        elif pattern == WorkloadPattern.SIMILAR:
            # Larger semantic cache for similar
            exact_ratio = 0.3
            semantic_ratio = 0.7
            compression = True
        elif pattern == WorkloadPattern.DIVERSE:
            # Smaller caches for diverse
            exact_ratio = 0.5
            semantic_ratio = 0.5
            compression = True
        else:  # MIXED
            exact_ratio = 0.5
            semantic_ratio = 0.5
            compression = True
        
        # Calculate sizes (in MB)
        exact_mb = cache_memory * exact_ratio
        semantic_mb = cache_memory * semantic_ratio
        
        # Convert to sequence counts (assuming ~3KB per sequence)
        kb_per_sequence = 3
        exact_cache_size = int((exact_mb * 1024) / kb_per_sequence)
        semantic_cache_size = int((semantic_mb * 1024) / kb_per_sequence)
        
        # Adjust thresholds
        if metrics.hit_rate < 0.5:
            exact_threshold = 1.0  # Only perfect matches
            semantic_threshold = 0.80  # More lenient semantic
        elif metrics.hit_rate > 0.8:
            exact_threshold = 1.0
            semantic_threshold = 0.95  # More strict
        else:
            exact_threshold = 1.0
            semantic_threshold = 0.85
        
        # Estimate final memory
        estimated_memory = (exact_cache_size * 3 + semantic_cache_size * 3) / 1024
        if compression:
            estimated_memory *= 0.5  # With compression
        
        return CacheSizeRecommendation(
            exact_cache_size=exact_cache_size,
            semantic_cache_size=semantic_cache_size,
            exact_hit_threshold=exact_threshold,
            semantic_hit_threshold=semantic_threshold,
            compression_enabled=compression,
            compression_format="int8",
            estimated_memory_mb=estimated_memory
        )


class DynamicCacheAllocator:
    """
    Dynamically allocates cache resources based on demand.
    """
    
    def __init__(
        self,
        initial_exact_size: int = 100,
        initial_semantic_size: int = 500,
        max_total_sequences: int = 2000
    ):
        """Initialize allocator."""
        self.exact_capacity = initial_exact_size
        self.semantic_capacity = initial_semantic_size
        self.max_total = max_total_sequences
        
        self.exact_count = 0
        self.semantic_count = 0
        self.expansion_count = 0
    
    def can_allocate_exact(self) -> bool:
        """Check if can allocate to exact cache."""
        return self.exact_count < self.exact_capacity
    
    def can_allocate_semantic(self) -> bool:
        """Check if can allocate to semantic cache."""
        return self.semantic_count < self.semantic_capacity
    
    def allocate_exact(self) -> bool:
        """Allocate to exact cache."""
        if self.can_allocate_exact():
            self.exact_count += 1
            return True
        
        # Try to expand
        if self.exact_count + self.semantic_count < self.max_total:
            self.exact_capacity += 10
            self.exact_count += 1
            self.expansion_count += 1
            logger.debug(f"Expanded exact cache to {self.exact_capacity}")
            return True
        
        return False
    
    def allocate_semantic(self) -> bool:
        """Allocate to semantic cache."""
        if self.can_allocate_semantic():
            self.semantic_count += 1
            return True
        
        # Try to expand
        if self.exact_count + self.semantic_count < self.max_total:
            self.semantic_capacity += 10
            self.semantic_count += 1
            self.expansion_count += 1
            logger.debug(f"Expanded semantic cache to {self.semantic_capacity}")
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get allocator statistics."""
        return {
            "exact_utilization": self.exact_count / max(1, self.exact_capacity),
            "semantic_utilization": self.semantic_count / max(1, self.semantic_capacity),
            "total_utilization": (self.exact_count + self.semantic_count) / self.max_total,
            "expansions": self.expansion_count,
            "exact_capacity": self.exact_capacity,
            "semantic_capacity": self.semantic_capacity
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Adaptive Cache Sizing...")
    
    # Test threshold controller
    print("\n1. Testing Threshold Controller:")
    controller = AdaptiveThresholdController(initial_threshold=0.85)
    
    # Simulate hit pattern
    for i in range(100):
        hit = i % 3 < 2  # 67% hit rate
        controller.update(hit)
        if (i + 1) % 20 == 0:
            print(f"  After {i+1} accesses: hit_rate={controller.compute_hit_rate():.1%}, "
                  f"threshold={controller.adjust_threshold():.3f}")
    
    # Test workload characterizer
    print("\n2. Testing Workload Characterizer:")
    characterizer = WorkloadCharacterizer(window_size=100)
    
    for i in range(50):
        tokens = torch.randint(0, 1000, (100,))
        hit = i % 3 < 1  # 33% hit rate
        characterizer.add_sequence(tokens, hit=hit, hit_type="semantic" if hit else "miss")
    
    metrics = characterizer.characterize()
    pattern = characterizer.detect_pattern()
    
    print(f"  Metrics: hit_rate={metrics.hit_rate:.1%}, pattern={pattern.value}")
    
    # Test cache size optimizer
    print("\n3. Testing Cache Size Optimizer:")
    optimizer = CacheSizeOptimizer()
    
    # Add more sequences
    for i in range(100):
        tokens = torch.randint(0, 1000, (150,))
        characterizer.add_sequence(tokens, hit=i % 4 < 1, hit_type="exact" if i % 4 == 0 else "miss")
    
    recommendation = optimizer.recommend_size(available_memory_gb=8.0)
    
    print(f"  Exact cache: {recommendation.exact_cache_size}")
    print(f"  Semantic cache: {recommendation.semantic_cache_size}")
    print(f"  Compression: {recommendation.compression_enabled}")
    print(f"  Estimated memory: {recommendation.estimated_memory_mb:.1f}MB")
    
    print("\nAdaptive sizing tests passed!")
