"""
Batch Optimizer for High-Throughput Inference
==============================================

Dynamic batch size optimization based on:
- System load and GPU memory availability
- Request characteristics (prompt length, complexity)
- Latency SLA requirements
- Historical performance metrics

Sprint 4.1 - Batch Processing Engine
Created: 2026-01-06
"""

import torch
import logging
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Batch optimization strategies."""
    THROUGHPUT_FIRST = "throughput_first"      # Maximize tokens/sec
    LATENCY_FIRST = "latency_first"            # Minimize P99 latency
    BALANCED = "balanced"                       # Balance throughput/latency
    MEMORY_EFFICIENT = "memory_efficient"      # Minimize memory usage
    ADAPTIVE = "adaptive"                       # Dynamically adjust


@dataclass
class BatchMetrics:
    """Metrics for a processed batch."""
    batch_id: str
    batch_size: int
    total_tokens: int
    latency_ms: float
    throughput_tps: float
    gpu_memory_used_mb: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def tokens_per_request(self) -> float:
        return self.total_tokens / max(1, self.batch_size)
    
    @property
    def latency_per_token_ms(self) -> float:
        return self.latency_ms / max(1, self.total_tokens)


@dataclass
class OptimizerConfig:
    """Configuration for batch optimizer."""
    # Batch size limits
    min_batch_size: int = 1
    max_batch_size: int = 256
    default_batch_size: int = 32
    
    # Token limits
    max_batch_tokens: int = 8192
    max_sequence_length: int = 2048
    
    # Timing
    max_wait_time_ms: float = 50.0           # Max time to wait for batch
    min_fill_ratio: float = 0.5               # Min batch fill before execution
    
    # Performance targets
    target_latency_ms: float = 100.0          # Target P99 latency
    target_throughput_tps: float = 1000.0     # Target tokens/sec
    
    # Memory
    max_memory_usage_percent: float = 0.85    # Max GPU memory usage
    memory_buffer_mb: float = 500.0           # Reserve memory buffer
    
    # Optimization
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    learning_rate: float = 0.1                # Adaptation rate
    history_window: int = 100                  # Metrics history size
    
    # SLA
    sla_latency_p99_ms: float = 200.0
    sla_timeout_ms: float = 30000.0           # 30 second timeout


class BatchSizePredictor:
    """
    ML-based batch size predictor.
    
    Uses historical metrics to predict optimal batch size
    based on current system state and request characteristics.
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.history: deque = deque(maxlen=config.history_window)
        
        # Simple exponential smoothing for key metrics
        self._avg_latency_per_token = 1.0      # ms/token
        self._avg_memory_per_token = 0.1       # MB/token
        self._smoothing = 0.1
        
        # Learned adjustments
        self._batch_size_adjustment = 1.0
    
    def add_observation(self, metrics: BatchMetrics) -> None:
        """Add batch metrics observation."""
        self.history.append(metrics)
        
        # Update rolling averages
        latency_per_token = metrics.latency_per_token_ms
        memory_per_token = metrics.gpu_memory_used_mb / max(1, metrics.total_tokens)
        
        self._avg_latency_per_token = (
            self._smoothing * latency_per_token + 
            (1 - self._smoothing) * self._avg_latency_per_token
        )
        self._avg_memory_per_token = (
            self._smoothing * memory_per_token + 
            (1 - self._smoothing) * self._avg_memory_per_token
        )
    
    def predict_optimal_batch_size(
        self,
        pending_tokens: int,
        available_memory_mb: float,
        strategy: OptimizationStrategy
    ) -> int:
        """
        Predict optimal batch size given current state.
        
        Args:
            pending_tokens: Total tokens waiting to be processed
            available_memory_mb: Available GPU memory in MB
            strategy: Optimization strategy
        
        Returns:
            Optimal batch size
        """
        if strategy == OptimizationStrategy.THROUGHPUT_FIRST:
            # Maximize batch size within memory constraints
            memory_limit_batch = int(available_memory_mb / max(0.1, self._avg_memory_per_token))
            return min(memory_limit_batch, self.config.max_batch_size)
        
        elif strategy == OptimizationStrategy.LATENCY_FIRST:
            # Smaller batches for lower latency
            latency_limit_batch = int(self.config.target_latency_ms / max(0.1, self._avg_latency_per_token))
            return max(self.config.min_batch_size, min(latency_limit_batch, 32))
        
        elif strategy == OptimizationStrategy.MEMORY_EFFICIENT:
            # Conservative batch size
            return min(16, self.config.default_batch_size)
        
        elif strategy == OptimizationStrategy.BALANCED:
            # Balance throughput and latency
            latency_limit = int(self.config.target_latency_ms / max(0.1, self._avg_latency_per_token))
            memory_limit = int(available_memory_mb / max(0.1, self._avg_memory_per_token))
            return max(self.config.min_batch_size, min(latency_limit, memory_limit, self.config.max_batch_size // 2))
        
        else:  # ADAPTIVE
            # Use learned adjustment
            base_size = self.config.default_batch_size
            adjusted = int(base_size * self._batch_size_adjustment)
            return max(self.config.min_batch_size, min(adjusted, self.config.max_batch_size))
    
    def update_adjustment(self, achieved_latency_ms: float, target_latency_ms: float) -> None:
        """Update batch size adjustment based on performance."""
        ratio = target_latency_ms / max(1.0, achieved_latency_ms)
        
        # Adjust more aggressively if far from target
        if ratio > 1.1:  # Latency was lower than target - can increase batch
            self._batch_size_adjustment *= (1 + self.config.learning_rate)
        elif ratio < 0.9:  # Latency was higher than target - decrease batch
            self._batch_size_adjustment *= (1 - self.config.learning_rate)
        
        # Clamp adjustment factor
        self._batch_size_adjustment = max(0.5, min(2.0, self._batch_size_adjustment))


class MemoryEstimator:
    """
    Estimates GPU memory requirements for batches.
    
    Uses profiling data to predict memory usage before execution.
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
        
        # Memory profile (can be calibrated)
        self._base_memory_mb = 100.0               # Base model memory
        self._per_token_memory_mb = 0.05           # Per-token KV cache
        self._per_request_overhead_mb = 10.0       # Per-request overhead
        
        # Calibration history
        self._history: deque = deque(maxlen=50)
    
    def estimate_memory(
        self,
        batch_size: int,
        total_tokens: int,
        max_sequence_length: int
    ) -> float:
        """
        Estimate memory required for a batch.
        
        Args:
            batch_size: Number of requests in batch
            total_tokens: Total tokens to process
            max_sequence_length: Maximum sequence length
        
        Returns:
            Estimated memory in MB
        """
        token_memory = total_tokens * self._per_token_memory_mb
        request_overhead = batch_size * self._per_request_overhead_mb
        sequence_memory = batch_size * max_sequence_length * 0.02  # KV cache
        
        total = self._base_memory_mb + token_memory + request_overhead + sequence_memory
        return total * 1.1  # 10% safety margin
    
    def calibrate(self, actual_memory_mb: float, batch_size: int, total_tokens: int) -> None:
        """Calibrate memory estimates based on actual usage."""
        self._history.append((actual_memory_mb, batch_size, total_tokens))
        
        if len(self._history) >= 10:
            # Fit simple linear model
            memories = [h[0] for h in self._history]
            tokens = [h[2] for h in self._history]
            
            if len(set(tokens)) > 1:  # Avoid division by zero
                mean_mem = np.mean(memories)
                mean_tokens = np.mean(tokens)
                
                # Update per-token memory estimate
                self._per_token_memory_mb = mean_mem / max(1, mean_tokens) * 0.8
    
    def get_available_memory(self) -> float:
        """Get available GPU memory in MB."""
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            available = total - allocated - self.config.memory_buffer_mb
            return max(0, available)
        return 4096.0  # Default for CPU


class BatchOptimizer:
    """
    Main batch optimization engine.
    
    Dynamically adjusts batch sizes based on:
    - Current system load
    - Request characteristics
    - Performance targets
    - Memory constraints
    
    Features:
    - Adaptive batch sizing
    - SLA-aware optimization
    - Memory-aware scheduling
    - Throughput/latency tradeoff management
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self.predictor = BatchSizePredictor(self.config)
        self.memory_estimator = MemoryEstimator(self.config)
        
        # Current state
        self._current_batch_size = self.config.default_batch_size
        self._current_strategy = self.config.strategy
        
        # Metrics
        self._total_batches = 0
        self._total_tokens = 0
        self._latency_history: deque = deque(maxlen=100)
        self._throughput_history: deque = deque(maxlen=100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(
            f"BatchOptimizer initialized: strategy={self.config.strategy.value}, "
            f"batch_size={self.config.default_batch_size}, "
            f"max_tokens={self.config.max_batch_tokens}"
        )
    
    def get_optimal_batch_size(
        self,
        pending_requests: int,
        total_pending_tokens: int,
        avg_sequence_length: int = 100
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Calculate optimal batch size for current conditions.
        
        Args:
            pending_requests: Number of pending requests
            total_pending_tokens: Total tokens in pending requests
            avg_sequence_length: Average sequence length
        
        Returns:
            Tuple of (optimal_batch_size, metadata)
        """
        with self._lock:
            available_memory = self.memory_estimator.get_available_memory()
            
            # Get prediction from ML predictor
            predicted_size = self.predictor.predict_optimal_batch_size(
                pending_tokens=total_pending_tokens,
                available_memory_mb=available_memory,
                strategy=self._current_strategy
            )
            
            # Apply constraints
            # 1. Token limit
            token_constrained = self.config.max_batch_tokens // max(1, avg_sequence_length)
            
            # 2. Memory limit
            memory_per_batch = self.memory_estimator.estimate_memory(
                batch_size=1,
                total_tokens=avg_sequence_length,
                max_sequence_length=self.config.max_sequence_length
            )
            memory_constrained = int(available_memory / max(1, memory_per_batch))
            
            # 3. Pending requests limit
            request_constrained = pending_requests
            
            # Take minimum of all constraints
            optimal = min(
                predicted_size,
                token_constrained,
                memory_constrained,
                request_constrained,
                self.config.max_batch_size
            )
            
            optimal = max(self.config.min_batch_size, optimal)
            
            metadata = {
                "predicted_size": predicted_size,
                "token_constrained": token_constrained,
                "memory_constrained": memory_constrained,
                "request_constrained": request_constrained,
                "available_memory_mb": available_memory,
                "strategy": self._current_strategy.value
            }
            
            self._current_batch_size = optimal
            return optimal, metadata
    
    def should_execute_batch(
        self,
        current_batch_size: int,
        current_tokens: int,
        wait_time_ms: float
    ) -> Tuple[bool, str]:
        """
        Determine if batch should be executed now.
        
        Args:
            current_batch_size: Current number of requests in batch
            current_tokens: Current tokens in batch
            wait_time_ms: Time spent waiting for more requests
        
        Returns:
            Tuple of (should_execute, reason)
        """
        # Immediate execution conditions
        if current_batch_size >= self._current_batch_size:
            return True, "batch_size_reached"
        
        if current_tokens >= self.config.max_batch_tokens:
            return True, "token_limit_reached"
        
        if wait_time_ms >= self.config.max_wait_time_ms:
            return True, "wait_timeout"
        
        # Check fill ratio
        fill_ratio = current_batch_size / max(1, self._current_batch_size)
        if fill_ratio >= self.config.min_fill_ratio and wait_time_ms > self.config.max_wait_time_ms / 2:
            return True, "fill_ratio_met"
        
        return False, "waiting"
    
    def record_batch_result(
        self,
        batch_size: int,
        total_tokens: int,
        latency_ms: float,
        memory_used_mb: float
    ) -> None:
        """
        Record batch execution results for optimization.
        
        Args:
            batch_size: Executed batch size
            total_tokens: Tokens processed
            latency_ms: Execution latency
            memory_used_mb: Memory used
        """
        with self._lock:
            self._total_batches += 1
            self._total_tokens += total_tokens
            
            throughput = total_tokens / max(0.001, latency_ms / 1000)
            
            self._latency_history.append(latency_ms)
            self._throughput_history.append(throughput)
            
            # Update predictor
            metrics = BatchMetrics(
                batch_id=f"batch_{self._total_batches}",
                batch_size=batch_size,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                throughput_tps=throughput,
                gpu_memory_used_mb=memory_used_mb
            )
            self.predictor.add_observation(metrics)
            
            # Update memory estimator
            self.memory_estimator.calibrate(memory_used_mb, batch_size, total_tokens)
            
            # Adapt batch size based on latency performance
            if self._current_strategy == OptimizationStrategy.ADAPTIVE:
                self.predictor.update_adjustment(latency_ms, self.config.target_latency_ms)
            
            logger.debug(
                f"Batch recorded: size={batch_size}, tokens={total_tokens}, "
                f"latency={latency_ms:.2f}ms, throughput={throughput:.1f} tok/s"
            )
    
    def set_strategy(self, strategy: OptimizationStrategy) -> None:
        """Set optimization strategy."""
        with self._lock:
            self._current_strategy = strategy
            logger.info(f"Optimization strategy set to {strategy.value}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get optimizer metrics."""
        with self._lock:
            avg_latency = np.mean(self._latency_history) if self._latency_history else 0.0
            p99_latency = np.percentile(self._latency_history, 99) if self._latency_history else 0.0
            avg_throughput = np.mean(self._throughput_history) if self._throughput_history else 0.0
            
            return {
                "total_batches": self._total_batches,
                "total_tokens": self._total_tokens,
                "current_batch_size": self._current_batch_size,
                "current_strategy": self._current_strategy.value,
                "avg_latency_ms": avg_latency,
                "p99_latency_ms": p99_latency,
                "avg_throughput_tps": avg_throughput,
                "batch_size_adjustment": self.predictor._batch_size_adjustment
            }
    
    def reset_metrics(self) -> None:
        """Reset optimizer metrics."""
        with self._lock:
            self._total_batches = 0
            self._total_tokens = 0
            self._latency_history.clear()
            self._throughput_history.clear()


class AdaptiveBatchOptimizer(BatchOptimizer):
    """
    Advanced adaptive batch optimizer with reinforcement learning.
    
    Uses Thompson Sampling to explore optimal batch sizes while
    balancing exploration vs exploitation.
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        super().__init__(config)
        
        # Thompson Sampling state
        # For each batch size bucket, track alpha (successes) and beta (failures)
        self._batch_buckets = [8, 16, 32, 64, 128, 256]
        self._alphas = {b: 1.0 for b in self._batch_buckets}
        self._betas = {b: 1.0 for b in self._batch_buckets}
        
        # Exploration parameters
        self._exploration_rate = 0.1
        self._exploitation_count = 0
    
    def _sample_batch_size(self) -> int:
        """Sample batch size using Thompson Sampling."""
        samples = {}
        for bucket in self._batch_buckets:
            samples[bucket] = np.random.beta(self._alphas[bucket], self._betas[bucket])
        
        # Select bucket with highest sample
        best_bucket = max(samples, key=samples.get)
        return best_bucket
    
    def _update_bucket(self, batch_size: int, success: bool) -> None:
        """Update Thompson Sampling parameters."""
        # Find closest bucket
        closest = min(self._batch_buckets, key=lambda b: abs(b - batch_size))
        
        if success:
            self._alphas[closest] += 1.0
        else:
            self._betas[closest] += 1.0
    
    def get_optimal_batch_size(
        self,
        pending_requests: int,
        total_pending_tokens: int,
        avg_sequence_length: int = 100
    ) -> Tuple[int, Dict[str, Any]]:
        """Get optimal batch size with exploration."""
        with self._lock:
            # Decide whether to explore or exploit
            if np.random.random() < self._exploration_rate:
                # Explore: use Thompson Sampling
                sampled = self._sample_batch_size()
                metadata = {"source": "thompson_sampling", "sampled_bucket": sampled}
                return min(sampled, pending_requests, self.config.max_batch_size), metadata
            else:
                # Exploit: use base optimizer
                self._exploitation_count += 1
                return super().get_optimal_batch_size(
                    pending_requests, total_pending_tokens, avg_sequence_length
                )
    
    def record_batch_result(
        self,
        batch_size: int,
        total_tokens: int,
        latency_ms: float,
        memory_used_mb: float
    ) -> None:
        """Record result and update Thompson Sampling."""
        super().record_batch_result(batch_size, total_tokens, latency_ms, memory_used_mb)
        
        # Determine success based on SLA
        success = latency_ms <= self.config.sla_latency_p99_ms
        self._update_bucket(batch_size, success)


# Convenience factory function
def create_batch_optimizer(
    strategy: str = "adaptive",
    max_batch_size: int = 256,
    target_latency_ms: float = 100.0,
    **kwargs
) -> BatchOptimizer:
    """
    Factory function to create batch optimizer.
    
    Args:
        strategy: Optimization strategy name
        max_batch_size: Maximum batch size
        target_latency_ms: Target latency in ms
        **kwargs: Additional config options
    
    Returns:
        Configured BatchOptimizer
    """
    strategy_map = {
        "throughput": OptimizationStrategy.THROUGHPUT_FIRST,
        "latency": OptimizationStrategy.LATENCY_FIRST,
        "balanced": OptimizationStrategy.BALANCED,
        "memory": OptimizationStrategy.MEMORY_EFFICIENT,
        "adaptive": OptimizationStrategy.ADAPTIVE
    }
    
    config = OptimizerConfig(
        max_batch_size=max_batch_size,
        target_latency_ms=target_latency_ms,
        strategy=strategy_map.get(strategy, OptimizationStrategy.ADAPTIVE),
        **kwargs
    )
    
    if strategy == "adaptive":
        return AdaptiveBatchOptimizer(config)
    return BatchOptimizer(config)
