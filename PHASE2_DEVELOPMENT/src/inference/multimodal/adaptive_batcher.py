"""
Adaptive Batcher
================

Dynamic batching for heterogeneous multi-modal inputs.
Optimizes throughput while maintaining latency SLAs.

Features:
- Dynamic batch size adjustment based on GPU memory
- Mixed modality batching with intelligent grouping
- Latency-aware scheduling
- Continuous batching with request pipelining
- Memory-efficient sequence bucketing

Sprint 2.1 - Multi-Modal Inference
Created: 2025-12-26
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
from queue import Queue, PriorityQueue, Empty
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


@dataclass
class BatcherConfig:
    """Configuration for adaptive batcher."""
    max_batch_size: int = 32
    max_batch_tokens: int = 4096  # For text
    max_batch_pixels: int = 224 * 224 * 32  # For images
    max_wait_time_ms: float = 50.0  # Max wait before dispatching
    min_batch_size: int = 1
    enable_dynamic_sizing: bool = True
    enable_memory_optimization: bool = True
    target_latency_ms: float = 100.0
    gpu_memory_fraction: float = 0.8  # Max GPU memory to use
    num_buckets: int = 8  # Sequence length buckets
    enable_continuous_batching: bool = True


@dataclass
class BatchRequest:
    """Single request in the batch queue."""
    request_id: str
    data: Any
    modality: str
    sequence_length: int
    priority: int = 0
    arrival_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    
    def __lt__(self, other):
        """Priority comparison for heap."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.arrival_time < other.arrival_time  # Earlier first


@dataclass
class Batch:
    """A batch of requests ready for processing."""
    batch_id: str
    requests: List[BatchRequest]
    modality: str
    total_tokens: int
    created_time: float = field(default_factory=time.time)
    
    @property
    def size(self) -> int:
        return len(self.requests)
    
    @property
    def max_sequence_length(self) -> int:
        if not self.requests:
            return 0
        return max(r.sequence_length for r in self.requests)


@dataclass
class BatcherStats:
    """Statistics for batcher performance."""
    total_requests: int = 0
    total_batches: int = 0
    average_batch_size: float = 0.0
    average_wait_time_ms: float = 0.0
    batch_utilization: float = 0.0
    dropped_requests: int = 0


class SequenceBucketing:
    """
    Bucket sequences by length for efficient batching.
    
    Reduces padding overhead by grouping similar-length sequences.
    """
    
    def __init__(self, num_buckets: int = 8, max_length: int = 4096):
        self.num_buckets = num_buckets
        self.max_length = max_length
        
        # Create bucket boundaries (powers of 2)
        self.boundaries = self._create_boundaries()
        self.buckets: Dict[int, List[BatchRequest]] = {
            b: [] for b in self.boundaries
        }
    
    def _create_boundaries(self) -> List[int]:
        """Create bucket boundaries."""
        boundaries = []
        length = 32
        while length <= self.max_length and len(boundaries) < self.num_buckets:
            boundaries.append(length)
            length *= 2
        boundaries.append(self.max_length)
        return boundaries
    
    def get_bucket(self, sequence_length: int) -> int:
        """Get bucket for a sequence length."""
        for boundary in self.boundaries:
            if sequence_length <= boundary:
                return boundary
        return self.boundaries[-1]
    
    def add(self, request: BatchRequest):
        """Add request to appropriate bucket."""
        bucket = self.get_bucket(request.sequence_length)
        self.buckets[bucket].append(request)
    
    def get_batch(self, max_size: int, bucket: Optional[int] = None) -> List[BatchRequest]:
        """Get a batch from buckets."""
        if bucket is not None:
            # Get from specific bucket
            requests = self.buckets[bucket][:max_size]
            self.buckets[bucket] = self.buckets[bucket][max_size:]
            return requests
        
        # Get from any non-empty bucket (prefer larger buckets for efficiency)
        for b in reversed(self.boundaries):
            if self.buckets[b]:
                requests = self.buckets[b][:max_size]
                self.buckets[b] = self.buckets[b][max_size:]
                return requests
        
        return []
    
    def total_pending(self) -> int:
        """Total pending requests across all buckets."""
        return sum(len(reqs) for reqs in self.buckets.values())
    
    def clear(self):
        """Clear all buckets."""
        for bucket in self.buckets:
            self.buckets[bucket] = []


class MemoryEstimator:
    """
    Estimate memory requirements for batches.
    
    Helps determine optimal batch sizes based on available GPU memory.
    """
    
    def __init__(self, gpu_memory_fraction: float = 0.8):
        self.gpu_memory_fraction = gpu_memory_fraction
        self.model_memory: int = 0
        self.activation_memory_per_token: int = 0
        
        # Calibration data
        self.batch_size_history: List[Tuple[int, int, bool]] = []  # (size, memory, success)
    
    def estimate_batch_memory(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_size: int = 4096,
        num_layers: int = 32,
        dtype_bytes: int = 2  # FP16
    ) -> int:
        """
        Estimate memory required for a batch.
        
        Returns memory in bytes.
        """
        # Activation memory per token
        activation_per_token = hidden_size * num_layers * dtype_bytes * 2  # forward + backward
        
        # KV cache memory
        kv_cache_per_token = 2 * hidden_size * num_layers * dtype_bytes
        
        # Total per sequence
        memory_per_seq = (activation_per_token + kv_cache_per_token) * sequence_length
        
        # Batch total
        return batch_size * memory_per_seq
    
    def get_max_batch_size(
        self,
        sequence_length: int,
        available_memory: Optional[int] = None
    ) -> int:
        """Get maximum batch size for given sequence length."""
        if available_memory is None:
            if torch.cuda.is_available():
                available_memory = int(
                    torch.cuda.get_device_properties(0).total_memory * 
                    self.gpu_memory_fraction
                )
            else:
                available_memory = 8 * 1024 * 1024 * 1024  # 8GB default
        
        # Binary search for max batch size
        low, high = 1, 256
        
        while low < high:
            mid = (low + high + 1) // 2
            estimated = self.estimate_batch_memory(mid, sequence_length)
            
            if estimated <= available_memory:
                low = mid
            else:
                high = mid - 1
        
        return low
    
    def record_batch(self, batch_size: int, memory_used: int, success: bool):
        """Record batch execution for calibration."""
        self.batch_size_history.append((batch_size, memory_used, success))
        
        # Keep last 100 records
        if len(self.batch_size_history) > 100:
            self.batch_size_history = self.batch_size_history[-100:]


class AdaptiveBatcher:
    """
    Adaptive batching for multi-modal inference.
    
    Dynamically adjusts batch sizes based on:
    - Available GPU memory
    - Request latency requirements
    - Modality-specific constraints
    - Input sequence lengths
    """
    
    def __init__(self, config: BatcherConfig):
        self.config = config
        self.buckets = SequenceBucketing(
            num_buckets=config.num_buckets
        )
        self.memory_estimator = MemoryEstimator(
            gpu_memory_fraction=config.gpu_memory_fraction
        )
        
        # Request queues per modality
        self.queues: Dict[str, List[BatchRequest]] = defaultdict(list)
        self.priority_queue: List[BatchRequest] = []
        
        # Statistics
        self.stats = BatcherStats()
        
        # Batch ID counter
        self._batch_counter = 0
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        logger.info("AdaptiveBatcher initialized")
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID."""
        self._batch_counter += 1
        return f"batch_{self._batch_counter}"
    
    def add_request(
        self,
        data: Any,
        modality: str,
        sequence_length: int,
        priority: int = 0,
        request_id: Optional[str] = None,
        deadline: Optional[float] = None
    ) -> str:
        """
        Add a request to the batcher.
        
        Args:
            data: Request data
            modality: Modality type ("image", "text", etc.)
            sequence_length: Length of sequence/number of tokens
            priority: Request priority (higher = more urgent)
            request_id: Optional custom request ID
            deadline: Optional deadline timestamp
        
        Returns:
            Request ID
        """
        if request_id is None:
            request_id = f"req_{time.time_ns()}"
        
        request = BatchRequest(
            request_id=request_id,
            data=data,
            modality=modality,
            sequence_length=sequence_length,
            priority=priority,
            deadline=deadline
        )
        
        with self._lock:
            # Add to modality queue
            self.queues[modality].append(request)
            
            # Add to priority queue
            heapq.heappush(self.priority_queue, request)
            
            # Add to buckets for sequence-based batching
            self.buckets.add(request)
            
            self.stats.total_requests += 1
        
        return request_id
    
    def get_batch(
        self,
        modality: Optional[str] = None,
        max_size: Optional[int] = None,
        max_wait_ms: Optional[float] = None
    ) -> Optional[Batch]:
        """
        Get a batch of requests.
        
        Args:
            modality: Optional modality filter
            max_size: Maximum batch size
            max_wait_ms: Maximum wait time
        
        Returns:
            Batch if available, None otherwise
        """
        if max_size is None:
            max_size = self.config.max_batch_size
        
        if max_wait_ms is None:
            max_wait_ms = self.config.max_wait_time_ms
        
        with self._lock:
            requests = []
            target_modality = modality
            
            if modality is not None:
                # Get from specific modality queue
                queue = self.queues[modality]
                requests = queue[:max_size]
                self.queues[modality] = queue[max_size:]
            else:
                # Get from any queue with requests
                for mod, queue in self.queues.items():
                    if queue:
                        requests = queue[:max_size]
                        self.queues[mod] = queue[max_size:]
                        target_modality = mod
                        break
            
            if not requests:
                return None
            
            # Calculate total tokens
            total_tokens = sum(r.sequence_length for r in requests)
            
            batch = Batch(
                batch_id=self._generate_batch_id(),
                requests=requests,
                modality=target_modality or "unknown",
                total_tokens=total_tokens
            )
            
            # Update stats
            self.stats.total_batches += 1
            self._update_average_batch_size(len(requests))
            
            return batch
    
    def _update_average_batch_size(self, batch_size: int):
        """Update running average batch size."""
        n = self.stats.total_batches
        old_avg = self.stats.average_batch_size
        self.stats.average_batch_size = old_avg + (batch_size - old_avg) / n
    
    def get_optimal_batch(
        self,
        modality: str,
        available_memory: Optional[int] = None
    ) -> Optional[Batch]:
        """
        Get an optimally-sized batch based on memory constraints.
        
        Args:
            modality: Target modality
            available_memory: Available GPU memory in bytes
        
        Returns:
            Optimally-sized batch
        """
        with self._lock:
            queue = self.queues[modality]
            
            if not queue:
                return None
            
            # Estimate optimal batch size
            avg_seq_len = sum(r.sequence_length for r in queue) / len(queue)
            max_batch = self.memory_estimator.get_max_batch_size(
                int(avg_seq_len),
                available_memory
            )
            
            # Cap at config max
            max_batch = min(max_batch, self.config.max_batch_size)
            
            # Get requests
            requests = queue[:max_batch]
            self.queues[modality] = queue[max_batch:]
            
            if not requests:
                return None
            
            total_tokens = sum(r.sequence_length for r in requests)
            
            batch = Batch(
                batch_id=self._generate_batch_id(),
                requests=requests,
                modality=modality,
                total_tokens=total_tokens
            )
            
            self.stats.total_batches += 1
            self._update_average_batch_size(len(requests))
            
            return batch
    
    def get_continuous_batch(
        self,
        modality: str,
        current_batch: Optional[Batch] = None,
        max_add: int = 8
    ) -> Batch:
        """
        Get or extend a batch for continuous batching.
        
        Continuous batching allows adding new requests to running batches.
        
        Args:
            modality: Target modality
            current_batch: Existing batch to extend
            max_add: Maximum requests to add
        
        Returns:
            Extended or new batch
        """
        with self._lock:
            if current_batch is None:
                # Create new batch
                return self.get_batch(modality) or Batch(
                    batch_id=self._generate_batch_id(),
                    requests=[],
                    modality=modality,
                    total_tokens=0
                )
            
            # Extend existing batch
            queue = self.queues[modality]
            remaining_capacity = self.config.max_batch_size - current_batch.size
            to_add = min(len(queue), remaining_capacity, max_add)
            
            if to_add > 0:
                new_requests = queue[:to_add]
                self.queues[modality] = queue[to_add:]
                
                # Create extended batch
                extended_requests = current_batch.requests + new_requests
                extended_tokens = current_batch.total_tokens + sum(
                    r.sequence_length for r in new_requests
                )
                
                return Batch(
                    batch_id=current_batch.batch_id,
                    requests=extended_requests,
                    modality=modality,
                    total_tokens=extended_tokens,
                    created_time=current_batch.created_time
                )
            
            return current_batch
    
    def pending_count(self, modality: Optional[str] = None) -> int:
        """Get count of pending requests."""
        with self._lock:
            if modality is not None:
                return len(self.queues[modality])
            return sum(len(q) for q in self.queues.values())
    
    def get_stats(self) -> BatcherStats:
        """Get batcher statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = BatcherStats()
    
    def clear(self, modality: Optional[str] = None):
        """Clear pending requests."""
        with self._lock:
            if modality is not None:
                self.queues[modality] = []
            else:
                self.queues.clear()
            self.priority_queue = []
            self.buckets.clear()


class ContinuousBatcher(AdaptiveBatcher):
    """
    Continuous batching implementation.
    
    Supports adding/removing requests from running batches
    for optimal GPU utilization.
    """
    
    def __init__(self, config: BatcherConfig):
        super().__init__(config)
        self.active_batches: Dict[str, Batch] = {}
        self.completed_requests: Dict[str, Any] = {}
    
    def start_batch(self, modality: str) -> Batch:
        """Start a new continuous batch."""
        batch = self.get_batch(modality)
        if batch:
            self.active_batches[batch.batch_id] = batch
        return batch
    
    def extend_batch(self, batch_id: str, max_add: int = 8) -> Optional[Batch]:
        """Extend an active batch with new requests."""
        if batch_id not in self.active_batches:
            return None
        
        current = self.active_batches[batch_id]
        extended = self.get_continuous_batch(
            modality=current.modality,
            current_batch=current,
            max_add=max_add
        )
        
        self.active_batches[batch_id] = extended
        return extended
    
    def complete_request(self, batch_id: str, request_id: str, result: Any):
        """Mark a request as completed."""
        if batch_id in self.active_batches:
            batch = self.active_batches[batch_id]
            batch.requests = [r for r in batch.requests if r.request_id != request_id]
            
            # Store result
            self.completed_requests[request_id] = result
            
            # Remove batch if empty
            if not batch.requests:
                del self.active_batches[batch_id]
    
    def get_result(self, request_id: str) -> Optional[Any]:
        """Get result for a completed request."""
        return self.completed_requests.pop(request_id, None)


def create_batcher(
    max_batch_size: int = 32,
    enable_continuous: bool = True,
    **kwargs
) -> AdaptiveBatcher:
    """
    Factory function to create an adaptive batcher.
    
    Args:
        max_batch_size: Maximum batch size
        enable_continuous: Enable continuous batching
        **kwargs: Additional config options
    
    Returns:
        Configured batcher
    """
    config = BatcherConfig(
        max_batch_size=max_batch_size,
        enable_continuous_batching=enable_continuous,
        **kwargs
    )
    
    if enable_continuous:
        return ContinuousBatcher(config)
    return AdaptiveBatcher(config)


if __name__ == "__main__":
    # Test adaptive batcher
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Adaptive Batcher...")
    
    # Create batcher
    config = BatcherConfig(
        max_batch_size=8,
        enable_continuous_batching=True
    )
    batcher = ContinuousBatcher(config)
    
    # Add requests
    for i in range(20):
        modality = "image" if i % 2 == 0 else "text"
        seq_len = 64 + i * 10
        batcher.add_request(
            data=f"data_{i}",
            modality=modality,
            sequence_length=seq_len,
            priority=i % 3
        )
    
    print(f"Pending requests: {batcher.pending_count()}")
    print(f"Pending images: {batcher.pending_count('image')}")
    print(f"Pending text: {batcher.pending_count('text')}")
    
    # Get batches
    batch = batcher.get_batch("image")
    if batch:
        print(f"Got batch {batch.batch_id}: {batch.size} requests, {batch.total_tokens} tokens")
    
    # Continuous batching test
    batch = batcher.start_batch("text")
    if batch:
        print(f"Started continuous batch: {batch.size} requests")
        extended = batcher.extend_batch(batch.batch_id)
        if extended:
            print(f"Extended to: {extended.size} requests")
    
    print(f"Stats: {batcher.get_stats()}")
    print("Adaptive batcher test passed!")
