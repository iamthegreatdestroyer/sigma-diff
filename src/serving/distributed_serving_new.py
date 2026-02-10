"""
Task 1.1.11: Distributed Serving Infrastructure

Comprehensive serving layer for distributed inference with:
- Async request handling with priority queuing
- Dynamic batching with token-level optimization
- Multi-GPU load balancing and request routing
- Health monitoring and automatic failover
- Metrics collection and performance tracking
- Support for streaming responses and long-running inference

Architecture:
  DistributedServingEngine (main orchestrator)
  |- RequestQueue (FIFO + priority)
  |- DynamicBatcher (token-level batching)
  |- LoadBalancer (multi-GPU distribution)
  |- HealthMonitor (GPU + request health)
  |- MetricsCollector (performance tracking)
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class RequestState(Enum):
    """Request lifecycle states."""
    QUEUED = "queued"
    BATCHING = "batching"
    EXECUTING = "executing"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceRequest:
    """Single inference request."""
    request_id: str
    prompt_tokens: torch.Tensor
    max_tokens: int
    priority: RequestPriority = RequestPriority.NORMAL
    temperature: float = 1.0
    top_p: float = 0.95
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    timeout_ms: float = 10000.0  # Default 10s timeout
    
    # Tracking
    state: RequestState = RequestState.QUEUED
    queue_time_ms: float = 0.0
    batch_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    
    def __lt__(self, other: "InferenceRequest") -> bool:
        """Priority queue ordering (higher priority first, then FIFO)."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at
    
    def is_timed_out(self) -> bool:
        """Check if request has timed out."""
        elapsed = (time.time() - self.created_at) * 1000
        return elapsed > self.timeout_ms


@dataclass
class InferenceBatch:
    """Batch of requests for inference."""
    batch_id: str
    request_ids: List[str]
    tokens: torch.Tensor  # [batch_size, seq_len]
    max_tokens: int
    
    gpu_id: int = 0
    created_at: float = field(default_factory=time.time)
    execution_time_ms: float = 0.0


@dataclass
class InferenceResponse:
    """Response for inference request."""
    request_id: str
    generated_tokens: torch.Tensor
    logits: Optional[torch.Tensor] = None
    
    # Metrics
    prompt_tokens: int = 0
    generated_count: int = 0
    queue_time_ms: float = 0.0
    batch_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    tokens_per_second: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ============================================================================
# Request Queue
# ============================================================================

class RequestQueue:
    """Priority queue for managing inference requests with timeout handling and fine-grained locking."""
    
    def __init__(self, max_queue_size: int = 10000):
        """Initialize request queue."""
        self.max_queue_size = max_queue_size
        self.queue: List[InferenceRequest] = []
        self.request_map: Dict[str, InferenceRequest] = {}

        # Fine-grained locks: separate locks for different operations
        self.enqueue_lock = asyncio.Lock()  # For enqueue operations
        self.dequeue_lock = asyncio.Lock()  # For dequeue operations
        self.map_lock = asyncio.Lock()      # For request_map operations

        # Statistics
        self.total_enqueued = 0
        self.total_dequeued = 0
        self.total_timeouts = 0
    
    async def enqueue(self, request: InferenceRequest) -> bool:
        """
        Enqueue a request.
        
        Args:
            request: Request to enqueue
        
        Returns:
            True if enqueued, False if queue full
        """
        async with self.enqueue_lock:
            if len(self.queue) >= self.max_queue_size:
                logger.warning(f"Request queue full: {len(self.queue)}/{self.max_queue_size}")
                return False
            
            heapq.heappush(self.queue, request)
            async with self.map_lock:
                self.request_map[request.request_id] = request
            self.total_enqueued += 1
            
            logger.debug(f"Enqueued request {request.request_id}")
            return True
    
    async def dequeue(self, count: int = 1) -> List[InferenceRequest]:
        """
        Dequeue top N requests.
        
        Args:
            count: Number of requests to dequeue
        
        Returns:
            List of dequeued requests
    async def dequeue(self, count: int = 1) -> List[InferenceRequest]:
        """
        Dequeue top N requests.
        
        Args:
            count: Number of requests to dequeue
        
        Returns:
            List of dequeued requests
        """
        async with self.dequeue_lock:
            dequeued = []
            
            # Remove timed-out requests first
            non_timeout = []
            while self.queue:
                req = heapq.heappop(self.queue)
                
                if req.is_timed_out():
                    async with self.map_lock:
                        del self.request_map[req.request_id]
                    self.total_timeouts += 1
                    logger.warning(f"Request {req.request_id} timed out")
                else:
                    non_timeout.append(req)
            
            # Restore non-timeout requests
            for req in non_timeout:
                heapq.heappush(self.queue, req)
            
            # Dequeue requested number
            while self.queue and len(dequeued) < count:
                req = heapq.heappop(self.queue)
                dequeued.append(req)
                req.state = RequestState.BATCHING
                self.total_dequeued += 1
            
            return dequeued
    
    async def get_size(self) -> int:
        """Get queue size."""
        async with self.dequeue_lock:
            return len(self.queue)
    
    async def cancel(self, request_id: str) -> bool:
        """Cancel a request."""
        async with self.map_lock:
            if request_id in self.request_map:
                req = self.request_map.pop(request_id)
                req.state = RequestState.CANCELLED
                # Remove from queue (need dequeue lock)
                async with self.dequeue_lock:
                    self.queue = [r for r in self.queue if r.request_id != request_id]
                    heapq.heapify(self.queue)
                return True
        return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        async with self.dequeue_lock:
            queue_size = len(self.queue)
        async with self.map_lock:
            map_size = len(self.request_map)
        
        return {
            "queue_size": queue_size,
            "map_size": map_size,
            "total_enqueued": self.total_enqueued,
                "total_dequeued": self.total_dequeued,
                "total_timeouts": self.total_timeouts,
            }


# ============================================================================
# Dynamic Batcher
# ============================================================================

class DynamicBatcher:
    """
    Dynamic batching with token-level optimization.
    
    Groups requests into batches to maximize GPU utilization while
    respecting latency constraints.
    """
    
    def __init__(self, 
                 max_batch_size: int = 128,
                 max_batch_tokens: int = 4096,
                 max_wait_ms: float = 100.0):
        """
        Initialize dynamic batcher.
        
        Args:
            max_batch_size: Maximum requests per batch
            max_batch_tokens: Maximum tokens per batch
            max_wait_ms: Maximum time to wait before batching
        """
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_wait_ms = max_wait_ms
        
        self.pending_requests: List[InferenceRequest] = []
        self.lock = asyncio.Lock()
        
        # Statistics
        self.total_batches = 0
        self.total_tokens_batched = 0
        self.avg_batch_size = 0.0
    
    async def add_requests(self, requests: List[InferenceRequest]):
        """Add requests to pending list."""
        async with self.lock:
            self.pending_requests.extend(requests)
    
    async def form_batches(self) -> List[InferenceBatch]:
        """
        Form batches from pending requests.
        
        Returns:
            List of formed batches
        """
        async with self.lock:
            if not self.pending_requests:
                return []
            
            batches = []
            remaining = self.pending_requests.copy()
            self.pending_requests = []
            
            while remaining:
                # Form next batch
                batch_requests = []
                batch_tokens = 0
                batch_size = 0
                
                non_included = []
                
                for req in remaining:
                    seq_len = req.prompt_tokens.shape[0]
                    
                    # Check if we can add this request
                    if (batch_size < self.max_batch_size and
                        batch_tokens + seq_len <= self.max_batch_tokens):
                        batch_requests.append(req)
                        batch_tokens += seq_len
                        batch_size += 1
                        req.state = RequestState.EXECUTING
                    else:
                        non_included.append(req)
                
                remaining = non_included
                
                if batch_requests:
                    # Create batch
                    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
                    request_ids = [r.request_id for r in batch_requests]
                    
                    # Pad tokens to same length
                    max_seq_len = max(r.prompt_tokens.shape[0] for r in batch_requests)
                    batch_tensor = self._create_padded_batch(batch_requests, max_seq_len)
                    
                    batch = InferenceBatch(
                        batch_id=batch_id,
                        request_ids=request_ids,
                        tokens=batch_tensor,
                        max_tokens=max(r.max_tokens for r in batch_requests)
                    )
                    
                    batches.append(batch)
                    
                    self.total_batches += 1
                    self.total_tokens_batched += batch_tokens
                    self.avg_batch_size = self.total_tokens_batched / self.total_batches
                    
                    logger.debug(f"Formed batch {batch_id}: {batch_size} requests, {batch_tokens} tokens")
            
            # Re-add remaining requests
            self.pending_requests = remaining
            
            return batches
    
    def _create_padded_batch(self, requests: List[InferenceRequest],
                              max_seq_len: int) -> torch.Tensor:
        """Create padded batch tensor."""
        batch_size = len(requests)
        batch = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        for i, req in enumerate(requests):
            seq_len = req.prompt_tokens.shape[0]
            batch[i, :seq_len] = req.prompt_tokens
        
        return batch
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        async with self.lock:
            return {
                "total_batches": self.total_batches,
                "total_tokens_batched": self.total_tokens_batched,
                "avg_batch_size": self.avg_batch_size,
                "pending_requests": len(self.pending_requests),
            }


# ============================================================================
# Load Balancer
# ============================================================================

class LoadBalancer:
    """Multi-GPU load balancer with health-aware routing and fine-grained locking."""
    
    def __init__(self, num_gpus: int = 1):
        """Initialize load balancer."""
        self.num_gpus = num_gpus
        self.gpu_loads: Dict[int, float] = {i: 0.0 for i in range(num_gpus)}
        self.gpu_health: Dict[int, bool] = {i: True for i in range(num_gpus)}

        # Fine-grained locks: one per GPU for state updates
        self.gpu_locks = {i: asyncio.Lock() for i in range(num_gpus)}
        # Global lock for selection operations (reads all GPU states)
        self.selection_lock = asyncio.Lock()

        # Statistics
        self.total_routed = 0
        self.batch_distribution = defaultdict(int)
    
    async def select_gpu(self) -> int:
        """
        Select best GPU for next batch.

        Returns:
            GPU ID with lowest load
        """
        async with self.selection_lock:
            # Filter healthy GPUs
            healthy_gpus = [i for i in range(self.num_gpus) if self.gpu_health[i]]

            if not healthy_gpus:
                logger.error("No healthy GPUs available")
                raise RuntimeError("No healthy GPUs")

            # Select GPU with lowest load
            gpu_id = min(healthy_gpus, key=lambda i: self.gpu_loads[i])

            self.total_routed += 1
            self.batch_distribution[gpu_id] += 1

            return gpu_id
    
    async def update_load(self, gpu_id: int, load: float):
        """Update GPU load."""
        async with self.gpu_locks[gpu_id]:
            self.gpu_loads[gpu_id] = load
    
    async def set_health(self, gpu_id: int, healthy: bool):
        """Update GPU health status."""
        async with self.lock:
            old_status = self.gpu_health[gpu_id]
            self.gpu_health[gpu_id] = healthy
            
            if old_status and not healthy:
                logger.warning(f"GPU {gpu_id} marked unhealthy")
            elif not old_status and healthy:
                logger.info(f"GPU {gpu_id} recovered")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        async with self.lock:
            return {
                "total_routed": self.total_routed,
                "batch_distribution": dict(self.batch_distribution),
                "gpu_loads": dict(self.gpu_loads),
                "gpu_health": dict(self.gpu_health),
            }


# ============================================================================
# Health Monitor
# ============================================================================

class HealthMonitor:
    """Monitor health of GPUs and serving system."""
    
    def __init__(self, num_gpus: int = 1):
        """Initialize health monitor."""
        self.num_gpus = num_gpus
        self.gpu_temps: Dict[int, float] = {}
        self.gpu_memory: Dict[int, float] = {}
        self.error_counts: Dict[int, int] = {i: 0 for i in range(num_gpus)}
        
        self.lock = asyncio.Lock()
    
    async def check_gpu_health(self, gpu_id: int) -> bool:
        """
        Check if GPU is healthy.
        
        Returns:
            True if GPU is healthy
        """
        async with self.lock:
            # Check error count (too many errors = unhealthy)
            if self.error_counts[gpu_id] > 10:
                return False
            
            # In real implementation, check temperature, memory, etc.
            return True
    
    async def record_error(self, gpu_id: int):
        """Record an error for GPU."""
        async with self.lock:
            self.error_counts[gpu_id] += 1
    
    async def reset_errors(self, gpu_id: int):
        """Reset error count for GPU."""
        async with self.lock:
            self.error_counts[gpu_id] = 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get health statistics."""
        async with self.lock:
            return {
                "error_counts": dict(self.error_counts),
                "gpu_temps": dict(self.gpu_temps),
                "gpu_memory": dict(self.gpu_memory),
            }


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """Collect and track serving metrics with fine-grained locking."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.request_latencies: List[float] = []
        self.batch_sizes: List[int] = []
        self.throughputs: List[float] = []
        
        # Per-GPU metrics
        self.gpu_utilizations: Dict[int, List[float]] = defaultdict(list)
        self.gpu_memory_usage: Dict[int, List[float]] = defaultdict(list)
        
        # Fine-grained locks
        self.record_lock = asyncio.Lock()  # For recording operations
        self.stats_lock = asyncio.Lock()   # For statistics computation
        self.start_time = time.time()
    
    async def record_request(self, response: InferenceResponse):
        """Record request metrics."""
        async with self.record_lock:
            self.request_latencies.append(response.total_time_ms)
            
            if response.total_time_ms > 0:
                throughput = response.generated_count / (response.total_time_ms / 1000)
                self.throughputs.append(throughput)
    
    async def record_batch(self, batch: InferenceBatch, execution_time_ms: float):
        """Record batch metrics."""
        async with self.record_lock:
            self.batch_sizes.append(len(batch.request_ids))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get metrics statistics."""
        async with self.stats_lock:
            if not self.request_latencies:
                return {"status": "no_data"}
            
            uptime_s = time.time() - self.start_time
            total_requests = len(self.request_latencies)
            
            return {
                "uptime_seconds": uptime_s,
                "total_requests": total_requests,
                "avg_latency_ms": sum(self.request_latencies) / len(self.request_latencies),
                "p50_latency_ms": sorted(self.request_latencies)[len(self.request_latencies)//2],
                "p99_latency_ms": sorted(self.request_latencies)[int(len(self.request_latencies)*0.99)],
                "max_latency_ms": max(self.request_latencies),
                "avg_throughput_tps": sum(self.throughputs) / len(self.throughputs) if self.throughputs else 0,
                "avg_batch_size": sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0,
                "requests_per_second": total_requests / uptime_s if uptime_s > 0 else 0,
            }


# ============================================================================
# Distributed Serving Engine
# ============================================================================

class DistributedServingEngine:
    """
    Main distributed serving engine combining all components.
    
    Orchestrates request handling, batching, GPU distribution, and monitoring.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 num_gpus: int = 1,
                 max_batch_size: int = 128,
                 max_batch_tokens: int = 4096):
        """
        Initialize serving engine.
        
        Args:
            model: Neural network model
            num_gpus: Number of available GPUs
            max_batch_size: Maximum batch size
            max_batch_tokens: Maximum tokens per batch
        """
        self.model = model
        self.num_gpus = num_gpus
        
        # Components
        self.request_queue = RequestQueue()
        self.batcher = DynamicBatcher(max_batch_size, max_batch_tokens)
        self.load_balancer = LoadBalancer(num_gpus)
        self.health_monitor = HealthMonitor(num_gpus)
        self.metrics = MetricsCollector()
        
        # State
        self.running = False
        self.processed_requests: Dict[str, InferenceResponse] = {}
        self.lock = asyncio.Lock()
        
        logger.info(f"DistributedServingEngine initialized: {num_gpus} GPUs")
    
    async def submit_request(self, request: InferenceRequest) -> str:
        """
        Submit an inference request.
        
        Args:
            request: Request to process
        
        Returns:
            Request ID
        """
        if not await self.request_queue.enqueue(request):
            raise RuntimeError("Request queue full")
        
        return request.request_id
    
    async def get_response(self, request_id: str, timeout_s: float = 30.0) -> InferenceResponse:
        """
        Get response for a request.
        
        Args:
            request_id: Request ID
            timeout_s: Timeout in seconds
        
        Returns:
            Response
        
        Raises:
            TimeoutError: If response not ready within timeout
        """
        start_time = time.time()
        
        while True:
            async with self.lock:
                if request_id in self.processed_requests:
                    return self.processed_requests.pop(request_id)
            
            elapsed = time.time() - start_time
            if elapsed > timeout_s:
                raise TimeoutError(f"Request {request_id} timed out")
            
            await asyncio.sleep(0.01)  # Poll interval
    
    async def serving_loop(self):
        """
        Main serving loop.
        
        Continuously:
        1. Dequeue requests
        2. Form batches
        3. Execute on GPUs
        4. Return responses
        """
        self.running = True
        logger.info("Serving loop started")
        
        while self.running:
            try:
                # Dequeue requests
                requests = await self.request_queue.dequeue(count=self.max_batch_size)
                
                if not requests:
                    await asyncio.sleep(0.01)
                    continue
                
                # Add to batcher
                await self.batcher.add_requests(requests)
                
                # Form batches
                batches = await self.batcher.form_batches()
                
                # Process batches
                for batch in batches:
                    await self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Error in serving loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: InferenceBatch):
        """Process a single batch."""
        try:
            # Select GPU
            gpu_id = await self.load_balancer.select_gpu()
            
            # Move batch to GPU
            batch.tokens = batch.tokens.cuda(gpu_id)
            
            # Execute inference
            exec_start = time.time()
            with torch.no_grad():
                logits = self.model(batch.tokens)
            batch.execution_time_ms = (time.time() - exec_start) * 1000
            
            # Generate responses
            for i, request_id in enumerate(batch.request_ids):
                # Simplified: take argmax of logits
                generated = torch.argmax(logits[i], dim=-1)
                
                response = InferenceResponse(
                    request_id=request_id,
                    generated_tokens=generated,
                    generated_count=generated.shape[0],
                    execution_time_ms=batch.execution_time_ms
                )
                
                async with self.lock:
                    self.processed_requests[request_id] = response
                
                # Record metrics
                await self.metrics.record_request(response)
            
            # Reset GPU errors on success
            await self.health_monitor.reset_errors(gpu_id)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {e}")
            
            # Mark batch as failed and record errors
            gpu_id = batch.gpu_id
            await self.health_monitor.record_error(gpu_id)
            
            # Check if GPU is still healthy
            is_healthy = await self.health_monitor.check_gpu_health(gpu_id)
            await self.load_balancer.set_health(gpu_id, is_healthy)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive serving statistics."""
        return {
            "request_queue": await self.request_queue.get_stats(),
            "batcher": await self.batcher.get_stats(),
            "load_balancer": await self.load_balancer.get_stats(),
            "health_monitor": await self.health_monitor.get_stats(),
            "metrics": await self.metrics.get_stats(),
        }
    
    async def shutdown(self):
        """Shutdown the serving engine."""
        self.running = False
        logger.info("Serving engine shutdown complete")


if __name__ == "__main__":
    logger.info("Distributed Serving Engine module ready")
