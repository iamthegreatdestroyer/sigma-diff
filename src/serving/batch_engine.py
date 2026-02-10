"""
Batch Engine for Inference Requests

Manages batching of inference requests with per-GPU locks for optimal
multi-threaded performance.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


class BatchPriority(Enum):
    """Priority levels for batch processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class InferenceRequest:
    """Represents an inference request."""
    request_id: str
    input_ids: List[int]
    priority: BatchPriority
    created_at: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InferenceBatch:
    """Represents a batch of inference requests."""
    batch_id: str
    requests: List[InferenceRequest]
    gpu_id: int
    priority: BatchPriority
    created_at: float


class BatchEngine:
    """
    Batch engine with per-GPU locks for optimal MT performance.

    Features:
    - Per-GPU request queues with fine-grained locking
    - Priority-based batching
    - Automatic batch formation and processing
    """

    def __init__(self, max_batch_size: int = 32, max_batch_tokens: int = 4096):
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens

        # Per-GPU request queues (one per GPU)
        self.gpu_queues: Dict[int, asyncio.Queue] = {}
        self.gpu_locks: Dict[int, asyncio.Lock] = {}

        # Batch processing state
        self.processing_lock = asyncio.Lock()
        self.active_batches: Dict[str, InferenceBatch] = {}

        # Statistics
        self.total_requests = 0
        self.total_batches = 0

    async def initialize_gpu_queues(self, gpu_ids: List[int]):
        """Initialize queues and locks for specified GPUs."""
        for gpu_id in gpu_ids:
            self.gpu_queues[gpu_id] = asyncio.Queue()
            self.gpu_locks[gpu_id] = asyncio.Lock()

    async def submit_request(self, request: InferenceRequest) -> str:
        """
        Submit a request to the appropriate GPU queue.

        Uses round-robin GPU selection for load balancing.
        """
        # Simple round-robin GPU selection
        gpu_id = hash(request.request_id) % len(self.gpu_queues)

        # Add to GPU-specific queue
        async with self.gpu_locks[gpu_id]:
            await self.gpu_queues[gpu_id].put(request)
            self.total_requests += 1

        return f"batch_{gpu_id}_{int(time.time() * 1000)}"

    async def get_next_batch(self, gpu_id: int) -> Optional[InferenceBatch]:
        """
        Get next batch for processing on specified GPU.

        Uses per-GPU locking to avoid contention.
        """
        async with self.gpu_locks[gpu_id]:
            if self.gpu_queues[gpu_id].empty():
                return None

            # Collect requests for batch
            requests = []
            batch_tokens = 0

            while (len(requests) < self.max_batch_size and
                   not self.gpu_queues[gpu_id].empty()):

                try:
                    request = self.gpu_queues[gpu_id].get_nowait()
                    request_tokens = len(request.input_ids)

                    # Check if adding this request would exceed token limit
                    if batch_tokens + request_tokens > self.max_batch_tokens:
                        # Put request back if it would exceed limit
                        await self.gpu_queues[gpu_id].put(request)
                        break

                    requests.append(request)
                    batch_tokens += request_tokens

                except asyncio.QueueEmpty:
                    break

            if not requests:
                return None

            # Create batch
            batch_id = f"batch_{gpu_id}_{int(time.time() * 1000)}"
            priority = max(r.priority for r in requests)

            batch = InferenceBatch(
                batch_id=batch_id,
                requests=requests,
                gpu_id=gpu_id,
                priority=priority,
                created_at=time.time()
            )

            async with self.processing_lock:
                self.active_batches[batch_id] = batch
                self.total_batches += 1

            return batch

    async def complete_batch(self, batch_id: str):
        """Mark batch as completed."""
        async with self.processing_lock:
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics for all GPU queues."""
        stats = {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "active_batches": len(self.active_batches),
            "gpu_queues": {}
        }

        for gpu_id, queue in self.gpu_queues.items():
            async with self.gpu_locks[gpu_id]:
                queue_size = queue.qsize()

            stats["gpu_queues"][gpu_id] = {
                "queue_size": queue_size,
                "lock": "available"  # Simplified for stats
            }

        return stats