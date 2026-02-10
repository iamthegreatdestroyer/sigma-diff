"""
GPU Coordinator for Distributed Inference

Manages GPU allocation, health monitoring, and load balancing across
multiple GPUs with fine-grained locking to minimize contention.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

# Import lock-free logger
from ..serving.lockfree_logger import lockfree_logger


@dataclass
class GPUStatus:
    """GPU status information."""
    gpu_id: int
    memory_used: int
    memory_total: int
    utilization: float
    temperature: float
    healthy: bool
    last_updated: float


class GPUCoordinator:
    """
    Coordinates GPU operations with fine-grained locking.

    Features:
    - Per-GPU locks for independent operations
    - Atomic status updates
    - Load balancing across healthy GPUs
    - Health monitoring and failover
    """

    def __init__(self, num_gpus: int = 4):
        """
        Initialize GPU coordinator.

        Args:
            num_gpus: Number of GPUs to manage
        """
        self.num_gpus = num_gpus

        # GPU status tracking
        self.gpu_status: Dict[int, GPUStatus] = {}
        self.gpu_loads: Dict[int, int] = {i: 0 for i in range(num_gpus)}

        # Fine-grained locks: one per GPU for independent operations
        self.gpu_locks = {i: asyncio.Lock() for i in range(num_gpus)}

        # Global coordination lock (minimal use)
        self.coordination_lock = asyncio.Lock()

        # Health monitoring
        self.unhealthy_gpus: set = set()
        self.failover_count: Dict[int, int] = defaultdict(int)

        # Statistics
        self.total_operations = 0
        self.failed_operations = 0

    async def allocate_gpu(self, memory_required: int, preferred_gpu: Optional[int] = None) -> Optional[int]:
        """
        Allocate a GPU for an operation.

        Args:
            memory_required: Memory required in bytes
            preferred_gpu: Preferred GPU ID

        Returns:
            GPU ID or None if no suitable GPU available
        """
        # Try preferred GPU first
        if preferred_gpu is not None and preferred_gpu not in self.unhealthy_gpus:
            if await self._check_gpu_availability(preferred_gpu, memory_required):
                async with self.gpu_locks[preferred_gpu]:
                    self.gpu_loads[preferred_gpu] += 1
                    self.total_operations += 1
                return preferred_gpu

        # Find best available GPU
        candidates = []
        for gpu_id in range(self.num_gpus):
            if gpu_id in self.unhealthy_gpus:
                continue

            if await self._check_gpu_availability(gpu_id, memory_required):
                load = self.gpu_loads[gpu_id]
                candidates.append((load, gpu_id))

        if not candidates:
            self.failed_operations += 1
            await lockfree_logger.warning("No suitable GPU available for allocation")
            return None

        # Select GPU with lowest load
        candidates.sort()  # Sort by load (ascending)
        _, selected_gpu = candidates[0]

        async with self.gpu_locks[selected_gpu]:
            self.gpu_loads[selected_gpu] += 1
            self.total_operations += 1

        await lockfree_logger.debug(f"Allocated GPU {selected_gpu} for {memory_required} bytes")
        return selected_gpu

    async def release_gpu(self, gpu_id: int):
        """Release a GPU allocation."""
        if gpu_id in self.gpu_loads:
            async with self.gpu_locks[gpu_id]:
                if self.gpu_loads[gpu_id] > 0:
                    self.gpu_loads[gpu_id] -= 1
                    await lockfree_logger.debug(f"Released GPU {gpu_id}")

    async def update_gpu_status(self, gpu_id: int, status: GPUStatus):
        """Update GPU status information."""
        async with self.gpu_locks[gpu_id]:
            self.gpu_status[gpu_id] = status

            # Update health status
            was_healthy = gpu_id not in self.unhealthy_gpus
            is_healthy = status.healthy

            if was_healthy and not is_healthy:
                # GPU became unhealthy
                async with self.coordination_lock:
                    self.unhealthy_gpus.add(gpu_id)
                    self.failover_count[gpu_id] += 1
                await lockfree_logger.warning(f"GPU {gpu_id} marked unhealthy")
            elif not was_healthy and is_healthy:
                # GPU recovered
                async with self.coordination_lock:
                    self.unhealthy_gpus.discard(gpu_id)
                await lockfree_logger.info(f"GPU {gpu_id} recovered")

    async def get_gpu_status(self, gpu_id: int) -> Optional[GPUStatus]:
        """Get status of a specific GPU."""
        async with self.gpu_locks[gpu_id]:
            return self.gpu_status.get(gpu_id)

    async def get_all_status(self) -> Dict[int, GPUStatus]:
        """Get status of all GPUs."""
        # Collect status from all GPUs (read-only, no locking needed for consistency)
        return dict(self.gpu_status)

    async def get_load_distribution(self) -> Dict[int, int]:
        """Get current load distribution across GPUs."""
        # Read-only operation, no locking needed
        return dict(self.gpu_loads)

    async def get_healthy_gpus(self) -> List[int]:
        """Get list of healthy GPU IDs."""
        async with self.coordination_lock:
            return [i for i in range(self.num_gpus) if i not in self.unhealthy_gpus]

    async def _check_gpu_availability(self, gpu_id: int, memory_required: int) -> bool:
        """Check if GPU is available for allocation."""
        status = self.gpu_status.get(gpu_id)
        if not status or not status.healthy:
            return False

        # Check memory availability
        available_memory = status.memory_total - status.memory_used
        return available_memory >= memory_required

    async def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        async with self.coordination_lock:
            healthy_count = self.num_gpus - len(self.unhealthy_gpus)
            total_failover = sum(self.failover_count.values())

            return {
                "total_gpus": self.num_gpus,
                "healthy_gpus": healthy_count,
                "unhealthy_gpus": list(self.unhealthy_gpus),
                "gpu_loads": dict(self.gpu_loads),
                "total_operations": self.total_operations,
                "failed_operations": self.failed_operations,
                "failover_counts": dict(self.failover_count),
                "total_failovers": total_failover,
                "success_rate": (self.total_operations / max(self.total_operations + self.failed_operations, 1))
            }


# Global GPU coordinator instance
gpu_coordinator = GPUCoordinator()


async def allocate_gpu(memory_required: int, preferred_gpu: Optional[int] = None) -> Optional[int]:
    """Convenience function for GPU allocation."""
    return await gpu_coordinator.allocate_gpu(memory_required, preferred_gpu)


async def release_gpu(gpu_id: int):
    """Convenience function for GPU release."""
    await gpu_coordinator.release_gpu(gpu_id)


async def get_gpu_stats() -> Dict[str, Any]:
    """Convenience function for GPU coordinator stats."""
    return await gpu_coordinator.get_stats()