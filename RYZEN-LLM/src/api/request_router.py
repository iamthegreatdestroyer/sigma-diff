"""
Request Router for Distributed Inference
========================================

Routes incoming requests to appropriate GPUs based on load balancing,
health status, and request characteristics.

Key Features:
- Load-aware GPU selection
- Health-based failover
- Request prioritization
- Performance optimization
- Metrics collection

Routing Strategies:
- Round-robin: Simple distribution
- Least-loaded: Optimal utilization
- GPU-affinity: Request stickiness
- Health-weighted: Prioritize healthy GPUs
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from enum import Enum

from src.distributed.multi_gpu_orchestrator import DistributedOrchestrator, GPUHealthMonitor

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Request routing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    GPU_AFFINITY = "gpu_affinity"
    HEALTH_WEIGHTED = "health_weighted"


class RequestRouter:
    """
    Routes requests to optimal GPUs based on current system state.

    Considers GPU health, load, and request characteristics to make
    intelligent routing decisions.
    """

    def __init__(
        self,
        orchestrator: DistributedOrchestrator,
        health_monitor: GPUHealthMonitor,
        strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED,
        enable_affinity: bool = False
    ):
        self.orchestrator = orchestrator
        self.health_monitor = health_monitor
        self.strategy = strategy
        self.enable_affinity = enable_affinity

        # Routing state
        self.round_robin_index = 0
        self.request_affinity: Dict[str, int] = {}  # client_id -> gpu_id

        # Metrics
        self.routing_stats = {
            "total_requests": 0,
            "strategy_distribution": {},
            "failover_events": 0,
            "avg_routing_time": 0.0
        }

    async def route_request(self, request: Any) -> int:
        """
        Route a request to the optimal GPU.

        Args:
            request: Request object with routing hints

        Returns:
            GPU ID to route the request to
        """
        start_time = time.time()

        try:
            # Get healthy GPUs
            healthy_gpus = self.health_monitor.get_healthy_devices()
            if not healthy_gpus:
                raise RuntimeError("No healthy GPUs available")

            # Apply routing strategy
            gpu_id = await self._apply_routing_strategy(request, healthy_gpus)

            # Update metrics
            routing_time = time.time() - start_time
            self._update_metrics(gpu_id, routing_time)

            logger.debug(f"Routed request to GPU {gpu_id} using {self.strategy.value}")
            return gpu_id

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            self.routing_stats["failover_events"] += 1
            # Fallback to first healthy GPU
            healthy_gpus = self.health_monitor.get_healthy_devices()
            return healthy_gpus[0] if healthy_gpus else 0

    async def _apply_routing_strategy(self, request: Any, healthy_gpus: List[int]) -> int:
        """Apply the configured routing strategy."""
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_routing(healthy_gpus)

        elif self.strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_routing(healthy_gpus)

        elif self.strategy == RoutingStrategy.GPU_AFFINITY:
            return self._affinity_routing(request, healthy_gpus)

        elif self.strategy == RoutingStrategy.HEALTH_WEIGHTED:
            return self._health_weighted_routing(healthy_gpus)

        else:
            # Default to least loaded
            return self._least_loaded_routing(healthy_gpus)

    def _round_robin_routing(self, healthy_gpus: List[int]) -> int:
        """Simple round-robin routing."""
        gpu_id = healthy_gpus[self.round_robin_index % len(healthy_gpus)]
        self.round_robin_index += 1
        return gpu_id

    def _least_loaded_routing(self, healthy_gpus: List[int]) -> int:
        """Route to the least loaded healthy GPU."""
        best_gpu = None
        best_score = float('inf')

        for gpu_id in healthy_gpus:
            # Calculate load score (lower is better)
            stats = self.health_monitor.get_stats(gpu_id)
            if stats:
                # Score based on utilization and active requests
                load_score = stats.utilization + (stats.active_requests * 10)
                if load_score < best_score:
                    best_score = load_score
                    best_gpu = gpu_id

        return best_gpu if best_gpu is not None else healthy_gpus[0]

    def _affinity_routing(self, request: Any, healthy_gpus: List[int]) -> int:
        """Route based on client affinity."""
        if not self.enable_affinity:
            return self._least_loaded_routing(healthy_gpus)

        # Extract client identifier (could be from request headers, etc.)
        client_id = getattr(request, 'client_id', None) or getattr(request, 'user_id', None)
        if client_id is None:
            # No affinity info, use least loaded
            return self._least_loaded_routing(healthy_gpus)

        # Check existing affinity
        if client_id in self.request_affinity:
            gpu_id = self.request_affinity[client_id]
            if gpu_id in healthy_gpus:
                return gpu_id
            else:
                # GPU no longer healthy, remove affinity
                del self.request_affinity[client_id]

        # Assign new affinity to least loaded GPU
        gpu_id = self._least_loaded_routing(healthy_gpus)
        self.request_affinity[client_id] = gpu_id
        return gpu_id

    def _health_weighted_routing(self, healthy_gpus: List[int]) -> int:
        """Route weighted by GPU health scores."""
        gpu_scores = []

        for gpu_id in healthy_gpus:
            stats = self.health_monitor.get_stats(gpu_id)
            if stats:
                # Health score based on temperature and utilization
                # Lower temperature and utilization = higher score
                health_score = max(0, 100 - stats.temperature) + max(0, 100 - stats.utilization)
                gpu_scores.append((gpu_id, health_score))
            else:
                # No stats available, neutral score
                gpu_scores.append((gpu_id, 50))

        # Select GPU with highest health score
        gpu_scores.sort(key=lambda x: x[1], reverse=True)
        return gpu_scores[0][0]

    def _update_metrics(self, gpu_id: int, routing_time: float):
        """Update routing metrics."""
        self.routing_stats["total_requests"] += 1

        # Update strategy distribution
        strategy_key = self.strategy.value
        self.routing_stats["strategy_distribution"][strategy_key] = \
            self.routing_stats["strategy_distribution"].get(strategy_key, 0) + 1

        # Update average routing time
        current_avg = self.routing_stats["avg_routing_time"]
        total_requests = self.routing_stats["total_requests"]
        self.routing_stats["avg_routing_time"] = \
            (current_avg * (total_requests - 1) + routing_time) / total_requests

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            **self.routing_stats,
            "active_affinities": len(self.request_affinity),
            "current_strategy": self.strategy.value,
            "affinity_enabled": self.enable_affinity
        }

    def set_routing_strategy(self, strategy: RoutingStrategy):
        """Change routing strategy."""
        logger.info(f"Changing routing strategy from {self.strategy.value} to {strategy.value}")
        self.strategy = strategy

    def clear_affinities(self):
        """Clear all request affinities."""
        cleared_count = len(self.request_affinity)
        self.request_affinity.clear()
        logger.info(f"Cleared {cleared_count} request affinities")

    async def optimize_routing(self):
        """
        Periodically optimize routing parameters based on system performance.

        This could adapt routing strategy based on observed performance,
        clear stale affinities, or adjust load balancing parameters.
        """
        # Clear stale affinities (older than 1 hour)
        current_time = time.time()
        stale_clients = [
            client_id for client_id, gpu_id in self.request_affinity.items()
            if self.health_monitor.get_stats(gpu_id) and
               current_time - self.health_monitor.get_stats(gpu_id).last_heartbeat > 3600
        ]

        for client_id in stale_clients:
            del self.request_affinity[client_id]

        if stale_clients:
            logger.info(f"Cleaned up {len(stale_clients)} stale affinities")

        # Could add more sophisticated optimization here
        # - Analyze routing performance
        # - Adjust strategy based on load patterns
        # - Optimize affinity assignments
