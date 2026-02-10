"""
Load Balancer for Distributed Inference.

Implements multiple load balancing strategies:
- Round Robin
- Least Connections
- Weighted Round Robin
- Consistent Hashing
- Latency-Based

Sprint 3.4 - Serving Layer
Created: 2026-01-06
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
import random

logger = logging.getLogger(__name__)


class LoadBalancerStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    LATENCY_BASED = "latency_based"
    RANDOM = "random"


@dataclass
class BackendNode:
    """Represents a backend inference node."""
    id: str
    host: str
    port: int
    weight: int = 1
    healthy: bool = True
    active_connections: int = 0
    avg_latency_ms: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    last_health_check: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"
    
    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""
    strategy: LoadBalancerStrategy = LoadBalancerStrategy.ROUND_ROBIN
    health_check_interval: float = 10.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    connection_timeout: float = 5.0
    max_connections_per_node: int = 100


class LoadBalancer(ABC):
    """Abstract base class for load balancers."""
    
    @abstractmethod
    def select_node(self, key: Optional[str] = None) -> Optional[BackendNode]:
        """Select a backend node for a request."""
        pass
    
    @abstractmethod
    def add_node(self, node: BackendNode) -> None:
        """Add a backend node."""
        pass
    
    @abstractmethod
    def remove_node(self, node_id: str) -> None:
        """Remove a backend node."""
        pass
    
    @abstractmethod
    def mark_healthy(self, node_id: str) -> None:
        """Mark a node as healthy."""
        pass
    
    @abstractmethod
    def mark_unhealthy(self, node_id: str) -> None:
        """Mark a node as unhealthy."""
        pass


class RoundRobinBalancer(LoadBalancer):
    """Round-robin load balancer."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.nodes: Dict[str, BackendNode] = {}
        self._current_index = 0
        self._lock = asyncio.Lock()
    
    def add_node(self, node: BackendNode) -> None:
        self.nodes[node.id] = node
        logger.info(f"Added node {node.id} to round-robin balancer")
    
    def remove_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Removed node {node_id} from round-robin balancer")
    
    def mark_healthy(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].healthy = True
    
    def mark_unhealthy(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].healthy = False
    
    def select_node(self, key: Optional[str] = None) -> Optional[BackendNode]:
        healthy_nodes = [n for n in self.nodes.values() if n.healthy]
        if not healthy_nodes:
            return None
        
        node = healthy_nodes[self._current_index % len(healthy_nodes)]
        self._current_index = (self._current_index + 1) % len(healthy_nodes)
        return node
    
    def get_healthy_nodes(self) -> List[BackendNode]:
        """Get list of healthy nodes."""
        return [n for n in self.nodes.values() if n.healthy]


class LeastConnectionsBalancer(LoadBalancer):
    """Least-connections load balancer."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.nodes: Dict[str, BackendNode] = {}
        self._lock = asyncio.Lock()
    
    def add_node(self, node: BackendNode) -> None:
        self.nodes[node.id] = node
        logger.info(f"Added node {node.id} to least-connections balancer")
    
    def remove_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            del self.nodes[node_id]
    
    def mark_healthy(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].healthy = True
    
    def mark_unhealthy(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].healthy = False
    
    def select_node(self, key: Optional[str] = None) -> Optional[BackendNode]:
        healthy_nodes = [n for n in self.nodes.values() if n.healthy]
        if not healthy_nodes:
            return None
        
        # Select node with fewest active connections
        return min(healthy_nodes, key=lambda n: n.active_connections)
    
    def acquire_connection(self, node_id: str) -> bool:
        """Acquire a connection slot on a node."""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        if node.active_connections >= self.config.max_connections_per_node:
            return False
        
        node.active_connections += 1
        return True
    
    def release_connection(self, node_id: str) -> None:
        """Release a connection slot on a node."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.active_connections = max(0, node.active_connections - 1)


class WeightedRoundRobinBalancer(LoadBalancer):
    """Weighted round-robin load balancer."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.nodes: Dict[str, BackendNode] = {}
        self._weighted_list: List[str] = []
        self._current_index = 0
    
    def add_node(self, node: BackendNode) -> None:
        self.nodes[node.id] = node
        self._rebuild_weighted_list()
        logger.info(f"Added node {node.id} with weight {node.weight}")
    
    def remove_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._rebuild_weighted_list()
    
    def mark_healthy(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].healthy = True
            self._rebuild_weighted_list()
    
    def mark_unhealthy(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].healthy = False
            self._rebuild_weighted_list()
    
    def _rebuild_weighted_list(self) -> None:
        """Rebuild the weighted node list."""
        self._weighted_list = []
        for node in self.nodes.values():
            if node.healthy:
                self._weighted_list.extend([node.id] * node.weight)
        self._current_index = 0
    
    def select_node(self, key: Optional[str] = None) -> Optional[BackendNode]:
        if not self._weighted_list:
            return None
        
        node_id = self._weighted_list[self._current_index % len(self._weighted_list)]
        self._current_index = (self._current_index + 1) % len(self._weighted_list)
        return self.nodes.get(node_id)


class ConsistentHashBalancer(LoadBalancer):
    """Consistent hash load balancer for session affinity."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None, replicas: int = 100):
        self.config = config or LoadBalancerConfig()
        self.nodes: Dict[str, BackendNode] = {}
        self.replicas = replicas
        self._ring: Dict[int, str] = {}
        self._sorted_keys: List[int] = []
    
    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: BackendNode) -> None:
        self.nodes[node.id] = node
        for i in range(self.replicas):
            hash_key = self._hash(f"{node.id}:{i}")
            self._ring[hash_key] = node.id
        self._sorted_keys = sorted(self._ring.keys())
        logger.info(f"Added node {node.id} to consistent hash ring")
    
    def remove_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            del self.nodes[node_id]
            # Remove from ring
            self._ring = {k: v for k, v in self._ring.items() if v != node_id}
            self._sorted_keys = sorted(self._ring.keys())
    
    def mark_healthy(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].healthy = True
    
    def mark_unhealthy(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].healthy = False
    
    def select_node(self, key: Optional[str] = None) -> Optional[BackendNode]:
        if not self._sorted_keys:
            return None
        
        if key is None:
            key = str(random.random())
        
        hash_val = self._hash(key)
        
        # Binary search for the first key >= hash_val
        for ring_key in self._sorted_keys:
            if ring_key >= hash_val:
                node_id = self._ring[ring_key]
                node = self.nodes.get(node_id)
                if node and node.healthy:
                    return node
        
        # Wrap around to first key
        node_id = self._ring[self._sorted_keys[0]]
        node = self.nodes.get(node_id)
        return node if node and node.healthy else None


class LatencyBasedBalancer(LoadBalancer):
    """Latency-based load balancer."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.nodes: Dict[str, BackendNode] = {}
        self._latency_window: Dict[str, List[float]] = {}
        self._window_size = 100
    
    def add_node(self, node: BackendNode) -> None:
        self.nodes[node.id] = node
        self._latency_window[node.id] = []
        logger.info(f"Added node {node.id} to latency-based balancer")
    
    def remove_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            del self.nodes[node_id]
            del self._latency_window[node_id]
    
    def mark_healthy(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].healthy = True
    
    def mark_unhealthy(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].healthy = False
    
    def record_latency(self, node_id: str, latency_ms: float) -> None:
        """Record a latency measurement for a node."""
        if node_id in self._latency_window:
            window = self._latency_window[node_id]
            window.append(latency_ms)
            if len(window) > self._window_size:
                window.pop(0)
            
            # Update node's average latency
            self.nodes[node_id].avg_latency_ms = sum(window) / len(window)
    
    def select_node(self, key: Optional[str] = None) -> Optional[BackendNode]:
        healthy_nodes = [n for n in self.nodes.values() if n.healthy]
        if not healthy_nodes:
            return None
        
        # Select node with lowest average latency
        # If no latency data, use default of 0
        return min(healthy_nodes, key=lambda n: n.avg_latency_ms)


class LoadBalancerFactory:
    """Factory for creating load balancers."""
    
    @staticmethod
    def create(
        strategy: LoadBalancerStrategy,
        config: Optional[LoadBalancerConfig] = None
    ) -> LoadBalancer:
        """Create a load balancer with the specified strategy."""
        if strategy == LoadBalancerStrategy.ROUND_ROBIN:
            return RoundRobinBalancer(config)
        elif strategy == LoadBalancerStrategy.LEAST_CONNECTIONS:
            return LeastConnectionsBalancer(config)
        elif strategy == LoadBalancerStrategy.WEIGHTED_ROUND_ROBIN:
            return WeightedRoundRobinBalancer(config)
        elif strategy == LoadBalancerStrategy.CONSISTENT_HASH:
            return ConsistentHashBalancer(config)
        elif strategy == LoadBalancerStrategy.LATENCY_BASED:
            return LatencyBasedBalancer(config)
        elif strategy == LoadBalancerStrategy.RANDOM:
            return RoundRobinBalancer(config)  # Fallback
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class HealthChecker:
    """Health checker for backend nodes."""
    
    def __init__(
        self,
        balancer: LoadBalancer,
        check_func: Callable[[BackendNode], bool],
        interval: float = 10.0,
        unhealthy_threshold: int = 3,
        healthy_threshold: int = 2
    ):
        self.balancer = balancer
        self.check_func = check_func
        self.interval = interval
        self.unhealthy_threshold = unhealthy_threshold
        self.healthy_threshold = healthy_threshold
        self._failure_counts: Dict[str, int] = {}
        self._success_counts: Dict[str, int] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the health checker."""
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info("Health checker started")
    
    async def stop(self) -> None:
        """Stop the health checker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")
    
    async def _check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            await self._check_all_nodes()
            await asyncio.sleep(self.interval)
    
    async def _check_all_nodes(self) -> None:
        """Check health of all nodes."""
        if not hasattr(self.balancer, 'nodes'):
            return
        
        for node_id, node in self.balancer.nodes.items():
            try:
                healthy = self.check_func(node)
                
                if healthy:
                    self._success_counts[node_id] = self._success_counts.get(node_id, 0) + 1
                    self._failure_counts[node_id] = 0
                    
                    if self._success_counts[node_id] >= self.healthy_threshold:
                        if not node.healthy:
                            self.balancer.mark_healthy(node_id)
                            logger.info(f"Node {node_id} marked healthy")
                else:
                    self._failure_counts[node_id] = self._failure_counts.get(node_id, 0) + 1
                    self._success_counts[node_id] = 0
                    
                    if self._failure_counts[node_id] >= self.unhealthy_threshold:
                        if node.healthy:
                            self.balancer.mark_unhealthy(node_id)
                            logger.warning(f"Node {node_id} marked unhealthy")
                
                node.last_health_check = time.time()
                
            except Exception as e:
                logger.error(f"Health check failed for {node_id}: {e}")
                self._failure_counts[node_id] = self._failure_counts.get(node_id, 0) + 1
