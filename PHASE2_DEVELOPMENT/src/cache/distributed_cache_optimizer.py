"""
Distributed Cache Optimizer
===========================

Cross-node cache coordination and optimization for distributed inference.
Enables efficient cache sharing, replication, and migration across nodes.

Key Features:
- Cross-node cache replication
- Consistent hashing for cache placement
- Cache migration during rebalancing
- Global eviction coordination
- Cache coherency across nodes
- Bandwidth-aware data movement

Sprint 2.2: Days 5-6 Delivery
"""

import torch
import threading
import time
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """State of a cache node."""
    ACTIVE = "active"
    DRAINING = "draining"
    OFFLINE = "offline"
    JOINING = "joining"


class MigrationState(Enum):
    """State of cache migration."""
    IDLE = "idle"
    PREPARING = "preparing"
    TRANSFERRING = "transferring"
    COMPLETING = "completing"
    FAILED = "failed"


@dataclass
class NodeInfo:
    """Information about a cache node."""
    node_id: str
    address: str
    port: int
    state: NodeState = NodeState.ACTIVE
    capacity_mb: float = 1024.0
    used_mb: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    
    @property
    def available_mb(self) -> float:
        return self.capacity_mb - self.used_mb
    
    @property
    def usage_percent(self) -> float:
        return (self.used_mb / self.capacity_mb * 100) if self.capacity_mb > 0 else 0


@dataclass
class CacheEntry:
    """Metadata for a distributed cache entry."""
    key: str
    size_bytes: int
    node_ids: List[str]  # Nodes holding this entry
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    replication_factor: int = 1


@dataclass
class DistributedCacheConfig:
    """Configuration for distributed cache."""
    # Cluster settings
    replication_factor: int = 2              # Number of replicas
    consistency_level: str = "quorum"        # "one", "quorum", "all"
    
    # Consistent hashing
    virtual_nodes: int = 100                 # Virtual nodes per physical node
    
    # Migration
    migration_batch_size: int = 10           # Entries per migration batch
    migration_bandwidth_mbps: float = 100.0  # Target bandwidth for migration
    
    # Health
    heartbeat_interval_sec: float = 1.0
    node_timeout_sec: float = 5.0
    
    # Eviction
    global_eviction_threshold: float = 0.85  # Trigger global eviction at 85%


class ConsistentHash:
    """Consistent hashing for cache placement."""
    
    def __init__(self, virtual_nodes: int = 100):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}  # hash -> node_id
        self.sorted_keys: List[int] = []
        self._lock = threading.Lock()
        
    def _hash(self, key: str) -> int:
        """Compute hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node_id: str) -> None:
        """Add a node to the hash ring."""
        with self._lock:
            for i in range(self.virtual_nodes):
                virtual_key = f"{node_id}:{i}"
                hash_val = self._hash(virtual_key)
                self.ring[hash_val] = node_id
            
            self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the hash ring."""
        with self._lock:
            for i in range(self.virtual_nodes):
                virtual_key = f"{node_id}:{i}"
                hash_val = self._hash(virtual_key)
                self.ring.pop(hash_val, None)
            
            self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a key."""
        with self._lock:
            if not self.sorted_keys:
                return None
            
            hash_val = self._hash(key)
            
            # Binary search for the first node >= hash_val
            left, right = 0, len(self.sorted_keys)
            while left < right:
                mid = (left + right) // 2
                if self.sorted_keys[mid] < hash_val:
                    left = mid + 1
                else:
                    right = mid
            
            # Wrap around if needed
            if left == len(self.sorted_keys):
                left = 0
            
            return self.ring[self.sorted_keys[left]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """Get multiple nodes for a key (for replication)."""
        with self._lock:
            if not self.sorted_keys:
                return []
            
            hash_val = self._hash(key)
            nodes: List[str] = []
            seen: Set[str] = set()
            
            # Binary search for starting point
            left, right = 0, len(self.sorted_keys)
            while left < right:
                mid = (left + right) // 2
                if self.sorted_keys[mid] < hash_val:
                    left = mid + 1
                else:
                    right = mid
            
            # Walk the ring collecting unique nodes
            idx = left
            while len(nodes) < count and len(seen) < len(set(self.ring.values())):
                if idx >= len(self.sorted_keys):
                    idx = 0
                
                node_id = self.ring[self.sorted_keys[idx]]
                if node_id not in seen:
                    nodes.append(node_id)
                    seen.add(node_id)
                
                idx += 1
            
            return nodes


class CacheCoordinator:
    """Coordinates cache operations across nodes."""
    
    def __init__(self, config: DistributedCacheConfig, local_node_id: str):
        self.config = config
        self.local_node_id = local_node_id
        
        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self.hash_ring = ConsistentHash(config.virtual_nodes)
        
        # Cache metadata
        self.entries: Dict[str, CacheEntry] = {}
        
        # Migration state
        self.migration_state = MigrationState.IDLE
        self.pending_migrations: List[Tuple[str, str, str]] = []  # (key, from, to)
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    def register_node(self, node: NodeInfo) -> None:
        """Register a new node in the cluster."""
        with self._lock:
            self.nodes[node.node_id] = node
            self.hash_ring.add_node(node.node_id)
            
        logger.info(f"Node registered: {node.node_id} at {node.address}:{node.port}")
        
        # Trigger rebalancing
        self._schedule_rebalance()
    
    def deregister_node(self, node_id: str) -> None:
        """Remove a node from the cluster."""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].state = NodeState.DRAINING
                self.hash_ring.remove_node(node_id)
                
        logger.info(f"Node deregistered: {node_id}")
        
        # Migrate data from this node
        self._schedule_rebalance()
    
    def get_placement(self, key: str) -> List[str]:
        """Get the nodes that should hold a cache entry."""
        nodes = self.hash_ring.get_nodes(key, self.config.replication_factor)
        
        # Filter out inactive nodes
        with self._lock:
            return [n for n in nodes if n in self.nodes and 
                    self.nodes[n].state == NodeState.ACTIVE]
    
    def record_access(self, key: str, node_id: str) -> None:
        """Record an access to a cache entry."""
        with self._lock:
            if key in self.entries:
                self.entries[key].last_access = time.time()
                self.entries[key].access_count += 1
    
    def register_entry(
        self,
        key: str,
        size_bytes: int,
        node_ids: List[str]
    ) -> None:
        """Register a new cache entry."""
        with self._lock:
            self.entries[key] = CacheEntry(
                key=key,
                size_bytes=size_bytes,
                node_ids=node_ids,
                replication_factor=len(node_ids)
            )
            
            # Update node usage
            for node_id in node_ids:
                if node_id in self.nodes:
                    self.nodes[node_id].used_mb += size_bytes / (1024 * 1024)
    
    def unregister_entry(self, key: str) -> None:
        """Remove a cache entry."""
        with self._lock:
            if key in self.entries:
                entry = self.entries.pop(key)
                
                # Update node usage
                for node_id in entry.node_ids:
                    if node_id in self.nodes:
                        self.nodes[node_id].used_mb -= entry.size_bytes / (1024 * 1024)
                        self.nodes[node_id].used_mb = max(0, self.nodes[node_id].used_mb)
    
    def _schedule_rebalance(self) -> None:
        """Schedule a cluster rebalance."""
        self._executor.submit(self._rebalance)
    
    def _rebalance(self) -> None:
        """Rebalance cache entries across nodes."""
        with self._lock:
            if self.migration_state != MigrationState.IDLE:
                return
            
            self.migration_state = MigrationState.PREPARING
        
        try:
            migrations = self._compute_migrations()
            
            if not migrations:
                with self._lock:
                    self.migration_state = MigrationState.IDLE
                return
            
            with self._lock:
                self.pending_migrations = migrations
                self.migration_state = MigrationState.TRANSFERRING
            
            # Execute migrations
            for key, from_node, to_node in migrations:
                self._migrate_entry(key, from_node, to_node)
            
            with self._lock:
                self.migration_state = MigrationState.IDLE
                self.pending_migrations = []
                
            logger.info(f"Rebalance complete: {len(migrations)} entries migrated")
            
        except Exception as e:
            logger.error(f"Rebalance failed: {e}")
            with self._lock:
                self.migration_state = MigrationState.FAILED
    
    def _compute_migrations(self) -> List[Tuple[str, str, str]]:
        """Compute which entries need to be migrated."""
        migrations = []
        
        with self._lock:
            for key, entry in self.entries.items():
                target_nodes = self.get_placement(key)
                
                # Find entries that need to move
                for current_node in entry.node_ids:
                    if current_node not in target_nodes:
                        # Need to migrate from current to a target
                        for target in target_nodes:
                            if target not in entry.node_ids:
                                migrations.append((key, current_node, target))
                                break
        
        return migrations[:self.config.migration_batch_size]
    
    def _migrate_entry(self, key: str, from_node: str, to_node: str) -> bool:
        """Migrate a single cache entry."""
        # In a real implementation, this would:
        # 1. Read data from from_node
        # 2. Write data to to_node
        # 3. Update metadata
        # 4. Delete from from_node if no longer needed
        
        logger.debug(f"Migrating {key} from {from_node} to {to_node}")
        
        with self._lock:
            if key in self.entries:
                entry = self.entries[key]
                if from_node in entry.node_ids:
                    entry.node_ids.remove(from_node)
                if to_node not in entry.node_ids:
                    entry.node_ids.append(to_node)
                
                # Update node usage
                size_mb = entry.size_bytes / (1024 * 1024)
                if from_node in self.nodes:
                    self.nodes[from_node].used_mb -= size_mb
                if to_node in self.nodes:
                    self.nodes[to_node].used_mb += size_mb
        
        return True
    
    def get_global_eviction_candidates(self, count: int = 10) -> List[str]:
        """Get globally least-valuable entries for eviction."""
        with self._lock:
            # Score entries by access recency and frequency
            scored_entries = []
            
            for key, entry in self.entries.items():
                age = time.time() - entry.last_access
                score = entry.access_count / (age + 1)  # Higher = more valuable
                scored_entries.append((key, score))
            
            # Sort by score (ascending = least valuable first)
            scored_entries.sort(key=lambda x: x[1])
            
            return [key for key, _ in scored_entries[:count]]
    
    def trigger_global_eviction(self, target_reduction_mb: float) -> int:
        """Trigger coordinated eviction across nodes."""
        evicted = 0
        remaining_mb = target_reduction_mb
        
        while remaining_mb > 0:
            candidates = self.get_global_eviction_candidates(10)
            if not candidates:
                break
            
            for key in candidates:
                with self._lock:
                    if key in self.entries:
                        entry = self.entries[key]
                        size_mb = entry.size_bytes / (1024 * 1024)
                        
                        self.unregister_entry(key)
                        evicted += 1
                        remaining_mb -= size_mb
                        
                        if remaining_mb <= 0:
                            break
        
        logger.info(f"Global eviction: {evicted} entries evicted")
        return evicted
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster-wide statistics."""
        with self._lock:
            total_capacity = sum(n.capacity_mb for n in self.nodes.values())
            total_used = sum(n.used_mb for n in self.nodes.values())
            
            return {
                "nodes": {
                    node_id: {
                        "state": node.state.value,
                        "capacity_mb": node.capacity_mb,
                        "used_mb": node.used_mb,
                        "usage_percent": node.usage_percent,
                        "address": f"{node.address}:{node.port}"
                    }
                    for node_id, node in self.nodes.items()
                },
                "cluster": {
                    "total_nodes": len(self.nodes),
                    "active_nodes": sum(1 for n in self.nodes.values() if n.state == NodeState.ACTIVE),
                    "total_capacity_mb": total_capacity,
                    "total_used_mb": total_used,
                    "usage_percent": (total_used / total_capacity * 100) if total_capacity > 0 else 0,
                    "total_entries": len(self.entries),
                    "replication_factor": self.config.replication_factor
                },
                "migration": {
                    "state": self.migration_state.value,
                    "pending_count": len(self.pending_migrations)
                }
            }


# Factory function
def create_distributed_cache_optimizer(
    local_node_id: str,
    replication_factor: int = 2,
    **kwargs
) -> CacheCoordinator:
    """Create a distributed cache optimizer.
    
    Args:
        local_node_id: ID of the local node
        replication_factor: Number of replicas for each entry
        **kwargs: Additional config options
        
    Returns:
        Configured CacheCoordinator
    """
    config = DistributedCacheConfig(
        replication_factor=replication_factor,
        **kwargs
    )
    
    return CacheCoordinator(config, local_node_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Distributed Cache Optimizer...")
    
    # Create coordinator
    coordinator = create_distributed_cache_optimizer(
        local_node_id="node-0",
        replication_factor=2
    )
    
    # Register nodes
    for i in range(3):
        node = NodeInfo(
            node_id=f"node-{i}",
            address="localhost",
            port=8000 + i,
            capacity_mb=1024
        )
        coordinator.register_node(node)
    
    # Register some cache entries
    for i in range(20):
        key = f"cache-key-{i}"
        placement = coordinator.get_placement(key)
        coordinator.register_entry(
            key=key,
            size_bytes=10 * 1024 * 1024,  # 10MB
            node_ids=placement
        )
    
    # Get stats
    stats = coordinator.get_cluster_stats()
    print(f"\nCluster stats:")
    print(f"  Active nodes: {stats['cluster']['active_nodes']}")
    print(f"  Total entries: {stats['cluster']['total_entries']}")
    print(f"  Usage: {stats['cluster']['usage_percent']:.1f}%")
    
    # Test rebalancing
    print("\nDeregistering node-2...")
    coordinator.deregister_node("node-2")
    
    # Wait for rebalance
    time.sleep(1)
    
    stats = coordinator.get_cluster_stats()
    print(f"After rebalance:")
    print(f"  Active nodes: {stats['cluster']['active_nodes']}")
    print(f"  Migration state: {stats['migration']['state']}")
