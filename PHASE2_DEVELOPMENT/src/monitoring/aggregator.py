"""
Metrics Aggregation for Distributed Systems.

This module provides aggregation strategies for collecting and combining
metrics from multiple nodes in a distributed LLM inference system.

Classes:
    AggregationStrategy: Enum of aggregation methods
    MetricsAggregator: Aggregates metrics from multiple sources
    DistributedCollector: Collects metrics from remote nodes
"""

from __future__ import annotations

import time
import threading
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import logging

from .metrics import MetricRegistry, MetricType

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategies for aggregating metrics across nodes."""
    SUM = "sum"           # Sum values across all nodes
    AVERAGE = "average"   # Average values across nodes
    MAX = "max"           # Maximum value across nodes
    MIN = "min"           # Minimum value across nodes
    LATEST = "latest"     # Most recent value
    COUNT = "count"       # Count of non-zero values
    PERCENTILE_50 = "p50" # 50th percentile
    PERCENTILE_90 = "p90" # 90th percentile
    PERCENTILE_99 = "p99" # 99th percentile


@dataclass
class NodeMetrics:
    """Metrics snapshot from a single node."""
    node_id: str
    timestamp: float
    metrics: Dict[str, Any]
    is_healthy: bool = True
    latency_ms: float = 0.0
    
    def age_seconds(self) -> float:
        """Get age of this snapshot in seconds."""
        return time.time() - self.timestamp


@dataclass
class AggregationRule:
    """Rule for how to aggregate a specific metric."""
    metric_name: str
    strategy: AggregationStrategy
    labels_to_preserve: List[str] = field(default_factory=list)
    labels_to_aggregate: List[str] = field(default_factory=list)
    default_value: float = 0.0


class MetricsAggregator:
    """
    Aggregates metrics from multiple nodes/sources.
    
    Supports various aggregation strategies and can handle
    metrics with labels, combining them appropriately.
    
    Example:
        aggregator = MetricsAggregator()
        
        # Add rules for specific metrics
        aggregator.add_rule(AggregationRule(
            metric_name="tokens_generated_total",
            strategy=AggregationStrategy.SUM
        ))
        
        # Register node metrics
        aggregator.update_node("node-1", metrics_1)
        aggregator.update_node("node-2", metrics_2)
        
        # Get aggregated view
        aggregated = aggregator.aggregate()
    """
    
    def __init__(
        self,
        default_strategy: AggregationStrategy = AggregationStrategy.SUM,
        max_node_age_seconds: float = 60.0
    ):
        """
        Initialize the aggregator.
        
        Args:
            default_strategy: Default aggregation strategy for metrics without rules
            max_node_age_seconds: Max age before considering node metrics stale
        """
        self._default_strategy = default_strategy
        self._max_node_age = max_node_age_seconds
        self._nodes: Dict[str, NodeMetrics] = {}
        self._rules: Dict[str, AggregationRule] = {}
        self._lock = threading.RLock()
    
    @property
    def node_count(self) -> int:
        """Get number of registered nodes."""
        return len(self._nodes)
    
    @property
    def healthy_node_count(self) -> int:
        """Get number of healthy nodes."""
        with self._lock:
            return sum(1 for n in self._nodes.values() if n.is_healthy)
    
    def add_rule(self, rule: AggregationRule) -> None:
        """
        Add an aggregation rule for a metric.
        
        Args:
            rule: The aggregation rule to add
        """
        with self._lock:
            self._rules[rule.metric_name] = rule
    
    def remove_rule(self, metric_name: str) -> None:
        """Remove an aggregation rule."""
        with self._lock:
            self._rules.pop(metric_name, None)
    
    def update_node(
        self,
        node_id: str,
        metrics: Dict[str, Any],
        is_healthy: bool = True,
        latency_ms: float = 0.0
    ) -> None:
        """
        Update metrics from a node.
        
        Args:
            node_id: Unique identifier for the node
            metrics: Dictionary of metric values
            is_healthy: Whether the node is healthy
            latency_ms: Latency to fetch these metrics
        """
        with self._lock:
            self._nodes[node_id] = NodeMetrics(
                node_id=node_id,
                timestamp=time.time(),
                metrics=metrics,
                is_healthy=is_healthy,
                latency_ms=latency_ms
            )
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the aggregator."""
        with self._lock:
            self._nodes.pop(node_id, None)
    
    def get_node(self, node_id: str) -> Optional[NodeMetrics]:
        """Get metrics for a specific node."""
        with self._lock:
            return self._nodes.get(node_id)
    
    def get_node_ids(self) -> List[str]:
        """Get list of all node IDs."""
        with self._lock:
            return list(self._nodes.keys())
    
    def _get_active_nodes(self) -> List[NodeMetrics]:
        """Get list of active (non-stale) nodes."""
        now = time.time()
        return [
            node for node in self._nodes.values()
            if (now - node.timestamp) < self._max_node_age
        ]
    
    def _aggregate_values(
        self,
        values: List[float],
        strategy: AggregationStrategy
    ) -> float:
        """Apply aggregation strategy to a list of values."""
        if not values:
            return 0.0
        
        if strategy == AggregationStrategy.SUM:
            return sum(values)
        elif strategy == AggregationStrategy.AVERAGE:
            return statistics.mean(values)
        elif strategy == AggregationStrategy.MAX:
            return max(values)
        elif strategy == AggregationStrategy.MIN:
            return min(values)
        elif strategy == AggregationStrategy.LATEST:
            return values[-1]
        elif strategy == AggregationStrategy.COUNT:
            return float(sum(1 for v in values if v > 0))
        elif strategy == AggregationStrategy.PERCENTILE_50:
            return statistics.median(values)
        elif strategy == AggregationStrategy.PERCENTILE_90:
            return self._percentile(values, 90)
        elif strategy == AggregationStrategy.PERCENTILE_99:
            return self._percentile(values, 99)
        else:
            return sum(values)
    
    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]
    
    def aggregate(self, include_stale: bool = False) -> Dict[str, Any]:
        """
        Aggregate metrics from all nodes.
        
        Args:
            include_stale: Whether to include stale node metrics
            
        Returns:
            Dictionary of aggregated metrics
        """
        with self._lock:
            nodes = list(self._nodes.values()) if include_stale else self._get_active_nodes()
            
            if not nodes:
                return {}
            
            result: Dict[str, Any] = {}
            
            # Collect all unique metric names across nodes
            all_metrics: Set[str] = set()
            for node in nodes:
                all_metrics.update(node.metrics.keys())
            
            # Aggregate each metric
            for metric_name in all_metrics:
                rule = self._rules.get(metric_name)
                strategy = rule.strategy if rule else self._default_strategy
                
                # Collect values from all nodes
                values = []
                for node in nodes:
                    if metric_name in node.metrics:
                        value = node.metrics[metric_name]
                        if isinstance(value, (int, float)):
                            values.append(float(value))
                        elif isinstance(value, dict):
                            # Handle labeled metrics - aggregate each label combo
                            for label_key, label_value in value.items():
                                if isinstance(label_value, (int, float)):
                                    # Store for later aggregation by label
                                    result_key = f"{metric_name}_{label_key}"
                                    if result_key not in result:
                                        result[result_key] = []
                                    result[result_key].append(float(label_value))
                
                if values:
                    result[metric_name] = self._aggregate_values(values, strategy)
            
            # Second pass: aggregate labeled metrics
            for key in list(result.keys()):
                if isinstance(result[key], list):
                    strategy = self._default_strategy
                    # Try to find a matching rule
                    base_metric = key.rsplit('_', 1)[0]
                    if base_metric in self._rules:
                        strategy = self._rules[base_metric].strategy
                    result[key] = self._aggregate_values(result[key], strategy)
            
            # Add metadata
            result["_aggregation_metadata"] = {
                "node_count": len(nodes),
                "healthy_nodes": sum(1 for n in nodes if n.is_healthy),
                "timestamp": time.time(),
                "include_stale": include_stale
            }
            
            return result
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """
        Get overall cluster health metrics.
        
        Returns:
            Dictionary with cluster health information
        """
        with self._lock:
            nodes = self._get_active_nodes()
            
            if not nodes:
                return {
                    "status": "unknown",
                    "total_nodes": 0,
                    "healthy_nodes": 0,
                    "unhealthy_nodes": 0,
                    "stale_nodes": len(self._nodes)
                }
            
            healthy = sum(1 for n in nodes if n.is_healthy)
            unhealthy = len(nodes) - healthy
            stale = len(self._nodes) - len(nodes)
            
            # Calculate average latency
            latencies = [n.latency_ms for n in nodes if n.latency_ms > 0]
            avg_latency = statistics.mean(latencies) if latencies else 0.0
            
            # Determine overall status
            if healthy == len(nodes) and stale == 0:
                status = "healthy"
            elif healthy > len(nodes) / 2:
                status = "degraded"
            else:
                status = "critical"
            
            return {
                "status": status,
                "total_nodes": len(self._nodes),
                "active_nodes": len(nodes),
                "healthy_nodes": healthy,
                "unhealthy_nodes": unhealthy,
                "stale_nodes": stale,
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max(latencies) if latencies else 0.0
            }
    
    def clear(self) -> None:
        """Clear all node metrics."""
        with self._lock:
            self._nodes.clear()


class DistributedCollector:
    """
    Collects metrics from remote nodes via HTTP/gRPC.
    
    Periodically fetches metrics from configured endpoints
    and updates the aggregator.
    """
    
    def __init__(
        self,
        aggregator: MetricsAggregator,
        collection_interval: float = 15.0,
        timeout_seconds: float = 5.0
    ):
        """
        Initialize the distributed collector.
        
        Args:
            aggregator: The aggregator to update
            collection_interval: Seconds between collection cycles
            timeout_seconds: Timeout for remote calls
        """
        self._aggregator = aggregator
        self._interval = collection_interval
        self._timeout = timeout_seconds
        self._endpoints: Dict[str, str] = {}
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._fetch_fn: Optional[Callable[[str], Dict[str, Any]]] = None
    
    def register_node(self, node_id: str, endpoint: str) -> None:
        """
        Register a node endpoint for collection.
        
        Args:
            node_id: Unique identifier for the node
            endpoint: HTTP/gRPC endpoint URL
        """
        self._endpoints[node_id] = endpoint
    
    def unregister_node(self, node_id: str) -> None:
        """Remove a node from collection."""
        self._endpoints.pop(node_id, None)
        self._aggregator.remove_node(node_id)
    
    def set_fetch_function(self, fn: Callable[[str], Dict[str, Any]]) -> None:
        """
        Set custom function for fetching metrics from endpoints.
        
        Args:
            fn: Function that takes endpoint URL and returns metrics dict
        """
        self._fetch_fn = fn
    
    def _default_fetch(self, endpoint: str) -> Dict[str, Any]:
        """Default fetch function (placeholder for actual HTTP call)."""
        # In real implementation, this would make HTTP/gRPC call
        # For now, return empty dict to allow testing
        return {}
    
    def _collect_from_node(self, node_id: str, endpoint: str) -> None:
        """Collect metrics from a single node."""
        fetch_fn = self._fetch_fn or self._default_fetch
        
        start_time = time.time()
        try:
            metrics = fetch_fn(endpoint)
            latency_ms = (time.time() - start_time) * 1000
            
            self._aggregator.update_node(
                node_id=node_id,
                metrics=metrics,
                is_healthy=True,
                latency_ms=latency_ms
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to collect from {node_id}: {e}")
            
            self._aggregator.update_node(
                node_id=node_id,
                metrics={},
                is_healthy=False,
                latency_ms=latency_ms
            )
    
    def collect_once(self) -> None:
        """Perform a single collection cycle."""
        threads = []
        for node_id, endpoint in self._endpoints.items():
            t = threading.Thread(
                target=self._collect_from_node,
                args=(node_id, endpoint),
                daemon=True
            )
            t.start()
            threads.append(t)
        
        # Wait for all to complete
        for t in threads:
            t.join(timeout=self._timeout)
    
    def start(self) -> None:
        """Start background collection."""
        if self._collection_thread is not None and self._collection_thread.is_alive():
            return
        
        self._stop_event.clear()
        
        def collection_loop():
            while not self._stop_event.is_set():
                self.collect_once()
                self._stop_event.wait(self._interval)
        
        self._collection_thread = threading.Thread(
            target=collection_loop,
            daemon=True,
            name="distributed-collector"
        )
        self._collection_thread.start()
        logger.info("Started distributed metrics collection")
    
    def stop(self) -> None:
        """Stop background collection."""
        self._stop_event.set()
        if self._collection_thread is not None:
            self._collection_thread.join(timeout=5.0)
            self._collection_thread = None
        logger.info("Stopped distributed metrics collection")
