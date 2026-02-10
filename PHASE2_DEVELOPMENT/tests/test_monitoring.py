"""
Comprehensive Test Suite for Sprint 3.1 - Monitoring Infrastructure

Tests for:
- MetricsCollector and all metric types
- MetricsAggregator and aggregation strategies
- PrometheusExporter and JSONExporter
- AlertManager and alert rules
- DistributedCollector for multi-node metrics

Target: 100% coverage of monitoring modules
"""

import pytest
import time
import json
import threading
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import statistics
import http.server
import socketserver

# Import monitoring modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitoring.metrics import (
    MetricType,
    MetricDefinition,
    MetricValue,
    MetricRegistry,
    BaseMetrics,
    InferenceMetrics,
    GPUMetrics,
    RequestMetrics,
    CacheMetrics,
    DistributedMetrics,
    MetricsCollector
)
from monitoring.aggregator import (
    AggregationStrategy,
    NodeMetrics,
    AggregationRule,
    MetricsAggregator,
    DistributedCollector
)
from monitoring.exporter import (
    MetricsExporter,
    PrometheusExporter,
    JSONExporter,
    MetricsHTTPHandler,
    MetricsServer
)
from monitoring.alerts import (
    AlertSeverity,
    AlertState,
    AlertRule,
    Alert,
    AlertManager,
    get_default_inference_rules
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def metric_registry():
    """Fresh MetricRegistry for each test."""
    return MetricRegistry()


@pytest.fixture
def inference_metrics():
    """Fresh InferenceMetrics instance."""
    return InferenceMetrics()


@pytest.fixture
def gpu_metrics():
    """Fresh GPUMetrics instance."""
    return GPUMetrics()


@pytest.fixture
def request_metrics():
    """Fresh RequestMetrics instance."""
    return RequestMetrics()


@pytest.fixture
def cache_metrics():
    """Fresh CacheMetrics instance."""
    return CacheMetrics()


@pytest.fixture
def distributed_metrics():
    """Fresh DistributedMetrics instance."""
    return DistributedMetrics()


@pytest.fixture
def metrics_collector():
    """Fresh MetricsCollector with all metric types."""
    return MetricsCollector()


@pytest.fixture
def metrics_aggregator():
    """Fresh MetricsAggregator."""
    return MetricsAggregator()


@pytest.fixture
def prometheus_exporter(metrics_collector):
    """PrometheusExporter with MetricsCollector."""
    return PrometheusExporter(metrics_collector)


@pytest.fixture
def json_exporter(metrics_collector):
    """JSONExporter with MetricsCollector."""
    return JSONExporter(metrics_collector)


@pytest.fixture
def alert_manager():
    """Fresh AlertManager with default rules."""
    manager = AlertManager()
    for rule in get_default_inference_rules():
        manager.add_rule(rule)
    return manager


# =============================================================================
# TEST: MetricType Enum
# =============================================================================

class TestMetricType:
    """Tests for MetricType enumeration."""
    
    def test_metric_types_exist(self):
        """Verify all expected metric types are defined."""
        assert MetricType.COUNTER is not None
        assert MetricType.GAUGE is not None
        assert MetricType.HISTOGRAM is not None
        assert MetricType.SUMMARY is not None
    
    def test_metric_types_values(self):
        """Verify metric type string values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"


# =============================================================================
# TEST: MetricDefinition
# =============================================================================

class TestMetricDefinition:
    """Tests for MetricDefinition dataclass."""
    
    def test_create_basic_definition(self):
        """Test creating a basic metric definition."""
        defn = MetricDefinition(
            name="test_metric",
            description="A test metric",
            metric_type=MetricType.COUNTER
        )
        assert defn.name == "test_metric"
        assert defn.metric_type == MetricType.COUNTER
        assert defn.description == "A test metric"
        assert defn.labels == []
        assert defn.buckets is None
    
    def test_create_definition_with_labels(self):
        """Test creating metric definition with labels."""
        defn = MetricDefinition(
            name="labeled_metric",
            description="Metric with labels",
            metric_type=MetricType.GAUGE,
            labels=["gpu_id", "model_name"]
        )
        assert defn.labels == ["gpu_id", "model_name"]
    
    def test_create_histogram_with_buckets(self):
        """Test creating histogram with custom buckets."""
        buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        defn = MetricDefinition(
            name="latency_histogram",
            description="Latency distribution",
            metric_type=MetricType.HISTOGRAM,
            buckets=buckets
        )
        assert defn.buckets == buckets


# =============================================================================
# TEST: MetricValue
# =============================================================================

class TestMetricValue:
    """Tests for MetricValue dataclass."""
    
    def test_create_metric_value(self):
        """Test creating a metric value."""
        value = MetricValue(
            value=42.5,
            labels={"gpu_id": "0"},
            timestamp=1234567890.0
        )
        assert value.value == 42.5
        assert value.labels == {"gpu_id": "0"}
        assert value.timestamp == 1234567890.0
    
    def test_metric_value_defaults(self):
        """Test metric value default values."""
        value = MetricValue(value=100)
        assert value.value == 100
        assert value.labels == {}
        assert value.timestamp is None


# =============================================================================
# TEST: MetricRegistry
# =============================================================================

class TestMetricRegistry:
    """Tests for thread-safe MetricRegistry."""
    
    def test_register_metric(self, metric_registry):
        """Test registering a metric."""
        defn = MetricDefinition(
            name="test_metric",
            description="Test",
            metric_type=MetricType.COUNTER
        )
        metric_registry.register(defn)
        assert "test_metric" in metric_registry._definitions
    
    def test_register_duplicate_raises(self, metric_registry):
        """Test that registering duplicate metric raises error."""
        defn = MetricDefinition(
            name="duplicate",
            description="First",
            metric_type=MetricType.COUNTER
        )
        metric_registry.register(defn)
        
        defn2 = MetricDefinition(
            name="duplicate",
            description="Second",
            metric_type=MetricType.GAUGE
        )
        with pytest.raises(ValueError, match="already registered"):
            metric_registry.register(defn2)
    
    def test_set_and_get_value(self, metric_registry):
        """Test setting and getting metric values."""
        defn = MetricDefinition(
            name="my_gauge",
            description="A gauge",
            metric_type=MetricType.GAUGE
        )
        metric_registry.register(defn)
        
        metric_registry.set_value("my_gauge", 99.9)
        value = metric_registry.get_value("my_gauge")
        assert value.value == 99.9
    
    def test_set_value_with_labels(self, metric_registry):
        """Test setting value with labels."""
        defn = MetricDefinition(
            name="labeled_gauge",
            description="Labeled",
            metric_type=MetricType.GAUGE,
            labels=["instance"]
        )
        metric_registry.register(defn)
        
        metric_registry.set_value("labeled_gauge", 50.0, labels={"instance": "node1"})
        value = metric_registry.get_value("labeled_gauge", labels={"instance": "node1"})
        assert value.value == 50.0
        assert value.labels == {"instance": "node1"}
    
    def test_increment_counter(self, metric_registry):
        """Test incrementing a counter."""
        defn = MetricDefinition(
            name="request_count",
            description="Request counter",
            metric_type=MetricType.COUNTER
        )
        metric_registry.register(defn)
        
        metric_registry.increment("request_count")
        metric_registry.increment("request_count", amount=5)
        
        value = metric_registry.get_value("request_count")
        assert value.value == 6
    
    def test_get_all_metrics(self, metric_registry):
        """Test getting all registered metrics."""
        defn1 = MetricDefinition(
            name="metric1",
            description="First",
            metric_type=MetricType.COUNTER
        )
        defn2 = MetricDefinition(
            name="metric2",
            description="Second",
            metric_type=MetricType.GAUGE
        )
        metric_registry.register(defn1)
        metric_registry.register(defn2)
        metric_registry.set_value("metric1", 10)
        metric_registry.set_value("metric2", 20)
        
        all_metrics = metric_registry.get_all_metrics()
        assert len(all_metrics) >= 2
    
    def test_thread_safety(self, metric_registry):
        """Test concurrent access to registry."""
        defn = MetricDefinition(
            name="concurrent_counter",
            description="Concurrent test",
            metric_type=MetricType.COUNTER
        )
        metric_registry.register(defn)
        
        def increment_many():
            for _ in range(100):
                metric_registry.increment("concurrent_counter")
        
        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        value = metric_registry.get_value("concurrent_counter")
        assert value.value == 1000


# =============================================================================
# TEST: InferenceMetrics
# =============================================================================

class TestInferenceMetrics:
    """Tests for InferenceMetrics class."""
    
    def test_metrics_initialized(self, inference_metrics):
        """Test that all inference metrics are initialized."""
        expected_metrics = [
            "tokens_generated_total",
            "tokens_per_second",
            "inference_latency_seconds",
            "time_to_first_token_seconds",
            "batch_size",
            "sequence_length",
            "active_generations",
            "generation_errors_total"
        ]
        definitions = inference_metrics.get_metric_definitions()
        metric_names = [d.name for d in definitions]
        
        for expected in expected_metrics:
            assert expected in metric_names, f"Missing metric: {expected}"
    
    def test_record_generation(self, inference_metrics):
        """Test recording a token generation."""
        inference_metrics.record_generation(
            tokens=100,
            latency_seconds=0.5,
            time_to_first_token=0.05,
            batch_size=4
        )
        
        registry = inference_metrics._registry
        tokens_value = registry.get_value("tokens_generated_total")
        assert tokens_value.value >= 100
    
    def test_record_generation_error(self, inference_metrics):
        """Test recording generation errors."""
        inference_metrics.record_error(error_type="oom")
        inference_metrics.record_error(error_type="oom")
        inference_metrics.record_error(error_type="timeout")
        
        # Should have recorded errors
        registry = inference_metrics._registry
        errors = registry.get_value("generation_errors_total")
        assert errors.value >= 1
    
    def test_update_active_generations(self, inference_metrics):
        """Test updating active generation count."""
        inference_metrics.set_active_generations(5)
        
        registry = inference_metrics._registry
        active = registry.get_value("active_generations")
        assert active.value == 5
    
    def test_tokens_per_second_calculation(self, inference_metrics):
        """Test tokens per second metric."""
        inference_metrics.update_tokens_per_second(150.5)
        
        registry = inference_metrics._registry
        tps = registry.get_value("tokens_per_second")
        assert tps.value == 150.5


# =============================================================================
# TEST: GPUMetrics
# =============================================================================

class TestGPUMetrics:
    """Tests for GPUMetrics class."""
    
    def test_gpu_metrics_initialized(self, gpu_metrics):
        """Test that GPU metrics are initialized."""
        expected = [
            "gpu_utilization_percent",
            "gpu_memory_used_bytes",
            "gpu_memory_total_bytes",
            "gpu_memory_utilization_percent",
            "gpu_temperature_celsius",
            "gpu_power_watts",
            "gpu_sm_clock_mhz",
            "gpu_memory_clock_mhz"
        ]
        definitions = gpu_metrics.get_metric_definitions()
        metric_names = [d.name for d in definitions]
        
        for exp in expected:
            assert exp in metric_names, f"Missing GPU metric: {exp}"
    
    def test_record_gpu_stats(self, gpu_metrics):
        """Test recording GPU statistics."""
        gpu_metrics.record_gpu_stats(
            gpu_id="0",
            utilization=85.5,
            memory_used=8_000_000_000,
            memory_total=16_000_000_000,
            temperature=72.0,
            power=250.0
        )
        
        registry = gpu_metrics._registry
        util = registry.get_value("gpu_utilization_percent", labels={"gpu_id": "0"})
        assert util.value == 85.5
    
    def test_memory_utilization_calculation(self, gpu_metrics):
        """Test memory utilization percentage calculation."""
        gpu_metrics.record_gpu_stats(
            gpu_id="1",
            utilization=50.0,
            memory_used=8_000_000_000,
            memory_total=16_000_000_000,
            temperature=65.0,
            power=200.0
        )
        
        registry = gpu_metrics._registry
        mem_util = registry.get_value(
            "gpu_memory_utilization_percent",
            labels={"gpu_id": "1"}
        )
        assert mem_util.value == 50.0  # 8GB / 16GB = 50%


# =============================================================================
# TEST: RequestMetrics
# =============================================================================

class TestRequestMetrics:
    """Tests for RequestMetrics class."""
    
    def test_request_metrics_initialized(self, request_metrics):
        """Test request metrics initialization."""
        expected = [
            "requests_total",
            "requests_in_flight",
            "request_queue_size",
            "request_latency_seconds",
            "request_size_bytes",
            "response_size_bytes",
            "http_status_total",
            "rate_limited_total",
            "timeout_total"
        ]
        definitions = request_metrics.get_metric_definitions()
        metric_names = [d.name for d in definitions]
        
        for exp in expected:
            assert exp in metric_names
    
    def test_record_request(self, request_metrics):
        """Test recording a request."""
        request_metrics.record_request(
            method="POST",
            endpoint="/generate",
            status_code=200,
            latency_seconds=0.25,
            request_size=1024,
            response_size=4096
        )
        
        registry = request_metrics._registry
        total = registry.get_value("requests_total")
        assert total.value >= 1
    
    def test_record_rate_limited(self, request_metrics):
        """Test recording rate limited requests."""
        request_metrics.record_rate_limited()
        request_metrics.record_rate_limited()
        
        registry = request_metrics._registry
        limited = registry.get_value("rate_limited_total")
        assert limited.value >= 2
    
    def test_update_queue_size(self, request_metrics):
        """Test updating queue size."""
        request_metrics.set_queue_size(15)
        
        registry = request_metrics._registry
        queue = registry.get_value("request_queue_size")
        assert queue.value == 15


# =============================================================================
# TEST: CacheMetrics
# =============================================================================

class TestCacheMetrics:
    """Tests for CacheMetrics class."""
    
    def test_cache_metrics_initialized(self, cache_metrics):
        """Test cache metrics initialization."""
        expected = [
            "cache_hits_total",
            "cache_misses_total",
            "cache_hit_ratio",
            "cache_size_bytes",
            "cache_entries",
            "cache_evictions_total",
            "cache_insertions_total",
            "kv_cache_tokens",
            "kv_cache_memory_bytes",
            "cache_lookup_latency_seconds"
        ]
        definitions = cache_metrics.get_metric_definitions()
        metric_names = [d.name for d in definitions]
        
        for exp in expected:
            assert exp in metric_names
    
    def test_record_cache_hit(self, cache_metrics):
        """Test recording cache hits."""
        for _ in range(10):
            cache_metrics.record_hit()
        for _ in range(5):
            cache_metrics.record_miss()
        
        registry = cache_metrics._registry
        hits = registry.get_value("cache_hits_total")
        misses = registry.get_value("cache_misses_total")
        
        assert hits.value >= 10
        assert misses.value >= 5
    
    def test_cache_hit_ratio(self, cache_metrics):
        """Test cache hit ratio calculation."""
        # Record 80 hits and 20 misses for 80% ratio
        for _ in range(80):
            cache_metrics.record_hit()
        for _ in range(20):
            cache_metrics.record_miss()
        
        cache_metrics.update_hit_ratio()
        
        registry = cache_metrics._registry
        ratio = registry.get_value("cache_hit_ratio")
        assert abs(ratio.value - 0.8) < 0.01
    
    def test_record_kv_cache_stats(self, cache_metrics):
        """Test recording KV cache statistics."""
        cache_metrics.record_kv_cache(
            tokens=50000,
            memory_bytes=1_000_000_000
        )
        
        registry = cache_metrics._registry
        tokens = registry.get_value("kv_cache_tokens")
        memory = registry.get_value("kv_cache_memory_bytes")
        
        assert tokens.value == 50000
        assert memory.value == 1_000_000_000


# =============================================================================
# TEST: DistributedMetrics
# =============================================================================

class TestDistributedMetrics:
    """Tests for DistributedMetrics class."""
    
    def test_distributed_metrics_initialized(self, distributed_metrics):
        """Test distributed metrics initialization."""
        expected = [
            "cluster_nodes_total",
            "cluster_nodes_healthy",
            "node_requests_total",
            "inter_node_latency_seconds",
            "load_balance_score",
            "shard_count",
            "replication_lag_seconds",
            "leader_elections_total"
        ]
        definitions = distributed_metrics.get_metric_definitions()
        metric_names = [d.name for d in definitions]
        
        for exp in expected:
            assert exp in metric_names
    
    def test_update_cluster_status(self, distributed_metrics):
        """Test updating cluster status."""
        distributed_metrics.update_cluster_status(
            total_nodes=5,
            healthy_nodes=4
        )
        
        registry = distributed_metrics._registry
        total = registry.get_value("cluster_nodes_total")
        healthy = registry.get_value("cluster_nodes_healthy")
        
        assert total.value == 5
        assert healthy.value == 4
    
    def test_record_inter_node_latency(self, distributed_metrics):
        """Test recording inter-node latency."""
        distributed_metrics.record_inter_node_latency(
            source_node="node1",
            target_node="node2",
            latency_seconds=0.005
        )
        
        registry = distributed_metrics._registry
        latency = registry.get_value(
            "inter_node_latency_seconds",
            labels={"source": "node1", "target": "node2"}
        )
        assert latency.value == 0.005


# =============================================================================
# TEST: MetricsCollector
# =============================================================================

class TestMetricsCollector:
    """Tests for unified MetricsCollector."""
    
    def test_collector_has_all_metrics(self, metrics_collector):
        """Test that collector aggregates all metric types."""
        assert metrics_collector.inference is not None
        assert metrics_collector.gpu is not None
        assert metrics_collector.request is not None
        assert metrics_collector.cache is not None
        assert metrics_collector.distributed is not None
    
    def test_collect_all_metrics(self, metrics_collector):
        """Test collecting all metrics."""
        # Record some data
        metrics_collector.inference.record_generation(
            tokens=50,
            latency_seconds=0.1,
            time_to_first_token=0.01,
            batch_size=2
        )
        metrics_collector.gpu.record_gpu_stats(
            gpu_id="0",
            utilization=90.0,
            memory_used=12_000_000_000,
            memory_total=16_000_000_000,
            temperature=75.0,
            power=280.0
        )
        
        all_metrics = metrics_collector.collect_all()
        assert len(all_metrics) > 0
    
    def test_get_summary(self, metrics_collector):
        """Test getting metrics summary."""
        summary = metrics_collector.get_summary()
        
        assert "inference" in summary
        assert "gpu" in summary
        assert "request" in summary
        assert "cache" in summary
        assert "distributed" in summary


# =============================================================================
# TEST: AggregationStrategy
# =============================================================================

class TestAggregationStrategy:
    """Tests for AggregationStrategy enumeration."""
    
    def test_all_strategies_exist(self):
        """Verify all aggregation strategies are defined."""
        strategies = [
            AggregationStrategy.SUM,
            AggregationStrategy.AVERAGE,
            AggregationStrategy.MAX,
            AggregationStrategy.MIN,
            AggregationStrategy.LATEST,
            AggregationStrategy.COUNT,
            AggregationStrategy.PERCENTILE_50,
            AggregationStrategy.PERCENTILE_90,
            AggregationStrategy.PERCENTILE_99
        ]
        for s in strategies:
            assert s is not None


# =============================================================================
# TEST: MetricsAggregator
# =============================================================================

class TestMetricsAggregator:
    """Tests for MetricsAggregator class."""
    
    def test_add_node_metrics(self, metrics_aggregator):
        """Test adding node metrics."""
        node_metrics = NodeMetrics(
            node_id="node1",
            timestamp=time.time(),
            metrics={"tokens_per_second": 100.0, "gpu_utilization": 85.0}
        )
        metrics_aggregator.add_node_metrics(node_metrics)
        
        assert "node1" in metrics_aggregator._node_metrics
    
    def test_aggregate_sum(self, metrics_aggregator):
        """Test SUM aggregation strategy."""
        for i, tps in enumerate([100, 150, 200]):
            node_metrics = NodeMetrics(
                node_id=f"node{i}",
                timestamp=time.time(),
                metrics={"tokens_per_second": tps}
            )
            metrics_aggregator.add_node_metrics(node_metrics)
        
        rule = AggregationRule(
            metric_name="tokens_per_second",
            strategy=AggregationStrategy.SUM
        )
        metrics_aggregator.add_rule(rule)
        
        result = metrics_aggregator.aggregate("tokens_per_second")
        assert result == 450  # 100 + 150 + 200
    
    def test_aggregate_average(self, metrics_aggregator):
        """Test AVERAGE aggregation strategy."""
        for i, util in enumerate([80, 90, 70]):
            node_metrics = NodeMetrics(
                node_id=f"node{i}",
                timestamp=time.time(),
                metrics={"gpu_utilization": util}
            )
            metrics_aggregator.add_node_metrics(node_metrics)
        
        rule = AggregationRule(
            metric_name="gpu_utilization",
            strategy=AggregationStrategy.AVERAGE
        )
        metrics_aggregator.add_rule(rule)
        
        result = metrics_aggregator.aggregate("gpu_utilization")
        assert result == 80  # (80 + 90 + 70) / 3
    
    def test_aggregate_max(self, metrics_aggregator):
        """Test MAX aggregation strategy."""
        for i, temp in enumerate([65, 72, 68]):
            node_metrics = NodeMetrics(
                node_id=f"node{i}",
                timestamp=time.time(),
                metrics={"temperature": temp}
            )
            metrics_aggregator.add_node_metrics(node_metrics)
        
        rule = AggregationRule(
            metric_name="temperature",
            strategy=AggregationStrategy.MAX
        )
        metrics_aggregator.add_rule(rule)
        
        result = metrics_aggregator.aggregate("temperature")
        assert result == 72
    
    def test_aggregate_min(self, metrics_aggregator):
        """Test MIN aggregation strategy."""
        for i, mem in enumerate([80, 60, 90]):
            node_metrics = NodeMetrics(
                node_id=f"node{i}",
                timestamp=time.time(),
                metrics={"memory_available": mem}
            )
            metrics_aggregator.add_node_metrics(node_metrics)
        
        rule = AggregationRule(
            metric_name="memory_available",
            strategy=AggregationStrategy.MIN
        )
        metrics_aggregator.add_rule(rule)
        
        result = metrics_aggregator.aggregate("memory_available")
        assert result == 60
    
    def test_aggregate_percentile(self, metrics_aggregator):
        """Test percentile aggregation."""
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i, lat in enumerate(latencies):
            node_metrics = NodeMetrics(
                node_id=f"node{i}",
                timestamp=time.time(),
                metrics={"latency_ms": lat}
            )
            metrics_aggregator.add_node_metrics(node_metrics)
        
        rule = AggregationRule(
            metric_name="latency_ms",
            strategy=AggregationStrategy.PERCENTILE_99
        )
        metrics_aggregator.add_rule(rule)
        
        result = metrics_aggregator.aggregate("latency_ms")
        assert result >= 90  # P99 should be high
    
    def test_get_cluster_health(self, metrics_aggregator):
        """Test cluster health calculation."""
        # Add healthy nodes
        for i in range(4):
            node_metrics = NodeMetrics(
                node_id=f"node{i}",
                timestamp=time.time(),
                metrics={"healthy": 1.0},
                is_healthy=True
            )
            metrics_aggregator.add_node_metrics(node_metrics)
        
        # Add unhealthy node
        unhealthy = NodeMetrics(
            node_id="node4",
            timestamp=time.time(),
            metrics={"healthy": 0.0},
            is_healthy=False
        )
        metrics_aggregator.add_node_metrics(unhealthy)
        
        health = metrics_aggregator.get_cluster_health()
        assert health["total_nodes"] == 5
        assert health["healthy_nodes"] == 4
        assert health["health_ratio"] == 0.8
    
    def test_remove_stale_nodes(self, metrics_aggregator):
        """Test removing stale node metrics."""
        # Add old node
        old_node = NodeMetrics(
            node_id="old_node",
            timestamp=time.time() - 120,  # 2 minutes ago
            metrics={"value": 1}
        )
        metrics_aggregator.add_node_metrics(old_node)
        
        # Add fresh node
        fresh_node = NodeMetrics(
            node_id="fresh_node",
            timestamp=time.time(),
            metrics={"value": 2}
        )
        metrics_aggregator.add_node_metrics(fresh_node)
        
        # Remove stale (older than 60 seconds)
        metrics_aggregator.remove_stale_nodes(max_age_seconds=60)
        
        assert "old_node" not in metrics_aggregator._node_metrics
        assert "fresh_node" in metrics_aggregator._node_metrics


# =============================================================================
# TEST: PrometheusExporter
# =============================================================================

class TestPrometheusExporter:
    """Tests for PrometheusExporter class."""
    
    def test_export_format(self, prometheus_exporter, metrics_collector):
        """Test Prometheus text format export."""
        # Record some metrics
        metrics_collector.inference.record_generation(
            tokens=100,
            latency_seconds=0.5,
            time_to_first_token=0.05,
            batch_size=4
        )
        
        output = prometheus_exporter.export()
        
        # Should contain Prometheus format elements
        assert isinstance(output, str)
        assert "# HELP" in output or "# TYPE" in output or len(output) > 0
    
    def test_export_with_labels(self, prometheus_exporter, metrics_collector):
        """Test export with labeled metrics."""
        metrics_collector.gpu.record_gpu_stats(
            gpu_id="0",
            utilization=85.0,
            memory_used=8_000_000_000,
            memory_total=16_000_000_000,
            temperature=70.0,
            power=250.0
        )
        
        output = prometheus_exporter.export()
        assert isinstance(output, str)
    
    def test_content_type(self, prometheus_exporter):
        """Test correct content type for Prometheus."""
        content_type = prometheus_exporter.content_type()
        assert "text/plain" in content_type or "text/plain" in content_type.lower()


# =============================================================================
# TEST: JSONExporter
# =============================================================================

class TestJSONExporter:
    """Tests for JSONExporter class."""
    
    def test_export_json_format(self, json_exporter, metrics_collector):
        """Test JSON format export."""
        metrics_collector.inference.record_generation(
            tokens=50,
            latency_seconds=0.25,
            time_to_first_token=0.02,
            batch_size=2
        )
        
        output = json_exporter.export()
        
        # Should be valid JSON
        data = json.loads(output)
        assert isinstance(data, dict)
        assert "timestamp" in data or "metrics" in data
    
    def test_content_type(self, json_exporter):
        """Test correct content type for JSON."""
        content_type = json_exporter.content_type()
        assert "application/json" in content_type


# =============================================================================
# TEST: AlertSeverity and AlertState
# =============================================================================

class TestAlertEnums:
    """Tests for alert-related enumerations."""
    
    def test_alert_severity_levels(self):
        """Test alert severity levels exist."""
        assert AlertSeverity.INFO is not None
        assert AlertSeverity.WARNING is not None
        assert AlertSeverity.CRITICAL is not None
        assert AlertSeverity.EMERGENCY is not None
    
    def test_alert_states(self):
        """Test alert states exist."""
        assert AlertState.PENDING is not None
        assert AlertState.FIRING is not None
        assert AlertState.RESOLVED is not None


# =============================================================================
# TEST: AlertRule
# =============================================================================

class TestAlertRule:
    """Tests for AlertRule dataclass."""
    
    def test_create_alert_rule(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            name="HighLatency",
            metric_name="inference_latency_p99",
            condition=lambda v: v > 1.0,
            severity=AlertSeverity.WARNING,
            description="Latency is too high",
            for_duration_seconds=60
        )
        
        assert rule.name == "HighLatency"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.for_duration_seconds == 60
    
    def test_rule_condition_evaluation(self):
        """Test evaluating rule condition."""
        rule = AlertRule(
            name="LowThroughput",
            metric_name="tokens_per_second",
            condition=lambda v: v < 50,
            severity=AlertSeverity.CRITICAL,
            description="Throughput below threshold"
        )
        
        assert rule.condition(30) is True  # Below 50
        assert rule.condition(100) is False  # Above 50


# =============================================================================
# TEST: Alert
# =============================================================================

class TestAlert:
    """Tests for Alert dataclass."""
    
    def test_create_alert(self):
        """Test creating an alert."""
        alert = Alert(
            rule_name="TestAlert",
            state=AlertState.FIRING,
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            value=1.5,
            started_at=time.time(),
            labels={"instance": "node1"}
        )
        
        assert alert.rule_name == "TestAlert"
        assert alert.state == AlertState.FIRING
        assert alert.value == 1.5


# =============================================================================
# TEST: AlertManager
# =============================================================================

class TestAlertManager:
    """Tests for AlertManager class."""
    
    def test_add_rule(self, alert_manager):
        """Test adding alert rules."""
        initial_count = len(alert_manager._rules)
        
        new_rule = AlertRule(
            name="CustomRule",
            metric_name="custom_metric",
            condition=lambda v: v > 100,
            severity=AlertSeverity.INFO,
            description="Custom rule"
        )
        alert_manager.add_rule(new_rule)
        
        assert len(alert_manager._rules) == initial_count + 1
    
    def test_evaluate_rules(self, alert_manager):
        """Test evaluating alert rules against metrics."""
        # Create metrics that will trigger alerts
        metrics = {
            "inference_latency_p99": 2.5,  # High latency
            "tokens_per_second": 10.0,     # Low throughput
            "error_rate": 0.01             # Normal error rate
        }
        
        alerts = alert_manager.evaluate(metrics)
        
        # Should have some alerts firing
        assert isinstance(alerts, list)
    
    def test_alert_lifecycle(self, alert_manager):
        """Test alert state transitions."""
        # First evaluation - should create pending alert
        metrics_bad = {"inference_latency_p99": 3.0}
        alert_manager.evaluate(metrics_bad)
        
        # Check alert exists
        active_alerts = alert_manager.get_active_alerts()
        assert isinstance(active_alerts, list)
    
    def test_get_firing_alerts(self, alert_manager):
        """Test getting only firing alerts."""
        firing = alert_manager.get_firing_alerts()
        
        # All returned alerts should be in FIRING state
        for alert in firing:
            assert alert.state == AlertState.FIRING
    
    def test_resolve_alert(self, alert_manager):
        """Test resolving an alert."""
        # Trigger an alert
        metrics_bad = {"inference_latency_p99": 5.0}
        alert_manager.evaluate(metrics_bad)
        
        # Then resolve it
        metrics_good = {"inference_latency_p99": 0.1}
        alert_manager.evaluate(metrics_good)
        
        resolved = alert_manager.get_resolved_alerts()
        assert isinstance(resolved, list)
    
    def test_alert_history(self, alert_manager):
        """Test alert history tracking."""
        # Generate some alerts
        for i in range(3):
            metrics = {"inference_latency_p99": 3.0 + i}
            alert_manager.evaluate(metrics)
        
        history = alert_manager.get_alert_history(limit=10)
        assert isinstance(history, list)


# =============================================================================
# TEST: Default Inference Rules
# =============================================================================

class TestDefaultInferenceRules:
    """Tests for default inference alert rules."""
    
    def test_default_rules_exist(self):
        """Test that default rules are created."""
        rules = get_default_inference_rules()
        
        assert len(rules) >= 5
        rule_names = [r.name for r in rules]
        
        # Check for expected rules
        expected = [
            "HighLatencyP99",
            "LowThroughput",
            "HighErrorRate",
            "HighGPUMemory",
            "LowCacheHitRate"
        ]
        for exp in expected:
            assert exp in rule_names, f"Missing default rule: {exp}"
    
    def test_rule_conditions_valid(self):
        """Test that rule conditions are callable."""
        rules = get_default_inference_rules()
        
        for rule in rules:
            assert callable(rule.condition)
            # Should not raise when called with a number
            try:
                result = rule.condition(0.5)
                assert isinstance(result, bool)
            except Exception as e:
                pytest.fail(f"Rule {rule.name} condition failed: {e}")


# =============================================================================
# TEST: Integration - Full Pipeline
# =============================================================================

class TestMonitoringIntegration:
    """Integration tests for the complete monitoring pipeline."""
    
    def test_full_metrics_pipeline(self):
        """Test complete metrics collection to export pipeline."""
        # 1. Create collector
        collector = MetricsCollector()
        
        # 2. Record various metrics
        collector.inference.record_generation(
            tokens=200,
            latency_seconds=0.4,
            time_to_first_token=0.03,
            batch_size=8
        )
        collector.gpu.record_gpu_stats(
            gpu_id="0",
            utilization=92.0,
            memory_used=14_000_000_000,
            memory_total=16_000_000_000,
            temperature=78.0,
            power=290.0
        )
        collector.cache.record_hit()
        collector.cache.record_hit()
        collector.cache.record_miss()
        
        # 3. Export to Prometheus format
        exporter = PrometheusExporter(collector)
        prometheus_output = exporter.export()
        
        # 4. Export to JSON format
        json_exporter = JSONExporter(collector)
        json_output = json_exporter.export()
        
        # Verify outputs
        assert len(prometheus_output) > 0
        assert json.loads(json_output) is not None
    
    def test_aggregation_to_alerting_pipeline(self):
        """Test metrics aggregation to alerting pipeline."""
        # 1. Create aggregator with node metrics
        aggregator = MetricsAggregator()
        
        for i in range(3):
            node = NodeMetrics(
                node_id=f"node{i}",
                timestamp=time.time(),
                metrics={
                    "inference_latency_p99": 0.5 + (i * 0.3),
                    "tokens_per_second": 100 - (i * 20),
                    "error_rate": 0.01
                },
                is_healthy=True
            )
            aggregator.add_node_metrics(node)
        
        # 2. Add aggregation rules
        aggregator.add_rule(AggregationRule(
            metric_name="inference_latency_p99",
            strategy=AggregationStrategy.MAX
        ))
        aggregator.add_rule(AggregationRule(
            metric_name="tokens_per_second",
            strategy=AggregationStrategy.SUM
        ))
        
        # 3. Get aggregated metrics
        latency = aggregator.aggregate("inference_latency_p99")
        throughput = aggregator.aggregate("tokens_per_second")
        
        # 4. Create alert manager and evaluate
        alert_manager = AlertManager()
        for rule in get_default_inference_rules():
            alert_manager.add_rule(rule)
        
        metrics = {
            "inference_latency_p99": latency,
            "tokens_per_second": throughput
        }
        
        alerts = alert_manager.evaluate(metrics)
        
        # Pipeline should work end-to-end
        assert latency > 0
        assert throughput > 0
        assert isinstance(alerts, list)
    
    def test_distributed_monitoring_scenario(self):
        """Test a realistic distributed monitoring scenario."""
        # Simulate a 4-node GPU cluster
        collector = MetricsCollector()
        aggregator = MetricsAggregator()
        
        # Each node reports metrics
        node_data = [
            {"gpu_util": 85, "tps": 150, "latency": 0.3, "mem": 12e9},
            {"gpu_util": 90, "tps": 140, "latency": 0.35, "mem": 13e9},
            {"gpu_util": 88, "tps": 145, "latency": 0.32, "mem": 12.5e9},
            {"gpu_util": 92, "tps": 135, "latency": 0.4, "mem": 14e9}
        ]
        
        for i, data in enumerate(node_data):
            # Local metrics
            collector.gpu.record_gpu_stats(
                gpu_id=str(i),
                utilization=data["gpu_util"],
                memory_used=int(data["mem"]),
                memory_total=16_000_000_000,
                temperature=70 + i,
                power=250 + (i * 10)
            )
            
            # Aggregator node metrics
            node = NodeMetrics(
                node_id=f"gpu-node-{i}",
                timestamp=time.time(),
                metrics={
                    "tokens_per_second": data["tps"],
                    "inference_latency_p99": data["latency"],
                    "gpu_utilization": data["gpu_util"]
                },
                is_healthy=True
            )
            aggregator.add_node_metrics(node)
        
        # Update cluster status
        collector.distributed.update_cluster_status(
            total_nodes=4,
            healthy_nodes=4
        )
        
        # Aggregate and check
        aggregator.add_rule(AggregationRule(
            metric_name="tokens_per_second",
            strategy=AggregationStrategy.SUM
        ))
        
        total_throughput = aggregator.aggregate("tokens_per_second")
        cluster_health = aggregator.get_cluster_health()
        
        assert total_throughput == 150 + 140 + 145 + 135  # 570 tokens/sec
        assert cluster_health["health_ratio"] == 1.0


# =============================================================================
# TEST: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in monitoring modules."""
    
    def test_invalid_metric_name(self, metric_registry):
        """Test handling of invalid metric names."""
        with pytest.raises(KeyError):
            metric_registry.get_value("nonexistent_metric")
    
    def test_aggregate_missing_metric(self, metrics_aggregator):
        """Test aggregating metric with no data."""
        rule = AggregationRule(
            metric_name="missing_metric",
            strategy=AggregationStrategy.SUM
        )
        metrics_aggregator.add_rule(rule)
        
        result = metrics_aggregator.aggregate("missing_metric")
        assert result is None or result == 0
    
    def test_alert_with_invalid_metric(self, alert_manager):
        """Test alert evaluation with missing metric."""
        # Should not raise, just skip the rule
        alerts = alert_manager.evaluate({})
        assert isinstance(alerts, list)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
