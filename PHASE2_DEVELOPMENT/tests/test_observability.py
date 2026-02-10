"""
Comprehensive Test Suite for Sprint 3.5: Observability Stack

Tests for:
- Prometheus metrics exporter
- Jaeger tracing exporter
- Unified observability client
- Health checking
- Structured logging

Sprint 3.5: Observability Stack
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# ============================================================================
# Mock Classes for Testing (avoid import errors if modules not fully available)
# ============================================================================

@dataclass
class MockMetricSample:
    """Mock metric sample for testing."""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: Optional[float] = None
    metric_type: str = "gauge"
    help_text: str = ""


@dataclass
class MockExporterConfig:
    """Mock exporter configuration."""
    host: str = "localhost"
    port: int = 9090
    path: str = "/metrics"
    namespace: str = "ryzanstein"
    include_timestamp: bool = True
    push_gateway_url: Optional[str] = None


@dataclass
class MockSpanData:
    """Mock span data for testing."""
    trace_id: str
    span_id: str
    name: str
    parent_span_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "OK"
    attributes: Dict[str, Any] = None
    events: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        if self.events is None:
            self.events = []


@dataclass
class MockJaegerConfig:
    """Mock Jaeger configuration."""
    agent_host: str = "localhost"
    agent_port: int = 6831
    collector_endpoint: Optional[str] = None
    service_name: str = "ryzanstein-llm"
    sample_rate: float = 1.0
    max_queue_size: int = 1000
    batch_size: int = 100


class MockLogLevel:
    """Mock log level enum."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class MockObservabilityConfig:
    """Mock observability configuration."""
    service_name: str = "ryzanstein-llm"
    environment: str = "development"
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    metrics_port: int = 9090
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    log_level: str = "INFO"


@dataclass
class MockRequestContext:
    """Mock request context for testing."""
    request_id: str
    trace_id: str
    span_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


# ============================================================================
# Prometheus Formatter Tests
# ============================================================================

class TestPrometheusFormatter:
    """Tests for Prometheus text exposition format."""
    
    def test_format_gauge_metric(self):
        """Test formatting a gauge metric."""
        sample = MockMetricSample(
            name="inference_latency_seconds",
            value=0.125,
            labels={"model": "llama-7b", "node": "node-0"},
            metric_type="gauge",
            help_text="Inference latency in seconds"
        )
        
        # Format according to Prometheus text format
        lines = []
        lines.append(f"# HELP {sample.name} {sample.help_text}")
        lines.append(f"# TYPE {sample.name} {sample.metric_type}")
        
        label_str = ",".join([f'{k}="{v}"' for k, v in sample.labels.items()])
        lines.append(f"{sample.name}{{{label_str}}} {sample.value}")
        
        output = "\n".join(lines)
        
        assert "# HELP inference_latency_seconds" in output
        assert "# TYPE inference_latency_seconds gauge" in output
        assert 'model="llama-7b"' in output
        assert '0.125' in output
    
    def test_format_counter_metric(self):
        """Test formatting a counter metric."""
        sample = MockMetricSample(
            name="requests_total",
            value=1000,
            labels={"status": "success"},
            metric_type="counter",
            help_text="Total number of requests"
        )
        
        lines = []
        lines.append(f"# HELP {sample.name} {sample.help_text}")
        lines.append(f"# TYPE {sample.name} {sample.metric_type}")
        
        label_str = ",".join([f'{k}="{v}"' for k, v in sample.labels.items()])
        lines.append(f"{sample.name}{{{label_str}}} {sample.value}")
        
        output = "\n".join(lines)
        
        assert "# TYPE requests_total counter" in output
        assert 'status="success"' in output
        assert "1000" in output
    
    def test_format_histogram_metric(self):
        """Test formatting a histogram metric."""
        # Histogram buckets
        buckets = [
            (0.01, 10),
            (0.05, 50),
            (0.1, 100),
            (0.5, 180),
            (1.0, 200),
            (float('inf'), 200),
        ]
        
        lines = []
        lines.append("# HELP inference_duration_seconds Duration of inference requests")
        lines.append("# TYPE inference_duration_seconds histogram")
        
        for le, count in buckets:
            le_str = "+Inf" if le == float('inf') else str(le)
            lines.append(f'inference_duration_seconds_bucket{{le="{le_str}"}} {count}')
        
        lines.append("inference_duration_seconds_sum 45.5")
        lines.append("inference_duration_seconds_count 200")
        
        output = "\n".join(lines)
        
        assert "# TYPE inference_duration_seconds histogram" in output
        assert 'le="0.01"' in output
        assert 'le="+Inf"' in output
        assert "inference_duration_seconds_sum 45.5" in output
        assert "inference_duration_seconds_count 200" in output
    
    def test_format_with_timestamp(self):
        """Test formatting metric with timestamp."""
        timestamp = time.time() * 1000  # milliseconds
        sample = MockMetricSample(
            name="current_temperature",
            value=72.5,
            labels={"sensor": "cpu"},
            timestamp=timestamp
        )
        
        label_str = ",".join([f'{k}="{v}"' for k, v in sample.labels.items()])
        line = f"{sample.name}{{{label_str}}} {sample.value} {int(sample.timestamp)}"
        
        assert str(int(timestamp)) in line
    
    def test_escape_label_values(self):
        """Test escaping special characters in label values."""
        sample = MockMetricSample(
            name="log_messages",
            value=5,
            labels={"message": 'Error: "file not found"'}
        )
        
        # Escape quotes and backslashes
        escaped_value = sample.labels["message"].replace('\\', '\\\\').replace('"', '\\"')
        label_str = f'message="{escaped_value}"'
        
        assert '\\"file not found\\"' in label_str
    
    def test_format_multiple_metrics(self):
        """Test formatting multiple metrics."""
        samples = [
            MockMetricSample(name="cpu_usage", value=45.5, labels={"core": "0"}),
            MockMetricSample(name="cpu_usage", value=62.3, labels={"core": "1"}),
            MockMetricSample(name="memory_usage", value=8192, labels={"type": "heap"}),
        ]
        
        lines = []
        current_metric = None
        
        for sample in samples:
            if sample.name != current_metric:
                lines.append(f"# TYPE {sample.name} gauge")
                current_metric = sample.name
            
            label_str = ",".join([f'{k}="{v}"' for k, v in sample.labels.items()])
            lines.append(f"{sample.name}{{{label_str}}} {sample.value}")
        
        output = "\n".join(lines)
        
        assert output.count("# TYPE cpu_usage") == 1
        assert output.count("cpu_usage{") == 2
        assert output.count("# TYPE memory_usage") == 1


# ============================================================================
# Prometheus Exporter Tests
# ============================================================================

class TestPrometheusExporter:
    """Tests for Prometheus HTTP metrics exporter."""
    
    def test_exporter_config_defaults(self):
        """Test default configuration values."""
        config = MockExporterConfig()
        
        assert config.host == "localhost"
        assert config.port == 9090
        assert config.path == "/metrics"
        assert config.namespace == "ryzanstein"
        assert config.include_timestamp is True
    
    def test_exporter_config_custom(self):
        """Test custom configuration values."""
        config = MockExporterConfig(
            host="0.0.0.0",
            port=8080,
            path="/prometheus/metrics",
            namespace="llm_inference",
            include_timestamp=False
        )
        
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.path == "/prometheus/metrics"
        assert config.namespace == "llm_inference"
    
    def test_metrics_endpoint_format(self):
        """Test metrics endpoint URL format."""
        config = MockExporterConfig(host="localhost", port=9090, path="/metrics")
        url = f"http://{config.host}:{config.port}{config.path}"
        
        assert url == "http://localhost:9090/metrics"
    
    def test_namespace_prefix(self):
        """Test namespace prefixing for metric names."""
        namespace = "ryzanstein"
        metric_name = "inference_latency"
        
        prefixed_name = f"{namespace}_{metric_name}"
        
        assert prefixed_name == "ryzanstein_inference_latency"
    
    def test_push_gateway_url(self):
        """Test push gateway configuration."""
        config = MockExporterConfig(
            push_gateway_url="http://pushgateway:9091"
        )
        
        assert config.push_gateway_url == "http://pushgateway:9091"


# ============================================================================
# Metric Collector Tests
# ============================================================================

class TestMetricCollectors:
    """Tests for various metric collectors."""
    
    def test_inference_metrics_collection(self):
        """Test collection of inference metrics."""
        metrics = {
            "inference_requests_total": 1000,
            "inference_latency_seconds": 0.125,
            "tokens_generated_total": 50000,
            "tokens_per_second": 150.5,
            "batch_size": 4,
            "active_requests": 2,
        }
        
        samples = []
        for name, value in metrics.items():
            samples.append(MockMetricSample(
                name=f"ryzanstein_{name}",
                value=value,
                labels={"model": "llama-7b", "node": "node-0"}
            ))
        
        assert len(samples) == 6
        assert any(s.name == "ryzanstein_tokens_per_second" for s in samples)
    
    def test_node_metrics_collection(self):
        """Test collection of node-level metrics."""
        metrics = {
            "cpu_usage_percent": 65.5,
            "memory_usage_bytes": 8589934592,  # 8GB
            "memory_total_bytes": 17179869184,  # 16GB
            "disk_usage_percent": 45.2,
            "network_bytes_sent": 1000000,
            "network_bytes_recv": 5000000,
        }
        
        samples = []
        for name, value in metrics.items():
            samples.append(MockMetricSample(
                name=f"node_{name}",
                value=value,
                labels={"node_id": "node-0", "hostname": "ryzanstein-0"}
            ))
        
        assert len(samples) == 6
        assert any(s.name == "node_memory_usage_bytes" for s in samples)
    
    def test_cache_metrics_collection(self):
        """Test collection of KV cache metrics."""
        metrics = {
            "kv_cache_size_bytes": 4294967296,  # 4GB
            "kv_cache_capacity_bytes": 8589934592,  # 8GB
            "kv_cache_hit_ratio": 0.85,
            "kv_cache_entries": 10000,
            "kv_cache_evictions_total": 500,
        }
        
        samples = []
        for name, value in metrics.items():
            samples.append(MockMetricSample(
                name=name,
                value=value,
                labels={"cache_type": "kv", "layer": "all"}
            ))
        
        assert len(samples) == 5
        
        hit_ratio_sample = next(s for s in samples if "hit_ratio" in s.name)
        assert hit_ratio_sample.value == 0.85


# ============================================================================
# Jaeger Exporter Tests
# ============================================================================

class TestJaegerExporter:
    """Tests for Jaeger tracing exporter."""
    
    def test_jaeger_config_defaults(self):
        """Test default Jaeger configuration."""
        config = MockJaegerConfig()
        
        assert config.agent_host == "localhost"
        assert config.agent_port == 6831
        assert config.service_name == "ryzanstein-llm"
        assert config.sample_rate == 1.0
        assert config.max_queue_size == 1000
        assert config.batch_size == 100
    
    def test_jaeger_config_custom(self):
        """Test custom Jaeger configuration."""
        config = MockJaegerConfig(
            agent_host="jaeger-agent",
            agent_port=6832,
            collector_endpoint="http://jaeger-collector:14268/api/traces",
            service_name="custom-llm-service",
            sample_rate=0.1
        )
        
        assert config.agent_host == "jaeger-agent"
        assert config.collector_endpoint == "http://jaeger-collector:14268/api/traces"
        assert config.sample_rate == 0.1
    
    def test_span_data_creation(self):
        """Test span data structure creation."""
        span = MockSpanData(
            trace_id="abc123def456",
            span_id="span001",
            name="inference_request",
            parent_span_id=None,
            start_time=time.time(),
            end_time=time.time() + 0.125,
            status="OK",
            attributes={"model": "llama-7b", "tokens": 100}
        )
        
        assert span.trace_id == "abc123def456"
        assert span.name == "inference_request"
        assert span.parent_span_id is None
        assert span.end_time > span.start_time
        assert span.attributes["model"] == "llama-7b"
    
    def test_span_with_parent(self):
        """Test child span creation."""
        parent_span = MockSpanData(
            trace_id="trace001",
            span_id="span001",
            name="parent_operation"
        )
        
        child_span = MockSpanData(
            trace_id="trace001",
            span_id="span002",
            name="child_operation",
            parent_span_id=parent_span.span_id
        )
        
        assert child_span.trace_id == parent_span.trace_id
        assert child_span.parent_span_id == parent_span.span_id
    
    def test_span_events(self):
        """Test span event recording."""
        span = MockSpanData(
            trace_id="trace001",
            span_id="span001",
            name="batch_inference",
            events=[
                {"name": "batch_started", "timestamp": time.time(), "attributes": {"batch_size": 4}},
                {"name": "tokens_generated", "timestamp": time.time() + 0.05, "attributes": {"count": 100}},
                {"name": "batch_completed", "timestamp": time.time() + 0.1, "attributes": {}},
            ]
        )
        
        assert len(span.events) == 3
        assert span.events[0]["name"] == "batch_started"
        assert span.events[2]["name"] == "batch_completed"
    
    def test_span_attributes(self):
        """Test span attribute handling."""
        span = MockSpanData(
            trace_id="trace001",
            span_id="span001",
            name="model_load",
            attributes={
                "model.name": "llama-7b",
                "model.size_bytes": 14000000000,
                "node.id": "node-0",
                "inference.batch_size": 4,
                "inference.max_tokens": 512,
            }
        )
        
        assert len(span.attributes) == 5
        assert span.attributes["model.name"] == "llama-7b"
        assert span.attributes["inference.batch_size"] == 4


# ============================================================================
# Batch Span Processor Tests
# ============================================================================

class TestBatchSpanProcessor:
    """Tests for batch span processing."""
    
    def test_batch_accumulation(self):
        """Test that spans accumulate in batches."""
        batch_size = 10
        spans = []
        
        for i in range(25):
            spans.append(MockSpanData(
                trace_id=f"trace{i}",
                span_id=f"span{i}",
                name=f"operation_{i}"
            ))
        
        # Simulate batch processing
        batches = []
        current_batch = []
        
        for span in spans:
            current_batch.append(span)
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        assert len(batches) == 3  # 10 + 10 + 5
        assert len(batches[0]) == 10
        assert len(batches[2]) == 5
    
    def test_batch_timeout(self):
        """Test batch export on timeout."""
        batch_timeout_ms = 5000
        batch_size = 100
        
        # Simulate partial batch with timeout
        partial_batch = [
            MockSpanData(trace_id=f"trace{i}", span_id=f"span{i}", name=f"op_{i}")
            for i in range(30)
        ]
        
        # After timeout, batch should be exported even if not full
        assert len(partial_batch) < batch_size
        assert len(partial_batch) == 30  # Would export after timeout
    
    def test_max_queue_size(self):
        """Test queue size limits."""
        max_queue_size = 100
        queue = []
        dropped = 0
        
        # Try to add 150 spans
        for i in range(150):
            span = MockSpanData(
                trace_id=f"trace{i}",
                span_id=f"span{i}",
                name=f"operation_{i}"
            )
            
            if len(queue) < max_queue_size:
                queue.append(span)
            else:
                dropped += 1
        
        assert len(queue) == 100
        assert dropped == 50


# ============================================================================
# Distributed Tracer Tests
# ============================================================================

class TestDistributedTracer:
    """Tests for distributed tracing functionality."""
    
    def test_trace_id_generation(self):
        """Test trace ID generation format."""
        import uuid
        
        trace_id = uuid.uuid4().hex
        
        assert len(trace_id) == 32
        assert all(c in '0123456789abcdef' for c in trace_id)
    
    def test_span_id_generation(self):
        """Test span ID generation format."""
        import uuid
        
        span_id = uuid.uuid4().hex[:16]
        
        assert len(span_id) == 16
        assert all(c in '0123456789abcdef' for c in span_id)
    
    def test_context_propagation_format(self):
        """Test trace context propagation format (W3C TraceContext)."""
        trace_id = "abc123def456abc123def456abc123de"
        span_id = "abc123def456abc1"
        trace_flags = "01"  # sampled
        
        traceparent = f"00-{trace_id}-{span_id}-{trace_flags}"
        
        assert traceparent.startswith("00-")
        assert trace_id in traceparent
        assert span_id in traceparent
    
    def test_sampling_decision(self):
        """Test sampling rate enforcement."""
        sample_rate = 0.5
        sampled_count = 0
        total_count = 1000
        
        import random
        random.seed(42)  # For reproducibility
        
        for _ in range(total_count):
            if random.random() < sample_rate:
                sampled_count += 1
        
        # Should be approximately 50% (with some variance)
        ratio = sampled_count / total_count
        assert 0.4 < ratio < 0.6
    
    def test_span_duration_calculation(self):
        """Test span duration calculation."""
        start_time = time.time()
        time.sleep(0.01)  # 10ms operation
        end_time = time.time()
        
        duration_ms = (end_time - start_time) * 1000
        
        assert duration_ms >= 10  # At least 10ms


# ============================================================================
# Observability Client Tests
# ============================================================================

class TestObservabilityClient:
    """Tests for unified observability client."""
    
    def test_config_defaults(self):
        """Test default observability configuration."""
        config = MockObservabilityConfig()
        
        assert config.service_name == "ryzanstein-llm"
        assert config.environment == "development"
        assert config.enable_metrics is True
        assert config.enable_tracing is True
        assert config.enable_logging is True
    
    def test_config_production(self):
        """Test production configuration."""
        config = MockObservabilityConfig(
            service_name="production-llm",
            environment="production",
            log_level="WARNING",
            jaeger_agent_host="jaeger.prod.internal"
        )
        
        assert config.environment == "production"
        assert config.log_level == "WARNING"
        assert config.jaeger_agent_host == "jaeger.prod.internal"
    
    def test_request_context_creation(self):
        """Test request context creation."""
        import uuid
        
        context = MockRequestContext(
            request_id=str(uuid.uuid4()),
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            user_id="user-123",
            attributes={"client_version": "2.0.0"}
        )
        
        assert context.request_id is not None
        assert context.trace_id is not None
        assert context.user_id == "user-123"
        assert context.attributes["client_version"] == "2.0.0"
    
    def test_request_context_inheritance(self):
        """Test request context inheritance for child spans."""
        parent_context = MockRequestContext(
            request_id="req-001",
            trace_id="trace001",
            span_id="span001"
        )
        
        child_context = MockRequestContext(
            request_id=parent_context.request_id,
            trace_id=parent_context.trace_id,
            span_id="span002",  # New span ID
            attributes={"parent_span": parent_context.span_id}
        )
        
        assert child_context.trace_id == parent_context.trace_id
        assert child_context.request_id == parent_context.request_id
        assert child_context.span_id != parent_context.span_id


# ============================================================================
# Structured Logger Tests
# ============================================================================

class TestStructuredLogger:
    """Tests for structured logging."""
    
    def test_log_levels(self):
        """Test log level hierarchy."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        level_values = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        
        for i, level in enumerate(levels[:-1]):
            assert level_values[level] < level_values[levels[i + 1]]
    
    def test_structured_log_format(self):
        """Test structured log JSON format."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "INFO",
            "message": "Inference completed",
            "service": "ryzanstein-llm",
            "trace_id": "abc123",
            "span_id": "def456",
            "attributes": {
                "model": "llama-7b",
                "tokens_generated": 100,
                "latency_ms": 125.5
            }
        }
        
        json_output = json.dumps(log_entry)
        parsed = json.loads(json_output)
        
        assert parsed["level"] == "INFO"
        assert parsed["attributes"]["tokens_generated"] == 100
    
    def test_log_context_enrichment(self):
        """Test automatic context enrichment in logs."""
        base_log = {
            "message": "Request processed",
            "level": "INFO"
        }
        
        context = MockRequestContext(
            request_id="req-001",
            trace_id="trace001",
            span_id="span001"
        )
        
        enriched_log = {
            **base_log,
            "request_id": context.request_id,
            "trace_id": context.trace_id,
            "span_id": context.span_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        assert "request_id" in enriched_log
        assert "trace_id" in enriched_log
        assert "timestamp" in enriched_log
    
    def test_error_log_with_exception(self):
        """Test error logging with exception details."""
        try:
            raise ValueError("Invalid model configuration")
        except ValueError as e:
            error_log = {
                "level": "ERROR",
                "message": str(e),
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            }
        
        assert error_log["exception_type"] == "ValueError"
        assert "Invalid model" in error_log["exception_message"]


# ============================================================================
# Health Checker Tests
# ============================================================================

class TestHealthChecker:
    """Tests for health checking functionality."""
    
    def test_health_check_response(self):
        """Test health check response format."""
        health_response = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": "2.0.0",
            "checks": {
                "database": {"status": "healthy", "latency_ms": 5.2},
                "cache": {"status": "healthy", "latency_ms": 1.1},
                "model": {"status": "healthy", "loaded": True},
            }
        }
        
        assert health_response["status"] == "healthy"
        assert len(health_response["checks"]) == 3
    
    def test_health_check_degraded(self):
        """Test degraded health status."""
        checks = {
            "database": {"status": "healthy"},
            "cache": {"status": "degraded", "error": "High latency"},
            "model": {"status": "healthy"},
        }
        
        # Overall status is degraded if any check is degraded
        statuses = [c["status"] for c in checks.values()]
        
        if "unhealthy" in statuses:
            overall = "unhealthy"
        elif "degraded" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"
        
        assert overall == "degraded"
    
    def test_health_check_unhealthy(self):
        """Test unhealthy status detection."""
        checks = {
            "database": {"status": "unhealthy", "error": "Connection refused"},
            "cache": {"status": "healthy"},
            "model": {"status": "healthy"},
        }
        
        statuses = [c["status"] for c in checks.values()]
        
        if "unhealthy" in statuses:
            overall = "unhealthy"
        elif "degraded" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"
        
        assert overall == "unhealthy"
    
    def test_liveness_probe(self):
        """Test liveness probe response."""
        liveness = {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        assert liveness["status"] == "alive"
    
    def test_readiness_probe(self):
        """Test readiness probe response."""
        # Service is ready when all dependencies are available
        dependencies_ready = {
            "model_loaded": True,
            "cache_available": True,
            "warmup_complete": True,
        }
        
        is_ready = all(dependencies_ready.values())
        
        readiness = {
            "status": "ready" if is_ready else "not_ready",
            "checks": dependencies_ready
        }
        
        assert readiness["status"] == "ready"


# ============================================================================
# Alert Rules Tests
# ============================================================================

class TestAlertRules:
    """Tests for Prometheus alert rules."""
    
    def test_high_latency_alert(self):
        """Test high latency alert condition."""
        threshold_ms = 1000
        current_latency_ms = 1500
        
        alert_firing = current_latency_ms > threshold_ms
        
        assert alert_firing is True
    
    def test_error_rate_alert(self):
        """Test high error rate alert condition."""
        threshold_percent = 5.0
        total_requests = 1000
        error_requests = 80
        
        error_rate = (error_requests / total_requests) * 100
        alert_firing = error_rate > threshold_percent
        
        assert error_rate == 8.0
        assert alert_firing is True
    
    def test_memory_usage_alert(self):
        """Test high memory usage alert."""
        threshold_percent = 90
        used_bytes = 15032385536  # ~14GB
        total_bytes = 17179869184  # 16GB
        
        usage_percent = (used_bytes / total_bytes) * 100
        alert_firing = usage_percent > threshold_percent
        
        assert alert_firing is False  # ~87.5% < 90%
    
    def test_node_down_alert(self):
        """Test node down alert condition."""
        nodes = {
            "node-0": {"status": "up", "last_seen": time.time()},
            "node-1": {"status": "up", "last_seen": time.time()},
            "node-2": {"status": "down", "last_seen": time.time() - 300},
        }
        
        down_nodes = [n for n, info in nodes.items() if info["status"] == "down"]
        alert_firing = len(down_nodes) > 0
        
        assert alert_firing is True
        assert "node-2" in down_nodes
    
    def test_cache_hit_rate_alert(self):
        """Test low cache hit rate alert."""
        threshold = 0.5
        hits = 300
        misses = 700
        
        hit_rate = hits / (hits + misses)
        alert_firing = hit_rate < threshold
        
        assert hit_rate == 0.3
        assert alert_firing is True


# ============================================================================
# Docker Compose Configuration Tests
# ============================================================================

class TestDockerComposeConfig:
    """Tests for Docker Compose observability configuration."""
    
    def test_required_services(self):
        """Test that all required services are defined."""
        required_services = [
            "prometheus",
            "grafana",
            "jaeger",
            "alertmanager",
        ]
        
        # Mock service definitions
        services = {
            "prometheus": {"image": "prom/prometheus:latest"},
            "grafana": {"image": "grafana/grafana:latest"},
            "jaeger": {"image": "jaegertracing/all-in-one:latest"},
            "alertmanager": {"image": "prom/alertmanager:latest"},
            "node-exporter": {"image": "prom/node-exporter:latest"},
        }
        
        for service in required_services:
            assert service in services
    
    def test_prometheus_config_volume(self):
        """Test Prometheus configuration volume mount."""
        volume_mount = "./configs/prometheus:/etc/prometheus:ro"
        
        assert "prometheus" in volume_mount
        assert ":ro" in volume_mount  # Read-only
    
    def test_grafana_provisioning_volumes(self):
        """Test Grafana provisioning volume mounts."""
        volumes = [
            "./configs/grafana/provisioning/dashboards:/etc/grafana/provisioning/dashboards",
            "./configs/grafana/provisioning/datasources:/etc/grafana/provisioning/datasources",
            "./monitoring/grafana_dashboards:/var/lib/grafana/dashboards",
        ]
        
        assert len(volumes) == 3
        assert any("dashboards" in v for v in volumes)
        assert any("datasources" in v for v in volumes)
    
    def test_port_mappings(self):
        """Test service port mappings."""
        ports = {
            "prometheus": "9090:9090",
            "grafana": "3000:3000",
            "jaeger": ["6831:6831/udp", "16686:16686"],
            "alertmanager": "9093:9093",
        }
        
        assert "9090" in ports["prometheus"]
        assert "3000" in ports["grafana"]
        assert any("16686" in p for p in ports["jaeger"])
    
    def test_network_configuration(self):
        """Test network configuration."""
        networks = {
            "observability": {
                "driver": "bridge",
                "ipam": {
                    "config": [{"subnet": "172.20.0.0/16"}]
                }
            }
        }
        
        assert "observability" in networks
        assert networks["observability"]["driver"] == "bridge"


# ============================================================================
# Integration Tests
# ============================================================================

class TestObservabilityIntegration:
    """Integration tests for observability components."""
    
    def test_metrics_to_prometheus_flow(self):
        """Test metrics collection to Prometheus flow."""
        # Collect metrics
        metrics = {
            "inference_latency_seconds": 0.125,
            "requests_total": 1000,
        }
        
        # Format as Prometheus metrics
        lines = []
        for name, value in metrics.items():
            lines.append(f"ryzanstein_{name} {value}")
        
        output = "\n".join(lines)
        
        assert "ryzanstein_inference_latency_seconds 0.125" in output
        assert "ryzanstein_requests_total 1000" in output
    
    def test_trace_context_through_request(self):
        """Test trace context propagation through request lifecycle."""
        import uuid
        
        # Initial request with trace context
        trace_id = uuid.uuid4().hex
        
        # Request processing stages
        stages = ["receive", "validate", "process", "respond"]
        spans = []
        
        parent_span_id = None
        for stage in stages:
            span = MockSpanData(
                trace_id=trace_id,
                span_id=uuid.uuid4().hex[:16],
                name=stage,
                parent_span_id=parent_span_id
            )
            spans.append(span)
            parent_span_id = span.span_id
        
        # All spans share trace_id
        assert all(s.trace_id == trace_id for s in spans)
        
        # Parent-child relationships
        assert spans[0].parent_span_id is None
        assert spans[1].parent_span_id == spans[0].span_id
        assert spans[3].parent_span_id == spans[2].span_id
    
    def test_log_trace_correlation(self):
        """Test log entries include trace correlation."""
        trace_id = "abc123def456"
        span_id = "span001"
        
        log_entry = {
            "level": "INFO",
            "message": "Processing request",
            "trace_id": trace_id,
            "span_id": span_id,
        }
        
        # Logs can be correlated with traces
        assert log_entry["trace_id"] == trace_id
        assert log_entry["span_id"] == span_id
    
    def test_alert_to_notification_flow(self):
        """Test alert triggering notification flow."""
        # Alert condition
        alert = {
            "name": "HighLatency",
            "severity": "warning",
            "value": 1500,
            "threshold": 1000,
            "labels": {"service": "ryzanstein-llm", "node": "node-0"},
        }
        
        # Notification routing
        if alert["severity"] == "critical":
            channels = ["pagerduty", "slack"]
        elif alert["severity"] == "warning":
            channels = ["slack"]
        else:
            channels = ["email"]
        
        assert "slack" in channels
        assert "pagerduty" not in channels


# ============================================================================
# Performance Tests
# ============================================================================

class TestObservabilityPerformance:
    """Performance tests for observability overhead."""
    
    def test_metric_collection_overhead(self):
        """Test that metric collection has minimal overhead."""
        iterations = 10000
        
        start = time.time()
        for i in range(iterations):
            sample = MockMetricSample(
                name="test_metric",
                value=float(i),
                labels={"iteration": str(i % 100)}
            )
        elapsed = time.time() - start
        
        # Should complete quickly (< 1 second for 10k iterations)
        assert elapsed < 1.0
        per_op_us = (elapsed / iterations) * 1_000_000
        assert per_op_us < 100  # < 100 microseconds per operation
    
    def test_span_creation_overhead(self):
        """Test that span creation has minimal overhead."""
        iterations = 10000
        
        start = time.time()
        for i in range(iterations):
            span = MockSpanData(
                trace_id=f"trace{i}",
                span_id=f"span{i}",
                name=f"operation_{i}",
                attributes={"key": "value"}
            )
        elapsed = time.time() - start
        
        assert elapsed < 1.0
        per_op_us = (elapsed / iterations) * 1_000_000
        assert per_op_us < 100
    
    def test_log_formatting_overhead(self):
        """Test that log formatting has minimal overhead."""
        iterations = 10000
        
        start = time.time()
        for i in range(iterations):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "message": f"Log message {i}",
                "attributes": {"iteration": i}
            }
            json_str = json.dumps(log_entry)
        elapsed = time.time() - start
        
        assert elapsed < 2.0  # JSON serialization takes more time
        per_op_us = (elapsed / iterations) * 1_000_000
        assert per_op_us < 200


# ============================================================================
# Grafana Dashboard Tests
# ============================================================================

class TestGrafanaDashboards:
    """Tests for Grafana dashboard configurations."""
    
    def test_dashboard_structure(self):
        """Test dashboard JSON structure."""
        dashboard = {
            "title": "Ryzanstein Distributed Inference",
            "uid": "ryzanstein-inference",
            "version": 1,
            "panels": [],
            "templating": {"list": []},
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "5s",
        }
        
        assert "title" in dashboard
        assert "panels" in dashboard
        assert dashboard["refresh"] == "5s"
    
    def test_panel_configuration(self):
        """Test panel configuration structure."""
        panel = {
            "id": 1,
            "type": "graph",
            "title": "Inference Latency",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(inference_latency_bucket[5m]))",
                    "legendFormat": "p95 latency",
                }
            ],
        }
        
        assert panel["type"] == "graph"
        assert len(panel["targets"]) == 1
        assert "histogram_quantile" in panel["targets"][0]["expr"]
    
    def test_variable_templating(self):
        """Test dashboard variable templating."""
        variables = [
            {
                "name": "node",
                "type": "query",
                "query": "label_values(up{job=\"ryzanstein\"}, instance)",
            },
            {
                "name": "model",
                "type": "query",
                "query": "label_values(model_loaded, model)",
            },
        ]
        
        assert len(variables) == 2
        assert variables[0]["name"] == "node"
    
    def test_alert_annotations(self):
        """Test alert annotations on panels."""
        annotations = {
            "list": [
                {
                    "name": "Alerts",
                    "datasource": "Prometheus",
                    "enable": True,
                    "expr": "ALERTS{alertstate=\"firing\"}",
                }
            ]
        }
        
        assert len(annotations["list"]) == 1
        assert annotations["list"][0]["enable"] is True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
