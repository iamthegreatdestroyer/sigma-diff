"""
Production Monitoring and Observability
=======================================

Comprehensive monitoring, metrics collection, and observability
for production distributed inference systems.

Key Features:
- Real-time metrics collection (latency, throughput, errors)
- Distributed tracing and request correlation
- Health monitoring and alerting
- Performance profiling and bottleneck detection
- Resource utilization tracking
- Custom dashboards and visualization
"""

import torch
import torch.distributed as dist
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics for monitoring."""
    COUNTER = "counter"      # Monotonically increasing value
    GAUGE = "gauge"         # Point-in-time value
    HISTOGRAM = "histogram" # Distribution of values
    SUMMARY = "summary"     # Quantiles over sliding window


@dataclass
class MetricValue:
    """Individual metric measurement."""
    name: str
    value: Union[int, float]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class TraceSpan:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


class MetricsCollector:
    """
    High-performance metrics collection for distributed systems.

    Features:
    - Low-overhead metrics collection
    - Distributed aggregation
    - Real-time alerting thresholds
    - Historical data retention
    - Export to monitoring backends
    """

    def __init__(
        self,
        collection_interval: float = 1.0,
        retention_period: int = 3600,  # 1 hour
        enable_distributed_aggregation: bool = True
    ):
        self.collection_interval = collection_interval
        self.retention_period = retention_period
        self.enable_distributed_aggregation = enable_distributed_aggregation

        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        # Alerting
        self.alert_thresholds: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}

        # Aggregation
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = {}

        # Threading
        self.collection_thread: Optional[threading.Thread] = None
        self.running = False

        # Initialize standard metrics
        self._init_standard_metrics()

        logger.info("MetricsCollector initialized")

    def _init_standard_metrics(self):
        """Initialize standard system metrics."""
        # Inference metrics
        self.create_counter("inference_requests_total", "Total inference requests")
        self.create_gauge("inference_latency_ms", "Inference latency in milliseconds")
        self.create_gauge("inference_throughput_req_per_sec", "Requests per second")
        self.create_histogram("inference_batch_sizes", "Batch size distribution")

        # Resource metrics
        self.create_gauge("gpu_memory_used_mb", "GPU memory used in MB")
        self.create_gauge("gpu_memory_total_mb", "Total GPU memory in MB")
        self.create_gauge("gpu_utilization_percent", "GPU utilization percentage")
        self.create_gauge("cpu_usage_percent", "CPU usage percentage")

        # Error metrics
        self.create_counter("errors_total", "Total errors by type", labels=["error_type"])
        self.create_gauge("error_rate_per_minute", "Errors per minute")

        # Cache metrics
        self.create_gauge("cache_hit_rate", "Cache hit rate percentage")
        self.create_gauge("cache_memory_used_mb", "Cache memory used in MB")

    def create_counter(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Create a counter metric."""
        self.counters[name] = 0
        logger.debug(f"Created counter metric: {name}")

    def create_gauge(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Create a gauge metric."""
        self.gauges[name] = 0.0
        logger.debug(f"Created gauge metric: {name}")

    def create_histogram(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Create a histogram metric."""
        self.histograms[name] = []
        logger.debug(f"Created histogram metric: {name}")

    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        self.counters[key] += value

        # Record metric
        metric = MetricValue(
            name=name,
            value=self.counters[key],
            timestamp=time.time(),
            labels=labels or {},
            metric_type=MetricType.COUNTER
        )
        self._store_metric(metric)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self.gauges[key] = value

        # Record metric
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            metric_type=MetricType.GAUGE
        )
        self._store_metric(metric)

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value in a histogram."""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)

        # Keep only recent observations
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-500:]

        # Record metric
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            metric_type=MetricType.HISTOGRAM
        )
        self._store_metric(metric)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Make a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _store_metric(self, metric: MetricValue):
        """Store metric in circular buffer."""
        self.metrics[metric.name].append(metric)

        # Clean old metrics
        cutoff_time = time.time() - self.retention_period
        while self.metrics[metric.name] and self.metrics[metric.name][0].timestamp < cutoff_time:
            self.metrics[metric.name].popleft()

    def start_collection(self):
        """Start metrics collection thread."""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")

    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._check_alerts()
                self._aggregate_distributed_metrics()

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # GPU metrics
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_used = torch.cuda.memory_allocated(i) / 1024 / 1024  # MB
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024  # MB

                    self.set_gauge("gpu_memory_used_mb", memory_used, {"gpu": str(i)})
                    self.set_gauge("gpu_memory_total_mb", memory_total, {"gpu": str(i)})

                    # GPU utilization (simplified)
                    utilization = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0.0
                    self.set_gauge("gpu_utilization_percent", utilization, {"gpu": str(i)})

            # CPU metrics (simplified)
            import psutil
            cpu_percent = psutil.cpu_percent(interval=None)
            self.set_gauge("cpu_usage_percent", cpu_percent)

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def _check_alerts(self):
        """Check metric thresholds and trigger alerts."""
        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name in self.gauges:
                value = self.gauges[metric_name]

                for threshold_name, threshold_config in thresholds.items():
                    threshold_value = threshold_config.get("value", 0)
                    comparison = threshold_config.get("comparison", "gt")  # gt, lt, eq

                    alert_triggered = False
                    if comparison == "gt" and value > threshold_value:
                        alert_triggered = True
                    elif comparison == "lt" and value < threshold_value:
                        alert_triggered = True
                    elif comparison == "eq" and abs(value - threshold_value) < 0.001:
                        alert_triggered = True

                    if alert_triggered:
                        alert_key = f"{metric_name}:{threshold_name}"
                        if alert_key not in self.active_alerts:
                            self.active_alerts[alert_key] = {
                                "metric": metric_name,
                                "threshold": threshold_name,
                                "value": value,
                                "threshold_value": threshold_value,
                                "timestamp": time.time(),
                                "message": threshold_config.get("message", f"Alert: {metric_name} {comparison} {threshold_value}")
                            }
                            logger.warning(f"ALERT: {self.active_alerts[alert_key]['message']}")

    def _aggregate_distributed_metrics(self):
        """Aggregate metrics across distributed nodes."""
        if not self.enable_distributed_aggregation or not dist.is_initialized():
            return

        try:
            # Aggregate counters across nodes
            for name, value in self.counters.items():
                if "total" in name:  # Only aggregate totals
                    tensor = torch.tensor([float(value)], dtype=torch.float32)
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    self.aggregated_metrics[name] = {"sum": tensor.item()}

            # Aggregate gauges (average)
            for name, value in self.gauges.items():
                tensor = torch.tensor([float(value)], dtype=torch.float32)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                avg_value = tensor.item() / dist.get_world_size()
                self.aggregated_metrics[name] = {"average": avg_value}

        except Exception as e:
            logger.warning(f"Failed to aggregate distributed metrics: {e}")

    def set_alert_threshold(
        self,
        metric_name: str,
        threshold_name: str,
        value: float,
        comparison: str = "gt",
        message: Optional[str] = None
    ):
        """Set an alert threshold for a metric."""
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}

        self.alert_thresholds[metric_name][threshold_name] = {
            "value": value,
            "comparison": comparison,
            "message": message or f"{metric_name} {comparison} {value}"
        }

        logger.info(f"Set alert threshold: {metric_name} {comparison} {value}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0
                }
                for name, values in self.histograms.items()
            },
            "active_alerts": self.active_alerts,
            "aggregated_metrics": self.aggregated_metrics,
            "retention_info": {
                "collection_interval": self.collection_interval,
                "retention_period": self.retention_period,
                "total_metrics_stored": sum(len(metrics) for metrics in self.metrics.values())
            }
        }

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        summary = self.get_metrics_summary()

        if format == "json":
            return json.dumps(summary, indent=2, default=str)
        elif format == "prometheus":
            return self._export_prometheus_format(summary)
        else:
            return str(summary)

    def _export_prometheus_format(self, summary: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Counters
        for name, value in summary["counters"].items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # Gauges
        for name, value in summary["gauges"].items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        # Histograms (simplified)
        for name, stats in summary["histograms"].items():
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"{name}_count {stats['count']}")
            lines.append(f"{name}_sum {stats['avg'] * stats['count']}")

        return "\n".join(lines)


class DistributedTracer:
    """
    Distributed tracing for request correlation and performance analysis.

    Features:
    - Request tracing across distributed nodes
    - Performance bottleneck identification
    - Service dependency mapping
    - Trace sampling and retention
    """

    def __init__(
        self,
        service_name: str,
        sample_rate: float = 1.0,
        max_traces: int = 10000,
        trace_timeout: float = 300.0  # 5 minutes
    ):
        self.service_name = service_name
        self.sample_rate = sample_rate
        self.max_traces = max_traces
        self.trace_timeout = trace_timeout

        # Trace storage
        self.active_traces: Dict[str, TraceSpan] = {}
        self.completed_traces: deque = deque(maxlen=max_traces)

        # Performance tracking
        self.span_durations: Dict[str, List[float]] = defaultdict(list)

        logger.info(f"DistributedTracer initialized for service: {service_name}")

    def start_span(
        self,
        operation: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new trace span."""
        if trace_id is None:
            trace_id = self._generate_trace_id()

        span_id = self._generate_span_id()

        # Sample decision
        if torch.rand(1).item() > self.sample_rate:
            return span_id  # Return span_id but don't track

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
            tags=tags or {}
        )

        self.active_traces[span_id] = span
        return span_id

    def end_span(self, span_id: str, tags: Optional[Dict[str, Any]] = None):
        """End a trace span."""
        if span_id not in self.active_traces:
            return

        span = self.active_traces[span_id]
        span.end_time = time.time()
        span.duration = span.end_time - span.start_time

        if tags:
            span.tags.update(tags)

        # Move to completed traces
        self.completed_traces.append(span)
        del self.active_traces[span_id]

        # Track performance
        self.span_durations[span.operation].append(span.duration)

        # Keep only recent durations
        if len(self.span_durations[span.operation]) > 1000:
            self.span_durations[span.operation] = self.span_durations[span.operation][-500:]

    def add_span_event(self, span_id: str, event: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to a span."""
        if span_id not in self.active_traces:
            return

        event_data = {
            "name": event,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }

        self.active_traces[span_id].events.append(event_data)

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        spans = []
        for span in self.completed_traces:
            if span.trace_id == trace_id:
                spans.append(span)
        return spans

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for operations."""
        stats = {}
        for operation, durations in self.span_durations.items():
            if durations:
                stats[operation] = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "p50": self._percentile(durations, 50),
                    "p95": self._percentile(durations, 95),
                    "p99": self._percentile(durations, 99)
                }
        return stats

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from data."""
        if not data:
            return 0.0
        data_sorted = sorted(data)
        index = int(len(data_sorted) * percentile / 100)
        return data_sorted[min(index, len(data_sorted) - 1)]

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return f"{self.service_name}-{int(time.time() * 1000000)}-{torch.randint(0, 1000000, (1,)).item()}"

    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return f"span-{torch.randint(0, 1000000, (1,)).item()}"

    def cleanup_expired_traces(self):
        """Clean up expired active traces."""
        current_time = time.time()
        expired_spans = []

        for span_id, span in self.active_traces.items():
            if current_time - span.start_time > self.trace_timeout:
                expired_spans.append(span_id)

        for span_id in expired_spans:
            span = self.active_traces[span_id]
            span.tags["expired"] = True
            self.end_span(span_id, {"expired": True})

    def export_traces(self, format: str = "json") -> str:
        """Export traces in specified format."""
        traces = list(self.completed_traces)

        if format == "json":
            return json.dumps([self._span_to_dict(span) for span in traces], indent=2, default=str)
        else:
            return str(traces)

    def _span_to_dict(self, span: TraceSpan) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "operation": span.operation,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration": span.duration,
            "tags": span.tags,
            "events": span.events
        }


class HealthMonitor:
    """
    Health monitoring for distributed inference system.

    Features:
    - Component health checks
    - Dependency monitoring
    - Automated health reporting
    - Failure prediction
    """

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval

        # Health checks
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}

        # Dependencies
        self.dependencies: Dict[str, List[str]] = defaultdict(list)

        # Threading
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False

        # Initialize standard health checks
        self._init_standard_checks()

        logger.info("HealthMonitor initialized")

    def _init_standard_checks(self):
        """Initialize standard health checks."""
        self.add_health_check("gpu_available", self._check_gpu_available)
        self.add_health_check("memory_sufficient", self._check_memory_sufficient)
        self.add_health_check("network_connectivity", self._check_network_connectivity)
        self.add_health_check("disk_space", self._check_disk_space)

    def add_health_check(self, name: str, check_func: Callable[[], bool]):
        """Add a custom health check."""
        self.health_checks[name] = check_func
        logger.debug(f"Added health check: {name}")

    def add_dependency(self, component: str, dependency: str):
        """Add a dependency relationship."""
        self.dependencies[component].append(dependency)

    def start_monitoring(self):
        """Start health monitoring."""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")

    def _monitoring_loop(self):
        """Main health monitoring loop."""
        while self.running:
            try:
                self._run_health_checks()
                self._update_overall_health()

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")

    def _run_health_checks(self):
        """Run all health checks."""
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = check_func()
                status = "healthy" if is_healthy else "unhealthy"

                self.health_status[name] = {
                    "status": status,
                    "last_check": time.time(),
                    "error_count": self.health_status.get(name, {}).get("error_count", 0)
                }

                if not is_healthy:
                    self.health_status[name]["error_count"] += 1
                    logger.warning(f"Health check failed: {name}")

            except Exception as e:
                logger.error(f"Health check error for {name}: {e}")
                self.health_status[name] = {
                    "status": "error",
                    "last_check": time.time(),
                    "error": str(e),
                    "error_count": self.health_status.get(name, {}).get("error_count", 0) + 1
                }

    def _update_overall_health(self):
        """Update overall system health based on component health."""
        total_checks = len(self.health_status)
        healthy_checks = sum(1 for status in self.health_status.values() if status["status"] == "healthy")

        overall_health = "healthy" if healthy_checks == total_checks else "degraded"
        if healthy_checks < total_checks * 0.5:
            overall_health = "unhealthy"

        self.health_status["_overall"] = {
            "status": overall_health,
            "healthy_checks": healthy_checks,
            "total_checks": total_checks,
            "last_update": time.time()
        }

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available() and torch.cuda.device_count() > 0

    def _check_memory_sufficient(self) -> bool:
        """Check if memory is sufficient."""
        if not torch.cuda.is_available():
            return True

        try:
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            return memory_used < 0.9  # Less than 90% used
        except:
            return True

    def _check_network_connectivity(self) -> bool:
        """Check network connectivity."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except:
            return False

    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            stat = os.statvfs('.')
            free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            return free_space_gb > 1.0  # At least 1GB free
        except:
            return True

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "overall_health": self.health_status.get("_overall", {}),
            "component_health": {
                name: status for name, status in self.health_status.items() if name != "_overall"
            },
            "dependencies": dict(self.dependencies),
            "monitoring_config": {
                "check_interval": self.check_interval,
                "total_checks": len(self.health_checks)
            }
        }

    def is_system_healthy(self) -> bool:
        """Check if overall system is healthy."""
        overall = self.health_status.get("_overall", {})
        return overall.get("status") == "healthy"
