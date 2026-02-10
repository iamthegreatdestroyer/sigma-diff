"""
Prometheus HTTP Exporter for Distributed Inference.

Provides HTTP endpoints for Prometheus metric scraping:
- /metrics - Standard Prometheus metrics endpoint
- /health - Health check endpoint
- /ready - Readiness probe

Sprint 3.5 - Observability Stack
Created: 2026-01-06
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MetricSample:
    """A single metric sample for export."""
    name: str
    labels: Dict[str, str]
    value: float
    timestamp: Optional[float] = None
    metric_type: str = "gauge"
    help_text: str = ""


@dataclass
class ExporterConfig:
    """Configuration for Prometheus exporter."""
    host: str = "0.0.0.0"
    port: int = 9100
    path: str = "/metrics"
    enable_default_metrics: bool = True
    namespace: str = "llm_inference"


class PrometheusFormatter:
    """Formats metrics in Prometheus text exposition format."""
    
    @staticmethod
    def format_metric_name(name: str, namespace: str = "") -> str:
        """Format metric name according to Prometheus naming conventions."""
        # Replace dashes and dots with underscores
        formatted = name.replace("-", "_").replace(".", "_")
        if namespace:
            return f"{namespace}_{formatted}"
        return formatted
    
    @staticmethod
    def format_labels(labels: Dict[str, str]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        
        parts = []
        for key, value in sorted(labels.items()):
            # Escape special characters in label values
            escaped_value = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            parts.append(f'{key}="{escaped_value}"')
        
        return "{" + ",".join(parts) + "}"
    
    @staticmethod
    def format_sample(sample: MetricSample, namespace: str = "") -> str:
        """Format a single metric sample."""
        name = PrometheusFormatter.format_metric_name(sample.name, namespace)
        labels = PrometheusFormatter.format_labels(sample.labels)
        
        if sample.timestamp:
            return f"{name}{labels} {sample.value} {int(sample.timestamp * 1000)}"
        return f"{name}{labels} {sample.value}"
    
    @staticmethod
    def format_metrics(
        samples: List[MetricSample],
        namespace: str = ""
    ) -> str:
        """Format all metrics in Prometheus exposition format."""
        output_lines = []
        
        # Group samples by metric name for TYPE and HELP annotations
        metrics_by_name: Dict[str, List[MetricSample]] = {}
        for sample in samples:
            if sample.name not in metrics_by_name:
                metrics_by_name[sample.name] = []
            metrics_by_name[sample.name].append(sample)
        
        for metric_name, metric_samples in sorted(metrics_by_name.items()):
            full_name = PrometheusFormatter.format_metric_name(metric_name, namespace)
            first_sample = metric_samples[0]
            
            # Add HELP line
            if first_sample.help_text:
                output_lines.append(f"# HELP {full_name} {first_sample.help_text}")
            
            # Add TYPE line
            output_lines.append(f"# TYPE {full_name} {first_sample.metric_type}")
            
            # Add all samples
            for sample in metric_samples:
                output_lines.append(
                    PrometheusFormatter.format_sample(sample, namespace)
                )
        
        return "\n".join(output_lines) + "\n"


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    @abstractmethod
    def collect(self) -> List[MetricSample]:
        """Collect current metric samples."""
        pass


class InferenceMetricsCollector(MetricCollector):
    """Collects inference-related metrics."""
    
    def __init__(self):
        self.requests_total = 0
        self.tokens_generated = 0
        self.request_latencies: List[float] = []
        self.batch_sizes: List[int] = []
        self.errors_total = 0
        self._lock = threading.Lock()
    
    def record_request(
        self,
        latency_ms: float,
        tokens: int,
        batch_size: int = 1,
        error: bool = False
    ) -> None:
        """Record a completed request."""
        with self._lock:
            self.requests_total += 1
            self.tokens_generated += tokens
            self.request_latencies.append(latency_ms)
            self.batch_sizes.append(batch_size)
            if error:
                self.errors_total += 1
            
            # Keep sliding window
            if len(self.request_latencies) > 1000:
                self.request_latencies = self.request_latencies[-1000:]
                self.batch_sizes = self.batch_sizes[-1000:]
    
    def collect(self) -> List[MetricSample]:
        """Collect inference metrics."""
        samples = []
        
        with self._lock:
            # Total requests counter
            samples.append(MetricSample(
                name="requests_total",
                labels={},
                value=float(self.requests_total),
                metric_type="counter",
                help_text="Total number of inference requests processed"
            ))
            
            # Total tokens counter
            samples.append(MetricSample(
                name="tokens_generated_total",
                labels={},
                value=float(self.tokens_generated),
                metric_type="counter",
                help_text="Total number of tokens generated"
            ))
            
            # Error counter
            samples.append(MetricSample(
                name="errors_total",
                labels={},
                value=float(self.errors_total),
                metric_type="counter",
                help_text="Total number of errors"
            ))
            
            # Latency statistics
            if self.request_latencies:
                avg_latency = sum(self.request_latencies) / len(self.request_latencies)
                samples.append(MetricSample(
                    name="request_latency_ms",
                    labels={"quantile": "avg"},
                    value=avg_latency,
                    metric_type="gauge",
                    help_text="Request latency in milliseconds"
                ))
                
                sorted_latencies = sorted(self.request_latencies)
                for quantile in [0.5, 0.9, 0.95, 0.99]:
                    idx = int(len(sorted_latencies) * quantile)
                    samples.append(MetricSample(
                        name="request_latency_ms",
                        labels={"quantile": str(quantile)},
                        value=sorted_latencies[min(idx, len(sorted_latencies) - 1)],
                        metric_type="gauge",
                        help_text="Request latency in milliseconds"
                    ))
            
            # Throughput (tokens per second)
            if self.request_latencies:
                total_time_sec = sum(self.request_latencies) / 1000.0
                if total_time_sec > 0:
                    throughput = self.tokens_generated / total_time_sec
                    samples.append(MetricSample(
                        name="throughput_tokens_per_sec",
                        labels={},
                        value=throughput,
                        metric_type="gauge",
                        help_text="Token generation throughput"
                    ))
            
            # Average batch size
            if self.batch_sizes:
                avg_batch = sum(self.batch_sizes) / len(self.batch_sizes)
                samples.append(MetricSample(
                    name="batch_size_avg",
                    labels={},
                    value=avg_batch,
                    metric_type="gauge",
                    help_text="Average batch size"
                ))
        
        return samples


class NodeMetricsCollector(MetricCollector):
    """Collects distributed node metrics."""
    
    def __init__(self, node_id: str = "node-0"):
        self.node_id = node_id
        self.healthy = True
        self.active_connections = 0
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.gpu_utilization = 0.0
        self.gpu_memory_used = 0.0
        self.gpu_memory_total = 0.0
        self._lock = threading.Lock()
    
    def update_health(self, healthy: bool) -> None:
        with self._lock:
            self.healthy = healthy
    
    def update_resources(
        self,
        cpu: float = 0.0,
        memory: float = 0.0,
        gpu_util: float = 0.0,
        gpu_mem_used: float = 0.0,
        gpu_mem_total: float = 0.0
    ) -> None:
        with self._lock:
            self.cpu_usage = cpu
            self.memory_usage = memory
            self.gpu_utilization = gpu_util
            self.gpu_memory_used = gpu_mem_used
            self.gpu_memory_total = gpu_mem_total
    
    def update_connections(self, count: int) -> None:
        with self._lock:
            self.active_connections = count
    
    def collect(self) -> List[MetricSample]:
        samples = []
        labels = {"node_id": self.node_id}
        
        with self._lock:
            samples.append(MetricSample(
                name="node_healthy",
                labels=labels,
                value=1.0 if self.healthy else 0.0,
                metric_type="gauge",
                help_text="Node health status (1=healthy, 0=unhealthy)"
            ))
            
            samples.append(MetricSample(
                name="node_active_connections",
                labels=labels,
                value=float(self.active_connections),
                metric_type="gauge",
                help_text="Number of active connections"
            ))
            
            samples.append(MetricSample(
                name="cpu_usage_percent",
                labels=labels,
                value=self.cpu_usage,
                metric_type="gauge",
                help_text="CPU usage percentage"
            ))
            
            samples.append(MetricSample(
                name="memory_usage_percent",
                labels=labels,
                value=self.memory_usage,
                metric_type="gauge",
                help_text="Memory usage percentage"
            ))
            
            samples.append(MetricSample(
                name="gpu_utilization_percent",
                labels=labels,
                value=self.gpu_utilization,
                metric_type="gauge",
                help_text="GPU utilization percentage"
            ))
            
            samples.append(MetricSample(
                name="gpu_memory_used_bytes",
                labels=labels,
                value=self.gpu_memory_used,
                metric_type="gauge",
                help_text="GPU memory used in bytes"
            ))
            
            samples.append(MetricSample(
                name="gpu_memory_total_bytes",
                labels=labels,
                value=self.gpu_memory_total,
                metric_type="gauge",
                help_text="Total GPU memory in bytes"
            ))
        
        return samples


class CacheMetricsCollector(MetricCollector):
    """Collects cache-related metrics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_bytes = 0
        self.capacity_bytes = 0
        self._lock = threading.Lock()
    
    def record_hit(self) -> None:
        with self._lock:
            self.hits += 1
    
    def record_miss(self) -> None:
        with self._lock:
            self.misses += 1
    
    def record_eviction(self) -> None:
        with self._lock:
            self.evictions += 1
    
    def update_size(self, size: int, capacity: int) -> None:
        with self._lock:
            self.size_bytes = size
            self.capacity_bytes = capacity
    
    def collect(self) -> List[MetricSample]:
        samples = []
        
        with self._lock:
            samples.append(MetricSample(
                name="cache_hits_total",
                labels={},
                value=float(self.hits),
                metric_type="counter",
                help_text="Total cache hits"
            ))
            
            samples.append(MetricSample(
                name="cache_misses_total",
                labels={},
                value=float(self.misses),
                metric_type="counter",
                help_text="Total cache misses"
            ))
            
            samples.append(MetricSample(
                name="cache_evictions_total",
                labels={},
                value=float(self.evictions),
                metric_type="counter",
                help_text="Total cache evictions"
            ))
            
            samples.append(MetricSample(
                name="cache_size_bytes",
                labels={},
                value=float(self.size_bytes),
                metric_type="gauge",
                help_text="Current cache size in bytes"
            ))
            
            samples.append(MetricSample(
                name="cache_capacity_bytes",
                labels={},
                value=float(self.capacity_bytes),
                metric_type="gauge",
                help_text="Cache capacity in bytes"
            ))
            
            # Hit rate
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            samples.append(MetricSample(
                name="cache_hit_rate",
                labels={},
                value=hit_rate,
                metric_type="gauge",
                help_text="Cache hit rate (0.0 to 1.0)"
            ))
        
        return samples


class PrometheusExporter:
    """
    HTTP server for Prometheus metric exposition.
    
    Exposes metrics at /metrics endpoint for Prometheus scraping.
    """
    
    def __init__(self, config: Optional[ExporterConfig] = None):
        self.config = config or ExporterConfig()
        self.collectors: List[MetricCollector] = []
        self.formatter = PrometheusFormatter()
        self._running = False
        self._start_time = time.time()
    
    def register_collector(self, collector: MetricCollector) -> None:
        """Register a metric collector."""
        self.collectors.append(collector)
        logger.info(f"Registered collector: {collector.__class__.__name__}")
    
    def collect_all(self) -> List[MetricSample]:
        """Collect metrics from all registered collectors."""
        all_samples = []
        
        # Add default process metrics
        if self.config.enable_default_metrics:
            all_samples.extend(self._collect_process_metrics())
        
        # Collect from all registered collectors
        for collector in self.collectors:
            try:
                samples = collector.collect()
                all_samples.extend(samples)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
        
        return all_samples
    
    def _collect_process_metrics(self) -> List[MetricSample]:
        """Collect default process metrics."""
        samples = []
        
        # Process uptime
        uptime = time.time() - self._start_time
        samples.append(MetricSample(
            name="process_uptime_seconds",
            labels={},
            value=uptime,
            metric_type="gauge",
            help_text="Process uptime in seconds"
        ))
        
        # Process start time
        samples.append(MetricSample(
            name="process_start_time_seconds",
            labels={},
            value=self._start_time,
            metric_type="gauge",
            help_text="Process start time as Unix timestamp"
        ))
        
        return samples
    
    def get_metrics(self) -> str:
        """Get formatted metrics string."""
        samples = self.collect_all()
        return self.formatter.format_metrics(samples, self.config.namespace)
    
    def get_health(self) -> Dict[str, Any]:
        """Get health check response."""
        return {
            "status": "healthy",
            "uptime_seconds": time.time() - self._start_time,
            "collectors": len(self.collectors)
        }
    
    def get_ready(self) -> Dict[str, Any]:
        """Get readiness check response."""
        return {
            "ready": True,
            "collectors_registered": len(self.collectors)
        }
    
    async def handle_request(self, path: str) -> Tuple[int, str, str]:
        """
        Handle HTTP request.
        
        Returns:
            Tuple of (status_code, content_type, body)
        """
        if path == self.config.path or path == "/metrics":
            return 200, "text/plain; charset=utf-8", self.get_metrics()
        elif path == "/health":
            return 200, "application/json", json.dumps(self.get_health())
        elif path == "/ready":
            return 200, "application/json", json.dumps(self.get_ready())
        else:
            return 404, "text/plain", "Not Found"


def create_exporter(
    node_id: str = "node-0",
    config: Optional[ExporterConfig] = None
) -> Tuple[PrometheusExporter, InferenceMetricsCollector, NodeMetricsCollector, CacheMetricsCollector]:
    """
    Create a fully configured Prometheus exporter with collectors.
    
    Returns:
        Tuple of (exporter, inference_collector, node_collector, cache_collector)
    """
    exporter = PrometheusExporter(config)
    
    inference = InferenceMetricsCollector()
    node = NodeMetricsCollector(node_id)
    cache = CacheMetricsCollector()
    
    exporter.register_collector(inference)
    exporter.register_collector(node)
    exporter.register_collector(cache)
    
    return exporter, inference, node, cache
