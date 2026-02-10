"""
Prometheus Metrics Collection for Distributed LLM Inference.

This module provides comprehensive metrics collection using prometheus_client,
with custom metrics for inference workloads, GPU utilization, and distributed
system health.

Classes:
    MetricsCollector: Main metrics collection orchestrator
    InferenceMetrics: Tokens/sec, latency, batch size metrics
    GPUMetrics: GPU memory, utilization, temperature tracking
    RequestMetrics: Request counts, errors, queue depth
    CacheMetrics: KV cache hit rates, evictions, memory usage
    DistributedMetrics: Node health, network latency, load balancing
"""

from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from contextlib import contextmanager
import logging

# Prometheus client types (compatible with prometheus_client library)
# Using abstractions to allow testing without actual prometheus_client dependency

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of Prometheus metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition for a Prometheus metric."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[Tuple[float, ...]] = None  # For histograms
    quantiles: Optional[Tuple[Tuple[float, float], ...]] = None  # For summaries
    
    def __post_init__(self):
        """Validate metric definition."""
        if self.metric_type == MetricType.HISTOGRAM and self.buckets is None:
            # Default latency buckets (in seconds)
            self.buckets = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        if self.metric_type == MetricType.SUMMARY and self.quantiles is None:
            # Default percentiles: 50th, 90th, 95th, 99th
            self.quantiles = ((0.5, 0.05), (0.9, 0.01), (0.95, 0.005), (0.99, 0.001))


@dataclass
class MetricValue:
    """A metric value with optional labels."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricRegistry:
    """
    Registry for all metrics in the system.
    
    Thread-safe storage and retrieval of metric definitions and values.
    Implements the Prometheus data model.
    """
    
    def __init__(self, namespace: str = "llm_inference"):
        """
        Initialize the metric registry.
        
        Args:
            namespace: Prefix for all metric names
        """
        self._namespace = namespace
        self._definitions: Dict[str, MetricDefinition] = {}
        self._counters: Dict[str, Dict[Tuple[str, ...], float]] = {}
        self._gauges: Dict[str, Dict[Tuple[str, ...], float]] = {}
        self._histograms: Dict[str, Dict[Tuple[str, ...], List[float]]] = {}
        self._summaries: Dict[str, Dict[Tuple[str, ...], List[float]]] = {}
        self._lock = threading.RLock()
    
    @property
    def namespace(self) -> str:
        """Get the metric namespace."""
        return self._namespace
    
    def _full_name(self, name: str) -> str:
        """Get full metric name with namespace."""
        return f"{self._namespace}_{name}"
    
    def register(self, definition: MetricDefinition) -> None:
        """
        Register a new metric definition.
        
        Args:
            definition: The metric definition to register
            
        Raises:
            ValueError: If metric already registered with different definition
        """
        full_name = self._full_name(definition.name)
        with self._lock:
            if full_name in self._definitions:
                existing = self._definitions[full_name]
                if existing.metric_type != definition.metric_type:
                    raise ValueError(
                        f"Metric {full_name} already registered as {existing.metric_type}, "
                        f"cannot register as {definition.metric_type}"
                    )
            self._definitions[full_name] = definition
            
            # Initialize storage based on type
            if definition.metric_type == MetricType.COUNTER:
                self._counters.setdefault(full_name, {})
            elif definition.metric_type == MetricType.GAUGE:
                self._gauges.setdefault(full_name, {})
            elif definition.metric_type == MetricType.HISTOGRAM:
                self._histograms.setdefault(full_name, {})
            elif definition.metric_type == MetricType.SUMMARY:
                self._summaries.setdefault(full_name, {})
    
    def _labels_key(self, labels: Dict[str, str], definition: MetricDefinition) -> Tuple[str, ...]:
        """Convert labels dict to hashable tuple key."""
        return tuple(labels.get(label, "") for label in definition.labels)
    
    def inc_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name (without namespace)
            value: Amount to increment (must be >= 0)
            labels: Optional label values
        """
        if value < 0:
            raise ValueError("Counter can only be incremented")
        
        full_name = self._full_name(name)
        labels = labels or {}
        
        with self._lock:
            definition = self._definitions.get(full_name)
            if definition is None:
                raise ValueError(f"Metric {full_name} not registered")
            
            key = self._labels_key(labels, definition)
            self._counters[full_name][key] = self._counters[full_name].get(key, 0.0) + value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Metric name (without namespace)
            value: Value to set
            labels: Optional label values
        """
        full_name = self._full_name(name)
        labels = labels or {}
        
        with self._lock:
            definition = self._definitions.get(full_name)
            if definition is None:
                raise ValueError(f"Metric {full_name} not registered")
            
            key = self._labels_key(labels, definition)
            self._gauges[full_name][key] = value
    
    def inc_gauge(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a gauge metric."""
        full_name = self._full_name(name)
        labels = labels or {}
        
        with self._lock:
            definition = self._definitions.get(full_name)
            if definition is None:
                raise ValueError(f"Metric {full_name} not registered")
            
            key = self._labels_key(labels, definition)
            current = self._gauges[full_name].get(key, 0.0)
            self._gauges[full_name][key] = current + value
    
    def dec_gauge(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement a gauge metric."""
        self.inc_gauge(name, -value, labels)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record an observation in a histogram.
        
        Args:
            name: Metric name (without namespace)
            value: Observed value
            labels: Optional label values
        """
        full_name = self._full_name(name)
        labels = labels or {}
        
        with self._lock:
            definition = self._definitions.get(full_name)
            if definition is None:
                raise ValueError(f"Metric {full_name} not registered")
            
            key = self._labels_key(labels, definition)
            if key not in self._histograms[full_name]:
                self._histograms[full_name][key] = []
            self._histograms[full_name][key].append(value)
    
    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record an observation in a summary.
        
        Args:
            name: Metric name (without namespace)
            value: Observed value
            labels: Optional label values
        """
        full_name = self._full_name(name)
        labels = labels or {}
        
        with self._lock:
            definition = self._definitions.get(full_name)
            if definition is None:
                raise ValueError(f"Metric {full_name} not registered")
            
            key = self._labels_key(labels, definition)
            if key not in self._summaries[full_name]:
                self._summaries[full_name][key] = []
            
            # Keep only recent observations for summary (sliding window)
            observations = self._summaries[full_name][key]
            observations.append(value)
            # Keep last 1000 observations
            if len(observations) > 1000:
                self._summaries[full_name][key] = observations[-1000:]
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        full_name = self._full_name(name)
        labels = labels or {}
        
        with self._lock:
            definition = self._definitions.get(full_name)
            if definition is None:
                return 0.0
            key = self._labels_key(labels, definition)
            return self._counters.get(full_name, {}).get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        full_name = self._full_name(name)
        labels = labels or {}
        
        with self._lock:
            definition = self._definitions.get(full_name)
            if definition is None:
                return 0.0
            key = self._labels_key(labels, definition)
            return self._gauges.get(full_name, {}).get(key, 0.0)
    
    def get_histogram_buckets(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[float, int]:
        """
        Get histogram bucket counts.
        
        Returns:
            Dict mapping bucket upper bound to count of observations <= that bound
        """
        full_name = self._full_name(name)
        labels = labels or {}
        
        with self._lock:
            definition = self._definitions.get(full_name)
            if definition is None:
                return {}
            
            key = self._labels_key(labels, definition)
            observations = self._histograms.get(full_name, {}).get(key, [])
            
            if not definition.buckets:
                return {}
            
            buckets = {}
            for bucket in definition.buckets:
                buckets[bucket] = sum(1 for v in observations if v <= bucket)
            buckets[float('inf')] = len(observations)
            
            return buckets
    
    def get_summary_quantiles(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[float, float]:
        """
        Get summary quantile values.
        
        Returns:
            Dict mapping quantile (e.g., 0.99) to value at that percentile
        """
        full_name = self._full_name(name)
        labels = labels or {}
        
        with self._lock:
            definition = self._definitions.get(full_name)
            if definition is None:
                return {}
            
            key = self._labels_key(labels, definition)
            observations = self._summaries.get(full_name, {}).get(key, [])
            
            if not observations or not definition.quantiles:
                return {}
            
            sorted_obs = sorted(observations)
            n = len(sorted_obs)
            
            quantiles = {}
            for quantile, _ in definition.quantiles:
                idx = int(quantile * n)
                idx = min(idx, n - 1)
                quantiles[quantile] = sorted_obs[idx]
            
            return quantiles
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            result = {}
            
            for name in self._definitions:
                definition = self._definitions[name]
                
                if definition.metric_type == MetricType.COUNTER:
                    result[name] = dict(self._counters.get(name, {}))
                elif definition.metric_type == MetricType.GAUGE:
                    result[name] = dict(self._gauges.get(name, {}))
                elif definition.metric_type == MetricType.HISTOGRAM:
                    result[name] = {
                        str(k): self.get_histogram_buckets(
                            definition.name, 
                            dict(zip(definition.labels, k))
                        )
                        for k in self._histograms.get(name, {})
                    }
                elif definition.metric_type == MetricType.SUMMARY:
                    result[name] = {
                        str(k): self.get_summary_quantiles(
                            definition.name,
                            dict(zip(definition.labels, k))
                        )
                        for k in self._summaries.get(name, {})
                    }
            
            return result
    
    def clear(self) -> None:
        """Clear all metric values (but keep definitions)."""
        with self._lock:
            for name in self._counters:
                self._counters[name].clear()
            for name in self._gauges:
                self._gauges[name].clear()
            for name in self._histograms:
                self._histograms[name].clear()
            for name in self._summaries:
                self._summaries[name].clear()


class BaseMetrics(ABC):
    """Base class for metric collectors."""
    
    def __init__(self, registry: Optional[MetricRegistry] = None):
        """
        Initialize base metrics.
        
        Args:
            registry: The metric registry to use
        """
        self._registry = registry if registry is not None else MetricRegistry()
        self._register_metrics()
    
    @property
    def registry(self) -> MetricRegistry:
        """Get the metric registry."""
        return self._registry
    
    @abstractmethod
    def _register_metrics(self) -> None:
        """Register all metrics for this collector. Must be implemented by subclasses."""
        pass


class InferenceMetrics(BaseMetrics):
    """
    Metrics for LLM inference operations.
    
    Tracks:
    - Tokens generated per second
    - Inference latency (time to first token, total generation time)
    - Batch sizes
    - Generation throughput
    """
    
    def _register_metrics(self) -> None:
        """Register inference-specific metrics."""
        # Counters
        self._registry.register(MetricDefinition(
            name="tokens_generated_total",
            description="Total number of tokens generated",
            metric_type=MetricType.COUNTER,
            labels=["model", "request_type"]
        ))
        
        self._registry.register(MetricDefinition(
            name="inference_requests_total",
            description="Total number of inference requests",
            metric_type=MetricType.COUNTER,
            labels=["model", "status"]
        ))
        
        # Gauges
        self._registry.register(MetricDefinition(
            name="tokens_per_second",
            description="Current tokens generated per second",
            metric_type=MetricType.GAUGE,
            labels=["model"]
        ))
        
        self._registry.register(MetricDefinition(
            name="active_generations",
            description="Number of currently active generation requests",
            metric_type=MetricType.GAUGE,
            labels=["model"]
        ))
        
        self._registry.register(MetricDefinition(
            name="batch_size_current",
            description="Current dynamic batch size",
            metric_type=MetricType.GAUGE,
            labels=["model"]
        ))
        
        # Histograms
        self._registry.register(MetricDefinition(
            name="time_to_first_token_seconds",
            description="Time from request to first token generated",
            metric_type=MetricType.HISTOGRAM,
            labels=["model"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        ))
        
        self._registry.register(MetricDefinition(
            name="generation_duration_seconds",
            description="Total time for complete generation",
            metric_type=MetricType.HISTOGRAM,
            labels=["model"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        ))
        
        self._registry.register(MetricDefinition(
            name="tokens_per_request",
            description="Distribution of tokens generated per request",
            metric_type=MetricType.HISTOGRAM,
            labels=["model"],
            buckets=(1, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000)
        ))
        
        # Summaries
        self._registry.register(MetricDefinition(
            name="inference_latency_summary",
            description="Summary of inference latencies",
            metric_type=MetricType.SUMMARY,
            labels=["model", "quantile"]
        ))
    
    def record_tokens_generated(self, count: int, model: str, request_type: str = "generate") -> None:
        """Record tokens generated."""
        self._registry.inc_counter(
            "tokens_generated_total",
            count,
            {"model": model, "request_type": request_type}
        )
    
    def record_request(self, model: str, status: str = "success") -> None:
        """Record an inference request."""
        self._registry.inc_counter(
            "inference_requests_total",
            1.0,
            {"model": model, "status": status}
        )
    
    def set_tokens_per_second(self, tps: float, model: str) -> None:
        """Set current tokens per second rate."""
        self._registry.set_gauge("tokens_per_second", tps, {"model": model})
    
    def set_active_generations(self, count: int, model: str) -> None:
        """Set number of active generations."""
        self._registry.set_gauge("active_generations", float(count), {"model": model})
    
    def set_batch_size(self, size: int, model: str) -> None:
        """Set current batch size."""
        self._registry.set_gauge("batch_size_current", float(size), {"model": model})
    
    def record_time_to_first_token(self, seconds: float, model: str) -> None:
        """Record time to first token."""
        self._registry.observe_histogram(
            "time_to_first_token_seconds",
            seconds,
            {"model": model}
        )
    
    def record_generation_duration(self, seconds: float, model: str) -> None:
        """Record total generation duration."""
        self._registry.observe_histogram(
            "generation_duration_seconds",
            seconds,
            {"model": model}
        )
    
    def record_tokens_per_request(self, count: int, model: str) -> None:
        """Record tokens generated in a single request."""
        self._registry.observe_histogram(
            "tokens_per_request",
            float(count),
            {"model": model}
        )
    
    def record_latency(self, seconds: float, model: str) -> None:
        """Record inference latency for summary."""
        self._registry.observe_summary(
            "inference_latency_summary",
            seconds,
            {"model": model, "quantile": "all"}
        )
    
    @contextmanager
    def track_generation(self, model: str):
        """
        Context manager to track a generation request.
        
        Automatically records:
        - Active generation count
        - Generation duration
        - Request completion
        
        Example:
            with metrics.track_generation("llama-7b") as tracker:
                tracker.record_first_token()  # When first token generated
                # ... generate tokens ...
                tracker.set_tokens(150)  # Total tokens at end
        """
        class GenerationTracker:
            def __init__(self, metrics: InferenceMetrics, model: str):
                self._metrics = metrics
                self._model = model
                self._start_time = time.time()
                self._first_token_time: Optional[float] = None
                self._tokens = 0
            
            def record_first_token(self) -> None:
                if self._first_token_time is None:
                    self._first_token_time = time.time()
                    ttft = self._first_token_time - self._start_time
                    self._metrics.record_time_to_first_token(ttft, self._model)
            
            def set_tokens(self, count: int) -> None:
                self._tokens = count
        
        tracker = GenerationTracker(self, model)
        self._registry.inc_gauge("active_generations", 1.0, {"model": model})
        
        try:
            yield tracker
            status = "success"
        except Exception:
            status = "error"
            raise
        finally:
            self._registry.dec_gauge("active_generations", 1.0, {"model": model})
            
            duration = time.time() - tracker._start_time
            self.record_generation_duration(duration, model)
            self.record_request(model, status)
            
            if tracker._tokens > 0:
                self.record_tokens_per_request(tracker._tokens, model)
                self.record_tokens_generated(tracker._tokens, model)


class GPUMetrics(BaseMetrics):
    """
    Metrics for GPU resource tracking.
    
    Tracks:
    - GPU memory usage (allocated, reserved, free)
    - GPU utilization percentage
    - GPU temperature
    - Multi-GPU load balancing metrics
    """
    
    def _register_metrics(self) -> None:
        """Register GPU-specific metrics."""
        # Memory gauges
        self._registry.register(MetricDefinition(
            name="gpu_memory_allocated_bytes",
            description="GPU memory currently allocated",
            metric_type=MetricType.GAUGE,
            labels=["gpu_id", "node"]
        ))
        
        self._registry.register(MetricDefinition(
            name="gpu_memory_reserved_bytes",
            description="GPU memory currently reserved",
            metric_type=MetricType.GAUGE,
            labels=["gpu_id", "node"]
        ))
        
        self._registry.register(MetricDefinition(
            name="gpu_memory_total_bytes",
            description="Total GPU memory available",
            metric_type=MetricType.GAUGE,
            labels=["gpu_id", "node"]
        ))
        
        self._registry.register(MetricDefinition(
            name="gpu_memory_utilization_ratio",
            description="GPU memory utilization (0-1)",
            metric_type=MetricType.GAUGE,
            labels=["gpu_id", "node"]
        ))
        
        # Utilization
        self._registry.register(MetricDefinition(
            name="gpu_utilization_percent",
            description="GPU compute utilization percentage",
            metric_type=MetricType.GAUGE,
            labels=["gpu_id", "node"]
        ))
        
        self._registry.register(MetricDefinition(
            name="gpu_temperature_celsius",
            description="GPU temperature in Celsius",
            metric_type=MetricType.GAUGE,
            labels=["gpu_id", "node"]
        ))
        
        self._registry.register(MetricDefinition(
            name="gpu_power_watts",
            description="GPU power consumption in watts",
            metric_type=MetricType.GAUGE,
            labels=["gpu_id", "node"]
        ))
        
        # Histograms for utilization patterns
        self._registry.register(MetricDefinition(
            name="gpu_utilization_histogram",
            description="Distribution of GPU utilization",
            metric_type=MetricType.HISTOGRAM,
            labels=["gpu_id", "node"],
            buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        ))
    
    def set_memory_allocated(self, bytes_: int, gpu_id: str, node: str = "local") -> None:
        """Set GPU memory allocated."""
        self._registry.set_gauge(
            "gpu_memory_allocated_bytes",
            float(bytes_),
            {"gpu_id": gpu_id, "node": node}
        )
    
    def set_memory_reserved(self, bytes_: int, gpu_id: str, node: str = "local") -> None:
        """Set GPU memory reserved."""
        self._registry.set_gauge(
            "gpu_memory_reserved_bytes",
            float(bytes_),
            {"gpu_id": gpu_id, "node": node}
        )
    
    def set_memory_total(self, bytes_: int, gpu_id: str, node: str = "local") -> None:
        """Set total GPU memory."""
        self._registry.set_gauge(
            "gpu_memory_total_bytes",
            float(bytes_),
            {"gpu_id": gpu_id, "node": node}
        )
    
    def set_memory_utilization(self, ratio: float, gpu_id: str, node: str = "local") -> None:
        """Set GPU memory utilization ratio (0-1)."""
        self._registry.set_gauge(
            "gpu_memory_utilization_ratio",
            ratio,
            {"gpu_id": gpu_id, "node": node}
        )
    
    def set_utilization(self, percent: float, gpu_id: str, node: str = "local") -> None:
        """Set GPU compute utilization percentage."""
        self._registry.set_gauge(
            "gpu_utilization_percent",
            percent,
            {"gpu_id": gpu_id, "node": node}
        )
        self._registry.observe_histogram(
            "gpu_utilization_histogram",
            percent,
            {"gpu_id": gpu_id, "node": node}
        )
    
    def set_temperature(self, celsius: float, gpu_id: str, node: str = "local") -> None:
        """Set GPU temperature."""
        self._registry.set_gauge(
            "gpu_temperature_celsius",
            celsius,
            {"gpu_id": gpu_id, "node": node}
        )
    
    def set_power(self, watts: float, gpu_id: str, node: str = "local") -> None:
        """Set GPU power consumption."""
        self._registry.set_gauge(
            "gpu_power_watts",
            watts,
            {"gpu_id": gpu_id, "node": node}
        )
    
    def update_gpu_stats(
        self,
        gpu_id: str,
        memory_allocated: int,
        memory_total: int,
        utilization: float,
        temperature: Optional[float] = None,
        power: Optional[float] = None,
        node: str = "local"
    ) -> None:
        """Update all GPU stats at once."""
        self.set_memory_allocated(memory_allocated, gpu_id, node)
        self.set_memory_total(memory_total, gpu_id, node)
        self.set_memory_utilization(memory_allocated / memory_total if memory_total > 0 else 0, gpu_id, node)
        self.set_utilization(utilization, gpu_id, node)
        
        if temperature is not None:
            self.set_temperature(temperature, gpu_id, node)
        if power is not None:
            self.set_power(power, gpu_id, node)


class RequestMetrics(BaseMetrics):
    """
    Metrics for request handling.
    
    Tracks:
    - Request counts by type, status, and endpoint
    - Request queue depth
    - Error rates
    - Rate limiting metrics
    """
    
    def _register_metrics(self) -> None:
        """Register request-specific metrics."""
        # Counters
        self._registry.register(MetricDefinition(
            name="requests_total",
            description="Total number of requests received",
            metric_type=MetricType.COUNTER,
            labels=["endpoint", "method", "status_code"]
        ))
        
        self._registry.register(MetricDefinition(
            name="request_errors_total",
            description="Total number of request errors",
            metric_type=MetricType.COUNTER,
            labels=["endpoint", "error_type"]
        ))
        
        self._registry.register(MetricDefinition(
            name="rate_limited_requests_total",
            description="Total number of rate-limited requests",
            metric_type=MetricType.COUNTER,
            labels=["endpoint", "client_id"]
        ))
        
        # Gauges
        self._registry.register(MetricDefinition(
            name="request_queue_depth",
            description="Current number of requests in queue",
            metric_type=MetricType.GAUGE,
            labels=["priority"]
        ))
        
        self._registry.register(MetricDefinition(
            name="active_connections",
            description="Number of active client connections",
            metric_type=MetricType.GAUGE,
            labels=["protocol"]
        ))
        
        # Histograms
        self._registry.register(MetricDefinition(
            name="request_duration_seconds",
            description="Request processing duration",
            metric_type=MetricType.HISTOGRAM,
            labels=["endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        ))
        
        self._registry.register(MetricDefinition(
            name="request_size_bytes",
            description="Request payload size in bytes",
            metric_type=MetricType.HISTOGRAM,
            labels=["endpoint"],
            buckets=(100, 1000, 10000, 100000, 1000000)
        ))
        
        self._registry.register(MetricDefinition(
            name="response_size_bytes",
            description="Response payload size in bytes",
            metric_type=MetricType.HISTOGRAM,
            labels=["endpoint"],
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
        ))
    
    def record_request(
        self,
        endpoint: str,
        method: str = "POST",
        status_code: int = 200,
        duration: Optional[float] = None,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None
    ) -> None:
        """Record a completed request."""
        self._registry.inc_counter(
            "requests_total",
            1.0,
            {"endpoint": endpoint, "method": method, "status_code": str(status_code)}
        )
        
        if duration is not None:
            self._registry.observe_histogram(
                "request_duration_seconds",
                duration,
                {"endpoint": endpoint}
            )
        
        if request_size is not None:
            self._registry.observe_histogram(
                "request_size_bytes",
                float(request_size),
                {"endpoint": endpoint}
            )
        
        if response_size is not None:
            self._registry.observe_histogram(
                "response_size_bytes",
                float(response_size),
                {"endpoint": endpoint}
            )
    
    def record_error(self, endpoint: str, error_type: str) -> None:
        """Record a request error."""
        self._registry.inc_counter(
            "request_errors_total",
            1.0,
            {"endpoint": endpoint, "error_type": error_type}
        )
    
    def record_rate_limited(self, endpoint: str, client_id: str = "anonymous") -> None:
        """Record a rate-limited request."""
        self._registry.inc_counter(
            "rate_limited_requests_total",
            1.0,
            {"endpoint": endpoint, "client_id": client_id}
        )
    
    def set_queue_depth(self, depth: int, priority: str = "normal") -> None:
        """Set current request queue depth."""
        self._registry.set_gauge(
            "request_queue_depth",
            float(depth),
            {"priority": priority}
        )
    
    def set_active_connections(self, count: int, protocol: str = "http") -> None:
        """Set number of active connections."""
        self._registry.set_gauge(
            "active_connections",
            float(count),
            {"protocol": protocol}
        )
    
    @contextmanager
    def track_request(self, endpoint: str, method: str = "POST"):
        """
        Context manager to track a request.
        
        Example:
            with request_metrics.track_request("/v1/generate") as tracker:
                tracker.set_request_size(1024)
                result = process_request(...)
                tracker.set_response_size(len(result))
        """
        class RequestTracker:
            def __init__(self):
                self.request_size: Optional[int] = None
                self.response_size: Optional[int] = None
            
            def set_request_size(self, size: int) -> None:
                self.request_size = size
            
            def set_response_size(self, size: int) -> None:
                self.response_size = size
        
        tracker = RequestTracker()
        start_time = time.time()
        status_code = 200
        
        try:
            yield tracker
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time
            self.record_request(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                duration=duration,
                request_size=tracker.request_size,
                response_size=tracker.response_size
            )


class CacheMetrics(BaseMetrics):
    """
    Metrics for KV cache operations.
    
    Tracks:
    - Cache hit/miss rates
    - Cache memory usage
    - Eviction counts
    - Cache entry age
    """
    
    def _register_metrics(self) -> None:
        """Register cache-specific metrics."""
        # Counters
        self._registry.register(MetricDefinition(
            name="cache_hits_total",
            description="Total number of cache hits",
            metric_type=MetricType.COUNTER,
            labels=["cache_type"]
        ))
        
        self._registry.register(MetricDefinition(
            name="cache_misses_total",
            description="Total number of cache misses",
            metric_type=MetricType.COUNTER,
            labels=["cache_type"]
        ))
        
        self._registry.register(MetricDefinition(
            name="cache_evictions_total",
            description="Total number of cache evictions",
            metric_type=MetricType.COUNTER,
            labels=["cache_type", "reason"]
        ))
        
        # Gauges
        self._registry.register(MetricDefinition(
            name="cache_entries",
            description="Current number of cache entries",
            metric_type=MetricType.GAUGE,
            labels=["cache_type"]
        ))
        
        self._registry.register(MetricDefinition(
            name="cache_memory_bytes",
            description="Current cache memory usage in bytes",
            metric_type=MetricType.GAUGE,
            labels=["cache_type"]
        ))
        
        self._registry.register(MetricDefinition(
            name="cache_hit_ratio",
            description="Cache hit ratio (0-1)",
            metric_type=MetricType.GAUGE,
            labels=["cache_type"]
        ))
        
        self._registry.register(MetricDefinition(
            name="cache_capacity_ratio",
            description="Cache capacity utilization (0-1)",
            metric_type=MetricType.GAUGE,
            labels=["cache_type"]
        ))
        
        # Histograms
        self._registry.register(MetricDefinition(
            name="cache_entry_age_seconds",
            description="Age of cache entries when accessed",
            metric_type=MetricType.HISTOGRAM,
            labels=["cache_type"],
            buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600)
        ))
        
        self._registry.register(MetricDefinition(
            name="cache_lookup_duration_seconds",
            description="Time to perform cache lookup",
            metric_type=MetricType.HISTOGRAM,
            labels=["cache_type"],
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
        ))
    
    def record_hit(self, cache_type: str = "kv_cache") -> None:
        """Record a cache hit."""
        self._registry.inc_counter("cache_hits_total", 1.0, {"cache_type": cache_type})
    
    def record_miss(self, cache_type: str = "kv_cache") -> None:
        """Record a cache miss."""
        self._registry.inc_counter("cache_misses_total", 1.0, {"cache_type": cache_type})
    
    def record_eviction(self, cache_type: str = "kv_cache", reason: str = "capacity") -> None:
        """Record a cache eviction."""
        self._registry.inc_counter(
            "cache_evictions_total",
            1.0,
            {"cache_type": cache_type, "reason": reason}
        )
    
    def set_entries(self, count: int, cache_type: str = "kv_cache") -> None:
        """Set current number of cache entries."""
        self._registry.set_gauge("cache_entries", float(count), {"cache_type": cache_type})
    
    def set_memory_bytes(self, bytes_: int, cache_type: str = "kv_cache") -> None:
        """Set current cache memory usage."""
        self._registry.set_gauge("cache_memory_bytes", float(bytes_), {"cache_type": cache_type})
    
    def set_hit_ratio(self, ratio: float, cache_type: str = "kv_cache") -> None:
        """Set cache hit ratio (0-1)."""
        self._registry.set_gauge("cache_hit_ratio", ratio, {"cache_type": cache_type})
    
    def set_capacity_ratio(self, ratio: float, cache_type: str = "kv_cache") -> None:
        """Set cache capacity utilization (0-1)."""
        self._registry.set_gauge("cache_capacity_ratio", ratio, {"cache_type": cache_type})
    
    def record_entry_age(self, age_seconds: float, cache_type: str = "kv_cache") -> None:
        """Record the age of an accessed cache entry."""
        self._registry.observe_histogram(
            "cache_entry_age_seconds",
            age_seconds,
            {"cache_type": cache_type}
        )
    
    def record_lookup_duration(self, seconds: float, cache_type: str = "kv_cache") -> None:
        """Record cache lookup duration."""
        self._registry.observe_histogram(
            "cache_lookup_duration_seconds",
            seconds,
            {"cache_type": cache_type}
        )
    
    def update_hit_ratio(self, cache_type: str = "kv_cache") -> None:
        """Compute and update hit ratio from counters."""
        hits = self._registry.get_counter("cache_hits_total", {"cache_type": cache_type})
        misses = self._registry.get_counter("cache_misses_total", {"cache_type": cache_type})
        total = hits + misses
        
        if total > 0:
            self.set_hit_ratio(hits / total, cache_type)


class DistributedMetrics(BaseMetrics):
    """
    Metrics for distributed system operations.
    
    Tracks:
    - Node health and status
    - Network latency between nodes
    - Load balancing metrics
    - Synchronization metrics
    """
    
    def _register_metrics(self) -> None:
        """Register distributed-specific metrics."""
        # Gauges
        self._registry.register(MetricDefinition(
            name="cluster_nodes_total",
            description="Total number of nodes in cluster",
            metric_type=MetricType.GAUGE,
            labels=["status"]
        ))
        
        self._registry.register(MetricDefinition(
            name="node_health_score",
            description="Node health score (0-1)",
            metric_type=MetricType.GAUGE,
            labels=["node_id"]
        ))
        
        self._registry.register(MetricDefinition(
            name="node_load",
            description="Current load on node (0-1)",
            metric_type=MetricType.GAUGE,
            labels=["node_id"]
        ))
        
        self._registry.register(MetricDefinition(
            name="cluster_load_imbalance",
            description="Standard deviation of node loads",
            metric_type=MetricType.GAUGE,
            labels=[]
        ))
        
        # Counters
        self._registry.register(MetricDefinition(
            name="node_failures_total",
            description="Total node failures detected",
            metric_type=MetricType.COUNTER,
            labels=["node_id", "failure_type"]
        ))
        
        self._registry.register(MetricDefinition(
            name="requests_routed_total",
            description="Total requests routed to nodes",
            metric_type=MetricType.COUNTER,
            labels=["source_node", "target_node"]
        ))
        
        self._registry.register(MetricDefinition(
            name="sync_operations_total",
            description="Total synchronization operations",
            metric_type=MetricType.COUNTER,
            labels=["operation_type", "status"]
        ))
        
        # Histograms
        self._registry.register(MetricDefinition(
            name="node_latency_seconds",
            description="Network latency to node",
            metric_type=MetricType.HISTOGRAM,
            labels=["source_node", "target_node"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        ))
        
        self._registry.register(MetricDefinition(
            name="sync_duration_seconds",
            description="Duration of sync operations",
            metric_type=MetricType.HISTOGRAM,
            labels=["operation_type"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)
        ))
    
    def set_cluster_nodes(self, count: int, status: str = "healthy") -> None:
        """Set total nodes in cluster by status."""
        self._registry.set_gauge("cluster_nodes_total", float(count), {"status": status})
    
    def set_node_health(self, node_id: str, score: float) -> None:
        """Set node health score (0-1)."""
        self._registry.set_gauge("node_health_score", score, {"node_id": node_id})
    
    def set_node_load(self, node_id: str, load: float) -> None:
        """Set current node load (0-1)."""
        self._registry.set_gauge("node_load", load, {"node_id": node_id})
    
    def set_load_imbalance(self, std_dev: float) -> None:
        """Set cluster load imbalance (std dev of node loads)."""
        self._registry.set_gauge("cluster_load_imbalance", std_dev, {})
    
    def record_node_failure(self, node_id: str, failure_type: str) -> None:
        """Record a node failure."""
        self._registry.inc_counter(
            "node_failures_total",
            1.0,
            {"node_id": node_id, "failure_type": failure_type}
        )
    
    def record_request_routed(self, source: str, target: str) -> None:
        """Record a request being routed."""
        self._registry.inc_counter(
            "requests_routed_total",
            1.0,
            {"source_node": source, "target_node": target}
        )
    
    def record_sync_operation(self, operation_type: str, status: str, duration: float) -> None:
        """Record a synchronization operation."""
        self._registry.inc_counter(
            "sync_operations_total",
            1.0,
            {"operation_type": operation_type, "status": status}
        )
        self._registry.observe_histogram(
            "sync_duration_seconds",
            duration,
            {"operation_type": operation_type}
        )
    
    def record_node_latency(self, source: str, target: str, latency: float) -> None:
        """Record network latency between nodes."""
        self._registry.observe_histogram(
            "node_latency_seconds",
            latency,
            {"source_node": source, "target_node": target}
        )


class MetricsCollector:
    """
    Main orchestrator for all metrics collection.
    
    Provides a unified interface for accessing all metric types
    and handles initialization, collection scheduling, and export.
    
    Example:
        collector = MetricsCollector(namespace="my_app")
        
        # Access specific metrics
        collector.inference.record_tokens_generated(100, "llama-7b")
        collector.gpu.set_utilization(75.5, "gpu:0")
        collector.cache.record_hit()
        
        # Export all metrics
        metrics_data = collector.collect()
    """
    
    def __init__(self, namespace: str = "llm_inference"):
        """
        Initialize the metrics collector.
        
        Args:
            namespace: Prefix for all metric names
        """
        self._registry = MetricRegistry(namespace)
        
        # Initialize all metric collectors
        self._inference = InferenceMetrics(self._registry)
        self._gpu = GPUMetrics(self._registry)
        self._request = RequestMetrics(self._registry)
        self._cache = CacheMetrics(self._registry)
        self._distributed = DistributedMetrics(self._registry)
        
        # Background collection state
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_collection = threading.Event()
        self._collection_callbacks: List[Callable[[], None]] = []
        
        logger.info(f"MetricsCollector initialized with namespace: {namespace}")
    
    @property
    def registry(self) -> MetricRegistry:
        """Get the underlying metric registry."""
        return self._registry
    
    @property
    def inference(self) -> InferenceMetrics:
        """Get inference metrics collector."""
        return self._inference
    
    @property
    def gpu(self) -> GPUMetrics:
        """Get GPU metrics collector."""
        return self._gpu
    
    @property
    def request(self) -> RequestMetrics:
        """Get request metrics collector."""
        return self._request
    
    @property
    def cache(self) -> CacheMetrics:
        """Get cache metrics collector."""
        return self._cache
    
    @property
    def distributed(self) -> DistributedMetrics:
        """Get distributed metrics collector."""
        return self._distributed
    
    def register_collection_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to run during each collection cycle.
        
        Args:
            callback: Function to call during collection
        """
        self._collection_callbacks.append(callback)
    
    def start_background_collection(self, interval_seconds: float = 15.0) -> None:
        """
        Start background metric collection.
        
        Args:
            interval_seconds: Collection interval in seconds
        """
        if self._collection_thread is not None and self._collection_thread.is_alive():
            logger.warning("Background collection already running")
            return
        
        self._stop_collection.clear()
        
        def collection_loop():
            while not self._stop_collection.is_set():
                try:
                    for callback in self._collection_callbacks:
                        callback()
                except Exception as e:
                    logger.error(f"Error in collection callback: {e}")
                
                self._stop_collection.wait(interval_seconds)
        
        self._collection_thread = threading.Thread(
            target=collection_loop,
            daemon=True,
            name="metrics-collector"
        )
        self._collection_thread.start()
        logger.info(f"Started background collection with {interval_seconds}s interval")
    
    def stop_background_collection(self) -> None:
        """Stop background metric collection."""
        self._stop_collection.set()
        if self._collection_thread is not None:
            self._collection_thread.join(timeout=5.0)
            self._collection_thread = None
        logger.info("Stopped background collection")
    
    def collect(self) -> Dict[str, Any]:
        """
        Collect all current metric values.
        
        Returns:
            Dictionary of all metric values
        """
        return self._registry.get_all_metrics()
    
    def reset(self) -> None:
        """Reset all metric values."""
        self._registry.clear()
        logger.info("All metrics reset")
