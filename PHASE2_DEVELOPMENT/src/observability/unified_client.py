"""
Unified Observability Client for Distributed Inference.

Provides a single entry point for:
- Prometheus metrics
- Jaeger distributed tracing  
- Structured logging
- Health checks

Sprint 3.5 - Observability Stack
Created: 2026-01-06
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from enum import Enum

from src.monitoring.prometheus_exporter import (
    PrometheusExporter, ExporterConfig,
    InferenceMetricsCollector, NodeMetricsCollector, CacheMetricsCollector,
    MetricSample, create_exporter
)
from src.tracing.jaeger_exporter import (
    DistributedTracer, JaegerConfig, SpanData,
    create_jaeger_tracer, create_test_tracer
)

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ObservabilityConfig:
    """Configuration for unified observability."""
    # Service identification
    service_name: str = "llm-inference"
    node_id: str = "node-0"
    environment: str = "development"
    version: str = "1.0.0"
    
    # Prometheus settings
    metrics_enabled: bool = True
    metrics_port: int = 9100
    metrics_path: str = "/metrics"
    
    # Jaeger settings
    tracing_enabled: bool = True
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    jaeger_use_http: bool = False
    jaeger_collector_endpoint: Optional[str] = None
    
    # Logging settings
    logging_enabled: bool = True
    log_level: str = "INFO"
    structured_logging: bool = True
    
    # Health check settings
    health_check_interval: float = 30.0


@dataclass
class RequestContext:
    """Context for tracking a request across the system."""
    request_id: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    user_id: Optional[str] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "user_id": self.user_id,
            "model": self.model,
            **self.metadata
        }


class StructuredLogger:
    """
    Structured logging with context propagation.
    
    Outputs JSON-formatted logs with trace context for correlation.
    """
    
    def __init__(self, name: str, config: ObservabilityConfig):
        self.name = name
        self.config = config
        self._context_var = threading.local()
        
        # Configure Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, config.log_level))
    
    def set_context(self, context: RequestContext) -> None:
        """Set the current request context."""
        self._context_var.context = context
    
    def clear_context(self) -> None:
        """Clear the current request context."""
        self._context_var.context = None
    
    def get_context(self) -> Optional[RequestContext]:
        """Get the current request context."""
        return getattr(self._context_var, 'context', None)
    
    def _format_message(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format a log message with context."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "service": self.config.service_name,
            "node_id": self.config.node_id,
            "environment": self.config.environment,
            "message": message
        }
        
        # Add request context if available
        context = self.get_context()
        if context:
            log_entry["trace_id"] = context.trace_id
            log_entry["span_id"] = context.span_id
            log_entry["request_id"] = context.request_id
        
        # Add extra fields
        if extra:
            log_entry["extra"] = extra
        
        return log_entry
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        entry = self._format_message("debug", message, kwargs or None)
        self._logger.debug(str(entry))
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        entry = self._format_message("info", message, kwargs or None)
        self._logger.info(str(entry))
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        entry = self._format_message("warning", message, kwargs or None)
        self._logger.warning(str(entry))
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
        entry = self._format_message("error", message, kwargs or None)
        self._logger.error(str(entry))
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        entry = self._format_message("critical", message, kwargs or None)
        self._logger.critical(str(entry))


class HealthChecker:
    """
    System health checker with component monitoring.
    
    Tracks health of various system components and provides
    aggregated health status.
    """
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self._components: Dict[str, Callable[[], bool]] = {}
        self._last_check: Dict[str, bool] = {}
        self._check_times: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def register_component(
        self,
        name: str,
        check_fn: Callable[[], bool]
    ) -> None:
        """
        Register a component health check.
        
        Args:
            name: Component name
            check_fn: Function that returns True if healthy
        """
        with self._lock:
            self._components[name] = check_fn
            self._last_check[name] = True
            self._check_times[name] = 0.0
    
    def check_component(self, name: str) -> bool:
        """Check health of a specific component."""
        with self._lock:
            if name not in self._components:
                return False
            
            try:
                check_fn = self._components[name]
                result = check_fn()
                self._last_check[name] = result
                self._check_times[name] = time.time()
                return result
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                self._last_check[name] = False
                return False
    
    def check_all(self) -> Dict[str, Any]:
        """Check health of all components."""
        results = {}
        all_healthy = True
        
        with self._lock:
            for name in self._components:
                healthy = self.check_component(name)
                results[name] = {
                    "healthy": healthy,
                    "last_check": self._check_times.get(name, 0.0)
                }
                if not healthy:
                    all_healthy = False
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "components": results,
            "timestamp": time.time()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get cached health status without running checks."""
        with self._lock:
            all_healthy = all(self._last_check.values()) if self._last_check else True
            return {
                "status": "healthy" if all_healthy else "unhealthy",
                "components": {
                    name: {"healthy": healthy}
                    for name, healthy in self._last_check.items()
                }
            }


class ObservabilityClient:
    """
    Unified observability client.
    
    Single entry point for all observability concerns:
    - Metrics (Prometheus)
    - Tracing (Jaeger)
    - Logging (Structured)
    - Health (Component checks)
    """
    
    def __init__(self, config: Optional[ObservabilityConfig] = None):
        self.config = config or ObservabilityConfig()
        
        # Initialize components
        self._init_metrics()
        self._init_tracing()
        self._init_logging()
        self._init_health()
        
        self._started = False
        self._lock = threading.Lock()
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        if self.config.metrics_enabled:
            exporter_config = ExporterConfig(
                port=self.config.metrics_port,
                path=self.config.metrics_path,
                namespace=self.config.service_name.replace("-", "_")
            )
            
            self.metrics_exporter, self.inference_metrics, \
                self.node_metrics, self.cache_metrics = create_exporter(
                    node_id=self.config.node_id,
                    config=exporter_config
                )
        else:
            self.metrics_exporter = None
            self.inference_metrics = None
            self.node_metrics = None
            self.cache_metrics = None
    
    def _init_tracing(self) -> None:
        """Initialize Jaeger tracing."""
        if self.config.tracing_enabled:
            jaeger_config = JaegerConfig(
                agent_host=self.config.jaeger_agent_host,
                agent_port=self.config.jaeger_agent_port,
                service_name=self.config.service_name,
                collector_endpoint=self.config.jaeger_collector_endpoint
            )
            
            self.tracer = create_jaeger_tracer(
                service_name=self.config.service_name,
                config=jaeger_config,
                use_http=self.config.jaeger_use_http
            )
        else:
            self.tracer = None
    
    def _init_logging(self) -> None:
        """Initialize structured logging."""
        self.logger = StructuredLogger(
            name=self.config.service_name,
            config=self.config
        )
    
    def _init_health(self) -> None:
        """Initialize health checker."""
        self.health = HealthChecker(self.config)
        
        # Register self-checks
        if self.config.metrics_enabled:
            self.health.register_component(
                "metrics",
                lambda: self.metrics_exporter is not None
            )
        
        if self.config.tracing_enabled:
            self.health.register_component(
                "tracing",
                lambda: self.tracer is not None
            )
    
    def start(self) -> None:
        """Start all observability components."""
        with self._lock:
            if self._started:
                return
            
            if self.tracer:
                self.tracer.start()
            
            self._started = True
            self.logger.info("Observability client started")
    
    def shutdown(self) -> None:
        """Shutdown all observability components."""
        with self._lock:
            if not self._started:
                return
            
            if self.tracer:
                self.tracer.shutdown()
            
            self._started = False
            self.logger.info("Observability client shutdown")
    
    # ==========================================================================
    # Context Management
    # ==========================================================================
    
    def create_context(
        self,
        request_id: Optional[str] = None,
        model: Optional[str] = None,
        user_id: Optional[str] = None,
        **metadata
    ) -> RequestContext:
        """
        Create a new request context.
        
        Automatically creates trace and span IDs.
        """
        import uuid
        
        request_id = request_id or str(uuid.uuid4())
        trace_id = uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        
        return RequestContext(
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id,
            model=model,
            user_id=user_id,
            metadata=metadata
        )
    
    @contextmanager
    def trace_request(
        self,
        operation: str,
        context: Optional[RequestContext] = None,
        **tags
    ):
        """
        Context manager for tracing a request.
        
        Handles span creation, logging, and metrics recording.
        
        Usage:
            with client.trace_request("inference", context) as span:
                # do work
                span.tags["result"] = "success"
        """
        if context is None:
            context = self.create_context()
        
        # Set logging context
        self.logger.set_context(context)
        
        # Start span
        span = None
        if self.tracer:
            span = self.tracer.start_span(
                operation_name=operation,
                trace_id=context.trace_id,
                parent_span_id=context.parent_span_id,
                kind="server",
                tags=tags
            )
        
        start_time = time.time()
        error_occurred = False
        
        try:
            yield span or context
        except Exception as e:
            error_occurred = True
            self.logger.error(f"Error in {operation}", error=e)
            if span:
                self.tracer.end_span(span, status="error", error_message=str(e))
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            # End span if no error
            if span and not error_occurred:
                self.tracer.end_span(span, status="ok")
            
            # Record metrics
            if self.inference_metrics:
                self.inference_metrics.record_request(
                    latency_ms=duration_ms,
                    tokens=tags.get("tokens", 0),
                    batch_size=tags.get("batch_size", 1),
                    error=error_occurred
                )
            
            # Clear logging context
            self.logger.clear_context()
    
    # ==========================================================================
    # Metrics Shortcuts
    # ==========================================================================
    
    def record_request(
        self,
        latency_ms: float,
        tokens: int,
        batch_size: int = 1,
        error: bool = False
    ) -> None:
        """Record an inference request."""
        if self.inference_metrics:
            self.inference_metrics.record_request(
                latency_ms=latency_ms,
                tokens=tokens,
                batch_size=batch_size,
                error=error
            )
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        if self.cache_metrics:
            self.cache_metrics.record_hit()
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        if self.cache_metrics:
            self.cache_metrics.record_miss()
    
    def update_node_health(self, healthy: bool) -> None:
        """Update node health status."""
        if self.node_metrics:
            self.node_metrics.update_health(healthy)
    
    def update_node_resources(
        self,
        cpu: float = 0.0,
        memory: float = 0.0,
        gpu_util: float = 0.0,
        gpu_mem_used: float = 0.0,
        gpu_mem_total: float = 0.0
    ) -> None:
        """Update node resource usage."""
        if self.node_metrics:
            self.node_metrics.update_resources(
                cpu=cpu,
                memory=memory,
                gpu_util=gpu_util,
                gpu_mem_used=gpu_mem_used,
                gpu_mem_total=gpu_mem_total
            )
    
    # ==========================================================================
    # Tracing Shortcuts
    # ==========================================================================
    
    def start_span(
        self,
        operation: str,
        parent: Optional[SpanData] = None,
        **tags
    ) -> Optional[SpanData]:
        """Start a new span."""
        if self.tracer:
            return self.tracer.start_span(
                operation_name=operation,
                parent_span_id=parent.span_id if parent else None,
                trace_id=parent.trace_id if parent else None,
                tags=tags
            )
        return None
    
    def end_span(
        self,
        span: Optional[SpanData],
        status: str = "ok",
        error: Optional[str] = None
    ) -> None:
        """End a span."""
        if self.tracer and span:
            self.tracer.end_span(span, status=status, error_message=error)
    
    # ==========================================================================
    # Health & Metrics Endpoints
    # ==========================================================================
    
    def get_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        if self.metrics_exporter:
            return self.metrics_exporter.get_metrics()
        return ""
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        return self.health.check_all()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "service": self.config.service_name,
            "node_id": self.config.node_id,
            "environment": self.config.environment,
            "version": self.config.version
        }
        
        if self.tracer:
            stats["tracing"] = self.tracer.get_stats()
        
        stats["health"] = self.health.get_status()
        
        return stats


def create_observability_client(
    service_name: str = "llm-inference",
    node_id: str = "node-0",
    environment: str = "development",
    **kwargs
) -> ObservabilityClient:
    """
    Create and start an observability client.
    
    Convenience function for quick setup.
    """
    config = ObservabilityConfig(
        service_name=service_name,
        node_id=node_id,
        environment=environment,
        **kwargs
    )
    
    client = ObservabilityClient(config)
    client.start()
    
    return client


def create_test_client() -> ObservabilityClient:
    """
    Create an observability client for testing.
    
    Uses in-memory exporters, suitable for unit tests.
    """
    config = ObservabilityConfig(
        service_name="test-service",
        node_id="test-node",
        environment="test",
        metrics_enabled=True,
        tracing_enabled=True,
        logging_enabled=True
    )
    
    return ObservabilityClient(config)
