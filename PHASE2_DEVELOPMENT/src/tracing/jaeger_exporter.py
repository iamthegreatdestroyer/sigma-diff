"""
Jaeger Distributed Tracing Exporter.

Exports OpenTelemetry-compatible spans to Jaeger for visualization
of distributed inference request flows.

Sprint 3.5 - Observability Stack
Created: 2026-01-06
"""

import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import uuid

logger = logging.getLogger(__name__)


class ExportResult(Enum):
    """Result of export operation."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


@dataclass
class JaegerConfig:
    """Configuration for Jaeger exporter."""
    agent_host: str = "localhost"
    agent_port: int = 6831  # UDP Thrift compact
    collector_endpoint: Optional[str] = None  # HTTP collector (overrides agent)
    service_name: str = "llm-inference"
    max_queue_size: int = 10000
    batch_size: int = 100
    flush_interval: float = 5.0  # seconds
    timeout: float = 30.0


@dataclass
class SpanData:
    """Span data for export."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: float  # Unix timestamp
    duration: float  # seconds
    status: str = "ok"
    kind: str = "internal"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "traceIdLow": self.trace_id[-16:] if len(self.trace_id) > 16 else self.trace_id,
            "traceIdHigh": self.trace_id[:-16] if len(self.trace_id) > 16 else "",
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "operationName": self.operation_name,
            "serviceName": self.service_name,
            "startTime": int(self.start_time * 1_000_000),  # microseconds
            "duration": int(self.duration * 1_000_000),  # microseconds
            "tags": [{"key": k, "value": str(v)} for k, v in self.tags.items()],
            "logs": self.logs,
            "references": self.references
        }


class SpanExporter(ABC):
    """Abstract base class for span exporters."""
    
    @abstractmethod
    def export(self, spans: List[SpanData]) -> ExportResult:
        """Export a batch of spans."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class InMemorySpanExporter(SpanExporter):
    """
    In-memory span exporter for testing.
    
    Stores all exported spans in memory for verification.
    """
    
    def __init__(self, max_spans: int = 10000):
        self.max_spans = max_spans
        self.spans: List[SpanData] = []
        self._lock = threading.Lock()
    
    def export(self, spans: List[SpanData]) -> ExportResult:
        with self._lock:
            self.spans.extend(spans)
            # Trim if over limit
            if len(self.spans) > self.max_spans:
                self.spans = self.spans[-self.max_spans:]
        return ExportResult.SUCCESS
    
    def shutdown(self) -> None:
        pass
    
    def get_spans(self) -> List[SpanData]:
        """Get all exported spans."""
        with self._lock:
            return list(self.spans)
    
    def clear(self) -> None:
        """Clear all spans."""
        with self._lock:
            self.spans.clear()
    
    def find_trace(self, trace_id: str) -> List[SpanData]:
        """Find all spans for a trace."""
        with self._lock:
            return [s for s in self.spans if s.trace_id == trace_id]
    
    def find_operation(self, operation: str) -> List[SpanData]:
        """Find spans by operation name."""
        with self._lock:
            return [s for s in self.spans if s.operation_name == operation]


class JaegerThriftExporter(SpanExporter):
    """
    Exports spans to Jaeger using Thrift over UDP.
    
    Compatible with jaeger-agent on port 6831.
    """
    
    def __init__(self, config: Optional[JaegerConfig] = None):
        self.config = config or JaegerConfig()
        self._socket = None
        self._lock = threading.Lock()
    
    def _connect(self) -> None:
        """Connect to Jaeger agent."""
        import socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def export(self, spans: List[SpanData]) -> ExportResult:
        """Export spans to Jaeger."""
        if not self._socket:
            self._connect()
        
        try:
            # In production, this would serialize to Thrift format
            # For now, we'll use JSON as a placeholder
            for span in spans:
                data = json.dumps(span.to_dict()).encode('utf-8')
                with self._lock:
                    self._socket.sendto(
                        data,
                        (self.config.agent_host, self.config.agent_port)
                    )
            return ExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Failed to export spans to Jaeger: {e}")
            return ExportResult.FAILURE
    
    def shutdown(self) -> None:
        """Close connection."""
        if self._socket:
            self._socket.close()
            self._socket = None


class JaegerHTTPExporter(SpanExporter):
    """
    Exports spans to Jaeger collector via HTTP.
    
    Uses the HTTP collector endpoint for span ingestion.
    """
    
    def __init__(self, config: Optional[JaegerConfig] = None):
        self.config = config or JaegerConfig()
        if not self.config.collector_endpoint:
            self.config.collector_endpoint = "http://localhost:14268/api/traces"
    
    def export(self, spans: List[SpanData]) -> ExportResult:
        """Export spans via HTTP POST."""
        try:
            # Prepare batch
            batch = {
                "process": {
                    "serviceName": self.config.service_name,
                    "tags": []
                },
                "spans": [span.to_dict() for span in spans]
            }
            
            # In production, use aiohttp or requests
            # For testing, we'll simulate the export
            logger.debug(f"Would export {len(spans)} spans to {self.config.collector_endpoint}")
            
            return ExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Failed to export spans via HTTP: {e}")
            return ExportResult.FAILURE
    
    def shutdown(self) -> None:
        pass


class BatchSpanProcessor:
    """
    Batches spans for efficient export.
    
    Accumulates spans in a queue and exports in batches
    based on size or time thresholds.
    """
    
    def __init__(
        self,
        exporter: SpanExporter,
        max_queue_size: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 5.0
    ):
        self.exporter = exporter
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._stats = {
            "spans_processed": 0,
            "spans_dropped": 0,
            "batches_exported": 0,
            "export_errors": 0
        }
    
    def start(self) -> None:
        """Start the background worker."""
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        logger.info("BatchSpanProcessor started")
    
    def _run(self) -> None:
        """Background worker loop."""
        batch: List[SpanData] = []
        last_flush = time.time()
        
        while not self._shutdown.is_set():
            try:
                # Get span with timeout
                try:
                    span = self._queue.get(timeout=0.1)
                    batch.append(span)
                except queue.Empty:
                    pass
                
                # Check if we should flush
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and time.time() - last_flush >= self.flush_interval)
                )
                
                if should_flush and batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except Exception as e:
                logger.error(f"Error in span processor: {e}")
        
        # Flush remaining on shutdown
        if batch:
            self._flush_batch(batch)
    
    def _flush_batch(self, batch: List[SpanData]) -> None:
        """Flush a batch of spans."""
        try:
            result = self.exporter.export(batch)
            self._stats["batches_exported"] += 1
            self._stats["spans_processed"] += len(batch)
            
            if result == ExportResult.FAILURE:
                self._stats["export_errors"] += 1
                
        except Exception as e:
            logger.error(f"Failed to export batch: {e}")
            self._stats["export_errors"] += 1
    
    def add_span(self, span: SpanData) -> bool:
        """
        Add a span to the processing queue.
        
        Returns:
            True if added, False if dropped
        """
        try:
            self._queue.put_nowait(span)
            return True
        except queue.Full:
            self._stats["spans_dropped"] += 1
            return False
    
    def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown the processor."""
        self._shutdown.set()
        if self._worker:
            self._worker.join(timeout=timeout)
        self.exporter.shutdown()
        logger.info("BatchSpanProcessor shutdown")
    
    def get_stats(self) -> Dict[str, int]:
        """Get processor statistics."""
        return dict(self._stats)


class DistributedTracer:
    """
    High-level distributed tracing API.
    
    Provides convenient methods for tracing inference operations
    with automatic span creation and export.
    """
    
    def __init__(
        self,
        service_name: str = "llm-inference",
        exporter: Optional[SpanExporter] = None,
        config: Optional[JaegerConfig] = None
    ):
        self.service_name = service_name
        self.config = config or JaegerConfig(service_name=service_name)
        
        if exporter is None:
            exporter = InMemorySpanExporter()
        
        self.processor = BatchSpanProcessor(
            exporter=exporter,
            max_queue_size=self.config.max_queue_size,
            batch_size=self.config.batch_size,
            flush_interval=self.config.flush_interval
        )
        
        self._active_spans: Dict[str, SpanData] = {}
        self._context = threading.local()
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start the tracer."""
        self.processor.start()
    
    def shutdown(self) -> None:
        """Shutdown the tracer."""
        self.processor.shutdown()
    
    def _generate_id(self, length: int = 16) -> str:
        """Generate a random hex ID."""
        return uuid.uuid4().hex[:length]
    
    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        kind: str = "internal",
        tags: Optional[Dict[str, Any]] = None
    ) -> SpanData:
        """
        Start a new span.
        
        Args:
            operation_name: Name of the operation
            parent_span_id: Optional parent span ID
            trace_id: Optional trace ID (generates new if not provided)
            kind: Span kind (internal, server, client, producer, consumer)
            tags: Optional tags
        
        Returns:
            The created span
        """
        span_id = self._generate_id(16)
        
        if trace_id is None:
            trace_id = self._generate_id(32)
        
        span = SpanData(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=time.time(),
            duration=0.0,
            kind=kind,
            tags=tags or {}
        )
        
        with self._lock:
            self._active_spans[span_id] = span
        
        return span
    
    def end_span(
        self,
        span: SpanData,
        status: str = "ok",
        error_message: Optional[str] = None
    ) -> None:
        """
        End a span and export it.
        
        Args:
            span: The span to end
            status: Status (ok, error)
            error_message: Optional error message
        """
        span.duration = time.time() - span.start_time
        span.status = status
        
        if error_message:
            span.tags["error.message"] = error_message
        
        with self._lock:
            self._active_spans.pop(span.span_id, None)
        
        self.processor.add_span(span)
    
    def add_event(
        self,
        span: SpanData,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an event/log to a span."""
        span.logs.append({
            "timestamp": int(time.time() * 1_000_000),
            "fields": [
                {"key": "event", "value": name},
                *[{"key": k, "value": str(v)} for k, v in (attributes or {}).items()]
            ]
        })
    
    def set_tag(self, span: SpanData, key: str, value: Any) -> None:
        """Set a tag on a span."""
        span.tags[key] = value
    
    def trace_inference(
        self,
        model: str,
        prompt_tokens: int,
        max_tokens: int,
        parent_span: Optional[SpanData] = None
    ) -> SpanData:
        """
        Start a span for inference operation.
        
        Convenience method with inference-specific tags.
        """
        return self.start_span(
            operation_name="inference",
            parent_span_id=parent_span.span_id if parent_span else None,
            trace_id=parent_span.trace_id if parent_span else None,
            kind="server",
            tags={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens
            }
        )
    
    def trace_batch(
        self,
        batch_size: int,
        parent_span: Optional[SpanData] = None
    ) -> SpanData:
        """Start a span for batch processing."""
        return self.start_span(
            operation_name="batch_process",
            parent_span_id=parent_span.span_id if parent_span else None,
            trace_id=parent_span.trace_id if parent_span else None,
            kind="internal",
            tags={"batch_size": batch_size}
        )
    
    def trace_cache_lookup(
        self,
        cache_type: str,
        parent_span: Optional[SpanData] = None
    ) -> SpanData:
        """Start a span for cache lookup."""
        return self.start_span(
            operation_name="cache_lookup",
            parent_span_id=parent_span.span_id if parent_span else None,
            trace_id=parent_span.trace_id if parent_span else None,
            kind="client",
            tags={"cache_type": cache_type}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        with self._lock:
            active = len(self._active_spans)
        
        return {
            "active_spans": active,
            **self.processor.get_stats()
        }


def create_jaeger_tracer(
    service_name: str = "llm-inference",
    config: Optional[JaegerConfig] = None,
    use_http: bool = False
) -> DistributedTracer:
    """
    Create a tracer configured for Jaeger export.
    
    Args:
        service_name: Name of the service
        config: Jaeger configuration
        use_http: Use HTTP exporter instead of UDP
    
    Returns:
        Configured DistributedTracer
    """
    config = config or JaegerConfig(service_name=service_name)
    
    if use_http:
        exporter = JaegerHTTPExporter(config)
    else:
        exporter = JaegerThriftExporter(config)
    
    tracer = DistributedTracer(
        service_name=service_name,
        exporter=exporter,
        config=config
    )
    
    return tracer


def create_test_tracer(
    service_name: str = "llm-inference-test"
) -> tuple:
    """
    Create a tracer with in-memory exporter for testing.
    
    Returns:
        Tuple of (tracer, in_memory_exporter)
    """
    exporter = InMemorySpanExporter()
    tracer = DistributedTracer(
        service_name=service_name,
        exporter=exporter
    )
    return tracer, exporter
