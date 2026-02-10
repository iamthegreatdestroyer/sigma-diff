"""
OpenTelemetry-based Distributed Tracer for LLM Inference.

Provides comprehensive tracing for:
- Inference requests across distributed workers
- Token generation timing
- Model loading and caching operations
- Batch processing workflows
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from contextlib import contextmanager
import uuid


class SpanKind(Enum):
    """Type of span in the trace."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Status of a span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Immutable context for a span."""
    trace_id: str
    span_id: str
    trace_flags: int = 1  # Sampled by default
    trace_state: Dict[str, str] = field(default_factory=dict)
    is_remote: bool = False
    
    def is_valid(self) -> bool:
        """Check if context is valid."""
        return bool(self.trace_id and self.span_id)


@dataclass
class SpanEvent:
    """Event recorded during span execution."""
    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """Link to another span."""
    context: SpanContext
    attributes: Dict[str, Any] = field(default_factory=dict)


class Span:
    """
    Represents a single operation within a trace.
    
    Spans are the building blocks of distributed traces, tracking
    individual operations with timing, attributes, and events.
    """
    
    def __init__(
        self,
        name: str,
        context: SpanContext,
        parent_context: Optional[SpanContext] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanLink]] = None,
    ):
        self.name = name
        self.context = context
        self.parent_context = parent_context
        self.kind = kind
        self.attributes: Dict[str, Any] = attributes or {}
        self.links: List[SpanLink] = links or []
        self.events: List[SpanEvent] = []
        self.status = SpanStatus.UNSET
        self.status_message: Optional[str] = None
        
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self._ended = False
        self._lock = threading.Lock()
    
    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a single attribute."""
        with self._lock:
            if not self._ended:
                self.attributes[key] = value
        return self
    
    def set_attributes(self, attributes: Dict[str, Any]) -> "Span":
        """Set multiple attributes."""
        with self._lock:
            if not self._ended:
                self.attributes.update(attributes)
        return self
    
    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> "Span":
        """Add an event to the span."""
        with self._lock:
            if not self._ended:
                event = SpanEvent(
                    name=name,
                    timestamp=timestamp or time.time(),
                    attributes=attributes or {},
                )
                self.events.append(event)
        return self
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> "Span":
        """Set span status."""
        with self._lock:
            if not self._ended:
                self.status = status
                self.status_message = message
        return self
    
    def record_exception(self, exception: Exception) -> "Span":
        """Record an exception."""
        self.add_event(
            "exception",
            attributes={
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )
        self.set_status(SpanStatus.ERROR, str(exception))
        return self
    
    def end(self, end_time: Optional[float] = None) -> None:
        """End the span."""
        with self._lock:
            if not self._ended:
                self.end_time = end_time or time.time()
                self._ended = True
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000
    
    @property
    def trace_id(self) -> str:
        """Get the trace ID for this span."""
        return self.context.trace_id
    
    @property
    def span_id(self) -> str:
        """Get the span ID for this span."""
        return self.context.span_id
    
    def is_recording(self) -> bool:
        """Check if the span is still recording (not ended)."""
        return not self._ended
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute value by key."""
        return self.attributes.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.parent_context.span_id if self.parent_context else None,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
            "links": [
                {"trace_id": l.context.trace_id, "span_id": l.context.span_id}
                for l in self.links
            ],
        }


class LLMTracer:
    """
    Distributed tracer optimized for LLM inference workloads.
    
    Features:
    - Automatic trace context propagation
    - LLM-specific span attributes
    - Integration with Jaeger/Zipkin exporters
    - Sampling strategies for high-volume traces
    """
    
    def __init__(
        self,
        service_name: str,
        sampling_rate: float = 1.0,
        max_spans_per_trace: int = 1000,
        span_processors: Optional[List[Any]] = None,
    ):
        self.service_name = service_name
        self.sampling_rate = sampling_rate
        self.max_spans_per_trace = max_spans_per_trace
        self.span_processors = span_processors or []
        
        self._active_spans: Dict[str, List[Span]] = {}
        self._completed_spans: List[Span] = []
        self._lock = threading.Lock()
        
        # Thread-local storage for current span
        self._current_span = threading.local()
    
    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return uuid.uuid4().hex
    
    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return uuid.uuid4().hex[:16]
    
    def _should_sample(self) -> bool:
        """Determine if trace should be sampled."""
        import random
        return random.random() < self.sampling_rate
    
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent_context: Optional[SpanContext] = None,
        links: Optional[List[SpanLink]] = None,
    ) -> Span:
        """
        Start a new span.
        
        Args:
            name: Span name
            kind: Type of span
            attributes: Initial attributes
            parent_context: Parent span context
            links: Links to other spans
            
        Returns:
            New span instance
        """
        # Get or create trace context
        if parent_context:
            trace_id = parent_context.trace_id
        elif hasattr(self._current_span, 'span') and self._current_span.span:
            trace_id = self._current_span.span.context.trace_id
            parent_context = self._current_span.span.context
        else:
            trace_id = self._generate_trace_id()
        
        # Create span context
        context = SpanContext(
            trace_id=trace_id,
            span_id=self._generate_span_id(),
        )
        
        # Add service name to attributes
        attrs = {"service.name": self.service_name}
        if attributes:
            attrs.update(attributes)
        
        # Create span
        span = Span(
            name=name,
            context=context,
            parent_context=parent_context,
            kind=kind,
            attributes=attrs,
            links=links,
        )
        
        # Track active span
        with self._lock:
            if trace_id not in self._active_spans:
                self._active_spans[trace_id] = []
            self._active_spans[trace_id].append(span)
        
        # Notify processors
        for processor in self.span_processors:
            if hasattr(processor, 'on_start'):
                processor.on_start(span, parent_context)
        
        return span
    
    def end_span(self, span: Span) -> None:
        """End a span and process it."""
        span.end()
        
        # Remove from active spans
        with self._lock:
            trace_id = span.context.trace_id
            if trace_id in self._active_spans:
                self._active_spans[trace_id] = [
                    s for s in self._active_spans[trace_id]
                    if s.context.span_id != span.context.span_id
                ]
                if not self._active_spans[trace_id]:
                    del self._active_spans[trace_id]
            
            self._completed_spans.append(span)
        
        # Notify processors
        for processor in self.span_processors:
            if hasattr(processor, 'on_end'):
                processor.on_end(span)
    
    @contextmanager
    def trace(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracing an operation.
        
        Usage:
            with tracer.trace("operation_name") as span:
                # Do work
                span.set_attribute("key", "value")
        """
        span = self.start_span(name, kind=kind, attributes=attributes)
        old_span = getattr(self._current_span, 'span', None)
        self._current_span.span = span
        
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            self._current_span.span = old_span
            self.end_span(span)
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return getattr(self._current_span, 'span', None)
    
    def get_active_traces(self) -> Dict[str, int]:
        """Get active trace IDs and span counts."""
        with self._lock:
            return {
                trace_id: len(spans)
                for trace_id, spans in self._active_spans.items()
            }
    
    def get_completed_spans(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recently completed spans."""
        with self._lock:
            return [span.to_dict() for span in self._completed_spans[-limit:]]
    
    def flush(self) -> None:
        """Flush all pending spans to processors."""
        for processor in self.span_processors:
            if hasattr(processor, 'force_flush'):
                processor.force_flush()
    
    def shutdown(self) -> None:
        """Shutdown the tracer."""
        self.flush()
        for processor in self.span_processors:
            if hasattr(processor, 'shutdown'):
                processor.shutdown()


# Global tracer instance
_global_tracer: Optional[LLMTracer] = None
_tracer_lock = threading.Lock()


def init_tracing(
    service_name: str,
    sampling_rate: float = 1.0,
    span_processors: Optional[List[Any]] = None,
) -> LLMTracer:
    """
    Initialize global tracing.
    
    Args:
        service_name: Name of the service
        sampling_rate: Fraction of traces to sample (0.0-1.0)
        span_processors: List of span processors
        
    Returns:
        Configured tracer instance
    """
    global _global_tracer
    
    with _tracer_lock:
        _global_tracer = LLMTracer(
            service_name=service_name,
            sampling_rate=sampling_rate,
            span_processors=span_processors or [],
        )
    
    return _global_tracer


def get_tracer() -> LLMTracer:
    """Get the global tracer instance."""
    global _global_tracer
    
    if _global_tracer is None:
        with _tracer_lock:
            if _global_tracer is None:
                _global_tracer = LLMTracer(service_name="default")
    
    return _global_tracer


def shutdown_tracing() -> None:
    """Shutdown global tracing."""
    global _global_tracer
    
    with _tracer_lock:
        if _global_tracer:
            _global_tracer.shutdown()
            _global_tracer = None
