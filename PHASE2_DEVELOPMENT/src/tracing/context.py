"""
Tracing Context Propagation for Distributed Systems.

Implements W3C Trace Context standard for cross-service
trace propagation with support for:
- HTTP headers
- gRPC metadata
- Message queue headers
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager

from .tracer import SpanContext, Span, get_tracer


# W3C Trace Context header names
TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"


@dataclass
class TracingContext:
    """
    Container for distributed tracing context.
    
    Holds all context needed for trace propagation including:
    - Trace and span IDs
    - Sampling decisions
    - Vendor-specific trace state
    - Baggage items
    """
    
    trace_id: str
    span_id: str
    trace_flags: int = 1
    trace_state: Dict[str, str] = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)
    is_valid: bool = True
    
    @classmethod
    def from_span_context(cls, ctx: SpanContext) -> "TracingContext":
        """Create from SpanContext."""
        return cls(
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            trace_flags=ctx.trace_flags,
            trace_state=ctx.trace_state.copy(),
            is_valid=ctx.is_valid(),
        )
    
    def to_span_context(self) -> SpanContext:
        """Convert to SpanContext."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            trace_flags=self.trace_flags,
            trace_state=self.trace_state.copy(),
            is_remote=True,
        )
    
    def with_baggage(self, key: str, value: str) -> "TracingContext":
        """Create new context with additional baggage item."""
        new_baggage = self.baggage.copy()
        new_baggage[key] = value
        return TracingContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            trace_flags=self.trace_flags,
            trace_state=self.trace_state.copy(),
            baggage=new_baggage,
            is_valid=self.is_valid,
        )
    
    def is_sampled(self) -> bool:
        """Check if trace is sampled."""
        return bool(self.trace_flags & 0x01)


# Thread-local context storage
_context_storage = threading.local()


def get_current_context() -> Optional[TracingContext]:
    """
    Get the current tracing context from thread-local storage.
    
    Returns:
        Current TracingContext or None
    """
    return getattr(_context_storage, 'context', None)


def set_current_context(context: Optional[TracingContext]) -> None:
    """
    Set the current tracing context.
    
    Args:
        context: TracingContext to set (or None to clear)
    """
    _context_storage.context = context


@contextmanager
def use_context(context: TracingContext):
    """
    Context manager to temporarily use a specific tracing context.
    
    Args:
        context: TracingContext to use
        
    Yields:
        The context being used
    """
    old_context = get_current_context()
    set_current_context(context)
    try:
        yield context
    finally:
        set_current_context(old_context)


def inject_context(
    carrier: Dict[str, str],
    context: Optional[TracingContext] = None,
) -> Dict[str, str]:
    """
    Inject tracing context into a carrier (e.g., HTTP headers).
    
    Implements W3C Trace Context format:
    - traceparent: {version}-{trace_id}-{span_id}-{trace_flags}
    - tracestate: vendor-specific key=value pairs
    
    Args:
        carrier: Dictionary to inject headers into
        context: Context to inject (uses current if not provided)
        
    Returns:
        Carrier with injected headers
    """
    ctx = context or get_current_context()
    
    if ctx is None:
        # Try to get from current span
        tracer = get_tracer()
        span = tracer.get_current_span()
        if span:
            ctx = TracingContext.from_span_context(span.context)
    
    if ctx and ctx.is_valid:
        # Format traceparent: version-trace_id-span_id-trace_flags
        version = "00"
        trace_flags = f"{ctx.trace_flags:02x}"
        traceparent = f"{version}-{ctx.trace_id}-{ctx.span_id}-{trace_flags}"
        carrier[TRACEPARENT_HEADER] = traceparent
        
        # Format tracestate: key1=value1,key2=value2
        if ctx.trace_state:
            tracestate = ",".join(
                f"{k}={v}" for k, v in ctx.trace_state.items()
            )
            carrier[TRACESTATE_HEADER] = tracestate
        
        # Inject baggage
        for key, value in ctx.baggage.items():
            carrier[f"baggage-{key}"] = value
    
    return carrier


def extract_context(carrier: Dict[str, str]) -> Optional[TracingContext]:
    """
    Extract tracing context from a carrier (e.g., HTTP headers).
    
    Parses W3C Trace Context format.
    
    Args:
        carrier: Dictionary containing trace headers
        
    Returns:
        Extracted TracingContext or None
    """
    # Case-insensitive header lookup
    headers = {k.lower(): v for k, v in carrier.items()}
    
    traceparent = headers.get(TRACEPARENT_HEADER)
    if not traceparent:
        return None
    
    try:
        parts = traceparent.split("-")
        if len(parts) != 4:
            return None
        
        version, trace_id, span_id, trace_flags = parts
        
        # Validate version
        if version != "00":
            return None
        
        # Parse trace_flags
        flags = int(trace_flags, 16)
        
        # Parse tracestate
        trace_state: Dict[str, str] = {}
        tracestate = headers.get(TRACESTATE_HEADER)
        if tracestate:
            for pair in tracestate.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    trace_state[key.strip()] = value.strip()
        
        # Extract baggage
        baggage: Dict[str, str] = {}
        for key, value in headers.items():
            if key.startswith("baggage-"):
                baggage_key = key[8:]  # Remove "baggage-" prefix
                baggage[baggage_key] = value
        
        return TracingContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=flags,
            trace_state=trace_state,
            baggage=baggage,
            is_valid=True,
        )
        
    except (ValueError, IndexError):
        return None


def create_child_context(
    parent: Optional[TracingContext] = None,
) -> TracingContext:
    """
    Create a child context with a new span ID.
    
    Args:
        parent: Parent context (uses current if not provided)
        
    Returns:
        New child TracingContext
    """
    import uuid
    
    parent = parent or get_current_context()
    
    if parent:
        return TracingContext(
            trace_id=parent.trace_id,
            span_id=uuid.uuid4().hex[:16],
            trace_flags=parent.trace_flags,
            trace_state=parent.trace_state.copy(),
            baggage=parent.baggage.copy(),
            is_valid=True,
        )
    else:
        return TracingContext(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            trace_flags=1,
            is_valid=True,
        )


class ContextCarrier:
    """
    Generic carrier for context propagation.
    
    Supports multiple transport formats.
    """
    
    def __init__(self, format_type: str = "http"):
        self.format_type = format_type
        self.headers: Dict[str, str] = {}
    
    def inject(self, context: Optional[TracingContext] = None) -> Dict[str, str]:
        """Inject context into carrier."""
        self.headers = inject_context({}, context)
        return self.headers
    
    def extract(self) -> Optional[TracingContext]:
        """Extract context from carrier."""
        return extract_context(self.headers)
    
    def set_header(self, key: str, value: str) -> None:
        """Set a header value."""
        self.headers[key] = value
    
    def get_header(self, key: str) -> Optional[str]:
        """Get a header value."""
        return self.headers.get(key)


class GRPCContextCarrier(ContextCarrier):
    """Carrier for gRPC metadata propagation."""
    
    def __init__(self):
        super().__init__(format_type="grpc")
    
    def to_metadata(self) -> list:
        """Convert to gRPC metadata format."""
        return [(k, v) for k, v in self.headers.items()]
    
    def from_metadata(self, metadata: list) -> "GRPCContextCarrier":
        """Load from gRPC metadata."""
        self.headers = {k: v for k, v in metadata}
        return self


class MessageQueueCarrier(ContextCarrier):
    """Carrier for message queue header propagation."""
    
    def __init__(self):
        super().__init__(format_type="mq")
    
    def to_message_properties(self) -> Dict[str, str]:
        """Convert to message properties."""
        return self.headers.copy()
    
    def from_message_properties(
        self, properties: Dict[str, str]
    ) -> "MessageQueueCarrier":
        """Load from message properties."""
        self.headers = properties.copy()
        return self
