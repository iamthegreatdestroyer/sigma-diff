"""
Distributed Tracing Module for LLM Inference.

Provides OpenTelemetry-based distributed tracing with:
- Automatic span creation and context propagation
- Custom span processors for LLM metrics
- Jaeger/Zipkin exporter support
- Correlation with logging system
"""

from .tracer import (
    LLMTracer,
    Span,
    SpanKind,
    SpanStatus,
    get_tracer,
    init_tracing,
    shutdown_tracing,
)
from .context import (
    TracingContext,
    ContextCarrier,
    get_current_context,
    inject_context,
    extract_context,
)
from .span_processor import (
    LLMSpanProcessor,
    MetricsSpanProcessor,
    BatchSpanProcessor,
    JaegerSpanExporter,
)

__all__ = [
    # Tracer
    "LLMTracer",
    "Span",
    "SpanKind",
    "SpanStatus",
    "get_tracer",
    "init_tracing",
    "shutdown_tracing",
    # Context
    "TracingContext",
    "ContextCarrier",
    "get_current_context",
    "inject_context",
    "extract_context",
    # Span Processors
    "LLMSpanProcessor",
    "MetricsSpanProcessor",
    "BatchSpanProcessor",
    "JaegerSpanExporter",
]

__version__ = "1.0.0"
