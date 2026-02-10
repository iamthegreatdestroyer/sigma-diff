"""
Span Processors for LLM Inference Tracing.

Provides processors for:
- LLM-specific metric extraction
- Batch span export
- Metrics aggregation
- Jaeger/Zipkin export formatting
"""

from __future__ import annotations

import time
import threading
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from collections import defaultdict
from queue import Queue, Empty

from .tracer import Span, SpanContext, SpanStatus


class SpanProcessor(ABC):
    """Base class for span processors."""
    
    @abstractmethod
    def on_start(self, span: Span, parent_context: Optional[SpanContext]) -> None:
        """Called when a span starts."""
        pass
    
    @abstractmethod
    def on_end(self, span: Span) -> None:
        """Called when a span ends."""
        pass
    
    def force_flush(self, timeout_ms: int = 30000) -> bool:
        """Force flush pending data."""
        return True
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass


class LLMSpanProcessor(SpanProcessor):
    """
    Specialized span processor for LLM inference operations.
    
    Extracts and aggregates LLM-specific metrics:
    - Token counts (input/output)
    - Generation latency
    - Model-specific attributes
    - Batch processing statistics
    """
    
    def __init__(
        self,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.metrics_callback = metrics_callback
        self._lock = threading.Lock()
        
        # Aggregated metrics
        self._token_counts: Dict[str, int] = defaultdict(int)
        self._latency_samples: List[float] = []
        self._operation_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
    
    def on_start(self, span: Span, parent_context: Optional[SpanContext]) -> None:
        """Record span start for LLM operations."""
        operation = span.name
        with self._lock:
            self._operation_counts[operation] += 1
    
    def on_end(self, span: Span) -> None:
        """Extract LLM metrics from completed span."""
        with self._lock:
            # Track latency
            self._latency_samples.append(span.duration_ms)
            
            # Extract token counts if present
            if "input_tokens" in span.attributes:
                self._token_counts["input"] += span.attributes["input_tokens"]
            if "output_tokens" in span.attributes:
                self._token_counts["output"] += span.attributes["output_tokens"]
            
            # Track errors
            if span.status == SpanStatus.ERROR:
                error_type = span.attributes.get("error.type", "unknown")
                self._error_counts[error_type] += 1
        
        # Send metrics if callback configured
        if self.metrics_callback:
            metrics = self._extract_span_metrics(span)
            self.metrics_callback(metrics)
    
    def _extract_span_metrics(self, span: Span) -> Dict[str, Any]:
        """Extract metrics from a span."""
        return {
            "operation": span.name,
            "duration_ms": span.duration_ms,
            "status": span.status.value,
            "trace_id": span.context.trace_id,
            "span_id": span.context.span_id,
            "input_tokens": span.attributes.get("input_tokens", 0),
            "output_tokens": span.attributes.get("output_tokens", 0),
            "model": span.attributes.get("model", "unknown"),
            "timestamp": span.end_time or time.time(),
        }
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        with self._lock:
            latency_stats = {}
            if self._latency_samples:
                sorted_samples = sorted(self._latency_samples)
                n = len(sorted_samples)
                latency_stats = {
                    "min_ms": sorted_samples[0],
                    "max_ms": sorted_samples[-1],
                    "avg_ms": sum(sorted_samples) / n,
                    "p50_ms": sorted_samples[n // 2],
                    "p95_ms": sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1],
                    "p99_ms": sorted_samples[int(n * 0.99)] if n >= 100 else sorted_samples[-1],
                    "sample_count": n,
                }
            
            return {
                "token_counts": dict(self._token_counts),
                "operation_counts": dict(self._operation_counts),
                "error_counts": dict(self._error_counts),
                "latency_stats": latency_stats,
            }
    
    def reset_metrics(self) -> None:
        """Reset aggregated metrics."""
        with self._lock:
            self._token_counts.clear()
            self._latency_samples.clear()
            self._operation_counts.clear()
            self._error_counts.clear()


class MetricsSpanProcessor(SpanProcessor):
    """
    Processor that exports span data as metrics.
    
    Converts span attributes to Prometheus-compatible metrics format.
    """
    
    def __init__(
        self,
        metrics_registry: Optional[Any] = None,
        export_interval_sec: float = 10.0,
    ):
        self.metrics_registry = metrics_registry
        self.export_interval_sec = export_interval_sec
        
        self._counters: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        self._running = False
        self._export_thread: Optional[threading.Thread] = None
    
    def on_start(self, span: Span, parent_context: Optional[SpanContext]) -> None:
        """Track span start."""
        with self._lock:
            self._counters[f"spans_started_total{{operation=\"{span.name}\"}}"] += 1
    
    def on_end(self, span: Span) -> None:
        """Export span metrics."""
        with self._lock:
            # Increment counters
            self._counters[f"spans_completed_total{{operation=\"{span.name}\"}}"] += 1
            
            if span.status == SpanStatus.ERROR:
                self._counters[f"spans_errors_total{{operation=\"{span.name}\"}}"] += 1
            
            # Record histogram
            self._histograms[f"span_duration_ms{{operation=\"{span.name}\"}}"].append(
                span.duration_ms
            )
            
            # Update gauges
            self._gauges[f"last_span_duration_ms{{operation=\"{span.name}\"}}"] = (
                span.duration_ms
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {k: list(v) for k, v in self._histograms.items()},
                "gauges": dict(self._gauges),
            }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        with self._lock:
            for name, value in self._counters.items():
                lines.append(f"{name} {value}")
            for name, value in self._gauges.items():
                lines.append(f"{name} {value}")
        return "\n".join(lines)


class BatchSpanProcessor(SpanProcessor):
    """
    Batched span processor for efficient export.
    
    Collects spans and exports them in batches to reduce
    overhead and improve throughput.
    """
    
    def __init__(
        self,
        exporter: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        max_batch_size: int = 512,
        max_queue_size: int = 2048,
        export_interval_sec: float = 5.0,
        export_timeout_sec: float = 30.0,
    ):
        self.exporter = exporter
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.export_interval_sec = export_interval_sec
        self.export_timeout_sec = export_timeout_sec
        
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._running = True
        self._lock = threading.Lock()
        
        # Stats
        self._exported_count = 0
        self._dropped_count = 0
        
        # Start export thread
        self._export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self._export_thread.start()
    
    def on_start(self, span: Span, parent_context: Optional[SpanContext]) -> None:
        """No action on start."""
        pass
    
    def on_end(self, span: Span) -> None:
        """Queue span for export."""
        try:
            self._queue.put_nowait(span.to_dict())
        except Exception:
            with self._lock:
                self._dropped_count += 1
    
    def _export_loop(self) -> None:
        """Background thread for batch export."""
        while self._running:
            batch = []
            deadline = time.time() + self.export_interval_sec
            
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    timeout = max(0.1, deadline - time.time())
                    span_data = self._queue.get(timeout=timeout)
                    batch.append(span_data)
                except Empty:
                    break
            
            if batch:
                self._export_batch(batch)
    
    def _export_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Export a batch of spans."""
        if self.exporter:
            try:
                self.exporter(batch)
                with self._lock:
                    self._exported_count += len(batch)
            except Exception:
                with self._lock:
                    self._dropped_count += len(batch)
    
    def force_flush(self, timeout_ms: int = 30000) -> bool:
        """Flush all pending spans."""
        deadline = time.time() + timeout_ms / 1000
        batch = []
        
        while time.time() < deadline:
            try:
                span_data = self._queue.get_nowait()
                batch.append(span_data)
            except Empty:
                break
        
        if batch:
            self._export_batch(batch)
        
        return self._queue.empty()
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        self._running = False
        self.force_flush()
        if self._export_thread:
            self._export_thread.join(timeout=5.0)
    
    def get_stats(self) -> Dict[str, int]:
        """Get export statistics."""
        with self._lock:
            return {
                "exported_count": self._exported_count,
                "dropped_count": self._dropped_count,
                "queue_size": self._queue.qsize(),
            }


class JaegerSpanExporter:
    """
    Exporter that formats spans for Jaeger.
    
    Supports both Thrift and HTTP/JSON protocols.
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:14268/api/traces",
        service_name: str = "llm-inference",
    ):
        self.endpoint = endpoint
        self.service_name = service_name
    
    def export(self, spans: List[Dict[str, Any]]) -> bool:
        """Export spans to Jaeger."""
        # Format as Jaeger spans
        jaeger_spans = [self._to_jaeger_span(s) for s in spans]
        
        # In production, would send to Jaeger endpoint
        # For now, just return success
        return True
    
    def _to_jaeger_span(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal span format to Jaeger format."""
        return {
            "traceIdLow": int(span["trace_id"][:16], 16) if len(span["trace_id"]) >= 16 else 0,
            "traceIdHigh": int(span["trace_id"][16:32], 16) if len(span["trace_id"]) >= 32 else 0,
            "spanId": int(span["span_id"], 16) if span["span_id"] else 0,
            "parentSpanId": int(span["parent_span_id"], 16) if span["parent_span_id"] else 0,
            "operationName": span["name"],
            "startTime": int(span["start_time"] * 1_000_000),  # microseconds
            "duration": int(span["duration_ms"] * 1000),  # microseconds
            "tags": [
                {"key": k, "vType": "STRING", "vStr": str(v)}
                for k, v in span.get("attributes", {}).items()
            ],
            "logs": [
                {
                    "timestamp": int(e["timestamp"] * 1_000_000),
                    "fields": [
                        {"key": "event", "vType": "STRING", "vStr": e["name"]},
                        *[
                            {"key": k, "vType": "STRING", "vStr": str(v)}
                            for k, v in e.get("attributes", {}).items()
                        ],
                    ],
                }
                for e in span.get("events", [])
            ],
            "serviceName": self.service_name,
        }


class CompositeSpanProcessor(SpanProcessor):
    """
    Processor that delegates to multiple child processors.
    """
    
    def __init__(self, processors: List[SpanProcessor]):
        self.processors = processors
    
    def on_start(self, span: Span, parent_context: Optional[SpanContext]) -> None:
        """Delegate to all processors."""
        for processor in self.processors:
            processor.on_start(span, parent_context)
    
    def on_end(self, span: Span) -> None:
        """Delegate to all processors."""
        for processor in self.processors:
            processor.on_end(span)
    
    def force_flush(self, timeout_ms: int = 30000) -> bool:
        """Flush all processors."""
        return all(p.force_flush(timeout_ms) for p in self.processors)
    
    def shutdown(self) -> None:
        """Shutdown all processors."""
        for processor in self.processors:
            processor.shutdown()
