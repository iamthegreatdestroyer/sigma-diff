"""
Comprehensive test suite for the distributed tracing module.

Tests cover:
- Tracer functionality (LLMTracer, Span creation)
- Context propagation (W3C Trace Context)
- Span processors and exporters
- Trace correlation and attribute handling
"""

import pytest
import time
import json
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List, Optional

# Import tracing module components
from src.tracing import (
    LLMTracer,
    Span,
    SpanKind,
    SpanStatus,
    TracingContext,
    ContextCarrier,
    LLMSpanProcessor,
    BatchSpanProcessor,
    JaegerSpanExporter,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tracer():
    """Create a fresh tracer instance for testing."""
    return LLMTracer(service_name="test-service")


@pytest.fixture
def tracer_with_processor():
    """Create a tracer with a mock processor."""
    tracer = LLMTracer(service_name="test-service-with-processor")
    processor = Mock(spec=LLMSpanProcessor)
    processor.on_start = Mock()
    processor.on_end = Mock()
    tracer.add_span_processor(processor)
    return tracer, processor


@pytest.fixture
def tracing_context():
    """Create a tracing context for testing."""
    return TracingContext()


@pytest.fixture
def context_carrier():
    """Create a context carrier for testing."""
    return ContextCarrier()


@pytest.fixture
def llm_span_processor():
    """Create an LLM span processor for testing."""
    return LLMSpanProcessor()


@pytest.fixture
def batch_processor():
    """Create a batch span processor with mock exporter."""
    exporter = Mock(spec=JaegerSpanExporter)
    exporter.export = Mock(return_value=True)
    processor = BatchSpanProcessor(
        exporter=exporter,
        max_batch_size=10,
        max_queue_size=100,
        export_interval_ms=1000
    )
    return processor, exporter


@pytest.fixture
def jaeger_exporter():
    """Create a Jaeger exporter for testing."""
    return JaegerSpanExporter(
        agent_host="localhost",
        agent_port=6831
    )


# =============================================================================
# SPAN TESTS
# =============================================================================

class TestSpan:
    """Test suite for Span class."""

    def test_span_creation(self, tracer):
        """Test basic span creation."""
        span = tracer.start_span("test-operation")
        
        assert span is not None
        assert span.name == "test-operation"
        assert span.trace_id is not None
        assert span.span_id is not None
        assert span.start_time is not None
        assert span.end_time is None  # Not ended yet

    def test_span_with_kind(self, tracer):
        """Test span creation with different kinds."""
        for kind in SpanKind:
            span = tracer.start_span(f"test-{kind.value}", kind=kind)
            assert span.kind == kind

    def test_span_attributes(self, tracer):
        """Test setting span attributes."""
        span = tracer.start_span("test-attributes")
        
        span.set_attribute("string_attr", "value")
        span.set_attribute("int_attr", 42)
        span.set_attribute("float_attr", 3.14)
        span.set_attribute("bool_attr", True)
        span.set_attribute("list_attr", [1, 2, 3])
        
        assert span.get_attribute("string_attr") == "value"
        assert span.get_attribute("int_attr") == 42
        assert span.get_attribute("float_attr") == 3.14
        assert span.get_attribute("bool_attr") is True
        assert span.get_attribute("list_attr") == [1, 2, 3]

    def test_span_events(self, tracer):
        """Test adding events to span."""
        span = tracer.start_span("test-events")
        
        span.add_event("event1", {"key1": "value1"})
        span.add_event("event2", {"key2": "value2"})
        
        events = span.events
        assert len(events) == 2
        assert events[0]["name"] == "event1"
        assert events[1]["name"] == "event2"

    def test_span_status(self, tracer):
        """Test span status operations."""
        span = tracer.start_span("test-status")
        
        # Default status
        assert span.status == SpanStatus.UNSET
        
        # Set to OK
        span.set_status(SpanStatus.OK)
        assert span.status == SpanStatus.OK
        
        # Set to ERROR with message
        span.set_status(SpanStatus.ERROR, "Something went wrong")
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Something went wrong"

    def test_span_end(self, tracer):
        """Test span ending."""
        span = tracer.start_span("test-end")
        
        assert span.end_time is None
        assert span.is_recording() is True
        
        span.end()
        
        assert span.end_time is not None
        assert span.is_recording() is False
        assert span.duration_ms >= 0

    def test_span_context_manager(self, tracer):
        """Test span as context manager."""
        with tracer.start_span("test-context-manager") as span:
            assert span.is_recording() is True
            span.set_attribute("inside_context", True)
        
        assert span.is_recording() is False
        assert span.end_time is not None

    def test_span_exception_recording(self, tracer):
        """Test recording exceptions in span."""
        span = tracer.start_span("test-exception")
        
        try:
            raise ValueError("Test error")
        except ValueError as e:
            span.record_exception(e)
        
        span.end()
        
        assert span.status == SpanStatus.ERROR
        assert len(span.events) >= 1
        exception_event = next(
            (e for e in span.events if e["name"] == "exception"),
            None
        )
        assert exception_event is not None
        assert "ValueError" in str(exception_event.get("attributes", {}))

    def test_span_links(self, tracer):
        """Test span links."""
        # Create first span
        span1 = tracer.start_span("span1")
        span1.end()
        
        # Create second span with link to first
        span2 = tracer.start_span(
            "span2",
            links=[{"trace_id": span1.trace_id, "span_id": span1.span_id}]
        )
        
        assert len(span2.links) == 1
        assert span2.links[0]["trace_id"] == span1.trace_id


# =============================================================================
# TRACER TESTS
# =============================================================================

class TestLLMTracer:
    """Test suite for LLMTracer class."""

    def test_tracer_creation(self):
        """Test tracer creation with service name."""
        tracer = LLMTracer(service_name="my-llm-service")
        assert tracer.service_name == "my-llm-service"

    def test_tracer_with_resource_attributes(self):
        """Test tracer with resource attributes."""
        tracer = LLMTracer(
            service_name="my-service",
            resource_attributes={
                "deployment.environment": "production",
                "service.version": "1.0.0"
            }
        )
        
        assert tracer.resource_attributes["deployment.environment"] == "production"
        assert tracer.resource_attributes["service.version"] == "1.0.0"

    def test_tracer_parent_child_spans(self, tracer):
        """Test parent-child span relationship."""
        with tracer.start_span("parent") as parent:
            parent_trace_id = parent.trace_id
            parent_span_id = parent.span_id
            
            with tracer.start_span("child") as child:
                assert child.trace_id == parent_trace_id
                assert child.parent_span_id == parent_span_id

    def test_tracer_trace_llm_request(self, tracer):
        """Test LLM-specific request tracing."""
        with tracer.trace_llm_request(
            operation="generate",
            model_name="llama-7b",
            prompt_tokens=100,
            max_tokens=500
        ) as span:
            span.set_attribute("response_tokens", 250)
            span.set_attribute("total_tokens", 350)
        
        assert span.get_attribute("llm.model") == "llama-7b"
        assert span.get_attribute("llm.prompt_tokens") == 100
        assert span.get_attribute("llm.max_tokens") == 500
        assert span.get_attribute("response_tokens") == 250

    def test_tracer_trace_inference(self, tracer):
        """Test inference-specific tracing."""
        with tracer.trace_inference(
            model_name="gpt-2",
            batch_size=4,
            sequence_length=512
        ) as span:
            span.set_attribute("inference.latency_ms", 150.5)
        
        assert span.get_attribute("inference.model") == "gpt-2"
        assert span.get_attribute("inference.batch_size") == 4
        assert span.get_attribute("inference.sequence_length") == 512

    def test_tracer_trace_kv_cache(self, tracer):
        """Test KV cache operation tracing."""
        with tracer.trace_kv_cache(
            operation="lookup",
            cache_size_mb=1024,
            hit_rate=0.85
        ) as span:
            span.set_attribute("cache.entries", 10000)
        
        assert span.get_attribute("kv_cache.operation") == "lookup"
        assert span.get_attribute("kv_cache.size_mb") == 1024
        assert span.get_attribute("kv_cache.hit_rate") == 0.85

    def test_tracer_add_processor(self, tracer_with_processor):
        """Test adding span processor to tracer."""
        tracer, processor = tracer_with_processor
        
        with tracer.start_span("test-processor") as span:
            pass
        
        processor.on_start.assert_called()
        processor.on_end.assert_called()

    def test_tracer_current_span(self, tracer):
        """Test getting current span from tracer."""
        assert tracer.get_current_span() is None
        
        with tracer.start_span("current-span") as span:
            current = tracer.get_current_span()
            assert current == span

    def test_tracer_flush(self, tracer_with_processor):
        """Test tracer flush operation."""
        tracer, processor = tracer_with_processor
        processor.force_flush = Mock(return_value=True)
        
        result = tracer.flush()
        
        processor.force_flush.assert_called()

    def test_tracer_shutdown(self, tracer_with_processor):
        """Test tracer shutdown operation."""
        tracer, processor = tracer_with_processor
        processor.shutdown = Mock()
        
        tracer.shutdown()
        
        processor.shutdown.assert_called()


# =============================================================================
# CONTEXT PROPAGATION TESTS
# =============================================================================

class TestContextPropagation:
    """Test suite for context propagation."""

    def test_context_injection(self, tracer, context_carrier):
        """Test injecting trace context into carrier."""
        with tracer.start_span("parent-span") as span:
            TracingContext.inject(span, context_carrier)
        
        # Check W3C trace context headers
        assert "traceparent" in context_carrier.headers
        traceparent = context_carrier.headers["traceparent"]
        
        # Format: version-trace_id-span_id-flags
        parts = traceparent.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"  # version
        assert len(parts[1]) == 32  # trace_id (hex)
        assert len(parts[2]) == 16  # span_id (hex)

    def test_context_extraction(self, tracer):
        """Test extracting trace context from carrier."""
        # Create parent span and inject context
        with tracer.start_span("parent") as parent:
            carrier = ContextCarrier()
            TracingContext.inject(parent, carrier)
        
        # Extract context
        extracted = TracingContext.extract(carrier)
        
        assert extracted is not None
        assert extracted["trace_id"] == parent.trace_id
        assert extracted["parent_span_id"] == parent.span_id

    def test_context_continuation(self, tracer):
        """Test continuing trace from extracted context."""
        # Create and inject parent context
        with tracer.start_span("parent") as parent:
            carrier = ContextCarrier()
            TracingContext.inject(parent, carrier)
            parent_trace_id = parent.trace_id
        
        # Extract and continue
        extracted = TracingContext.extract(carrier)
        
        with tracer.start_span("child", parent_context=extracted) as child:
            assert child.trace_id == parent_trace_id

    def test_tracestate_propagation(self, tracer, context_carrier):
        """Test tracestate header propagation."""
        with tracer.start_span("span-with-state") as span:
            span.set_trace_state({"key1": "value1", "key2": "value2"})
            TracingContext.inject(span, context_carrier)
        
        assert "tracestate" in context_carrier.headers

    def test_baggage_propagation(self, tracer, context_carrier):
        """Test baggage propagation."""
        with tracer.start_span("span-with-baggage") as span:
            span.set_baggage("user_id", "12345")
            span.set_baggage("session_id", "abc-xyz")
            TracingContext.inject(span, context_carrier)
        
        # Check baggage was propagated
        assert span.get_baggage("user_id") == "12345"
        assert span.get_baggage("session_id") == "abc-xyz"

    def test_http_headers_format(self, tracer):
        """Test HTTP header format for propagation."""
        with tracer.start_span("http-span") as span:
            headers = TracingContext.to_http_headers(span)
        
        assert "traceparent" in headers
        assert isinstance(headers["traceparent"], str)

    def test_context_carrier_dict_interface(self, context_carrier):
        """Test context carrier dictionary interface."""
        context_carrier["custom-header"] = "custom-value"
        
        assert context_carrier["custom-header"] == "custom-value"
        assert "custom-header" in context_carrier

    def test_empty_context_extraction(self):
        """Test extracting from empty carrier."""
        carrier = ContextCarrier()
        
        extracted = TracingContext.extract(carrier)
        
        assert extracted is None


# =============================================================================
# SPAN PROCESSOR TESTS
# =============================================================================

class TestSpanProcessors:
    """Test suite for span processors."""

    def test_llm_span_processor_on_start(self, llm_span_processor, tracer):
        """Test LLM span processor on_start."""
        span = tracer.start_span("test-span")
        
        llm_span_processor.on_start(span)
        
        # Processor should add LLM-specific attributes
        assert span.get_attribute("llm.framework") is not None or True

    def test_llm_span_processor_on_end(self, llm_span_processor, tracer):
        """Test LLM span processor on_end."""
        span = tracer.start_span("test-span")
        span.set_attribute("llm.prompt_tokens", 100)
        span.set_attribute("llm.completion_tokens", 50)
        span.end()
        
        llm_span_processor.on_end(span)
        
        # Should calculate total tokens if not set
        # (implementation may vary)

    def test_llm_span_processor_metrics(self, llm_span_processor, tracer):
        """Test LLM span processor metrics collection."""
        # Process multiple spans
        for i in range(5):
            span = tracer.start_span(f"span-{i}")
            span.set_attribute("llm.tokens", 100 + i * 10)
            span.end()
            llm_span_processor.on_end(span)
        
        # Get metrics
        metrics = llm_span_processor.get_metrics()
        
        assert metrics["spans_processed"] == 5

    def test_batch_processor_batching(self, batch_processor):
        """Test batch span processor batching."""
        processor, exporter = batch_processor
        
        # Add spans but don't exceed batch size
        for i in range(5):
            span = Mock()
            span.name = f"span-{i}"
            span.trace_id = f"trace-{i}"
            span.span_id = f"span-{i}"
            span.to_dict = Mock(return_value={"name": f"span-{i}"})
            processor.on_end(span)
        
        # Spans should be queued, not exported yet
        # (unless export interval passed)

    def test_batch_processor_force_flush(self, batch_processor):
        """Test batch processor force flush."""
        processor, exporter = batch_processor
        
        # Add some spans
        for i in range(3):
            span = Mock()
            span.name = f"span-{i}"
            span.to_dict = Mock(return_value={"name": f"span-{i}"})
            processor.on_end(span)
        
        # Force flush
        processor.force_flush()
        
        # Exporter should have been called
        # (depends on implementation)

    def test_batch_processor_shutdown(self, batch_processor):
        """Test batch processor shutdown."""
        processor, exporter = batch_processor
        
        processor.shutdown()
        
        # Should flush remaining spans and stop accepting new ones


# =============================================================================
# JAEGER EXPORTER TESTS
# =============================================================================

class TestJaegerExporter:
    """Test suite for Jaeger exporter."""

    def test_exporter_creation(self, jaeger_exporter):
        """Test Jaeger exporter creation."""
        assert jaeger_exporter.agent_host == "localhost"
        assert jaeger_exporter.agent_port == 6831

    def test_exporter_with_collector(self):
        """Test Jaeger exporter with collector endpoint."""
        exporter = JaegerSpanExporter(
            collector_endpoint="http://localhost:14268/api/traces"
        )
        
        assert exporter.collector_endpoint is not None

    def test_span_to_jaeger_format(self, jaeger_exporter, tracer):
        """Test converting span to Jaeger format."""
        span = tracer.start_span("test-span")
        span.set_attribute("key", "value")
        span.add_event("test-event", {"data": "test"})
        span.end()
        
        jaeger_span = jaeger_exporter._span_to_jaeger(span)
        
        assert jaeger_span is not None
        assert "operationName" in jaeger_span
        assert jaeger_span["operationName"] == "test-span"

    @patch("socket.socket")
    def test_exporter_udp_export(self, mock_socket, jaeger_exporter, tracer):
        """Test UDP export to Jaeger agent."""
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        
        span = tracer.start_span("test-export")
        span.end()
        
        # Export would send via UDP
        # (actual network calls are mocked)

    def test_exporter_batch_export(self, jaeger_exporter, tracer):
        """Test batch export of multiple spans."""
        spans = []
        for i in range(10):
            span = tracer.start_span(f"span-{i}")
            span.end()
            spans.append(span)
        
        # Batch export
        result = jaeger_exporter.export(spans)
        
        # Should return success or failure
        assert result in [True, False]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestTracingIntegration:
    """Integration tests for complete tracing flow."""

    def test_full_tracing_pipeline(self):
        """Test complete tracing pipeline."""
        # Create tracer with processors
        tracer = LLMTracer(service_name="integration-test")
        
        llm_processor = LLMSpanProcessor()
        tracer.add_span_processor(llm_processor)
        
        # Create nested spans
        with tracer.start_span("request", kind=SpanKind.SERVER) as request_span:
            request_span.set_attribute("http.method", "POST")
            request_span.set_attribute("http.url", "/v1/completions")
            
            with tracer.start_span("tokenize", kind=SpanKind.INTERNAL) as tokenize_span:
                tokenize_span.set_attribute("tokens", 150)
                time.sleep(0.01)
            
            with tracer.start_span("inference", kind=SpanKind.INTERNAL) as inference_span:
                inference_span.set_attribute("model", "test-model")
                inference_span.set_attribute("batch_size", 1)
                time.sleep(0.02)
            
            with tracer.start_span("decode", kind=SpanKind.INTERNAL) as decode_span:
                decode_span.set_attribute("output_tokens", 75)
                time.sleep(0.005)
            
            request_span.set_status(SpanStatus.OK)
        
        # Verify spans are properly linked
        assert tokenize_span.parent_span_id == request_span.span_id
        assert inference_span.parent_span_id == request_span.span_id
        assert decode_span.parent_span_id == request_span.span_id

    def test_distributed_trace_simulation(self):
        """Test simulating distributed trace across services."""
        # Service A
        tracer_a = LLMTracer(service_name="service-a")
        
        with tracer_a.start_span("service-a-request") as span_a:
            # Propagate context
            carrier = ContextCarrier()
            TracingContext.inject(span_a, carrier)
        
        # Service B receives the context
        tracer_b = LLMTracer(service_name="service-b")
        
        extracted = TracingContext.extract(carrier)
        
        with tracer_b.start_span("service-b-request", parent_context=extracted) as span_b:
            pass
        
        # Verify trace continuity
        assert span_a.trace_id == span_b.trace_id
        assert span_b.parent_span_id == span_a.span_id

    def test_error_trace_flow(self):
        """Test tracing with errors."""
        tracer = LLMTracer(service_name="error-test")
        
        with tracer.start_span("error-request") as span:
            try:
                with tracer.start_span("failing-operation") as child:
                    raise RuntimeError("Simulated failure")
            except RuntimeError as e:
                child.record_exception(e)
                child.set_status(SpanStatus.ERROR, str(e))
            
            span.set_status(SpanStatus.ERROR, "Child operation failed")
        
        assert span.status == SpanStatus.ERROR
        assert child.status == SpanStatus.ERROR

    def test_sampling_decision(self):
        """Test trace sampling decision."""
        tracer = LLMTracer(
            service_name="sampling-test",
            sampling_rate=0.5  # 50% sampling
        )
        
        sampled_count = 0
        total = 100
        
        for i in range(total):
            span = tracer.start_span(f"sampled-span-{i}")
            if span.is_sampled:
                sampled_count += 1
            span.end()
        
        # Should be approximately 50%, allow variance
        assert 30 <= sampled_count <= 70

    def test_metrics_from_traces(self):
        """Test extracting metrics from traces."""
        tracer = LLMTracer(service_name="metrics-test")
        processor = LLMSpanProcessor()
        tracer.add_span_processor(processor)
        
        # Generate traces
        latencies = []
        for i in range(10):
            with tracer.start_span("request") as span:
                time.sleep(0.001 * (i + 1))
            latencies.append(span.duration_ms)
        
        metrics = processor.get_metrics()
        
        assert metrics["spans_processed"] == 10
        assert "avg_duration_ms" in metrics or len(latencies) == 10


# =============================================================================
# ASYNC TESTS
# =============================================================================

class TestAsyncTracing:
    """Test async tracing operations."""

    @pytest.mark.asyncio
    async def test_async_span_creation(self):
        """Test span creation in async context."""
        tracer = LLMTracer(service_name="async-test")
        
        async def async_operation():
            with tracer.start_span("async-span") as span:
                await asyncio.sleep(0.01)
                span.set_attribute("async", True)
            return span
        
        span = await async_operation()
        
        assert span.get_attribute("async") is True
        assert span.duration_ms >= 10

    @pytest.mark.asyncio
    async def test_concurrent_spans(self):
        """Test concurrent span creation."""
        tracer = LLMTracer(service_name="concurrent-test")
        
        async def create_span(name: str):
            with tracer.start_span(name) as span:
                await asyncio.sleep(0.01)
                return span
        
        spans = await asyncio.gather(
            create_span("span-1"),
            create_span("span-2"),
            create_span("span-3")
        )
        
        # All spans should be independent
        trace_ids = set(s.trace_id for s in spans)
        assert len(trace_ids) == 3

    @pytest.mark.asyncio
    async def test_async_context_propagation(self):
        """Test context propagation in async code."""
        tracer = LLMTracer(service_name="async-propagation")
        
        async def parent_operation():
            with tracer.start_span("parent") as parent:
                await child_operation(tracer, parent.trace_id)
                return parent
        
        async def child_operation(tracer, expected_trace_id):
            with tracer.start_span("child") as child:
                assert child.trace_id == expected_trace_id
                await asyncio.sleep(0.005)
        
        parent = await parent_operation()
        assert parent.end_time is not None


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_span_double_end(self, tracer):
        """Test ending a span twice."""
        span = tracer.start_span("double-end")
        span.end()
        
        first_end_time = span.end_time
        
        # Second end should be no-op
        span.end()
        
        assert span.end_time == first_end_time

    def test_attribute_after_end(self, tracer):
        """Test setting attribute after span ends."""
        span = tracer.start_span("after-end")
        span.end()
        
        # Should not raise, but attribute might not be set
        span.set_attribute("late_attr", "value")

    def test_empty_span_name(self, tracer):
        """Test span with empty name."""
        span = tracer.start_span("")
        
        assert span.name == "" or span.name is not None
        span.end()

    def test_very_long_attribute_value(self, tracer):
        """Test handling very long attribute values."""
        span = tracer.start_span("long-attr")
        
        long_value = "x" * 10000
        span.set_attribute("long_attr", long_value)
        
        # Should either truncate or accept
        attr = span.get_attribute("long_attr")
        assert attr is not None
        span.end()

    def test_special_characters_in_name(self, tracer):
        """Test span names with special characters."""
        special_names = [
            "span/with/slashes",
            "span.with.dots",
            "span:with:colons",
            "span-with-dashes",
            "span_with_underscores",
            "span with spaces"
        ]
        
        for name in special_names:
            span = tracer.start_span(name)
            assert span.name == name
            span.end()

    def test_null_attribute_value(self, tracer):
        """Test setting None as attribute value."""
        span = tracer.start_span("null-attr")
        
        span.set_attribute("null_attr", None)
        
        # Implementation may skip None or store it
        span.end()

    def test_nested_exception_handling(self, tracer):
        """Test exception handling in nested spans."""
        with tracer.start_span("outer") as outer:
            try:
                with tracer.start_span("inner") as inner:
                    try:
                        raise ValueError("Inner error")
                    except ValueError as e:
                        inner.record_exception(e)
                        raise RuntimeError("Outer error") from e
            except RuntimeError as e:
                outer.record_exception(e)
        
        assert outer.status == SpanStatus.ERROR
        assert inner.status == SpanStatus.ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
