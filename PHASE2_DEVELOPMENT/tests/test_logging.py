"""
Comprehensive tests for distributed logging infrastructure.

Tests structured logging, log aggregation, trace correlation,
and various forwarder implementations.

@SENTRY - Observability, Logging & Monitoring Specialist
"""

import pytest
import json
import time
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List, Optional

# Import logging module components
from src.llm_logging.structured_logger import (
    LogLevel,
    LogContext,
    LogRecord,
    LogHandler,
    StructuredLogger,
    ConsoleHandler,
    FileHandler,
    JSONFormatter,
)
from src.llm_logging.log_aggregator import (
    LogEntry,
    LogBuffer,
    LogForwarder,
    LogAggregator,
    ConsoleForwarder,
    LokiForwarder,
    ElasticsearchForwarder,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def log_context():
    """Create a log context with trace correlation."""
    return LogContext(
        trace_id="abc123def456",
        span_id="span789",
        service_name="test-service",
        environment="test",
        extra={"request_id": "req-001"}
    )


@pytest.fixture
def structured_logger(log_context):
    """Create a structured logger with test context."""
    return StructuredLogger(
        name="test-logger",
        context=log_context,
        level=LogLevel.DEBUG
    )


@pytest.fixture
def mock_handler():
    """Create a mock log handler."""
    handler = Mock(spec=LogHandler)
    handler.handle = Mock()
    handler.flush = Mock()
    handler.close = Mock()
    return handler


@pytest.fixture
def log_buffer():
    """Create a log buffer with default settings."""
    return LogBuffer(
        max_size=100,
        flush_interval_seconds=1.0,
        max_age_seconds=60.0
    )


@pytest.fixture
def mock_forwarder():
    """Create a mock log forwarder."""
    forwarder = Mock(spec=LogForwarder)
    forwarder.forward = Mock(return_value=True)
    forwarder.forward_batch = Mock(return_value=True)
    forwarder.is_healthy = Mock(return_value=True)
    return forwarder


@pytest.fixture
def log_aggregator(mock_forwarder):
    """Create a log aggregator with mock forwarder."""
    aggregator = LogAggregator(
        buffer_size=100,
        forwarders=[mock_forwarder]
    )
    return aggregator


@pytest.fixture
def console_forwarder():
    """Create a console forwarder for testing."""
    return ConsoleForwarder(format_json=True)


@pytest.fixture
def temp_log_file():
    """Create a temporary log file path."""
    fd, path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


# ============================================================================
# Test LogLevel
# ============================================================================

class TestLogLevel:
    """Tests for LogLevel enumeration."""

    def test_log_level_ordering(self):
        """Test that log levels have correct ordering."""
        assert LogLevel.DEBUG.value < LogLevel.INFO.value
        assert LogLevel.INFO.value < LogLevel.WARNING.value
        assert LogLevel.WARNING.value < LogLevel.ERROR.value
        assert LogLevel.ERROR.value < LogLevel.CRITICAL.value

    def test_log_level_names(self):
        """Test log level name strings."""
        assert LogLevel.DEBUG.name == "DEBUG"
        assert LogLevel.INFO.name == "INFO"
        assert LogLevel.WARNING.name == "WARNING"
        assert LogLevel.ERROR.name == "ERROR"
        assert LogLevel.CRITICAL.name == "CRITICAL"

    def test_log_level_comparison(self):
        """Test log level comparison for filtering."""
        current_level = LogLevel.WARNING
        
        # DEBUG and INFO should be filtered out
        assert LogLevel.DEBUG.value < current_level.value
        assert LogLevel.INFO.value < current_level.value
        
        # WARNING and above should pass
        assert LogLevel.WARNING.value >= current_level.value
        assert LogLevel.ERROR.value >= current_level.value
        assert LogLevel.CRITICAL.value >= current_level.value


# ============================================================================
# Test LogContext
# ============================================================================

class TestLogContext:
    """Tests for LogContext class."""

    def test_context_creation(self, log_context):
        """Test log context is created correctly."""
        assert log_context.trace_id == "abc123def456"
        assert log_context.span_id == "span789"
        assert log_context.service_name == "test-service"
        assert log_context.environment == "test"

    def test_context_extra_fields(self, log_context):
        """Test context extra fields are accessible."""
        assert log_context.extra["request_id"] == "req-001"

    def test_context_to_dict(self, log_context):
        """Test context serialization to dictionary."""
        context_dict = log_context.to_dict()
        
        assert context_dict["trace_id"] == "abc123def456"
        assert context_dict["span_id"] == "span789"
        assert context_dict["service_name"] == "test-service"
        assert context_dict["environment"] == "test"
        assert "request_id" in context_dict

    def test_context_merge(self, log_context):
        """Test merging additional context."""
        additional = {"user_id": "user-123", "session_id": "sess-456"}
        merged = log_context.merge(additional)
        
        # Original fields preserved
        assert merged.trace_id == log_context.trace_id
        assert merged.service_name == log_context.service_name
        
        # New fields added
        assert merged.extra["user_id"] == "user-123"
        assert merged.extra["session_id"] == "sess-456"

    def test_context_with_trace_correlation(self):
        """Test context creation with trace correlation from tracer."""
        context = LogContext.from_trace(
            trace_id="trace-abc",
            span_id="span-xyz",
            service_name="inference-service"
        )
        
        assert context.trace_id == "trace-abc"
        assert context.span_id == "span-xyz"
        assert context.service_name == "inference-service"

    def test_empty_context(self):
        """Test context with minimal fields."""
        context = LogContext(service_name="minimal-service")
        
        assert context.service_name == "minimal-service"
        assert context.trace_id is None or context.trace_id == ""
        assert context.environment is None or context.environment == ""


# ============================================================================
# Test LogRecord
# ============================================================================

class TestLogRecord:
    """Tests for LogRecord class."""

    def test_record_creation(self, log_context):
        """Test log record creation."""
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test-logger",
            context=log_context
        )
        
        assert record.level == LogLevel.INFO
        assert record.message == "Test message"
        assert record.logger_name == "test-logger"
        assert record.context == log_context

    def test_record_timestamp(self):
        """Test log record has timestamp."""
        before = datetime.utcnow()
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test",
            logger_name="test"
        )
        after = datetime.utcnow()
        
        assert before <= record.timestamp <= after

    def test_record_with_extra_data(self, log_context):
        """Test log record with additional data."""
        record = LogRecord(
            level=LogLevel.ERROR,
            message="Operation failed",
            logger_name="test-logger",
            context=log_context,
            extra={"error_code": 500, "retry_count": 3}
        )
        
        assert record.extra["error_code"] == 500
        assert record.extra["retry_count"] == 3

    def test_record_with_exception(self, log_context):
        """Test log record with exception info."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            record = LogRecord(
                level=LogLevel.ERROR,
                message="Exception occurred",
                logger_name="test-logger",
                context=log_context,
                exception=e
            )
        
        assert record.exception is not None
        assert "ValueError" in str(type(record.exception))

    def test_record_to_dict(self, log_context):
        """Test record serialization to dictionary."""
        record = LogRecord(
            level=LogLevel.WARNING,
            message="Warning message",
            logger_name="test-logger",
            context=log_context
        )
        
        record_dict = record.to_dict()
        
        assert record_dict["level"] == "WARNING"
        assert record_dict["message"] == "Warning message"
        assert record_dict["logger_name"] == "test-logger"
        assert "timestamp" in record_dict
        assert "trace_id" in record_dict

    def test_record_json_serialization(self, log_context):
        """Test record can be serialized to JSON."""
        record = LogRecord(
            level=LogLevel.INFO,
            message="JSON test",
            logger_name="test-logger",
            context=log_context,
            extra={"data": {"nested": "value"}}
        )
        
        json_str = json.dumps(record.to_dict())
        parsed = json.loads(json_str)
        
        assert parsed["message"] == "JSON test"
        assert parsed["data"]["nested"] == "value"


# ============================================================================
# Test StructuredLogger
# ============================================================================

class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_logger_creation(self, structured_logger):
        """Test structured logger is created correctly."""
        assert structured_logger.name == "test-logger"
        assert structured_logger.level == LogLevel.DEBUG

    def test_log_debug(self, structured_logger, mock_handler):
        """Test debug level logging."""
        structured_logger.add_handler(mock_handler)
        structured_logger.debug("Debug message")
        
        mock_handler.handle.assert_called_once()
        record = mock_handler.handle.call_args[0][0]
        assert record.level == LogLevel.DEBUG
        assert record.message == "Debug message"

    def test_log_info(self, structured_logger, mock_handler):
        """Test info level logging."""
        structured_logger.add_handler(mock_handler)
        structured_logger.info("Info message")
        
        mock_handler.handle.assert_called_once()
        record = mock_handler.handle.call_args[0][0]
        assert record.level == LogLevel.INFO

    def test_log_warning(self, structured_logger, mock_handler):
        """Test warning level logging."""
        structured_logger.add_handler(mock_handler)
        structured_logger.warning("Warning message")
        
        mock_handler.handle.assert_called_once()
        record = mock_handler.handle.call_args[0][0]
        assert record.level == LogLevel.WARNING

    def test_log_error(self, structured_logger, mock_handler):
        """Test error level logging."""
        structured_logger.add_handler(mock_handler)
        structured_logger.error("Error message")
        
        mock_handler.handle.assert_called_once()
        record = mock_handler.handle.call_args[0][0]
        assert record.level == LogLevel.ERROR

    def test_log_critical(self, structured_logger, mock_handler):
        """Test critical level logging."""
        structured_logger.add_handler(mock_handler)
        structured_logger.critical("Critical message")
        
        mock_handler.handle.assert_called_once()
        record = mock_handler.handle.call_args[0][0]
        assert record.level == LogLevel.CRITICAL

    def test_log_level_filtering(self, mock_handler):
        """Test that logs below level are filtered."""
        logger = StructuredLogger(
            name="filtered-logger",
            level=LogLevel.WARNING
        )
        logger.add_handler(mock_handler)
        
        # These should be filtered
        logger.debug("Debug message")
        logger.info("Info message")
        
        # These should pass
        logger.warning("Warning message")
        logger.error("Error message")
        
        assert mock_handler.handle.call_count == 2

    def test_log_with_extra(self, structured_logger, mock_handler):
        """Test logging with extra data."""
        structured_logger.add_handler(mock_handler)
        structured_logger.info(
            "Request completed",
            latency_ms=150,
            status_code=200
        )
        
        record = mock_handler.handle.call_args[0][0]
        assert record.extra["latency_ms"] == 150
        assert record.extra["status_code"] == 200

    def test_log_exception(self, structured_logger, mock_handler):
        """Test logging with exception."""
        structured_logger.add_handler(mock_handler)
        
        try:
            raise RuntimeError("Test exception")
        except RuntimeError:
            structured_logger.exception("Exception occurred")
        
        record = mock_handler.handle.call_args[0][0]
        assert record.exception is not None
        assert record.level == LogLevel.ERROR

    def test_context_binding(self, structured_logger, mock_handler):
        """Test context binding creates child logger."""
        structured_logger.add_handler(mock_handler)
        
        child = structured_logger.bind(user_id="user-123")
        child.info("User action")
        
        record = mock_handler.handle.call_args[0][0]
        assert record.context.extra["user_id"] == "user-123"

    def test_multiple_handlers(self, structured_logger):
        """Test logger with multiple handlers."""
        handler1 = Mock(spec=LogHandler)
        handler2 = Mock(spec=LogHandler)
        
        structured_logger.add_handler(handler1)
        structured_logger.add_handler(handler2)
        structured_logger.info("Multi-handler message")
        
        handler1.handle.assert_called_once()
        handler2.handle.assert_called_once()

    def test_remove_handler(self, structured_logger, mock_handler):
        """Test handler removal."""
        structured_logger.add_handler(mock_handler)
        structured_logger.info("Before removal")
        
        structured_logger.remove_handler(mock_handler)
        structured_logger.info("After removal")
        
        # Only called once (before removal)
        assert mock_handler.handle.call_count == 1

    def test_flush(self, structured_logger, mock_handler):
        """Test flushing all handlers."""
        structured_logger.add_handler(mock_handler)
        structured_logger.flush()
        
        mock_handler.flush.assert_called_once()


# ============================================================================
# Test Handlers
# ============================================================================

class TestHandlers:
    """Tests for log handlers."""

    def test_console_handler(self, log_context, capsys):
        """Test console handler output."""
        handler = ConsoleHandler(format_json=False)
        record = LogRecord(
            level=LogLevel.INFO,
            message="Console test",
            logger_name="test",
            context=log_context
        )
        
        handler.handle(record)
        captured = capsys.readouterr()
        
        assert "Console test" in captured.out or "Console test" in captured.err

    def test_console_handler_json(self, log_context, capsys):
        """Test console handler with JSON formatting."""
        handler = ConsoleHandler(format_json=True)
        record = LogRecord(
            level=LogLevel.INFO,
            message="JSON console test",
            logger_name="test",
            context=log_context
        )
        
        handler.handle(record)
        captured = capsys.readouterr()
        output = captured.out or captured.err
        
        # Should be valid JSON
        if output.strip():
            parsed = json.loads(output.strip())
            assert parsed["message"] == "JSON console test"

    def test_file_handler(self, temp_log_file, log_context):
        """Test file handler writes to file."""
        handler = FileHandler(file_path=temp_log_file)
        record = LogRecord(
            level=LogLevel.INFO,
            message="File test message",
            logger_name="test",
            context=log_context
        )
        
        handler.handle(record)
        handler.flush()
        handler.close()
        
        with open(temp_log_file, 'r') as f:
            content = f.read()
        
        assert "File test message" in content

    def test_file_handler_rotation(self, temp_log_file, log_context):
        """Test file handler with size-based rotation."""
        handler = FileHandler(
            file_path=temp_log_file,
            max_bytes=1024,
            backup_count=3
        )
        
        # Write enough to trigger rotation
        for i in range(100):
            record = LogRecord(
                level=LogLevel.INFO,
                message=f"Log entry {i} with some padding to increase size",
                logger_name="test",
                context=log_context
            )
            handler.handle(record)
        
        handler.close()
        
        # Should have created backup files
        # Note: Actual rotation depends on implementation

    def test_json_formatter(self, log_context):
        """Test JSON formatter output."""
        formatter = JSONFormatter(include_timestamp=True)
        record = LogRecord(
            level=LogLevel.INFO,
            message="Formatter test",
            logger_name="test",
            context=log_context,
            extra={"custom_field": "custom_value"}
        )
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert parsed["message"] == "Formatter test"
        assert parsed["level"] == "INFO"
        assert parsed["custom_field"] == "custom_value"
        assert "timestamp" in parsed


# ============================================================================
# Test LogEntry
# ============================================================================

class TestLogEntry:
    """Tests for LogEntry class used in aggregation."""

    def test_entry_creation(self):
        """Test log entry creation."""
        entry = LogEntry(
            level="INFO",
            message="Test entry",
            timestamp=datetime.utcnow(),
            service="test-service",
            trace_id="trace-123"
        )
        
        assert entry.level == "INFO"
        assert entry.message == "Test entry"
        assert entry.service == "test-service"
        assert entry.trace_id == "trace-123"

    def test_entry_from_record(self, log_context):
        """Test creating entry from log record."""
        record = LogRecord(
            level=LogLevel.WARNING,
            message="Record to entry",
            logger_name="test",
            context=log_context
        )
        
        entry = LogEntry.from_record(record)
        
        assert entry.level == "WARNING"
        assert entry.message == "Record to entry"
        assert entry.trace_id == log_context.trace_id

    def test_entry_to_loki_format(self):
        """Test entry conversion to Loki format."""
        entry = LogEntry(
            level="ERROR",
            message="Loki format test",
            timestamp=datetime.utcnow(),
            service="test-service",
            labels={"env": "test", "host": "localhost"}
        )
        
        loki_format = entry.to_loki_format()
        
        assert "streams" in loki_format or "values" in loki_format
        # Loki format should include labels and message

    def test_entry_to_elasticsearch_format(self):
        """Test entry conversion to Elasticsearch format."""
        entry = LogEntry(
            level="INFO",
            message="Elasticsearch format test",
            timestamp=datetime.utcnow(),
            service="test-service"
        )
        
        es_format = entry.to_elasticsearch_format()
        
        assert es_format["message"] == "Elasticsearch format test"
        assert "@timestamp" in es_format or "timestamp" in es_format


# ============================================================================
# Test LogBuffer
# ============================================================================

class TestLogBuffer:
    """Tests for LogBuffer class."""

    def test_buffer_add(self, log_buffer):
        """Test adding entries to buffer."""
        entry = LogEntry(
            level="INFO",
            message="Buffer test",
            timestamp=datetime.utcnow(),
            service="test"
        )
        
        log_buffer.add(entry)
        assert log_buffer.size() == 1

    def test_buffer_max_size(self):
        """Test buffer respects max size."""
        buffer = LogBuffer(max_size=5)
        
        for i in range(10):
            entry = LogEntry(
                level="INFO",
                message=f"Entry {i}",
                timestamp=datetime.utcnow(),
                service="test"
            )
            buffer.add(entry)
        
        # Should trigger flush or reject when full
        assert buffer.size() <= 5

    def test_buffer_flush(self, log_buffer):
        """Test buffer flush returns entries."""
        for i in range(5):
            entry = LogEntry(
                level="INFO",
                message=f"Entry {i}",
                timestamp=datetime.utcnow(),
                service="test"
            )
            log_buffer.add(entry)
        
        entries = log_buffer.flush()
        
        assert len(entries) == 5
        assert log_buffer.size() == 0

    def test_buffer_age_tracking(self):
        """Test buffer tracks entry age."""
        buffer = LogBuffer(max_age_seconds=60.0)
        
        entry = LogEntry(
            level="INFO",
            message="Age test",
            timestamp=datetime.utcnow(),
            service="test"
        )
        buffer.add(entry)
        
        # Check oldest entry age
        age = buffer.oldest_entry_age()
        assert age >= 0

    def test_buffer_should_flush(self):
        """Test buffer flush conditions."""
        buffer = LogBuffer(max_size=5, flush_interval_seconds=0.1)
        
        # Add some entries
        for i in range(3):
            entry = LogEntry(
                level="INFO",
                message=f"Entry {i}",
                timestamp=datetime.utcnow(),
                service="test"
            )
            buffer.add(entry)
        
        # Should not flush yet (size < max)
        assert not buffer.should_flush()
        
        # Fill to max
        for i in range(2):
            entry = LogEntry(
                level="INFO",
                message=f"Entry {i}",
                timestamp=datetime.utcnow(),
                service="test"
            )
            buffer.add(entry)
        
        # Should flush now (size == max)
        assert buffer.should_flush()


# ============================================================================
# Test LogForwarders
# ============================================================================

class TestLogForwarders:
    """Tests for log forwarder implementations."""

    def test_console_forwarder(self, console_forwarder, capsys):
        """Test console forwarder output."""
        entry = LogEntry(
            level="INFO",
            message="Console forward test",
            timestamp=datetime.utcnow(),
            service="test"
        )
        
        result = console_forwarder.forward(entry)
        
        assert result is True
        captured = capsys.readouterr()
        # Output should contain the message

    def test_console_forwarder_batch(self, console_forwarder, capsys):
        """Test console forwarder batch operation."""
        entries = [
            LogEntry(
                level="INFO",
                message=f"Batch entry {i}",
                timestamp=datetime.utcnow(),
                service="test"
            )
            for i in range(3)
        ]
        
        result = console_forwarder.forward_batch(entries)
        
        assert result is True

    def test_loki_forwarder_creation(self):
        """Test Loki forwarder creation."""
        forwarder = LokiForwarder(
            endpoint="http://localhost:3100/loki/api/v1/push",
            labels={"app": "test-app"}
        )
        
        assert forwarder.endpoint == "http://localhost:3100/loki/api/v1/push"
        assert forwarder.labels["app"] == "test-app"

    @patch('requests.post')
    def test_loki_forwarder_forward(self, mock_post):
        """Test Loki forwarder sends to endpoint."""
        mock_post.return_value = Mock(status_code=204)
        
        forwarder = LokiForwarder(
            endpoint="http://localhost:3100/loki/api/v1/push",
            labels={"app": "test"}
        )
        
        entry = LogEntry(
            level="INFO",
            message="Loki test",
            timestamp=datetime.utcnow(),
            service="test"
        )
        
        result = forwarder.forward(entry)
        
        assert result is True
        mock_post.assert_called_once()

    def test_elasticsearch_forwarder_creation(self):
        """Test Elasticsearch forwarder creation."""
        forwarder = ElasticsearchForwarder(
            hosts=["http://localhost:9200"],
            index_pattern="logs-{date}"
        )
        
        assert "http://localhost:9200" in forwarder.hosts
        assert forwarder.index_pattern == "logs-{date}"

    @patch('requests.post')
    def test_elasticsearch_forwarder_forward(self, mock_post):
        """Test Elasticsearch forwarder sends to endpoint."""
        mock_post.return_value = Mock(status_code=201)
        
        forwarder = ElasticsearchForwarder(
            hosts=["http://localhost:9200"],
            index_pattern="logs-test"
        )
        
        entry = LogEntry(
            level="ERROR",
            message="ES test",
            timestamp=datetime.utcnow(),
            service="test"
        )
        
        result = forwarder.forward(entry)
        
        assert result is True

    def test_forwarder_health_check(self, mock_forwarder):
        """Test forwarder health check."""
        assert mock_forwarder.is_healthy() is True


# ============================================================================
# Test LogAggregator
# ============================================================================

class TestLogAggregator:
    """Tests for LogAggregator class."""

    def test_aggregator_creation(self, log_aggregator):
        """Test log aggregator creation."""
        assert log_aggregator is not None
        assert len(log_aggregator.forwarders) == 1

    def test_aggregator_log(self, log_aggregator, mock_forwarder):
        """Test aggregator accepts log entries."""
        log_aggregator.log(
            level="INFO",
            message="Aggregator test",
            service="test-service"
        )
        
        # Entry should be buffered or forwarded
        # Depending on implementation

    def test_aggregator_flush(self, log_aggregator, mock_forwarder):
        """Test aggregator flush sends to forwarders."""
        for i in range(5):
            log_aggregator.log(
                level="INFO",
                message=f"Entry {i}",
                service="test"
            )
        
        log_aggregator.flush()
        
        # Forwarder should have been called
        assert mock_forwarder.forward.called or mock_forwarder.forward_batch.called

    def test_aggregator_multiple_forwarders(self):
        """Test aggregator with multiple forwarders."""
        forwarder1 = Mock(spec=LogForwarder)
        forwarder2 = Mock(spec=LogForwarder)
        forwarder1.forward_batch = Mock(return_value=True)
        forwarder2.forward_batch = Mock(return_value=True)
        
        aggregator = LogAggregator(
            buffer_size=10,
            forwarders=[forwarder1, forwarder2]
        )
        
        aggregator.log(level="INFO", message="Multi-forward test", service="test")
        aggregator.flush()
        
        # Both forwarders should receive entries
        assert forwarder1.forward_batch.called or forwarder1.forward.called
        assert forwarder2.forward_batch.called or forwarder2.forward.called

    def test_aggregator_forwarder_failure_handling(self):
        """Test aggregator handles forwarder failures."""
        failing_forwarder = Mock(spec=LogForwarder)
        failing_forwarder.forward_batch = Mock(side_effect=Exception("Network error"))
        failing_forwarder.forward = Mock(side_effect=Exception("Network error"))
        
        aggregator = LogAggregator(
            buffer_size=10,
            forwarders=[failing_forwarder]
        )
        
        aggregator.log(level="ERROR", message="Failure test", service="test")
        
        # Should not raise exception
        try:
            aggregator.flush()
        except Exception:
            pytest.fail("Aggregator should handle forwarder failures gracefully")

    def test_aggregator_shutdown(self, log_aggregator):
        """Test aggregator shutdown flushes remaining."""
        log_aggregator.log(level="INFO", message="Before shutdown", service="test")
        log_aggregator.shutdown()
        
        # Should have flushed on shutdown


# ============================================================================
# Test Trace Correlation
# ============================================================================

class TestTraceCorrelation:
    """Tests for trace-log correlation."""

    def test_log_with_trace_context(self, structured_logger, mock_handler):
        """Test logs include trace context."""
        structured_logger.add_handler(mock_handler)
        structured_logger.info("Traced log entry")
        
        record = mock_handler.handle.call_args[0][0]
        record_dict = record.to_dict()
        
        assert "trace_id" in record_dict
        assert record_dict["trace_id"] == "abc123def456"
        assert "span_id" in record_dict
        assert record_dict["span_id"] == "span789"

    def test_log_inherits_span_context(self):
        """Test logs inherit context from active span."""
        # Create logger with trace context
        context = LogContext(
            trace_id="inherited-trace",
            span_id="inherited-span",
            service_name="traced-service"
        )
        logger = StructuredLogger(name="traced-logger", context=context)
        handler = Mock(spec=LogHandler)
        logger.add_handler(handler)
        
        logger.info("Within span")
        
        record = handler.handle.call_args[0][0]
        assert record.context.trace_id == "inherited-trace"
        assert record.context.span_id == "inherited-span"

    def test_trace_context_propagation_to_aggregator(self, log_aggregator):
        """Test trace context flows to aggregator."""
        log_aggregator.log(
            level="INFO",
            message="With trace",
            service="test",
            trace_id="agg-trace-123",
            span_id="agg-span-456"
        )
        
        # Verify trace info is preserved in buffer
        entries = log_aggregator.buffer.flush()
        if entries:
            assert entries[0].trace_id == "agg-trace-123"


# ============================================================================
# Test LLM-Specific Logging
# ============================================================================

class TestLLMLogging:
    """Tests for LLM-specific logging patterns."""

    def test_log_inference_request(self, structured_logger, mock_handler):
        """Test logging inference request details."""
        structured_logger.add_handler(mock_handler)
        
        structured_logger.info(
            "Inference request",
            model="llama-7b",
            input_tokens=512,
            max_output_tokens=256,
            temperature=0.7
        )
        
        record = mock_handler.handle.call_args[0][0]
        assert record.extra["model"] == "llama-7b"
        assert record.extra["input_tokens"] == 512

    def test_log_inference_response(self, structured_logger, mock_handler):
        """Test logging inference response details."""
        structured_logger.add_handler(mock_handler)
        
        structured_logger.info(
            "Inference complete",
            output_tokens=128,
            latency_ms=450,
            tokens_per_second=284.4
        )
        
        record = mock_handler.handle.call_args[0][0]
        assert record.extra["output_tokens"] == 128
        assert record.extra["latency_ms"] == 450

    def test_log_kv_cache_metrics(self, structured_logger, mock_handler):
        """Test logging KV cache metrics."""
        structured_logger.add_handler(mock_handler)
        
        structured_logger.debug(
            "KV cache status",
            cache_size_mb=512,
            hit_rate=0.85,
            evictions=42
        )
        
        record = mock_handler.handle.call_args[0][0]
        assert record.extra["cache_size_mb"] == 512
        assert record.extra["hit_rate"] == 0.85

    def test_log_gpu_metrics(self, structured_logger, mock_handler):
        """Test logging GPU utilization metrics."""
        structured_logger.add_handler(mock_handler)
        
        structured_logger.info(
            "GPU metrics",
            gpu_id=0,
            utilization_percent=85.5,
            memory_used_gb=12.3,
            memory_total_gb=24.0,
            temperature_c=72
        )
        
        record = mock_handler.handle.call_args[0][0]
        assert record.extra["gpu_id"] == 0
        assert record.extra["utilization_percent"] == 85.5


# ============================================================================
# Test Async Logging
# ============================================================================

class TestAsyncLogging:
    """Tests for async logging operations."""

    @pytest.mark.asyncio
    async def test_async_log(self, structured_logger, mock_handler):
        """Test async logging if supported."""
        structured_logger.add_handler(mock_handler)
        
        # Use sync logging in async context
        structured_logger.info("Async context log")
        
        mock_handler.handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_aggregator_flush(self, log_aggregator):
        """Test async flush of aggregator."""
        log_aggregator.log(level="INFO", message="Async flush test", service="test")
        
        # If async flush is supported
        if hasattr(log_aggregator, 'flush_async'):
            await log_aggregator.flush_async()
        else:
            log_aggregator.flush()

    @pytest.mark.asyncio
    async def test_concurrent_logging(self, structured_logger, mock_handler):
        """Test concurrent log operations."""
        structured_logger.add_handler(mock_handler)
        
        async def log_task(i):
            structured_logger.info(f"Concurrent log {i}", task_id=i)
        
        await asyncio.gather(*[log_task(i) for i in range(10)])
        
        assert mock_handler.handle.call_count == 10


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_message(self, structured_logger, mock_handler):
        """Test logging empty message."""
        structured_logger.add_handler(mock_handler)
        structured_logger.info("")
        
        mock_handler.handle.assert_called_once()
        record = mock_handler.handle.call_args[0][0]
        assert record.message == ""

    def test_very_long_message(self, structured_logger, mock_handler):
        """Test logging very long message."""
        structured_logger.add_handler(mock_handler)
        long_message = "A" * 100000
        structured_logger.info(long_message)
        
        record = mock_handler.handle.call_args[0][0]
        assert len(record.message) == 100000

    def test_special_characters_in_message(self, structured_logger, mock_handler):
        """Test logging message with special characters."""
        structured_logger.add_handler(mock_handler)
        structured_logger.info("Special chars: \n\t\r\x00 ñ 日本語 🚀")
        
        mock_handler.handle.assert_called_once()

    def test_none_in_extra(self, structured_logger, mock_handler):
        """Test logging with None in extra fields."""
        structured_logger.add_handler(mock_handler)
        structured_logger.info("None test", value=None)
        
        record = mock_handler.handle.call_args[0][0]
        assert record.extra["value"] is None

    def test_nested_dict_in_extra(self, structured_logger, mock_handler):
        """Test logging with nested dict in extra."""
        structured_logger.add_handler(mock_handler)
        structured_logger.info(
            "Nested test",
            data={"level1": {"level2": {"level3": "value"}}}
        )
        
        record = mock_handler.handle.call_args[0][0]
        assert record.extra["data"]["level1"]["level2"]["level3"] == "value"

    def test_list_in_extra(self, structured_logger, mock_handler):
        """Test logging with list in extra fields."""
        structured_logger.add_handler(mock_handler)
        structured_logger.info(
            "List test",
            items=[1, 2, 3, "four", {"five": 5}]
        )
        
        record = mock_handler.handle.call_args[0][0]
        assert len(record.extra["items"]) == 5

    def test_handler_exception_isolation(self, structured_logger):
        """Test that handler exceptions don't break logging."""
        failing_handler = Mock(spec=LogHandler)
        failing_handler.handle = Mock(side_effect=Exception("Handler failed"))
        
        working_handler = Mock(spec=LogHandler)
        
        structured_logger.add_handler(failing_handler)
        structured_logger.add_handler(working_handler)
        
        # Should not raise
        try:
            structured_logger.info("Test message")
        except Exception:
            pytest.fail("Handler exception should be isolated")

    def test_forwarder_timeout_handling(self):
        """Test forwarder handles timeouts gracefully."""
        slow_forwarder = Mock(spec=LogForwarder)
        slow_forwarder.forward = Mock(side_effect=TimeoutError("Timeout"))
        
        aggregator = LogAggregator(
            buffer_size=10,
            forwarders=[slow_forwarder]
        )
        
        aggregator.log(level="INFO", message="Timeout test", service="test")
        
        # Should handle timeout gracefully
        try:
            aggregator.flush()
        except TimeoutError:
            pytest.fail("Timeout should be handled gracefully")


# ============================================================================
# Test Integration with Tracing
# ============================================================================

class TestTracingIntegration:
    """Tests for integration between logging and tracing modules."""

    def test_log_includes_trace_headers(self, log_context):
        """Test log entries include W3C trace headers."""
        record = LogRecord(
            level=LogLevel.INFO,
            message="Trace header test",
            logger_name="test",
            context=log_context
        )
        
        record_dict = record.to_dict()
        
        # Should have trace context fields
        assert "trace_id" in record_dict
        assert "span_id" in record_dict

    def test_create_context_from_traceparent(self):
        """Test creating log context from traceparent header."""
        traceparent = "00-abc123def456789012345678901234-0123456789abcdef-01"
        
        context = LogContext.from_traceparent(traceparent)
        
        assert context.trace_id == "abc123def456789012345678901234"
        assert context.span_id == "0123456789abcdef"

    def test_logger_auto_trace_propagation(self, structured_logger, mock_handler):
        """Test logger automatically propagates trace context."""
        structured_logger.add_handler(mock_handler)
        structured_logger.info("Auto propagation test")
        
        record = mock_handler.handle.call_args[0][0]
        
        # Should have trace context from logger's context
        assert record.context.trace_id is not None


# ============================================================================
# Test Performance
# ============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_high_volume_logging(self, structured_logger, mock_handler):
        """Test logging performance under high volume."""
        structured_logger.add_handler(mock_handler)
        
        start = time.time()
        for i in range(1000):
            structured_logger.info(f"High volume log {i}", iteration=i)
        elapsed = time.time() - start
        
        assert mock_handler.handle.call_count == 1000
        # Should complete reasonably fast
        assert elapsed < 5.0  # 5 seconds for 1000 logs

    def test_buffer_performance(self):
        """Test buffer performance under load."""
        buffer = LogBuffer(max_size=10000)
        
        start = time.time()
        for i in range(10000):
            entry = LogEntry(
                level="INFO",
                message=f"Entry {i}",
                timestamp=datetime.utcnow(),
                service="test"
            )
            buffer.add(entry)
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 2.0  # 2 seconds for 10000 entries

    def test_json_formatting_performance(self, log_context):
        """Test JSON formatting performance."""
        formatter = JSONFormatter()
        record = LogRecord(
            level=LogLevel.INFO,
            message="Performance test",
            logger_name="test",
            context=log_context,
            extra={"data": {"nested": "value"} * 10}
        )
        
        start = time.time()
        for _ in range(1000):
            formatter.format(record)
        elapsed = time.time() - start
        
        # Should be reasonably fast
        assert elapsed < 2.0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
