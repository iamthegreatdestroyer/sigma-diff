"""
Structured Logger with Distributed Tracing Correlation

Provides JSON-structured logging that integrates with OpenTelemetry
tracing for full observability across distributed LLM inference.
"""

import json
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, TextIO, Union
from contextlib import contextmanager


class LogLevel(IntEnum):
    """Log severity levels following standard conventions."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    
    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """Convert string to LogLevel."""
        mapping = {
            "trace": cls.TRACE,
            "debug": cls.DEBUG,
            "info": cls.INFO,
            "warning": cls.WARNING,
            "warn": cls.WARNING,
            "error": cls.ERROR,
            "critical": cls.CRITICAL,
            "fatal": cls.CRITICAL,
        }
        return mapping.get(level.lower(), cls.INFO)


@dataclass
class LogContext:
    """Context information attached to log records."""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    service_name: str = "llm-inference"
    environment: str = "development"
    version: str = "1.0.0"
    host: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        result = {
            "service": self.service_name,
            "environment": self.environment,
            "version": self.version,
        }
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.parent_span_id:
            result["parent_span_id"] = self.parent_span_id
        if self.host:
            result["host"] = self.host
        result.update(self.extra)
        return result
    
    def with_trace(self, trace_id: str, span_id: str, parent_span_id: Optional[str] = None) -> "LogContext":
        """Create new context with trace information."""
        return LogContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            service_name=self.service_name,
            environment=self.environment,
            version=self.version,
            host=self.host,
            extra=self.extra.copy(),
        )


@dataclass
class LogRecord:
    """Structured log record with all metadata."""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    context: LogContext
    attributes: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """Serialize record to JSON string."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name,
            "level_value": int(self.level),
            "message": self.message,
            "logger": self.logger_name,
            **self.context.to_dict(),
        }
        if self.attributes:
            data["attributes"] = self.attributes
        if self.exception:
            data["exception"] = self.exception
        return json.dumps(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name,
            "level_value": int(self.level),
            "message": self.message,
            "logger": self.logger_name,
            **self.context.to_dict(),
        }
        if self.attributes:
            data["attributes"] = self.attributes
        if self.exception:
            data["exception"] = self.exception
        return data


class LogHandler(ABC):
    """Abstract base class for log handlers."""
    
    def __init__(self, level: LogLevel = LogLevel.DEBUG):
        self.level = level
        self._lock = threading.Lock()
    
    def should_handle(self, record: LogRecord) -> bool:
        """Check if this handler should process the record."""
        return record.level >= self.level
    
    @abstractmethod
    def emit(self, record: LogRecord) -> None:
        """Emit a log record."""
        pass
    
    def flush(self) -> None:
        """Flush any buffered output."""
        pass
    
    def close(self) -> None:
        """Clean up handler resources."""
        pass


class StreamHandler(LogHandler):
    """Handler that writes to a stream (stdout/stderr)."""
    
    def __init__(
        self,
        stream: TextIO = sys.stdout,
        level: LogLevel = LogLevel.DEBUG,
        json_format: bool = True,
    ):
        super().__init__(level)
        self.stream = stream
        self.json_format = json_format
    
    def emit(self, record: LogRecord) -> None:
        """Write record to stream."""
        if not self.should_handle(record):
            return
        
        with self._lock:
            try:
                if self.json_format:
                    output = record.to_json()
                else:
                    output = self._format_text(record)
                self.stream.write(output + "\n")
                self.stream.flush()
            except Exception:
                pass  # Fail silently on logging errors
    
    def _format_text(self, record: LogRecord) -> str:
        """Format record as human-readable text."""
        parts = [
            record.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"[{record.level.name:8}]",
            f"[{record.logger_name}]",
        ]
        if record.context.trace_id:
            parts.append(f"[trace:{record.context.trace_id[:8]}]")
        parts.append(record.message)
        return " ".join(parts)
    
    def flush(self) -> None:
        """Flush the stream."""
        with self._lock:
            self.stream.flush()


class FileHandler(LogHandler):
    """Handler that writes to a file with rotation support."""
    
    def __init__(
        self,
        filepath: str,
        level: LogLevel = LogLevel.DEBUG,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        json_format: bool = True,
    ):
        super().__init__(level)
        self.filepath = filepath
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.json_format = json_format
        self._file: Optional[TextIO] = None
        self._current_size = 0
        self._open_file()
    
    def _open_file(self) -> None:
        """Open the log file."""
        import os
        self._file = open(self.filepath, "a", encoding="utf-8")
        self._current_size = os.path.getsize(self.filepath) if os.path.exists(self.filepath) else 0
    
    def _rotate(self) -> None:
        """Rotate log files."""
        import os
        if self._file:
            self._file.close()
        
        # Rotate existing backups
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.filepath}.{i}"
            dst = f"{self.filepath}.{i + 1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
        
        # Rename current file
        if os.path.exists(self.filepath):
            dst = f"{self.filepath}.1"
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(self.filepath, dst)
        
        self._open_file()
    
    def emit(self, record: LogRecord) -> None:
        """Write record to file."""
        if not self.should_handle(record):
            return
        
        with self._lock:
            try:
                if self.json_format:
                    output = record.to_json()
                else:
                    output = f"{record.timestamp.isoformat()} [{record.level.name}] {record.message}"
                
                output_bytes = len(output.encode("utf-8")) + 1
                if self._current_size + output_bytes > self.max_bytes:
                    self._rotate()
                
                if self._file:
                    self._file.write(output + "\n")
                    self._file.flush()
                    self._current_size += output_bytes
            except Exception:
                pass
    
    def close(self) -> None:
        """Close the file."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class AsyncHandler(LogHandler):
    """Async handler that buffers logs and writes in background."""
    
    def __init__(
        self,
        delegate: LogHandler,
        buffer_size: int = 1000,
        flush_interval: float = 1.0,
    ):
        super().__init__(delegate.level)
        self.delegate = delegate
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._buffer: List[LogRecord] = []
        self._shutdown = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
    
    def emit(self, record: LogRecord) -> None:
        """Add record to buffer."""
        if not self.should_handle(record):
            return
        
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Flush buffer to delegate handler."""
        records = self._buffer
        self._buffer = []
        for record in records:
            self.delegate.emit(record)
        self.delegate.flush()
    
    def _flush_loop(self) -> None:
        """Background flush loop."""
        while not self._shutdown.is_set():
            self._shutdown.wait(self.flush_interval)
            with self._lock:
                if self._buffer:
                    self._flush_buffer()
    
    def flush(self) -> None:
        """Flush all pending records."""
        with self._lock:
            self._flush_buffer()
    
    def close(self) -> None:
        """Shutdown background thread."""
        self._shutdown.set()
        self._flush_thread.join(timeout=5.0)
        self.flush()
        self.delegate.close()


class StructuredLogger:
    """
    Structured logger with trace correlation.
    
    Provides JSON-formatted logging with automatic trace context
    injection for correlation with distributed tracing.
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        context: Optional[LogContext] = None,
    ):
        self.name = name
        self.level = level
        self.context = context or LogContext()
        self._handlers: List[LogHandler] = []
        self._filters: List[Callable[[LogRecord], bool]] = []
        self._lock = threading.Lock()
    
    def add_handler(self, handler: LogHandler) -> None:
        """Add a log handler."""
        with self._lock:
            self._handlers.append(handler)
    
    def remove_handler(self, handler: LogHandler) -> None:
        """Remove a log handler."""
        with self._lock:
            self._handlers.remove(handler)
    
    def add_filter(self, filter_func: Callable[[LogRecord], bool]) -> None:
        """Add a filter function."""
        with self._lock:
            self._filters.append(filter_func)
    
    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level."""
        self.level = level
    
    def set_context(self, context: LogContext) -> None:
        """Set the logger context."""
        self.context = context
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message at this level should be logged."""
        return level >= self.level
    
    def _create_record(
        self,
        level: LogLevel,
        message: str,
        attributes: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
    ) -> LogRecord:
        """Create a log record."""
        exc_info = None
        if exception:
            import traceback
            exc_info = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
            }
        
        # Try to get current trace context
        context = self.context
        try:
            from ..tracing.context import get_current_context
            trace_ctx = get_current_context()
            if trace_ctx and trace_ctx.trace_id:
                context = context.with_trace(
                    trace_ctx.trace_id,
                    trace_ctx.span_id,
                    trace_ctx.parent_span_id,
                )
        except ImportError:
            pass
        
        return LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
            logger_name=self.name,
            context=context,
            attributes=attributes or {},
            exception=exc_info,
        )
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        attributes: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        """Internal log method."""
        if not self._should_log(level):
            return
        
        record = self._create_record(level, message, attributes, exception)
        
        # Apply filters
        for filter_func in self._filters:
            if not filter_func(record):
                return
        
        # Emit to handlers
        for handler in self._handlers:
            handler.emit(record)
    
    def trace(self, message: str, **attributes: Any) -> None:
        """Log at TRACE level."""
        self._log(LogLevel.TRACE, message, attributes)
    
    def debug(self, message: str, **attributes: Any) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, attributes)
    
    def info(self, message: str, **attributes: Any) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, attributes)
    
    def warning(self, message: str, **attributes: Any) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, attributes)
    
    def warn(self, message: str, **attributes: Any) -> None:
        """Alias for warning."""
        self.warning(message, **attributes)
    
    def error(self, message: str, exception: Optional[Exception] = None, **attributes: Any) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, attributes, exception)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **attributes: Any) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, message, attributes, exception)
    
    def exception(self, message: str, exception: Exception, **attributes: Any) -> None:
        """Log an exception at ERROR level."""
        self.error(message, exception=exception, **attributes)
    
    @contextmanager
    def operation(self, name: str, **attributes: Any):
        """Context manager for logging operation start/end."""
        start_time = time.time()
        self.info(f"Starting {name}", operation=name, **attributes)
        try:
            yield
            duration = time.time() - start_time
            self.info(
                f"Completed {name}",
                operation=name,
                duration_ms=duration * 1000,
                status="success",
                **attributes,
            )
        except Exception as e:
            duration = time.time() - start_time
            self.error(
                f"Failed {name}",
                exception=e,
                operation=name,
                duration_ms=duration * 1000,
                status="error",
                **attributes,
            )
            raise
    
    def child(self, name: str) -> "StructuredLogger":
        """Create a child logger with inherited context."""
        child_name = f"{self.name}.{name}"
        child_logger = StructuredLogger(child_name, self.level, self.context)
        for handler in self._handlers:
            child_logger.add_handler(handler)
        for filter_func in self._filters:
            child_logger.add_filter(filter_func)
        return child_logger


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}
_global_context = LogContext()
_global_handlers: List[LogHandler] = []
_global_level = LogLevel.INFO
_registry_lock = threading.Lock()


def configure_logging(
    level: Union[LogLevel, str] = LogLevel.INFO,
    json_format: bool = True,
    service_name: str = "llm-inference",
    environment: str = "development",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure global logging settings.
    
    Args:
        level: Minimum log level
        json_format: Use JSON format (True) or text format (False)
        service_name: Name of the service
        environment: Environment name (development, staging, production)
        log_file: Optional file path for file logging
    """
    global _global_context, _global_handlers, _global_level
    
    if isinstance(level, str):
        level = LogLevel.from_string(level)
    
    _global_level = level
    _global_context = LogContext(
        service_name=service_name,
        environment=environment,
    )
    
    # Clear existing handlers
    for handler in _global_handlers:
        handler.close()
    _global_handlers.clear()
    
    # Add console handler
    console_handler = StreamHandler(
        stream=sys.stdout,
        level=level,
        json_format=json_format,
    )
    _global_handlers.append(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = FileHandler(
            filepath=log_file,
            level=level,
            json_format=json_format,
        )
        _global_handlers.append(file_handler)


def set_global_context(**kwargs: Any) -> None:
    """Set global context attributes."""
    global _global_context
    _global_context.extra.update(kwargs)


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a logger by name.
    
    Args:
        name: Logger name (typically module name)
    
    Returns:
        StructuredLogger instance
    """
    global _loggers
    
    with _registry_lock:
        if name not in _loggers:
            logger = StructuredLogger(name, _global_level, _global_context)
            for handler in _global_handlers:
                logger.add_handler(handler)
            _loggers[name] = logger
        return _loggers[name]


# Alias for backward compatibility - ConsoleHandler is StreamHandler with stdout
ConsoleHandler = StreamHandler


class JSONFormatter:
    """
    Formatter that outputs log records as JSON.
    
    Provides configurable JSON formatting for log records with options
    for pretty printing and custom field inclusion.
    """
    
    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_trace: bool = True,
        pretty: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize JSON formatter.
        
        Args:
            include_timestamp: Include timestamp in output
            include_level: Include log level in output
            include_logger: Include logger name in output
            include_trace: Include trace/span IDs if available
            pretty: Use pretty printing (indented JSON)
            extra_fields: Additional fields to include in every record
        """
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_trace = include_trace
        self.pretty = pretty
        self.extra_fields = extra_fields or {}
    
    def format(self, record: LogRecord) -> str:
        """
        Format a log record as JSON string.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted string
        """
        output: Dict[str, Any] = {}
        
        if self.include_timestamp:
            output["timestamp"] = record.timestamp.isoformat()
        
        if self.include_level:
            output["level"] = record.level.name
        
        if self.include_logger:
            output["logger"] = record.logger_name
        
        output["message"] = record.message
        
        if self.include_trace and record.context:
            if record.context.trace_id:
                output["trace_id"] = record.context.trace_id
            if record.context.span_id:
                output["span_id"] = record.context.span_id
        
        # Add extra fields from record context
        if record.context and record.context.extra:
            output.update(record.context.extra)
        
        # Add formatter's extra fields
        if self.extra_fields:
            output.update(self.extra_fields)
        
        if self.pretty:
            return json.dumps(output, indent=2, default=str)
        return json.dumps(output, default=str)
