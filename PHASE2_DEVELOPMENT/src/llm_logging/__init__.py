"""
Structured Logging Module for LLM Inference Framework

Provides structured JSON logging with distributed tracing correlation,
log aggregation, and integration with observability platforms.

Key Features:
- Structured JSON logging with trace context
- Log level management and filtering
- Log aggregation and forwarding
- Correlation with distributed traces
- Performance metric logging
"""

from .structured_logger import (
    StructuredLogger,
    LogLevel,
    LogRecord,
    LogContext,
    LogHandler,
    ConsoleHandler,
    FileHandler,
    JSONFormatter,
    get_logger,
    configure_logging,
    set_global_context,
)
from .log_aggregator import (
    LogAggregator,
    LogBuffer,
    LogForwarder,
    LogEntry,
    LokiForwarder,
    ElasticsearchForwarder,
    ConsoleForwarder,
)

__all__ = [
    # Structured Logger
    "StructuredLogger",
    "LogLevel",
    "LogRecord",
    "LogContext",
    "LogHandler",
    "ConsoleHandler",
    "FileHandler",
    "JSONFormatter",
    "get_logger",
    "configure_logging",
    "set_global_context",
    # Log Aggregator
    "LogAggregator",
    "LogBuffer",
    "LogForwarder",
    "LogEntry",
    "LokiForwarder",
    "ElasticsearchForwarder",
    "ConsoleForwarder",
]

__version__ = "1.0.0"
