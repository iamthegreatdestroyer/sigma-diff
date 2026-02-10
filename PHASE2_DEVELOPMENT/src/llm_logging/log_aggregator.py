"""
Log Aggregator and Forwarder

Provides log buffering, aggregation, and forwarding to various
backend systems (Loki, Elasticsearch, etc.) for centralized
log management.
"""

import json
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Set
from queue import Queue, Empty
import hashlib


@dataclass
class LogEntry:
    """A log entry for aggregation."""
    timestamp: datetime
    level: str
    message: str
    logger: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    service: str = "llm-inference"
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "logger": self.logger,
            "service": self.service,
        }
        if self.trace_id:
            data["trace_id"] = self.trace_id
        if self.span_id:
            data["span_id"] = self.span_id
        if self.attributes:
            data.update(self.attributes)
        return data
    
    def fingerprint(self) -> str:
        """Generate a fingerprint for deduplication."""
        key = f"{self.level}:{self.logger}:{self.message}"
        return hashlib.md5(key.encode()).hexdigest()[:16]


class LogBuffer:
    """
    Thread-safe log buffer with overflow protection.
    
    Buffers log entries before forwarding to reduce network
    overhead and handle bursts.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        overflow_strategy: str = "drop_oldest",
    ):
        """
        Initialize buffer.
        
        Args:
            max_size: Maximum number of entries to buffer
            overflow_strategy: What to do on overflow ("drop_oldest", "drop_newest", "block")
        """
        self.max_size = max_size
        self.overflow_strategy = overflow_strategy
        self._buffer: Deque[LogEntry] = deque(maxlen=max_size if overflow_strategy == "drop_oldest" else None)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._dropped_count = 0
    
    def add(self, entry: LogEntry, timeout: Optional[float] = None) -> bool:
        """
        Add entry to buffer.
        
        Args:
            entry: Log entry to add
            timeout: Timeout for blocking (only if overflow_strategy="block")
        
        Returns:
            True if entry was added, False if dropped
        """
        with self._lock:
            if self.overflow_strategy == "drop_oldest":
                # deque with maxlen handles this automatically
                self._buffer.append(entry)
                self._not_empty.notify()
                return True
            
            elif self.overflow_strategy == "drop_newest":
                if len(self._buffer) >= self.max_size:
                    self._dropped_count += 1
                    return False
                self._buffer.append(entry)
                self._not_empty.notify()
                return True
            
            else:  # block
                end_time = time.time() + timeout if timeout else None
                while len(self._buffer) >= self.max_size:
                    if end_time:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            return False
                        self._not_full.wait(remaining)
                    else:
                        self._not_full.wait()
                
                self._buffer.append(entry)
                self._not_empty.notify()
                return True
    
    def get_batch(self, max_batch: int = 100, timeout: Optional[float] = None) -> List[LogEntry]:
        """
        Get a batch of entries.
        
        Args:
            max_batch: Maximum entries to return
            timeout: Timeout waiting for entries
        
        Returns:
            List of log entries
        """
        with self._lock:
            # Wait for entries if empty
            if not self._buffer and timeout:
                self._not_empty.wait(timeout)
            
            # Get up to max_batch entries
            batch = []
            for _ in range(min(max_batch, len(self._buffer))):
                batch.append(self._buffer.popleft())
            
            if batch:
                self._not_full.notify()
            
            return batch
    
    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)
    
    def dropped_count(self) -> int:
        """Get count of dropped entries."""
        return self._dropped_count
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._not_full.notify_all()


class LogForwarder(ABC):
    """Abstract base class for log forwarders."""
    
    @abstractmethod
    def forward(self, entries: List[LogEntry]) -> bool:
        """
        Forward log entries to backend.
        
        Args:
            entries: List of log entries
        
        Returns:
            True if forwarding succeeded
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the forwarder."""
        pass


class ConsoleForwarder(LogForwarder):
    """Forwarder that prints to console (for testing/debugging)."""
    
    def __init__(self, json_format: bool = True):
        self.json_format = json_format
    
    def forward(self, entries: List[LogEntry]) -> bool:
        """Print entries to console."""
        for entry in entries:
            if self.json_format:
                print(json.dumps(entry.to_dict()))
            else:
                print(f"{entry.timestamp.isoformat()} [{entry.level}] {entry.logger}: {entry.message}")
        return True
    
    def close(self) -> None:
        """Nothing to close."""
        pass


class LokiForwarder(LogForwarder):
    """
    Forward logs to Grafana Loki.
    
    Uses the Loki push API to send log streams.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:3100/loki/api/v1/push",
        labels: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        batch_size: int = 100,
    ):
        self.url = url
        self.labels = labels or {"job": "llm-inference"}
        self.timeout = timeout
        self.batch_size = batch_size
        self._session = None
    
    def _get_session(self):
        """Lazy initialization of HTTP session."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
            except ImportError:
                raise ImportError("requests library required for LokiForwarder")
        return self._session
    
    def forward(self, entries: List[LogEntry]) -> bool:
        """Forward entries to Loki."""
        if not entries:
            return True
        
        try:
            session = self._get_session()
            
            # Group entries by labels
            streams: Dict[str, List[tuple]] = {}
            for entry in entries:
                # Create label set
                labels = {**self.labels}
                labels["level"] = entry.level.lower()
                labels["service"] = entry.service
                if entry.logger:
                    labels["logger"] = entry.logger
                
                label_str = self._format_labels(labels)
                if label_str not in streams:
                    streams[label_str] = []
                
                # Loki expects nanosecond timestamps
                ts_ns = str(int(entry.timestamp.timestamp() * 1e9))
                streams[label_str].append((ts_ns, json.dumps(entry.to_dict())))
            
            # Build payload
            payload = {
                "streams": [
                    {
                        "stream": self._parse_labels(label_str),
                        "values": values,
                    }
                    for label_str, values in streams.items()
                ]
            }
            
            # Send to Loki
            response = session.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            return response.status_code == 204
        
        except Exception:
            return False
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels as Loki label string."""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ", ".join(parts) + "}"
    
    def _parse_labels(self, label_str: str) -> Dict[str, str]:
        """Parse label string back to dict."""
        labels = {}
        # Simple parsing - assumes well-formed label string
        content = label_str.strip("{}").split(", ")
        for part in content:
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k] = v.strip('"')
        return labels
    
    def close(self) -> None:
        """Close session."""
        if self._session:
            self._session.close()
            self._session = None


class ElasticsearchForwarder(LogForwarder):
    """
    Forward logs to Elasticsearch.
    
    Uses the bulk API for efficient indexing.
    """
    
    def __init__(
        self,
        hosts: List[str] = None,
        index_pattern: str = "logs-{service}-{date}",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.hosts = hosts or ["http://localhost:9200"]
        self.index_pattern = index_pattern
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None
    
    def _get_session(self):
        """Lazy initialization of HTTP session."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
            except ImportError:
                raise ImportError("requests library required for ElasticsearchForwarder")
        return self._session
    
    def _get_index(self, entry: LogEntry) -> str:
        """Get index name for entry."""
        return self.index_pattern.format(
            service=entry.service.lower().replace(" ", "-"),
            date=entry.timestamp.strftime("%Y.%m.%d"),
        )
    
    def forward(self, entries: List[LogEntry]) -> bool:
        """Forward entries to Elasticsearch."""
        if not entries:
            return True
        
        try:
            session = self._get_session()
            
            # Build bulk request
            lines = []
            for entry in entries:
                index = self._get_index(entry)
                lines.append(json.dumps({"index": {"_index": index}}))
                lines.append(json.dumps(entry.to_dict()))
            
            bulk_body = "\n".join(lines) + "\n"
            
            # Send to Elasticsearch
            for host in self.hosts:
                try:
                    response = session.post(
                        f"{host}/_bulk",
                        data=bulk_body,
                        headers={"Content-Type": "application/x-ndjson"},
                        timeout=self.timeout,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return not result.get("errors", True)
                except Exception:
                    continue
            
            return False
        
        except Exception:
            return False
    
    def close(self) -> None:
        """Close session."""
        if self._session:
            self._session.close()
            self._session = None


class LogAggregator:
    """
    Log aggregator with buffering and forwarding.
    
    Collects log entries, buffers them, and forwards to configured
    backends in batches for efficiency.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        forwarders: Optional[List[LogForwarder]] = None,
    ):
        """
        Initialize aggregator.
        
        Args:
            buffer_size: Maximum entries to buffer
            batch_size: Entries per batch when forwarding
            flush_interval: Seconds between flush attempts
            forwarders: List of log forwarders
        """
        self.buffer = LogBuffer(max_size=buffer_size)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.forwarders = forwarders or []
        
        # Metrics
        self._forwarded_count = 0
        self._failed_count = 0
        self._dedup_count = 0
        
        # Deduplication
        self._seen_fingerprints: Set[str] = set()
        self._fingerprint_window = 60.0  # seconds
        self._last_fingerprint_clear = time.time()
        
        # Background flush
        self._shutdown = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start background forwarding."""
        self._flush_thread.start()
    
    def stop(self, timeout: float = 10.0) -> None:
        """Stop background forwarding and flush remaining."""
        self._shutdown.set()
        self._flush_thread.join(timeout=timeout)
        self.flush()
        for forwarder in self.forwarders:
            forwarder.close()
    
    def add_forwarder(self, forwarder: LogForwarder) -> None:
        """Add a log forwarder."""
        self.forwarders.append(forwarder)
    
    def add(
        self,
        message: str,
        level: str = "INFO",
        logger: str = "app",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        deduplicate: bool = True,
        **attributes: Any,
    ) -> bool:
        """
        Add a log entry.
        
        Args:
            message: Log message
            level: Log level
            logger: Logger name
            trace_id: Trace ID for correlation
            span_id: Span ID for correlation
            deduplicate: Whether to deduplicate similar messages
            **attributes: Additional attributes
        
        Returns:
            True if entry was added
        """
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level.upper(),
            message=message,
            logger=logger,
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes,
        )
        
        # Deduplication
        if deduplicate:
            fingerprint = entry.fingerprint()
            current_time = time.time()
            
            with self._lock:
                # Clear old fingerprints periodically
                if current_time - self._last_fingerprint_clear > self._fingerprint_window:
                    self._seen_fingerprints.clear()
                    self._last_fingerprint_clear = current_time
                
                if fingerprint in self._seen_fingerprints:
                    self._dedup_count += 1
                    return False
                
                self._seen_fingerprints.add(fingerprint)
        
        return self.buffer.add(entry)
    
    def flush(self) -> int:
        """
        Flush buffered entries to forwarders.
        
        Returns:
            Number of entries forwarded
        """
        total_forwarded = 0
        
        while True:
            batch = self.buffer.get_batch(self.batch_size, timeout=0.1)
            if not batch:
                break
            
            success = True
            for forwarder in self.forwarders:
                if not forwarder.forward(batch):
                    success = False
            
            if success:
                self._forwarded_count += len(batch)
                total_forwarded += len(batch)
            else:
                self._failed_count += len(batch)
        
        return total_forwarded
    
    def _flush_loop(self) -> None:
        """Background flush loop."""
        while not self._shutdown.is_set():
            self._shutdown.wait(self.flush_interval)
            if not self._shutdown.is_set():
                self.flush()
    
    def stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "buffer_size": self.buffer.size(),
            "forwarded": self._forwarded_count,
            "failed": self._failed_count,
            "deduplicated": self._dedup_count,
            "dropped": self.buffer.dropped_count(),
        }


# Convenience function for quick setup
def create_aggregator(
    loki_url: Optional[str] = None,
    elasticsearch_hosts: Optional[List[str]] = None,
    console: bool = False,
) -> LogAggregator:
    """
    Create a log aggregator with common configurations.
    
    Args:
        loki_url: Loki push API URL
        elasticsearch_hosts: Elasticsearch host URLs
        console: Whether to also print to console
    
    Returns:
        Configured LogAggregator
    """
    forwarders = []
    
    if console:
        forwarders.append(ConsoleForwarder())
    
    if loki_url:
        forwarders.append(LokiForwarder(url=loki_url))
    
    if elasticsearch_hosts:
        forwarders.append(ElasticsearchForwarder(hosts=elasticsearch_hosts))
    
    aggregator = LogAggregator(forwarders=forwarders)
    aggregator.start()
    return aggregator
