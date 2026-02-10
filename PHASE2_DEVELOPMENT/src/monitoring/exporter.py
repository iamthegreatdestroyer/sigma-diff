"""
Metrics Export for Prometheus and other monitoring systems.

This module provides exporters that format metrics for consumption
by external monitoring systems like Prometheus.

Classes:
    MetricsExporter: Abstract base for exporters
    PrometheusExporter: Exports metrics in Prometheus text format
    JSONExporter: Exports metrics as JSON
"""

from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
import json

from .metrics import MetricRegistry, MetricType, MetricsCollector

logger = logging.getLogger(__name__)


class MetricsExporter(ABC):
    """Abstract base class for metrics exporters."""
    
    @abstractmethod
    def export(self, registry: MetricRegistry) -> str:
        """
        Export metrics from the registry.
        
        Args:
            registry: The metric registry to export
            
        Returns:
            Formatted metrics string
        """
        pass
    
    @abstractmethod
    def content_type(self) -> str:
        """Get the content type for this export format."""
        pass


class PrometheusExporter(MetricsExporter):
    """
    Export metrics in Prometheus text exposition format.
    
    Follows the Prometheus text format specification:
    https://prometheus.io/docs/instrumenting/exposition_formats/
    
    Example output:
        # HELP llm_inference_tokens_generated_total Total number of tokens generated
        # TYPE llm_inference_tokens_generated_total counter
        llm_inference_tokens_generated_total{model="llama-7b"} 12345
    """
    
    def __init__(self, include_timestamp: bool = False):
        """
        Initialize the exporter.
        
        Args:
            include_timestamp: Whether to include timestamps in output
        """
        self._include_timestamp = include_timestamp
    
    def content_type(self) -> str:
        """Get content type for Prometheus format."""
        return "text/plain; version=0.0.4; charset=utf-8"
    
    def export(self, registry: MetricRegistry) -> str:
        """
        Export metrics in Prometheus text format.
        
        Args:
            registry: The metric registry to export
            
        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        metrics = registry.get_all_metrics()
        definitions = registry._definitions
        
        for metric_name, values in metrics.items():
            if metric_name.startswith("_"):
                continue
                
            definition = definitions.get(metric_name)
            if definition is None:
                continue
            
            # Add HELP and TYPE lines
            full_name = metric_name
            lines.append(f"# HELP {full_name} {definition.description}")
            lines.append(f"# TYPE {full_name} {definition.metric_type.value}")
            
            # Format values based on metric type
            if definition.metric_type == MetricType.COUNTER:
                self._format_counter(lines, full_name, values, definition)
            elif definition.metric_type == MetricType.GAUGE:
                self._format_gauge(lines, full_name, values, definition)
            elif definition.metric_type == MetricType.HISTOGRAM:
                self._format_histogram(lines, full_name, values, definition)
            elif definition.metric_type == MetricType.SUMMARY:
                self._format_summary(lines, full_name, values, definition)
            
            lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)
    
    def _format_labels(self, label_values: tuple, label_names: List[str]) -> str:
        """Format label key-value pairs."""
        if not label_names or not label_values:
            return ""
        
        pairs = []
        for name, value in zip(label_names, label_values):
            # Escape special characters in label values
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            pairs.append(f'{name}="{escaped}"')
        
        return "{" + ",".join(pairs) + "}"
    
    def _format_counter(
        self,
        lines: List[str],
        name: str,
        values: Dict,
        definition
    ) -> None:
        """Format counter metrics."""
        for label_key, value in values.items():
            labels = self._format_labels(label_key, definition.labels)
            lines.append(f"{name}{labels} {value}")
    
    def _format_gauge(
        self,
        lines: List[str],
        name: str,
        values: Dict,
        definition
    ) -> None:
        """Format gauge metrics."""
        for label_key, value in values.items():
            labels = self._format_labels(label_key, definition.labels)
            lines.append(f"{name}{labels} {value}")
    
    def _format_histogram(
        self,
        lines: List[str],
        name: str,
        values: Dict,
        definition
    ) -> None:
        """Format histogram metrics with buckets."""
        for label_key_str, buckets in values.items():
            # Parse label key (stored as string representation of tuple)
            try:
                if label_key_str.startswith("(") and label_key_str.endswith(")"):
                    label_key = eval(label_key_str)
                else:
                    label_key = (label_key_str,) if label_key_str else ()
            except:
                label_key = ()
            
            if isinstance(buckets, dict):
                total_count = 0
                total_sum = 0.0
                
                for bucket_bound, count in sorted(buckets.items()):
                    if bucket_bound == float('inf'):
                        bucket_label = '+Inf'
                    else:
                        bucket_label = str(bucket_bound)
                    
                    labels_base = self._format_labels(label_key, definition.labels)
                    if labels_base:
                        labels = labels_base[:-1] + f',le="{bucket_label}"' + "}"
                    else:
                        labels = '{le="' + bucket_label + '"}'
                    
                    lines.append(f"{name}_bucket{labels} {count}")
                    total_count = max(total_count, count)
                
                # Add _count and _sum
                labels = self._format_labels(label_key, definition.labels)
                lines.append(f"{name}_count{labels} {total_count}")
                lines.append(f"{name}_sum{labels} {total_sum}")
    
    def _format_summary(
        self,
        lines: List[str],
        name: str,
        values: Dict,
        definition
    ) -> None:
        """Format summary metrics with quantiles."""
        for label_key_str, quantiles in values.items():
            try:
                if label_key_str.startswith("(") and label_key_str.endswith(")"):
                    label_key = eval(label_key_str)
                else:
                    label_key = (label_key_str,) if label_key_str else ()
            except:
                label_key = ()
            
            if isinstance(quantiles, dict):
                count = 0
                total = 0.0
                
                for quantile, value in sorted(quantiles.items()):
                    labels_base = self._format_labels(label_key, definition.labels)
                    if labels_base:
                        labels = labels_base[:-1] + f',quantile="{quantile}"' + "}"
                    else:
                        labels = '{quantile="' + str(quantile) + '"}'
                    
                    lines.append(f"{name}{labels} {value}")
                
                # Add _count and _sum (placeholders)
                labels = self._format_labels(label_key, definition.labels)
                lines.append(f"{name}_count{labels} {count}")
                lines.append(f"{name}_sum{labels} {total}")


class JSONExporter(MetricsExporter):
    """
    Export metrics as JSON.
    
    Useful for custom dashboards and debugging.
    """
    
    def __init__(self, pretty: bool = False):
        """
        Initialize the exporter.
        
        Args:
            pretty: Whether to pretty-print the JSON
        """
        self._pretty = pretty
    
    def content_type(self) -> str:
        """Get content type for JSON format."""
        return "application/json"
    
    def export(self, registry: MetricRegistry) -> str:
        """
        Export metrics as JSON.
        
        Args:
            registry: The metric registry to export
            
        Returns:
            JSON-formatted metrics string
        """
        data = {
            "timestamp": time.time(),
            "namespace": registry.namespace,
            "metrics": registry.get_all_metrics()
        }
        
        if self._pretty:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for serving metrics."""
    
    # Class-level references set by MetricsServer
    collector: Optional[MetricsCollector] = None
    exporter: Optional[MetricsExporter] = None
    
    def log_message(self, format: str, *args) -> None:
        """Override to use our logger."""
        logger.debug(f"HTTP: {format % args}")
    
    def do_GET(self) -> None:
        """Handle GET request for metrics."""
        if self.path == "/metrics" or self.path == "/":
            self._serve_metrics()
        elif self.path == "/health":
            self._serve_health()
        else:
            self.send_error(404, "Not Found")
    
    def _serve_metrics(self) -> None:
        """Serve metrics endpoint."""
        if self.collector is None or self.exporter is None:
            self.send_error(500, "Server not configured")
            return
        
        try:
            content = self.exporter.export(self.collector.registry)
            content_bytes = content.encode("utf-8")
            
            self.send_response(200)
            self.send_header("Content-Type", self.exporter.content_type())
            self.send_header("Content-Length", str(len(content_bytes)))
            self.end_headers()
            self.wfile.write(content_bytes)
        except Exception as e:
            logger.error(f"Error serving metrics: {e}")
            self.send_error(500, str(e))
    
    def _serve_health(self) -> None:
        """Serve health check endpoint."""
        response = json.dumps({"status": "healthy", "timestamp": time.time()})
        response_bytes = response.encode("utf-8")
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)


class MetricsServer:
    """
    HTTP server for exposing metrics to Prometheus.
    
    Serves the /metrics endpoint for Prometheus scraping.
    
    Example:
        collector = MetricsCollector()
        server = MetricsServer(collector, port=9090)
        server.start()
        
        # ... later ...
        server.stop()
    """
    
    def __init__(
        self,
        collector: MetricsCollector,
        port: int = 9090,
        host: str = "0.0.0.0",
        exporter: Optional[MetricsExporter] = None
    ):
        """
        Initialize the metrics server.
        
        Args:
            collector: The metrics collector to serve
            port: Port to listen on
            host: Host to bind to
            exporter: Optional custom exporter (defaults to Prometheus)
        """
        self._collector = collector
        self._port = port
        self._host = host
        self._exporter = exporter or PrometheusExporter()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
    
    @property
    def port(self) -> int:
        """Get the server port."""
        return self._port
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._thread is not None and self._thread.is_alive()
    
    def start(self) -> None:
        """Start the metrics server."""
        if self.is_running:
            logger.warning("Metrics server already running")
            return
        
        # Configure handler
        MetricsHTTPHandler.collector = self._collector
        MetricsHTTPHandler.exporter = self._exporter
        
        # Create server
        self._server = HTTPServer((self._host, self._port), MetricsHTTPHandler)
        
        # Start in background thread
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="metrics-server"
        )
        self._thread.start()
        
        logger.info(f"Metrics server started on {self._host}:{self._port}")
    
    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        
        logger.info("Metrics server stopped")
    
    def get_url(self) -> str:
        """Get the metrics URL."""
        return f"http://{self._host}:{self._port}/metrics"
