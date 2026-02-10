"""
Metrics Collection for Distributed Inference
============================================

Collects and aggregates metrics from distributed GPU inference.
Provides real-time monitoring and performance analytics.

Key Features:
- Request/response metrics
- GPU utilization tracking
- Latency and throughput monitoring
- Error rate analysis
- Historical data aggregation

Metrics Types:
- Counter: Monotonically increasing values
- Gauge: Point-in-time measurements
- Histogram: Distribution of values
- Summary: Quantiles over sliding time window
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric measurement."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series data for a metric."""
    name: str
    points: List[MetricPoint] = field(default_factory=list)
    max_points: int = 1000  # Keep last 1000 points

    def add_point(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a new measurement point."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )

        self.points.append(point)

        # Maintain max points limit
        if len(self.points) > self.max_points:
            self.points.pop(0)

    def get_recent_points(self, seconds: float = 60.0) -> List[MetricPoint]:
        """Get points from the last N seconds."""
        cutoff_time = time.time() - seconds
        return [p for p in self.points if p.timestamp >= cutoff_time]

    def get_aggregate(self, seconds: float = 60.0) -> Dict[str, float]:
        """Get aggregated statistics for recent points."""
        recent_points = self.get_recent_points(seconds)

        if not recent_points:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}

        values = [p.value for p in recent_points]

        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }


class MetricsCollector:
    """
    Collects and aggregates metrics for distributed inference.

    Provides real-time monitoring capabilities and historical analysis.
    """

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.retention_seconds = retention_hours * 3600

        # Metric storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.lock = threading.Lock()

        # Initialize standard metrics
        self._initialize_standard_metrics()

        # Cleanup task
        self.cleanup_task = None

    def _initialize_standard_metrics(self):
        """Initialize standard metric series."""
        standard_metrics = [
            "requests_total",
            "requests_success",
            "requests_failed",
            "request_latency_ms",
            "gpu_utilization_percent",
            "gpu_memory_used_mb",
            "batch_size_avg",
            "tokens_generated_total",
            "tokens_processed_total",
            "error_rate_percent"
        ]

        for metric_name in standard_metrics:
            self.metrics[metric_name] = MetricSeries(metric_name)

    async def record_request(
        self,
        gpu_id: int,
        tokens_processed: int,
        success: bool,
        latency_ms: Optional[float] = None,
        error_type: Optional[str] = None
    ):
        """Record a request completion."""
        with self.lock:
            # Total requests
            self.metrics["requests_total"].add_point(1.0, {"gpu_id": str(gpu_id)})

            # Success/failure
            if success:
                self.metrics["requests_success"].add_point(1.0, {"gpu_id": str(gpu_id)})
            else:
                self.metrics["requests_failed"].add_point(1.0, {"gpu_id": str(gpu_id)})
                if error_type:
                    # Could add error type specific metrics
                    pass

            # Latency
            if latency_ms is not None:
                self.metrics["request_latency_ms"].add_point(latency_ms, {"gpu_id": str(gpu_id)})

            # Tokens
            self.metrics["tokens_processed_total"].add_point(float(tokens_processed), {"gpu_id": str(gpu_id)})

    def record_gpu_stats(self, gpu_id: int, utilization: float, memory_used_mb: float):
        """Record GPU utilization statistics."""
        with self.lock:
            self.metrics["gpu_utilization_percent"].add_point(utilization, {"gpu_id": str(gpu_id)})
            self.metrics["gpu_memory_used_mb"].add_point(memory_used_mb, {"gpu_id": str(gpu_id)})

    def record_batch_stats(self, gpu_id: int, batch_size: int, tokens_generated: int):
        """Record batch processing statistics."""
        with self.lock:
            self.metrics["batch_size_avg"].add_point(float(batch_size), {"gpu_id": str(gpu_id)})
            self.metrics["tokens_generated_total"].add_point(float(tokens_generated), {"gpu_id": str(gpu_id)})

    def record_error_rate(self, gpu_id: int, error_rate: float):
        """Record error rate for a GPU."""
        with self.lock:
            self.metrics["error_rate_percent"].add_point(error_rate, {"gpu_id": str(gpu_id)})

    def get_metric_summary(self, metric_name: str, seconds: float = 300.0) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self.lock:
            if metric_name not in self.metrics:
                return {"error": f"Metric {metric_name} not found"}

            series = self.metrics[metric_name]
            aggregate = series.get_aggregate(seconds)

            return {
                "metric": metric_name,
                "period_seconds": seconds,
                **aggregate,
                "latest_value": series.points[-1].value if series.points else None,
                "latest_timestamp": series.points[-1].timestamp if series.points else None
            }

    def get_gpu_summary(self, gpu_id: int, seconds: float = 300.0) -> Dict[str, Any]:
        """Get comprehensive summary for a specific GPU."""
        with self.lock:
            gpu_label = {"gpu_id": str(gpu_id)}

            return {
                "gpu_id": gpu_id,
                "period_seconds": seconds,
                "requests_total": self._get_metric_for_labels("requests_total", gpu_label, seconds),
                "requests_success": self._get_metric_for_labels("requests_success", gpu_label, seconds),
                "requests_failed": self._get_metric_for_labels("requests_failed", gpu_label, seconds),
                "avg_latency_ms": self._get_metric_for_labels("request_latency_ms", gpu_label, seconds),
                "avg_utilization_percent": self._get_metric_for_labels("gpu_utilization_percent", gpu_label, seconds),
                "avg_memory_used_mb": self._get_metric_for_labels("gpu_memory_used_mb", gpu_label, seconds),
                "avg_batch_size": self._get_metric_for_labels("batch_size_avg", gpu_label, seconds),
                "tokens_processed": self._get_metric_for_labels("tokens_processed_total", gpu_label, seconds),
                "tokens_generated": self._get_metric_for_labels("tokens_generated_total", gpu_label, seconds),
                "error_rate_percent": self._get_metric_for_labels("error_rate_percent", gpu_label, seconds)
            }

    def _get_metric_for_labels(self, metric_name: str, labels: Dict[str, str], seconds: float) -> Dict[str, float]:
        """Get metric aggregate filtered by labels."""
        if metric_name not in self.metrics:
            return {"count": 0, "avg": 0}

        series = self.metrics[metric_name]
        recent_points = series.get_recent_points(seconds)

        # Filter by labels
        filtered_points = [
            p for p in recent_points
            if all(p.labels.get(k) == v for k, v in labels.items())
        ]

        if not filtered_points:
            return {"count": 0, "avg": 0}

        values = [p.value for p in filtered_points]
        return {
            "count": len(values),
            "avg": sum(values) / len(values)
        }

    def get_system_summary(self, seconds: float = 300.0) -> Dict[str, Any]:
        """Get system-wide metrics summary."""
        with self.lock:
            # Aggregate across all GPUs
            total_requests = 0
            total_success = 0
            total_failed = 0
            total_tokens = 0
            gpu_summaries = []

            # Get unique GPU IDs from metrics
            gpu_ids = set()
            for series in self.metrics.values():
                for point in series.points:
                    if "gpu_id" in point.labels:
                        gpu_ids.add(int(point.labels["gpu_id"]))

            for gpu_id in sorted(gpu_ids):
                gpu_summary = self.get_gpu_summary(gpu_id, seconds)
                gpu_summaries.append(gpu_summary)

                total_requests += gpu_summary["requests_total"]["count"]
                total_success += gpu_summary["requests_success"]["count"]
                total_failed += gpu_summary["requests_failed"]["count"]
                total_tokens += gpu_summary["tokens_processed"]["count"]

            # Calculate system-wide averages
            avg_latency = sum(
                gpu["avg_latency_ms"]["avg"] for gpu in gpu_summaries
                if gpu["avg_latency_ms"]["count"] > 0
            ) / len([g for g in gpu_summaries if g["avg_latency_ms"]["count"] > 0]) \
                if gpu_summaries else 0

            avg_utilization = sum(
                gpu["avg_utilization_percent"]["avg"] for gpu in gpu_summaries
                if gpu["avg_utilization_percent"]["count"] > 0
            ) / len([g for g in gpu_summaries if g["avg_utilization_percent"]["count"] > 0]) \
                if gpu_summaries else 0

            error_rate = (total_failed / total_requests * 100) if total_requests > 0 else 0

            return {
                "period_seconds": seconds,
                "total_gpus": len(gpu_summaries),
                "total_requests": total_requests,
                "total_success": total_success,
                "total_failed": total_failed,
                "total_tokens_processed": total_tokens,
                "system_avg_latency_ms": avg_latency,
                "system_avg_utilization_percent": avg_utilization,
                "system_error_rate_percent": error_rate,
                "gpu_summaries": gpu_summaries,
                "timestamp": time.time()
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all available metrics."""
        with self.lock:
            return {
                metric_name: {
                    "count": len(series.points),
                    "latest_value": series.points[-1].value if series.points else None,
                    "latest_timestamp": series.points[-1].timestamp if series.points else None
                }
                for metric_name, series in self.metrics.items()
            }

    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        with self.lock:
            cutoff_time = time.time() - self.retention_seconds

            for series in self.metrics.values():
                # Remove old points
                series.points = [p for p in series.points if p.timestamp >= cutoff_time]

    def start_cleanup_task(self):
        """Start periodic cleanup task."""
        if self.cleanup_task is None:
            self.cleanup_task = threading.Thread(
                target=self._cleanup_worker,
                daemon=True
            )
            self.cleanup_task.start()

    def _cleanup_worker(self):
        """Background cleanup worker."""
        while True:
            time.sleep(3600)  # Clean up every hour
            self.cleanup_old_metrics()
            logger.debug("Cleaned up old metrics")

    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format."""
        import json

        data = {
            "export_time": time.time(),
            "retention_hours": self.retention_hours,
            "metrics": {}
        }

        with self.lock:
            for name, series in self.metrics.items():
                data["metrics"][name] = {
                    "points": [
                        {
                            "timestamp": p.timestamp,
                            "value": p.value,
                            "labels": p.labels
                        }
                        for p in series.points[-100:]  # Export last 100 points
                    ]
                }

        if format == "json":
            return json.dumps(data, indent=2)
        else:
            return json.dumps(data)  # Compact JSON
