#!/usr/bin/env python3
"""
Phase 2: Training Metrics Collector

Real-time collection and analysis of training and optimization metrics:
- Training loss and accuracy per batch and epoch
- Inference latency (TTFT, throughput)
- Compression effectiveness (embedding compression ratio, reconstruction error)
- Kernel performance (memory access patterns, cache hits)
- Overall E2E speedup

Outputs:
- training_metrics_report.json: Comprehensive epoch-by-epoch report
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchMetric:
    """Single batch metric record."""
    epoch: int
    batch: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    loss: float = 0.0
    learning_rate: float = 0.0
    throughput: float = 0.0  # samples/sec
    batch_duration_sec: float = 0.0
    
    # Compression metrics
    compression_ratio: Optional[float] = None
    embedding_reconstruction_error: Optional[float] = None
    
    # Inference latency metrics
    ttft_ms: Optional[float] = None  # Time to first token
    throughput_tokens_sec: Optional[float] = None
    
    # Kernel metrics
    kernel_speedup: Optional[float] = None
    cache_hit_rate: Optional[float] = None


@dataclass
class EpochMetric:
    """Single epoch metric record."""
    epoch: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None
    duration_sec: float = 0.0
    num_batches: int = 0
    learning_rate: float = 0.0
    throughput: float = 0.0  # samples/sec
    
    # Optimization metrics
    compression_ratio: float = 1.0
    avg_embedding_error: Optional[float] = None
    kernel_speedup: float = 1.0
    inference_speedup: float = 1.0
    combined_speedup: float = 1.0
    
    # Training stability
    loss_improvement: float = 0.0  # Change from previous epoch
    gradient_norm: Optional[float] = None
    weight_norm: Optional[float] = None


class TrainingMetricsCollector:
    """
    Collect and analyze training metrics in real-time.
    
    Features:
    - Batch-level metric recording
    - Epoch-level aggregation
    - Real-time analysis and flagging
    - JSON export for post-training analysis
    - Performance anomaly detection
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.batch_metrics: List[BatchMetric] = []
        self.epoch_metrics: List[EpochMetric] = []
        
        # Running statistics
        self.batch_count = 0
        self.epoch_count = 0
        self.start_time = datetime.now()
        
        # Thresholds for anomaly detection
        self.loss_spike_threshold = 1.5  # 50% increase triggers alert
        self.accuracy_threshold_low = 0.85
        self.accuracy_threshold_high = 0.999
        
        logger.info("TrainingMetricsCollector initialized")
    
    def record_batch_metric(self, metric_dict: Dict[str, Any]) -> None:
        """
        Record metrics for a single batch.
        
        Args:
            metric_dict: Dictionary with batch metrics
                Required: epoch, batch, loss, throughput
                Optional: compression_ratio, ttft_ms, kernel_speedup, etc.
        """
        batch_metric = BatchMetric(
            epoch=metric_dict.get('epoch', 0),
            batch=metric_dict.get('batch', self.batch_count),
            loss=metric_dict.get('loss', 0.0),
            learning_rate=metric_dict.get('learning_rate', 0.0),
            throughput=metric_dict.get('throughput', 0.0),
            batch_duration_sec=metric_dict.get('batch_duration_sec', 0.0),
            compression_ratio=metric_dict.get('compression_ratio'),
            embedding_reconstruction_error=metric_dict.get('embedding_reconstruction_error'),
            ttft_ms=metric_dict.get('ttft_ms'),
            throughput_tokens_sec=metric_dict.get('throughput_tokens_sec'),
            kernel_speedup=metric_dict.get('kernel_speedup'),
            cache_hit_rate=metric_dict.get('cache_hit_rate')
        )
        
        self.batch_metrics.append(batch_metric)
        self.batch_count += 1
        
        # Check for anomalies
        self._check_batch_anomalies(batch_metric)
    
    def record_epoch_metric(self, metric_dict: Dict[str, Any]) -> None:
        """
        Record metrics for a completed epoch.
        
        Args:
            metric_dict: Dictionary with epoch metrics
                Required: epoch, train_loss, duration_sec
                Optional: val_loss, accuracy, compression_ratio, kernel_speedup, etc.
        """
        # Compute loss improvement if available
        loss_improvement = 0.0
        if len(self.epoch_metrics) > 0:
            prev_loss = self.epoch_metrics[-1].train_loss
            curr_loss = metric_dict.get('train_loss', 0.0)
            if prev_loss > 0:
                loss_improvement = (prev_loss - curr_loss) / prev_loss
        
        epoch_metric = EpochMetric(
            epoch=metric_dict.get('epoch', self.epoch_count),
            train_loss=metric_dict.get('train_loss', 0.0),
            val_loss=metric_dict.get('val_loss'),
            accuracy=metric_dict.get('accuracy'),
            duration_sec=metric_dict.get('duration_sec', 0.0),
            num_batches=metric_dict.get('num_batches', 0),
            learning_rate=metric_dict.get('learning_rate', 0.0),
            throughput=metric_dict.get('throughput', 0.0),
            compression_ratio=metric_dict.get('compression_ratio', 1.0),
            avg_embedding_error=metric_dict.get('avg_embedding_error'),
            kernel_speedup=metric_dict.get('kernel_speedup', 1.0),
            inference_speedup=metric_dict.get('inference_speedup', 1.0),
            combined_speedup=metric_dict.get('combined_speedup', 1.0),
            loss_improvement=loss_improvement,
            gradient_norm=metric_dict.get('gradient_norm'),
            weight_norm=metric_dict.get('weight_norm')
        )
        
        self.epoch_metrics.append(epoch_metric)
        self.epoch_count += 1
        
        # Check for anomalies
        self._check_epoch_anomalies(epoch_metric)
    
    def _check_batch_anomalies(self, metric: BatchMetric) -> None:
        """Check for anomalies in batch metrics."""
        if metric.loss > 1000:
            logger.warning(f"Batch {metric.batch} - Extremely high loss detected: {metric.loss:.2f}")
        
        if metric.throughput <= 0:
            logger.warning(f"Batch {metric.batch} - Invalid throughput: {metric.throughput:.2f}")
        
        if metric.batch_duration_sec > 60:  # Batch took >60 seconds
            logger.warning(f"Batch {metric.batch} - Very high batch time: {metric.batch_duration_sec:.1f}s")
    
    def _check_epoch_anomalies(self, metric: EpochMetric) -> None:
        """Check for anomalies in epoch metrics."""
        # Loss spike detection
        if len(self.epoch_metrics) > 1:
            prev_loss = self.epoch_metrics[-2].train_loss
            curr_loss = metric.train_loss
            if prev_loss > 0:
                loss_ratio = curr_loss / prev_loss
                if loss_ratio > self.loss_spike_threshold:
                    logger.warning(
                        f"Epoch {metric.epoch} - Loss spike detected: "
                        f"{prev_loss:.4f} → {curr_loss:.4f} ({loss_ratio:.2f}x)"
                    )
        
        # Accuracy anomalies
        if metric.accuracy is not None:
            if metric.accuracy < self.accuracy_threshold_low:
                logger.warning(f"Epoch {metric.epoch} - Low accuracy: {metric.accuracy:.4f}")
            elif metric.accuracy > self.accuracy_threshold_high:
                logger.info(f"Epoch {metric.epoch} - Excellent accuracy: {metric.accuracy:.4f}")
        
        # Training time anomalies
        if metric.duration_sec > 1000 and len(self.epoch_metrics) > 2:
            avg_duration = np.mean([m.duration_sec for m in self.epoch_metrics[:-1]])
            if metric.duration_sec > avg_duration * 2:
                logger.warning(
                    f"Epoch {metric.epoch} - Very long epoch duration: "
                    f"{metric.duration_sec:.1f}s (avg: {avg_duration:.1f}s)"
                )
    
    def get_epoch_summary(self, epoch: int) -> Optional[Dict[str, Any]]:
        """
        Get summary for a specific epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Dictionary with epoch summary or None if not found
        """
        for metric in self.epoch_metrics:
            if metric.epoch == epoch:
                return asdict(metric)
        return None
    
    def get_batch_summary(self, epoch: int, batch: int) -> Optional[Dict[str, Any]]:
        """
        Get summary for a specific batch.
        
        Args:
            epoch: Epoch number
            batch: Batch number
            
        Returns:
            Dictionary with batch summary or None if not found
        """
        for metric in self.batch_metrics:
            if metric.epoch == epoch and metric.batch == batch:
                return asdict(metric)
        return None
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all metrics."""
        if not self.epoch_metrics:
            return {}
        
        losses = [m.train_loss for m in self.epoch_metrics]
        durations = [m.duration_sec for m in self.epoch_metrics]
        throughputs = [m.throughput for m in self.epoch_metrics if m.throughput > 0]
        speedups = [m.combined_speedup for m in self.epoch_metrics]
        
        return {
            'num_epochs': len(self.epoch_metrics),
            'total_duration_sec': sum(durations),
            'avg_epoch_duration_sec': np.mean(durations),
            'min_epoch_duration_sec': np.min(durations),
            'max_epoch_duration_sec': np.max(durations),
            
            'final_loss': losses[-1],
            'initial_loss': losses[0],
            'total_loss_improvement': (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0,
            'avg_loss': np.mean(losses),
            'min_loss': np.min(losses),
            'max_loss': np.max(losses),
            
            'avg_throughput': np.mean(throughputs) if throughputs else 0,
            'max_throughput': np.max(throughputs) if throughputs else 0,
            'min_throughput': np.min(throughputs) if throughputs else 0,
            
            'avg_speedup': np.mean(speedups),
            'max_speedup': np.max(speedups),
            'min_speedup': np.min(speedups),
            'target_speedup_achieved': max(speedups) >= 3.0 if speedups else False
        }
    
    def export_metrics(self, output_path: str) -> None:
        """
        Export all metrics to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare report
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_epochs': self.epoch_count,
                'total_batches': self.batch_count,
                'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600
            },
            'statistics': self.compute_statistics(),
            'epoch_metrics': [asdict(m) for m in self.epoch_metrics],
            'batch_metrics_sample': [
                asdict(m) for m in self.batch_metrics[::max(1, len(self.batch_metrics) // 100)]
            ]  # Sample every 100th batch
        }
        
        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {output_path}")
    
    def print_summary(self) -> None:
        """Print summary statistics to console."""
        stats = self.compute_statistics()
        
        print("\n" + "="*60)
        print("TRAINING METRICS SUMMARY")
        print("="*60)
        
        if 'num_epochs' in stats:
            print(f"\nTraining Summary:")
            print(f"  Epochs: {stats['num_epochs']}")
            print(f"  Total Duration: {stats['total_duration_sec']/60:.1f} minutes")
            print(f"  Avg Epoch Time: {stats['avg_epoch_duration_sec']:.1f}s")
        
        if 'final_loss' in stats:
            print(f"\nLoss Metrics:")
            print(f"  Initial Loss: {stats['initial_loss']:.4f}")
            print(f"  Final Loss: {stats['final_loss']:.4f}")
            print(f"  Improvement: {stats['total_loss_improvement']*100:.1f}%")
            print(f"  Avg Loss: {stats['avg_loss']:.4f}")
        
        if 'avg_throughput' in stats and stats['avg_throughput'] > 0:
            print(f"\nThroughput Metrics:")
            print(f"  Avg: {stats['avg_throughput']:.1f} samples/sec")
            print(f"  Max: {stats['max_throughput']:.1f} samples/sec")
            print(f"  Min: {stats['min_throughput']:.1f} samples/sec")
        
        if 'avg_speedup' in stats:
            print(f"\nOptimization Speedup:")
            print(f"  Average: {stats['avg_speedup']:.2f}x")
            print(f"  Max: {stats['max_speedup']:.2f}x")
            print(f"  Min: {stats['min_speedup']:.2f}x")
            print(f"  3-5x Target Achieved: {'✅ YES' if stats['target_speedup_achieved'] else '⚠️ NO'}")
        
        print("\n" + "="*60)
