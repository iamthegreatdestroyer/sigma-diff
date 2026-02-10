#!/usr/bin/env python3
"""
Phase 2: Model Inference Validator

End-to-end inference performance tracking and validation:
- Load baseline and optimized models
- Execute inference benchmarks
- Measure TTFT (Time To First Token) and throughput
- Compare against Phase 1 baseline metrics
- Generate comprehensive validation reports

Outputs:
- inference_validation_report.json: E2E speedup validation
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import statistics

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetric:
    """Single inference run metric."""
    model_type: str  # "baseline" or "optimized"
    batch_size: int
    sequence_length: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Latency metrics (milliseconds)
    ttft_ms: float = 0.0  # Time to first token
    tpot_ms: float = 0.0  # Time per output token
    
    # Throughput metrics
    throughput_tokens_sec: float = 0.0
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_p99_ms: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    
    # Quality metrics
    output_length: int = 0
    inference_successful: bool = True
    error_message: Optional[str] = None
    
    # Optimization metrics
    compression_ratio: Optional[float] = None
    kernel_speedup: Optional[float] = None
    total_speedup: Optional[float] = None


@dataclass
class ValidationResults:
    """Complete validation results comparing baseline vs optimized."""
    validation_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    baseline_metrics: List[InferenceMetric] = field(default_factory=list)
    optimized_metrics: List[InferenceMetric] = field(default_factory=list)
    
    # Aggregated statistics
    baseline_ttft_mean_ms: float = 0.0
    optimized_ttft_mean_ms: float = 0.0
    ttft_speedup: float = 1.0
    
    baseline_throughput_mean: float = 0.0
    optimized_throughput_mean: float = 0.0
    throughput_improvement: float = 1.0
    
    # Memory efficiency
    baseline_peak_memory_mb: float = 0.0
    optimized_peak_memory_mb: float = 0.0
    memory_reduction_percent: float = 0.0
    
    # Success rates
    baseline_success_rate: float = 1.0
    optimized_success_rate: float = 1.0
    
    # Phase 1 baseline comparisons
    phase1_ttft_target_ms: float = 0.0
    phase1_throughput_target: float = 0.0
    ttft_vs_phase1: float = 1.0
    throughput_vs_phase1: float = 1.0
    
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


class ModelInferenceValidator:
    """Validates inference performance of baseline and optimized models."""
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = "reports"):
        """
        Initialize validator.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for output reports
        """
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 1 baseline targets from architecture review
        self.phase1_ttft_target_ms = 120.0  # Target from Phase 1 spec
        self.phase1_throughput_target = 25.0  # tokens/second from Phase 1
        
        self.results = ValidationResults()
        logger.info("ModelInferenceValidator initialized")
    
    def load_model(self, model_path: str, device: str = "cpu") -> Optional[torch.nn.Module]:
        """
        Load PyTorch model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to load model to ("cpu" or "cuda")
            
        Returns:
            Loaded model or None if loading fails
        """
        try:
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                model = torch.load(model_path, map_location=device)
            elif model_path.endswith('.safetensors'):
                # Placeholder for safetensors loading
                logger.warning(f"safetensors loader not implemented, using default loading")
                model = torch.load(model_path, map_location=device)
            else:
                logger.error(f"Unknown model format: {model_path}")
                return None
            
            model.eval()
            model.to(device)
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    
    def measure_ttft(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        device: str = "cpu",
        num_runs: int = 5
    ) -> Tuple[float, float]:
        """
        Measure Time To First Token (TTFT).
        
        Args:
            model: Model to benchmark
            input_ids: Input token IDs
            device: Device to run on
            num_runs: Number of runs for averaging
            
        Returns:
            Tuple of (mean_ttft_ms, std_ttft_ms)
        """
        ttft_times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                # Ensure model is on correct device
                input_ids = input_ids.to(device)
                
                # Warm up GPU/CPU cache
                _ = model(input_ids)
                
                # Measure TTFT
                torch.cuda.synchronize() if device == "cuda" else None
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model(input_ids)
                
                torch.cuda.synchronize() if device == "cuda" else None
                elapsed_ms = (time.time() - start_time) * 1000
                ttft_times.append(elapsed_ms)
        
        return statistics.mean(ttft_times), statistics.stdev(ttft_times) if len(ttft_times) > 1 else 0.0
    
    def measure_throughput(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        num_tokens_generate: int = 100,
        device: str = "cpu"
    ) -> float:
        """
        Measure inference throughput (tokens/second).
        
        Args:
            model: Model to benchmark
            input_ids: Input token IDs
            num_tokens_generate: Number of tokens to generate
            device: Device to run on
            
        Returns:
            Throughput in tokens/second
        """
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            # Warm up
            _ = model(input_ids)
            
            # Measure throughput
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()
            
            current_ids = input_ids
            for _ in range(num_tokens_generate):
                outputs = model(current_ids)
                # Simulate next token (in real scenario, would sample from logits)
                current_ids = outputs[:, -1:, :].argmax(dim=-1) if len(outputs.shape) == 3 else outputs
            
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed_sec = time.time() - start_time
        
        throughput = num_tokens_generate / elapsed_sec if elapsed_sec > 0 else 0.0
        return throughput
    
    def measure_memory(self, model: torch.nn.Module, device: str = "cpu") -> Tuple[float, float]:
        """
        Measure peak and average memory usage.
        
        Args:
            model: Model to measure
            device: Device to measure on
            
        Returns:
            Tuple of (peak_memory_mb, average_memory_mb)
        """
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Run inference
            dummy_input = torch.randint(0, 32000, (1, 512)).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
            return peak_memory, peak_memory  # Simplified: peak ≈ average for demo
        else:
            # CPU memory detection - simplified
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 ** 2), memory_info.rss / (1024 ** 2)  # in MB
    
    def validate_baseline(
        self,
        model: torch.nn.Module,
        test_data: torch.Tensor,
        batch_size: int = 1,
        device: str = "cpu",
        num_runs: int = 5
    ) -> None:
        """
        Run inference validation for baseline model.
        
        Args:
            model: Baseline model
            test_data: Test input tensor
            batch_size: Batch size for inference
            device: Device to run on
            num_runs: Number of validation runs
        """
        logger.info(f"Validating baseline model on {device}")
        
        for run_idx in range(num_runs):
            try:
                # Prepare input
                if len(test_data.shape) == 1:
                    input_ids = test_data.unsqueeze(0).to(device)
                else:
                    input_ids = test_data[:batch_size].to(device)
                
                # Measure TTFT
                ttft_mean, ttft_std = self.measure_ttft(model, input_ids, device, num_runs=1)
                
                # Measure throughput
                throughput = self.measure_throughput(model, input_ids, num_tokens_generate=50, device=device)
                
                # Measure memory
                peak_mem, avg_mem = self.measure_memory(model, device)
                
                # Create metric record
                metric = InferenceMetric(
                    model_type="baseline",
                    batch_size=batch_size,
                    sequence_length=input_ids.shape[1] if len(input_ids.shape) > 1 else 1,
                    ttft_ms=ttft_mean,
                    throughput_tokens_sec=throughput,
                    peak_memory_mb=peak_mem,
                    average_memory_mb=avg_mem,
                    inference_successful=True
                )
                
                self.results.baseline_metrics.append(metric)
                logger.info(
                    f"Baseline run {run_idx + 1}: "
                    f"TTFT={ttft_mean:.2f}ms, "
                    f"Throughput={throughput:.2f} tok/s, "
                    f"Memory={peak_mem:.2f}MB"
                )
                
            except Exception as e:
                logger.error(f"Baseline validation run {run_idx + 1} failed: {e}")
                metric = InferenceMetric(
                    model_type="baseline",
                    batch_size=batch_size,
                    sequence_length=0,
                    inference_successful=False,
                    error_message=str(e)
                )
                self.results.baseline_metrics.append(metric)
        
        # Compute aggregated statistics
        if self.results.baseline_metrics:
            successful_metrics = [m for m in self.results.baseline_metrics if m.inference_successful]
            if successful_metrics:
                ttfts = [m.ttft_ms for m in successful_metrics]
                throughputs = [m.throughput_tokens_sec for m in successful_metrics]
                
                self.results.baseline_ttft_mean_ms = statistics.mean(ttfts)
                self.results.baseline_throughput_mean = statistics.mean(throughputs)
                self.results.baseline_peak_memory_mb = successful_metrics[0].peak_memory_mb
                self.results.baseline_success_rate = len(successful_metrics) / len(self.results.baseline_metrics)
    
    def validate_optimized(
        self,
        model: torch.nn.Module,
        test_data: torch.Tensor,
        batch_size: int = 1,
        device: str = "cpu",
        num_runs: int = 5
    ) -> None:
        """
        Run inference validation for optimized model.
        
        Args:
            model: Optimized model with Phase 1 optimizations
            test_data: Test input tensor
            batch_size: Batch size for inference
            device: Device to run on
            num_runs: Number of validation runs
        """
        logger.info(f"Validating optimized model on {device}")
        
        for run_idx in range(num_runs):
            try:
                # Prepare input
                if len(test_data.shape) == 1:
                    input_ids = test_data.unsqueeze(0).to(device)
                else:
                    input_ids = test_data[:batch_size].to(device)
                
                # Measure TTFT
                ttft_mean, ttft_std = self.measure_ttft(model, input_ids, device, num_runs=1)
                
                # Measure throughput
                throughput = self.measure_throughput(model, input_ids, num_tokens_generate=50, device=device)
                
                # Measure memory
                peak_mem, avg_mem = self.measure_memory(model, device)
                
                # Create metric record with speedup calculations
                metric = InferenceMetric(
                    model_type="optimized",
                    batch_size=batch_size,
                    sequence_length=input_ids.shape[1] if len(input_ids.shape) > 1 else 1,
                    ttft_ms=ttft_mean,
                    throughput_tokens_sec=throughput,
                    peak_memory_mb=peak_mem,
                    average_memory_mb=avg_mem,
                    inference_successful=True
                )
                
                self.results.optimized_metrics.append(metric)
                logger.info(
                    f"Optimized run {run_idx + 1}: "
                    f"TTFT={ttft_mean:.2f}ms, "
                    f"Throughput={throughput:.2f} tok/s, "
                    f"Memory={peak_mem:.2f}MB"
                )
                
            except Exception as e:
                logger.error(f"Optimized validation run {run_idx + 1} failed: {e}")
                metric = InferenceMetric(
                    model_type="optimized",
                    batch_size=batch_size,
                    sequence_length=0,
                    inference_successful=False,
                    error_message=str(e)
                )
                self.results.optimized_metrics.append(metric)
        
        # Compute aggregated statistics
        if self.results.optimized_metrics:
            successful_metrics = [m for m in self.results.optimized_metrics if m.inference_successful]
            if successful_metrics:
                ttfts = [m.ttft_ms for m in successful_metrics]
                throughputs = [m.throughput_tokens_sec for m in successful_metrics]
                
                self.results.optimized_ttft_mean_ms = statistics.mean(ttfts)
                self.results.optimized_throughput_mean = statistics.mean(throughputs)
                self.results.optimized_peak_memory_mb = successful_metrics[0].peak_memory_mb
                self.results.optimized_success_rate = len(successful_metrics) / len(self.results.optimized_metrics)
    
    def compute_speedups(self) -> None:
        """Compute speedup metrics comparing baseline vs optimized."""
        # TTFT speedup
        if self.results.baseline_ttft_mean_ms > 0:
            self.results.ttft_speedup = self.results.baseline_ttft_mean_ms / self.results.optimized_ttft_mean_ms
        
        # Throughput improvement
        if self.results.baseline_throughput_mean > 0:
            self.results.throughput_improvement = self.results.optimized_throughput_mean / self.results.baseline_throughput_mean
        
        # Memory reduction
        if self.results.baseline_peak_memory_mb > 0:
            self.results.memory_reduction_percent = (
                (self.results.baseline_peak_memory_mb - self.results.optimized_peak_memory_mb) /
                self.results.baseline_peak_memory_mb * 100
            )
        
        # Comparison vs Phase 1 targets
        if self.phase1_ttft_target_ms > 0:
            self.results.ttft_vs_phase1 = self.phase1_ttft_target_ms / self.results.optimized_ttft_mean_ms
        
        if self.phase1_throughput_target > 0:
            self.results.throughput_vs_phase1 = self.results.optimized_throughput_mean / self.phase1_throughput_target
        
        logger.info(
            f"Speedups computed: "
            f"TTFT={self.results.ttft_speedup:.2f}x, "
            f"Throughput={self.results.throughput_improvement:.2f}x, "
            f"Memory reduction={self.results.memory_reduction_percent:.1f}%"
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Validation report dictionary
        """
        self.compute_speedups()
        
        report = {
            "validation_id": self.results.validation_id,
            "timestamp": self.results.validation_timestamp,
            "baseline": {
                "ttft_mean_ms": round(self.results.baseline_ttft_mean_ms, 2),
                "throughput_mean_tokens_sec": round(self.results.baseline_throughput_mean, 2),
                "peak_memory_mb": round(self.results.baseline_peak_memory_mb, 2),
                "success_rate": round(self.results.baseline_success_rate, 3),
                "num_runs": len(self.results.baseline_metrics)
            },
            "optimized": {
                "ttft_mean_ms": round(self.results.optimized_ttft_mean_ms, 2),
                "throughput_mean_tokens_sec": round(self.results.optimized_throughput_mean, 2),
                "peak_memory_mb": round(self.results.optimized_peak_memory_mb, 2),
                "success_rate": round(self.results.optimized_success_rate, 3),
                "num_runs": len(self.results.optimized_metrics)
            },
            "speedups": {
                "ttft_speedup": round(self.results.ttft_speedup, 2),
                "throughput_improvement": round(self.results.throughput_improvement, 2),
                "memory_reduction_percent": round(self.results.memory_reduction_percent, 1)
            },
            "phase1_comparison": {
                "ttft_target_ms": self.phase1_ttft_target_ms,
                "throughput_target_tokens_sec": self.phase1_throughput_target,
                "ttft_vs_target": round(self.results.ttft_vs_phase1, 2),
                "throughput_vs_target": round(self.results.throughput_vs_phase1, 2)
            },
            "notes": self.results.notes,
            "all_baseline_metrics": [asdict(m) for m in self.results.baseline_metrics],
            "all_optimized_metrics": [asdict(m) for m in self.results.optimized_metrics]
        }
        
        return report
    
    def save_report(self, output_filename: str = "inference_validation_report.json") -> Path:
        """
        Save validation report to JSON file.
        
        Args:
            output_filename: Output filename (default: inference_validation_report.json)
            
        Returns:
            Path to saved report file
        """
        report = self.generate_report()
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize validator
    validator = ModelInferenceValidator(output_dir="reports")
    
    # In real usage:
    # baseline_model = validator.load_model("checkpoints/baseline_model.pt")
    # optimized_model = validator.load_model("checkpoints/optimized_model.pt")
    # test_data = torch.randint(0, 32000, (100, 512))
    #
    # validator.validate_baseline(baseline_model, test_data, num_runs=5)
    # validator.validate_optimized(optimized_model, test_data, num_runs=5)
    # report_path = validator.save_report()
    
    logger.info("ModelInferenceValidator ready for use")
