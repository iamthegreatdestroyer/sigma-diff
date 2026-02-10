#!/usr/bin/env python3
"""
Phase 2: Optimization Validator

Validates that all optimizations are working correctly:
- Kernel implementations match baseline inference
- Compression maintains accuracy within thresholds
- Speedup targets are met (3-5x baseline)
- Memory efficiency improvements verified
- E2E correctness across all layers

Outputs:
- validation_report.json: Pass/fail for each optimization
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status values."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    status: ValidationStatus
    actual: Any
    expected: Any
    tolerance: float = 0.0
    message: str = ""
    timestamp: str = field(default_factory=lambda: "")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'actual': self.actual,
            'expected': self.expected,
            'tolerance': self.tolerance,
            'message': self.message,
            'timestamp': self.timestamp
        }


class OptimizationValidator:
    """
    Validate all optimization components.
    
    Checks:
    1. Kernel Performance: Speedup, memory, latency
    2. Embedding Compression: Accuracy preservation, ratio
    3. KV Cache Optimization: Memory savings, latency
    4. Speculative Decoding: Acceptance rate, overhead
    5. E2E System: Overall speedup, correctness
    """
    
    def __init__(self):
        """Initialize validator."""
        self.results: List[ValidationResult] = []
        
        # Optimization thresholds
        self.speedup_min = 3.0  # Minimum 3x speedup
        self.speedup_max = 10.0  # Not unrealistic-ly high
        self.accuracy_loss_max = 0.01  # Max 1% accuracy drop
        self.compression_ratio_min = 2.0  # Minimum 2x compression
        self.latency_improvement_min = 0.2  # Min 20% improvement
        
        logger.info("OptimizationValidator initialized")
    
    def validate_kernel_performance(
        self,
        baseline_latency_ms: float,
        optimized_latency_ms: float,
        baseline_memory_mb: float,
        optimized_memory_mb: float
    ) -> ValidationResult:
        """
        Validate kernel performance improvements.
        
        Args:
            baseline_latency_ms: Baseline latency in ms
            optimized_latency_ms: Optimized latency in ms
            baseline_memory_mb: Baseline memory in MB
            optimized_memory_mb: Optimized memory in MB
            
        Returns:
            ValidationResult for kernel performance
        """
        speedup = baseline_latency_ms / optimized_latency_ms if optimized_latency_ms > 0 else 0
        memory_reduction = (baseline_memory_mb - optimized_memory_mb) / baseline_memory_mb
        
        status = ValidationStatus.PASS
        message = f"Kernel speedup: {speedup:.2f}x, Memory reduction: {memory_reduction*100:.1f}%"
        
        if speedup < self.speedup_min:
            status = ValidationStatus.FAIL
            message += f" - Speedup below target ({self.speedup_min:.1f}x)"
        elif speedup < 2.5:
            status = ValidationStatus.WARNING
            message += " - Speedup below optimal"
        
        if memory_reduction < 0.1:
            message += " - Minimal memory improvement"
        
        result = ValidationResult(
            name="Kernel Performance",
            status=status,
            actual={'speedup': speedup, 'memory_reduction': memory_reduction},
            expected={'speedup': self.speedup_min, 'memory_reduction': 0.3},
            message=message
        )
        
        self.results.append(result)
        return result
    
    def validate_embedding_compression(
        self,
        compression_ratio: float,
        baseline_accuracy: float,
        compressed_accuracy: float
    ) -> ValidationResult:
        """
        Validate embedding compression effectiveness.
        
        Args:
            compression_ratio: Compression ratio (higher is better)
            baseline_accuracy: Baseline model accuracy
            compressed_accuracy: Accuracy with compression
            
        Returns:
            ValidationResult for embedding compression
        """
        accuracy_drop = baseline_accuracy - compressed_accuracy
        accuracy_drop_percent = accuracy_drop / baseline_accuracy if baseline_accuracy > 0 else 0
        
        status = ValidationStatus.PASS
        message = f"Compression: {compression_ratio:.2f}x, Accuracy drop: {accuracy_drop_percent*100:.2f}%"
        
        # Check compression ratio
        if compression_ratio < self.compression_ratio_min:
            status = ValidationStatus.FAIL
            message += f" - Compression below target ({self.compression_ratio_min:.1f}x)"
        
        # Check accuracy preservation
        if accuracy_drop_percent > self.accuracy_loss_max:
            status = ValidationStatus.FAIL
            message += f" - Accuracy drop exceeds tolerance ({self.accuracy_loss_max*100:.1f}%)"
        elif accuracy_drop_percent > self.accuracy_loss_max * 0.5:
            status = ValidationStatus.WARNING
            message += " - Accuracy drop moderate"
        
        result = ValidationResult(
            name="Embedding Compression",
            status=status,
            actual={'compression_ratio': compression_ratio, 'accuracy_loss': accuracy_drop_percent},
            expected={'compression_ratio': self.compression_ratio_min, 'accuracy_loss': self.accuracy_loss_max},
            message=message
        )
        
        self.results.append(result)
        return result
    
    def validate_kv_cache_optimization(
        self,
        baseline_memory_mb: float,
        optimized_memory_mb: float,
        baseline_latency_ms: float,
        optimized_latency_ms: float
    ) -> ValidationResult:
        """
        Validate KV cache optimization.
        
        Args:
            baseline_memory_mb: Baseline memory usage
            optimized_memory_mb: Optimized memory usage
            baseline_latency_ms: Baseline latency
            optimized_latency_ms: Optimized latency
            
        Returns:
            ValidationResult for KV cache optimization
        """
        memory_savings = (baseline_memory_mb - optimized_memory_mb) / baseline_memory_mb
        latency_improvement = (baseline_latency_ms - optimized_latency_ms) / baseline_latency_ms
        
        status = ValidationStatus.PASS
        message = f"Memory savings: {memory_savings*100:.1f}%, Latency improvement: {latency_improvement*100:.1f}%"
        
        if memory_savings < 0.2:
            status = ValidationStatus.WARNING
            message += " - Minimal memory savings"
        
        if latency_improvement < self.latency_improvement_min:
            status = ValidationStatus.WARNING
            message += f" - Latency improvement below target ({self.latency_improvement_min*100:.0f}%)"
        
        result = ValidationResult(
            name="KV Cache Optimization",
            status=status,
            actual={'memory_savings': memory_savings, 'latency_improvement': latency_improvement},
            expected={'memory_savings': 0.3, 'latency_improvement': self.latency_improvement_min},
            message=message
        )
        
        self.results.append(result)
        return result
    
    def validate_speculative_decoding(
        self,
        acceptance_rate: float,
        overhead_percent: float,
        baseline_latency_ms: float,
        speculative_latency_ms: float
    ) -> ValidationResult:
        """
        Validate speculative decoding effectiveness.
        
        Args:
            acceptance_rate: Percentage of speculative tokens accepted
            overhead_percent: Overhead of speculation (% increase in compute)
            baseline_latency_ms: Baseline decoding latency
            speculative_latency_ms: Latency with speculation
            
        Returns:
            ValidationResult for speculative decoding
        """
        speedup = baseline_latency_ms / speculative_latency_ms if speculative_latency_ms > 0 else 1.0
        net_benefit = speedup > 1.0 and acceptance_rate > 0.5
        
        status = ValidationStatus.PASS if net_benefit else ValidationStatus.WARNING
        message = f"Acceptance rate: {acceptance_rate*100:.1f}%, Overhead: {overhead_percent:.1f}%, Speedup: {speedup:.2f}x"
        
        if acceptance_rate < 0.5:
            status = ValidationStatus.WARNING
            message += " - Low acceptance rate reduces benefit"
        
        if overhead_percent > 50:
            status = ValidationStatus.WARNING
            message += " - High speculation overhead"
        
        if speedup < 1.0:
            status = ValidationStatus.FAIL
            message += " - Speculation slower than baseline"
        
        result = ValidationResult(
            name="Speculative Decoding",
            status=status,
            actual={'acceptance_rate': acceptance_rate, 'overhead': overhead_percent, 'speedup': speedup},
            expected={'acceptance_rate': 0.7, 'overhead': 20.0, 'speedup': 1.5},
            message=message
        )
        
        self.results.append(result)
        return result
    
    def validate_end_to_end_system(
        self,
        baseline_latency_ms: float,
        optimized_latency_ms: float,
        baseline_memory_mb: float,
        optimized_memory_mb: float,
        baseline_accuracy: float,
        optimized_accuracy: float
    ) -> ValidationResult:
        """
        Validate end-to-end system performance.
        
        Args:
            baseline_latency_ms: Baseline E2E latency
            optimized_latency_ms: Optimized E2E latency
            baseline_memory_mb: Baseline memory
            optimized_memory_mb: Optimized memory
            baseline_accuracy: Baseline accuracy
            optimized_accuracy: Optimized accuracy
            
        Returns:
            ValidationResult for E2E system
        """
        speedup = baseline_latency_ms / optimized_latency_ms if optimized_latency_ms > 0 else 1.0
        memory_reduction = (baseline_memory_mb - optimized_memory_mb) / baseline_memory_mb
        accuracy_drop = (baseline_accuracy - optimized_accuracy) / baseline_accuracy
        
        # E2E pass: speedup >= 3x, memory reduction >= 20%, accuracy drop <= 1%
        all_pass = (
            speedup >= self.speedup_min and
            memory_reduction >= 0.2 and
            accuracy_drop <= self.accuracy_loss_max
        )
        
        status = ValidationStatus.PASS if all_pass else ValidationStatus.FAIL
        message = f"E2E Speedup: {speedup:.2f}x, Memory reduction: {memory_reduction*100:.1f}%, Accuracy drop: {accuracy_drop*100:.2f}%"
        
        if not all_pass:
            reasons = []
            if speedup < self.speedup_min:
                reasons.append(f"speedup {speedup:.2f}x < {self.speedup_min:.1f}x")
            if memory_reduction < 0.2:
                reasons.append(f"memory savings {memory_reduction*100:.1f}% < 20%")
            if accuracy_drop > self.accuracy_loss_max:
                reasons.append(f"accuracy loss {accuracy_drop*100:.2f}% > {self.accuracy_loss_max*100:.1f}%")
            message += " - Failed: " + ", ".join(reasons)
        
        result = ValidationResult(
            name="End-to-End System",
            status=status,
            actual={'speedup': speedup, 'memory_reduction': memory_reduction, 'accuracy_loss': accuracy_drop},
            expected={
                'speedup': self.speedup_min,
                'memory_reduction': 0.3,
                'accuracy_loss': self.accuracy_loss_max
            },
            message=message
        )
        
        self.results.append(result)
        return result
    
    def validate_correctness(
        self,
        baseline_output: np.ndarray,
        optimized_output: np.ndarray,
        tolerance: float = 1e-5
    ) -> ValidationResult:
        """
        Validate numerical correctness of optimizations.
        
        Args:
            baseline_output: Baseline model output
            optimized_output: Optimized model output
            tolerance: Maximum allowed difference
            
        Returns:
            ValidationResult for correctness
        """
        if baseline_output.shape != optimized_output.shape:
            status = ValidationStatus.FAIL
            message = f"Output shape mismatch: {baseline_output.shape} vs {optimized_output.shape}"
            max_diff = float('inf')
            mean_diff = float('inf')
        else:
            diff = np.abs(baseline_output - optimized_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            status = ValidationStatus.PASS if max_diff <= tolerance else ValidationStatus.FAIL
            message = f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}, Tolerance: {tolerance:.2e}"
        
        result = ValidationResult(
            name="Numerical Correctness",
            status=status,
            actual={'max_diff': float(max_diff), 'mean_diff': float(mean_diff)},
            expected={'max_diff': tolerance},
            tolerance=tolerance,
            message=message
        )
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.results:
            return {'status': 'NO_TESTS_RUN', 'total': 0}
        
        passed = sum(1 for r in self.results if r.status == ValidationStatus.PASS)
        failed = sum(1 for r in self.results if r.status == ValidationStatus.FAIL)
        warnings = sum(1 for r in self.results if r.status == ValidationStatus.WARNING)
        
        overall = ValidationStatus.PASS
        if failed > 0:
            overall = ValidationStatus.FAIL
        elif warnings > 0:
            overall = ValidationStatus.WARNING
        
        return {
            'overall_status': overall.value,
            'passed': passed,
            'failed': failed,
            'warnings': warnings,
            'total': len(self.results),
            'pass_rate': passed / len(self.results) if self.results else 0
        }
    
    def export_validation_report(self, output_path: str) -> None:
        """
        Export validation report to JSON.
        
        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'summary': self.get_summary(),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report exported to {output_path}")
    
    def print_summary(self) -> None:
        """Print validation summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("OPTIMIZATION VALIDATION SUMMARY")
        print("="*60)
        
        print(f"\nOverall Status: {summary.get('overall_status', 'UNKNOWN')}")
        print(f"  Passed: {summary.get('passed', 0)}/{summary.get('total', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}/{summary.get('total', 0)}")
        print(f"  Warnings: {summary.get('warnings', 0)}/{summary.get('total', 0)}")
        print(f"  Pass Rate: {summary.get('pass_rate', 0)*100:.1f}%")
        
        if self.results:
            print("\nDetailed Results:")
            for result in self.results:
                status_symbol = "✅" if result.status == ValidationStatus.PASS else "❌" if result.status == ValidationStatus.FAIL else "⚠️"
                print(f"\n{status_symbol} {result.name}")
                print(f"   {result.message}")
        
        print("\n" + "="*60)
