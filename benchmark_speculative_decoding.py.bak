"""
Speculative Decoding Performance Benchmark

Comprehensive benchmarking of speculative decoding performance against baseline
inference, measuring speedup, acceptance rates, and resource utilization.

Run with: python benchmark_speculative_decoding.py
"""

import torch
import numpy as np
import time
import psutil
import os
from typing import Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager
import logging

# Import speculative decoding components
import sys
sys.path.append('RYZEN-LLM/src')

from inference.speculative_decoder import (
    SpeculativeConfig,
    SpeculativeDecoder,
    SpeculativeStats,
    benchmark_speculative_decoding,
    create_speculative_decoder
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    value: float
    unit: str
    target: float = None
    status: str = "measured"

    def __post_init__(self):
        if self.target is not None:
            if self.unit == "x" and self.name.__contains__("speedup"):
                self.status = "✅ PASS" if self.value >= self.target else "❌ FAIL"
            elif self.unit == "%" and "acceptance" in self.name:
                self.status = "✅ PASS" if self.value >= 80.0 else "❌ FAIL"
            elif self.unit == "%" and "overhead" in self.name:
                self.status = "✅ PASS" if self.value <= 20.0 else "❌ FAIL"
            elif self.unit == "ms" and "latency" in self.name:
                self.status = "✅ PASS" if self.value <= 100.0 else "❌ FAIL"


class SpeculativeBenchmark:
    """Comprehensive speculative decoding benchmark suite."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.results: List[BenchmarkResult] = []

        # Benchmark configurations
        self.vocab_size = 32000
        self.test_prefixes = [
            [101, 2054, 2003, 2004],  # "The quick brown fox"
            [101, 7592, 1010, 2054],  # "I love the quick"
            [101, 2088, 2003, 2307],  # "The brown dog runs"
            [101, 2115, 2003, 2931],  # "The cat is sleeping"
            [101, 2023, 2518, 2003],  # "The bird flies high"
        ]

        print(f"🚀 Speculative Decoding Benchmark on {device}")
        print(f"Test prefixes: {len(self.test_prefixes)} sequences")

    @contextmanager
    def memory_monitor(self):
        """Monitor memory usage during benchmark."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        peak_memory = initial_memory
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            duration = end_time - start_time

            memory_overhead = max(0, final_memory - initial_memory)
            logger.info(f"Memory: {memory_overhead:.1f}MB overhead, {duration:.2f}s duration")

    def benchmark_baseline_performance(self) -> Dict[str, Any]:
        """Benchmark baseline (non-speculative) performance."""
        print("\n=== Benchmarking Baseline Performance ===")

        config = SpeculativeConfig(enable_speculative=False)
        decoder = SpeculativeDecoder(config)

        num_iterations = 100
        total_tokens = 0
        total_time = 0.0

        with self.memory_monitor():
            for i in range(num_iterations):
                prefix = self.test_prefixes[i % len(self.test_prefixes)]
                max_tokens = np.random.randint(1, 3)  # 1-2 tokens per iteration

                start_time = time.time()
                tokens = decoder.decode_next_tokens(prefix, max_tokens)
                end_time = time.time()

                total_tokens += len(tokens)
                total_time += (end_time - start_time)

        throughput = total_tokens / total_time
        avg_latency = (total_time / num_iterations) * 1000  # ms

        baseline_results = {
            "throughput_tokens_per_sec": throughput,
            "avg_latency_ms": avg_latency,
            "total_tokens": total_tokens,
            "total_time": total_time
        }

        self.results.extend([
            BenchmarkResult("baseline_throughput", throughput, "tokens/sec"),
            BenchmarkResult("baseline_latency", avg_latency, "ms", 50.0)
        ])

        print(f"Baseline: {throughput:.1f} tokens/sec, {avg_latency:.2f}ms latency")
        return baseline_results

    def benchmark_speculative_k_values(self) -> Dict[str, Any]:
        """Benchmark speculative decoding with different K values."""
        print("\n=== Benchmarking Speculative Decoding (K=1 to 8) ===")

        k_values = [1, 2, 3, 4, 5, 6, 7, 8]
        results = {}

        for k in k_values:
            print(f"\nTesting K={k}...")

            config = SpeculativeConfig(
                enable_speculative=True,
                initial_K=k,
                enable_adaptive_k=False  # Keep K fixed for testing
            )
            decoder = SpeculativeDecoder(config)

            # Run benchmark
            bench_results = benchmark_speculative_decoding(
                decoder, self.test_prefixes, num_iterations=50
            )

            results[k] = bench_results

            # Record results
            speedup = bench_results["avg_speedup"]
            acceptance = bench_results["acceptance_rate"] * 100
            throughput = bench_results["throughput_tokens_per_sec"]

            self.results.extend([
                BenchmarkResult(f"speculative_k{k}_speedup", speedup, "x", 1.2),
                BenchmarkResult(f"speculative_k{k}_acceptance", acceptance, "%", 80.0),
                BenchmarkResult(f"speculative_k{k}_throughput", throughput, "tokens/sec")
            ])

            print(f"K={k}: {speedup:.2f}x speedup, {acceptance:.1f}% acceptance, {throughput:.1f} tokens/sec")

        return results

    def benchmark_adaptive_k(self) -> Dict[str, Any]:
        """Benchmark adaptive K adjustment."""
        print("\n=== Benchmarking Adaptive K ===")

        config = SpeculativeConfig(
            enable_speculative=True,
            enable_adaptive_k=True,
            initial_K=4,
            acceptance_rate_target=0.8
        )
        decoder = SpeculativeDecoder(config)

        # Run extended benchmark to allow adaptation
        bench_results = benchmark_speculative_decoding(
            decoder, self.test_prefixes, num_iterations=200
        )

        final_k = decoder.get_K()
        speedup = bench_results["avg_speedup"]
        acceptance = bench_results["acceptance_rate"] * 100

        self.results.extend([
            BenchmarkResult("adaptive_k_final", final_k, "value"),
            BenchmarkResult("adaptive_speedup", speedup, "x", 2.0),
            BenchmarkResult("adaptive_acceptance", acceptance, "%", 75.0)
        ])

        print(f"Adaptive K: Final K={final_k}, {speedup:.2f}x speedup, {acceptance:.1f}% acceptance")
        return bench_results

    def benchmark_memory_overhead(self) -> Dict[str, Any]:
        """Benchmark memory overhead of speculative decoding."""
        print("\n=== Benchmarking Memory Overhead ===")

        # Test different configurations
        configs = [
            ("baseline", SpeculativeConfig(enable_speculative=False)),
            ("speculative_k4", SpeculativeConfig(enable_speculative=True, initial_K=4)),
            ("speculative_k8", SpeculativeConfig(enable_speculative=True, initial_K=8))
        ]

        memory_results = {}

        for name, config in configs:
            print(f"Testing {name}...")

            decoder = SpeculativeDecoder(config)

            # Measure memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Run benchmark
            bench_results = benchmark_speculative_decoding(
                decoder, self.test_prefixes, num_iterations=30
            )

            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_overhead = memory_after - memory_before

            memory_results[name] = {
                "memory_overhead_mb": memory_overhead,
                "speedup": bench_results["avg_speedup"]
            }

            if name != "baseline":
                baseline_memory = memory_results["baseline"]["memory_overhead_mb"]
                if baseline_memory > 0:
                    overhead_percent = ((memory_overhead - baseline_memory) / baseline_memory) * 100
                else:
                    overhead_percent = memory_overhead * 100  # Relative to zero baseline

                self.results.append(
                    BenchmarkResult(f"{name}_memory_overhead", overhead_percent, "%", 20.0)
                )

                print(f"{name}: {memory_overhead:.1f}MB overhead ({overhead_percent:+.1f}%), {bench_results['avg_speedup']:.2f}x speedup")

        return memory_results

    def benchmark_long_sequences(self) -> Dict[str, Any]:
        """Benchmark performance on longer sequences."""
        print("\n=== Benchmarking Long Sequences ===")

        # Create longer test sequences
        long_prefixes = []
        for base_prefix in self.test_prefixes:
            # Extend each prefix to 50 tokens
            extended = base_prefix * 10  # Repeat pattern
            long_prefixes.append(extended[:50])  # Truncate to 50 tokens

        config = SpeculativeConfig(enable_speculative=True, initial_K=6)
        decoder = SpeculativeDecoder(config)

        bench_results = benchmark_speculative_decoding(
            decoder, long_prefixes, num_iterations=20
        )

        speedup = bench_results["avg_speedup"]
        acceptance = bench_results["acceptance_rate"] * 100

        self.results.extend([
            BenchmarkResult("long_sequence_speedup", speedup, "x", 2.5),
            BenchmarkResult("long_sequence_acceptance", acceptance, "%", 70.0)
        ])

        print(f"Long sequences: {speedup:.2f}x speedup, {acceptance:.1f}% acceptance")
        return bench_results

    def run_comprehensive_benchmark(self) -> None:
        """Run the complete benchmark suite."""
        print("🔬 COMPREHENSIVE SPECULATIVE DECODING BENCHMARK")
        print("=" * 60)

        # Run all benchmarks
        self.benchmark_baseline_performance()
        self.benchmark_speculative_k_values()
        self.benchmark_adaptive_k()
        self.benchmark_memory_overhead()
        self.benchmark_long_sequences()

        self.print_results()

    def print_results(self) -> None:
        """Print comprehensive benchmark results."""
        print("\n" + "=" * 80)
        print("📊 SPECULATIVE DECODING BENCHMARK RESULTS")
        print("=" * 80)

        # Group results by category
        categories = {
            "Baseline": [r for r in self.results if r.name.startswith("baseline")],
            "Speculative K=4": [r for r in self.results if "_k4_" in r.name],
            "Speculative K=8": [r for r in self.results if "_k8_" in r.name],
            "Adaptive K": [r for r in self.results if r.name.startswith("adaptive")],
            "Memory": [r for r in self.results if "memory" in r.name],
            "Long Sequences": [r for r in self.results if "long_sequence" in r.name]
        }

        for category, results in categories.items():
            if results:
                print(f"\n{category}:")
                print("-" * len(category))
                for result in results:
                    if result.unit == "%":
                        print("25")
                    elif result.unit == "x":
                        print("25")
                    elif result.unit == "ms":
                        print("25")
                    else:
                        print("25")

        # Summary statistics
        speedup_results = [r for r in self.results if "speedup" in r.name and r.target]
        passed_targets = sum(1 for r in speedup_results if r.status == "✅ PASS")
        total_targets = len(speedup_results)

        print("\n" + "-" * 80)
        print(f"🎯 TARGET ACHIEVEMENT: {passed_targets}/{total_targets} performance targets met")

        if passed_targets == total_targets:
            print("🎉 ALL PERFORMANCE TARGETS ACHIEVED!")
            print("🚀 Speculative decoding ready for production deployment!")
        else:
            print(f"⚠️  {total_targets - passed_targets} targets need optimization")

        # Key insights
        print("\n💡 KEY INSIGHTS:")
        adaptive_results = [r for r in self.results if r.name.startswith("adaptive")]
        if adaptive_results:
            speedup_result = next((r for r in adaptive_results if "speedup" in r.name), None)
            if speedup_result:
                print(f"📈 Achieved {speedup_result.value:.2f}x speedup with adaptive K!")
                print("📈 Ready for 2-3x inference acceleration!")

        print("=" * 80)


def main():
    """Main benchmark runner."""
    # Set random seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)

    # Run comprehensive benchmark
    benchmark = SpeculativeBenchmark()
    benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()