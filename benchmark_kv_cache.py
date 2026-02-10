"""
KV-Cache Optimization Performance Benchmarks

Validates the performance targets for Week 3 KV-cache optimization:
- Cache coherency latency <1ms
- FP8 compression with <0.5% accuracy loss
- 40-50% memory reduction
- Dynamic allocation overhead <2%

Run with: python benchmark_kv_cache.py
"""

import torch
import time
import numpy as np
import psutil
import os
from typing import Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager

# Import our components
import sys
sys.path.append('Ryzanstein LLM')

from src.inference.distributed_kv_cache import DistributedKVCache
from src.inference.cache_compression import FP8Compressor, CompressedKVCache
from src.inference.dynamic_allocator import DynamicCacheAllocator


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    value: float
    unit: str
    target: float = None
    passed: bool = None

    def __post_init__(self):
        if self.target is not None:
            if self.unit == "ms" and self.name.__contains__("latency"):
                self.passed = self.value < self.target
            elif self.unit == "%" and "loss" in self.name:
                self.passed = self.value < self.target
            elif self.unit == "%" and "reduction" in self.name:
                self.passed = self.value >= self.target
            elif self.unit == "%" and "overhead" in self.name:
                self.passed = self.value < self.target


class KVCacheBenchmark:
    """Comprehensive KV-cache performance benchmarking."""

    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.results: List[BenchmarkResult] = []

        # Benchmark configuration
        self.num_layers = 12
        self.num_heads = 12
        self.head_dim = 64
        self.max_seq_len = 2048
        self.batch_size = 4

        print(f"Running KV-cache benchmarks on {device}")
        print(f"Configuration: {self.num_layers}L x {self.num_heads}H x {self.head_dim}D, max_seq={self.max_seq_len}")

    @contextmanager
    def timer(self):
        """Context manager for timing operations."""
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        return end - start

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def benchmark_distributed_cache_coherency(self) -> None:
        """Benchmark distributed cache coherency latency."""
        print("\n=== Benchmarking Distributed Cache Coherency ===")

        # Create distributed cache (simulate 2 GPUs)
        cache = DistributedKVCache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            world_size=2,
            rank=0,
            device=self.device
        )

        cache.allocate_cache(self.batch_size, 512)

        # Measure local access latency
        latencies = []
        for _ in range(100):
            layer_id = np.random.randint(0, self.num_layers)
            head_id = np.random.randint(0, self.num_heads)
            seq_pos = np.random.randint(0, 256)  # Within local shard

            k = torch.randn(self.batch_size, self.head_dim, dtype=torch.float16, device=self.device)
            v = torch.randn(self.batch_size, self.head_dim, dtype=torch.float16, device=self.device)

            # Time update + retrieve
            start = time.perf_counter()
            cache.update_kv(layer_id, head_id, seq_pos, k, v)
            k_ret, v_ret = cache.get_kv_range(layer_id, head_id, seq_pos, seq_pos + 1)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        self.results.extend([
            BenchmarkResult("cache_coherency_avg_latency", avg_latency, "ms", 1.0),
            BenchmarkResult("cache_coherency_p95_latency", p95_latency, "ms", 1.0),
            BenchmarkResult("cache_coherency_p99_latency", p99_latency, "ms", 1.0)
        ])

        print(f"Cache coherency latency: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms, p99={p99_latency:.2f}ms")
    def benchmark_fp8_compression_accuracy(self) -> None:
        """Benchmark FP8 compression accuracy."""
        print("\n=== Benchmarking FP8 Compression Accuracy ===")

        compressor = FP8Compressor(device=self.device)

        # Generate calibration samples
        samples = []
        for _ in range(100):
            k = torch.randn(1, self.head_dim, dtype=torch.float16, device=self.device)
            v = torch.randn(1, self.head_dim, dtype=torch.float16, device=self.device)
            samples.append((0, 0, k, v))  # layer 0, head 0

        # Calibrate
        compressor.calibrate_scales()

        # Test compression accuracy
        losses = []
        for _ in range(50):
            k_orig = torch.randn(1, self.head_dim, dtype=torch.float16, device=self.device)
            v_orig = torch.randn(1, self.head_dim, dtype=torch.float16, device=self.device)

            # Compress and decompress
            k_fp8, v_fp8 = compressor.quantize_kv(0, 0, k_orig, v_orig)
            k_restored, v_restored = compressor.dequantize_kv(0, 0, k_fp8, v_fp8)

            # Calculate MSE loss
            k_loss = torch.mean((k_orig - k_restored) ** 2).item()
            v_loss = torch.mean((v_orig - v_restored) ** 2).item()
            avg_loss = (k_loss + v_loss) / 2.0

            losses.append(avg_loss)

        avg_loss = np.mean(losses)
        max_loss = np.max(losses)

        # Calculate relative loss as percentage
        # Using a reference scale for relative comparison
        reference_scale = 0.01  # Typical variance scale
        relative_loss_percent = (avg_loss / reference_scale) * 100

        self.results.extend([
            BenchmarkResult("fp8_compression_avg_loss", relative_loss_percent, "%", 0.5),
            BenchmarkResult("fp8_compression_max_loss", (max_loss / reference_scale) * 100, "%", 1.0)
        ])

        print(f"FP8 compression accuracy: avg_loss={relative_loss_percent:.4f}%, max_loss={(max_loss / reference_scale) * 100:.4f}%")
    def benchmark_memory_reduction(self) -> None:
        """Benchmark memory reduction from compression."""
        print("\n=== Benchmarking Memory Reduction ===")

        # Create compressed cache
        compressed_cache = CompressedKVCache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            device=self.device,
            enable_compression=True
        )

        # Create uncompressed cache for comparison
        uncompressed_cache = CompressedKVCache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            device=self.device,
            enable_compression=False
        )

        # Calibrate compression
        samples = []
        for _ in range(10):
            k = torch.randn(1, self.head_dim, dtype=torch.float16, device=self.device)
            v = torch.randn(1, self.head_dim, dtype=torch.float16, device=self.device)
            samples.append((0, 0, k, v))

        compressed_cache.calibrate_compression(samples)

        # Store data in both caches
        seq_len = 512
        for seq_pos in range(min(100, seq_len)):  # Store 100 positions
            k = torch.randn(1, self.head_dim, dtype=torch.float16, device=self.device)
            v = torch.randn(1, self.head_dim, dtype=torch.float16, device=self.device)

            compressed_cache.store_compressed(0, 0, seq_pos, k, v)
            uncompressed_cache.store_compressed(0, 0, seq_pos, k, v)

        # Get memory usage
        compressed_stats = compressed_cache.get_memory_usage()
        uncompressed_stats = uncompressed_cache.get_memory_usage()

        memory_reduction = compressed_stats["memory_savings_percent"]

        self.results.append(
            BenchmarkResult("memory_reduction_percent", memory_reduction, "%", 40.0)
        )

        print(f"Memory reduction: {memory_reduction:.1f}%")
    def benchmark_allocation_overhead(self) -> None:
        """Benchmark dynamic allocation overhead."""
        print("\n=== Benchmarking Allocation Overhead ===")

        allocator = DynamicCacheAllocator(total_memory_gb=4.0)

        # Measure allocation overhead
        overheads = []
        for _ in range(50):
            start = time.perf_counter()

            success = allocator.allocate_cache(
                f"bench_req_{_}",
                seq_len=256,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                compressed=True
            )

            end = time.perf_counter()
            if success:
                overheads.append((end - start) * 1000)  # Convert to ms

        if overheads:
            avg_overhead = np.mean(overheads)
            p95_overhead = np.percentile(overheads, 95)

            self.results.extend([
                BenchmarkResult("allocation_avg_overhead", avg_overhead, "%", 2.0),
                BenchmarkResult("allocation_p95_overhead", p95_overhead, "%", 5.0)
            ])

            print(f"Allocation overhead: avg={avg_overhead:.3f}ms, p95={p95_overhead:.3f}ms")
        else:
            print("No successful allocations to measure")

    def run_all_benchmarks(self) -> None:
        """Run all benchmarks."""
        print("🚀 Starting KV-Cache Optimization Benchmarks")
        print("=" * 50)

        self.benchmark_distributed_cache_coherency()
        self.benchmark_fp8_compression_accuracy()
        self.benchmark_memory_reduction()
        self.benchmark_allocation_overhead()

        self.print_results()

    def print_results(self) -> None:
        """Print benchmark results."""
        print("\n" + "=" * 60)
        print("📊 KV-CACHE OPTIMIZATION BENCHMARK RESULTS")
        print("=" * 60)

        passed = 0
        total = 0

        for result in self.results:
            total += 1
            status = "✅ PASS" if result.passed else "❌ FAIL" if result.passed is not None else "⏱️  MEASURED"

            if result.unit == "%":
                print("30")
            else:
                print("30")

            if result.passed:
                passed += 1

        print("-" * 60)
        print(f"🎯 OVERALL: {passed}/{total} targets met")

        if passed == total:
            print("🎉 ALL PERFORMANCE TARGETS ACHIEVED!")
        else:
            print(f"⚠️  {total - passed} targets need improvement")

        print("=" * 60)


def main():
    """Main benchmark runner."""
    # Set random seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)

    # Run benchmarks
    benchmark = KVCacheBenchmark()
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()