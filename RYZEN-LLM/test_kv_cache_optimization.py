#!/usr/bin/env python3
"""
KV-Cache Optimization Test Suite
=================================

Comprehensive testing for Sprint 1.2: KV-Cache Optimization for Distributed.
Validates dynamic allocation, cache coherency, and advanced compression.

Test Scenarios:
1. Dynamic cache allocation efficiency
2. Cache coherency latency (<1ms target)
3. Advanced compression (40-50% memory reduction)
4. Workload-adaptive cache sizing
5. Distributed eviction policies

Usage:
    # Single GPU cache optimization test
    python test_kv_cache_optimization.py --test_suite dynamic_allocation

    # Multi-GPU coherency test
    torchrun --nproc_per_node=4 test_kv_cache_optimization.py --test_suite coherency

    # Full optimization test suite
    torchrun --nproc_per_node=4 test_kv_cache_optimization.py --test_suite full
"""

import torch
import torch.distributed as dist
import argparse
import time
import logging
import json
import os
import sys
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from distributed.sharded_kv_cache import ShardedKVCache
from distributed.cache_coherency import CoherenceProtocol
from inference.dynamic_cache_allocator import DynamicCacheAllocator
from inference.advanced_compression import CompressionManager, CompressionType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Optimization test result data structure."""
    test_name: str
    world_size: int
    rank: int
    memory_reduction_percent: float
    coherence_latency_ms: float
    allocation_overhead_percent: float
    compression_ratio: float
    cache_hit_rate: float
    success: bool
    error_message: str = ""


class KVCacheOptimizerTester:
    """Comprehensive KV-cache optimization tester."""

    def __init__(self, world_size: int, rank: int, args: argparse.Namespace):
        self.world_size = world_size
        self.rank = rank
        self.args = args

        # Test configuration
        self.config = self._create_test_config()

        # Results storage
        self.results: List[OptimizationResult] = []

        # Cache instance
        self.cache: ShardedKVCache = None

    def _create_test_config(self):
        """Create test configuration."""
        return {
            'max_seq_len': 2048,
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'cache_memory_mb': 512.0,  # Smaller for testing
            'compression_ratio': 0.5
        }

    def setup_cache(self):
        """Setup optimized KV-cache."""
        logger.info(f"Setting up optimized KV-cache on rank {self.rank}/{self.world_size}")

        self.cache = ShardedKVCache(
            max_seq_len=self.config['max_seq_len'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            world_size=self.world_size,
            rank=self.rank,
            compression_ratio=self.config['compression_ratio'],
            cache_memory_mb=self.config['cache_memory_mb'],
            coherence_protocol=CoherenceProtocol.ADAPTIVE
        )

        logger.info("Optimized KV-cache setup complete")

    def run_dynamic_allocation_test(self) -> OptimizationResult:
        """Test dynamic cache allocation efficiency."""
        logger.info("Running dynamic cache allocation test")

        # Simulate workload with varying sequence lengths
        workloads = [
            (100, "short sequences"),
            (500, "medium sequences"),
            (1000, "long sequences"),
            (2000, "very long sequences")
        ]

        start_memory = self.cache.get_memory_usage()['total_cache_mb']
        allocation_overheads = []

        for seq_len, desc in workloads:
            logger.info(f"Testing {desc} (seq_len={seq_len})")

            # Create test tensors
            batch_size = 1
            key = torch.randn(batch_size, seq_len, self.config['num_heads'], self.config['hidden_size'] // self.config['num_heads'])
            value = torch.randn(batch_size, seq_len, self.config['num_heads'], self.config['hidden_size'] // self.config['num_heads'])

            # Measure allocation overhead
            start_time = time.time()
            for layer in range(min(3, self.config['num_layers'])):  # Test first 3 layers
                self.cache.update(layer, key, value, 0)
            allocation_time = time.time() - start_time
            allocation_overheads.append(allocation_time)

        end_memory = self.cache.get_memory_usage()['total_cache_mb']
        avg_allocation_overhead = sum(allocation_overheads) / len(allocation_overheads)

        # Calculate overhead percentage (target <2%)
        total_operations_time = sum(allocation_overheads)
        overhead_percent = (avg_allocation_overhead / total_operations_time) * 100 if total_operations_time > 0 else 0

        return OptimizationResult(
            test_name="dynamic_allocation",
            world_size=self.world_size,
            rank=self.rank,
            memory_reduction_percent=0.0,  # Not applicable
            coherence_latency_ms=0.0,       # Not applicable
            allocation_overhead_percent=overhead_percent,
            compression_ratio=self.cache.get_memory_usage().get('avg_compression_ratio', 1.0),
            cache_hit_rate=0.0,  # Not measured
            success=overhead_percent < 2.0  # Target <2%
        )

    def run_coherency_test(self) -> OptimizationResult:
        """Test cache coherency latency."""
        logger.info("Running cache coherency test")

        # Create test data
        batch_size = 1
        seq_len = 100
        key = torch.randn(batch_size, seq_len, self.config['num_heads'], self.config['hidden_size'] // self.config['num_heads'])
        value = torch.randn(batch_size, seq_len, self.config['num_heads'], self.config['hidden_size'] // self.config['num_heads'])

        # Update cache
        self.cache.update(0, key, value, 0)

        # Measure coherence latency for multiple reads
        latencies = []
        num_reads = 10

        for i in range(num_reads):
            start_time = time.time()
            k, v = self.cache.get(0, 0, seq_len)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        coherence_stats = self.cache.coherence_manager.get_coherence_stats()
        actual_coherence_latency = coherence_stats.get('avg_coherence_latency_ms', avg_latency)

        return OptimizationResult(
            test_name="coherency",
            world_size=self.world_size,
            rank=self.rank,
            memory_reduction_percent=0.0,  # Not applicable
            coherence_latency_ms=actual_coherence_latency,
            allocation_overhead_percent=0.0,  # Not applicable
            compression_ratio=1.0,  # Not applicable
            cache_hit_rate=0.0,  # Not measured
            success=actual_coherence_latency < 1.0  # Target <1ms
        )

    def run_compression_test(self) -> OptimizationResult:
        """Test advanced compression effectiveness."""
        logger.info("Running compression test")

        # Create test data with different patterns
        test_patterns = [
            ("random", torch.randn(1, 500, self.config['num_heads'], self.config['hidden_size'] // self.config['num_heads'])),
            ("sparse", torch.randn(1, 500, self.config['num_heads'], self.config['hidden_size'] // self.config['num_heads']) * (torch.rand(1, 500, self.config['num_heads'], self.config['hidden_size'] // self.config['num_heads']) > 0.8).float()),
            ("sequential", torch.arange(1, 500 * self.config['num_heads'] * (self.config['hidden_size'] // self.config['num_heads']) + 1).float().reshape(1, 500, self.config['num_heads'], self.config['hidden_size'] // self.config['num_heads']))
        ]

        compression_ratios = []

        for pattern_name, data in test_patterns:
            logger.info(f"Testing compression on {pattern_name} data")

            # Measure original size
            original_size = data.numel() * data.element_size()

            # Update cache (this applies compression)
            self.cache.update(0, data, data, 0)

            # Get compressed size from cache info
            cache_info = self.cache.get_cache_info()
            compression_stats = cache_info.get('compression', {}).get('stats', {})

            if compression_stats:
                avg_ratio = sum(
                    layer_stats.get('avg_compression_ratio', 1.0)
                    for layer_stats in compression_stats.values()
                ) / len(compression_stats)
                compression_ratios.append(avg_ratio)

        avg_compression_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 1.0
        memory_reduction = (1.0 - avg_compression_ratio) * 100

        return OptimizationResult(
            test_name="compression",
            world_size=self.world_size,
            rank=self.rank,
            memory_reduction_percent=memory_reduction,
            coherence_latency_ms=0.0,  # Not applicable
            allocation_overhead_percent=0.0,  # Not applicable
            compression_ratio=avg_compression_ratio,
            cache_hit_rate=0.0,  # Not measured
            success=memory_reduction >= 40.0  # Target 40-50% reduction
        )

    def run_workload_adaptive_test(self) -> OptimizationResult:
        """Test workload-adaptive cache sizing."""
        logger.info("Running workload-adaptive test")

        # Simulate different workload patterns
        patterns = [
            ("read_heavy", [0.8, 0.1, 0.1]),  # 80% reads, 10% writes, 10% mixed
            ("write_heavy", [0.1, 0.8, 0.1]),
            ("balanced", [0.33, 0.33, 0.34])
        ]

        for pattern_name, (read_ratio, write_ratio, mixed_ratio) in patterns:
            logger.info(f"Testing {pattern_name} workload")

            # Simulate workload
            for i in range(10):
                seq_len = 100 + i * 50  # Increasing sequence lengths
                key = torch.randn(1, seq_len, self.config['num_heads'], self.config['hidden_size'] // self.config['num_heads'])
                value = torch.randn(1, seq_len, self.config['num_heads'], self.config['hidden_size'] // self.config['num_heads'])

                # Apply pattern
                if torch.rand(1).item() < read_ratio:
                    # Read operation
                    if i > 0:  # Only read if we have data
                        self.cache.get(0, 0, min(seq_len, 100))
                elif torch.rand(1).item() < write_ratio + read_ratio:
                    # Write operation
                    self.cache.update(0, key, value, 0)
                else:
                    # Mixed operation
                    self.cache.update(0, key, value, 0)
                    self.cache.get(0, 0, min(seq_len, 100))

        # Check if allocator adapted to workload
        workload_stats = self.cache.cache_allocator.get_workload_stats()
        hot_layers = workload_stats.get('hot_layers', [])

        # Success if we have identified hot layers (workload adaptation working)
        success = len(hot_layers) > 0

        return OptimizationResult(
            test_name="workload_adaptive",
            world_size=self.world_size,
            rank=self.rank,
            memory_reduction_percent=0.0,  # Not applicable
            coherence_latency_ms=0.0,       # Not applicable
            allocation_overhead_percent=0.0,  # Not applicable
            compression_ratio=1.0,  # Not applicable
            cache_hit_rate=0.0,  # Not measured
            success=success
        )

    def run_test_suite(self, test_suite: str) -> List[OptimizationResult]:
        """Run complete optimization test suite."""
        logger.info(f"Running optimization test suite: {test_suite}")

        results = []

        try:
            if test_suite in ["dynamic_allocation", "full"]:
                results.append(self.run_dynamic_allocation_test())

            if test_suite in ["coherency", "full"]:
                results.append(self.run_coherency_test())

            if test_suite in ["compression", "full"]:
                results.append(self.run_compression_test())

            if test_suite in ["workload_adaptive", "full"]:
                results.append(self.run_workload_adaptive_test())

        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            # Add failure result
            results.append(OptimizationResult(
                test_name=test_suite,
                world_size=self.world_size,
                rank=self.rank,
                memory_reduction_percent=0.0,
                coherence_latency_ms=0.0,
                allocation_overhead_percent=0.0,
                compression_ratio=1.0,
                cache_hit_rate=0.0,
                success=False,
                error_message=str(e)
            ))

        return results

    def save_results(self, results: List[OptimizationResult], output_file: str):
        """Save test results to file."""
        if self.rank == 0:  # Only rank 0 saves results
            results_dict = {
                "timestamp": time.time(),
                "world_size": self.world_size,
                "sprint": "1.2",
                "objective": "KV-Cache Optimization for Distributed",
                "tests": [asdict(result) for result in results]
            }

            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)

            logger.info(f"Optimization results saved to {output_file}")

    def print_summary(self, results: List[OptimizationResult]):
        """Print optimization test summary."""
        if self.rank == 0:
            logger.info("=== SPRINT 1.2: KV-CACHE OPTIMIZATION RESULTS ===")
            for result in results:
                status = "✅ PASS" if result.success else "❌ FAIL"
                logger.info(f"{status} {result.test_name}:")

                if result.test_name == "compression":
                    logger.info(f"   Memory reduction: {result.metrics.get('memory_reduction', 0):.1f}%")
                    logger.info(f"   Accuracy loss: {result.metrics.get('accuracy_loss', 0):.3f}%")
                elif result.test_name == "coherency":
                    logger.info(f"   Cache hit rate: {result.metrics.get('cache_hit_rate', 0):.3f}%")
                    logger.info(f"   Sync latency: {result.metrics.get('sync_latency', 0):.1f}ms")
                elif result.test_name == "dynamic_allocation":
                    logger.info(f"   Allocation efficiency: {result.metrics.get('allocation_efficiency', 0):.1f}%")
                elif result.test_name == "workload_adaptive":
                    logger.info(f"   Workload adaptation: {'Working' if result.success else 'Failed'}")

                if not result.success:
                    logger.error(f"   Error: {result.error_message}")

            # Overall success criteria check
            compression_success = any(r.test_name == "compression" and r.success for r in results)
            coherency_success = any(r.test_name == "coherency" and r.success for r in results)
            allocation_success = any(r.test_name == "dynamic_allocation" and r.success for r in results)

            overall_success = compression_success and coherency_success and allocation_success

            if overall_success:
                logger.info("🎉 SPRINT 1.2 SUCCESS: All optimization targets met!")
                logger.info("   ✅ Memory reduction 40-50% with advanced compression")
                logger.info("   ✅ Cache coherency latency <1ms")
                logger.info("   ✅ Dynamic allocation overhead <2%")
            else:
                logger.error("❌ SPRINT 1.2: Some optimization targets not met")
                if not compression_success:
                    logger.error("   - Compression target not achieved")
                if not coherency_success:
                    logger.error("   - Coherency latency target not achieved")
                if not allocation_success:
                    logger.error("   - Dynamic allocation target not achieved")


def main():
    """Main optimization test function."""
    parser = argparse.ArgumentParser(description='KV-Cache Optimization Tests for Sprint 1.2')
    parser.add_argument('--world_size', type=int, default=1,
                       help='Number of processes/GPUs')
    parser.add_argument('--test_suite', type=str, default='full',
                       choices=['dynamic_allocation', 'coherency', 'compression', 'workload_adaptive', 'full'],
                       help='Test suite to run')
    parser.add_argument('--output_file', type=str, default='kv_cache_optimization_results.json',
                       help='Output file for results')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize distributed environment if multi-GPU
    if args.world_size > 1:
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1

    logger.info(f"Starting KV-cache optimization tests on rank {rank}/{world_size}")

    # Create tester
    tester = KVCacheOptimizerTester(world_size, rank, args)

    try:
        # Setup optimized cache
        tester.setup_cache()

        # Run test suite
        results = tester.run_test_suite(args.test_suite)

        # Save and print results
        tester.save_results(results, args.output_file)
        tester.print_summary(results)

    except Exception as e:
        logger.error(f"Optimization test execution failed: {e}")
        sys.exit(1)

    finally:
        # Cleanup
        if tester.cache:
            tester.cache.coherence_manager.shutdown()

    logger.info("KV-cache optimization tests completed")


if __name__ == '__main__':
    main()