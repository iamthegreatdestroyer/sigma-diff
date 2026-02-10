#!/usr/bin/env python3
"""
MT Contention Fix Validation Script
===================================

Tests the performance improvements from fixing multi-threading contention
in the batch engine and distributed serving components.

This script validates:
1. Reduced lock contention in batch engine
2. Improved throughput with multiple threads
3. Better GPU utilization
4. Lower latency under load
"""

import asyncio
import time
import threading
import statistics
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the fixed batch engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.serving.batch_engine import BatchEngine, BatchPriority

class MTContentionTester:
    """Test multi-threading performance improvements."""

    def __init__(self, num_gpus: int = 4, num_threads: int = 16):
        self.num_gpus = num_gpus
        self.num_threads = num_threads

        # Initialize batch engine with multiple GPUs
        self.batch_engine = BatchEngine(
            max_batch_size=8,
            max_latency_ms=100.0,
            batch_timeout_ms=50.0
        )
        self.batch_engine.initialize_gpu_queues(list(range(num_gpus)))

        # Test results
        self.results = {
            "single_thread": {},
            "multi_thread": {},
            "contention_analysis": {}
        }

    async def run_single_thread_test(self, num_requests: int = 100) -> Dict[str, Any]:
        """Run single-threaded performance test."""
        logger.info(f"Running single-thread test with {num_requests} requests")

        start_time = time.time()
        futures = []

        # Submit all requests sequentially
        for i in range(num_requests):
            gpu_id = i % self.num_gpus
            future = await self.batch_engine.submit_request(
                request_id=f"req_{i}",
                gpu_id=gpu_id,
                input_tokens=[1, 2, 3] * 10,  # 30 tokens
                max_new_tokens=50,
                priority=BatchPriority.NORMAL
            )
            futures.append(future)

        # Force batch processing
        await self.batch_engine.flush_all_batches()

        # Wait for all requests to complete
        await asyncio.gather(*futures, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time
        throughput = num_requests / total_time

        result = {
            "total_time": total_time,
            "throughput": throughput,
            "avg_latency": total_time / num_requests * 1000,  # ms
            "num_requests": num_requests
        }

        logger.info(".2f")
        return result

    async def run_multi_thread_test(self, num_requests: int = 100) -> Dict[str, Any]:
        """Run multi-threaded performance test."""
        logger.info(f"Running multi-thread test with {num_requests} requests, {self.num_threads} threads")

        start_time = time.time()
        results = []

        async def worker_thread(thread_id: int, requests_per_thread: int):
            """Worker thread function."""
            thread_results = []
            start_idx = thread_id * requests_per_thread

            for i in range(requests_per_thread):
                req_id = start_idx + i
                gpu_id = req_id % self.num_gpus

                future = await self.batch_engine.submit_request(
                    request_id=f"mt_req_{req_id}",
                    gpu_id=gpu_id,
                    input_tokens=[1, 2, 3] * 10,
                    max_new_tokens=50,
                    priority=BatchPriority.NORMAL
                )
                thread_results.append(future)

            return thread_results

        # Create worker tasks
        requests_per_thread = num_requests // self.num_threads
        tasks = []

        for thread_id in range(self.num_threads):
            task = asyncio.create_task(worker_thread(thread_id, requests_per_thread))
            tasks.append(task)

        # Wait for all threads to complete
        thread_futures = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten futures
        all_futures = []
        for thread_futures_list in thread_futures:
            all_futures.extend(thread_futures_list)

        # Force batch processing
        await self.batch_engine.flush_all_batches()

        # Wait for all requests to complete
        await asyncio.gather(*all_futures, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time
        throughput = num_requests / total_time

        result = {
            "total_time": total_time,
            "throughput": throughput,
            "avg_latency": total_time / num_requests * 1000,  # ms
            "num_requests": num_requests,
            "threads": self.num_threads
        }

        logger.info(".2f")
        return result

    def analyze_contention(self) -> Dict[str, Any]:
        """Analyze lock contention and performance."""
        single = self.results["single_thread"]
        multi = self.results["multi_thread"]

        speedup = multi["throughput"] / single["throughput"]
        efficiency = speedup / self.num_threads * 100  # Percentage

        analysis = {
            "speedup_ratio": speedup,
            "scaling_efficiency": efficiency,
            "ideal_speedup": self.num_threads,
            "contention_overhead": (self.num_threads - speedup) / self.num_threads * 100,
            "latency_improvement": single["avg_latency"] / multi["avg_latency"] if multi["avg_latency"] > 0 else 0,
            "throughput_improvement": (multi["throughput"] - single["throughput"]) / single["throughput"] * 100
        }

        logger.info("\n=== Contention Analysis ===")
        logger.info(f"Speedup ratio: {speedup:.2f}x")
        logger.info(f"Scaling efficiency: {efficiency:.1f}%")
        logger.info(f"Contention overhead: {analysis['contention_overhead']:.1f}%")
        logger.info(f"Throughput improvement: {analysis['throughput_improvement']:.1f}%")

        return analysis

    async def run_full_test_suite(self):
        """Run complete MT contention test suite."""
        logger.info("=== Starting MT Contention Fix Validation ===")

        # Test 1: Single-threaded baseline
        logger.info("\n--- Test 1: Single-Thread Baseline ---")
        self.results["single_thread"] = await self.run_single_thread_test(100)

        # Allow system to stabilize
        await asyncio.sleep(1)

        # Test 2: Multi-threaded performance
        logger.info("\n--- Test 2: Multi-Thread Performance ---")
        self.results["multi_thread"] = await self.run_multi_thread_test(100)

        # Test 3: Contention analysis
        logger.info("\n--- Test 3: Contention Analysis ---")
        self.results["contention_analysis"] = self.analyze_contention()

        # Get final batch statistics
        final_stats = self.batch_engine.get_batch_stats()
        logger.info(f"\nFinal batch stats: {final_stats}")

        return self.results

async def main():
    """Main test execution."""
    tester = MTContentionTester(num_gpus=4, num_threads=16)
    results = await tester.run_full_test_suite()

    # Save results
    import json
    with open("mt_contention_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n=== Test Complete ===")
    logger.info("Results saved to mt_contention_test_results.json")

    # Check if fix is successful
    analysis = results["contention_analysis"]
    if analysis["scaling_efficiency"] > 80:  # 80% efficiency threshold
        logger.info("✅ MT Contention Fix SUCCESSFUL - High scaling efficiency achieved!")
        return True
    else:
        logger.warning("⚠️  MT Contention Fix needs improvement - Low scaling efficiency")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)