#!/usr/bin/env python3
"""
Simple MT Contention Fix Validation
===================================

Validates that the fine-grained locking in BatchEngine works correctly.
"""

import time
import threading
import concurrent.futures
from typing import List

# Import the fixed batch engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.serving.batch_engine import BatchEngine, BatchPriority

def test_lock_contention():
    """Test that fine-grained locks reduce contention."""
    print("=== Testing Fine-Grained Lock Implementation ===")

    # Initialize batch engine with multiple GPUs
    batch_engine = BatchEngine(max_batch_size=4, max_latency_ms=1000.0)
    batch_engine.initialize_gpu_queues(list(range(4)))  # 4 GPUs

    # Test 1: Sequential access (should be fast)
    print("\n--- Test 1: Sequential Access ---")
    start_time = time.time()

    for i in range(100):
        gpu_id = i % 4
        # This should use per-GPU locks, not global lock
        future = batch_engine.submit_request(
            request_id=f"seq_{i}",
            gpu_id=gpu_id,
            input_tokens=[1, 2, 3] * 5,
            max_new_tokens=10,
            priority=BatchPriority.NORMAL
        )

    seq_time = time.time() - start_time
    print(".3f")

    # Test 2: Concurrent access (should scale better with fine-grained locks)
    print("\n--- Test 2: Concurrent Access ---")
    start_time = time.time()

    def worker_thread(thread_id: int, num_requests: int = 25):
        """Worker thread submitting requests."""
        for i in range(num_requests):
            req_id = thread_id * num_requests + i
            gpu_id = req_id % 4
            future = batch_engine.submit_request(
                request_id=f"mt_{req_id}",
                gpu_id=gpu_id,
                input_tokens=[1, 2, 3] * 5,
                max_new_tokens=10,
                priority=BatchPriority.NORMAL
            )

    # Run with multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for thread_id in range(8):
            future = executor.submit(worker_thread, thread_id, 25)
            futures.append(future)

        # Wait for all threads
        for future in concurrent.futures.as_completed(futures):
            future.result()

    concurrent_time = time.time() - start_time
    print(".3f")

    # Test 3: Force batch processing
    print("\n--- Test 3: Batch Processing ---")
    import asyncio

    async def process_batches():
        await batch_engine.flush_all_batches()

    asyncio.run(process_batches())

    # Test 4: Statistics
    print("\n--- Test 4: Statistics ---")
    stats = batch_engine.get_batch_stats()
    print(f"Total requests processed: {stats.get('total_requests', 0)}")
    print(f"Batches processed: {stats.get('batches_processed', 0)}")
    print(f"Average batch size: {stats.get('avg_batch_size', 0):.1f}")
    print(f"Active batches: {stats.get('active_batches', 0)}")
    print(f"Queued requests: {stats.get('queued_requests', 0)}")

    # Test 5: Per-GPU status
    print("\n--- Test 5: Per-GPU Status ---")
    for gpu_id in range(4):
        status = batch_engine.get_gpu_queue_status(gpu_id)
        print(f"GPU {gpu_id}: {status.get('queued_requests', 0)} queued, "
              f"{status.get('active_batch_size', 0)} active")

    # Analysis
    print("\n=== Analysis ===")
    speedup = seq_time / concurrent_time if concurrent_time > 0 else 0
    print(".2f")

    if concurrent_time < seq_time * 2:  # Should be much faster with fine-grained locks
        print("✅ Fine-grained locking working - concurrent access is efficient!")
        return True
    else:
        print("⚠️  Potential lock contention - concurrent access is slow")
        return False

if __name__ == "__main__":
    success = test_lock_contention()
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    exit(0 if success else 1)