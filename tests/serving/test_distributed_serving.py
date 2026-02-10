"""
Task 1.1.11: Distributed Serving Infrastructure Tests

Comprehensive tests for:
- Request queue management (priority, timeout, capacity)
- Dynamic batching (formation, padding, token optimization)
- Load balancing (distribution, health-aware routing)
- Health monitoring (error tracking, recovery)
- Metrics collection (latency, throughput, utilization)
- End-to-end serving workflows
"""

import unittest
import asyncio
import time
from unittest.mock import Mock, MagicMock, patch

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.serving.distributed_serving import (
    InferenceRequest,
    InferenceBatch,
    InferenceResponse,
    RequestPriority,
    RequestState,
    RequestQueue,
    DynamicBatcher,
    LoadBalancer,
    HealthMonitor,
    MetricsCollector,
    DistributedServingEngine,
)


# ============================================================================
# Test Utilities
# ============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, vocab_size=32000, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)


def create_test_request(request_id="test_req", 
                       num_tokens=100,
                       priority=RequestPriority.NORMAL) -> InferenceRequest:
    """Create a test inference request."""
    tokens = torch.randint(0, 32000, (num_tokens,))
    return InferenceRequest(
        request_id=request_id,
        prompt_tokens=tokens,
        max_tokens=50,
        priority=priority,
        timeout_ms=10000.0
    )


# ============================================================================
# Request Queue Tests
# ============================================================================

class TestRequestQueue(unittest.TestCase):
    """Test request queue functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.queue = RequestQueue(max_queue_size=100)
    
    def tearDown(self):
        """Clean up."""
        self.loop.close()
    
    def test_enqueue_single_request(self):
        """Test enqueueing a single request."""
        req = create_test_request("req_1")
        
        result = self.loop.run_until_complete(self.queue.enqueue(req))
        self.assertTrue(result)
        
        size = self.loop.run_until_complete(self.queue.get_size())
        self.assertEqual(size, 1)
    
    def test_dequeue_fifo_order(self):
        """Test FIFO dequeue ordering."""
        reqs = [create_test_request(f"req_{i}") for i in range(3)]
        
        for req in reqs:
            self.loop.run_until_complete(self.queue.enqueue(req))
        
        dequeued = self.loop.run_until_complete(self.queue.dequeue(count=3))
        
        self.assertEqual(len(dequeued), 3)
        # Should maintain insertion order for same priority
        self.assertEqual(dequeued[0].request_id, "req_0")
    
    def test_dequeue_priority_order(self):
        """Test priority dequeue ordering."""
        reqs = [
            create_test_request("low", priority=RequestPriority.LOW),
            create_test_request("critical", priority=RequestPriority.CRITICAL),
            create_test_request("normal", priority=RequestPriority.NORMAL),
        ]
        
        for req in reqs:
            self.loop.run_until_complete(self.queue.enqueue(req))
        
        dequeued = self.loop.run_until_complete(self.queue.dequeue(count=3))
        
        # Critical should be first
        self.assertEqual(dequeued[0].priority, RequestPriority.CRITICAL)
        # Then normal
        self.assertEqual(dequeued[1].priority, RequestPriority.NORMAL)
        # Then low
        self.assertEqual(dequeued[2].priority, RequestPriority.LOW)
    
    def test_queue_full(self):
        """Test queue overflow handling."""
        queue = RequestQueue(max_queue_size=2)
        
        for i in range(2):
            req = create_test_request(f"req_{i}")
            result = self.loop.run_until_complete(queue.enqueue(req))
            self.assertTrue(result)
        
        # Next should fail
        req = create_test_request("req_3")
        result = self.loop.run_until_complete(queue.enqueue(req))
        self.assertFalse(result)
    
    def test_cancel_request(self):
        """Test request cancellation."""
        req = create_test_request("req_1")
        self.loop.run_until_complete(self.queue.enqueue(req))
        
        size_before = self.loop.run_until_complete(self.queue.get_size())
        self.assertEqual(size_before, 1)
        
        cancelled = self.loop.run_until_complete(self.queue.cancel("req_1"))
        self.assertTrue(cancelled)
        
        size_after = self.loop.run_until_complete(self.queue.get_size())
        self.assertEqual(size_after, 0)
    
    def test_queue_statistics(self):
        """Test queue statistics tracking."""
        for i in range(5):
            req = create_test_request(f"req_{i}")
            self.loop.run_until_complete(self.queue.enqueue(req))
        
        stats = self.loop.run_until_complete(self.queue.get_stats())
        
        self.assertEqual(stats["total_enqueued"], 5)
        self.assertEqual(stats["queue_size"], 5)


# ============================================================================
# Dynamic Batcher Tests
# ============================================================================

class TestDynamicBatcher(unittest.TestCase):
    """Test dynamic batching functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.batcher = DynamicBatcher(
            max_batch_size=4,
            max_batch_tokens=1000,
            max_wait_ms=100
        )
    
    def tearDown(self):
        """Clean up."""
        self.loop.close()
    
    def test_form_single_batch(self):
        """Test forming a single batch."""
        reqs = [create_test_request(f"req_{i}", num_tokens=100) for i in range(3)]
        
        self.loop.run_until_complete(self.batcher.add_requests(reqs))
        batches = self.loop.run_until_complete(self.batcher.form_batches())
        
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0].request_ids), 3)
    
    def test_form_multiple_batches(self):
        """Test forming multiple batches."""
        reqs = [create_test_request(f"req_{i}", num_tokens=100) for i in range(10)]
        
        self.loop.run_until_complete(self.batcher.add_requests(reqs))
        batches = self.loop.run_until_complete(self.batcher.form_batches())
        
        # With max_batch_size=4, should form at least 2 batches
        self.assertGreaterEqual(len(batches), 2)
        
        # Total requests should match
        total_requests = sum(len(b.request_ids) for b in batches)
        self.assertEqual(total_requests, 10)
    
    def test_batch_token_limit(self):
        """Test batch respects token limit."""
        batcher = DynamicBatcher(
            max_batch_size=10,
            max_batch_tokens=500,
        )
        
        reqs = [create_test_request(f"req_{i}", num_tokens=200) for i in range(4)]
        
        self.loop.run_until_complete(batcher.add_requests(reqs))
        batches = self.loop.run_until_complete(batcher.form_batches())
        
        # With 200 tokens per request and 500 token limit, should form 2 batches
        self.assertEqual(len(batches), 2)
    
    def test_batcher_statistics(self):
        """Test batcher statistics."""
        reqs = [create_test_request(f"req_{i}", num_tokens=100) for i in range(3)]
        
        self.loop.run_until_complete(self.batcher.add_requests(reqs))
        self.loop.run_until_complete(self.batcher.form_batches())
        
        stats = self.loop.run_until_complete(self.batcher.get_stats())
        
        self.assertEqual(stats["total_batches"], 1)
        self.assertGreater(stats["avg_batch_size"], 0)


# ============================================================================
# Load Balancer Tests
# ============================================================================

class TestLoadBalancer(unittest.TestCase):
    """Test load balancer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.balancer = LoadBalancer(num_gpus=4)
    
    def tearDown(self):
        """Clean up."""
        self.loop.close()
    
    def test_select_gpu_round_robin(self):
        """Test GPU selection rounds"""
        selected_gpus = []
        
        for i in range(12):
            gpu_id = self.loop.run_until_complete(self.balancer.select_gpu())
            selected_gpus.append(gpu_id)
        
        # Should distribute across GPUs
        self.assertGreater(len(set(selected_gpus)), 1)
    
    def test_select_gpu_respects_load(self):
        """Test GPU selection respects load."""
        # Set GPU 0 to high load
        self.loop.run_until_complete(self.balancer.update_load(0, 1.0))
        self.loop.run_until_complete(self.balancer.update_load(1, 0.1))
        
        gpu_id = self.loop.run_until_complete(self.balancer.select_gpu())
        
        # Should select GPU 1 (lower load)
        self.assertEqual(gpu_id, 1)
    
    def test_gpu_health_tracking(self):
        """Test GPU health status."""
        # Mark GPU 0 as unhealthy
        self.loop.run_until_complete(self.balancer.set_health(0, False))
        
        # Multiple selections should avoid unhealthy GPU
        for i in range(5):
            gpu_id = self.loop.run_until_complete(self.balancer.select_gpu())
            self.assertNotEqual(gpu_id, 0)
    
    def test_balancer_statistics(self):
        """Test balancer statistics."""
        for i in range(5):
            self.loop.run_until_complete(self.balancer.select_gpu())
        
        stats = self.loop.run_until_complete(self.balancer.get_stats())
        
        self.assertEqual(stats["total_routed"], 5)
        self.assertGreater(len(stats["batch_distribution"]), 0)


# ============================================================================
# Health Monitor Tests
# ============================================================================

class TestHealthMonitor(unittest.TestCase):
    """Test health monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.monitor = HealthMonitor(num_gpus=2)
    
    def tearDown(self):
        """Clean up."""
        self.loop.close()
    
    def test_healthy_gpu_initially(self):
        """Test GPU is healthy initially."""
        is_healthy = self.loop.run_until_complete(self.monitor.check_gpu_health(0))
        self.assertTrue(is_healthy)
    
    def test_record_errors(self):
        """Test error recording."""
        for i in range(5):
            self.loop.run_until_complete(self.monitor.record_error(0))
        
        stats = self.loop.run_until_complete(self.monitor.get_stats())
        self.assertEqual(stats["error_counts"][0], 5)
    
    def test_gpu_unhealthy_after_many_errors(self):
        """Test GPU becomes unhealthy after many errors."""
        for i in range(15):
            self.loop.run_until_complete(self.monitor.record_error(0))
        
        is_healthy = self.loop.run_until_complete(self.monitor.check_gpu_health(0))
        self.assertFalse(is_healthy)
    
    def test_reset_errors(self):
        """Test error reset."""
        for i in range(5):
            self.loop.run_until_complete(self.monitor.record_error(0))
        
        self.loop.run_until_complete(self.monitor.reset_errors(0))
        
        stats = self.loop.run_until_complete(self.monitor.get_stats())
        self.assertEqual(stats["error_counts"][0], 0)


# ============================================================================
# Metrics Collector Tests
# ============================================================================

class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.metrics = MetricsCollector()
    
    def tearDown(self):
        """Clean up."""
        self.loop.close()
    
    def test_record_request_latency(self):
        """Test request latency recording."""
        response = InferenceResponse(
            request_id="req_1",
            generated_tokens=torch.tensor([1, 2, 3]),
            generated_count=3,
            total_time_ms=50.0
        )
        
        self.loop.run_until_complete(self.metrics.record_request(response))
        
        stats = self.loop.run_until_complete(self.metrics.get_stats())
        self.assertEqual(stats["total_requests"], 1)
        self.assertEqual(stats["avg_latency_ms"], 50.0)
    
    def test_multiple_requests_statistics(self):
        """Test statistics with multiple requests."""
        for i in range(10):
            response = InferenceResponse(
                request_id=f"req_{i}",
                generated_tokens=torch.tensor([1, 2, 3]),
                generated_count=3,
                total_time_ms=50.0 + i*5  # Varying latencies
            )
            
            self.loop.run_until_complete(self.metrics.record_request(response))
        
        stats = self.loop.run_until_complete(self.metrics.get_stats())
        
        self.assertEqual(stats["total_requests"], 10)
        self.assertGreater(stats["avg_latency_ms"], 0)
        self.assertGreater(stats["p50_latency_ms"], 0)


# ============================================================================
# Integration Tests
# ============================================================================

class TestServingEngineIntegration(unittest.TestCase):
    """Test end-to-end serving engine integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.model = SimpleModel()
        self.engine = DistributedServingEngine(self.model, num_gpus=1)
    
    def tearDown(self):
        """Clean up."""
        self.loop.run_until_complete(self.engine.shutdown())
        self.loop.close()
    
    def test_submit_request(self):
        """Test submitting a request."""
        req = create_test_request("req_1")
        
        request_id = self.loop.run_until_complete(self.engine.submit_request(req))
        
        self.assertEqual(request_id, "req_1")
    
    def test_submit_multiple_requests(self):
        """Test submitting multiple requests."""
        for i in range(5):
            req = create_test_request(f"req_{i}")
            self.loop.run_until_complete(self.engine.submit_request(req))
        
        size = self.loop.run_until_complete(self.engine.request_queue.get_size())
        self.assertEqual(size, 5)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
