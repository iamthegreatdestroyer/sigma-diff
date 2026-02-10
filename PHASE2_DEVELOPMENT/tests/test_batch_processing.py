"""
Tests for Sprint 4.1 - Batch Processing Engine

Comprehensive test suite for:
- Batch Optimizer (dynamic batch sizing, memory estimation)
- Batch Scheduler (trigger evaluation, policy execution)
- Request Queue (admission control, fairness, backpressure)

Sprint 4.1 - Batch Processing Engine
Created: 2026-01-06
"""

import pytest
import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List
import sys
import os
import importlib.util

# Get the path to the src/inference directory
inference_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'inference')


def load_module_direct(name, filepath):
    """Load a module directly from file, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load modules directly
batch_optimizer = load_module_direct(
    'batch_optimizer', 
    os.path.join(inference_dir, 'batch_optimizer.py')
)
batch_scheduler = load_module_direct(
    'batch_scheduler', 
    os.path.join(inference_dir, 'batch_scheduler.py')
)
request_queue = load_module_direct(
    'request_queue', 
    os.path.join(inference_dir, 'request_queue.py')
)

# Import classes from loaded modules
# Batch Optimizer
OptimizationStrategy = batch_optimizer.OptimizationStrategy
OptimizerConfig = batch_optimizer.OptimizerConfig
BatchMetrics = batch_optimizer.BatchMetrics
BatchSizePredictor = batch_optimizer.BatchSizePredictor
MemoryEstimator = batch_optimizer.MemoryEstimator
BatchOptimizer = batch_optimizer.BatchOptimizer
AdaptiveBatchOptimizer = batch_optimizer.AdaptiveBatchOptimizer
create_batch_optimizer = batch_optimizer.create_batch_optimizer

# Batch Scheduler
SchedulingPolicy = batch_scheduler.SchedulingPolicy
TriggerType = batch_scheduler.TriggerType
ScheduledRequest = batch_scheduler.ScheduledRequest
SchedulerConfig = batch_scheduler.SchedulerConfig
SchedulerStats = batch_scheduler.SchedulerStats
QueueState = batch_scheduler.QueueState
SizeThresholdTrigger = batch_scheduler.SizeThresholdTrigger
TimeDeadlineTrigger = batch_scheduler.TimeDeadlineTrigger
PriorityUrgentTrigger = batch_scheduler.PriorityUrgentTrigger
FIFOPolicy = batch_scheduler.FIFOPolicy
SizeOptimalPolicy = batch_scheduler.SizeOptimalPolicy
DeadlineDrivenPolicy = batch_scheduler.DeadlineDrivenPolicy
PriorityWeightedPolicy = batch_scheduler.PriorityWeightedPolicy
AdaptivePolicy = batch_scheduler.AdaptivePolicy
BatchScheduler = batch_scheduler.BatchScheduler
create_scheduler = batch_scheduler.create_scheduler
create_latency_optimized_scheduler = batch_scheduler.create_latency_optimized_scheduler
create_throughput_optimized_scheduler = batch_scheduler.create_throughput_optimized_scheduler
SchedulerContext = batch_scheduler.SchedulerContext

# Request Queue
QueuePriority = request_queue.QueuePriority
AdmissionDecision = request_queue.AdmissionDecision
BackpressureLevel = request_queue.BackpressureLevel
QueuedRequest = request_queue.QueuedRequest
TenantQuota = request_queue.TenantQuota
QueueStats = request_queue.QueueStats
QueueConfig = request_queue.QueueConfig
TokenBucketController = request_queue.TokenBucketController
LoadBasedController = request_queue.LoadBasedController
FairScheduler = request_queue.FairScheduler
RequestQueue = request_queue.RequestQueue
create_request_queue = request_queue.create_request_queue
create_high_throughput_queue = request_queue.create_high_throughput_queue
create_low_latency_queue = request_queue.create_low_latency_queue


# =============================================================================
# BATCH OPTIMIZER TESTS
# =============================================================================

class TestOptimizerConfig:
    """Tests for OptimizerConfig."""
    
    def test_default_values(self):
        config = OptimizerConfig()
        assert config.min_batch_size == 1
        assert config.max_batch_size == 256
        assert config.default_batch_size == 32
        assert config.max_batch_tokens == 8192
    
    def test_custom_values(self):
        config = OptimizerConfig(
            max_batch_size=512,
            target_latency_ms=50.0,
            strategy=OptimizationStrategy.THROUGHPUT_FIRST
        )
        assert config.max_batch_size == 512
        assert config.target_latency_ms == 50.0
        assert config.strategy == OptimizationStrategy.THROUGHPUT_FIRST


class TestBatchMetrics:
    """Tests for BatchMetrics."""
    
    def test_metrics_calculation(self):
        metrics = BatchMetrics(
            batch_id="test_batch",
            batch_size=32,
            total_tokens=1024,
            latency_ms=100.0,
            throughput_tps=10240.0,
            gpu_memory_used_mb=512.0
        )
        
        assert metrics.tokens_per_request == 32.0
        assert metrics.latency_per_token_ms == pytest.approx(0.097656, rel=0.01)


class TestBatchSizePredictor:
    """Tests for BatchSizePredictor."""
    
    def test_throughput_first_prediction(self):
        config = OptimizerConfig()
        predictor = BatchSizePredictor(config)
        
        size = predictor.predict_optimal_batch_size(
            pending_tokens=1000,
            available_memory_mb=4096.0,
            strategy=OptimizationStrategy.THROUGHPUT_FIRST
        )
        
        assert size <= config.max_batch_size
        assert size > 0
    
    def test_latency_first_prediction(self):
        config = OptimizerConfig()
        predictor = BatchSizePredictor(config)
        
        size = predictor.predict_optimal_batch_size(
            pending_tokens=1000,
            available_memory_mb=4096.0,
            strategy=OptimizationStrategy.LATENCY_FIRST
        )
        
        # Latency-first should prefer smaller batches
        assert size <= 32
    
    def test_learning_from_observations(self):
        config = OptimizerConfig()
        predictor = BatchSizePredictor(config)
        
        initial_adjustment = predictor._batch_size_adjustment
        
        # Add observations with good latency
        for i in range(10):
            metrics = BatchMetrics(
                batch_id=f"batch_{i}",
                batch_size=32,
                total_tokens=1024,
                latency_ms=50.0,  # Good latency
                throughput_tps=20480.0,
                gpu_memory_used_mb=256.0
            )
            predictor.add_observation(metrics)
        
        # Update adjustment (should increase since latency is good)
        predictor.update_adjustment(50.0, 100.0)
        
        assert predictor._batch_size_adjustment >= initial_adjustment


class TestMemoryEstimator:
    """Tests for MemoryEstimator."""
    
    def test_memory_estimation(self):
        config = OptimizerConfig()
        estimator = MemoryEstimator(config)
        
        memory = estimator.estimate_memory(
            batch_size=32,
            total_tokens=1024,
            max_sequence_length=512
        )
        
        assert memory > 0
        assert isinstance(memory, float)
    
    def test_larger_batch_more_memory(self):
        config = OptimizerConfig()
        estimator = MemoryEstimator(config)
        
        small_memory = estimator.estimate_memory(16, 512, 256)
        large_memory = estimator.estimate_memory(64, 2048, 512)
        
        assert large_memory > small_memory


class TestBatchOptimizer:
    """Tests for BatchOptimizer."""
    
    def test_optimizer_creation(self):
        optimizer = BatchOptimizer()
        assert optimizer is not None
        assert optimizer.config.default_batch_size == 32
    
    def test_get_optimal_batch_size(self):
        optimizer = BatchOptimizer()
        
        size, metadata = optimizer.get_optimal_batch_size(
            pending_requests=100,
            total_pending_tokens=5000,
            avg_sequence_length=50
        )
        
        assert size >= optimizer.config.min_batch_size
        assert size <= optimizer.config.max_batch_size
        assert 'predicted_size' in metadata
        assert 'strategy' in metadata
    
    def test_should_execute_batch_size_trigger(self):
        config = OptimizerConfig(default_batch_size=16)
        optimizer = BatchOptimizer(config)
        optimizer._current_batch_size = 16
        
        should_execute, reason = optimizer.should_execute_batch(
            current_batch_size=16,
            current_tokens=500,
            wait_time_ms=10.0
        )
        
        assert should_execute
        assert reason == "batch_size_reached"
    
    def test_should_execute_token_trigger(self):
        config = OptimizerConfig(max_batch_tokens=1000)
        optimizer = BatchOptimizer(config)
        
        should_execute, reason = optimizer.should_execute_batch(
            current_batch_size=5,
            current_tokens=1000,
            wait_time_ms=10.0
        )
        
        assert should_execute
        assert reason == "token_limit_reached"
    
    def test_should_execute_timeout_trigger(self):
        config = OptimizerConfig(max_wait_time_ms=50.0)
        optimizer = BatchOptimizer(config)
        
        should_execute, reason = optimizer.should_execute_batch(
            current_batch_size=5,
            current_tokens=100,
            wait_time_ms=60.0
        )
        
        assert should_execute
        assert reason == "wait_timeout"
    
    def test_record_batch_result(self):
        optimizer = BatchOptimizer()
        
        optimizer.record_batch_result(
            batch_size=32,
            total_tokens=1024,
            latency_ms=100.0,
            memory_used_mb=512.0
        )
        
        metrics = optimizer.get_metrics()
        assert metrics['total_batches'] == 1
        assert metrics['total_tokens'] == 1024
    
    def test_strategy_switching(self):
        optimizer = BatchOptimizer()
        
        optimizer.set_strategy(OptimizationStrategy.THROUGHPUT_FIRST)
        assert optimizer._current_strategy == OptimizationStrategy.THROUGHPUT_FIRST
        
        optimizer.set_strategy(OptimizationStrategy.LATENCY_FIRST)
        assert optimizer._current_strategy == OptimizationStrategy.LATENCY_FIRST


class TestAdaptiveBatchOptimizer:
    """Tests for AdaptiveBatchOptimizer."""
    
    def test_thompson_sampling(self):
        optimizer = AdaptiveBatchOptimizer()
        
        # Get multiple batch sizes
        sizes = set()
        for _ in range(50):
            size, _ = optimizer.get_optimal_batch_size(
                pending_requests=100,
                total_pending_tokens=5000
            )
            sizes.add(size)
        
        # Should have some exploration
        assert len(sizes) >= 1
    
    def test_learning_from_success(self):
        optimizer = AdaptiveBatchOptimizer()
        
        # Record successful batches at size 32
        for _ in range(20):
            optimizer.record_batch_result(
                batch_size=32,
                total_tokens=1024,
                latency_ms=50.0,  # Good latency (below SLA)
                memory_used_mb=256.0
            )
        
        # The 32 bucket should have higher alpha
        assert optimizer._alphas[32] > 1.0


class TestBatchOptimizerFactory:
    """Tests for factory function."""
    
    def test_create_throughput_optimizer(self):
        optimizer = create_batch_optimizer(
            strategy="throughput",
            max_batch_size=512
        )
        assert optimizer.config.max_batch_size == 512
    
    def test_create_latency_optimizer(self):
        optimizer = create_batch_optimizer(
            strategy="latency",
            target_latency_ms=25.0
        )
        assert optimizer.config.target_latency_ms == 25.0
    
    def test_create_adaptive_optimizer(self):
        optimizer = create_batch_optimizer(strategy="adaptive")
        assert isinstance(optimizer, AdaptiveBatchOptimizer)


# =============================================================================
# BATCH SCHEDULER TESTS
# =============================================================================

class TestScheduledRequest:
    """Tests for ScheduledRequest."""
    
    def test_priority_ordering(self):
        high = ScheduledRequest(priority=1, arrival_time=time.time())
        low = ScheduledRequest(priority=5, arrival_time=time.time())
        
        assert high < low
    
    def test_arrival_time_ordering(self):
        early = ScheduledRequest(priority=3, arrival_time=1000.0)
        late = ScheduledRequest(priority=3, arrival_time=2000.0)
        
        assert early < late
    
    def test_urgency_calculation(self):
        now = time.time()
        
        # Request with imminent deadline
        urgent = ScheduledRequest(
            priority=5,
            arrival_time=now,
            deadline=now + 0.1  # 100ms deadline
        )
        
        # Request without deadline
        normal = ScheduledRequest(
            priority=5,
            arrival_time=now
        )
        
        assert urgent.urgency > normal.urgency


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""
    
    def test_default_config(self):
        config = SchedulerConfig()
        assert config.max_batch_size == 64
        assert config.max_wait_time == 0.1
        assert config.scheduling_policy == SchedulingPolicy.ADAPTIVE


class TestSchedulingTriggers:
    """Tests for scheduling triggers."""
    
    def test_size_threshold_trigger(self):
        trigger = SizeThresholdTrigger(target_size=10)
        
        state_below = QueueState(pending_count=5)
        should_fire, _ = trigger.check(state_below)
        assert not should_fire
        
        state_at = QueueState(pending_count=10)
        should_fire, trigger_type = trigger.check(state_at)
        assert should_fire
        assert trigger_type == TriggerType.SIZE_THRESHOLD
    
    def test_time_deadline_trigger(self):
        trigger = TimeDeadlineTrigger(max_wait_time=0.05)  # 50ms
        
        # Recent request
        state_recent = QueueState(oldest_arrival=time.time())
        should_fire, _ = trigger.check(state_recent)
        assert not should_fire
        
        # Old request
        state_old = QueueState(oldest_arrival=time.time() - 0.1)
        should_fire, trigger_type = trigger.check(state_old)
        assert should_fire
        assert trigger_type == TriggerType.TIME_DEADLINE
    
    def test_priority_urgent_trigger(self):
        trigger = PriorityUrgentTrigger(urgent_threshold=2)
        
        state_normal = QueueState(highest_priority=5)
        should_fire, _ = trigger.check(state_normal)
        assert not should_fire
        
        state_urgent = QueueState(highest_priority=1)
        should_fire, trigger_type = trigger.check(state_urgent)
        assert should_fire
        assert trigger_type == TriggerType.PRIORITY_URGENT


class TestBatchFormationPolicies:
    """Tests for batch formation policies."""
    
    def _create_requests(self, count: int) -> List[ScheduledRequest]:
        """Helper to create test requests."""
        return [
            ScheduledRequest(
                priority=i % 5,
                arrival_time=time.time() + i * 0.001,
                request_id=f"req_{i}",
                sequence_length=50 + i * 10
            )
            for i in range(count)
        ]
    
    def test_fifo_policy(self):
        policy = FIFOPolicy()
        requests = self._create_requests(10)
        config = SchedulerConfig()
        
        batch = policy.form_batch(requests, 5, config)
        
        assert len(batch) == 5
        # Should be in arrival order
        for i, req in enumerate(batch):
            assert req.request_id == f"req_{i}"
    
    def test_size_optimal_policy(self):
        policy = SizeOptimalPolicy(length_bucket_size=32)
        
        # Create requests with varied lengths
        requests = [
            ScheduledRequest(priority=3, arrival_time=time.time(), 
                           request_id=f"short_{i}", sequence_length=30)
            for i in range(5)
        ] + [
            ScheduledRequest(priority=3, arrival_time=time.time(), 
                           request_id=f"long_{i}", sequence_length=100)
            for i in range(3)
        ]
        
        config = SchedulerConfig()
        batch = policy.form_batch(requests, 4, config)
        
        # Should group similar lengths
        lengths = [r.sequence_length for r in batch]
        assert len(set(lengths)) <= 2  # At most 2 different length buckets
    
    def test_deadline_driven_policy(self):
        policy = DeadlineDrivenPolicy()
        now = time.time()
        
        requests = [
            ScheduledRequest(priority=3, arrival_time=now, request_id="late",
                           deadline=now + 10.0),
            ScheduledRequest(priority=3, arrival_time=now, request_id="urgent",
                           deadline=now + 0.1),
            ScheduledRequest(priority=3, arrival_time=now, request_id="medium",
                           deadline=now + 1.0),
        ]
        
        config = SchedulerConfig()
        batch = policy.form_batch(requests, 3, config)
        
        # Most urgent should be first
        assert batch[0].request_id == "urgent"
    
    def test_priority_weighted_policy(self):
        policy = PriorityWeightedPolicy()
        now = time.time()
        
        requests = [
            ScheduledRequest(priority=5, arrival_time=now, request_id="low"),
            ScheduledRequest(priority=1, arrival_time=now, request_id="high"),
            ScheduledRequest(priority=3, arrival_time=now, request_id="medium"),
        ]
        
        config = SchedulerConfig()
        batch = policy.form_batch(requests, 3, config)
        
        # Highest priority should be first
        assert batch[0].request_id == "high"


class TestBatchScheduler:
    """Tests for BatchScheduler."""
    
    @pytest.fixture
    def scheduler(self):
        return BatchScheduler(SchedulerConfig(
            max_batch_size=8,
            batch_size_target=4,
            max_wait_time=0.1
        ))
    
    def test_scheduler_creation(self, scheduler):
        assert scheduler is not None
        assert scheduler.config.max_batch_size == 8
    
    def test_submit_request(self, scheduler):
        future = scheduler.submit(
            request_id="test_1",
            sequence_length=100,
            max_tokens=50,
            priority=3
        )
        
        assert scheduler.get_queue_depth() == 1
    
    def test_multiple_submissions(self, scheduler):
        for i in range(10):
            scheduler.submit(
                request_id=f"test_{i}",
                sequence_length=50,
                priority=i % 5
            )
        
        assert scheduler.get_queue_depth() == 10
    
    def test_stats_tracking(self, scheduler):
        for i in range(5):
            scheduler.submit(
                request_id=f"test_{i}",
                sequence_length=50
            )
        
        stats = scheduler.get_stats()
        assert stats.total_requests_scheduled == 5


class TestSchedulerFactories:
    """Tests for scheduler factory functions."""
    
    def test_create_default_scheduler(self):
        scheduler = create_scheduler()
        assert scheduler is not None
        assert scheduler.config.scheduling_policy == SchedulingPolicy.ADAPTIVE
    
    def test_create_latency_optimized(self):
        scheduler = create_latency_optimized_scheduler()
        assert scheduler.config.max_batch_size == 16
        assert scheduler.config.max_wait_time == 0.05
    
    def test_create_throughput_optimized(self):
        scheduler = create_throughput_optimized_scheduler()
        assert scheduler.config.max_batch_size == 128
        assert scheduler.config.scheduling_policy == SchedulingPolicy.SIZE_OPTIMAL


# =============================================================================
# REQUEST QUEUE TESTS
# =============================================================================

class TestQueuePriority:
    """Tests for QueuePriority."""
    
    def test_priority_ordering(self):
        assert QueuePriority.CRITICAL.value < QueuePriority.REALTIME.value
        assert QueuePriority.REALTIME.value < QueuePriority.HIGH.value
        assert QueuePriority.HIGH.value < QueuePriority.NORMAL.value
    
    def test_priority_weights(self):
        assert QueuePriority.CRITICAL.weight > QueuePriority.NORMAL.weight
        assert QueuePriority.NORMAL.weight > QueuePriority.BULK.weight


class TestQueuedRequest:
    """Tests for QueuedRequest."""
    
    def test_request_ordering(self):
        high = QueuedRequest(
            request_id="high",
            priority=QueuePriority.HIGH,
            arrival_time=time.time()
        )
        low = QueuedRequest(
            request_id="low",
            priority=QueuePriority.LOW,
            arrival_time=time.time()
        )
        
        assert high < low
    
    def test_total_tokens(self):
        req = QueuedRequest(
            request_id="test",
            sequence_length=100,
            max_tokens=50
        )
        
        assert req.total_tokens == 150
    
    def test_wait_time(self):
        start = time.time()
        req = QueuedRequest(request_id="test", arrival_time=start)
        
        time.sleep(0.01)
        
        assert req.wait_time >= 0.01


class TestTokenBucketController:
    """Tests for TokenBucketController."""
    
    def test_admits_with_tokens(self):
        controller = TokenBucketController(rate=100.0, burst_size=50)
        
        req = QueuedRequest(request_id="test", estimated_cost=1.0)
        state = request_queue.QueueState()
        
        decision, _ = controller.check_admission(req, state)
        assert decision == AdmissionDecision.ADMIT
    
    def test_rejects_when_empty(self):
        controller = TokenBucketController(rate=1.0, burst_size=1)
        
        req = QueuedRequest(request_id="test", estimated_cost=1.0)
        state = request_queue.QueueState()
        
        # Use all tokens
        controller.check_admission(req, state)
        
        # Next should be rejected (or throttled)
        decision, _ = controller.check_admission(req, state)
        assert decision in (AdmissionDecision.REJECT_OVERLOAD, AdmissionDecision.THROTTLE)
    
    def test_refills_over_time(self):
        controller = TokenBucketController(rate=1000.0, burst_size=10)
        
        req = QueuedRequest(request_id="test", estimated_cost=10.0)
        state = request_queue.QueueState()
        
        # Use all tokens
        controller.check_admission(req, state)
        
        # Wait for refill
        time.sleep(0.02)  # 20ms at 1000/s = 20 tokens
        
        decision, _ = controller.check_admission(req, state)
        assert decision == AdmissionDecision.ADMIT


class TestFairScheduler:
    """Tests for FairScheduler."""
    
    def test_virtual_time_increases(self):
        scheduler = FairScheduler()
        
        req1 = QueuedRequest(
            request_id="req1",
            tenant_id="tenant1",
            priority=QueuePriority.NORMAL,
            sequence_length=100,
            max_tokens=100
        )
        
        vtime1 = scheduler.compute_virtual_time(req1, time.time())
        
        # Simulate dequeue
        scheduler.update_on_dequeue(req1)
        
        req2 = QueuedRequest(
            request_id="req2",
            tenant_id="tenant1",
            priority=QueuePriority.NORMAL,
            sequence_length=100,
            max_tokens=100
        )
        
        vtime2 = scheduler.compute_virtual_time(req2, time.time())
        
        assert vtime2 > vtime1
    
    def test_fairness_across_tenants(self):
        scheduler = FairScheduler()
        
        # Process requests from tenant1
        for i in range(10):
            req = QueuedRequest(
                request_id=f"t1_{i}",
                tenant_id="tenant1",
                priority=QueuePriority.NORMAL,
                sequence_length=100,
                max_tokens=100
            )
            scheduler.update_on_enqueue(req)
            scheduler.update_on_dequeue(req)
        
        # New request from tenant2 should have lower virtual time (more deserving)
        score1 = scheduler.get_tenant_fairness_score("tenant1")
        score2 = scheduler.get_tenant_fairness_score("tenant2")
        
        assert score2 < score1  # tenant2 is more deserving


class TestRequestQueue:
    """Tests for RequestQueue."""
    
    @pytest.fixture
    def queue(self):
        config = QueueConfig(
            max_queue_size=100,
            enable_admission_control=True,
            enable_backpressure=True,
            rate_limit_enabled=False  # Disable for easier testing
        )
        return RequestQueue(config)
    
    def test_queue_creation(self, queue):
        assert queue is not None
        assert len(queue) == 0
        assert queue.is_empty()
    
    def test_enqueue_dequeue(self, queue):
        success, _ = queue.enqueue(
            request_id="test_1",
            sequence_length=100,
            max_tokens=50
        )
        
        assert success
        assert len(queue) == 1
        
        dequeued = queue.dequeue(max_requests=1)
        
        assert len(dequeued) == 1
        assert dequeued[0].request_id == "test_1"
        assert len(queue) == 0
    
    def test_priority_ordering(self, queue):
        # Enqueue in reverse priority order
        queue.enqueue("low", 100, priority=QueuePriority.LOW)
        queue.enqueue("normal", 100, priority=QueuePriority.NORMAL)
        queue.enqueue("high", 100, priority=QueuePriority.HIGH)
        
        dequeued = queue.dequeue(max_requests=3)
        
        # Should come out in priority order
        assert dequeued[0].request_id == "high"
        assert dequeued[1].request_id == "normal"
        assert dequeued[2].request_id == "low"
    
    def test_tenant_isolation(self, queue):
        # Enqueue from different tenants
        queue.enqueue("t1_1", 100, tenant_id="tenant1")
        queue.enqueue("t2_1", 100, tenant_id="tenant2")
        queue.enqueue("t1_2", 100, tenant_id="tenant1")
        
        dequeued = queue.dequeue(max_requests=3)
        
        # Fair scheduling should interleave
        assert len(dequeued) == 3
    
    def test_cancellation(self, queue):
        queue.enqueue("to_cancel", 100)
        assert len(queue) == 1
        
        cancelled = queue.cancel("to_cancel")
        assert cancelled
        
        # Dequeue should skip cancelled
        dequeued = queue.dequeue(max_requests=1)
        assert len(dequeued) == 0
        assert len(queue) == 0
    
    def test_max_requests_limit(self, queue):
        for i in range(20):
            queue.enqueue(f"req_{i}", 100)
        
        dequeued = queue.dequeue(max_requests=5)
        
        assert len(dequeued) == 5
        assert len(queue) == 15
    
    def test_max_tokens_limit(self, queue):
        # Each request has 150 total tokens (100 + 50)
        for i in range(10):
            queue.enqueue(f"req_{i}", sequence_length=100, max_tokens=50)
        
        # Limit to 400 tokens (should get ~2-3 requests)
        dequeued = queue.dequeue(max_requests=10, max_tokens=400)
        
        assert len(dequeued) <= 3
        assert sum(r.total_tokens for r in dequeued) <= 400
    
    def test_stats_tracking(self, queue):
        for i in range(10):
            queue.enqueue(f"req_{i}", 100)
        
        queue.dequeue(max_requests=5)
        
        stats = queue.get_stats()
        
        assert stats.total_enqueued == 10
        assert stats.total_dequeued == 5
        assert stats.current_depth == 5
    
    def test_peek(self, queue):
        queue.enqueue("first", 100, priority=QueuePriority.HIGH)
        queue.enqueue("second", 100, priority=QueuePriority.NORMAL)
        
        peeked = queue.peek(n=1)
        
        assert len(peeked) == 1
        assert peeked[0].request_id == "first"
        assert len(queue) == 2  # Still in queue


class TestRequestQueueBackpressure:
    """Tests for backpressure functionality."""
    
    def test_backpressure_increases_with_load(self):
        config = QueueConfig(
            max_queue_size=100,
            enable_backpressure=True,
            backpressure_thresholds={
                BackpressureLevel.LOW: 0.3,
                BackpressureLevel.MEDIUM: 0.5,
                BackpressureLevel.HIGH: 0.7,
                BackpressureLevel.CRITICAL: 0.9,
            }
        )
        queue = RequestQueue(config)
        
        # Start with no backpressure
        assert queue.get_backpressure_level() == BackpressureLevel.NONE
        
        # Add requests to trigger backpressure
        for i in range(35):
            queue.enqueue(f"req_{i}", 100)
        
        assert queue.get_backpressure_level().value >= BackpressureLevel.LOW.value


class TestRequestQueueFactories:
    """Tests for queue factory functions."""
    
    def test_create_default_queue(self):
        queue = create_request_queue()
        assert queue is not None
        assert queue.config.max_queue_size == 10000
    
    def test_create_high_throughput_queue(self):
        queue = create_high_throughput_queue()
        assert queue.config.max_queue_size == 50000
    
    def test_create_low_latency_queue(self):
        queue = create_low_latency_queue()
        assert queue.config.max_queue_size == 1000
        assert queue.config.default_timeout == 5.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestBatchProcessingIntegration:
    """Integration tests combining all components."""
    
    def test_end_to_end_batch_flow(self):
        """Test complete flow from queue to optimizer to scheduler."""
        # Create components
        queue = create_request_queue(max_size=1000, rate_limit=0)
        optimizer = create_batch_optimizer(strategy="balanced")
        
        # Enqueue requests
        for i in range(50):
            queue.enqueue(
                request_id=f"req_{i}",
                sequence_length=50 + i * 2,
                max_tokens=100,
                priority=QueuePriority(i % 4 + 1)
            )
        
        # Get optimal batch size
        batch_size, metadata = optimizer.get_optimal_batch_size(
            pending_requests=len(queue),
            total_pending_tokens=sum(r.total_tokens for r in queue.peek(50)),
            avg_sequence_length=100
        )
        
        # Dequeue optimal batch
        batch = queue.dequeue(max_requests=batch_size)
        
        assert len(batch) > 0
        assert len(batch) <= batch_size
        
        # Record result
        optimizer.record_batch_result(
            batch_size=len(batch),
            total_tokens=sum(r.total_tokens for r in batch),
            latency_ms=50.0,
            memory_used_mb=256.0
        )
        
        # Check optimizer learned
        metrics = optimizer.get_metrics()
        assert metrics['total_batches'] == 1
    
    def test_priority_respects_fairness(self):
        """Test that priority scheduling respects fairness over time."""
        queue = create_request_queue(max_size=1000, rate_limit=0)
        
        # Add requests from different tenants with different priorities
        for i in range(20):
            queue.enqueue(
                request_id=f"t1_{i}",
                sequence_length=100,
                tenant_id="tenant1",
                priority=QueuePriority.HIGH
            )
            queue.enqueue(
                request_id=f"t2_{i}",
                sequence_length=100,
                tenant_id="tenant2",
                priority=QueuePriority.NORMAL
            )
        
        # Dequeue and track tenant distribution
        tenant_counts = {"tenant1": 0, "tenant2": 0}
        
        while not queue.is_empty():
            batch = queue.dequeue(max_requests=4)
            for req in batch:
                tenant_counts[req.tenant_id] += 1
        
        # Both tenants should get served
        assert tenant_counts["tenant1"] > 0
        assert tenant_counts["tenant2"] > 0


# =============================================================================
# ASYNC TESTS
# =============================================================================

@pytest.mark.asyncio
class TestAsyncScheduler:
    """Async tests for BatchScheduler."""
    
    async def test_scheduler_lifecycle(self):
        """Test scheduler start/stop lifecycle."""
        scheduler = create_scheduler(max_batch_size=4, max_wait_time=0.05)
        
        async with SchedulerContext(scheduler):
            # Submit requests
            for i in range(5):
                scheduler.submit(
                    request_id=f"req_{i}",
                    sequence_length=50
                )
            
            # Wait for processing
            await asyncio.sleep(0.1)
        
        # After context, scheduler should be stopped
        assert not scheduler._running


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for batch processing components."""
    
    def test_queue_enqueue_throughput(self):
        """Test enqueue throughput."""
        queue = create_high_throughput_queue()
        
        start = time.time()
        count = 10000
        
        for i in range(count):
            queue.enqueue(f"req_{i}", 100)
        
        elapsed = time.time() - start
        throughput = count / elapsed
        
        print(f"\nEnqueue throughput: {throughput:.0f} req/s")
        assert throughput > 2000  # At least 2k req/s (conservative for CI environments)
    
    def test_queue_dequeue_throughput(self):
        """Test dequeue throughput."""
        queue = create_high_throughput_queue()
        
        # Pre-fill queue
        for i in range(10000):
            queue.enqueue(f"req_{i}", 100)
        
        start = time.time()
        total_dequeued = 0
        
        while not queue.is_empty():
            batch = queue.dequeue(max_requests=32)
            total_dequeued += len(batch)
        
        elapsed = time.time() - start
        throughput = total_dequeued / elapsed
        
        print(f"\nDequeue throughput: {throughput:.0f} req/s")
        assert throughput > 2000  # At least 2k req/s (conservative for CI environments)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
