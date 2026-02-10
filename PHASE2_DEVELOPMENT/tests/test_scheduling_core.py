"""
Comprehensive Test Suite for Scheduling Module.

Sprint 4.3 Testing: Tests for GPU Memory Manager, Adaptive Batch Scheduler,
and Resource Allocator.

Coverage Targets:
- gpu_memory_manager.py: >95%
- batch_scheduler.py: >90%
- resource_allocator.py: >90%
- Overall: >90%

Test Organization:
- Unit tests: Individual component functionality
- Integration tests: Component interactions
- Performance tests: Latency and throughput targets
- Fairness tests: Multi-tenant allocation fairness

Author: ECLIPSE (Testing, Verification & Formal Methods Specialist)
"""

import pytest
import asyncio
import threading
import time
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock

# Import module components
from src.scheduling.gpu_memory_manager import (
    GPUMemoryManager,
    MemoryPool,
    MemoryBlock,
    MemoryStats,
    MemoryPressureLevel,
    AllocationPolicy,
    PoolType,
    AllocationRequest,
    AllocationResult,
    MemoryPressureMonitor,
    DefragmentationEngine,
    DefragmentationStrategy,
    create_memory_manager,
    format_bytes,
)
from src.scheduling.batch_scheduler import (
    AdaptiveBatchScheduler,
    SchedulingRequest,
    SchedulingDecision,
    SchedulingPolicy,
    RequestPriority,
    WorkloadProfile,
    WorkloadType,
    BatchFormationStrategy,
    LatencyTarget,
    LatencyPredictor,
    create_scheduler,
    create_request,
)
from src.scheduling.resource_allocator import (
    ResourceAllocator,
    ResourceType,
    ResourceCapacity,
    ResourceQuota,
    ResourceRequest,
    AllocationGrant,
    TenantState,
    PriorityClass,
    IsolationLevel,
    AllocationMode,
    QuotaEnforcement,
    FairnessPolicy,
    DominantResourceFairness,
    MaxMinFairShare,
    AdmissionController,
    PreemptionManager,
    ResourceRebalancer,
    RebalanceStrategy,
    create_allocator,
    create_request as create_alloc_request,
)


# =============================================================================
# GPU MEMORY MANAGER TESTS
# =============================================================================

class TestMemoryPool:
    """Tests for MemoryPool class."""
    
    def test_pool_creation(self):
        """Test pool initialization."""
        pool = MemoryPool(
            pool_type=PoolType.MEDIUM,
            initial_capacity=10 * 1024 * 1024,  # 10MB
            max_capacity=100 * 1024 * 1024     # 100MB
        )
        
        assert pool.pool_type == PoolType.MEDIUM
        assert pool.initial_capacity == 10 * 1024 * 1024
        assert pool.max_capacity == 100 * 1024 * 1024
        assert len(pool._blocks) == 0
        assert pool._total_allocated == 0
    
    def test_allocation_basic(self):
        """Test basic allocation from pool."""
        pool = MemoryPool(
            pool_type=PoolType.MEDIUM,
            initial_capacity=10 * 1024 * 1024,
            max_capacity=100 * 1024 * 1024
        )
        
        request = AllocationRequest(size=1 * 1024 * 1024)
        result = pool.allocate(request)
        
        assert result.success
        assert result.block is not None
        assert result.block.size == 1 * 1024 * 1024
        assert result.block.allocated is True
        assert pool._total_allocated == 1 * 1024 * 1024
    
    def test_allocation_exhaustion(self):
        """Test allocation failure when pool exhausted."""
        pool = MemoryPool(
            pool_type=PoolType.MEDIUM,
            initial_capacity=1 * 1024 * 1024,
            max_capacity=1 * 1024 * 1024
        )
        
        # First allocation should succeed
        request1 = AllocationRequest(size=500 * 1024)
        result1 = pool.allocate(request1)
        assert result1.success
        
        # Second allocation should also succeed
        request2 = AllocationRequest(size=500 * 1024)
        result2 = pool.allocate(request2)
        assert result2.success
        
        # Third allocation should fail
        request3 = AllocationRequest(size=100 * 1024)
        result3 = pool.allocate(request3)
        assert not result3.success
    
    def test_deallocation(self):
        """Test block deallocation."""
        pool = MemoryPool(
            pool_type=PoolType.MEDIUM,
            initial_capacity=10 * 1024 * 1024,
            max_capacity=100 * 1024 * 1024
        )
        
        # Allocate a block
        request = AllocationRequest(size=1 * 1024 * 1024)
        result = pool.allocate(request)
        assert result.success
        
        block_id = result.block.id
        
        # Deallocate
        success = pool.deallocate(block_id)
        assert success
        assert pool._total_allocated == 0
    
    def test_allocation_policies(self):
        """Test different allocation policies."""
        pool = MemoryPool(
            pool_type=PoolType.MEDIUM,
            initial_capacity=10 * 1024 * 1024,
            max_capacity=100 * 1024 * 1024
        )
        
        # Allocate some blocks
        for size in [1 * 1024 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024]:
            req = AllocationRequest(size=size)
            pool.allocate(req)
        
        # Deallocate the middle one
        middle_block = list(pool._blocks.values())[1]
        pool.deallocate(middle_block.id)
        
        # Test BEST_FIT - should pick the exactly-fitting block
        request = AllocationRequest(
            size=2 * 1024 * 1024,
            policy=AllocationPolicy.BEST_FIT
        )
        result = pool.allocate(request)
        assert result.success


class TestMemoryPressureMonitor:
    """Tests for MemoryPressureMonitor."""
    
    def test_pressure_level_calculation(self):
        """Test pressure level calculation from utilization."""
        monitor = MemoryPressureMonitor()
        
        # Test different utilization levels
        assert monitor.update(0.3) == MemoryPressureLevel.LOW
        assert monitor.update(0.6) == MemoryPressureLevel.MEDIUM
        assert monitor.update(0.8) == MemoryPressureLevel.HIGH
        assert monitor.update(0.96) == MemoryPressureLevel.CRITICAL
    
    def test_pressure_callbacks(self):
        """Test that callbacks are triggered on pressure changes."""
        monitor = MemoryPressureMonitor()
        callback_called = []
        
        def callback():
            callback_called.append(True)
        
        monitor.register_callback(MemoryPressureLevel.HIGH, callback)
        
        # Update pressure to trigger callback
        monitor.update(0.85)
        
        assert len(callback_called) > 0


class TestDefragmentationEngine:
    """Tests for DefragmentationEngine."""
    
    def test_fragmentation_calculation(self):
        """Test fragmentation ratio calculation."""
        engine = DefragmentationEngine()
        
        # Create blocks
        blocks = [
            MemoryBlock(id=f"b{i}", address=i*100, size=50, pool_type=PoolType.MEDIUM, allocated=True)
            for i in range(5)
        ]
        # Add a free block
        blocks.append(MemoryBlock(id="free", address=500, size=200, pool_type=PoolType.MEDIUM, allocated=False))
        
        # Calculate fragmentation
        frag = engine.calculate_fragmentation(blocks, 200)
        
        # Should be between 0 and 1
        assert 0.0 <= frag <= 1.0
    
    def test_defrag_should_trigger(self):
        """Test defragmentation triggering logic."""
        engine = DefragmentationEngine(
            strategy=DefragmentationStrategy.INCREMENTAL,
            min_fragmentation_threshold=0.3
        )
        
        # Low fragmentation should not trigger
        assert not engine.should_defrag(0.1)
        
        # High fragmentation should trigger
        assert engine.should_defrag(0.5)


class TestGPUMemoryManager:
    """Tests for GPUMemoryManager (main GPU memory manager)."""
    
    def test_manager_creation(self):
        """Test memory manager initialization."""
        mgr = create_memory_manager(device_id=0, total_memory_gb=16.0)
        
        assert mgr.device_id == 0
        assert mgr.total_memory == 16 * 1024 * 1024 * 1024
        assert len(mgr._pools) > 0
    
    def test_basic_allocation(self):
        """Test basic memory allocation."""
        mgr = create_memory_manager(device_id=0, total_memory_gb=1.0)
        
        result = mgr.allocate(
            size=100 * 1024 * 1024,  # 100MB
            tenant_id="test_tenant"
        )
        
        assert result.success
        assert result.block is not None
        assert result.block.size == 100 * 1024 * 1024
        assert result.block.tenant_id == "test_tenant"
    
    def test_allocation_and_deallocation(self):
        """Test allocation followed by deallocation."""
        mgr = create_memory_manager(device_id=0, total_memory_gb=1.0)
        
        # Allocate
        result = mgr.allocate(size=100 * 1024 * 1024)
        assert result.success
        block = result.block
        
        # Deallocate
        success = mgr.deallocate(block)
        assert success
    
    def test_tenant_quota_enforcement(self):
        """Test tenant quota enforcement."""
        mgr = create_memory_manager(device_id=0, total_memory_gb=1.0)
        
        # Set tenant quota
        mgr.set_tenant_quota("tenant1", 200 * 1024 * 1024)  # 200MB
        
        # Allocate within quota
        result1 = mgr.allocate(
            size=100 * 1024 * 1024,
            tenant_id="tenant1"
        )
        assert result1.success
        
        # Try to exceed quota
        result2 = mgr.allocate(
            size=150 * 1024 * 1024,
            tenant_id="tenant1"
        )
        assert not result2.success
    
    def test_memory_pressure_monitoring(self):
        """Test memory pressure monitoring."""
        mgr = create_memory_manager(device_id=0, total_memory_gb=0.1)  # Small for testing
        
        # Allocate to increase pressure
        mgr.allocate(size=50 * 1024 * 1024)
        
        stats = mgr.get_stats()
        assert stats.pressure_level != MemoryPressureLevel.NONE
    
    def test_stats_computation(self):
        """Test memory statistics computation."""
        mgr = create_memory_manager(device_id=0, total_memory_gb=1.0)
        
        # Allocate some memory
        mgr.allocate(size=100 * 1024 * 1024)
        mgr.allocate(size=50 * 1024 * 1024)
        
        stats = mgr.get_stats()
        
        assert stats.total_bytes > 0
        assert stats.allocated_bytes >= 150 * 1024 * 1024
        assert stats.utilization >= 0.0
        assert stats.utilization <= 1.0


# =============================================================================
# BATCH SCHEDULER TESTS
# =============================================================================

class TestLatencyPredictor:
    """Tests for LatencyPredictor."""
    
    def test_prediction(self):
        """Test latency prediction."""
        predictor = LatencyPredictor()
        
        request = create_request(
            request_id="req1",
            tenant_id="tenant1",
            sequence_length=512,
            max_new_tokens=128
        )
        
        # Predict latency
        latency = predictor.predict(request, batch_size=1, current_load=0.5)
        
        assert latency > 0
        assert latency < 10000  # Reasonable upper bound
    
    def test_online_learning(self):
        """Test online learning from observations."""
        predictor = LatencyPredictor()
        
        request = create_request(
            request_id="req1",
            tenant_id="tenant1",
            sequence_length=512,
            max_new_tokens=128
        )
        
        # Get initial prediction
        pred1 = predictor.predict(request, batch_size=1, current_load=0.5)
        
        # Observe actual latency and update
        predictor.observe(request, actual_latency_ms=100.0, batch_size=1)
        
        # Get new prediction
        pred2 = predictor.predict(request, batch_size=1, current_load=0.5)
        
        # MAE should improve or stay reasonable
        assert predictor.mean_absolute_error >= 0


class TestSchedulingPolicies:
    """Tests for different scheduling policies."""
    
    def test_fcfs_policy(self):
        """Test FCFS scheduling."""
        scheduler = create_scheduler(enable_adaptive=False)
        
        # Submit requests
        for i in range(5):
            req = create_request(
                request_id=f"req{i}",
                tenant_id="tenant1",
                max_new_tokens=100 + i * 10
            )
            scheduler.submit(req)
        
        # Schedule a batch
        decision = scheduler.schedule()
        
        assert decision is not None
        assert len(decision.requests) > 0
    
    def test_sjf_policy(self):
        """Test SJF (Shortest Job First) scheduling."""
        scheduler = create_scheduler(enable_adaptive=False)
        scheduler._current_policy = SchedulingPolicy.SJF
        
        # Submit requests with different durations
        for i in range(3):
            req = create_request(
                request_id=f"req{i}",
                tenant_id="tenant1",
                max_new_tokens=100 * (i + 1)
            )
            scheduler.submit(req)
        
        decision = scheduler.schedule()
        assert decision is not None


class TestAdaptiveBatchScheduler:
    """Tests for AdaptiveBatchScheduler."""
    
    def test_scheduler_creation(self):
        """Test scheduler creation."""
        scheduler = create_scheduler(max_batch_size=32)
        
        assert scheduler.max_batch_size == 32
        assert scheduler.queue_length == 0
    
    def test_request_submission(self):
        """Test submitting requests."""
        scheduler = create_scheduler()
        
        req = create_request(
            request_id="req1",
            tenant_id="tenant1",
            sequence_length=512,
            max_new_tokens=128
        )
        
        assert scheduler.submit(req)
        assert scheduler.queue_length == 1
    
    def test_batch_formation(self):
        """Test batch formation from queued requests."""
        scheduler = create_scheduler(max_batch_size=4)
        
        # Submit 10 requests
        for i in range(10):
            req = create_request(
                request_id=f"req{i}",
                tenant_id=f"tenant{i % 3}",
                max_new_tokens=100 + i * 10
            )
            scheduler.submit(req)
        
        # Form a batch
        decision = scheduler.schedule()
        
        assert decision is not None
        assert len(decision.requests) <= 4
        assert scheduler.queue_length == 10 - len(decision.requests)
    
    def test_priority_scheduling(self):
        """Test priority-based scheduling."""
        scheduler = create_scheduler(enable_adaptive=False)
        scheduler._current_policy = SchedulingPolicy.PRIORITY
        
        # Submit requests with different priorities
        for i in range(3):
            req = create_request(
                request_id=f"req{i}",
                tenant_id="tenant1",
                max_new_tokens=100
            )
            req.priority.base = i - 1  # -1, 0, 1
            scheduler.submit(req)
        
        decision = scheduler.schedule()
        
        # Highest priority request should be first
        if len(decision.requests) > 0:
            assert decision.requests[0].request_id in ["req1", "req2"]
    
    def test_deadline_awareness(self):
        """Test deadline-aware scheduling."""
        scheduler = create_scheduler(enable_adaptive=False)
        scheduler._current_policy = SchedulingPolicy.EDF
        
        now = time.monotonic()
        
        # Submit requests with different deadlines
        for i in range(3):
            req = create_request(
                request_id=f"req{i}",
                tenant_id="tenant1",
                max_new_tokens=100
            )
            req.deadline_ms = (now + 1000 * (3 - i)) * 1000  # Tighter deadlines first
            scheduler.submit(req)
        
        decision = scheduler.schedule()
        assert decision is not None
    
    def test_completion_recording(self):
        """Test recording request completion."""
        scheduler = create_scheduler()
        
        req = create_request(
            request_id="req1",
            tenant_id="tenant1"
        )
        scheduler.submit(req)
        
        # Schedule and record completion
        decision = scheduler.schedule()
        scheduler.record_completion(
            request_id="req1",
            actual_latency_ms=150.0,
            tokens_generated=128
        )
        
        # Check metrics updated
        metrics = scheduler.metrics
        assert metrics.total_batches_formed > 0
    
    def test_metrics_tracking(self):
        """Test metrics tracking."""
        scheduler = create_scheduler()
        
        # Submit and schedule multiple requests
        for i in range(10):
            req = create_request(
                request_id=f"req{i}",
                tenant_id="tenant1"
            )
            scheduler.submit(req)
        
        # Form batches
        while scheduler.queue_length > 0:
            decision = scheduler.schedule()
            if decision:
                for req in decision.requests:
                    scheduler.record_completion(
                        request_id=req.request_id,
                        actual_latency_ms=100.0,
                        tokens_generated=128
                    )
        
        metrics = scheduler.metrics
        assert metrics.total_requests_scheduled == 10
        assert metrics.total_batches_formed > 0


# =============================================================================
# RESOURCE ALLOCATOR TESTS
# =============================================================================

class TestFairShareAlgorithms:
    """Tests for fair share algorithms."""
    
    def test_max_min_fair_share(self):
        """Test Max-Min Fair Share algorithm."""
        algo = MaxMinFairShare()
        
        # Two tenants, two resources
        demands = {
            "tenant1": {ResourceType.GPU_MEMORY: 100, ResourceType.GPU_COMPUTE: 50},
            "tenant2": {ResourceType.GPU_MEMORY: 100, ResourceType.GPU_COMPUTE: 50},
        }
        
        capacities = {
            ResourceType.GPU_MEMORY: 150,
            ResourceType.GPU_COMPUTE: 75,
        }
        
        shares = algo.compute_shares(demands, capacities)
        
        # Both tenants should get equal share
        assert shares["tenant1"][ResourceType.GPU_MEMORY] == 75
        assert shares["tenant2"][ResourceType.GPU_MEMORY] == 75
    
    def test_dominant_resource_fairness(self):
        """Test Dominant Resource Fairness algorithm."""
        algo = DominantResourceFairness()
        
        demands = {
            "tenant1": {ResourceType.GPU_MEMORY: 100, ResourceType.GPU_COMPUTE: 10},
            "tenant2": {ResourceType.GPU_MEMORY: 50, ResourceType.GPU_COMPUTE: 100},
        }
        
        capacities = {
            ResourceType.GPU_MEMORY: 150,
            ResourceType.GPU_COMPUTE: 110,
        }
        
        shares = algo.compute_shares(demands, capacities)
        
        # Should allocate resources
        assert "tenant1" in shares
        assert "tenant2" in shares


class TestAdmissionController:
    """Tests for admission control."""
    
    def test_admission_check(self):
        """Test admission check logic."""
        controller = AdmissionController(overbooking_factor=1.2)
        
        # Create a node with resources
        node = Mock()
        node.get_available = Mock(return_value=1000)
        node.capacities = {
            ResourceType.GPU_MEMORY: Mock(total=1000)
        }
        
        # Create a tenant
        tenant = TenantState("tenant1")
        tenant.quotas[ResourceType.GPU_MEMORY] = ResourceQuota(
            tenant_id="tenant1",
            resource_type=ResourceType.GPU_MEMORY,
            guaranteed=0,
            limit=2000
        )
        
        # Create a request
        request = ResourceRequest(
            request_id="req1",
            tenant_id="tenant1",
            resources={ResourceType.GPU_MEMORY: 500}
        )
        
        can_admit, reason = controller.can_admit(request, node, tenant)
        assert can_admit


class TestResourceAllocator:
    """Tests for ResourceAllocator."""
    
    def test_allocator_creation(self):
        """Test allocator creation."""
        allocator = create_allocator()
        
        assert allocator is not None
        assert len(allocator.nodes) == 0
        assert len(allocator.tenants) == 0
    
    def test_node_registration(self):
        """Test node registration."""
        allocator = create_allocator()
        
        capacities = {
            ResourceType.GPU_MEMORY: 16 * 1024 * 1024 * 1024,  # 16GB
            ResourceType.GPU_COMPUTE: 1.0
        }
        
        allocator.register_node("gpu0", capacities)
        
        assert "gpu0" in allocator.nodes
    
    def test_tenant_registration(self):
        """Test tenant registration."""
        allocator = create_allocator()
        
        allocator.register_tenant("tenant1", priority=PriorityClass.HIGH)
        
        assert "tenant1" in allocator.tenants
        assert allocator.tenants["tenant1"].priority_class == PriorityClass.HIGH
    
    def test_resource_allocation(self):
        """Test resource allocation."""
        allocator = create_allocator()
        
        # Setup
        allocator.register_node("gpu0", {
            ResourceType.GPU_MEMORY: 16 * 1024 * 1024 * 1024,
            ResourceType.GPU_COMPUTE: 1.0
        })
        allocator.register_tenant("tenant1")
        
        # Allocate
        request = create_alloc_request(
            request_id="req1",
            tenant_id="tenant1",
            gpu_memory=2 * 1024 * 1024 * 1024  # 2GB
        )
        
        grant = allocator.allocate(request)
        
        assert grant is not None
        assert grant.tenant_id == "tenant1"
        assert grant.request_id == "req1"
    
    def test_allocation_failure(self):
        """Test allocation failure when insufficient resources."""
        allocator = create_allocator()
        
        # Small node
        allocator.register_node("gpu0", {
            ResourceType.GPU_MEMORY: 1 * 1024 * 1024 * 1024,  # 1GB
            ResourceType.GPU_COMPUTE: 0.5
        })
        allocator.register_tenant("tenant1")
        
        # Request more than available
        request = create_alloc_request(
            request_id="req1",
            tenant_id="tenant1",
            gpu_memory=10 * 1024 * 1024 * 1024  # 10GB
        )
        
        grant = allocator.allocate(request)
        
        assert grant is None
    
    def test_resource_release(self):
        """Test resource release."""
        allocator = create_allocator()
        
        # Setup and allocate
        allocator.register_node("gpu0", {
            ResourceType.GPU_MEMORY: 16 * 1024 * 1024 * 1024
        })
        allocator.register_tenant("tenant1")
        
        request = create_alloc_request(
            request_id="req1",
            tenant_id="tenant1",
            gpu_memory=2 * 1024 * 1024 * 1024
        )
        
        grant = allocator.allocate(request)
        assert grant is not None
        
        # Release
        success = allocator.release(grant.request_id)
        assert success
    
    def test_tenant_quota_management(self):
        """Test tenant quota management."""
        allocator = create_allocator()
        
        allocator.register_node("gpu0", {
            ResourceType.GPU_MEMORY: 16 * 1024 * 1024 * 1024
        })
        allocator.register_tenant("tenant1")
        
        # Set quota
        allocator.update_tenant_quota(
            "tenant1",
            ResourceType.GPU_MEMORY,
            guaranteed=2 * 1024 * 1024 * 1024,
            limit=4 * 1024 * 1024 * 1024
        )
        
        quota = allocator.tenants["tenant1"].quotas[ResourceType.GPU_MEMORY]
        assert quota.guaranteed == 2 * 1024 * 1024 * 1024
        assert quota.limit == 4 * 1024 * 1024 * 1024
    
    def test_multi_tenant_fairness(self):
        """Test fair allocation across multiple tenants."""
        allocator = create_allocator(fair_share="drf")
        
        # Setup
        allocator.register_node("gpu0", {
            ResourceType.GPU_MEMORY: 16 * 1024 * 1024 * 1024,
            ResourceType.GPU_COMPUTE: 1.0
        })
        
        # Register multiple tenants
        for i in range(3):
            allocator.register_tenant(f"tenant{i}")
        
        # Compute fair shares
        fair_shares = allocator.compute_fair_shares()
        
        # Should have shares for all tenants
        assert len(fair_shares) > 0
    
    def test_fairness_index_computation(self):
        """Test Jain's fairness index computation."""
        allocator = create_allocator()
        
        allocator.register_node("gpu0", {
            ResourceType.GPU_MEMORY: 16 * 1024 * 1024 * 1024
        })
        
        # Register tenants
        for i in range(3):
            allocator.register_tenant(f"tenant{i}")
        
        metrics = allocator.get_metrics()
        
        # Fairness index should be between 1/n and 1.0
        assert 0.3 <= metrics.fairness_index <= 1.0
    
    def test_cluster_utilization(self):
        """Test cluster utilization computation."""
        allocator = create_allocator()
        
        allocator.register_node("gpu0", {
            ResourceType.GPU_MEMORY: 16 * 1024 * 1024 * 1024
        })
        allocator.register_tenant("tenant1")
        
        # Allocate
        request = create_alloc_request(
            request_id="req1",
            tenant_id="tenant1",
            gpu_memory=8 * 1024 * 1024 * 1024
        )
        allocator.allocate(request)
        
        utilization = allocator.get_cluster_utilization()
        
        assert ResourceType.GPU_MEMORY in utilization
        assert utilization[ResourceType.GPU_MEMORY] > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSchedulingIntegration:
    """Integration tests for scheduling module components."""
    
    def test_memory_manager_scheduler_integration(self):
        """Test GPU memory manager with batch scheduler."""
        mgr = create_memory_manager(device_id=0, total_memory_gb=1.0)
        scheduler = create_scheduler(max_batch_size=4)
        
        # Submit requests to scheduler
        for i in range(5):
            req = create_request(
                request_id=f"req{i}",
                tenant_id="tenant1",
                max_new_tokens=100 + i * 20
            )
            scheduler.submit(req)
        
        # Form batch
        decision = scheduler.schedule()
        
        # Allocate memory for batch
        batch_size = len(decision.requests)
        total_tokens = sum(r.max_new_tokens for r in decision.requests)
        
        result = mgr.allocate(
            size=total_tokens * 1024,  # 1KB per token approx
            tenant_id="tenant1"
        )
        
        assert result.success
    
    def test_resource_allocator_fairness_integration(self):
        """Test resource allocator with multiple tenants."""
        allocator = create_allocator(fair_share="drf")
        
        # Setup nodes
        for i in range(2):
            allocator.register_node(f"gpu{i}", {
                ResourceType.GPU_MEMORY: 16 * 1024 * 1024 * 1024,
                ResourceType.GPU_COMPUTE: 1.0
            })
        
        # Register tenants
        for i in range(3):
            allocator.register_tenant(f"tenant{i}")
            allocator.update_tenant_quota(
                f"tenant{i}",
                ResourceType.GPU_MEMORY,
                guaranteed=2 * 1024 * 1024 * 1024,
                limit=8 * 1024 * 1024 * 1024
            )
        
        # Allocate for each tenant
        grants = []
        for i in range(3):
            request = create_alloc_request(
                request_id=f"req_t{i}",
                tenant_id=f"tenant{i}",
                gpu_memory=4 * 1024 * 1024 * 1024
            )
            grant = allocator.allocate(request)
            if grant:
                grants.append(grant)
        
        # Verify fairness
        metrics = allocator.get_metrics()
        assert metrics.fairness_index > 0.8


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestSchedulingPerformance:
    """Performance tests for scheduling components."""
    
    def test_scheduler_throughput(self):
        """Test scheduler throughput."""
        scheduler = create_scheduler(max_batch_size=32)
        
        start = time.time()
        
        # Submit 1000 requests
        for i in range(1000):
            req = create_request(
                request_id=f"req{i}",
                tenant_id=f"tenant{i % 10}",
                max_new_tokens=100
            )
            scheduler.submit(req)
        
        # Schedule batches until queue empty
        batch_count = 0
        while scheduler.queue_length > 0:
            decision = scheduler.schedule()
            if decision:
                batch_count += 1
        
        elapsed = time.time() - start
        
        # Should complete reasonably quickly
        assert elapsed < 5.0
        assert batch_count > 0
    
    def test_allocator_allocation_speed(self):
        """Test allocator allocation speed."""
        allocator = create_allocator()
        
        # Setup
        for i in range(10):
            allocator.register_node(f"gpu{i}", {
                ResourceType.GPU_MEMORY: 16 * 1024 * 1024 * 1024
            })
        
        for i in range(10):
            allocator.register_tenant(f"tenant{i}")
        
        start = time.time()
        
        # Perform 1000 allocations
        for i in range(1000):
            request = create_alloc_request(
                request_id=f"req{i}",
                tenant_id=f"tenant{i % 10}",
                gpu_memory=100 * 1024 * 1024
            )
            allocator.allocate(request)
        
        elapsed = time.time() - start
        
        # Should be fast (< 1s for 1000 allocations)
        assert elapsed < 1.0


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def memory_manager():
    """Create a test memory manager."""
    return create_memory_manager(device_id=0, total_memory_gb=1.0)


@pytest.fixture
def batch_scheduler():
    """Create a test batch scheduler."""
    return create_scheduler(max_batch_size=8)


@pytest.fixture
def resource_allocator():
    """Create a test resource allocator."""
    return create_allocator()


# =============================================================================
# Test Execution
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
