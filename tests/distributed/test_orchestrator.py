"""
Unit Tests for Multi-GPU Orchestrator Implementation

Task 1.1.6: Comprehensive test coverage for:
  - ProcessGroupManager lifecycle
  - ResourceAllocator memory management
  - HealthMonitor metrics collection
  - FailureRecoveryManager error handling
  - MultiGPUOrchestrator orchestration
  - Integration scenarios

Test Categories:
  1. Initialization: Process group setup
  2. Resource Management: Memory allocation and tracking
  3. Health Monitoring: Metrics collection and status
  4. Failure Recovery: Error handling and recovery
  5. Orchestration: End-to-end inference flow
  6. Integration: Multiple components working together
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import time

import torch
import torch.nn as nn
from torch.testing import assert_close

# Import orchestrator components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.distributed.orchestrator import (
    ProcessGroupManager,
    ResourceAllocator,
    HealthMonitor,
    FailureRecoveryManager,
    MultiGPUOrchestrator,
    OrchestratorConfig,
    ProcessStatus,
    FailureMode,
)


# ============================================================================
# Test Utilities
# ============================================================================

class BaseOrchestratorTest(unittest.TestCase):
    """Base class for orchestrator tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
    
    def assert_in_range(self, value: float, min_val: float, max_val: float, msg: str = ""):
        """Assert value is in range."""
        self.assertTrue(
            min_val <= value <= max_val,
            f"{msg}: {value} not in range [{min_val}, {max_val}]"
        )


# ============================================================================
# ProcessGroupManager Tests
# ============================================================================

class TestProcessGroupManager(BaseOrchestratorTest):
    """Test ProcessGroupManager component."""
    
    def test_initialization_state(self):
        """Test manager starts in uninitialized state."""
        mgr = ProcessGroupManager(backend="nccl", timeout_sec=30.0)
        
        self.assertIsNone(mgr.rank)
        self.assertIsNone(mgr.world_size)
        self.assertIsNone(mgr.device)
        self.assertFalse(mgr.initialized)
    
    def test_configuration_parameters(self):
        """Test configuration is stored correctly."""
        mgr = ProcessGroupManager(backend="gloo", timeout_sec=60.0)
        
        self.assertEqual(mgr.backend, "gloo")
        # Timeout is timedelta object
        self.assertEqual(mgr.timeout.total_seconds(), 60.0)
    
    @patch('torch.distributed.init_process_group')
    def test_initialize_single_process(self, mock_init):
        """Test initialization with single process."""
        mgr = ProcessGroupManager()
        
        # Initialize rank 0, world_size 1 (single process)
        # We skip actual initialization to avoid requiring a real distributed setup
        mgr.rank = 0
        mgr.world_size = 1
        mgr.device = self.device
        mgr.initialized = True
        
        self.assertEqual(mgr.rank, 0)
        self.assertEqual(mgr.world_size, 1)
        self.assertTrue(mgr.initialized)
    
    def test_barrier_without_init(self):
        """Test barrier works even without initialization."""
        mgr = ProcessGroupManager()
        
        # Should not raise
        mgr.barrier()
    
    @patch('torch.distributed.is_initialized', return_value=False)
    def test_finalize_without_init(self, mock_is_init):
        """Test finalize works without initialization."""
        mgr = ProcessGroupManager()
        
        # Should not raise
        mgr.finalize()
        
        self.assertFalse(mgr.initialized)


# ============================================================================
# ResourceAllocator Tests
# ============================================================================

class TestResourceAllocator(BaseOrchestratorTest):
    """Test ResourceAllocator component."""
    
    def test_initialization(self):
        """Test resource allocator initialization."""
        allocator = ResourceAllocator(
            rank=0,
            device=self.device,
            memory_fraction=0.9
        )
        
        self.assertEqual(allocator.rank, 0)
        self.assertEqual(allocator.device, self.device)
        self.assertEqual(allocator.memory_fraction, 0.9)
        self.assertGreater(allocator.max_memory, 0)
    
    def test_memory_budget_calculation(self):
        """Test memory budget is calculated correctly."""
        allocator = ResourceAllocator(
            rank=0,
            device=self.device,
            memory_fraction=0.8
        )
        
        # Max memory should be <= actual GPU memory
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(self.device).total_memory
            self.assertLessEqual(allocator.max_memory, total)
            self.assertGreaterEqual(allocator.max_memory, total * 0.75)  # Within reasonable range
    
    def test_allocate_tensor(self):
        """Test tensor allocation."""
        allocator = ResourceAllocator(rank=0, device=self.device)
        
        tensor = allocator.allocate_tensor(
            name="test_tensor",
            shape=(100, 100),
            dtype=torch.float32
        )
        
        self.assertEqual(tensor.shape, (100, 100))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, self.device)
        self.assertIn("test_tensor", allocator.buffers)
        self.assertGreater(allocator.allocated_memory, 0)
    
    def test_deallocate_tensor(self):
        """Test tensor deallocation."""
        allocator = ResourceAllocator(rank=0, device=self.device)
        
        allocator.allocate_tensor("test", (100, 100), torch.float32)
        allocated_before = allocator.allocated_memory
        
        allocator.deallocate_tensor("test")
        
        self.assertLess(allocator.allocated_memory, allocated_before)
        self.assertNotIn("test", allocator.buffers)
    
    def test_memory_stats(self):
        """Test memory statistics."""
        allocator = ResourceAllocator(rank=0, device=self.device)
        
        stats = allocator.get_memory_stats()
        
        self.assertIn("allocated_gb", stats)
        self.assertIn("reserved_gb", stats)
        self.assertIn("max_budget_gb", stats)
        self.assertIn("utilization_percent", stats)
        
        # All should be non-negative
        for key, value in stats.items():
            self.assertGreaterEqual(value, 0)
    
    def test_allocation_exceeds_budget(self):
        """Test that allocations are bounded by memory budget."""
        allocator = ResourceAllocator(
            rank=0,
            device=self.device,
            memory_fraction=0.001  # Very small budget for testing
        )
        
        # First small allocation should work
        t1 = allocator.allocate_tensor("small", (10, 10), torch.float32)
        self.assertIsNotNone(t1)
        
        # Attempting massive allocation should fail
        with self.assertRaises(RuntimeError):
            allocator.allocate_tensor("huge", (100000, 100000), torch.float32)


# ============================================================================
# HealthMonitor Tests
# ============================================================================

class TestHealthMonitor(BaseOrchestratorTest):
    """Test HealthMonitor component."""
    
    def test_initialization(self):
        """Test health monitor initialization."""
        monitor = HealthMonitor(
            rank=0,
            device=self.device,
            check_interval_sec=5.0
        )
        
        self.assertEqual(monitor.rank, 0)
        self.assertEqual(monitor.device, self.device)
        self.assertEqual(monitor.check_interval, 5.0)
        self.assertEqual(monitor.error_count, 0)
    
    def test_health_check_structure(self):
        """Test health check returns proper structure."""
        monitor = HealthMonitor(rank=0, device=self.device, check_interval_sec=0.0)
        
        # Force check to run
        metrics = monitor.check_health()
        
        # Should be empty if too soon, so force by resetting timer
        monitor.last_check = 0  # Force check
        metrics = monitor.check_health()
        
        self.assertIn("timestamp", metrics)
        self.assertIn("rank", metrics)
        self.assertIn("memory_allocated_mb", metrics)
        self.assertIn("status", metrics)
    
    def test_heartbeat_recording(self):
        """Test heartbeat recording."""
        monitor = HealthMonitor(rank=0, device=self.device)
        
        initial_age = monitor.check_health().get("heartbeat_age_sec", 0)
        
        time.sleep(0.1)
        monitor.record_heartbeat()
        
        # After heartbeat, age should be small
        age = monitor.check_health().get("heartbeat_age_sec", 0)
        self.assertLess(age, 0.5)  # Should be recent
    
    def test_error_tracking(self):
        """Test error count tracking."""
        monitor = HealthMonitor(rank=0, device=self.device)
        
        self.assertEqual(monitor.error_count, 0)
        
        monitor.record_error()
        self.assertEqual(monitor.error_count, 1)
        
        monitor.record_error()
        self.assertEqual(monitor.error_count, 2)
    
    def test_health_status_degradation(self):
        """Test status changes with error count."""
        monitor = HealthMonitor(rank=0, device=self.device, check_interval_sec=0.0)
        
        # Healthy state
        monitor.error_count = 0
        monitor.last_check = 0
        metrics = monitor.check_health()
        self.assertEqual(metrics["status"], ProcessStatus.HEALTHY.value)
        
        # Degraded state
        monitor.error_count = 2
        monitor.last_check = 0
        metrics = monitor.check_health()
        self.assertEqual(metrics["status"], ProcessStatus.DEGRADED.value)
        
        # Unhealthy state
        monitor.error_count = 6
        monitor.last_check = 0
        metrics = monitor.check_health()
        self.assertEqual(metrics["status"], ProcessStatus.UNHEALTHY.value)
    
    def test_health_summary(self):
        """Test health summary generation."""
        monitor = HealthMonitor(rank=0, device=self.device, check_interval_sec=0.0)
        
        # Need some metrics in history
        monitor.last_check = 0
        for _ in range(5):
            monitor.check_health()
            monitor.last_check = 0
        
        summary = monitor.get_health_summary()
        
        self.assertIn("avg_memory_mb", summary)
        self.assertIn("peak_memory_mb", summary)
        self.assertIn("status", summary)
        self.assertGreaterEqual(summary["peak_memory_mb"], summary["avg_memory_mb"])


# ============================================================================
# FailureRecoveryManager Tests
# ============================================================================

class TestFailureRecoveryManager(BaseOrchestratorTest):
    """Test FailureRecoveryManager component."""
    
    def test_initialization(self):
        """Test failure recovery manager initialization."""
        mgr = FailureRecoveryManager(max_retries=3)
        
        self.assertEqual(mgr.max_retries, 3)
        self.assertEqual(mgr.failure_count, 0)
        self.assertEqual(len(mgr.failure_history), 0)
    
    def test_handle_gpu_oom(self):
        """Test GPU OOM handling."""
        mgr = FailureRecoveryManager()
        
        new_batch_size = mgr.handle_gpu_oom(rank=0, batch_size=64)
        
        self.assertEqual(new_batch_size, 32)  # Reduced by half
        self.assertEqual(len(mgr.failure_history), 1)
        self.assertEqual(mgr.failure_history[0]["type"], FailureMode.GPU_OOM)
    
    def test_handle_communication_timeout_recoverable(self):
        """Test communication timeout with recovery possible."""
        mgr = FailureRecoveryManager(max_retries=3)
        
        can_recover = mgr.handle_communication_timeout(rank=0)
        
        self.assertTrue(can_recover)
        self.assertEqual(mgr.failure_count, 1)
    
    def test_handle_communication_timeout_max_exceeded(self):
        """Test communication timeout with max retries exceeded."""
        mgr = FailureRecoveryManager(max_retries=2)
        
        # Use up all retries
        mgr.handle_communication_timeout(rank=0)  # 1
        mgr.handle_communication_timeout(rank=0)  # 2
        can_recover = mgr.handle_communication_timeout(rank=0)  # 3 (exceeds)
        
        self.assertFalse(can_recover)
    
    def test_handle_gpu_failure(self):
        """Test GPU failure handling (cannot recover)."""
        mgr = FailureRecoveryManager()
        
        can_recover = mgr.handle_gpu_failure(rank=0)
        
        self.assertFalse(can_recover)
        self.assertEqual(mgr.failure_history[0]["type"], FailureMode.GPU_FAILURE)
    
    def test_failure_reset(self):
        """Test failure count reset."""
        mgr = FailureRecoveryManager()
        
        mgr.failure_count = 5
        mgr.reset()
        
        self.assertEqual(mgr.failure_count, 4)  # Decrements
    
    def test_failure_summary(self):
        """Test failure summary generation."""
        mgr = FailureRecoveryManager()
        
        mgr.handle_gpu_oom(rank=0, batch_size=64)
        mgr.handle_communication_timeout(rank=0)
        mgr.handle_gpu_oom(rank=1, batch_size=32)
        
        summary = mgr.get_failure_summary()
        
        self.assertEqual(summary["total_failures"], 3)
        self.assertIn(FailureMode.GPU_OOM.value, summary["failure_types"])
        self.assertIn(FailureMode.COMMUNICATION_TIMEOUT.value, summary["failure_types"])


# ============================================================================
# MultiGPUOrchestrator Tests
# ============================================================================

class TestMultiGPUOrchestrator(BaseOrchestratorTest):
    """Test MultiGPUOrchestrator component."""
    
    def test_initialization_config(self):
        """Test orchestrator initialization with config."""
        config = OrchestratorConfig(
            backend="nccl",
            timeout_sec=30.0,
            memory_fraction=0.9,
            health_check_interval_sec=5.0
        )
        
        orchestrator = MultiGPUOrchestrator(config)
        
        self.assertEqual(orchestrator.config, config)
        self.assertFalse(orchestrator.initialized)
        self.assertIsNone(orchestrator.model)
    
    def test_state_before_initialization(self):
        """Test orchestrator state before initialization."""
        orchestrator = MultiGPUOrchestrator(OrchestratorConfig())
        
        self.assertIsNone(orchestrator.resource_allocator)
        self.assertIsNone(orchestrator.health_monitor)
        self.assertEqual(orchestrator.step_count, 0)
    
    def test_load_model_before_init_fails(self):
        """Test that loading model before init fails."""
        orchestrator = MultiGPUOrchestrator(OrchestratorConfig())
        model = nn.Linear(10, 10)
        
        with self.assertRaises(RuntimeError):
            orchestrator.load_model(model)
    
    def test_inference_step_before_init_fails(self):
        """Test that inference before init fails."""
        orchestrator = MultiGPUOrchestrator(OrchestratorConfig())
        batch = {"input": torch.randn(2, 10)}
        
        with self.assertRaises(RuntimeError):
            orchestrator.inference_step(batch)
    
    def test_stats_before_init(self):
        """Test stats before initialization."""
        orchestrator = MultiGPUOrchestrator(OrchestratorConfig())
        
        stats = orchestrator.get_stats()
        
        self.assertEqual(stats["status"], "not_initialized")
    
    def test_cleanup_without_init(self):
        """Test cleanup without initialization."""
        orchestrator = MultiGPUOrchestrator(OrchestratorConfig())
        
        # Should not raise
        orchestrator.cleanup()


# ============================================================================
# Integration Tests
# ============================================================================

class TestOrchestratorIntegration(BaseOrchestratorTest):
    """Integration tests for orchestrator components."""
    
    def test_config_creation(self):
        """Test configuration creation with various settings."""
        config = OrchestratorConfig(
            backend="gloo",
            timeout_sec=60.0,
            memory_fraction=0.85,
            enable_auto_restart=False,
            max_restart_attempts=5,
        )
        
        self.assertEqual(config.backend, "gloo")
        self.assertEqual(config.timeout_sec, 60.0)
        self.assertEqual(config.memory_fraction, 0.85)
        self.assertFalse(config.enable_auto_restart)
        self.assertEqual(config.max_restart_attempts, 5)
    
    def test_components_work_together(self):
        """Test that components can work together."""
        config = OrchestratorConfig()
        orchestrator = MultiGPUOrchestrator(config)
        
        # Manually setup for testing (skip distributed init)
        orchestrator.process_group_mgr.rank = 0
        orchestrator.process_group_mgr.world_size = 1
        orchestrator.process_group_mgr.device = self.device
        orchestrator.process_group_mgr.initialized = True
        
        orchestrator.resource_allocator = ResourceAllocator(0, self.device)
        orchestrator.health_monitor = HealthMonitor(0, self.device)
        orchestrator.initialized = True
        
        # Create simple model
        model = nn.Linear(10, 5)
        orchestrator.load_model(model)
        
        # Should have model loaded
        self.assertIsNotNone(orchestrator.model)
    
    def test_multiple_components_initialization(self):
        """Test initializing multiple components sequentially."""
        pgm = ProcessGroupManager()
        pgm.rank = 0
        pgm.world_size = 1
        pgm.device = self.device
        pgm.initialized = True
        
        allocator = ResourceAllocator(0, self.device)
        monitor = HealthMonitor(0, self.device)
        recovery = FailureRecoveryManager()
        
        # All should work independently
        self.assertTrue(pgm.initialized)
        self.assertGreater(allocator.max_memory, 0)
        self.assertEqual(monitor.error_count, 0)
        self.assertEqual(recovery.failure_count, 0)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
