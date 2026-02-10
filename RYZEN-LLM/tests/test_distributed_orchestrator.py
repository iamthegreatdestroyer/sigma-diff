"""
Test Distributed Inference Orchestrator

Tests multi-GPU orchestration, health monitoring, and performance tracking.
"""

import pytest
import torch
import time
from unittest.mock import Mock, patch
from src.distributed.multi_gpu_orchestrator import (
    DistributedOrchestrator,
    GPUHealthMonitor,
    GPUStats,
    BackendType,
    spawn_distributed_processes,
    example_distributed_inference
)


class TestGPUHealthMonitor:
    """Test GPU health monitoring functionality."""

    def test_initialization(self):
        """Test health monitor initialization."""
        monitor = GPUHealthMonitor(device_count=4)
        assert monitor.device_count == 4
        assert len(monitor.stats) == 0

    def test_update_and_check_health(self):
        """Test updating stats and health checking."""
        monitor = GPUHealthMonitor(device_count=2)

        # Add healthy GPU stats
        healthy_stats = GPUStats(
            device_id=0,
            memory_used=2048,  # 2GB used
            memory_total=8192,  # 8GB total
            utilization=60.0,
            temperature=70.0,
            last_heartbeat=time.time(),
            active_requests=1,
            avg_latency=25.0
        )
        monitor.update_stats(0, healthy_stats)

        # Add unhealthy GPU stats (high temperature)
        unhealthy_stats = GPUStats(
            device_id=1,
            memory_used=2048,
            memory_total=8192,
            utilization=60.0,
            temperature=90.0,  # Too hot
            last_heartbeat=time.time(),
            active_requests=1,
            avg_latency=25.0
        )
        monitor.update_stats(1, unhealthy_stats)

        # Check health
        assert monitor.is_healthy(0) == True
        assert monitor.is_healthy(1) == False

        # Check healthy devices list
        healthy_devices = monitor.get_healthy_devices()
        assert 0 in healthy_devices
        assert 1 not in healthy_devices

    def test_least_loaded_device(self):
        """Test finding least loaded device."""
        monitor = GPUHealthMonitor(device_count=3)

        # Device 0: 80% utilization
        monitor.update_stats(0, GPUStats(0, 2048, 8192, 80.0, 70.0, time.time(), 1, 25.0))

        # Device 1: 50% utilization (least loaded)
        monitor.update_stats(1, GPUStats(1, 2048, 8192, 50.0, 70.0, time.time(), 1, 25.0))

        # Device 2: 70% utilization
        monitor.update_stats(2, GPUStats(2, 2048, 8192, 70.0, 70.0, time.time(), 1, 25.0))

        assert monitor.get_least_loaded_device() == 1


class TestDistributedOrchestrator:
    """Test distributed orchestrator functionality."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = DistributedOrchestrator(world_size=4)
        assert orchestrator.world_size == 4
        assert orchestrator.rank == -1
        assert orchestrator.is_initialized == False

    @patch('torch.cuda.is_available', return_value=False)
    def test_device_assignment_cpu(self, mock_cuda):
        """Test device assignment on CPU-only system."""
        orchestrator = DistributedOrchestrator(world_size=2)

        device_0 = orchestrator.get_device_for_rank(0)
        device_1 = orchestrator.get_device_for_rank(1)

        assert device_0 == torch.device('cpu')
        assert device_1 == torch.device('cpu')

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=4)
    def test_device_assignment_gpu(self, mock_count, mock_cuda):
        """Test device assignment on GPU system."""
        orchestrator = DistributedOrchestrator(world_size=2)

        device_0 = orchestrator.get_device_for_rank(0)
        device_1 = orchestrator.get_device_for_rank(1)
        device_2 = orchestrator.get_device_for_rank(2)  # Round-robin assignment

        assert device_0 == torch.device('cuda:0')
        assert device_1 == torch.device('cuda:1')
        assert device_2 == torch.device('cuda:2')  # 2 % 4 = 2

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        orchestrator = DistributedOrchestrator(world_size=1)

        # Add some performance data
        orchestrator.update_performance_stats(25.0, success=True)
        orchestrator.update_performance_stats(30.0, success=True)
        orchestrator.update_performance_stats(20.0, success=False)

        summary = orchestrator.get_performance_summary()

        assert summary['total_requests'] == 3
        assert abs(summary['average_latency'] - 25.0) < 0.1  # (25+30+20)/3 = 25
        assert summary['error_rate'] == 1/3  # 1 error out of 3 requests


class TestDistributedExecution:
    """Test distributed execution functionality."""

    def test_single_process_execution(self):
        """Test single-process distributed execution."""
        test_input = torch.randn(2, 4)

        # Run in single process mode
        result = spawn_distributed_processes(1, example_distributed_inference, test_input)

        # Result should be modified by rank 0
        expected = test_input + 0  # rank 0
        assert torch.allclose(result, expected)

    def test_example_inference_function(self):
        """Test the example inference function."""
        orchestrator = DistributedOrchestrator(world_size=1)
        # Don't initialize distributed group for this test
        orchestrator.rank = 0  # Set rank manually
        test_input = torch.randn(3, 5)

        result = example_distributed_inference(orchestrator, test_input)

        # Should add rank (0) to input
        expected = test_input + 0
        assert torch.allclose(result, expected)

        # Check performance stats were updated
        summary = orchestrator.get_performance_summary()
        assert summary['total_requests'] == 1
        assert summary['average_latency'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
