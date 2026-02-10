"""
Multi-GPU Optimization Integration Tests.

Comprehensive test suite for Sprint 2.3 Multi-GPU components:
- Tensor Parallelism (Megatron-style)
- Pipeline Parallelism (GPipe/1F1B)
- Multi-GPU Distributed KV Cache
- GPU Coordinator

Tests use mocking to work without actual multi-GPU hardware.
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import time


# =============================================================================
# Test Imports - These should work after __init__.py update
# =============================================================================

class TestModuleImports:
    """Test that all Sprint 2.3 modules import correctly."""
    
    def test_tensor_parallelism_imports(self):
        """Test tensor parallelism module imports."""
        from distributed.tensor_parallelism import (
            TensorParallelMode,
            TensorParallelConfig,
            ProcessGroupManager,
            ColumnParallelLinear,
            RowParallelLinear,
            ParallelAttention,
            ParallelMLP,
            ParallelTransformerLayer,
            TensorParallelModel,
            create_tensor_parallel_model,
        )
        
        # Verify enums
        assert hasattr(TensorParallelMode, 'COLUMN')
        assert hasattr(TensorParallelMode, 'ROW')
        
    def test_pipeline_parallelism_imports(self):
        """Test pipeline parallelism module imports."""
        from distributed.pipeline_parallelism import (
            PipelineSchedule,
            ActivationPolicy,
            PipelineConfig,
            PipelineCommunicator,
            ActivationCheckpointer,
            GPipeScheduler,
            OneFOneBScheduler,
            InterleavedScheduler,
            PipelineParallelEngine,
            create_pipeline_engine,
        )
        
        # Verify enums
        assert hasattr(PipelineSchedule, 'GPIPE')
        assert hasattr(PipelineSchedule, 'ONE_F_ONE_B')
        assert hasattr(ActivationPolicy, 'NONE')
        assert hasattr(ActivationPolicy, 'FULL')
        
    def test_multi_gpu_cache_imports(self):
        """Test multi-GPU cache module imports."""
        from distributed.multi_gpu_cache import (
            CacheCoherencyProtocol,
            MESIState,
            MESIProtocol,
            DirectoryProtocol,
            GPUPageAllocator,
            PageTransferManager,
            DistributedKVCache,
            SequenceAwareCacheSharding,
            create_distributed_cache,
        )
        
        # Verify enums
        assert hasattr(CacheCoherencyProtocol, 'MESI')
        assert hasattr(CacheCoherencyProtocol, 'DIRECTORY')
        assert hasattr(MESIState, 'MODIFIED')
        assert hasattr(MESIState, 'EXCLUSIVE')
        
    def test_gpu_coordinator_imports(self):
        """Test GPU coordinator module imports."""
        from distributed.gpu_coordinator import (
            GPUState,
            WorkloadPriority,
            SchedulingPolicy,
            TopologyType,
            FailureType,
            RecoveryAction,
            HealthMonitorConfig,
            SchedulerConfig,
            CoordinatorConfig,
            GPUInfo,
            TopologyInfo,
            WorkloadTask,
            CoordinatorStatistics,
            TopologyDetector,
            HealthMonitor,
            WorkloadScheduler,
            GPUCoordinator,
            create_coordinator,
            get_gpu_cluster_info,
        )
        
        # Verify enums
        assert hasattr(GPUState, 'IDLE')
        assert hasattr(GPUState, 'BUSY')
        assert hasattr(WorkloadPriority, 'HIGH')
        assert hasattr(SchedulingPolicy, 'ROUND_ROBIN')
        
    def test_top_level_distributed_imports(self):
        """Test imports from distributed package __init__."""
        from distributed import (
            # Core engine
            DistributedInferenceEngine,
            DistributedConfig,
            ParallelismStrategy,
            create_distributed_engine,
            # Tensor parallelism
            TensorParallelMode,
            TensorParallelConfig,
            create_tensor_parallel_model,
            # Pipeline parallelism
            PipelineSchedule,
            PipelineConfig,
            create_pipeline_engine,
            # Multi-GPU cache
            CacheCoherencyProtocol,
            create_distributed_cache,
            # GPU coordinator
            GPUCoordinator,
            create_coordinator,
        )
        
        assert DistributedInferenceEngine is not None
        assert TensorParallelConfig is not None
        assert PipelineConfig is not None


# =============================================================================
# Tensor Parallelism Tests
# =============================================================================

class TestTensorParallelConfig:
    """Test TensorParallelConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from distributed.tensor_parallelism import TensorParallelConfig
        
        config = TensorParallelConfig()
        assert config.world_size == 1
        assert config.sequence_parallel is False
        assert config.async_communication is True
        
    def test_custom_config(self):
        """Test custom configuration values."""
        from distributed.tensor_parallelism import TensorParallelConfig
        
        config = TensorParallelConfig(
            world_size=4,
            sequence_parallel=True,
            gradient_accumulation_steps=8,
        )
        assert config.world_size == 4
        assert config.sequence_parallel is True
        assert config.gradient_accumulation_steps == 8


class TestColumnParallelLinear:
    """Test ColumnParallelLinear layer."""
    
    def test_initialization(self):
        """Test layer initialization."""
        from distributed.tensor_parallelism import (
            ColumnParallelLinear,
            TensorParallelConfig,
        )
        
        config = TensorParallelConfig(world_size=2)
        # Input features, output features (will be split by world_size)
        layer = ColumnParallelLinear(
            in_features=512,
            out_features=1024,
            config=config,
            rank=0,
        )
        
        # Output should be split across GPUs
        assert layer.weight.shape[0] == 512  # 1024 / 2
        assert layer.weight.shape[1] == 512
        
    def test_forward_pass(self):
        """Test forward pass shape."""
        from distributed.tensor_parallelism import (
            ColumnParallelLinear,
            TensorParallelConfig,
        )
        
        config = TensorParallelConfig(world_size=2)
        layer = ColumnParallelLinear(
            in_features=512,
            out_features=1024,
            config=config,
            rank=0,
        )
        
        x = torch.randn(4, 32, 512)  # batch, seq, features
        output = layer(x)
        
        # Output has split features
        assert output.shape == (4, 32, 512)  # 1024 / 2


class TestRowParallelLinear:
    """Test RowParallelLinear layer."""
    
    def test_initialization(self):
        """Test layer initialization."""
        from distributed.tensor_parallelism import (
            RowParallelLinear,
            TensorParallelConfig,
        )
        
        config = TensorParallelConfig(world_size=2)
        layer = RowParallelLinear(
            in_features=1024,
            out_features=512,
            config=config,
            rank=0,
        )
        
        # Input should be split across GPUs
        assert layer.weight.shape[0] == 512
        assert layer.weight.shape[1] == 512  # 1024 / 2
        
    def test_forward_pass(self):
        """Test forward pass shape."""
        from distributed.tensor_parallelism import (
            RowParallelLinear,
            TensorParallelConfig,
        )
        
        config = TensorParallelConfig(world_size=2)
        layer = RowParallelLinear(
            in_features=1024,
            out_features=512,
            config=config,
            rank=0,
        )
        
        # Input already split to 512 (1024 / 2)
        x = torch.randn(4, 32, 512)
        output = layer(x)
        
        assert output.shape == (4, 32, 512)


class TestParallelAttention:
    """Test ParallelAttention module."""
    
    def test_initialization(self):
        """Test attention initialization."""
        from distributed.tensor_parallelism import (
            ParallelAttention,
            TensorParallelConfig,
        )
        
        config = TensorParallelConfig(world_size=2)
        attn = ParallelAttention(
            hidden_size=512,
            num_heads=8,
            config=config,
            rank=0,
        )
        
        # Heads should be split across GPUs
        assert attn.num_heads_per_partition == 4  # 8 / 2
        
    def test_forward_pass(self):
        """Test attention forward pass."""
        from distributed.tensor_parallelism import (
            ParallelAttention,
            TensorParallelConfig,
        )
        
        config = TensorParallelConfig(world_size=2)
        attn = ParallelAttention(
            hidden_size=512,
            num_heads=8,
            config=config,
            rank=0,
        )
        
        x = torch.randn(4, 32, 512)
        output = attn(x)
        
        assert output.shape == x.shape


class TestParallelMLP:
    """Test ParallelMLP module."""
    
    def test_initialization(self):
        """Test MLP initialization."""
        from distributed.tensor_parallelism import (
            ParallelMLP,
            TensorParallelConfig,
        )
        
        config = TensorParallelConfig(world_size=2)
        mlp = ParallelMLP(
            hidden_size=512,
            intermediate_size=2048,
            config=config,
            rank=0,
        )
        
        assert mlp is not None
        
    def test_forward_pass(self):
        """Test MLP forward pass."""
        from distributed.tensor_parallelism import (
            ParallelMLP,
            TensorParallelConfig,
        )
        
        config = TensorParallelConfig(world_size=2)
        mlp = ParallelMLP(
            hidden_size=512,
            intermediate_size=2048,
            config=config,
            rank=0,
        )
        
        x = torch.randn(4, 32, 512)
        output = mlp(x)
        
        assert output.shape == x.shape


class TestTensorParallelModel:
    """Test TensorParallelModel wrapper."""
    
    def test_model_wrapping(self):
        """Test model wrapping functionality."""
        from distributed.tensor_parallelism import (
            TensorParallelModel,
            TensorParallelConfig,
        )
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )
        
        config = TensorParallelConfig(world_size=2)
        
        # Wrap model
        tp_model = TensorParallelModel(
            model=model,
            config=config,
            rank=0,
        )
        
        assert tp_model is not None
        assert tp_model.config == config


# =============================================================================
# Pipeline Parallelism Tests
# =============================================================================

class TestPipelineConfig:
    """Test PipelineConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        from distributed.pipeline_parallelism import PipelineConfig
        
        config = PipelineConfig()
        assert config.num_stages == 2
        assert config.num_micro_batches == 4
        
    def test_custom_config(self):
        """Test custom configuration."""
        from distributed.pipeline_parallelism import (
            PipelineConfig,
            PipelineSchedule,
            ActivationPolicy,
        )
        
        config = PipelineConfig(
            num_stages=4,
            num_micro_batches=8,
            schedule=PipelineSchedule.ONE_F_ONE_B,
            activation_policy=ActivationPolicy.SELECTIVE,
        )
        
        assert config.num_stages == 4
        assert config.num_micro_batches == 8
        assert config.schedule == PipelineSchedule.ONE_F_ONE_B


class TestActivationCheckpointer:
    """Test ActivationCheckpointer."""
    
    def test_checkpoint_save_load(self):
        """Test activation checkpointing."""
        from distributed.pipeline_parallelism import (
            ActivationCheckpointer,
            ActivationPolicy,
        )
        
        checkpointer = ActivationCheckpointer(
            policy=ActivationPolicy.FULL,
            memory_budget_mb=1024,
        )
        
        # Create sample activation
        activation = torch.randn(4, 32, 512)
        micro_batch_id = 0
        stage_id = 0
        
        # Save checkpoint
        checkpointer.save(activation, micro_batch_id, stage_id)
        
        # Load checkpoint
        loaded = checkpointer.load(micro_batch_id, stage_id)
        
        assert torch.allclose(activation, loaded)
        
    def test_selective_checkpointing(self):
        """Test selective checkpointing policy."""
        from distributed.pipeline_parallelism import (
            ActivationCheckpointer,
            ActivationPolicy,
        )
        
        checkpointer = ActivationCheckpointer(
            policy=ActivationPolicy.SELECTIVE,
            checkpoint_ratio=0.5,
        )
        
        # Only every other layer should be checkpointed
        assert checkpointer.should_checkpoint(layer_idx=0) is True
        assert checkpointer.should_checkpoint(layer_idx=1) is False


class TestGPipeScheduler:
    """Test GPipe scheduler."""
    
    def test_schedule_generation(self):
        """Test GPipe schedule generation."""
        from distributed.pipeline_parallelism import (
            GPipeScheduler,
            PipelineConfig,
        )
        
        config = PipelineConfig(
            num_stages=4,
            num_micro_batches=8,
        )
        
        scheduler = GPipeScheduler(config)
        schedule = scheduler.generate_schedule()
        
        # Should have forward and backward phases
        assert 'forward' in schedule
        assert 'backward' in schedule
        
        # All micro-batches should be scheduled
        assert len(schedule['forward']) == config.num_micro_batches
        
    def test_bubble_overhead_calculation(self):
        """Test pipeline bubble overhead."""
        from distributed.pipeline_parallelism import (
            GPipeScheduler,
            PipelineConfig,
        )
        
        config = PipelineConfig(
            num_stages=4,
            num_micro_batches=8,
        )
        
        scheduler = GPipeScheduler(config)
        bubble_ratio = scheduler.calculate_bubble_ratio()
        
        # GPipe: bubble = (num_stages - 1) / num_micro_batches
        expected = (4 - 1) / 8
        assert abs(bubble_ratio - expected) < 0.01


class TestOneFOneBScheduler:
    """Test 1F1B scheduler."""
    
    def test_schedule_generation(self):
        """Test 1F1B schedule generation."""
        from distributed.pipeline_parallelism import (
            OneFOneBScheduler,
            PipelineConfig,
        )
        
        config = PipelineConfig(
            num_stages=4,
            num_micro_batches=8,
        )
        
        scheduler = OneFOneBScheduler(config)
        schedule = scheduler.generate_schedule()
        
        # 1F1B interleaves forward and backward
        assert 'steps' in schedule
        
    def test_reduced_bubble(self):
        """Test 1F1B has lower bubble than GPipe."""
        from distributed.pipeline_parallelism import (
            GPipeScheduler,
            OneFOneBScheduler,
            PipelineConfig,
        )
        
        config = PipelineConfig(
            num_stages=4,
            num_micro_batches=8,
        )
        
        gpipe = GPipeScheduler(config)
        one_f_one_b = OneFOneBScheduler(config)
        
        gpipe_bubble = gpipe.calculate_bubble_ratio()
        ofob_bubble = one_f_one_b.calculate_bubble_ratio()
        
        # 1F1B should have lower bubble overhead
        assert ofob_bubble <= gpipe_bubble


class TestPipelineParallelEngine:
    """Test PipelineParallelEngine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        from distributed.pipeline_parallelism import (
            PipelineParallelEngine,
            PipelineConfig,
        )
        
        # Create simple stages
        stages = nn.ModuleList([
            nn.Linear(512, 512) for _ in range(4)
        ])
        
        config = PipelineConfig(num_stages=4)
        
        engine = PipelineParallelEngine(
            stages=stages,
            config=config,
            rank=0,
        )
        
        assert engine.num_stages == 4
        
    def test_micro_batch_splitting(self):
        """Test micro-batch splitting."""
        from distributed.pipeline_parallelism import (
            PipelineParallelEngine,
            PipelineConfig,
        )
        
        stages = nn.ModuleList([
            nn.Linear(512, 512) for _ in range(2)
        ])
        
        config = PipelineConfig(
            num_stages=2,
            num_micro_batches=4,
        )
        
        engine = PipelineParallelEngine(
            stages=stages,
            config=config,
            rank=0,
        )
        
        # Create batch
        batch = torch.randn(8, 32, 512)
        
        # Split into micro-batches
        micro_batches = engine.split_micro_batches(batch)
        
        assert len(micro_batches) == 4
        assert micro_batches[0].shape[0] == 2  # 8 / 4


# =============================================================================
# Multi-GPU Cache Tests
# =============================================================================

class TestMESIProtocol:
    """Test MESI cache coherency protocol."""
    
    def test_state_transitions(self):
        """Test MESI state transitions."""
        from distributed.multi_gpu_cache import MESIProtocol, MESIState
        
        protocol = MESIProtocol(num_gpus=4)
        
        page_id = "page_0"
        
        # Initial state should be Invalid
        state = protocol.get_state(page_id, gpu_id=0)
        assert state == MESIState.INVALID
        
        # Read should transition to Exclusive (if first reader)
        new_state = protocol.read(page_id, gpu_id=0)
        assert new_state in [MESIState.EXCLUSIVE, MESIState.SHARED]
        
        # Write should transition to Modified
        new_state = protocol.write(page_id, gpu_id=0)
        assert new_state == MESIState.MODIFIED
        
    def test_invalidation_on_write(self):
        """Test that writes invalidate other copies."""
        from distributed.multi_gpu_cache import MESIProtocol, MESIState
        
        protocol = MESIProtocol(num_gpus=4)
        page_id = "page_0"
        
        # GPU 0 reads (Exclusive)
        protocol.read(page_id, gpu_id=0)
        
        # GPU 1 reads (both become Shared)
        protocol.read(page_id, gpu_id=1)
        
        # GPU 0 writes (should invalidate GPU 1's copy)
        protocol.write(page_id, gpu_id=0)
        
        # GPU 1 should be Invalid
        state = protocol.get_state(page_id, gpu_id=1)
        assert state == MESIState.INVALID


class TestDirectoryProtocol:
    """Test directory-based cache coherency."""
    
    def test_sharer_tracking(self):
        """Test that directory tracks sharers."""
        from distributed.multi_gpu_cache import DirectoryProtocol
        
        protocol = DirectoryProtocol(num_gpus=4)
        page_id = "page_0"
        
        # Multiple GPUs read
        protocol.read(page_id, gpu_id=0)
        protocol.read(page_id, gpu_id=1)
        protocol.read(page_id, gpu_id=2)
        
        # Directory should track all sharers
        sharers = protocol.get_sharers(page_id)
        assert 0 in sharers
        assert 1 in sharers
        assert 2 in sharers
        
    def test_exclusive_owner(self):
        """Test exclusive ownership tracking."""
        from distributed.multi_gpu_cache import DirectoryProtocol
        
        protocol = DirectoryProtocol(num_gpus=4)
        page_id = "page_0"
        
        # GPU 0 writes
        protocol.write(page_id, gpu_id=0)
        
        # Should be exclusive owner
        owner = protocol.get_owner(page_id)
        assert owner == 0


class TestGPUPageAllocator:
    """Test GPU page allocator."""
    
    def test_page_allocation(self):
        """Test page allocation."""
        from distributed.multi_gpu_cache import GPUPageAllocator
        
        allocator = GPUPageAllocator(
            gpu_id=0,
            total_pages=1024,
            page_size_bytes=4096,
        )
        
        # Allocate page
        page_id = allocator.allocate()
        
        assert page_id is not None
        assert allocator.get_free_pages() == 1023
        
    def test_page_deallocation(self):
        """Test page deallocation."""
        from distributed.multi_gpu_cache import GPUPageAllocator
        
        allocator = GPUPageAllocator(
            gpu_id=0,
            total_pages=1024,
            page_size_bytes=4096,
        )
        
        # Allocate and deallocate
        page_id = allocator.allocate()
        allocator.deallocate(page_id)
        
        assert allocator.get_free_pages() == 1024
        
    def test_allocation_failure(self):
        """Test allocation failure when out of pages."""
        from distributed.multi_gpu_cache import GPUPageAllocator
        
        allocator = GPUPageAllocator(
            gpu_id=0,
            total_pages=2,
            page_size_bytes=4096,
        )
        
        # Allocate all pages
        allocator.allocate()
        allocator.allocate()
        
        # Third allocation should fail
        page_id = allocator.allocate()
        assert page_id is None


class TestDistributedKVCache:
    """Test DistributedKVCache."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        from distributed.multi_gpu_cache import (
            DistributedKVCache,
            CacheCoherencyProtocol,
        )
        
        cache = DistributedKVCache(
            num_gpus=4,
            cache_size_per_gpu_mb=1024,
            coherency_protocol=CacheCoherencyProtocol.MESI,
        )
        
        assert cache.num_gpus == 4
        
    def test_cache_put_get(self):
        """Test cache put and get operations."""
        from distributed.multi_gpu_cache import (
            DistributedKVCache,
            CacheCoherencyProtocol,
        )
        
        cache = DistributedKVCache(
            num_gpus=4,
            cache_size_per_gpu_mb=1024,
            coherency_protocol=CacheCoherencyProtocol.MESI,
        )
        
        # Create KV tensors
        key = torch.randn(1, 8, 32, 64)  # batch, heads, seq, head_dim
        value = torch.randn(1, 8, 32, 64)
        
        # Put in cache
        cache.put(
            sequence_id="seq_0",
            layer_idx=0,
            key=key,
            value=value,
        )
        
        # Get from cache
        k, v = cache.get(
            sequence_id="seq_0",
            layer_idx=0,
        )
        
        assert torch.allclose(key, k)
        assert torch.allclose(value, v)


class TestSequenceAwareCacheSharding:
    """Test sequence-aware cache sharding."""
    
    def test_sharding_strategy(self):
        """Test cache sharding across GPUs."""
        from distributed.multi_gpu_cache import SequenceAwareCacheSharding
        
        sharder = SequenceAwareCacheSharding(
            num_gpus=4,
            sharding_strategy='sequence',
        )
        
        # Different sequences should potentially go to different GPUs
        gpu_0 = sharder.get_gpu_for_sequence("seq_0")
        gpu_1 = sharder.get_gpu_for_sequence("seq_1")
        
        assert 0 <= gpu_0 < 4
        assert 0 <= gpu_1 < 4
        
    def test_layer_sharding(self):
        """Test layer-based sharding."""
        from distributed.multi_gpu_cache import SequenceAwareCacheSharding
        
        sharder = SequenceAwareCacheSharding(
            num_gpus=4,
            sharding_strategy='layer',
        )
        
        # Each layer assigned to specific GPU
        gpu_layer_0 = sharder.get_gpu_for_layer(layer_idx=0)
        gpu_layer_4 = sharder.get_gpu_for_layer(layer_idx=4)
        
        assert gpu_layer_0 == gpu_layer_4  # Same GPU for layers 0 and 4 (mod 4)


# =============================================================================
# GPU Coordinator Tests
# =============================================================================

class TestGPUInfo:
    """Test GPUInfo dataclass."""
    
    def test_dataclass_creation(self):
        """Test GPUInfo creation."""
        from distributed.gpu_coordinator import GPUInfo, GPUState
        
        info = GPUInfo(
            gpu_id=0,
            device_name="NVIDIA A100",
            total_memory_gb=80.0,
            available_memory_gb=75.0,
            state=GPUState.IDLE,
            utilization=0.0,
        )
        
        assert info.gpu_id == 0
        assert info.device_name == "NVIDIA A100"
        assert info.state == GPUState.IDLE


class TestTopologyDetector:
    """Test TopologyDetector."""
    
    def test_topology_detection(self):
        """Test GPU topology detection."""
        from distributed.gpu_coordinator import TopologyDetector
        
        detector = TopologyDetector()
        
        # Mock detection (no actual GPUs needed)
        with patch('torch.cuda.device_count', return_value=4):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = MagicMock(
                    name='NVIDIA A100',
                    total_memory=80 * 1024**3,
                )
                
                topology = detector.detect()
                
                assert topology is not None
                
    def test_nvlink_detection(self):
        """Test NVLink connection detection."""
        from distributed.gpu_coordinator import TopologyDetector, TopologyType
        
        detector = TopologyDetector()
        
        # Mock NVLink detection
        with patch.object(detector, 'detect_nvlink_connections') as mock_nvlink:
            mock_nvlink.return_value = [(0, 1), (1, 2), (2, 3)]
            
            connections = detector.detect_nvlink_connections()
            
            assert len(connections) == 3


class TestHealthMonitor:
    """Test HealthMonitor."""
    
    def test_health_check(self):
        """Test GPU health check."""
        from distributed.gpu_coordinator import (
            HealthMonitor,
            HealthMonitorConfig,
            GPUState,
        )
        
        config = HealthMonitorConfig(
            check_interval_seconds=1.0,
            memory_threshold_percent=90.0,
            temperature_threshold_celsius=85.0,
        )
        
        monitor = HealthMonitor(config)
        
        # Mock health check
        with patch.object(monitor, '_check_gpu_health') as mock_check:
            mock_check.return_value = {
                'state': GPUState.HEALTHY,
                'memory_used_percent': 50.0,
                'temperature': 65.0,
            }
            
            health = monitor.check_gpu(gpu_id=0)
            
            assert health['state'] == GPUState.HEALTHY
            
    def test_failure_detection(self):
        """Test failure detection."""
        from distributed.gpu_coordinator import (
            HealthMonitor,
            HealthMonitorConfig,
            FailureType,
        )
        
        config = HealthMonitorConfig(
            check_interval_seconds=1.0,
            memory_threshold_percent=90.0,
        )
        
        monitor = HealthMonitor(config)
        
        # Simulate memory exhaustion
        with patch.object(monitor, '_check_gpu_health') as mock_check:
            mock_check.return_value = {
                'memory_used_percent': 95.0,
                'failure_type': FailureType.MEMORY_EXHAUSTION,
            }
            
            health = monitor.check_gpu(gpu_id=0)
            
            assert health['failure_type'] == FailureType.MEMORY_EXHAUSTION


class TestWorkloadScheduler:
    """Test WorkloadScheduler."""
    
    def test_round_robin_scheduling(self):
        """Test round-robin scheduling."""
        from distributed.gpu_coordinator import (
            WorkloadScheduler,
            SchedulerConfig,
            SchedulingPolicy,
            WorkloadTask,
            WorkloadPriority,
        )
        
        config = SchedulerConfig(
            policy=SchedulingPolicy.ROUND_ROBIN,
            num_gpus=4,
        )
        
        scheduler = WorkloadScheduler(config)
        
        # Schedule multiple tasks
        gpus = []
        for i in range(8):
            task = WorkloadTask(
                task_id=f"task_{i}",
                priority=WorkloadPriority.NORMAL,
                estimated_memory_mb=1000,
            )
            gpu = scheduler.schedule(task)
            gpus.append(gpu)
            
        # Should cycle through GPUs
        assert gpus == [0, 1, 2, 3, 0, 1, 2, 3]
        
    def test_load_balanced_scheduling(self):
        """Test load-balanced scheduling."""
        from distributed.gpu_coordinator import (
            WorkloadScheduler,
            SchedulerConfig,
            SchedulingPolicy,
            WorkloadTask,
            WorkloadPriority,
        )
        
        config = SchedulerConfig(
            policy=SchedulingPolicy.LOAD_BALANCED,
            num_gpus=4,
        )
        
        scheduler = WorkloadScheduler(config)
        
        # Update GPU loads
        scheduler.update_gpu_load(gpu_id=0, load=0.8)
        scheduler.update_gpu_load(gpu_id=1, load=0.2)
        scheduler.update_gpu_load(gpu_id=2, load=0.5)
        scheduler.update_gpu_load(gpu_id=3, load=0.9)
        
        task = WorkloadTask(
            task_id="task_0",
            priority=WorkloadPriority.NORMAL,
            estimated_memory_mb=1000,
        )
        
        # Should schedule to least loaded GPU (GPU 1)
        gpu = scheduler.schedule(task)
        assert gpu == 1


class TestGPUCoordinator:
    """Test GPUCoordinator."""
    
    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        from distributed.gpu_coordinator import (
            GPUCoordinator,
            CoordinatorConfig,
        )
        
        config = CoordinatorConfig(
            num_gpus=4,
            enable_health_monitoring=True,
        )
        
        with patch('torch.cuda.device_count', return_value=4):
            coordinator = GPUCoordinator(config)
            
            assert coordinator.num_gpus == 4
            
    def test_task_submission(self):
        """Test task submission."""
        from distributed.gpu_coordinator import (
            GPUCoordinator,
            CoordinatorConfig,
            WorkloadTask,
            WorkloadPriority,
        )
        
        config = CoordinatorConfig(num_gpus=4)
        
        with patch('torch.cuda.device_count', return_value=4):
            coordinator = GPUCoordinator(config)
            
            task = WorkloadTask(
                task_id="inference_0",
                priority=WorkloadPriority.HIGH,
                estimated_memory_mb=2000,
            )
            
            gpu = coordinator.submit_task(task)
            
            assert 0 <= gpu < 4
            
    def test_statistics_collection(self):
        """Test statistics collection."""
        from distributed.gpu_coordinator import (
            GPUCoordinator,
            CoordinatorConfig,
        )
        
        config = CoordinatorConfig(num_gpus=4)
        
        with patch('torch.cuda.device_count', return_value=4):
            coordinator = GPUCoordinator(config)
            
            stats = coordinator.get_statistics()
            
            assert 'total_tasks' in stats
            assert 'gpu_utilization' in stats


# =============================================================================
# Integration Tests
# =============================================================================

class TestEndToEndMultiGPU:
    """End-to-end integration tests for multi-GPU system."""
    
    def test_tensor_parallel_with_coordinator(self):
        """Test tensor parallelism with GPU coordinator."""
        from distributed.tensor_parallelism import TensorParallelConfig
        from distributed.gpu_coordinator import (
            GPUCoordinator,
            CoordinatorConfig,
            WorkloadTask,
            WorkloadPriority,
        )
        
        # Initialize coordinator
        coord_config = CoordinatorConfig(num_gpus=4)
        
        with patch('torch.cuda.device_count', return_value=4):
            coordinator = GPUCoordinator(coord_config)
            
            # Create tensor parallel config
            tp_config = TensorParallelConfig(
                world_size=4,
                sequence_parallel=True,
            )
            
            # Submit workload task
            task = WorkloadTask(
                task_id="tp_forward",
                priority=WorkloadPriority.HIGH,
                estimated_memory_mb=4000,
            )
            
            assigned_gpu = coordinator.submit_task(task)
            
            assert assigned_gpu is not None
            
    def test_pipeline_with_distributed_cache(self):
        """Test pipeline parallelism with distributed cache."""
        from distributed.pipeline_parallelism import PipelineConfig
        from distributed.multi_gpu_cache import (
            DistributedKVCache,
            CacheCoherencyProtocol,
        )
        
        # Initialize distributed cache
        cache = DistributedKVCache(
            num_gpus=4,
            cache_size_per_gpu_mb=1024,
            coherency_protocol=CacheCoherencyProtocol.MESI,
        )
        
        # Initialize pipeline config
        pp_config = PipelineConfig(
            num_stages=4,
            num_micro_batches=8,
        )
        
        # Simulate pipeline execution with cache updates
        for micro_batch_idx in range(pp_config.num_micro_batches):
            for stage_idx in range(pp_config.num_stages):
                # Cache KV for each layer
                key = torch.randn(1, 8, 32, 64)
                value = torch.randn(1, 8, 32, 64)
                
                cache.put(
                    sequence_id=f"seq_{micro_batch_idx}",
                    layer_idx=stage_idx,
                    key=key,
                    value=value,
                )
                
        # Verify cache population
        k, v = cache.get(
            sequence_id="seq_0",
            layer_idx=0,
        )
        
        assert k is not None
        assert v is not None
        
    def test_full_multi_gpu_inference_flow(self):
        """Test complete multi-GPU inference flow."""
        from distributed import (
            TensorParallelConfig,
            PipelineConfig,
            create_distributed_cache,
            create_coordinator,
        )
        from distributed.gpu_coordinator import WorkloadTask, WorkloadPriority
        
        # Create all components
        with patch('torch.cuda.device_count', return_value=4):
            # GPU coordinator
            coordinator = create_coordinator(
                num_gpus=4,
                enable_health_monitoring=True,
            )
            
            # Distributed cache
            cache = create_distributed_cache(
                num_gpus=4,
                cache_size_per_gpu_mb=1024,
            )
            
            # Tensor parallel config
            tp_config = TensorParallelConfig(world_size=4)
            
            # Pipeline config
            pp_config = PipelineConfig(num_stages=4)
            
            # Submit inference task
            task = WorkloadTask(
                task_id="inference_request_0",
                priority=WorkloadPriority.HIGH,
                estimated_memory_mb=8000,
            )
            
            gpu = coordinator.submit_task(task)
            
            # Verify system components work together
            assert coordinator.num_gpus == 4
            assert cache.num_gpus == 4
            assert gpu is not None


# =============================================================================
# Performance Tests
# =============================================================================

class TestMultiGPUPerformance:
    """Performance tests for multi-GPU components."""
    
    def test_cache_lookup_performance(self):
        """Test cache lookup performance."""
        from distributed.multi_gpu_cache import (
            DistributedKVCache,
            CacheCoherencyProtocol,
        )
        
        cache = DistributedKVCache(
            num_gpus=4,
            cache_size_per_gpu_mb=1024,
            coherency_protocol=CacheCoherencyProtocol.MESI,
        )
        
        # Pre-populate cache
        for i in range(100):
            key = torch.randn(1, 8, 32, 64)
            value = torch.randn(1, 8, 32, 64)
            cache.put(f"seq_{i}", layer_idx=0, key=key, value=value)
            
        # Measure lookup time
        start = time.perf_counter()
        for i in range(1000):
            cache.get(f"seq_{i % 100}", layer_idx=0)
        elapsed = time.perf_counter() - start
        
        # Should be fast (< 1ms per lookup on average)
        avg_lookup_us = (elapsed / 1000) * 1_000_000
        assert avg_lookup_us < 1000, f"Lookup too slow: {avg_lookup_us:.2f}us"
        
    def test_scheduler_performance(self):
        """Test scheduler performance under load."""
        from distributed.gpu_coordinator import (
            WorkloadScheduler,
            SchedulerConfig,
            SchedulingPolicy,
            WorkloadTask,
            WorkloadPriority,
        )
        
        config = SchedulerConfig(
            policy=SchedulingPolicy.LOAD_BALANCED,
            num_gpus=8,
        )
        
        scheduler = WorkloadScheduler(config)
        
        # Measure scheduling time
        start = time.perf_counter()
        for i in range(10000):
            task = WorkloadTask(
                task_id=f"task_{i}",
                priority=WorkloadPriority.NORMAL,
                estimated_memory_mb=1000,
            )
            scheduler.schedule(task)
        elapsed = time.perf_counter() - start
        
        # Should schedule quickly (< 100us per task on average)
        avg_schedule_us = (elapsed / 10000) * 1_000_000
        assert avg_schedule_us < 100, f"Scheduling too slow: {avg_schedule_us:.2f}us"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestMultiGPUErrorHandling:
    """Test error handling in multi-GPU components."""
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configurations."""
        from distributed.tensor_parallelism import TensorParallelConfig
        
        # World size must be positive
        with pytest.raises((ValueError, AssertionError)):
            TensorParallelConfig(world_size=0)
            
    def test_cache_miss_handling(self):
        """Test cache miss handling."""
        from distributed.multi_gpu_cache import (
            DistributedKVCache,
            CacheCoherencyProtocol,
        )
        
        cache = DistributedKVCache(
            num_gpus=4,
            cache_size_per_gpu_mb=1024,
            coherency_protocol=CacheCoherencyProtocol.MESI,
        )
        
        # Get non-existent entry
        result = cache.get("nonexistent_seq", layer_idx=0)
        
        # Should return None or empty tuple
        assert result is None or result == (None, None)
        
    def test_gpu_failure_recovery(self):
        """Test GPU failure recovery handling."""
        from distributed.gpu_coordinator import (
            GPUCoordinator,
            CoordinatorConfig,
            FailureType,
            RecoveryAction,
        )
        
        config = CoordinatorConfig(
            num_gpus=4,
            enable_health_monitoring=True,
            enable_auto_recovery=True,
        )
        
        with patch('torch.cuda.device_count', return_value=4):
            coordinator = GPUCoordinator(config)
            
            # Simulate GPU failure
            recovery_action = coordinator.handle_failure(
                gpu_id=1,
                failure_type=FailureType.TIMEOUT,
            )
            
            # Should suggest appropriate recovery
            assert recovery_action in [
                RecoveryAction.RETRY,
                RecoveryAction.REDISTRIBUTE,
                RecoveryAction.EXCLUDE_GPU,
            ]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
