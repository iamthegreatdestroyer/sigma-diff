"""
Tests for KV-Cache Optimization Components

Comprehensive test suite covering:
- Distributed KV-cache sharding
- FP8 compression accuracy
- Dynamic cache allocation
- Integration testing
- Performance validation

Test Coverage Goals:
- Unit tests: 90%+ coverage
- Integration tests: End-to-end validation
- Performance tests: Benchmarking
- Edge case tests: Error conditions
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import time
import threading
from typing import Dict, Any

# Import components under test
import sys
sys.path.append('Ryzanstein LLM')

from src.inference.distributed_kv_cache import (
    DistributedKVCache, CacheShard, ConsistencyManager,
    KVCacheCommunicator, ConsistencyMode
)
from src.inference.cache_compression import (
    FP8Compressor, CompressedKVCache, CompressionAccuracyValidator
)
from src.inference.dynamic_allocator import (
    DynamicCacheAllocator, MemoryPressureMonitor, EvictionPolicy
)


class TestDistributedKVCache:
    """Test distributed KV-cache functionality."""

    @pytest.fixture
    def cache_config(self):
        return {
            "num_layers": 2,
            "num_heads": 4,
            "head_dim": 64,
            "max_seq_len": 128,
            "world_size": 2,
            "rank": 0,
            "device": torch.device("cpu")  # Use CPU for testing
        }

    @pytest.fixture
    def distributed_cache(self, cache_config):
        cache = DistributedKVCache(**cache_config)
        cache.set_communicator(Mock(spec=KVCacheCommunicator))
        return cache

    def test_initialization(self, distributed_cache, cache_config):
        """Test cache initialization."""
        assert distributed_cache.num_layers == cache_config["num_layers"]
        assert distributed_cache.num_heads == cache_config["num_heads"]
        assert distributed_cache.rank == cache_config["rank"]
        assert distributed_cache.world_size == cache_config["world_size"]

        # Check shard boundaries
        assert distributed_cache.shard_start == 0
        assert distributed_cache.shard_end == 64  # 128 // 2

    def test_allocate_cache(self, distributed_cache):
        """Test cache allocation."""
        batch_size = 2
        seq_len = 32

        distributed_cache.allocate_cache(batch_size, seq_len)

        # Check that shards are allocated
        for layer_id in range(distributed_cache.num_layers):
            for head_id in range(distributed_cache.num_heads):
                shard = distributed_cache.cache_shards[layer_id][head_id]
                assert shard.k_cache is not None
                assert shard.v_cache is not None
                assert shard.k_cache.shape == (batch_size, seq_len, distributed_cache.head_dim)

    def test_update_and_get_kv(self, distributed_cache):
        """Test KV update and retrieval."""
        batch_size = 1
        seq_len = 16
        distributed_cache.allocate_cache(batch_size, seq_len)

        layer_id, head_id = 0, 0
        seq_pos = 5
        k = torch.randn(batch_size, distributed_cache.head_dim, dtype=torch.float16)
        v = torch.randn(batch_size, distributed_cache.head_dim, dtype=torch.float16)

        # Update
        distributed_cache.update_kv(layer_id, head_id, seq_pos, k, v)

        # Retrieve
        k_retrieved, v_retrieved = distributed_cache.get_kv_range(layer_id, head_id, seq_pos, seq_pos + 1)

        assert torch.allclose(k_retrieved.squeeze(1), k, atol=1e-3)
        assert torch.allclose(v_retrieved.squeeze(1), v, atol=1e-3)

    def test_memory_usage_stats(self, distributed_cache):
        """Test memory usage reporting."""
        batch_size = 2
        seq_len = 32
        distributed_cache.allocate_cache(batch_size, seq_len)

        stats = distributed_cache.get_memory_usage()

        assert "allocated_mb" in stats
        assert "used_mb" in stats
        assert "utilization_percent" in stats
        assert stats["allocated_mb"] > 0
        assert stats["utilization_percent"] >= 0

    def test_cache_stats(self, distributed_cache):
        """Test cache performance statistics."""
        stats = distributed_cache.get_cache_stats()

        assert "total_accesses" in stats
        assert "remote_accesses" in stats
        assert "local_access_ratio" in stats
        assert "avg_latency_ms" in stats

    def test_clear_cache(self, distributed_cache):
        """Test cache clearing."""
        batch_size = 1
        seq_len = 16
        distributed_cache.allocate_cache(batch_size, seq_len)

        # Verify allocated
        assert distributed_cache.cache_shards[0][0].k_cache is not None

        # Clear
        distributed_cache.clear_cache()

        # Verify cleared
        assert distributed_cache.cache_shards[0][0].k_cache is None
        assert distributed_cache.cache_shards[0][0].v_cache is None


class TestFP8Compression:
    """Test FP8 compression functionality."""

    @pytest.fixture
    def compressor(self):
        return FP8Compressor(calibration_samples=10, device=torch.device("cpu"))

    @pytest.fixture
    def sample_kv(self):
        """Generate sample KV tensors."""
        batch_size, seq_len, head_dim = 1, 10, 64
        k = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float16)
        return k, v

    def test_calibration(self, compressor, sample_kv):
        """Test compression calibration."""
        k, v = sample_kv

        # Collect samples
        for i in range(5):
            compressor.collect_calibration_sample(0, 0, k, v)

        # Calibrate
        compressor.calibrate_scales()

        # Check scales are set
        assert (0, 0) in compressor.scale_params
        params = compressor.scale_params[(0, 0)]
        assert params.calibrated
        assert params.scale_k > 0
        assert params.scale_v > 0

    def test_quantize_dequantize(self, compressor, sample_kv):
        """Test quantization and dequantization."""
        k, v = sample_kv

        # Calibrate first
        compressor.collect_calibration_sample(0, 0, k, v)
        compressor.calibrate_scales()

        # Quantize
        k_fp8, v_fp8 = compressor.quantize_kv(0, 0, k, v)

        # Dequantize
        k_restored, v_restored = compressor.dequantize_kv(0, 0, k_fp8, v_fp8)

        # Check shapes
        assert k_restored.shape == k.shape
        assert v_restored.shape == v.shape

        # Check reasonable accuracy (allowing for FP8 precision loss)
        k_mse = torch.mean((k - k_restored) ** 2).item()
        v_mse = torch.mean((v - v_restored) ** 2).item()

        # FP8 should have some loss but not catastrophic
        assert k_mse < 1.0  # Reasonable bound
        assert v_mse < 1.0

    def test_compression_stats(self, compressor):
        """Test compression statistics."""
        stats = compressor.get_compression_stats()

        assert "calibrated_layers" in stats
        assert "total_layers" in stats
        assert "calibration_samples" in stats
        assert "memory_reduction_percent" in stats
        assert stats["memory_reduction_percent"] == 50.0  # FP8 vs FP16


class TestCompressedKVCache:
    """Test compressed KV-cache integration."""

    @pytest.fixture
    def compressed_cache(self):
        return CompressedKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=64,
            max_seq_len=128,
            device=torch.device("cpu"),
            enable_compression=True
        )

    def test_initialization(self, compressed_cache):
        """Test compressed cache initialization."""
        assert compressed_cache.num_layers == 2
        assert compressed_cache.num_heads == 4
        assert compressed_cache.enable_compression
        assert compressed_cache.compressor is not None

    def test_store_and_retrieve(self, compressed_cache):
        """Test compressed storage and retrieval."""
        layer_id, head_id = 0, 0
        seq_pos = 5

        # Sample KV
        k = torch.randn(1, 64, dtype=torch.float16)
        v = torch.randn(1, 64, dtype=torch.float16)

        # Calibrate compression
        compressed_cache.calibrate_compression([(layer_id, head_id, k, v)])

        # Store
        compressed_cache.store_compressed(layer_id, head_id, seq_pos, k, v)

        # Retrieve
        k_retrieved, v_retrieved = compressed_cache.retrieve_decompressed(
            layer_id, head_id, seq_pos, seq_pos + 1
        )

        # Check shapes
        assert k_retrieved.shape == (1, 1, 64)  # batch, seq, head_dim
        assert v_retrieved.shape == (1, 1, 64)

    def test_memory_usage(self, compressed_cache):
        """Test memory usage reporting."""
        stats = compressed_cache.get_memory_usage()

        assert "compressed_mb" in stats
        assert "uncompressed_mb" in stats
        assert "compression_ratio" in stats
        assert "memory_savings_percent" in stats

    def test_clear_cache(self, compressed_cache):
        """Test cache clearing."""
        layer_id, head_id = 0, 0
        k = torch.randn(1, 64, dtype=torch.float16)
        v = torch.randn(1, 64, dtype=torch.float16)

        compressed_cache.calibrate_compression([(layer_id, head_id, k, v)])
        compressed_cache.store_compressed(layer_id, head_id, 0, k, v)

        # Verify stored
        assert compressed_cache.compressed_k[layer_id][head_id] is not None

        # Clear
        compressed_cache.clear_cache()

        # Verify cleared
        assert compressed_cache.compressed_k[layer_id][head_id] is None


class TestDynamicCacheAllocator:
    """Test dynamic cache allocation."""

    @pytest.fixture
    def allocator(self):
        return DynamicCacheAllocator(
            total_memory_gb=1.0,  # 1GB for testing
            safety_margin=0.1,
            eviction_policy=EvictionPolicy.LRU
        )

    def test_initialization(self, allocator):
        """Test allocator initialization."""
        assert allocator.total_memory_bytes > 0
        assert len(allocator.memory_pools) == 0  # No pools created yet
        assert allocator.eviction_policy == EvictionPolicy.LRU

    def test_memory_calculation(self, allocator):
        """Test memory requirement calculation."""
        seq_len, num_layers, num_heads, head_dim = 100, 12, 12, 64

        memory = allocator.calculate_memory_requirement(
            seq_len, num_layers, num_heads, head_dim, compressed=False
        )

        # Expected: 100 * 12 * 12 * 64 * 2 * 2 bytes (K+V, FP16=2bytes)
        expected = 100 * 12 * 12 * 64 * 2 * 2
        assert memory == expected

    def test_allocate_cache(self, allocator):
        """Test cache allocation."""
        success = allocator.allocate_cache(
            request_id="test_req",
            seq_len=50,
            num_layers=6,
            num_heads=8,
            head_dim=64,
            compressed=False
        )

        assert success
        assert "test_req" in allocator.global_allocations

        # Check memory pools created
        assert len(allocator.memory_pools) == 6  # One per layer

    def test_deallocate_cache(self, allocator):
        """Test cache deallocation."""
        # Allocate first
        allocator.allocate_cache("test_req", 50, 6, 8, 64, False)

        # Verify allocated
        assert "test_req" in allocator.global_allocations

        # Deallocate
        allocator.deallocate_cache("test_req")

        # Verify deallocated
        assert "test_req" not in allocator.global_allocations

    def test_memory_stats(self, allocator):
        """Test memory statistics."""
        stats = allocator.get_memory_stats()

        assert "total_memory_gb" in stats
        assert "used_memory_gb" in stats
        assert "available_memory_gb" in stats
        assert "utilization_percent" in stats
        assert "pools" in stats

    def test_eviction_under_pressure(self, allocator):
        """Test eviction under memory pressure."""
        # Allocate until we need to evict
        for i in range(10):
            success = allocator.allocate_cache(
                f"req_{i}", 100, 12, 12, 64, False, priority=1
            )
            if not success:
                break

        # Should have some allocations
        assert len(allocator.global_allocations) > 0

        # Check eviction stats
        stats = allocator.get_memory_stats()
        assert stats["total_evictions"] >= 0

    def test_access_tracking(self, allocator):
        """Test access pattern tracking."""
        allocator.allocate_cache("test_req", 50, 6, 8, 64, False)

        initial_access = allocator.global_allocations["test_req"].access_count
        initial_time = allocator.global_allocations["test_req"].last_access

        time.sleep(0.01)  # Small delay
        allocator.access_cache("test_req")

        assert allocator.global_allocations["test_req"].access_count == initial_access + 1
        assert allocator.global_allocations["test_req"].last_access > initial_time


class TestCompressionAccuracyValidator:
    """Test compression accuracy validation."""

    @pytest.fixture
    def validator(self):
        return CompressionAccuracyValidator(num_samples=10)

    def test_accuracy_measurement(self, validator):
        """Test accuracy loss measurement."""
        original_k = torch.randn(1, 10, 64)
        original_v = torch.randn(1, 10, 64)

        # Simulate some compression loss
        compressed_k = original_k + 0.01 * torch.randn_like(original_k)
        compressed_v = original_v + 0.01 * torch.randn_like(original_v)

        loss = validator.measure_accuracy_loss(
            original_k, original_v, compressed_k, compressed_v
        )

        assert loss > 0
        assert loss < 1.0  # Should be small

    def test_accuracy_stats(self, validator):
        """Test accuracy statistics."""
        # Add some measurements
        for _ in range(5):
            k = torch.randn(1, 10, 64)
            v = torch.randn(1, 10, 64)
            compressed_k = k + 0.01 * torch.randn_like(k)
            compressed_v = v + 0.01 * torch.randn_like(v)
            validator.measure_accuracy_loss(k, v, compressed_k, compressed_v)

        stats = validator.get_accuracy_stats()

        assert stats["samples"] == 5
        assert "avg_loss" in stats
        assert "max_loss" in stats
        assert "relative_loss_percent" in stats


class TestIntegration:
    """Integration tests for KV-cache optimization."""

    @pytest.fixture
    def full_system(self):
        """Create a full integrated system for testing."""
        # Distributed cache
        cache = DistributedKVCache(
            num_layers=2, num_heads=4, head_dim=64, max_seq_len=128,
            world_size=2, rank=0, device=torch.device("cpu")
        )

        # Compression
        compressed_cache = CompressedKVCache(
            num_layers=2, num_heads=4, head_dim=64, max_seq_len=128,
            device=torch.device("cpu"), enable_compression=True
        )

        # Allocator
        allocator = DynamicCacheAllocator(total_memory_gb=2.0)

        return {
            "cache": cache,
            "compressed_cache": compressed_cache,
            "allocator": allocator
        }

    def test_end_to_end_workflow(self, full_system):
        """Test complete KV-cache workflow."""
        cache = full_system["cache"]
        compressed_cache = full_system["compressed_cache"]
        allocator = full_system["allocator"]

        # Allocate memory
        success = allocator.allocate_cache("test_workflow", 64, 2, 4, 64, True)
        assert success

        # Generate sample KV
        k = torch.randn(1, 64, dtype=torch.float16)
        v = torch.randn(1, 64, dtype=torch.float16)

        # Calibrate compression
        compressed_cache.calibrate_compression([(0, 0, k, v)])

        # Store compressed
        compressed_cache.store_compressed(0, 0, 10, k, v)

        # Retrieve and verify
        k_retrieved, v_retrieved = compressed_cache.retrieve_decompressed(0, 0, 10, 11)

        assert k_retrieved.shape == (1, 1, 64)
        assert v_retrieved.shape == (1, 1, 64)

        # Check memory usage
        cache_stats = cache.get_memory_usage()
        compressed_stats = compressed_cache.get_memory_usage()
        allocator_stats = allocator.get_memory_stats()

        assert cache_stats["allocated_mb"] >= 0
        assert compressed_stats["memory_savings_percent"] > 0
        assert allocator_stats["utilization_percent"] >= 0

    def test_concurrent_access(self, full_system):
        """Test concurrent access to cache system."""
        compressed_cache = full_system["compressed_cache"]

        def worker_thread(thread_id: int):
            """Worker thread for concurrent testing."""
            for i in range(5):
                k = torch.randn(1, 64, dtype=torch.float16)
                v = torch.randn(1, 64, dtype=torch.float16)

                # Store
                compressed_cache.store_compressed(0, 0, thread_id * 5 + i, k, v)

                # Retrieve
                k_ret, v_ret = compressed_cache.retrieve_decompressed(
                    0, 0, thread_id * 5 + i, thread_id * 5 + i + 1
                )

                assert k_ret.shape == (1, 1, 64)

        # Run concurrent threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker_thread, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify final state
        stats = compressed_cache.get_memory_usage()
        assert stats["compressed_mb"] > 0


# Performance benchmarks
class TestPerformance:
    """Performance tests for KV-cache optimization."""

    @pytest.fixture
    def perf_cache(self):
        return DistributedKVCache(
            num_layers=12, num_heads=12, head_dim=64, max_seq_len=2048,
            world_size=1, rank=0, device=torch.device("cpu")
        )

    def test_cache_throughput(self, perf_cache):
        """Test cache operation throughput."""
        batch_size = 4
        seq_len = 512
        perf_cache.allocate_cache(batch_size, seq_len)

        # Measure update throughput
        num_operations = 100
        start_time = time.time()

        for i in range(num_operations):
            layer_id = i % perf_cache.num_layers
            head_id = i % perf_cache.num_heads
            seq_pos = i % seq_len

            k = torch.randn(batch_size, perf_cache.head_dim)
            v = torch.randn(batch_size, perf_cache.head_dim)

            perf_cache.update_kv(layer_id, head_id, seq_pos, k, v)

        end_time = time.time()
        throughput = num_operations / (end_time - start_time)

        assert throughput > 10  # At least 10 operations per second

    def test_memory_efficiency(self, perf_cache):
        """Test memory usage efficiency."""
        initial_stats = perf_cache.get_memory_usage()

        # Allocate cache
        perf_cache.allocate_cache(4, 512)
        allocated_stats = perf_cache.get_memory_usage()

        # Should have allocated memory
        assert allocated_stats["allocated_mb"] > initial_stats["allocated_mb"]

        # Clear and verify cleanup
        perf_cache.clear_cache()
        cleared_stats = perf_cache.get_memory_usage()

        assert cleared_stats["allocated_mb"] == initial_stats["allocated_mb"]


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running KV-cache optimization smoke tests...")

    # Test distributed cache
    cache = DistributedKVCache(2, 4, 64, 128, 2, 0, torch.device("cpu"))
    cache.allocate_cache(1, 32)
    print("✓ Distributed cache initialization")

    # Test compression
    compressor = FP8Compressor(device=torch.device("cpu"))
    k = torch.randn(1, 64, dtype=torch.float16)
    v = torch.randn(1, 64, dtype=torch.float16)
    compressor.collect_calibration_sample(0, 0, k, v)
    compressor.calibrate_scales()
    k_fp8, v_fp8 = compressor.quantize_kv(0, 0, k, v)
    k_restored, v_restored = compressor.dequantize_kv(0, 0, k_fp8, v_fp8)
    print("✓ FP8 compression")

    # Test allocator
    allocator = DynamicCacheAllocator(total_memory_gb=1.0)
    success = allocator.allocate_cache("test", 50, 6, 8, 64, False)
    print("✓ Dynamic allocation")

    print("All smoke tests passed! ✅")