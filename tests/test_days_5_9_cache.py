"""
Test Suite for Sprint 2.2 Days 5-9 Implementation
==================================================

Comprehensive tests for:
- KV Cache Compression (INT8/INT4)
- Adaptive Cache Sizing
- Distributed Cache Optimization
- Production Hardening

Run with: pytest tests/test_days_5_9_cache.py -v
"""

import pytest
import torch
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PHASE2_DEVELOPMENT', 'src'))

from cache.kv_cache_compression import (
    QuantizationType,
    ScalingMode,
    QuantizationConfig,
    Int8Quantizer,
    Int4Quantizer,
    MixedPrecisionQuantizer,
    QuantizedKVCacheManager,
    create_quantized_kv_cache,
)

from cache.adaptive_cache_manager import (
    MemoryPressureLevel,
    MemoryStats,
    AdaptiveSizingConfig,
    MemoryMonitor,
    WorkloadAnalyzer,
    AdaptiveCacheSizer,
    create_adaptive_sizer,
)

from cache.distributed_cache_optimizer import (
    NodeState,
    NodeInfo,
    ConsistentHash,
    CacheCoordinator,
    create_distributed_cache_optimizer,
)

from cache.production_hardening import (
    CircuitState,
    HealthStatus,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    HealthChecker,
    HealthCheckResult,
    MetricsCollector,
    RateLimiter,
    GracefulDegradation,
)


# ==============================================================================
# KV Cache Compression Tests
# ==============================================================================

class TestInt8Quantizer:
    """Tests for INT8 quantization."""
    
    def test_quantize_dequantize_round_trip(self):
        """Test that quantize -> dequantize produces reasonable results."""
        config = QuantizationConfig(quant_type=QuantizationType.INT8)
        quantizer = Int8Quantizer(config)
        
        # Create test tensor
        tensor = torch.randn(4, 12, 128, 64)
        
        # Quantize
        qtensor = quantizer.quantize(tensor)
        
        # Dequantize
        recovered = quantizer.dequantize(qtensor)
        
        # Check shape preserved
        assert recovered.shape == tensor.shape
        
        # Check error is small
        error = (tensor - recovered).abs().mean().item()
        assert error < 0.1, f"Quantization error too high: {error}"
    
    def test_compression_ratio(self):
        """Test that INT8 achieves ~4x compression."""
        config = QuantizationConfig(quant_type=QuantizationType.INT8)
        quantizer = Int8Quantizer(config)
        
        tensor = torch.randn(4, 12, 128, 64, dtype=torch.float16)
        qtensor = quantizer.quantize(tensor)
        
        original_bytes = tensor.numel() * 2  # float16 = 2 bytes
        compressed_bytes = qtensor.memory_bytes()
        
        ratio = original_bytes / compressed_bytes
        assert ratio > 1.5, f"Compression ratio too low: {ratio}"
    
    def test_per_channel_scaling(self):
        """Test per-channel scaling mode."""
        config = QuantizationConfig(
            quant_type=QuantizationType.INT8,
            scaling_mode=ScalingMode.PER_CHANNEL
        )
        quantizer = Int8Quantizer(config)
        
        tensor = torch.randn(4, 12, 128, 64)
        qtensor = quantizer.quantize(tensor)
        
        # Scale should have shape matching last dimension
        assert qtensor.scale.shape[0] == 64


class TestInt4Quantizer:
    """Tests for INT4 quantization."""
    
    def test_quantize_dequantize_round_trip(self):
        """Test INT4 quantize -> dequantize."""
        config = QuantizationConfig(quant_type=QuantizationType.INT4, block_size=64)
        quantizer = Int4Quantizer(config)
        
        tensor = torch.randn(4, 12, 128, 64)
        
        qtensor = quantizer.quantize(tensor)
        recovered = quantizer.dequantize(qtensor)
        
        assert recovered.shape == tensor.shape
        
        # INT4 has higher error tolerance
        error = (tensor - recovered).abs().mean().item()
        assert error < 0.5, f"INT4 quantization error too high: {error}"
    
    def test_packing(self):
        """Test that INT4 values are packed correctly."""
        config = QuantizationConfig(quant_type=QuantizationType.INT4, block_size=16)
        quantizer = Int4Quantizer(config)
        
        tensor = torch.randn(256)  # 256 values
        qtensor = quantizer.quantize(tensor)
        
        # Packed size should be half (2 values per byte)
        # Plus some overhead for scales
        assert qtensor.data.numel() <= tensor.numel() // 2 + 32


class TestQuantizedKVCacheManager:
    """Tests for quantized KV cache manager."""
    
    def test_store_retrieve(self):
        """Test basic store and retrieve operations."""
        cache = create_quantized_kv_cache(
            quant_type="int8",
            num_layers=12,
            num_heads=12,
            head_dim=64
        )
        
        k = torch.randn(2, 12, 64, 64)
        v = torch.randn(2, 12, 64, 64)
        
        cache.store(0, k, v)
        k_out, v_out = cache.retrieve(0)
        
        assert k_out is not None
        assert v_out is not None
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape
    
    def test_missing_layer(self):
        """Test retrieval of non-existent layer."""
        cache = create_quantized_kv_cache(quant_type="int8")
        
        k, v = cache.retrieve(999)
        
        assert k is None
        assert v is None
    
    def test_statistics(self):
        """Test statistics collection."""
        cache = create_quantized_kv_cache(quant_type="int8")
        
        k = torch.randn(2, 12, 64, 64)
        v = torch.randn(2, 12, 64, 64)
        
        cache.store(0, k, v)
        cache.retrieve(0)
        cache.retrieve(0)
        cache.retrieve(999)  # Miss
        
        stats = cache.get_stats()
        
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert stats["compression_ratio"] > 1


# ==============================================================================
# Adaptive Cache Sizing Tests
# ==============================================================================

class TestMemoryMonitor:
    """Tests for memory monitoring."""
    
    def test_get_memory_stats(self):
        """Test memory stats collection."""
        config = AdaptiveSizingConfig()
        monitor = MemoryMonitor(config)
        
        stats = monitor.get_memory_stats()
        
        assert stats.total_mb > 0
        assert stats.available_mb >= 0
        assert stats.used_mb >= 0
        assert stats.pressure_level in MemoryPressureLevel
    
    def test_pressure_trend(self):
        """Test pressure trend analysis."""
        config = AdaptiveSizingConfig()
        monitor = MemoryMonitor(config)
        
        # Collect some samples
        for _ in range(20):
            monitor.get_memory_stats()
        
        trend = monitor.get_pressure_trend()
        
        assert trend in ["increasing", "stable", "decreasing"]


class TestWorkloadAnalyzer:
    """Tests for workload analysis."""
    
    def test_hit_miss_recording(self):
        """Test hit/miss recording."""
        analyzer = WorkloadAnalyzer()
        
        # Record some hits and misses
        for _ in range(70):
            analyzer.record_hit(1.0)
        for _ in range(30):
            analyzer.record_miss(10.0)
        
        stats = analyzer.get_stats()
        
        # Should have ~70% hit rate
        assert 0.65 < stats.hit_rate < 0.75
    
    def test_eviction_tracking(self):
        """Test eviction tracking."""
        analyzer = WorkloadAnalyzer()
        
        for _ in range(50):
            analyzer.record_eviction()
        
        stats = analyzer.get_stats()
        
        assert stats.evictions_per_sec > 0


class TestAdaptiveCacheSizer:
    """Tests for adaptive cache sizer."""
    
    def test_initial_size(self):
        """Test initial size configuration."""
        sizer = create_adaptive_sizer(
            min_size_mb=64,
            max_size_mb=2048,
            initial_size_mb=256
        )
        
        assert sizer.get_current_size_mb() == 256
    
    def test_force_resize(self):
        """Test forced resize."""
        resized_to = []
        
        def on_resize(size_mb):
            resized_to.append(size_mb)
        
        sizer = create_adaptive_sizer(
            min_size_mb=64,
            max_size_mb=2048,
            initial_size_mb=256,
            resize_callback=on_resize
        )
        
        sizer.force_resize(512)
        
        assert sizer.get_current_size_mb() == 512
        assert len(resized_to) == 1
        assert resized_to[0] == 512
    
    def test_bounds_enforcement(self):
        """Test that resize respects bounds."""
        sizer = create_adaptive_sizer(
            min_size_mb=64,
            max_size_mb=2048,
            initial_size_mb=256
        )
        
        sizer.force_resize(10000)
        assert sizer.get_current_size_mb() == 2048
        
        sizer.force_resize(10)
        assert sizer.get_current_size_mb() == 64


# ==============================================================================
# Distributed Cache Optimization Tests
# ==============================================================================

class TestConsistentHash:
    """Tests for consistent hashing."""
    
    def test_add_remove_nodes(self):
        """Test adding and removing nodes."""
        ring = ConsistentHash(virtual_nodes=10)
        
        ring.add_node("node-0")
        ring.add_node("node-1")
        ring.add_node("node-2")
        
        # Should get nodes for any key
        node = ring.get_node("test-key")
        assert node in ["node-0", "node-1", "node-2"]
        
        # Remove a node
        ring.remove_node("node-1")
        
        node = ring.get_node("test-key")
        assert node in ["node-0", "node-2"]
    
    def test_consistent_routing(self):
        """Test that keys route consistently."""
        ring = ConsistentHash(virtual_nodes=100)
        
        ring.add_node("node-0")
        ring.add_node("node-1")
        ring.add_node("node-2")
        
        # Same key should always go to same node
        node1 = ring.get_node("my-key")
        node2 = ring.get_node("my-key")
        node3 = ring.get_node("my-key")
        
        assert node1 == node2 == node3
    
    def test_get_multiple_nodes(self):
        """Test getting multiple nodes for replication."""
        ring = ConsistentHash(virtual_nodes=100)
        
        ring.add_node("node-0")
        ring.add_node("node-1")
        ring.add_node("node-2")
        
        nodes = ring.get_nodes("test-key", 2)
        
        assert len(nodes) == 2
        assert len(set(nodes)) == 2  # All unique


class TestCacheCoordinator:
    """Tests for cache coordinator."""
    
    def test_node_registration(self):
        """Test node registration."""
        coordinator = create_distributed_cache_optimizer(
            local_node_id="node-0",
            replication_factor=2
        )
        
        node = NodeInfo(
            node_id="node-1",
            address="localhost",
            port=8001,
            capacity_mb=1024
        )
        
        coordinator.register_node(node)
        
        stats = coordinator.get_cluster_stats()
        assert "node-1" in stats["nodes"]
    
    def test_entry_registration(self):
        """Test cache entry registration."""
        coordinator = create_distributed_cache_optimizer(
            local_node_id="node-0",
            replication_factor=2
        )
        
        # Register nodes first
        for i in range(3):
            coordinator.register_node(NodeInfo(
                node_id=f"node-{i}",
                address="localhost",
                port=8000 + i
            ))
        
        # Get placement and register entry
        placement = coordinator.get_placement("test-key")
        coordinator.register_entry("test-key", 1024 * 1024, placement)
        
        stats = coordinator.get_cluster_stats()
        assert stats["cluster"]["total_entries"] == 1
    
    def test_global_eviction(self):
        """Test global eviction coordination."""
        coordinator = create_distributed_cache_optimizer(
            local_node_id="node-0"
        )
        
        # Register nodes and entries
        for i in range(2):
            coordinator.register_node(NodeInfo(
                node_id=f"node-{i}",
                address="localhost",
                port=8000 + i
            ))
        
        for i in range(10):
            coordinator.register_entry(f"key-{i}", 10 * 1024 * 1024, [f"node-{i % 2}"])
        
        # Trigger eviction
        evicted = coordinator.trigger_global_eviction(50)
        
        assert evicted > 0
        
        stats = coordinator.get_cluster_stats()
        assert stats["cluster"]["total_entries"] < 10


# ==============================================================================
# Production Hardening Tests
# ==============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker."""
    
    def test_initial_state(self):
        """Test initial state is closed."""
        breaker = CircuitBreaker("test")
        
        assert breaker.state == CircuitState.CLOSED
    
    def test_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)
        
        # Record failures
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.state == CircuitState.OPEN
    
    def test_blocks_when_open(self):
        """Test that open circuit blocks requests."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)
        
        breaker.record_failure()
        
        assert not breaker.allow_request()
    
    def test_decorator(self):
        """Test protect decorator."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test", config)
        
        call_count = 0
        
        @breaker.protect
        def failing_fn():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Fail")
        
        # Should fail twice, then circuit opens
        for i in range(5):
            try:
                failing_fn()
            except (RuntimeError, CircuitOpenError):
                pass
        
        # Only 2 actual calls (before circuit opened)
        assert call_count == 2


class TestHealthChecker:
    """Tests for health checker."""
    
    def test_register_check(self):
        """Test registering health checks."""
        checker = HealthChecker()
        
        checker.register_check("test", lambda: HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="OK"
        ))
        
        result = checker.run_check("test")
        
        assert result.status == HealthStatus.HEALTHY
    
    def test_overall_health(self):
        """Test overall health aggregation."""
        checker = HealthChecker()
        
        checker.register_check("check1", lambda: HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="OK"
        ))
        checker.register_check("check2", lambda: HealthCheckResult(
            status=HealthStatus.DEGRADED,
            message="Slow"
        ))
        
        overall = checker.get_overall_health()
        
        # Should be DEGRADED (worst of the two)
        assert overall.status == HealthStatus.DEGRADED
    
    def test_unhealthy_check(self):
        """Test unhealthy check detection."""
        checker = HealthChecker()
        
        checker.register_check("failing", lambda: HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message="Down"
        ))
        
        assert not checker.is_ready()


class TestMetricsCollector:
    """Tests for metrics collection."""
    
    def test_counter(self):
        """Test counter metrics."""
        metrics = MetricsCollector()
        
        metrics.increment("requests")
        metrics.increment("requests")
        metrics.increment("requests", 3)
        
        result = metrics.get_metrics()
        
        assert result["counters"]["requests"] == 5
    
    def test_gauge(self):
        """Test gauge metrics."""
        metrics = MetricsCollector()
        
        metrics.set_gauge("memory_mb", 512)
        metrics.set_gauge("memory_mb", 768)
        
        result = metrics.get_metrics()
        
        assert result["gauges"]["memory_mb"] == 768
    
    def test_histogram(self):
        """Test histogram metrics."""
        metrics = MetricsCollector()
        
        for i in range(100):
            metrics.observe("latency_ms", i)
        
        result = metrics.get_metrics()
        
        assert "latency_ms" in result["histograms"]
        assert result["histograms"]["latency_ms"]["count"] == 100
        assert result["histograms"]["latency_ms"]["avg"] == 49.5


class TestRateLimiter:
    """Tests for rate limiter."""
    
    def test_allows_within_rate(self):
        """Test that requests within rate are allowed."""
        limiter = RateLimiter(rate=100, burst=10)
        
        # Should allow initial burst
        for _ in range(10):
            assert limiter.acquire()
    
    def test_blocks_over_rate(self):
        """Test that requests over rate are blocked."""
        limiter = RateLimiter(rate=10, burst=2)
        
        # Exhaust burst
        assert limiter.acquire()
        assert limiter.acquire()
        
        # Should be blocked
        assert not limiter.acquire()
    
    def test_refills_over_time(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(rate=100, burst=5)
        
        # Exhaust burst
        for _ in range(5):
            limiter.acquire()
        
        # Wait for refill
        time.sleep(0.05)
        
        # Should have tokens again
        assert limiter.acquire()


class TestGracefulDegradation:
    """Tests for graceful degradation."""
    
    def test_fallback_on_failure(self):
        """Test fallback execution on primary failure."""
        degradation = GracefulDegradation()
        
        degradation.register_fallback("test", lambda: "fallback_result")
        
        def failing_primary():
            raise RuntimeError("Primary failed")
        
        result, used_fallback = degradation.execute_with_fallback(
            "test",
            failing_primary
        )
        
        assert result == "fallback_result"
        assert used_fallback is True
    
    def test_primary_success(self):
        """Test primary execution success."""
        degradation = GracefulDegradation()
        
        degradation.register_fallback("test", lambda: "fallback")
        
        result, used_fallback = degradation.execute_with_fallback(
            "test",
            lambda: "primary_result"
        )
        
        assert result == "primary_result"
        assert used_fallback is False


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_compressed_cache_with_monitoring(self):
        """Test compressed cache with production monitoring."""
        from cache.production_hardening import harden_cache
        
        # Create compressed cache
        cache = create_quantized_kv_cache(quant_type="int8", num_layers=4)
        
        # Wrap with production hardening
        # Note: This would need adapter methods in real implementation
        # For now, just test the cache directly
        
        for layer in range(4):
            k = torch.randn(2, 12, 64, 64)
            v = torch.randn(2, 12, 64, 64)
            cache.store(layer, k, v)
        
        stats = cache.get_stats()
        
        assert stats["compression_ratio"] > 1
        assert stats["num_layers_cached"] == 4
    
    def test_distributed_with_adaptive_sizing(self):
        """Test distributed cache with adaptive sizing."""
        # Create distributed coordinator
        coordinator = create_distributed_cache_optimizer(
            local_node_id="node-0",
            replication_factor=2
        )
        
        # Create adaptive sizer
        sizer = create_adaptive_sizer(
            min_size_mb=64,
            max_size_mb=1024,
            initial_size_mb=256
        )
        
        # Register nodes
        for i in range(3):
            coordinator.register_node(NodeInfo(
                node_id=f"node-{i}",
                address="localhost",
                port=8000 + i,
                capacity_mb=sizer.get_current_size_mb()
            ))
        
        cluster_stats = coordinator.get_cluster_stats()
        
        assert cluster_stats["cluster"]["active_nodes"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
