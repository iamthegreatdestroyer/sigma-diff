"""
Cache Compression & Adaptive Sizing Tests
==========================================

Tests for:
- KV cache compression (int8, int4, quantization)
- Adaptive threshold tuning
- Workload characterization
- Cache size optimization
- Dynamic allocation

Sprint 2.2 Days 5-6 - Cache Optimization Tests
Created: 2025-12-27
"""

import pytest
import torch
import time
from typing import Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache.compression import (
    QuantizationScheme,
    KVCacheCompressor,
    AdaptiveCompressionSelector,
    QuantizationAwareCache,
    CompressionFormat
)
from src.cache.adaptive_sizing import (
    AdaptiveThresholdController,
    WorkloadCharacterizer,
    CacheSizeOptimizer,
    DynamicCacheAllocator,
    WorkloadPattern
)


# ============================================================================
# Compression Tests
# ============================================================================

class TestQuantizationScheme:
    """Tests for quantization scheme."""
    
    def test_int8_quantization(self):
        """Test int8 quantization."""
        scheme = QuantizationScheme(bits=8)
        
        # Create test tensor
        x = torch.randn(100, 64)
        
        # Quantize
        scale, zp = scheme.compute_scale_zero_point(x)
        x_q, _, _ = scheme.quantize(x)
        
        assert x_q.dtype == torch.int8
        assert x_q.shape == x.shape
    
    def test_int4_quantization(self):
        """Test int4 quantization."""
        scheme = QuantizationScheme(bits=4)
        
        x = torch.randn(100, 64)
        x_q, scale, zp = scheme.quantize(x)
        
        assert x_q.dtype == torch.uint8
        assert x_q.shape == x.shape
    
    def test_quantization_dequantization(self):
        """Test quantization-dequantization round-trip."""
        scheme = QuantizationScheme(bits=8)
        
        x = torch.randn(50, 32)
        x_q, scale, zp = scheme.quantize(x)
        x_dq = scheme.dequantize(x_q, scale, zp)
        
        # Check error
        error = torch.abs(x - x_dq).mean().item()
        assert error < 0.1  # Reasonable error for int8
    
    def test_per_channel_quantization(self):
        """Test per-channel quantization."""
        scheme = QuantizationScheme(bits=8)
        
        x = torch.randn(10, 64)
        x_q, scale, zp = scheme.quantize(x, per_channel=True)
        
        assert x_q.shape == x.shape
        assert scale.shape[0] == 10  # Per-channel scale


class TestKVCacheCompressor:
    """Tests for KV cache compressor."""
    
    def test_compression_int8(self):
        """Test int8 compression."""
        compressor = KVCacheCompressor(format=CompressionFormat.INT8)
        
        k = torch.randn(1000, 64)
        v = torch.randn(1000, 64)
        
        compressed = compressor.compress(k, v)
        
        assert compressed is not None
        assert compressed.compression_ratio > 3.0  # Should be ~4x
    
    def test_compression_int4(self):
        """Test int4 compression."""
        compressor = KVCacheCompressor(format=CompressionFormat.INT4)
        
        k = torch.randn(2000, 64)
        v = torch.randn(2000, 64)
        
        compressed = compressor.compress(k, v)
        
        assert compressed is not None
        assert compressed.compression_ratio > 6.0  # Should be ~8x
    
    def test_decompression_accuracy(self):
        """Test decompression accuracy."""
        compressor = KVCacheCompressor(format=CompressionFormat.INT8)
        
        k = torch.randn(500, 64)
        v = torch.randn(500, 64)
        
        compressed = compressor.compress(k, v)
        k_dec, v_dec = compressor.decompress(compressed)
        
        k_error = torch.abs(k - k_dec).mean().item()
        v_error = torch.abs(v - v_dec).mean().item()
        
        assert k_error < 0.05
        assert v_error < 0.05
    
    def test_below_threshold_no_compress(self):
        """Test that small sequences aren't compressed."""
        compressor = KVCacheCompressor(
            format=CompressionFormat.INT8,
            compression_threshold=1000
        )
        
        k = torch.randn(100, 64)  # Below threshold
        v = torch.randn(100, 64)
        
        compressed = compressor.compress(k, v)
        
        assert compressed is None  # Not compressed
    
    def test_compression_statistics(self):
        """Test compression statistics."""
        compressor = KVCacheCompressor(format=CompressionFormat.INT8)
        
        # Compress multiple tensors
        for _ in range(5):
            k = torch.randn(1000, 64)
            v = torch.randn(1000, 64)
            compressor.compress(k, v)
        
        stats = compressor.get_statistics()
        
        assert stats["compression_ratio"] > 1.0
        assert stats["total_original_mb"] > 0


class TestQuantizationAwareCache:
    """Tests for quantization-aware cache."""
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving compressed cache."""
        cache = QuantizationAwareCache(format=CompressionFormat.INT8)
        
        k = torch.randn(1000, 64)
        v = torch.randn(1000, 64)
        
        ratio = cache.store("seq_1", k, v)
        assert ratio > 1.0
        
        k_ret, v_ret = cache.retrieve("seq_1")
        assert k_ret is not None
        assert k_ret.shape == k.shape
    
    def test_compression_format_selection(self):
        """Test adaptive format selection."""
        selector = AdaptiveCompressionSelector()
        
        # Test sensitivity estimation
        original = torch.randn(100, 64)
        compressed = torch.randn(100, 64)
        dequantized = original + torch.randn_like(original) * 0.01
        
        sensitivity = selector.estimate_sensitivity(original, compressed, dequantized)
        
        assert 0 <= sensitivity <= 1


# ============================================================================
# Adaptive Sizing Tests
# ============================================================================

class TestAdaptiveThresholdController:
    """Tests for adaptive threshold control."""
    
    def test_threshold_adjustment(self):
        """Test threshold adjustment based on hit rate."""
        controller = AdaptiveThresholdController(initial_threshold=0.85)
        
        # Simulate low hit rate (below target)
        for _ in range(30):
            controller.update(False)  # Miss
        
        for _ in range(10):
            controller.update(True)   # Hit (33% rate < 70% target)
        
        new_threshold = controller.adjust_threshold()
        assert new_threshold < 0.85  # Should lower threshold
    
    def test_hit_rate_computation(self):
        """Test hit rate computation."""
        controller = AdaptiveThresholdController()
        
        # 70% hits
        for _ in range(70):
            controller.update(True)
        for _ in range(30):
            controller.update(False)
        
        hit_rate = controller.compute_hit_rate()
        assert 0.65 < hit_rate < 0.75
    
    def test_threshold_bounds(self):
        """Test threshold stays within bounds."""
        controller = AdaptiveThresholdController(
            min_threshold=0.70,
            max_threshold=0.95
        )
        
        # Force many adjustments
        for _ in range(100):
            controller.update(True)
        
        threshold = controller.adjust_threshold()
        
        assert controller.min_threshold <= threshold <= controller.max_threshold


class TestWorkloadCharacterizer:
    """Tests for workload characterization."""
    
    def test_repetitive_pattern_detection(self):
        """Test detection of repetitive workload."""
        characterizer = WorkloadCharacterizer(window_size=100)
        
        # Repetitive: same sequence repeated
        seq = torch.tensor([1, 2, 3, 4, 5])
        for _ in range(50):
            characterizer.add_sequence(seq, hit=True, hit_type="exact")
        
        pattern = characterizer.detect_pattern()
        assert pattern == WorkloadPattern.REPETITIVE
    
    def test_diverse_pattern_detection(self):
        """Test detection of diverse workload."""
        characterizer = WorkloadCharacterizer(window_size=100)
        
        # Diverse: different sequences
        for i in range(50):
            seq = torch.randint(0, 1000, (100,))
            characterizer.add_sequence(seq, hit=False, hit_type="miss")
        
        pattern = characterizer.detect_pattern()
        assert pattern == WorkloadPattern.DIVERSE
    
    def test_metrics_computation(self):
        """Test workload metrics computation."""
        characterizer = WorkloadCharacterizer(window_size=100)
        
        for i in range(30):
            seq = torch.randint(0, 1000, (150,))
            hit = i % 3 == 0  # 33% hit rate
            characterizer.add_sequence(seq, hit=hit, hit_type="semantic" if hit else "miss")
        
        metrics = characterizer.characterize()
        
        assert 0 <= metrics.hit_rate <= 1
        assert metrics.avg_sequence_length > 0
        assert metrics.num_unique_sequences > 0


class TestCacheSizeOptimizer:
    """Tests for cache size optimization."""
    
    def test_size_recommendation(self):
        """Test cache size recommendation."""
        optimizer = CacheSizeOptimizer()
        
        # Add workload data
        for i in range(100):
            seq = torch.randint(0, 1000, (100,))
            optimizer.characterizer.add_sequence(
                seq,
                hit=i % 4 == 0,
                hit_type="exact" if i % 4 == 0 else "miss"
            )
        
        recommendation = optimizer.recommend_size(available_memory_gb=8.0)
        
        assert recommendation.exact_cache_size > 0
        assert recommendation.semantic_cache_size > 0
        assert recommendation.estimated_memory_mb > 0
    
    def test_compression_recommendation(self):
        """Test compression is recommended for diverse workloads."""
        optimizer = CacheSizeOptimizer()
        
        # Add diverse workload
        for i in range(100):
            seq = torch.randint(0, 10000, (200,))
            optimizer.characterizer.add_sequence(seq, hit=False)
        
        recommendation = optimizer.recommend_size()
        
        # Diverse workloads should use compression
        assert recommendation.compression_enabled


class TestDynamicCacheAllocator:
    """Tests for dynamic cache allocation."""
    
    def test_exact_cache_allocation(self):
        """Test exact cache allocation."""
        allocator = DynamicCacheAllocator(
            initial_exact_size=10,
            initial_semantic_size=10
        )
        
        for i in range(10):
            assert allocator.allocate_exact()
        
        # Should be at capacity
        assert not allocator.can_allocate_exact()
    
    def test_capacity_expansion(self):
        """Test automatic capacity expansion."""
        allocator = DynamicCacheAllocator(
            initial_exact_size=5,
            initial_semantic_size=5,
            max_total_sequences=100
        )
        
        # Fill initial capacity
        for i in range(5):
            allocator.allocate_exact()
        
        # Expand beyond initial
        for i in range(10):
            assert allocator.allocate_exact()
        
        assert allocator.expansion_count > 0
    
    def test_allocator_statistics(self):
        """Test allocator statistics."""
        allocator = DynamicCacheAllocator(
            initial_exact_size=20,
            initial_semantic_size=20
        )
        
        for _ in range(15):
            allocator.allocate_exact()
        
        for _ in range(10):
            allocator.allocate_semantic()
        
        stats = allocator.get_statistics()
        
        assert 0 <= stats["exact_utilization"] <= 1
        assert 0 <= stats["semantic_utilization"] <= 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestCompressionIntegration:
    """Integration tests for compression."""
    
    def test_compression_speedup(self):
        """Test that compressed cache provides speedup."""
        cache = QuantizationAwareCache(format=CompressionFormat.INT8)
        
        k = torch.randn(2000, 64)
        v = torch.randn(2000, 64)
        
        # Store compressed
        cache.store("seq_1", k, v)
        
        # Time retrieval
        start = time.time()
        for _ in range(100):
            _ = cache.retrieve("seq_1")
        elapsed = time.time() - start
        
        # Should be fast (even with decompression)
        assert elapsed < 1.0  # 100 retrievals in under 1 second


class TestAdaptiveIntegration:
    """Integration tests for adaptive sizing."""
    
    def test_full_adaptation_cycle(self):
        """Test complete adaptation cycle."""
        controller = AdaptiveThresholdController()
        characterizer = WorkloadCharacterizer()
        optimizer = CacheSizeOptimizer()
        
        # Simulate workload
        for i in range(200):
            # Create varied workload
            if i < 50:
                # Repetitive phase
                seq = torch.tensor([1, 2, 3, 4, 5])
                hit = i % 2 == 0
            else:
                # Similar phase
                seq = torch.randint(1, 10, (50,))
                hit = i % 4 == 0
            
            # Update systems
            characterizer.add_sequence(seq, hit=hit)
            controller.update(hit)
            controller.adjust_threshold()
        
        # Get recommendations
        metrics = characterizer.characterize()
        recommendation = optimizer.recommend_size()
        
        assert metrics.num_total_sequences == 200
        assert recommendation.exact_cache_size > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestCompressionPerformance:
    """Performance tests for compression."""
    
    def test_compression_throughput(self):
        """Test compression throughput."""
        compressor = KVCacheCompressor(format=CompressionFormat.INT8)
        
        start = time.time()
        
        for _ in range(100):
            k = torch.randn(1000, 64)
            v = torch.randn(1000, 64)
            compressor.compress(k, v)
        
        elapsed = time.time() - start
        
        # Should compress 100 KV pairs in reasonable time
        throughput = 100 / elapsed
        assert throughput > 10  # At least 10 per second


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
