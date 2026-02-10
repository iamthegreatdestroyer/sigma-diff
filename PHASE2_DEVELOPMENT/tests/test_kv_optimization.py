"""
KV Cache Optimization - Comprehensive Test Suite
Sprint 4.4 - Task 5: Testing

Tests for all KV cache optimization components:
- Semantic compression (Task 1)
- Eviction policies (Task 2)
- Memory layout (Task 3)
- Benchmarks (Task 4)

Target: >90% code coverage
"""

import pytest
import numpy as np
from typing import Tuple
import tempfile
import os


# =============================================================================
# Task 1: Semantic Compression Tests
# =============================================================================

class TestSemanticCompression:
    """Test semantic compression algorithms."""
    
    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample KV cache data."""
        keys = np.random.randn(64, 512).astype(np.float32)  # 64 tokens, 512 dims
        values = np.random.randn(64, 512).astype(np.float32)
        return keys, values
    
    def test_low_rank_compression(self, sample_data):
        """Test low-rank approximation compression."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_semantic_compression import LowRankCompression
        
        keys, values = sample_data
        lr = LowRankCompression(rank_fraction=0.5)
        
        U, S, Vt, meta = lr.compress(keys)
        
        # Check dimensions
        assert U.shape[0] == keys.shape[0]
        assert U.shape[1] <= keys.shape[1]  # Reduced rank
        assert len(S) == U.shape[1]
        assert Vt.shape[0] == U.shape[1]
        
        # Check reconstruction
        reconstructed = lr.decompress(U, S, Vt, meta)
        assert reconstructed.shape == keys.shape
        
        # Check error is small
        error = np.mean(np.abs(reconstructed - keys))
        assert error < 0.5, f"Reconstruction error too high: {error}"
    
    def test_quantization_int8(self, sample_data):
        """Test INT8 quantization."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_semantic_compression import QuantizationCompression, QuantizationType
        
        keys, _ = sample_data
        quant = QuantizationCompression(QuantizationType.INT8)
        
        quantized, meta = quant.compress(keys)
        
        # Check dtype
        assert quantized.dtype == np.int8
        
        # Check metadata
        assert "min_val" in meta
        assert "max_val" in meta
        assert "scale" in meta
        
        # Check dequantization
        dequantized = quant.decompress(quantized, meta)
        assert dequantized.shape == keys.shape
        
        # Check error <1% for INT8
        relative_error = np.mean(np.abs(dequantized - keys)) / np.mean(np.abs(keys))
        assert relative_error < 0.01, f"Quantization error too high: {relative_error:.2%}"
    
    def test_quantization_int4(self, sample_data):
        """Test INT4 quantization."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_semantic_compression import QuantizationCompression, QuantizationType
        
        keys, _ = sample_data
        quant = QuantizationCompression(QuantizationType.INT4)
        
        quantized, meta = quant.compress(keys)
        
        # INT4 packed: 2 values per byte
        expected_size = (np.prod(keys.shape) + 1) // 2
        assert quantized.nbytes <= expected_size * 2  # Upper bound
        
        # Check dequantization works
        dequantized = quant.decompress(quantized, meta)
        assert dequantized.shape == keys.shape
    
    def test_clustering_compression(self, sample_data):
        """Test token clustering compression."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_semantic_compression import ClusteringCompression
        
        keys, _ = sample_data
        clustering = ClusteringCompression(num_clusters=32)
        
        centers, assignments, meta = clustering.compress(keys)
        
        # Check clusters
        assert centers.shape[0] == 32
        assert assignments.shape[0] == keys.shape[0]
        assert np.all(assignments >= 0) and np.all(assignments < 32)
        
        # Check reconstruction
        reconstructed = clustering.decompress(centers, assignments, meta)
        assert reconstructed.shape == keys.shape
    
    def test_compression_engine_hybrid(self, sample_data):
        """Test adaptive compression engine."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_semantic_compression import create_compression_engine
        
        keys, values = sample_data
        engine = create_compression_engine(method="hybrid", rank_fraction=0.5, quant_type="int8")
        
        compressed, stats = engine.compress_batch(keys, values, original_keys=keys)
        
        # Check stats
        assert stats.original_size > 0
        assert stats.compressed_size > 0
        assert stats.compression_ratio > 0
        assert stats.compression_ratio <= 1.0
        
        # Check memory savings
        assert stats.memory_saved_percent >= 0


# =============================================================================
# Task 2: Eviction Policy Tests
# =============================================================================

class TestEvictionPolicies:
    """Test cache eviction policies."""
    
    def test_lru_cache_basic(self):
        """Test LRU cache basic operations."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_cache_eviction import LRUCache
        
        cache = LRUCache(max_size_bytes=1000)
        
        # Add items
        cache.put("key1", "value1", size_bytes=200)
        cache.put("key2", "value2", size_bytes=200)
        
        # Get items
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        
        # Check stats
        assert cache.stats.total_hits == 2
        assert cache.stats.total_accesses == 2
    
    def test_lru_cache_eviction(self):
        """Test LRU eviction under memory pressure."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_cache_eviction import LRUCache
        
        cache = LRUCache(max_size_bytes=500)
        
        # Fill cache
        cache.put("key1", "value1", size_bytes=200)
        cache.put("key2", "value2", size_bytes=200)
        
        # Access key1 to mark as recently used
        cache.get("key1")
        
        # Add new item, should evict key2 (LRU)
        cache.put("key3", "value3", size_bytes=200)
        
        # key1 should still be there, key2 should be gone
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        
        assert cache.stats.total_evictions == 1
    
    def test_lfu_cache(self):
        """Test LFU cache."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_cache_eviction import LFUCache
        
        cache = LFUCache(max_size_bytes=500)
        
        # Add items
        cache.put("key1", "value1", size_bytes=200)
        cache.put("key2", "value2", size_bytes=200)
        
        # Access key1 multiple times (increase frequency)
        for _ in range(5):
            cache.get("key1")
        
        # Access key2 once
        cache.get("key2")
        
        # Add new item, should evict key2 (lower frequency)
        cache.put("key3", "value3", size_bytes=200)
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted (lower frequency)
        assert cache.get("key3") == "value3"
    
    def test_fifo_cache(self):
        """Test FIFO cache."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_cache_eviction import FIFOCache
        
        cache = FIFOCache(max_size_bytes=500)
        
        # Add items in order
        cache.put("key1", "value1", size_bytes=200)
        cache.put("key2", "value2", size_bytes=200)
        
        # Add new item, should evict key1 (first in)
        cache.put("key3", "value3", size_bytes=200)
        
        assert cache.get("key1") is None  # Evicted (FIFO)
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_hybrid_cache_adaptation(self):
        """Test hybrid cache policy adaptation."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_cache_eviction import HybridAdaptiveCache
        
        cache = HybridAdaptiveCache(max_size_bytes=1000)
        
        # Add some items and access patterns
        for i in range(50):
            key = f"key{i}"
            cache.put(key, f"value{i}", size_bytes=10)
            cache.get(key)
        
        # Cache should still be functional
        assert cache.stats.total_accesses > 0


# =============================================================================
# Task 3: Memory Layout Tests
# =============================================================================

class TestMemoryLayout:
    """Test memory layout optimization."""
    
    @pytest.fixture
    def sample_matrix(self) -> np.ndarray:
        """Generate sample matrix."""
        return np.random.randn(128, 1024).astype(np.float32)
    
    def test_aligned_layout(self, sample_matrix):
        """Test cache-line aligned layout."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_memory_layout import AlignedLayoutOptimizer
        
        opt = AlignedLayoutOptimizer(alignment=64)
        aligned, meta = opt.layout(sample_matrix)
        
        # Check alignment
        assert aligned.shape[0] == sample_matrix.shape[0]
        assert aligned.shape[1] >= sample_matrix.shape[1]
        
        # Check restoration
        restored = opt.restore(aligned, meta)
        assert restored.shape == sample_matrix.shape
        assert np.allclose(restored, sample_matrix)
    
    def test_blocked_layout(self, sample_matrix):
        """Test blocked memory layout."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_memory_layout import BlockedLayoutOptimizer
        
        opt = BlockedLayoutOptimizer(block_size=256)
        blocked, meta = opt.layout(sample_matrix)
        
        # Check block structure
        assert len(blocked.shape) == 3
        assert blocked.shape[0] == sample_matrix.shape[0]
        
        # Check restoration
        restored = opt.restore(blocked, meta)
        assert restored.shape == sample_matrix.shape
    
    def test_interleaved_layout(self, sample_matrix):
        """Test interleaved K/V layout."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_memory_layout import InterleavedLayoutOptimizer
        
        values = np.random.randn(*sample_matrix.shape).astype(np.float32)
        
        opt = InterleavedLayoutOptimizer()
        interleaved, meta = opt.layout(sample_matrix, values)
        
        # Check interleaving
        assert interleaved.shape[1] == sample_matrix.shape[1] * 2
        
        # Check restoration
        restored_k, restored_v = opt.restore(interleaved, meta)
        assert np.allclose(restored_k, sample_matrix)
        assert np.allclose(restored_v, values)
    
    def test_columnar_layout(self, sample_matrix):
        """Test column-major layout."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_memory_layout import ColumnarLayoutOptimizer
        
        opt = ColumnarLayoutOptimizer()
        columnar, meta = opt.layout(sample_matrix)
        
        # Check transpose
        assert columnar.shape == (sample_matrix.shape[1], sample_matrix.shape[0])
        
        # Check restoration
        restored = opt.restore(columnar, meta)
        assert restored.shape == sample_matrix.shape
        assert np.allclose(restored, sample_matrix)
    
    def test_access_pattern_analyzer(self):
        """Test access pattern analysis."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_memory_layout import AccessPatternAnalyzer
        
        analyzer = AccessPatternAnalyzer()
        
        # Record spatially local accesses
        for i in range(100):
            analyzer.record_access(i, i)  # Diagonal pattern
        
        spatial = analyzer.compute_spatial_locality()
        assert 0 <= spatial <= 1
        
        temporal = analyzer.compute_temporal_locality()
        assert 0 <= temporal <= 1
    
    def test_memory_pool_reuse(self):
        """Test memory pool for buffer reuse."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_memory_layout import MemoryPool
        
        pool = MemoryPool(buffer_size=1024, num_buffers=4)
        
        # Allocate and deallocate
        buf_id1, buf1 = pool.allocate()
        assert buf1.shape[0] == 1024
        
        buf_id2, buf2 = pool.allocate()
        
        # Return first buffer
        assert pool.deallocate(buf_id1)
        
        # Allocate again, should reuse
        buf_id3, buf3 = pool.allocate()
        assert buf3 is buf1  # Same buffer object
        
        # Check reuse rate
        assert pool.reuse_rate > 0


# =============================================================================
# Task 4: Benchmark Tests
# =============================================================================

class TestBenchmarks:
    """Test benchmark suite."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_benchmark import KVCacheBenchmark
        
        bench = KVCacheBenchmark(hidden_dim=4096, max_seq_len=2048)
        assert bench.hidden_dim == 4096
        assert bench.max_seq_len == 2048
        assert len(bench.results) == 0
    
    def test_baseline_benchmark(self):
        """Test baseline benchmark."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_benchmark import KVCacheBenchmark
        
        bench = KVCacheBenchmark(hidden_dim=512, max_seq_len=256)
        result = bench.benchmark_baseline()
        
        assert result.name == "Baseline (No Optimization)"
        assert result.latency.mean_ms > 0
        assert result.throughput.tokens_per_sec > 0
        assert result.memory.compression_ratio == 1.0
        assert result.cache.hit_rate == 0.0
    
    def test_compression_benchmark(self):
        """Test compression benchmark."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_benchmark import KVCacheBenchmark
        
        bench = KVCacheBenchmark(hidden_dim=512, max_seq_len=256)
        result = bench.benchmark_with_compression()
        
        assert result.name == "With Semantic Compression (INT8 + Low-Rank)"
        assert result.memory.compression_ratio > 1.0  # Some compression
        assert result.cache.hit_rate > 0.0
    
    def test_eviction_benchmark(self):
        """Test eviction policy benchmark."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_benchmark import KVCacheBenchmark
        
        bench = KVCacheBenchmark(hidden_dim=512, max_seq_len=256)
        result = bench.benchmark_with_eviction()
        
        assert result.name == "With LRU Eviction Policy"
        assert result.cache.hit_rate > 0.0
        assert result.cache.eviction_count >= 0
    
    def test_layout_benchmark(self):
        """Test memory layout benchmark."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_benchmark import KVCacheBenchmark
        
        bench = KVCacheBenchmark(hidden_dim=512, max_seq_len=256)
        result = bench.benchmark_with_layout()
        
        assert result.name == "With Optimized Memory Layout (Aligned + Blocked)"
        assert result.latency.mean_ms > 0
    
    def test_combined_benchmark(self):
        """Test combined optimizations benchmark."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_benchmark import KVCacheBenchmark
        
        bench = KVCacheBenchmark(hidden_dim=512, max_seq_len=256)
        result = bench.benchmark_combined()
        
        assert result.name == "Combined Optimization (Compression + Eviction + Layout)"
        # Combined should show best improvements
        assert result.memory.compression_ratio > 1.0
        assert result.cache.hit_rate > 0.6


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_compression_with_eviction(self):
        """Test compression working with eviction."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_semantic_compression import create_compression_engine
        from PHASE2_DEVELOPMENT.src.optimization.kv_cache_eviction import create_cache
        
        # Create components
        engine = create_compression_engine(method="hybrid")
        cache = create_cache(policy="lru", max_size_bytes=1024*1024)
        
        # Generate data
        keys = np.random.randn(64, 512).astype(np.float32)
        values = np.random.randn(64, 512).astype(np.float32)
        
        # Compress and cache
        compressed, stats = engine.compress_batch(keys, values)
        cache.put("compressed_kv", compressed, size_bytes=stats.compressed_size)
        
        # Retrieve from cache
        result = cache.get("compressed_kv")
        assert result is not None
        assert cache.stats.total_hits > 0
    
    def test_layout_with_compression(self):
        """Test memory layout with compression."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_memory_layout import create_layout_manager
        from PHASE2_DEVELOPMENT.src.optimization.kv_semantic_compression import create_compression_engine
        
        # Create components
        manager = create_layout_manager(hidden_dim=512, max_seq_len=256)
        engine = create_compression_engine(method="quantization")
        
        # Generate data
        keys = np.random.randn(64, 512).astype(np.float32)
        values = np.random.randn(64, 512).astype(np.float32)
        
        # Apply layout optimization
        optimized, layout_stats = manager.optimize_layout(keys, values)
        assert layout_stats.locality_score > 0
        
        # Then compress
        compressed, comp_stats = engine.compress_batch(keys, values)
        assert comp_stats.compression_ratio > 0
    
    def test_all_components_together(self):
        """Test all components working together."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_semantic_compression import create_compression_engine
        from PHASE2_DEVELOPMENT.src.optimization.kv_cache_eviction import create_cache
        from PHASE2_DEVELOPMENT.src.optimization.kv_memory_layout import create_layout_manager
        from PHASE2_DEVELOPMENT.src.optimization.kv_benchmark import KVCacheBenchmark
        
        # Create all components
        compression_engine = create_compression_engine(method="hybrid")
        cache = create_cache(policy="hybrid")
        layout_manager = create_layout_manager(hidden_dim=512, max_seq_len=256)
        benchmark = KVCacheBenchmark(hidden_dim=512, max_seq_len=256)
        
        # Run workflow
        keys = np.random.randn(64, 512).astype(np.float32)
        values = np.random.randn(64, 512).astype(np.float32)
        
        # Apply layout
        optimized_layout, _ = layout_manager.optimize_layout(keys, values)
        
        # Compress
        compressed, comp_stats = compression_engine.compress_batch(keys, values)
        
        # Cache
        cache.put("optimized", compressed, size_bytes=comp_stats.compressed_size)
        result = cache.get("optimized")
        
        assert result is not None
        assert cache.stats.total_hits > 0
        
        # Run benchmarks
        results = benchmark.run_all_benchmarks()
        assert len(results) == 5


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    def test_large_scale_compression(self):
        """Test compression on large matrices."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_semantic_compression import create_compression_engine
        
        engine = create_compression_engine(method="hybrid")
        
        # Large batch
        keys = np.random.randn(1024, 4096).astype(np.float32)
        values = np.random.randn(1024, 4096).astype(np.float32)
        
        compressed, stats = engine.compress_batch(keys, values)
        
        # Should still achieve reasonable compression
        assert stats.compression_ratio > 1.0
        assert stats.compression_time_ms < 1000  # Under 1 second
    
    def test_cache_under_memory_pressure(self):
        """Test cache behavior under memory pressure."""
        from PHASE2_DEVELOPMENT.src.optimization.kv_cache_eviction import create_cache
        
        cache = create_cache(policy="lru", max_size_bytes=100*1024)  # 100KB
        
        # Add many items until eviction occurs
        for i in range(100):
            cache.put(f"key{i}", f"value{i}", size_bytes=2048)
        
        # Should have evicted some items
        assert cache.stats.total_evictions > 0
        assert cache.stats.current_size_bytes <= cache.stats.max_size_bytes


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
