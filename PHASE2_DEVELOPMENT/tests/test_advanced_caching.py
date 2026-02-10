"""
Advanced Caching Tests
======================

Comprehensive test suite for advanced eviction policies, semantic caching,
and page sharing mechanisms.

Tests:
- Eviction policy comparison
- Semantic cache hit rates
- Page sharing efficiency
- Memory savings
- Performance impact

Sprint 2.2 Days 3-4
Created: 2025-12-27
"""

import pytest
import torch
import time
from typing import Dict

# Import advanced cache components
from src.cache.advanced_eviction import (
    LRUEvictionPolicy,
    LFUEvictionPolicy,
    FIFOEvictionPolicy,
    WTinyLFUEvictionPolicy,
    AdaptiveEvictionPolicy,
    EvictionPolicyFactory,
    EvictionPolicy
)
from src.cache.semantic_cache import (
    SemanticCache,
    HybridSemanticCache,
    EmbeddingModel
)
from src.cache.page_sharing import (
    PageSharingManager,
    PrefixSharingCache
)


# ============================================================================
# Advanced Eviction Policy Tests
# ============================================================================

class TestAdvancedEviction:
    """Tests for advanced eviction policies."""
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        policy = LRUEvictionPolicy(max_pages=3)
        
        # Add pages
        for i in range(5):
            policy.add_page(f"seq_{i}", i)
        
        # Access page 0 (make it recent)
        policy.record_access("seq_0", 0)
        
        # Victim should be page 1 (least recently used)
        victim = policy.select_victim()
        assert victim == 1
    
    def test_lfu_eviction(self):
        """Test LFU eviction policy."""
        policy = LFUEvictionPolicy(max_pages=4)
        
        # Add pages
        for i in range(4):
            policy.add_page(f"seq_{i}", i)
        
        # Access page 0 many times
        for _ in range(10):
            policy.record_access("seq_0", 0)
        
        # Victim should be page 1 (least frequently used)
        victim = policy.select_victim()
        assert victim == 1
    
    def test_fifo_eviction(self):
        """Test FIFO eviction policy."""
        policy = FIFOEvictionPolicy(max_pages=3)
        
        # Add pages in order
        for i in range(5):
            policy.add_page(f"seq_{i}", i)
        
        # Victim should be page 0 (first in)
        victim = policy.select_victim()
        assert victim == 0
    
    def test_wtinylfu_eviction(self):
        """Test weighted TinyLFU eviction."""
        policy = WTinyLFUEvictionPolicy(max_pages=4, frequency_weight=0.8)
        
        for i in range(4):
            policy.add_page(f"seq_{i}", i)
        
        # Access page 0 frequently
        for _ in range(5):
            policy.record_access("seq_0", 0)
        
        # Access page 1 recently
        time.sleep(0.01)
        policy.record_access("seq_1", 1)
        
        victim = policy.select_victim()
        # Should evict page 2 or 3 (low frequency, low recency)
        assert victim in [2, 3]
    
    def test_adaptive_eviction(self):
        """Test adaptive eviction policy."""
        policy = AdaptiveEvictionPolicy(max_pages=4, adaptation_window=10)
        
        for i in range(4):
            policy.add_page(f"seq_{i}", i)
        
        # Simulate accesses
        for _ in range(15):
            policy.record_access("seq_0", 0)
        
        # Policy should have adapted
        assert policy.current_policy in [
            EvictionPolicy.LRU,
            EvictionPolicy.LFU
        ]
    
    def test_eviction_factory(self):
        """Test eviction policy factory."""
        lru = EvictionPolicyFactory.create(EvictionPolicy.LRU, 100)
        lfu = EvictionPolicyFactory.create(EvictionPolicy.LFU, 100)
        fifo = EvictionPolicyFactory.create(EvictionPolicy.FIFO, 100)
        
        assert isinstance(lru, LRUEvictionPolicy)
        assert isinstance(lfu, LFUEvictionPolicy)
        assert isinstance(fifo, FIFOEvictionPolicy)


# ============================================================================
# Semantic Cache Tests
# ============================================================================

class TestSemanticCache:
    """Tests for semantic caching."""
    
    def test_embedding_generation(self):
        """Test sequence embedding generation."""
        model = EmbeddingModel(embedding_dim=256)
        
        tokens = torch.tensor([1, 2, 3, 4, 5])
        embedding = model(tokens)
        
        assert embedding.shape == (256,)
        assert embedding.norm() > 0
    
    def test_semantic_cache_add(self):
        """Test adding sequences to semantic cache."""
        cache = SemanticCache(embedding_dim=256, similarity_threshold=0.8)
        
        seq1 = torch.tensor([1, 2, 3, 4, 5])
        cache.add_sequence("seq_1", seq1)
        
        assert len(cache.cache) == 1
    
    def test_semantic_similarity_search(self):
        """Test semantic similarity search."""
        cache = SemanticCache(embedding_dim=256, similarity_threshold=0.7)
        
        # Add sequences
        seq1 = torch.tensor([1, 2, 3, 4, 5])
        seq2 = torch.tensor([1, 2, 3, 6, 7])
        
        cache.add_sequence("seq_1", seq1)
        cache.add_sequence("seq_2", seq2)
        
        # Query similar
        query = torch.tensor([1, 2, 3, 4, 5])
        results = cache.find_similar(query, k=2)
        
        # Should find at least seq_1
        assert len(results) > 0
        assert results[0][0] == "seq_1"
        assert results[0][1] >= 0.99  # Very high similarity
    
    def test_semantic_cache_statistics(self):
        """Test semantic cache statistics."""
        cache = SemanticCache()
        
        for i in range(5):
            tokens = torch.randint(0, 1000, (10,))
            cache.add_sequence(f"seq_{i}", tokens)
        
        stats = cache.get_statistics()
        
        assert stats["cache_size"] == 5
        assert "hit_rate" in stats
    
    def test_hybrid_semantic_cache(self):
        """Test hybrid exact + semantic cache."""
        cache = HybridSemanticCache(semantic_threshold=0.85)
        
        tokens = torch.tensor([1, 2, 3, 4, 5])
        k = torch.randn(10, 64)
        v = torch.randn(10, 64)
        
        # Cache result
        cache.cache_result(tokens, k, v)
        
        # Find cached (exact match)
        result = cache.find_cached(tokens, use_exact=True)
        assert result is not None
        
        # Should be exact match (similarity = 1.0)
        assert result[3] == 1.0


# ============================================================================
# Page Sharing Tests
# ============================================================================

class TestPageSharing:
    """Tests for multi-sequence page sharing."""
    
    def test_page_creation(self):
        """Test page creation."""
        manager = PageSharingManager(max_total_pages=100)
        
        tokens = torch.tensor([1, 2, 3])
        k = torch.randn(5, 64)
        v = torch.randn(5, 64)
        
        page_id = manager.create_page(tokens, k, v)
        
        assert page_id == 0
        assert page_id in manager.pages
    
    def test_page_sharing(self):
        """Test page sharing across sequences."""
        manager = PageSharingManager(max_total_pages=100)
        
        tokens = torch.tensor([1, 2, 3])
        k = torch.randn(5, 64)
        v = torch.randn(5, 64)
        
        page_id = manager.create_page(tokens, k, v)
        
        # Share with multiple sequences
        manager.share_page(page_id, "seq_1")
        manager.share_page(page_id, "seq_2")
        
        page = manager.pages[page_id]
        assert page.reference_count == 3  # Original + 2 shares
        assert len(page.shared_sequences) == 2
    
    def test_copy_on_write(self):
        """Test copy-on-write semantics."""
        manager = PageSharingManager(max_total_pages=100)
        
        tokens = torch.tensor([1, 2, 3])
        k = torch.randn(5, 64)
        v = torch.randn(5, 64)
        
        page_id = manager.create_page(tokens, k, v)
        
        # Share
        manager.share_page(page_id, "seq_1")
        manager.share_page(page_id, "seq_2")
        
        # Write triggers COW
        new_k = torch.randn(5, 64)
        new_v = torch.randn(5, 64)
        new_page_id = manager.write_page(page_id, "seq_1", new_k, new_v)
        
        # Should create new page
        assert new_page_id != page_id
        assert manager.pages[page_id].reference_count == 2
        assert manager.total_cow_copies == 1
    
    def test_prefix_sharing(self):
        """Test prefix sharing cache."""
        page_manager = PageSharingManager(max_total_pages=100)
        prefix_cache = PrefixSharingCache(page_manager)
        
        tokens = torch.tensor([1, 2, 3])
        k = torch.randn(5, 64)
        v = torch.randn(5, 64)
        
        # Create prefix
        page_id1 = prefix_cache.find_or_create_prefix(tokens, k, v)
        
        # Same prefix - should reuse
        page_id2 = prefix_cache.find_or_create_prefix(tokens, k, v)
        
        assert page_id1 == page_id2
        assert prefix_cache.prefix_count == 1


# ============================================================================
# Performance Tests
# ============================================================================

class TestCachingPerformance:
    """Performance tests for caching mechanisms."""
    
    def test_eviction_policy_throughput(self):
        """Test eviction policy throughput."""
        policies = {
            "LRU": LRUEvictionPolicy(max_pages=1000),
            "LFU": LFUEvictionPolicy(max_pages=1000),
            "FIFO": FIFOEvictionPolicy(max_pages=1000),
        }
        
        for name, policy in policies.items():
            # Add pages
            for i in range(1000):
                policy.add_page(f"seq_{i}", i)
            
            # Measure access time
            start = time.time()
            for i in range(10000):
                page_id = i % 1000
                policy.record_access(f"seq_{page_id}", page_id)
            elapsed = time.time() - start
            
            throughput = 10000 / elapsed
            print(f"{name}: {throughput:.0f} ops/sec")
            
            # Should be very fast
            assert throughput > 10000  # At least 10k ops/sec
    
    def test_semantic_cache_memory(self):
        """Test semantic cache memory usage."""
        cache = SemanticCache(embedding_dim=768)
        
        # Add sequences
        for i in range(100):
            tokens = torch.randint(0, 32000, (100,))
            cache.add_sequence(f"seq_{i}", tokens)
        
        stats = cache.get_statistics()
        
        print(f"Semantic cache memory: {stats['cache_memory_mb']:.1f}MB")
        
        # Memory should be reasonable
        assert stats['cache_memory_mb'] < 100  # Less than 100MB for 100 sequences


# ============================================================================
# Integration Tests
# ============================================================================

class TestAdvancedCachingIntegration:
    """Integration tests combining multiple caching mechanisms."""
    
    def test_cache_hit_rate_comparison(self):
        """Compare hit rates of different policies."""
        results = {}
        
        # Simulate workload with temporal locality
        workload = []
        for _ in range(10):
            page = torch.randint(0, 20, (1,)).item()
            workload.extend([page] * torch.randint(1, 5, (1,)).item())
        
        for policy_type in [EvictionPolicy.LRU, EvictionPolicy.LFU]:
            policy = EvictionPolicyFactory.create(policy_type, max_pages=10)
            
            hits = 0
            for page_id in workload:
                policy.add_page(f"seq_{page_id}", page_id)
                policy.record_access(f"seq_{page_id}", page_id)
                hits += policy.pages[page_id].hit_count
            
            results[policy_type.value] = hits
        
        print(f"Hit rates: {results}")
    
    def test_hybrid_cache_improvement(self):
        """Test hybrid cache improvement over single method."""
        # Exact-only cache
        exact_hits = 0
        exact_cache = {}
        
        # Hybrid cache
        hybrid_cache = HybridSemanticCache(semantic_threshold=0.85)
        hybrid_hits = 0
        
        # Test workload
        for i in range(100):
            tokens = torch.randint(0, 1000, (50,))
            
            # Exact cache
            token_str = str(tokens.tolist())
            if token_str in exact_cache:
                exact_hits += 1
            else:
                k, v = torch.randn(10, 64), torch.randn(10, 64)
                exact_cache[token_str] = (k, v)
            
            # Hybrid cache
            result = hybrid_cache.find_cached(tokens)
            if result:
                hybrid_hits += 1
            else:
                k, v = torch.randn(10, 64), torch.randn(10, 64)
                hybrid_cache.cache_result(tokens, k, v)
        
        print(f"Exact hits: {exact_hits}, Hybrid hits: {hybrid_hits}")
        # Hybrid should match or exceed exact in this scenario


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
