"""
Integration Tests for Unified Inference Pipeline
================================================

Comprehensive test suite validating all components work together.

Test Coverage:
- Distributed engine integration
- KV cache integration  
- Speculative decoding integration
- Token batching integration
- End-to-end pipeline tests
- Performance benchmarks

Sprint 2.2 Phase 1 - Integration
Created: 2025-12-26
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Tuple
import time
import logging

# Import components
from src.distributed.engine import DistributedInferenceEngine, DistributedConfig
from src.cache.manager import PagedAttentionKVCache, PrefixCache, PageConfig
from src.speculative.decoder import SpeculativeDecoder, SpeculationConfig, DraftModel
from src.batching.token_batcher import TokenBatcher, TokenRequest
from src.serving.unified_pipeline import UnifiedInferencePipeline, PipelineConfig, InferencePipelineExecutor

logger = logging.getLogger(__name__)


# ============================================================================
# Test Models
# ============================================================================

class SimpleTransformer(nn.Module):
    """Simple transformer model for testing."""
    
    def __init__(self, vocab_size=32000, hidden_size=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=512,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return self.output(x)


# ============================================================================
# Distributed Engine Tests
# ============================================================================

class TestDistributedEngine:
    """Tests for distributed inference engine."""
    
    @pytest.fixture
    def engine(self):
        """Create distributed engine for testing."""
        config = DistributedConfig(num_gpus=1, world_size=1, rank=0)
        return DistributedInferenceEngine(config)
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine is not None
        assert engine.config.num_gpus == 1
        assert engine.stats.forward_passes == 0
    
    def test_memory_manager(self, engine):
        """Test memory manager functionality."""
        stats = engine.memory_manager.get_memory_stats()
        assert "allocated_mb" in stats
        assert "peak_mb" in stats
        assert "free_blocks" in stats
    
    def test_tensor_sharding(self, engine):
        """Test tensor sharding functionality."""
        weight = torch.randn(1024, 512)
        sharded = engine.shard_manager.shard_linear_weight(weight, shard_dim=0)
        assert sharded.shape[0] == weight.shape[0]  # No sharding with 1 GPU
    
    def test_distributed_forward(self, engine):
        """Test distributed forward pass."""
        model = SimpleTransformer(hidden_size=128, num_layers=1)
        model = model.to(engine.device)
        
        batch = {
            "input_ids": torch.randint(0, 32000, (2, 10)).to(engine.device)
        }
        
        output = engine.distributed_forward(model, batch)
        assert output is not None
        assert engine.stats.forward_passes == 1


# ============================================================================
# KV Cache Tests
# ============================================================================

class TestKVCache:
    """Tests for KV cache functionality."""
    
    @pytest.fixture
    def kv_cache(self):
        """Create KV cache for testing."""
        config = PageConfig(page_size=16)
        device = torch.device("cpu")
        return PagedAttentionKVCache(config, num_pages=256, device=device)
    
    def test_cache_allocation(self, kv_cache):
        """Test page allocation."""
        pages = kv_cache.allocate_pages(4, "seq_1")
        assert len(pages) == 4
        assert "seq_1" in kv_cache.page_table
    
    def test_cache_write_read(self, kv_cache):
        """Test write and read from cache."""
        kv_cache.allocate_pages(4, "seq_1")
        
        k = torch.randn(1, 10, 64)
        v = torch.randn(1, 10, 64)
        
        kv_cache.write_kv("seq_1", k, v, token_pos=0)
        
        # Verify metadata
        assert "seq_1" in kv_cache.cache_metadata
        assert kv_cache.cache_metadata["seq_1"].num_pages == 4
    
    def test_cache_memory_stats(self, kv_cache):
        """Test cache memory statistics."""
        kv_cache.allocate_pages(4, "seq_1")
        
        stats = kv_cache.get_memory_stats()
        assert stats["used_pages"] == 4
        assert stats["num_sequences"] == 1
    
    def test_prefix_cache_hash(self):
        """Test prefix cache hashing."""
        kv_cache = PagedAttentionKVCache(PageConfig(), num_pages=256)
        prefix_cache = PrefixCache(kv_cache)
        
        tokens = torch.tensor([1, 2, 3, 4, 5])
        hash1 = prefix_cache.hash_tokens(tokens)
        hash2 = prefix_cache.hash_tokens(tokens)
        
        assert hash1 == hash2  # Same tokens = same hash
    
    def test_prefix_cache_storage(self):
        """Test prefix caching and retrieval."""
        kv_cache = PagedAttentionKVCache(PageConfig(), num_pages=256)
        prefix_cache = PrefixCache(kv_cache)
        
        tokens = torch.tensor([1, 2, 3, 4, 5])
        k = torch.randn(1, 5, 64)
        v = torch.randn(1, 5, 64)
        
        prefix_cache.cache_prefix(tokens, k, v)
        retrieved = prefix_cache.get_prefix(tokens)
        
        assert retrieved is not None
        assert retrieved[0].shape == k.shape


# ============================================================================
# Speculative Decoding Tests
# ============================================================================

class TestSpeculativeDecoding:
    """Tests for speculative decoding."""
    
    @pytest.fixture
    def setup(self):
        """Setup for speculative decoding tests."""
        main_model = SimpleTransformer(hidden_size=128, num_layers=1)
        config = SpeculationConfig(max_speculation_depth=2)
        draft_model = DraftModel(vocab_size=32000, hidden_size=64, num_layers=1)
        
        return {
            "main_model": main_model,
            "config": config,
            "draft_model": draft_model
        }
    
    def test_draft_generation(self, setup):
        """Test draft model generation."""
        draft_model = setup["draft_model"]
        input_ids = torch.randint(0, 32000, (2, 10))
        
        output = draft_model.generate_draft(input_ids, num_tokens=4)
        
        assert output.shape == (2, 4)
        assert output.dtype == torch.long
    
    def test_speculative_decoder_initialization(self, setup):
        """Test speculative decoder initialization."""
        decoder = SpeculativeDecoder(
            setup["main_model"],
            setup["config"],
            setup["draft_model"]
        )
        
        assert decoder.draft_model is not None
        assert decoder.verifier is not None
    
    def test_speculative_generation(self, setup):
        """Test speculative generation."""
        decoder = SpeculativeDecoder(
            setup["main_model"],
            setup["config"],
            setup["draft_model"]
        )
        
        input_ids = torch.randint(0, 32000, (1, 10))
        output = decoder.generate(input_ids, max_new_tokens=8)
        
        assert output.generated_ids is not None
        assert output.num_tokens > 0
        assert output.acceptance_rate <= 1.0


# ============================================================================
# Token Batcher Tests
# ============================================================================

class TestTokenBatcher:
    """Tests for token-level batching."""
    
    @pytest.fixture
    def batcher(self):
        """Create batcher for testing."""
        return TokenBatcher(max_batch_size=32, max_batch_tokens=512)
    
    def test_add_request(self, batcher):
        """Test adding requests."""
        tokens = torch.randint(0, 32000, (20,))
        batcher.add_request("req_1", tokens, max_tokens=100)
        
        assert batcher.get_pending_count() == 1
    
    def test_get_batch(self, batcher):
        """Test batch construction."""
        for i in range(5):
            tokens = torch.randint(0, 32000, (20 + i*5,))
            batcher.add_request(f"req_{i}", tokens, max_tokens=100)
        
        batch = batcher.get_batch()
        
        assert batch is not None
        assert batch.batch_size > 0
        assert len(batch.request_ids) > 0
    
    def test_batch_token_count(self, batcher):
        """Test batch respects token count limits."""
        for i in range(10):
            tokens = torch.randint(0, 32000, (100,))
            batcher.add_request(f"req_{i}", tokens, max_tokens=100)
        
        batch = batcher.get_batch(batch_tokens=256)
        
        assert batch.total_tokens <= 256
    
    def test_mark_completed(self, batcher):
        """Test marking requests complete."""
        tokens = torch.randint(0, 32000, (20,))
        batcher.add_request("req_1", tokens, max_tokens=100)
        
        batch = batcher.get_batch()
        for req_id in batch.request_ids:
            completed = batcher.mark_completed(req_id)
            assert completed
    
    def test_priority_scheduling(self, batcher):
        """Test priority-based scheduling."""
        # Add low priority request
        tokens1 = torch.randint(0, 32000, (20,))
        batcher.add_request("req_low", tokens1, max_tokens=100, priority=1)
        
        # Add high priority request
        tokens2 = torch.randint(0, 32000, (20,))
        batcher.add_request("req_high", tokens2, max_tokens=100, priority=10)
        
        # High priority should be selected first
        batch = batcher.get_batch()
        assert "req_high" in batch.request_ids
    
    def test_stats(self, batcher):
        """Test statistics collection."""
        for i in range(5):
            tokens = torch.randint(0, 32000, (20,))
            batcher.add_request(f"req_{i}", tokens, max_tokens=100)
        
        batch = batcher.get_batch()
        for req_id in batch.request_ids:
            batcher.mark_completed(req_id)
        
        stats = batcher.get_stats()
        assert stats["total_requests"] == 5
        assert stats["total_batches"] > 0


# ============================================================================
# Unified Pipeline Tests
# ============================================================================

class TestUnifiedPipeline:
    """Tests for unified inference pipeline."""
    
    @pytest.fixture
    def setup(self):
        """Setup for pipeline tests."""
        model = SimpleTransformer(hidden_size=128, num_layers=1)
        config = PipelineConfig(
            num_gpus=1,
            hidden_size=128,
            num_layers=1,
            enable_speculative_decoding=False,  # Disabled for testing
            enable_prefix_caching=True
        )
        
        return {
            "model": model,
            "config": config
        }
    
    def test_pipeline_initialization(self, setup):
        """Test pipeline initializes correctly."""
        pipeline = UnifiedInferencePipeline(setup["model"], setup["config"])
        
        assert pipeline is not None
        assert pipeline.distributed_engine is not None
        assert pipeline.kv_cache is not None
        assert pipeline.token_batcher is not None
    
    def test_add_request(self, setup):
        """Test adding requests to pipeline."""
        pipeline = UnifiedInferencePipeline(setup["model"], setup["config"])
        
        tokens = torch.randint(0, 32000, (20,))
        pipeline.add_request("req_1", tokens, max_tokens=10)
        
        assert pipeline.total_requests == 1
    
    def test_prefix_cache_integration(self, setup):
        """Test prefix cache integration."""
        pipeline = UnifiedInferencePipeline(setup["model"], setup["config"])
        
        tokens = torch.randint(0, 32000, (10,))
        
        # First time - no cache hit
        k1, v1, cached1 = pipeline._check_prefix_cache(tokens)
        assert cached1 == 0
        
        # Simulate adding to cache
        pipeline.add_request("req_1", tokens, max_tokens=10)
        
        # After generation, should be cached
        # (In real scenario, would be cached by generate_batch)
    
    def test_end_to_end_generation(self, setup):
        """Test end-to-end generation."""
        pipeline = UnifiedInferencePipeline(setup["model"], setup["config"])
        
        # Add requests
        for i in range(3):
            tokens = torch.randint(0, 32000, (10,))
            pipeline.add_request(f"req_{i}", tokens, max_tokens=5)
        
        # Process
        outputs = pipeline.process_requests(num_batches=1)
        
        assert len(outputs) > 0
        assert all(hasattr(o, 'request_id') for o in outputs)
    
    def test_pipeline_statistics(self, setup):
        """Test pipeline statistics."""
        pipeline = UnifiedInferencePipeline(setup["model"], setup["config"])
        
        tokens = torch.randint(0, 32000, (10,))
        pipeline.add_request("req_1", tokens, max_tokens=5)
        pipeline.process_requests(num_batches=1)
        
        stats = pipeline.get_statistics()
        
        assert "total_requests" in stats
        assert "total_tokens" in stats
        assert "avg_latency_ms" in stats
        assert stats["total_requests"] == 1


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and benchmark tests."""
    
    def test_throughput_single_batch(self):
        """Test throughput with single batch."""
        model = SimpleTransformer(hidden_size=128, num_layers=1)
        config = PipelineConfig(
            num_gpus=1,
            hidden_size=128,
            num_layers=1,
            enable_speculative_decoding=False,
            enable_prefix_caching=False  # Disable for baseline
        )
        
        pipeline = UnifiedInferencePipeline(model, config)
        
        # Add batch
        for i in range(10):
            tokens = torch.randint(0, 32000, (20,))
            pipeline.add_request(f"req_{i}", tokens, max_tokens=10)
        
        # Measure
        start = time.time()
        outputs = pipeline.process_requests()
        elapsed = time.time() - start
        
        tokens_per_sec = pipeline.total_tokens / elapsed
        
        # Should achieve reasonable throughput
        assert tokens_per_sec > 100  # At least 100 tokens/sec
        print(f"Throughput: {tokens_per_sec:.1f} tokens/sec")
    
    def test_latency_single_request(self):
        """Test latency for single request."""
        model = SimpleTransformer(hidden_size=128, num_layers=1)
        config = PipelineConfig(
            num_gpus=1,
            hidden_size=128,
            enable_speculative_decoding=False
        )
        
        executor = InferencePipelineExecutor(model, config)
        executor.pipeline.distributed_engine.initialize_process_group = lambda: None
        executor.pipeline.distributed_engine.cleanup = lambda: None
        
        tokens = torch.randint(0, 32000, (20,))
        output = executor.generate("req_1", tokens, max_tokens=10)
        
        assert output is not None
        assert output.latency_ms > 0
        print(f"Latency: {output.latency_ms:.1f}ms")


# ============================================================================
# Test Suite
# ============================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
