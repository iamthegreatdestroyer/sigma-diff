"""
Multi-Modal Inference Test Suite
================================

Comprehensive tests for Sprint 2.1 multi-modal components.

Test Coverage:
- Vision Encoder (VisionEncoder, preprocessing, caching)
- Fusion Layer (cross-attention, gated, perceiver strategies)
- Modality Router (detection, routing, batching)
- Adaptive Batcher (dynamic sizing, continuous batching)
- Pipeline Integration (end-to-end inference)

Sprint 2.1 - Multi-Modal Inference
Created: 2025-12-26
"""

import pytest
import torch
import torch.nn as nn
import time
from typing import List, Dict, Any

# Import components under test
import sys
sys.path.insert(0, str(__file__).rsplit('tests', 1)[0] + 'src')

from inference.multimodal.vision_encoder import (
    VisionEncoder, VisionEncoderConfig, VisionEncoderType,
    ImagePreprocessor, PatchEmbedding, VisionAttention
)
from inference.multimodal.fusion_layer import (
    CrossModalFusionLayer, FusionConfig, FusionStrategy,
    CrossModalAttention, FusionInput
)
from inference.multimodal.modality_router import (
    ModalityRouter, RouterConfig, Modality,
    ModalityDetector, EncoderRegistry
)
from inference.multimodal.adaptive_batcher import (
    AdaptiveBatcher, ContinuousBatcher, BatcherConfig,
    SequenceBucketing, MemoryEstimator
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def vision_config():
    """Create vision encoder config for testing."""
    return VisionEncoderConfig(
        encoder_type=VisionEncoderType.CLIP,
        image_size=224,
        patch_size=14,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=4,  # Reduced for faster testing
        intermediate_size=3072
    )


@pytest.fixture
def fusion_config():
    """Create fusion layer config for testing."""
    return FusionConfig(
        fusion_strategy=FusionStrategy.CROSS_ATTENTION,
        hidden_size=512,
        vision_hidden_size=768,
        text_hidden_size=1024,
        num_fusion_layers=2,
        num_attention_heads=8
    )


@pytest.fixture
def batcher_config():
    """Create batcher config for testing."""
    return BatcherConfig(
        max_batch_size=8,
        max_wait_time_ms=50.0,
        enable_continuous_batching=True
    )


# ============================================================================
# Vision Encoder Tests
# ============================================================================

class TestVisionEncoder:
    """Tests for VisionEncoder component."""
    
    def test_vision_encoder_init(self, vision_config, device):
        """Test vision encoder initialization."""
        encoder = VisionEncoder(vision_config).to(device)
        
        assert encoder is not None
        assert len(encoder.encoder_blocks) == vision_config.num_hidden_layers
    
    def test_vision_encoder_forward(self, vision_config, device):
        """Test vision encoder forward pass."""
        encoder = VisionEncoder(vision_config).to(device)
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        
        output = encoder(pixel_values=images)
        
        # Check output shapes
        expected_patches = (224 // 14) ** 2 + 1  # +1 for class token
        assert output.embeddings.shape == (batch_size, expected_patches, 768)
        assert output.pooled_output.shape == (batch_size, 768)
    
    def test_vision_encoder_batched_encoding(self, vision_config, device):
        """Test batched image encoding."""
        encoder = VisionEncoder(vision_config).to(device)
        
        images = torch.randn(16, 3, 224, 224).to(device)
        
        embeddings = encoder.encode_images(images, batch_size=4)
        
        assert embeddings.shape == (16, 768)
    
    def test_vision_encoder_caching(self, vision_config, device):
        """Test embedding caching."""
        vision_config.enable_caching = True
        encoder = VisionEncoder(vision_config).to(device)
        
        images = [torch.randn(3, 224, 224).to(device) for _ in range(4)]
        
        # First pass - should compute
        output1 = encoder(images=images)
        
        # Second pass with same images - should use cache
        output2 = encoder(images=images)
        
        # Cache should have entries
        assert len(encoder.embedding_cache) > 0
        
        # Clear cache
        encoder.clear_cache()
        assert len(encoder.embedding_cache) == 0
    
    def test_patch_embedding(self, device):
        """Test patch embedding module."""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=14,
            hidden_size=768
        ).to(device)
        
        images = torch.randn(2, 3, 224, 224).to(device)
        embeddings = patch_embed(images)
        
        # Should have 256 patches + 1 class token
        assert embeddings.shape == (2, 257, 768)
    
    def test_vision_attention(self, device):
        """Test vision attention module."""
        attention = VisionAttention(
            hidden_size=768,
            num_attention_heads=12,
            use_flash_attention=True
        ).to(device)
        
        hidden_states = torch.randn(2, 100, 768).to(device)
        output, _ = attention(hidden_states)
        
        assert output.shape == hidden_states.shape


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""
    
    def test_preprocessor_tensor_input(self, device):
        """Test preprocessing tensor input."""
        preprocessor = ImagePreprocessor(image_size=224, device=device)
        
        images = torch.randn(4, 3, 256, 256)
        result = preprocessor.preprocess(images)
        
        assert result.pixel_values.shape == (4, 3, 256, 256)
    
    def test_preprocessor_normalization(self, device):
        """Test image normalization."""
        preprocessor = ImagePreprocessor(image_size=224, device=device)
        
        # Create image with known values
        images = torch.ones(1, 3, 224, 224) * 0.5
        result = preprocessor.preprocess(images)
        
        # Check normalization was applied
        assert not torch.allclose(result.pixel_values, images.to(device))


# ============================================================================
# Fusion Layer Tests
# ============================================================================

class TestFusionLayer:
    """Tests for CrossModalFusionLayer component."""
    
    def test_fusion_layer_init(self, fusion_config, device):
        """Test fusion layer initialization."""
        fusion = CrossModalFusionLayer(fusion_config).to(device)
        
        assert fusion is not None
        assert fusion.config.fusion_strategy == FusionStrategy.CROSS_ATTENTION
    
    def test_cross_attention_fusion(self, fusion_config, device):
        """Test cross-attention fusion."""
        fusion = CrossModalFusionLayer(fusion_config).to(device)
        
        batch_size = 4
        vision_features = torch.randn(batch_size, 196, 768).to(device)
        text_features = torch.randn(batch_size, 32, 1024).to(device)
        
        output = fusion.fuse(vision_features, text_features)
        
        assert output.shape == (batch_size, 512)
    
    def test_gated_fusion(self, device):
        """Test gated fusion strategy."""
        config = FusionConfig(
            fusion_strategy=FusionStrategy.GATED,
            hidden_size=512,
            vision_hidden_size=768,
            text_hidden_size=1024
        )
        fusion = CrossModalFusionLayer(config).to(device)
        
        vision = torch.randn(4, 196, 768).to(device)
        text = torch.randn(4, 32, 1024).to(device)
        
        output = fusion.fuse(vision, text)
        
        assert output.shape == (4, 512)
    
    def test_perceiver_fusion(self, device):
        """Test perceiver-style fusion."""
        config = FusionConfig(
            fusion_strategy=FusionStrategy.PERCEIVER,
            hidden_size=512,
            vision_hidden_size=768,
            text_hidden_size=1024,
            num_latent_tokens=32
        )
        fusion = CrossModalFusionLayer(config).to(device)
        
        vision = torch.randn(4, 196, 768).to(device)
        text = torch.randn(4, 32, 1024).to(device)
        
        output = fusion.fuse(vision, text)
        
        assert output.shape == (4, 512)
    
    def test_early_fusion(self, device):
        """Test early fusion strategy."""
        config = FusionConfig(
            fusion_strategy=FusionStrategy.EARLY,
            hidden_size=512,
            vision_hidden_size=768,
            text_hidden_size=1024
        )
        fusion = CrossModalFusionLayer(config).to(device)
        
        vision = torch.randn(4, 196, 768).to(device)
        text = torch.randn(4, 32, 1024).to(device)
        
        output = fusion.fuse(vision, text)
        
        assert output.shape == (4, 512)
    
    def test_cross_modal_attention(self, device):
        """Test cross-modal attention module."""
        attention = CrossModalAttention(
            hidden_size=512,
            num_attention_heads=8
        ).to(device)
        
        query = torch.randn(4, 32, 512).to(device)
        key_value = torch.randn(4, 64, 512).to(device)
        
        output, _ = attention(query, key_value)
        
        assert output.shape == (4, 32, 512)


# ============================================================================
# Modality Router Tests
# ============================================================================

class TestModalityRouter:
    """Tests for ModalityRouter component."""
    
    def test_router_init(self):
        """Test router initialization."""
        config = RouterConfig(max_batch_size=16)
        router = ModalityRouter(config)
        
        assert router is not None
        assert router.config.max_batch_size == 16
    
    def test_modality_detection_tensor(self):
        """Test modality detection from tensor shapes."""
        detector = ModalityDetector()
        
        # 4D tensor (images)
        image_tensor = torch.randn(4, 3, 224, 224)
        assert detector.detect(image_tensor) == Modality.IMAGE
        
        # 2D tensor (text embeddings)
        text_tensor = torch.randn(32, 768)
        assert detector.detect(text_tensor) == Modality.TEXT
    
    def test_modality_detection_string(self):
        """Test modality detection from strings."""
        detector = ModalityDetector()
        
        assert detector.detect("image.jpg") == Modality.IMAGE
        assert detector.detect("photo.png") == Modality.IMAGE
        assert detector.detect("audio.mp3") == Modality.AUDIO
        assert detector.detect("video.mp4") == Modality.VIDEO
        assert detector.detect("Hello world") == Modality.TEXT
    
    def test_encoder_registry(self):
        """Test encoder registry."""
        registry = EncoderRegistry()
        
        def dummy_encoder(x):
            return torch.randn(len(x), 768)
        
        registry.register(Modality.IMAGE, dummy_encoder, {"model": "clip"})
        
        assert registry.has_encoder(Modality.IMAGE)
        assert not registry.has_encoder(Modality.AUDIO)
        assert registry.get_config(Modality.IMAGE) == {"model": "clip"}
    
    def test_routing(self):
        """Test input routing."""
        config = RouterConfig(max_batch_size=4)
        router = ModalityRouter(config)
        
        # Register dummy encoders
        router.register_encoder(Modality.IMAGE, lambda x: torch.randn(len(x), 768))
        router.register_encoder(Modality.TEXT, lambda x: torch.randn(len(x), 1024))
        
        # Mixed inputs
        inputs = [
            "Hello world",
            "image.jpg",
            "Another text",
            "photo.png"
        ]
        
        routed = router.route(inputs)
        
        assert len(routed) == 2  # One for images, one for text
    
    def test_router_stats(self):
        """Test router statistics."""
        config = RouterConfig()
        router = ModalityRouter(config)
        router.register_encoder(Modality.TEXT, lambda x: x)
        
        router.route(["text1", "text2", "text3"])
        
        stats = router.get_stats()
        assert stats.total_requests == 3
        assert "TEXT" in stats.requests_by_modality


# ============================================================================
# Adaptive Batcher Tests
# ============================================================================

class TestAdaptiveBatcher:
    """Tests for AdaptiveBatcher component."""
    
    def test_batcher_init(self, batcher_config):
        """Test batcher initialization."""
        batcher = AdaptiveBatcher(batcher_config)
        
        assert batcher is not None
        assert batcher.config.max_batch_size == 8
    
    def test_add_request(self, batcher_config):
        """Test adding requests."""
        batcher = AdaptiveBatcher(batcher_config)
        
        req_id = batcher.add_request(
            data="test data",
            modality="text",
            sequence_length=100
        )
        
        assert req_id is not None
        assert batcher.pending_count() == 1
    
    def test_get_batch(self, batcher_config):
        """Test getting batches."""
        batcher = AdaptiveBatcher(batcher_config)
        
        # Add requests
        for i in range(10):
            batcher.add_request(
                data=f"data_{i}",
                modality="text",
                sequence_length=50 + i * 10
            )
        
        batch = batcher.get_batch("text", max_size=4)
        
        assert batch is not None
        assert batch.size == 4
        assert batch.modality == "text"
    
    def test_sequence_bucketing(self):
        """Test sequence length bucketing."""
        buckets = SequenceBucketing(num_buckets=8, max_length=4096)
        
        from inference.multimodal.adaptive_batcher import BatchRequest
        
        requests = [
            BatchRequest("r1", "data1", "text", 50, 0),
            BatchRequest("r2", "data2", "text", 100, 0),
            BatchRequest("r3", "data3", "text", 200, 0),
        ]
        
        for req in requests:
            buckets.add(req)
        
        assert buckets.total_pending() == 3
        
        # Get batch from bucket
        batch = buckets.get_batch(2)
        assert len(batch) == 2
    
    def test_memory_estimator(self):
        """Test memory estimation."""
        estimator = MemoryEstimator(gpu_memory_fraction=0.8)
        
        # Estimate memory for batch
        memory = estimator.estimate_batch_memory(
            batch_size=16,
            sequence_length=512,
            hidden_size=4096,
            num_layers=32
        )
        
        assert memory > 0
        
        # Get max batch size
        max_batch = estimator.get_max_batch_size(512)
        assert max_batch >= 1
    
    def test_continuous_batcher(self, batcher_config):
        """Test continuous batching."""
        batcher = ContinuousBatcher(batcher_config)
        
        # Add requests
        for i in range(12):
            batcher.add_request(
                data=f"data_{i}",
                modality="text",
                sequence_length=100
            )
        
        # Start batch
        batch = batcher.start_batch("text")
        assert batch is not None
        
        # Extend batch
        extended = batcher.extend_batch(batch.batch_id, max_add=4)
        assert extended.size >= batch.size
        
        # Complete request
        if batch.requests:
            req_id = batch.requests[0].request_id
            batcher.complete_request(batch.batch_id, req_id, "result")
    
    def test_priority_scheduling(self, batcher_config):
        """Test priority-based scheduling."""
        batcher_config.priority_scheduling = True
        batcher = AdaptiveBatcher(batcher_config)
        
        # Add requests with different priorities
        batcher.add_request("low", "text", 100, priority=0)
        batcher.add_request("high", "text", 100, priority=10)
        batcher.add_request("medium", "text", 100, priority=5)
        
        # Get batch - should prioritize high priority
        batch = batcher.get_batch("text", max_size=3)
        
        # Check that higher priority requests come first
        priorities = [r.priority for r in batch.requests]
        assert priorities == sorted(priorities, reverse=True)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_vision_to_fusion(self, vision_config, fusion_config, device):
        """Test vision encoder to fusion layer integration."""
        # Create components
        vision_encoder = VisionEncoder(vision_config).to(device)
        fusion_layer = CrossModalFusionLayer(fusion_config).to(device)
        
        # Create inputs
        images = torch.randn(4, 3, 224, 224).to(device)
        text_features = torch.randn(4, 32, 1024).to(device)
        
        # Encode images
        vision_output = vision_encoder(pixel_values=images)
        
        # Fuse with text
        fusion_input = FusionInput(
            vision_features=vision_output.embeddings,
            text_features=text_features
        )
        
        # Update fusion config to match vision hidden size
        fusion_config.vision_hidden_size = 768
        fusion_layer = CrossModalFusionLayer(fusion_config).to(device)
        
        output = fusion_layer(fusion_input)
        
        assert output.fused_features is not None
    
    def test_router_with_batcher(self, device):
        """Test router and batcher integration."""
        # Create components
        router_config = RouterConfig(max_batch_size=8)
        router = ModalityRouter(router_config)
        
        batcher_config = BatcherConfig(max_batch_size=8)
        batcher = AdaptiveBatcher(batcher_config)
        
        # Register encoder that uses batcher
        def batched_encoder(inputs):
            for inp in inputs:
                batcher.add_request(inp, "image", 196)
            batch = batcher.get_batch("image")
            return torch.randn(batch.size, 768) if batch else None
        
        router.register_encoder(Modality.IMAGE, batched_encoder)
        
        # Route inputs
        inputs = ["img1.jpg", "img2.png", "img3.jpg"]
        result = router.process(inputs, Modality.IMAGE)
        
        assert Modality.IMAGE in result


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance benchmarks for multi-modal components."""
    
    def test_vision_encoder_throughput(self, vision_config, device):
        """Benchmark vision encoder throughput."""
        encoder = VisionEncoder(vision_config).to(device)
        encoder.eval()
        
        batch_size = 16
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                encoder(pixel_values=images)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        num_iterations = 10
        
        for _ in range(num_iterations):
            with torch.no_grad():
                encoder(pixel_values=images)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        throughput = (batch_size * num_iterations) / elapsed
        
        print(f"\nVision Encoder Throughput: {throughput:.1f} images/sec")
        assert throughput > 10  # Minimum expected throughput
    
    def test_fusion_layer_latency(self, fusion_config, device):
        """Benchmark fusion layer latency."""
        fusion = CrossModalFusionLayer(fusion_config).to(device)
        fusion.eval()
        
        vision = torch.randn(4, 196, 768).to(device)
        text = torch.randn(4, 32, 1024).to(device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                fusion.fuse(vision, text)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(20):
            start = time.time()
            with torch.no_grad():
                fusion.fuse(vision, text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"\nFusion Layer Latency: {avg_latency:.2f} ms")
        assert avg_latency < 100  # Max expected latency
    
    def test_batcher_overhead(self, batcher_config):
        """Benchmark batcher overhead."""
        batcher = AdaptiveBatcher(batcher_config)
        
        num_requests = 1000
        
        start = time.time()
        for i in range(num_requests):
            batcher.add_request(f"data_{i}", "text", 100)
        elapsed = time.time() - start
        
        overhead_per_request = (elapsed / num_requests) * 1000000  # microseconds
        
        print(f"\nBatcher Overhead: {overhead_per_request:.2f} µs/request")
        assert overhead_per_request < 100  # Max 100 microseconds per request


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
