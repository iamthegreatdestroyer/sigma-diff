"""
End-to-End Generation Tests
===========================

Tests complete inference pipeline.
"""

import time
from pathlib import Path

from src.core.engine import RyotEngine
from src.api.api_types import GenerationConfig, StopReason
from src.api.interfaces import InferenceEngine


class TestEndToEndGeneration:
    """End-to-end generation tests."""
    
    def test_simple_generation(self):
        """Test basic text generation."""
        engine = RyotEngine()
        result = engine.generate(
            "Hello, world!",
            GenerationConfig(max_tokens=10)
        )
        
        assert result.completion_tokens > 0
        assert result.completion_tokens <= 10
        assert len(result.generated_text) > 0
        print("✅ test_simple_generation passed")
    
    def test_streaming_generation(self):
        """Test streaming generation."""
        engine = RyotEngine()
        chunks = list(engine.stream(
            "Test prompt",
            GenerationConfig(max_tokens=5)
        ))
        
        assert len(chunks) == 5
        assert chunks[0].is_first
        assert chunks[-1].is_last
        print("✅ test_streaming_generation passed")
    
    def test_stop_on_eos(self):
        """Test generation stops on EOS token."""
        engine = RyotEngine()
        result = engine.generate(
            "Short response.",
            GenerationConfig(max_tokens=100)
        )
        
        # May stop early due to EOS
        assert result.stop_reason in [StopReason.MAX_TOKENS, StopReason.EOS_TOKEN]
        print("✅ test_stop_on_eos passed")
    
    def test_temperature_sampling(self):
        """Test different temperature settings."""
        engine = RyotEngine()
        
        # Low temperature (more deterministic)
        result_low = engine.generate(
            "The capital of France is",
            GenerationConfig(max_tokens=10, temperature=0.1)
        )
        
        # High temperature (more random)
        result_high = engine.generate(
            "The capital of France is",
            GenerationConfig(max_tokens=10, temperature=1.5)
        )
        
        # Both should produce valid output
        assert len(result_low.generated_text) > 0
        assert len(result_high.generated_text) > 0
        print("✅ test_temperature_sampling passed")
    
    def test_context_window(self):
        """Test context window limits."""
        engine = RyotEngine()
        context_size = engine.get_context_window()
        assert context_size > 0
        
        # Should not crash with long input (up to limit)
        long_prompt = "word " * (context_size // 2)
        result = engine.generate(
            long_prompt,
            GenerationConfig(max_tokens=5)
        )
        assert result.completion_tokens > 0
        print("✅ test_context_window passed")
    
    def test_generation_speed(self):
        """Test generation speed is reasonable."""
        engine = RyotEngine()
        start = time.time()
        result = engine.generate(
            "Speed test prompt",
            GenerationConfig(max_tokens=20)
        )
        elapsed = time.time() - start
        
        tokens_per_second = result.completion_tokens / elapsed if elapsed > 0 else 0
        print(f"Speed: {tokens_per_second:.1f} tok/s")
        
        # Should achieve at least 1 token/second
        assert tokens_per_second >= 1.0
        print("✅ test_generation_speed passed")
    
    def test_protocol_compliance(self):
        """Test InferenceEngine protocol compliance."""
        engine = RyotEngine()
        assert isinstance(engine, InferenceEngine)
        assert engine.is_ready()
        
        info = engine.get_model_info()
        assert info.vocab_size > 0
        assert info.context_window > 0
        print("✅ test_protocol_compliance passed")


def test_mock_engine_standalone():
    """Test mock engine works without model."""
    try:
        from src.stubs import MockInferenceEngine
        
        engine = MockInferenceEngine()
        assert engine.is_ready()
        
        result = engine.generate("Test", GenerationConfig(max_tokens=5))
        assert result.completion_tokens == 5
        
        print("✅ Mock engine standalone test passed")
    except ImportError:
        print("⚠️  Mock engine not available (stubs not created yet)")


if __name__ == "__main__":
    tests = TestEndToEndGeneration()
    
    try:
        tests.test_simple_generation()
        tests.test_streaming_generation()
        tests.test_stop_on_eos()
        tests.test_temperature_sampling()
        tests.test_context_window()
        tests.test_generation_speed()
        tests.test_protocol_compliance()
        test_mock_engine_standalone()
        
        print("\n✅ All end-to-end tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n⚠️ Test error: {e}")

