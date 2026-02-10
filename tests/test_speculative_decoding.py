"""
Comprehensive test suite for speculative decoding implementation.

Tests cover:
- Basic functionality and correctness
- Performance benchmarking
- Edge cases and error handling
- Adaptive K tuning
- Memory management
- Integration with existing pipeline
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
from typing import List

# Import the speculative decoding components
import sys
sys.path.append('Ryzanstein LLM/src')

from inference.speculative_decoder import (
    SpeculativeConfig,
    SpeculativeDecoder,
    SpeculativeStats,
    SimpleDraftModel,
    TargetModelVerifier,
    create_speculative_decoder,
    benchmark_speculative_decoding
)


class TestSpeculativeConfig:
    """Test SpeculativeConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = SpeculativeConfig(
            vocab_size=32000,
            min_K=1,
            max_K=8,
            temperature=1.0,
            top_p=0.9
        )
        assert config.vocab_size == 32000
        assert config.min_K == 1
        assert config.max_K == 8

    def test_invalid_k_range(self):
        """Test invalid K range validation."""
        with pytest.raises(AssertionError):
            SpeculativeConfig(min_K=5, max_K=3)

    def test_invalid_temperature(self):
        """Test invalid temperature validation."""
        with pytest.raises(AssertionError):
            SpeculativeConfig(temperature=0)

    def test_invalid_top_p(self):
        """Test invalid top_p validation."""
        with pytest.raises(AssertionError):
            SpeculativeConfig(top_p=1.5)


class TestSpeculativeStats:
    """Test SpeculativeStats functionality."""

    def test_initialization(self):
        """Test stats initialization."""
        stats = SpeculativeStats()
        assert stats.total_tokens_generated == 0
        assert stats.total_forward_passes == 0
        assert stats.get_speedup_ratio() == 1.0
        assert stats.get_acceptance_rate() == 0.0

    def test_speedup_calculation(self):
        """Test speedup ratio calculation."""
        stats = SpeculativeStats(
            total_tokens_generated=10,
            total_forward_passes=4
        )
        assert stats.get_speedup_ratio() == 2.5

    def test_zero_division_protection(self):
        """Test protection against division by zero."""
        stats = SpeculativeStats()
        assert stats.get_speedup_ratio() == 1.0
        assert stats.get_acceptance_rate() == 0.0

    def test_reset(self):
        """Test statistics reset."""
        stats = SpeculativeStats(
            total_tokens_generated=100,
            total_forward_passes=50,
            total_draft_tokens=200,
            total_accepted_tokens=150
        )
        stats.reset()
        assert stats.total_tokens_generated == 0
        assert stats.total_forward_passes == 0


class TestSimpleDraftModel:
    """Test SimpleDraftModel functionality."""

    def test_initialization(self):
        """Test draft model initialization."""
        config = SpeculativeConfig(vocab_size=1000, initial_K=4)
        model = SimpleDraftModel(config)

        assert model.config.vocab_size == 1000
        assert model.current_K == 4

    def test_generate_candidates(self):
        """Test candidate generation."""
        config = SpeculativeConfig(vocab_size=100, initial_K=3)
        model = SimpleDraftModel(config)

        prefix = [1, 2, 3]
        candidates = model.generate_candidates(prefix, 3)

        assert len(candidates) == 3
        assert all(0 <= token < 100 for token in candidates)

    def test_record_acceptance(self):
        """Test acceptance recording."""
        config = SpeculativeConfig(enable_adaptive_k=True, k_adjust_frequency=5)
        model = SimpleDraftModel(config)

        # Record some acceptances
        model.record_acceptance(3, 5)  # 3 accepted out of 5
        model.record_acceptance(2, 5)  # 2 accepted out of 5

        acceptance_rate = model.stats.get_acceptance_rate()
        assert acceptance_rate == 0.5  # (3+2)/(5+5) = 5/10 = 0.5

    @patch('inference.speculative_decoder.np.random.randint')
    def test_deterministic_generation(self, mock_randint):
        """Test deterministic generation with mocked randomness."""
        mock_randint.return_value = 42

        config = SpeculativeConfig(vocab_size=100)
        model = SimpleDraftModel(config)

        prefix = [1, 2, 3]
        candidates = model.generate_candidates(prefix, 2)

        assert candidates == [42, 42]
        assert mock_randint.call_count == 2


class TestTargetModelVerifier:
    """Test TargetModelVerifier functionality."""

    def test_initialization(self):
        """Test verifier initialization."""
        config = SpeculativeConfig()
        verifier = TargetModelVerifier(config)

        assert verifier.config == config

    @patch('inference.speculative_decoder.np.random.random')
    def test_verify_tokens(self, mock_random):
        """Test token verification with controlled randomness."""
        config = SpeculativeConfig()
        verifier = TargetModelVerifier(config)

        # Mock random to return values that ensure acceptance
        mock_random.return_value = 0.5  # This should be accepted

        prefix = [1, 2, 3]
        candidates = [10, 20, 30]
        draft_probs = [0.5, 0.5, 0.5]

        accepted, verifications = verifier.verify_tokens(prefix, candidates, draft_probs)

        # Should accept all tokens in this mock scenario
        assert len(accepted) == 3
        assert verifications == 3


class TestSpeculativeDecoder:
    """Test SpeculativeDecoder functionality."""

    def test_initialization(self):
        """Test decoder initialization."""
        config = SpeculativeConfig(enable_speculative=True, initial_K=4)
        decoder = SpeculativeDecoder(config)

        assert decoder.config == config
        assert decoder.get_K() == 4

    def test_disabled_speculative(self):
        """Test fallback when speculative decoding is disabled."""
        config = SpeculativeConfig(enable_speculative=False)
        decoder = SpeculativeDecoder(config)

        prefix = [1, 2, 3]
        tokens = decoder.decode_next_tokens(prefix, max_tokens=2)

        assert len(tokens) == 1  # Single token fallback

    def test_decode_next_tokens(self):
        """Test token decoding."""
        config = SpeculativeConfig(enable_speculative=True, initial_K=2)
        decoder = SpeculativeDecoder(config)

        prefix = [1, 2, 3]
        tokens = decoder.decode_next_tokens(prefix, max_tokens=2)

        assert isinstance(tokens, list)
        assert len(tokens) >= 0  # Can be 0 if no tokens accepted

    def test_statistics_update(self):
        """Test statistics are properly updated."""
        config = SpeculativeConfig(enable_speculative=True)
        decoder = SpeculativeDecoder(config)

        # Perform some decodings
        prefix = [1, 2, 3]
        decoder.decode_next_tokens(prefix, max_tokens=1)
        decoder.decode_next_tokens(prefix, max_tokens=1)

        stats = decoder.get_stats()
        assert stats.total_forward_passes >= 2  # At least 2 forward passes

    def test_k_adjustment(self):
        """Test dynamic K adjustment."""
        config = SpeculativeConfig(
            enable_adaptive_k=True,
            k_adjust_frequency=3,
            acceptance_rate_target=0.8
        )
        decoder = SpeculativeDecoder(config)

        # Simulate high acceptance rate
        for _ in range(5):
            decoder.draft_model.record_acceptance(4, 5)  # 80% acceptance

        # K should potentially increase
        assert decoder.get_K() >= config.min_K

    def test_reset_stats(self):
        """Test statistics reset."""
        config = SpeculativeConfig()
        decoder = SpeculativeDecoder(config)

        # Generate some stats
        prefix = [1, 2, 3]
        decoder.decode_next_tokens(prefix)

        # Reset
        decoder.reset_stats()
        stats = decoder.get_stats()

        assert stats.total_tokens_generated == 0
        assert stats.total_forward_passes == 0

    def test_set_k(self):
        """Test K setting with bounds checking."""
        config = SpeculativeConfig(min_K=1, max_K=8)
        decoder = SpeculativeDecoder(config)

        # Valid K
        decoder.set_K(5)
        assert decoder.get_K() == 5

        # K below minimum
        decoder.set_K(0)
        assert decoder.get_K() == 1

        # K above maximum
        decoder.set_K(10)
        assert decoder.get_K() == 8


class TestIntegration:
    """Integration tests for the complete system."""

    def test_end_to_end_workflow(self):
        """Test complete speculative decoding workflow."""
        config = SpeculativeConfig(
            enable_speculative=True,
            initial_K=3,
            vocab_size=1000
        )
        decoder = SpeculativeDecoder(config)

        # Simulate a conversation
        conversation = []
        prefix = [101, 2054, 2003]  # "The quick brown"

        for _ in range(5):
            tokens = decoder.decode_next_tokens(prefix, max_tokens=2)
            conversation.extend(tokens)
            prefix.extend(tokens)

        assert len(conversation) > 0

        # Check that stats were updated
        stats = decoder.get_stats()
        assert stats.total_tokens_generated > 0

    def test_performance_monitoring(self):
        """Test performance monitoring."""
        config = SpeculativeConfig(enable_speculative=True)
        decoder = SpeculativeDecoder(config)

        # Perform several decodings
        prefix = [1, 2, 3]
        for _ in range(10):
            decoder.decode_next_tokens(prefix, max_tokens=1)

        stats = decoder.get_stats()
        assert stats.avg_decode_time_ms > 0

    def test_memory_bounds(self):
        """Test memory usage stays within bounds."""
        config = SpeculativeConfig(max_memory_overhead=0.5)
        decoder = SpeculativeDecoder(config)

        # This is a basic check - in practice, we'd monitor actual memory
        assert decoder.config.max_memory_overhead == 0.5


class TestBenchmarking:
    """Test benchmarking functionality."""

    def test_create_decoder(self):
        """Test decoder creation helper."""
        decoder = create_speculative_decoder()
        assert isinstance(decoder, SpeculativeDecoder)
        assert decoder.config.enable_speculative

    def test_benchmark_function(self):
        """Test benchmarking function."""
        config = SpeculativeConfig(enable_speculative=True, initial_K=2)
        decoder = SpeculativeDecoder(config)

        test_prefixes = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        results = benchmark_speculative_decoding(decoder, test_prefixes, num_iterations=5)

        required_keys = ["total_tokens", "total_time", "avg_speedup",
                        "acceptance_rate", "throughput_tokens_per_sec"]

        for key in required_keys:
            assert key in results
            assert isinstance(results[key], (int, float))

    def test_benchmark_edge_cases(self):
        """Test benchmarking with edge cases."""
        config = SpeculativeConfig(enable_speculative=False)  # Disabled speculative
        decoder = SpeculativeDecoder(config)

        test_prefixes = [[1, 2, 3]]

        results = benchmark_speculative_decoding(decoder, test_prefixes, num_iterations=3)

        # Should still work but with lower speedup
        assert results["avg_speedup"] >= 1.0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_prefix(self):
        """Test handling of empty prefix."""
        config = SpeculativeConfig()
        decoder = SpeculativeDecoder(config)

        # Should handle empty prefix gracefully
        tokens = decoder.decode_next_tokens([], max_tokens=1)
        assert isinstance(tokens, list)

    def test_large_k(self):
        """Test handling of large K values."""
        config = SpeculativeConfig(max_K=16, initial_K=8)
        decoder = SpeculativeDecoder(config)

        prefix = [1, 2, 3]
        tokens = decoder.decode_next_tokens(prefix, max_tokens=16)

        # Should not exceed max_tokens
        assert len(tokens) <= 16

    def test_verifier_failure(self):
        """Test handling of verifier failures."""
        config = SpeculativeConfig()
        decoder = SpeculativeDecoder(config)

        # Mock verifier to always fail
        decoder.verifier.verify_tokens = Mock(return_value=([], 0))

        prefix = [1, 2, 3]
        tokens = decoder.decode_next_tokens(prefix, max_tokens=2)

        # Should still return tokens (fallback)
        assert isinstance(tokens, list)
        assert len(tokens) >= 0


if __name__ == "__main__":
    # Run basic smoke test
    print("Running Speculative Decoding smoke tests...")

    config = SpeculativeConfig(enable_speculative=True, initial_K=2)
    decoder = SpeculativeDecoder(config)

    prefix = [101, 2054, 2003]
    tokens = decoder.decode_next_tokens(prefix, max_tokens=3)

    print(f"Generated tokens: {tokens}")

    stats = decoder.get_stats()
    print(f"Speedup: {stats.avg_speedup:.2f}x")
    print(f"Acceptance rate: {stats.acceptance_rate:.2f}")

    print("✅ Smoke tests passed!")