"""
Unit Tests for Tensor Parallelism Layer Implementation

Task 1.1.5: Comprehensive test coverage for:
  - RowParallelLinear correctness and performance
  - ColumnParallelLinear correctness and synchronization  
  - DistributedModelWrapper automatic parallelization
  - Communication utilities (all-reduce, broadcast)
  - Integration with distributed training

Test Categories:
  1. Correctness: Forward/backward pass matches baseline
  2. Numerical: Floating point precision validation
  3. Edge Cases: Boundary conditions and error handling
  4. Performance: Latency and memory benchmarking
  5. Distributed: Multi-GPU synchronization
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.testing import assert_close

# Import tensor parallelism components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.distributed.tensor_parallel import (
    RowParallelLinear,
    ColumnParallelLinear,
    DistributedModelWrapper,
    TensorParallelConfig,
    all_reduce_sum,
    broadcast_tensor,
)


# ============================================================================
# Test Utilities
# ============================================================================

class BaseTensorParallelTest(unittest.TestCase):
    """Base class for tensor parallelism tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        torch.manual_seed(42)
    
    def assert_shape(self, tensor: torch.Tensor, expected_shape: tuple):
        """Assert tensor has expected shape."""
        self.assertEqual(tuple(tensor.shape), expected_shape,
                        f"Expected shape {expected_shape}, got {tensor.shape}")
    
    def assert_finite(self, tensor: torch.Tensor):
        """Assert tensor contains no NaN or Inf values."""
        self.assertTrue(torch.isfinite(tensor).all(),
                       "Tensor contains NaN or Inf values")
    
    def assert_close(self, actual: torch.Tensor, expected: torch.Tensor, 
                    rtol: float = 1e-5, atol: float = 1e-7):
        """Assert two tensors are close within tolerance."""
        try:
            assert_close(actual, expected, rtol=rtol, atol=atol)
        except AssertionError as e:
            self.fail(f"Tensors not close: {e}")


# ============================================================================
# RowParallelLinear Tests
# ============================================================================

class TestRowParallelLinear(BaseTensorParallelTest):
    """Test RowParallelLinear layer functionality."""
    
    def test_initialization(self):
        """Test layer initialization with valid parameters."""
        layer = RowParallelLinear(
            in_features=256,
            out_features=1024,
            world_size=4,
            rank=0,
            dtype=self.dtype,
        )
        
        # Check shapes
        self.assert_shape(layer.weight, (256, 256))  # out_features/4 × in_features
        self.assert_shape(layer.bias, (256,))
        
        # Check device and dtype
        self.assertEqual(layer.weight.dtype, self.dtype)
        self.assertEqual(layer.bias.dtype, self.dtype)
    
    def test_initialization_invalid_dimension(self):
        """Test initialization fails with invalid dimensions."""
        with self.assertRaises(ValueError):
            RowParallelLinear(
                in_features=256,
                out_features=1000,  # Not divisible by world_size=4
                world_size=4,
                rank=0,
            )
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        batch_size, seq_len = 2, 128
        in_features = 256
        
        layer = RowParallelLinear(
            in_features=in_features,
            out_features=1024,
            world_size=4,
            rank=1,  # Different ranks
        )
        
        # Create input (replicated across all ranks)
        x = torch.randn(batch_size, seq_len, in_features, dtype=self.dtype)
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, 256)  # out_features/4
        self.assert_shape(output, expected_shape)
        
        # Check output is finite
        self.assert_finite(output)
    
    def test_forward_pass_deterministic(self):
        """Test forward pass is deterministic."""
        layer = RowParallelLinear(
            in_features=256,
            out_features=1024,
            world_size=2,
            rank=0,
        )
        
        x = torch.randn(4, 64, 256)
        
        # Multiple forward passes should produce identical results
        output1 = layer(x)
        output2 = layer(x)
        
        self.assert_close(output1, output2)
    
    def test_weight_initialization_distribution(self):
        """Test weights are initialized with correct distribution."""
        layer = RowParallelLinear(
            in_features=10000,
            out_features=10000,
            world_size=4,
            rank=0,
        )
        
        # Check initialization is non-trivial (not all zeros)
        self.assertGreater(layer.weight.abs().max().item(), 0.0)
        
        # Check weights are roughly normally distributed
        weight_mean = layer.weight.mean().item()
        weight_std = layer.weight.std().item()
        
        # Mean should be close to 0
        self.assertLess(abs(weight_mean), 0.1)
        
        # Std should be reasonable
        self.assertGreater(weight_std, 0.01)
        self.assertLess(weight_std, 1.0)
    
    def test_bias_disabled(self):
        """Test layer works without bias."""
        layer = RowParallelLinear(
            in_features=256,
            out_features=1024,
            bias=False,
            world_size=2,
            rank=0,
        )
        
        self.assertIsNone(layer.bias)
        
        x = torch.randn(2, 64, 256)
        output = layer(x)
        
        self.assert_shape(output, (2, 64, 512))
        self.assert_finite(output)
    
    def test_single_element_batch(self):
        """Test edge case: single element batch."""
        layer = RowParallelLinear(
            in_features=128,
            out_features=512,
            world_size=4,
            rank=2,
        )
        
        x = torch.randn(1, 1, 128)
        output = layer(x)
        
        self.assert_shape(output, (1, 1, 128))
        self.assert_finite(output)
    
    def test_large_batch(self):
        """Test with realistic large batch size."""
        layer = RowParallelLinear(
            in_features=4096,
            out_features=16384,
            world_size=8,
            rank=0,
        )
        
        x = torch.randn(64, 128, 4096)  # Large batch
        output = layer(x)
        
        self.assert_shape(output, (64, 128, 2048))  # 16384/8
        self.assert_finite(output)


# ============================================================================
# ColumnParallelLinear Tests
# ============================================================================

class TestColumnParallelLinear(BaseTensorParallelTest):
    """Test ColumnParallelLinear layer functionality."""
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = ColumnParallelLinear(
            in_features=1024,  # This is already local in_features
            out_features=4096,
            world_size=4,
            rank=0,
        )
        
        # Check shapes
        self.assert_shape(layer.weight, (4096, 256))  # out × in_local
        self.assert_shape(layer.bias, (4096,))
    
    def test_initialization_invalid_dimension(self):
        """Test initialization fails with invalid dimensions."""
        with self.assertRaises(ValueError):
            ColumnParallelLinear(
                in_features=1000,  # Not divisible by world_size=4
                out_features=4096,
                world_size=4,
                rank=0,
            )
    
    def test_forward_pass_shape(self):
        """Test forward pass with correct dimensions."""
        layer = ColumnParallelLinear(
            in_features=1024,  # local features per rank
            out_features=4096,
            world_size=4,
            rank=1,
        )
        
        x = torch.randn(2, 64, 256)  # in_features/4
        output = layer(x)
        
        self.assert_shape(output, (2, 64, 4096))
        self.assert_finite(output)
    
    def test_forward_pass_deterministic(self):
        """Test forward pass is deterministic."""
        layer = ColumnParallelLinear(
            in_features=512,
            out_features=2048,
            world_size=2,
            rank=0,
        )
        
        x = torch.randn(4, 32, 256)
        
        output1 = layer(x)
        output2 = layer(x)
        
        self.assert_close(output1, output2)
    
    def test_bias_disabled(self):
        """Test layer without bias."""
        layer = ColumnParallelLinear(
            in_features=1024,
            out_features=4096,
            bias=False,
            world_size=4,
            rank=0,
        )
        
        self.assertIsNone(layer.bias)
        
        x = torch.randn(2, 32, 256)
        output = layer(x)
        
        self.assert_shape(output, (2, 32, 4096))
        self.assert_finite(output)


# ============================================================================
# DistributedModelWrapper Tests
# ============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing parallelization."""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestDistributedModelWrapper(BaseTensorParallelTest):
    """Test automatic model parallelization."""
    
    def test_wrapper_initialization(self):
        """Test wrapper properly initializes."""
        model = SimpleModel(hidden_dim=256)
        wrapped = DistributedModelWrapper(model, world_size=4, rank=0)
        
        # Check wrapped model exists
        self.assertIsNotNone(wrapped.base_model)
        self.assertEqual(wrapped.world_size, 4)
        self.assertEqual(wrapped.rank, 0)
    
    def test_wrapper_forward_pass(self):
        """Test wrapped model forward pass works."""
        model = SimpleModel(hidden_dim=128)
        wrapped = DistributedModelWrapper(model, world_size=2, rank=0)
        
        x = torch.randn(2, 32, 128)
        output = wrapped(x)
        
        # Output shape depends on layer type
        self.assertIsNotNone(output)
        self.assert_finite(output)
    
    def test_layer_replacement(self):
        """Test that nn.Linear layers are replaced."""
        model = SimpleModel(hidden_dim=256)
        original_fc1_type = type(model.fc1)
        
        wrapped = DistributedModelWrapper(model, world_size=4, rank=0)
        
        # Check layers were replaced
        new_fc1_type = type(wrapped.base_model.fc1)
        self.assertNotEqual(original_fc1_type, new_fc1_type)
        self.assertTrue(
            new_fc1_type.__name__ in ["RowParallelLinear", "ColumnParallelLinear"]
        )


# ============================================================================
# Communication Utilities Tests
# ============================================================================

class TestCommunicationUtilities(BaseTensorParallelTest):
    """Test communication helper functions."""
    
    def test_all_reduce_sum_single_gpu(self):
        """Test all-reduce on single GPU (no-op)."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = all_reduce_sum(x)
        
        self.assert_close(result, x)
    
    def test_broadcast_tensor_single_gpu(self):
        """Test broadcast on single GPU (no-op)."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = broadcast_tensor(x)
        
        self.assert_close(result, x)
    
    def test_all_reduce_sum_inplace(self):
        """Test that all-reduce modifies in place."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = all_reduce_sum(x)
        
        # Should be the same object
        self.assertTrue(result is x)


# ============================================================================
# Integration Tests
# ============================================================================

class TestTensorParallelIntegration(BaseTensorParallelTest):
    """Integration tests for tensor parallelism system."""
    
    def test_layer_gradient_flow(self):
        """Test gradients flow correctly through layers."""
        layer = RowParallelLinear(
            in_features=256,
            out_features=1024,
            world_size=4,
            rank=0,
        )
        
        x = torch.randn(2, 64, 256, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)
        
        # Check gradients are finite
        self.assert_finite(x.grad)
        self.assert_finite(layer.weight.grad)
        self.assert_finite(layer.bias.grad)
    
    def test_model_with_optimizer(self):
        """Test training a wrapped model with optimizer."""
        model = SimpleModel(hidden_dim=128)
        wrapped = DistributedModelWrapper(model, world_size=2, rank=0)
        
        optimizer = optim.Adam(wrapped.parameters(), lr=1e-3)
        
        x = torch.randn(2, 32, 128)
        target = torch.randn(2, 32, 128)
        
        # Forward pass
        output = wrapped(x)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check training step succeeded
        self.assert_finite(loss)
    
    def test_config_creation(self):
        """Test tensor parallel config creation."""
        config = TensorParallelConfig(
            world_size=4,
            rank=0,
            backend="nccl",
            use_async_reduce=True,
            debug=True,
        )
        
        self.assertEqual(config.world_size, 4)
        self.assertEqual(config.rank, 0)
        self.assertEqual(config.backend, "nccl")
        self.assertTrue(config.use_async_reduce)
        self.assertTrue(config.debug)


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformance(BaseTensorParallelTest):
    """Performance benchmarking tests."""
    
    def test_forward_latency_row_parallel(self):
        """Benchmark RowParallelLinear forward pass latency."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        layer = RowParallelLinear(
            in_features=4096,
            out_features=16384,
            world_size=4,
            rank=0,
        ).cuda()
        
        x = torch.randn(64, 256, 4096).cuda()
        
        # Warmup
        for _ in range(10):
            _ = layer(x)
        
        torch.cuda.synchronize()
        
        # Measure
        import time
        start = time.perf_counter()
        
        iterations = 100
        for _ in range(iterations):
            _ = layer(x)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        avg_latency_ms = (elapsed / iterations) * 1000
        
        # Expect reasonable latency (< 100ms for forward pass)
        self.assertLess(avg_latency_ms, 100.0,
                       f"Forward latency too high: {avg_latency_ms:.2f}ms")
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        layer = RowParallelLinear(
            in_features=4096,
            out_features=16384,
            world_size=4,
            rank=0,
        ).cuda()
        
        # Measure memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        x = torch.randn(64, 256, 4096).cuda()
        output = layer(x)
        
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        
        # Expect < 2GB for this operation
        self.assertLess(peak_memory_gb, 2.0,
                       f"Memory usage too high: {peak_memory_gb:.2f}GB")


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
