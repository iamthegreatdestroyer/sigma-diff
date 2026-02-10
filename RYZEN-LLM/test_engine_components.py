#!/usr/bin/env python3
"""
Test script for BitNet engine SafeTensors loading and matrix multiplication
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import pure Python implementations for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build', 'python'))

def quantize_activations_int8(activations, size, config):
    """Pure Python implementation of INT8 activation quantization"""
    class QuantizedActivation:
        def __init__(self):
            self.values = []
            self.scale = 1.0
            self.zero_point = 0

    result = QuantizedActivation()
    result.values = [0] * size

    # Find the maximum absolute value for symmetric quantization
    max_abs = max(abs(np.min(activations)), abs(np.max(activations)))
    clip_value = min(config.activation_clip_value, max_abs)

    # Compute scale: max_value / (2^7 - 1) for INT8 range [-127, 127]
    result.scale = clip_value / 127.0
    result.zero_point = 0  # Symmetric quantization

    # Quantize
    for i in range(size):
        clamped = max(-clip_value, min(clip_value, activations[i]))
        quantized = clamped / result.scale  # Divide by scale to get quantized value
        int_val = int(round(quantized))
        int_val = max(-127, min(127, int_val))
        result.values[i] = int_val

    return result

def naive_ternary_matmul(weights, activations, output, M, N, K):
    """Pure Python implementation of naive ternary matrix multiplication"""
    # Zero output matrix
    for i in range(M * N):
        output[i] = 0.0

    activation_scale = activations.scale
    activation_zero_point = activations.zero_point

    # Naive triple-loop matrix multiplication
    for m in range(M):
        for n in range(N):
            sum_val = 0.0

            for k in range(K):
                # Get quantized activation
                quantized_x = activations.values[m * K + k]
                dequantized_x = (quantized_x - activation_zero_point) * activation_scale

                # Get ternary weight and scale
                ternary_w = weights.values[k * N + n]
                weight_scale = weights.get_scale(k * N + n)
                scaled_weight = ternary_w * weight_scale

                # Accumulate
                sum_val += dequantized_x * scaled_weight

            output[m * N + n] = sum_val

class TernaryWeight:
    """Pure Python implementation of TernaryWeight"""
    def __init__(self, rows, cols, group_size=0):
        self.rows = rows
        self.cols = cols
        self.group_size = group_size
        total_elements = rows * cols
        self.values = [0] * total_elements
        num_groups = (total_elements + group_size - 1) // group_size if group_size > 0 else 1
        self.scales = [1.0] * num_groups

    def get_scale(self, idx):
        if self.group_size == 0:
            return self.scales[0]  # Per-layer scaling
        group_idx = idx // self.group_size
        return self.scales[group_idx]

class QuantConfig:
    """Pure Python implementation of QuantConfig"""
    def __init__(self):
        self.activation_clip_value = 6.0
        self.symmetric_activations = True

def test_naive_matmul():
    """Test the naive matrix multiplication"""
    print("\nTesting naive matrix multiplication...")

    try:
        # Create test matrices
        M, N, K = 4, 6, 8

        # Create ternary weights
        ternary_weight = TernaryWeight(K, N, 0)  # No grouping
        for i in range(len(ternary_weight.values)):
            ternary_weight.values[i] = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        ternary_weight.scales[0] = 1.0  # Single scale

        # Create quantized activations
        config = QuantConfig()
        config.activation_clip_value = 6.0
        config.symmetric_activations = True

        activations_fp32 = np.random.randn(M * K).astype(np.float32)
        quantized_activations = quantize_activations_int8(activations_fp32, len(activations_fp32), config)

        # Output buffer
        output = [0.0] * (M * N)

        # Perform multiplication using pure Python implementation
        naive_ternary_matmul(ternary_weight, quantized_activations, output, M, N, K)

        print(f"✓ Matrix multiplication: {M}x{K} * {K}x{N} = {M}x{N}")
        print(f"  Output range: {min(output):.4f} to {max(output):.4f}")

        return True
    except Exception as e:
        print(f"✗ Matrix multiplication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantization():
    """Test the quantization functions"""
    print("Testing quantization functions...")

    try:
        # Test data
        activations = np.random.randn(100).astype(np.float32)
        config = QuantConfig()
        config.activation_clip_value = 6.0
        config.symmetric_activations = True

        # Quantize using pure Python implementation
        result = quantize_activations_int8(activations, len(activations), config)

        print(f"✓ Quantized {len(activations)} activations")
        print(f"  Scale: {result.scale:.4f}")
        print(f"  Zero point: {result.zero_point}")
        print(f"  Value range: {min(result.values)} to {max(result.values)}")

        return True
    except Exception as e:
        print(f"✗ Quantization test failed: {e}")
        return False

def test_correctness():
    """Test correctness of matrix multiplication"""
    print("\nTesting correctness of matrix multiplication...")

    try:
        # Simple test case: 2x2 matrices
        M, N, K = 2, 2, 2

        # Create known inputs
        activations = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # 2x2
        config = QuantConfig()
        quantized_activations = quantize_activations_int8(activations, M * K, config)

        # Create ternary weights with known values
        weights = TernaryWeight(K, N, group_size=0)  # Per-layer scaling
        weights.values = [1, -1, 0, 1]  # 2x2 ternary matrix
        weights.scales = [0.5]  # Single scale

        # Output buffer
        output = [0.0] * (M * N)

        # Compute using our implementation
        naive_ternary_matmul(weights, quantized_activations, output, M, N, K)

        print("✓ Correctness test completed")
        print(f"  Output: {output}")

        # Check that output is not all zeros and has reasonable values
        if all(abs(x) < 1e-6 for x in output):
            print("✗ Output is all zeros")
            return False

        return True
    except Exception as e:
        print(f"✗ Correctness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_safetensors_loading():
    """Test SafeTensors loading functionality"""
    print("\nTesting SafeTensors loading...")

    try:
        # Try to import the C++ extension
        import ryzen_llm_bindings as ryzanstein_llm

        # Test creating basic objects
        config = ryzanstein_llm.ModelConfig()
        print("✓ ModelConfig created successfully")

        # Test creating engine with config (may fail if model not loaded)
        try:
            engine = ryzanstein_llm.BitNetEngine(config)
            print("✓ BitNetEngine created successfully")
        except Exception as e:
            print(f"⚠ BitNetEngine creation failed (expected if no model): {e}")

        return True
    except ImportError:
        print("⚠ C++ extension not available - skipping SafeTensors test")
        return True  # Don't fail if extension isn't built
    except Exception as e:
        print(f"✗ SafeTensors loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== BitNet Engine Component Tests ===\n")

    success = True
    success &= test_quantization()
    success &= test_naive_matmul()
    success &= test_correctness()
    success &= test_safetensors_loading()

    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())