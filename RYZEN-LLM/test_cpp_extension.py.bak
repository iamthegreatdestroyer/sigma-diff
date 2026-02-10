#!/usr/bin/env python3
"""
Test script for C++ extension functions after rebuild
This script tests the C++ implementations against the pure Python reference
"""

import numpy as np
import sys
import os
import time
from typing import List, Tuple

# Add src and build/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build', 'python'))

def test_cpp_vs_python():
    """Test C++ extension functions against Python reference implementations"""
    print("Testing C++ extension vs Python reference...")

    try:
        import ryzen_llm_bindings as ryzen_llm

        # Check if quantization functions are available
        has_quantize = hasattr(ryzen_llm, 'quantize_activations_int8')
        has_naive_matmul = hasattr(ryzen_llm, 'naive_ternary_matmul')

        if not (has_quantize and has_naive_matmul):
            print("❌ C++ quantization functions not available")
            print("Available functions:", [attr for attr in dir(ryzen_llm) if not attr.startswith('_')])
            return False

        print("✓ C++ quantization functions found")

        # Test quantization
        print("Testing quantization...")
        config = ryzen_llm.QuantConfig()
        config.activation_clip_value = 6.0

        activations = np.random.randn(100).astype(np.float32)

        # Python implementation
        py_result = quantize_activations_int8_py(activations, len(activations), config)

        # C++ implementation
        cpp_result = ryzen_llm.quantize_activations_int8(activations, len(activations), config)

        # Compare results
        assert abs(py_result.scale - cpp_result.scale) < 1e-6, f"Scale mismatch: {py_result.scale} vs {cpp_result.scale}"
        assert py_result.zero_point == cpp_result.zero_point, f"Zero point mismatch: {py_result.zero_point} vs {cpp_result.zero_point}"
        assert py_result.values == cpp_result.values, f"Values mismatch: {py_result.values[:10]} vs {cpp_result.values[:10]}"

        print("✓ Quantization C++ vs Python match")

        # Test matrix multiplication
        print("Testing matrix multiplication...")
        M, N, K = 8, 8, 8

        # Create test data
        weights = ryzen_llm.TernaryWeight(K, N, 4)  # group_size = 4
        for i in range(len(weights.values)):
            weights.values[i] = np.random.choice([-1, 0, 1])
            weights.scales[i // 4] = np.random.uniform(0.1, 1.0)

        output_py = [0.0] * (M * N)
        output_cpp = [0.0] * (M * N)

        # Python implementation
        naive_ternary_matmul_py(weights, py_result, output_py, M, N, K)

        # C++ implementation
        ryzen_llm.naive_ternary_matmul(weights, cpp_result, output_cpp, M, N, K)

        # Compare results
        max_diff = max(abs(a - b) for a, b in zip(output_py, output_cpp))
        assert max_diff < 1e-6, f"Matrix multiplication results differ by {max_diff}"

        print("✓ Matrix multiplication C++ vs Python match")

        # Performance comparison
        print("Benchmarking C++ vs Python performance...")
        iterations = 100

        # Benchmark Python
        start = time.perf_counter()
        for _ in range(iterations):
            output_temp = [0.0] * (M * N)
            naive_ternary_matmul_py(weights, py_result, output_temp, M, N, K)
        py_time = (time.perf_counter() - start) * 1000 / iterations

        # Benchmark C++
        start = time.perf_counter()
        for _ in range(iterations):
            output_temp = [0.0] * (M * N)
            ryzen_llm.naive_ternary_matmul(weights, cpp_result, output_temp, M, N, K)
        cpp_time = (time.perf_counter() - start) * 1000 / iterations

        print(f"  C++ time: {cpp_time:.2f} ms")
        print(f"  Speedup: {python_time / cpp_time:.2f}x")
        print(f"  Max error: {max_error:.2f}")
        return True

    except ImportError as e:
        print(f"❌ Cannot import C++ extension: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def quantize_activations_int8_py(activations: np.ndarray, size: int, config) -> 'QuantizedActivation':
    """Pure Python implementation of INT8 activation quantization"""
    class QuantizedActivation:
        def __init__(self):
            self.values: List[int] = []
            self.scale: float = 1.0
            self.zero_point: int = 0

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

def naive_ternary_matmul_py(weights, activations, output: List[float], M: int, N: int, K: int) -> None:
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

def main():
    print("=== C++ Extension Validation Tests ===\n")

    success = test_cpp_vs_python()

    if success:
        print("\n🎉 C++ extension validation passed!")
        print("\n📊 Summary:")
        print("- ✅ C++ and Python implementations match")
        print("- ✅ Quantization functions working")
        print("- ✅ Matrix multiplication algorithms verified")
        print("- ✅ Performance benchmarks completed")
        return 0
    else:
        print("\n❌ C++ extension validation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())