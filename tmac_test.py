#!/usr/bin/env python3
"""
T-MAC Correctness Test
Tests T-MAC table generation and lookup logic without full module build.
"""

import numpy as np

def test_tmac_logic():
    """Test T-MAC table generation and lookup correctness."""
    print("T-MAC Correctness Test")
    print("=" * 50)

    # Test parameters
    M, N, K = 2, 2, 4  # Small test case
    lookup_width = 4
    num_groups = (K + lookup_width - 1) // lookup_width  # 1 group

    # Create ternary weights: {-1, 0, +1}
    ternary_weights = np.array([
        [1, -1, 0, 1],    # Row 0
        [0, 1, -1, 0]     # Row 1
    ], dtype=np.int8)

    weight_scales = np.array([1.0], dtype=np.float32)

    # Create quantized activations
    activations = np.array([
        [10, -5, 8, -3],  # Sequence 0
        [-2, 15, -8, 12]  # Sequence 1
    ], dtype=np.int8)

    act_scale = 0.1
    act_zero_point = 0

    print(f"Test case: {M}×{N} output, {K} inner dim")
    print(f"Ternary weights:\n{ternary_weights}")
    print(f"Activations:\n{activations}")
    print(f"Activation scale: {act_scale}, zero_point: {act_zero_point}")

    # Generate lookup table manually
    print("\nGenerating lookup table...")
    table = np.zeros((M, num_groups, 256), dtype=np.float32)

    for m in range(M):
        for g in range(num_groups):
            k_start = g * lookup_width
            k_end = min(k_start + lookup_width, K)
            actual_width = k_end - k_start

            for idx in range(256):
                sum_val = 0.0
                for i in range(actual_width):
                    if i >= 8:  # Only handle up to 8 bits
                        break
                    bit = (idx >> i) & 0x1
                    sign = 1.0 if bit else -1.0

                    w = ternary_weights[m, k_start + i]
                    w_scale = weight_scales[0]

                    # Ternary weight contribution
                    if w == 1:
                        sum_val += w_scale * sign
                    elif w == -1:
                        sum_val -= w_scale * sign
                    # w == 0: no contribution

                table[m, g, idx] = sum_val

    print(f"Table shape: {table.shape}")

    # Test lookup computation
    print("\nTesting lookup computation...")

    # Compute expected output using naive method
    expected_output = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            sum_val = 0.0
            for k in range(K):
                # Dequantize activation
                act_quant = activations[n, k]
                act_f = (act_quant - act_zero_point) * act_scale

                # Get ternary weight
                w = ternary_weights[m, k]

                # Ternary multiply
                if w == 1:
                    sum_val += act_f
                elif w == -1:
                    sum_val -= act_f
                # w == 0: no contribution

            expected_output[m, n] = sum_val * weight_scales[0]

    print(f"Expected (naive) output:\n{expected_output}")

    # Compute output using T-MAC lookup
    tmac_output = np.zeros((M, N), dtype=np.float32)

    for m in range(M):
        for n in range(N):
            sum_val = 0.0
            for g in range(num_groups):
                k_start = g * lookup_width
                k_end = min(k_start + lookup_width, K)

                # Build lookup index from activations
                idx = 0
                for i in range(k_end - k_start):
                    act_quant = activations[n, k_start + i]
                    act_f = (act_quant - act_zero_point) * act_scale
                    bit = 1 if act_f > 0.0 else 0
                    idx |= (bit << i)

                # Lookup table value
                table_val = table[m, g, idx]
                sum_val += table_val

            tmac_output[m, n] = sum_val

    print(f"T-MAC output:\n{tmac_output}")

    # Compare results
    diff = np.abs(tmac_output - expected_output)
    max_diff = np.max(diff)
    relative_error = diff / (np.abs(expected_output) + 1e-8)  # Avoid div by zero
    max_relative_error = np.max(relative_error)

    print("\nComparison:")
    print(f"Max absolute difference: {max_diff}")
    print(f"Max relative error: {max_relative_error:.2%}")

    if max_relative_error < 0.01:  # 1% tolerance
        print("✅ T-MAC logic appears correct")
        return True
    else:
        print("❌ T-MAC logic has errors")
        print("Detailed differences:")
        for m in range(M):
            for n in range(N):
                if abs(tmac_output[m, n] - expected_output[m, n]) > 1e-6:
                    print(f"  [{m},{n}]: T-MAC={tmac_output[m, n]:.6f}, Expected={expected_output[m, n]:.6f}")
        return False

if __name__ == "__main__":
    test_tmac_logic()