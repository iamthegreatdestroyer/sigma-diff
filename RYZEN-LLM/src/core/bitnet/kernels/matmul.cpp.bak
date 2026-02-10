/*
 * RYZEN-LLM BitNet Matrix Multiplication Implementation
 * [REF:PHASE1-002] - Baseline Ternary Matmul (Correctness-First)
 */

#include "matmul.h"
#include <cmath>
#include <algorithm>

namespace ryzen_llm
{
    namespace bitnet
    {

        void naive_ternary_matmul(
            const TernaryWeight &weights,
            const QuantizedActivation &activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K)
        {
            // Y[M × N] = W[M × K] × X[K × N]
            // W is ternary {-1, 0, +1} with scaling factors
            // X is INT8 quantized with scale and zero point

            // Dequantize activation scale (convert INT8 back to approximate FP32 range)
            const float activation_scale = activations.scale;
            const int8_t activation_zero_point = activations.zero_point;

            // Zero output matrix
            for (uint32_t i = 0; i < M * N; ++i)
            {
                output[i] = 0.0f;
            }

            // Naive triple-nested loop
            for (uint32_t m = 0; m < M; ++m)
            {
                for (uint32_t n = 0; n < N; ++n)
                {
                    float accumulator = 0.0f;

                    for (uint32_t k = 0; k < K; ++k)
                    {
                        // Get ternary weight and its scale
                        const int8_t ternary_w = weights.values[m * K + k];
                        // Ensure index fits uint32_t expected by get_scale()
                        const float weight_scale = weights.get_scale(static_cast<uint32_t>(m * K + k));

                        // Get quantized activation
                        const int8_t quantized_x = activations.values[k * N + n];

                        // Dequantize activation: x_fp32 = (x_int8 - zero_point) * scale
                        const float dequantized_x =
                            (static_cast<float>(quantized_x) - activation_zero_point) * activation_scale;

                        // Multiply: ternary_weight * activation
                        // ternary_w is {-1, 0, +1}, so this is just: +x, 0, or -x
                        const float scaled_weight = static_cast<float>(ternary_w) * weight_scale;
                        accumulator += scaled_weight * dequantized_x;
                    }

                    output[m * N + n] = accumulator;
                }
            }
        }

        void naive_fp32_matmul(
            const float *weights,
            const float *activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K)
        {
            // Standard FP32 matmul: Y[M × N] = W[M × K] × X[K × N]

            // Zero output
            for (uint32_t i = 0; i < M * N; ++i)
            {
                output[i] = 0.0f;
            }

            for (uint32_t m = 0; m < M; ++m)
            {
                for (uint32_t n = 0; n < N; ++n)
                {
                    float accumulator = 0.0f;

                    for (uint32_t k = 0; k < K; ++k)
                    {
                        accumulator += weights[m * K + k] * activations[k * N + n];
                    }

                    output[m * N + n] = accumulator;
                }
            }
        }

        float compute_mse(
            const float *matrix_a,
            const float *matrix_b,
            size_t size)
        {
            double sum_squared_error = 0.0;

            for (size_t i = 0; i < size; ++i)
            {
                const double diff = static_cast<double>(matrix_a[i]) - static_cast<double>(matrix_b[i]);
                sum_squared_error += diff * diff;
            }

            return static_cast<float>(sum_squared_error / static_cast<double>(size));
        }

        float compute_max_error(
            const float *matrix_a,
            const float *matrix_b,
            size_t size)
        {
            float max_error = 0.0f;

            for (size_t i = 0; i < size; ++i)
            {
                const float error = std::abs(matrix_a[i] - matrix_b[i]);
                max_error = std::max(max_error, error);
            }

            return max_error;
        }

    } // namespace bitnet
} // namespace ryzen_llm
