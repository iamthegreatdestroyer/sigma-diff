/*
 * RYZEN-LLM BitNet Matrix Multiplication Kernels
 * [REF:PHASE1-002] - Ternary Weight Matrix Multiplication
 */

#pragma once

#include "../quantize.h"
#include <cstdint>

namespace ryzen_llm
{
    namespace bitnet
    {

        /**
         * Naive ternary matrix multiplication: Y = W × X
         *
         * This is the reference implementation for correctness validation.
         * No SIMD optimizations - pure scalar code for numerical accuracy baseline.
         *
         * @param weights Ternary weight matrix (M × K) in {-1, 0, +1}
         * @param activations Quantized activations (K × N) in INT8
         * @param output Output matrix (M × N) in FP32
         * @param M Number of output rows
         * @param N Number of output columns (batch size)
         * @param K Inner dimension (input features)
         */
        void naive_ternary_matmul(
            const TernaryWeight &weights,
            const QuantizedActivation &activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K);

        /**
         * Naive FP32 matrix multiplication: Y = W × X
         *
         * Reference implementation for accuracy comparison.
         * Used to compute numerical error vs quantized version.
         *
         * @param weights FP32 weight matrix (M × K)
         * @param activations FP32 activations (K × N)
         * @param output Output matrix (M × N) in FP32
         * @param M Number of output rows
         * @param N Number of output columns
         * @param K Inner dimension
         */
        void naive_fp32_matmul(
            const float *weights,
            const float *activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K);

        /**
         * Compute mean squared error between two matrices
         *
         * Used to validate quantized matmul accuracy.
         * Target: MSE < 1e-4 for acceptable quantization error.
         *
         * @param matrix_a First matrix (M × N)
         * @param matrix_b Second matrix (M × N)
         * @param size Total elements (M * N)
         * @return Mean squared error
         */
        float compute_mse(
            const float *matrix_a,
            const float *matrix_b,
            size_t size);

        /**
         * Compute maximum absolute error between two matrices
         *
         * Reports worst-case error for debugging.
         *
         * @param matrix_a First matrix
         * @param matrix_b Second matrix
         * @param size Total elements
         * @return Maximum absolute error
         */
        float compute_max_error(
            const float *matrix_a,
            const float *matrix_b,
            size_t size);

    } // namespace bitnet
} // namespace ryzen_llm
