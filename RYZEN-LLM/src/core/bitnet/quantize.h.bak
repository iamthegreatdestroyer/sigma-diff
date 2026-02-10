/*
 * RYZEN-LLM BitNet Quantization
 * [REF:PHASE1-001] - BitNet b1.58 Ternary Quantization
 *
 * This module implements ternary weight quantization (W ∈ {-1, 0, +1})
 * and INT8 activation quantization for BitNet b1.58 architecture.
 *
 * References:
 * - BitNet b1.58: https://arxiv.org/abs/2402.17764
 * - Quantization scheme maintains quality while enabling efficient CPU inference
 */

#ifndef RYZEN_LLM_BITNET_QUANTIZE_H
#define RYZEN_LLM_BITNET_QUANTIZE_H

#include <cstdint>
#include <vector>
#include <memory>

namespace ryzen_llm
{
    namespace bitnet
    {

        /**
         * Ternary weight representation
         *
         * Stores weights in {-1, 0, +1} format with per-layer or per-group scaling.
         * Memory layout: values are packed as int8_t for SIMD efficiency.
         */
        struct TernaryWeight
        {
            // Weight values: -1, 0, or +1 stored as int8_t
            std::vector<int8_t> values;

            // Scaling factors (per-layer or per-group)
            std::vector<float> scales;

            // Dimensions
            uint32_t rows;
            uint32_t cols;

            // Number of elements per scale group (0 = per-layer)
            uint32_t group_size;

            TernaryWeight() : rows(0), cols(0), group_size(0) {}

            TernaryWeight(uint32_t r, uint32_t c, uint32_t gs = 0)
                : rows(r), cols(c), group_size(gs)
            {
                values.resize(r * c);
                uint32_t num_groups = (gs > 0) ? ((r * c + gs - 1) / gs) : 1;
                scales.resize(num_groups);
            }

            /**
             * Get the scale factor for a given element index
             */
            float get_scale(uint32_t idx) const
            {
                if (group_size == 0)
                {
                    return scales[0]; // Per-layer scaling
                }
                uint32_t group_idx = idx / group_size;
                return scales[group_idx];
            }
        };

        /**
         * CPU-compatible ternary weight representation (no std::vector)
         *
         * Stores weights in {-1, 0, +1} format with per-layer or per-group scaling.
         * Uses raw pointers for CPU compatibility (no AVX-512 optimized constructors).
         */
        struct TernaryWeightCPU
        {
            // Weight values: -1, 0, or +1 stored as int8_t
            int8_t *values;

            // Scaling factors (per-layer or per-group)
            float *scales;

            // Dimensions
            uint32_t rows;
            uint32_t cols;

            // Number of elements per scale group (0 = per-layer)
            uint32_t group_size;

            // Number of scale groups
            uint32_t num_scales;

            TernaryWeightCPU() : values(nullptr), scales(nullptr), rows(0), cols(0), group_size(0), num_scales(0) {}

            TernaryWeightCPU(uint32_t r, uint32_t c, uint32_t gs = 0)
                : rows(r), cols(c), group_size(gs)
            {
                // Count total elements using nested loops to avoid AVX-512 multiplication
                uint32_t total_elements = 0;
                for (uint32_t ri = 0; ri < r; ++ri)
                {
                    for (uint32_t ci = 0; ci < c; ++ci)
                    {
                        total_elements++;
                    }
                }

                // Calculate number of scales using loops to avoid AVX-512 division/multiplication
                num_scales = 1;
                if (gs > 0)
                {
                    num_scales = 0;
                    uint32_t remaining = total_elements;
                    while (remaining > 0)
                    {
                        num_scales++;
                        if (remaining >= gs)
                        {
                            remaining -= gs;
                        }
                        else
                        {
                            remaining = 0;
                        }
                    }
                }

                // Allocate memory manually
                values = new int8_t[total_elements];
                scales = new float[num_scales];

                // Initialize to zero
                for (uint32_t i = 0; i < total_elements; ++i)
                {
                    values[i] = 0;
                }
                for (uint32_t i = 0; i < num_scales; ++i)
                {
                    scales[i] = 0.0f;
                }
            }

            ~TernaryWeightCPU()
            {
                if (values)
                    delete[] values;
                if (scales)
                    delete[] scales;
            }

            /**
             * Get the scale factor for a given element index
             */
            float get_scale(uint32_t idx) const
            {
                if (group_size == 0)
                {
                    return scales[0]; // Per-layer scaling
                }
                // Avoid division: use loop-based calculation
                uint32_t group_idx = 0;
                uint32_t temp_idx = idx;
                while (temp_idx >= group_size)
                {
                    group_idx++;
                    // Avoid subtraction: count down instead
                    uint32_t new_temp = 0;
                    for (uint32_t i = group_size; i < temp_idx; ++i)
                    {
                        new_temp++;
                    }
                    temp_idx = new_temp;
                }
                return scales[group_idx];
            }
        };

        /**
         * INT8 activation representation
         *
         * Activations quantized to [-128, 127] range with per-tensor scaling.
         */
        struct QuantizedActivation
        {
            std::vector<int8_t> values;
            float scale;
            int8_t zero_point;

            QuantizedActivation() : scale(1.0f), zero_point(0) {}

            explicit QuantizedActivation(size_t size)
                : values(size), scale(1.0f), zero_point(0) {}
        };

        /**
         * CPU-compatible INT8 activation representation (no std::vector)
         *
         * Activations quantized to [-128, 127] range with per-tensor scaling.
         * Uses raw pointers for CPU compatibility (no AVX-512 optimized constructors).
         */
        struct QuantizedActivationCPU
        {
            int8_t *values;
            float scale;
            int8_t zero_point;
            size_t size;

            QuantizedActivationCPU() : values(nullptr), scale(1.0f), zero_point(0), size(0) {}

            explicit QuantizedActivationCPU(size_t s)
                : scale(1.0f), zero_point(0), size(s)
            {
                values = new int8_t[s];
                for (size_t i = 0; i < s; ++i)
                {
                    values[i] = 0;
                }
            }

            ~QuantizedActivationCPU()
            {
                if (values)
                    delete[] values;
            }
        };

        /**
         * Quantization configuration
         */
        struct QuantConfig
        {
            // Weight quantization
            bool per_group_scaling;     // True: per-group, False: per-layer
            uint32_t weight_group_size; // Elements per group (if per_group_scaling)

            // Activation quantization
            float activation_clip_value; // Clip range for activations (e.g., 6.0)
            bool symmetric_activations;  // Symmetric [-c, c] vs asymmetric

            // Default configuration
            QuantConfig()
                : per_group_scaling(false),
                  weight_group_size(128),
                  activation_clip_value(6.0f),
                  symmetric_activations(true) {}
        };

        /**
         * Quantize FP32 weights to ternary {-1, 0, +1}
         *
         * Algorithm:
         * 1. Compute mean absolute value: α = mean(|W|)
         * 2. Quantize: W_q = sign(W) if |W| > α * threshold, else 0
         * 3. Compute scale: s = mean(|W|) / mean(|W_q|)
         *
         * @param weights Input FP32 weights [rows x cols]
         * @param rows Number of rows
         * @param cols Number of columns
         * @param config Quantization configuration
         * @return TernaryWeight structure
         */
        TernaryWeight quantize_weights_ternary(
            const float *weights,
            uint32_t rows,
            uint32_t cols,
            const QuantConfig &config = QuantConfig());

        /**
         * Quantize FP32 activations to INT8
         *
         * Algorithm:
         * 1. Clip activations to [-clip_value, clip_value]
         * 2. Compute scale: s = clip_value / 127
         * 3. Quantize: A_q = round(A / s)
         *
         * @param activations Input FP32 activations
         * @param size Number of elements
         * @param config Quantization configuration
         * @return QuantizedActivation structure
         */
        QuantizedActivation quantize_activations_int8(
            const float *activations,
            size_t size,
            const QuantConfig &config = QuantConfig());

        /**
         * Dequantize ternary weights back to FP32
         *
         * @param ternary_weight Quantized ternary weights
         * @param output Output buffer for FP32 weights [rows x cols]
         */
        void dequantize_weights(
            const TernaryWeight &ternary_weight,
            float *output);

        /**
         * Dequantize INT8 activations back to FP32
         *
         * @param quantized Quantized INT8 activations
         * @param output Output buffer for FP32 activations
         */
        void dequantize_activations(
            const QuantizedActivation &quantized,
            float *output);

        /**
         * Compute quantization error metrics
         *
         * @param original Original FP32 values
         * @param quantized Quantized values
         * @param size Number of elements
         * @return Mean squared error
         */
        float compute_quantization_error(
            const float *original,
            const float *quantized,
            size_t size);

        /**
         * Pack ternary weights for efficient storage
         *
         * Packs 4 ternary values into 1 byte (2 bits each: 00=-1, 01=0, 10=+1)
         * This reduces memory footprint by 4x compared to int8_t storage.
         *
         * @param ternary_weight Input ternary weights
         * @return Packed byte vector
         */
        std::vector<uint8_t> pack_ternary_weights(const TernaryWeight &ternary_weight);

        /**
         * Unpack ternary weights from compressed format
         *
         * @param packed Packed byte vector
         * @param rows Number of rows
         * @param cols Number of columns
         * @return TernaryWeight structure
         */
        TernaryWeight unpack_ternary_weights(
            const std::vector<uint8_t> &packed,
            uint32_t rows,
            uint32_t cols);

        /**
         * Naive ternary matrix multiplication (fallback implementation)
         *
         * Performs matrix multiplication: output[M][N] = activations[M][K] * weights[K][N]
         * Uses scalar operations for CPU compatibility when SIMD is not available.
         *
         * @param weights Ternary weight matrix (K x N)
         * @param activations Quantized activation matrix (M x K)
         * @param output Output matrix (M x N)
         * @param M Rows in activations / output
         * @param N Columns in weights / output
         * @param K Columns in activations / rows in weights
         */
        void naive_ternary_matmul(
            const TernaryWeight &weights,
            const QuantizedActivation &activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K);

        // Scalar-only versions for CPU compatibility (no SIMD instructions)

        /**
         * Scalar-only ternary weight quantization (CPU compatible)
         */
        TernaryWeightCPU quantize_weights_ternary_scalar(
            const float *weights,
            uint32_t rows,
            uint32_t cols,
            const QuantConfig &config = QuantConfig());

        /**
         * Scalar-only INT8 activation quantization (CPU compatible)
         */
        QuantizedActivationCPU quantize_activations_int8_scalar(
            const float *activations,
            size_t size,
            const QuantConfig &config = QuantConfig());

        /**
         * Scalar-only ternary weight dequantization (CPU compatible)
         */
        void dequantize_weights_scalar(
            const TernaryWeightCPU &weights,
            float *output,
            uint32_t rows,
            uint32_t cols);

        /**
         * Scalar-only INT8 activation dequantization (CPU compatible)
         */
        void dequantize_activations_scalar(
            const QuantizedActivationCPU &activations,
            float *output,
            size_t size);

        /**
         * Scalar-only quantization error computation (CPU compatible)
         */
        float compute_quantization_error_scalar(
            const float *original,
            const float *dequantized,
            size_t size);

        // CPU-compatible versions using raw pointers (no std::vector)

        /**
         * CPU-compatible ternary weight quantization (no AVX-512)
         */
        TernaryWeightCPU quantize_weights_ternary_cpu(
            const float *weights,
            uint32_t rows,
            uint32_t cols,
            const QuantConfig &config = QuantConfig());

        /**
         * CPU-compatible INT8 activation quantization (no AVX-512)
         */
        QuantizedActivationCPU quantize_activations_int8_cpu(
            const float *activations,
            size_t size,
            const QuantConfig &config = QuantConfig());

        /**
         * CPU-compatible ternary weight dequantization (no AVX-512)
         */
        void dequantize_weights_cpu(
            const TernaryWeightCPU &weights,
            float *output);

        /**
         * CPU-compatible INT8 activation dequantization (no AVX-512)
         */
        void dequantize_activations_cpu(
            const QuantizedActivationCPU &activations,
            float *output);

        /**
         * CPU-compatible quantization error computation (no AVX-512)
         */
        float compute_quantization_error_cpu(
            const float *original,
            const float *dequantized,
            size_t size);

    } // namespace bitnet
} // namespace ryzen_llm

#endif // RYZEN_LLM_BITNET_QUANTIZE_H
