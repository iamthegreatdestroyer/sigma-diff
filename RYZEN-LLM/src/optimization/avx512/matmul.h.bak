/*
 * RYZEN-LLM AVX-512 Optimized Matrix Multiplication
 * [REF:PHASE1-006] - VNNI-Accelerated Ternary×INT8 Matmul
 *
 * Target: 8-12× speedup vs naive baseline
 * Key Technologies:
 * - AVX-512 VNNI (Vector Neural Network Instructions)
 * - Tiled computation for cache efficiency
 * - Register blocking and loop unrolling
 * - CPU feature detection with runtime dispatch
 */

#pragma once

#include "../../core/bitnet/quantize.h"
#include <cstdint>
#include <string>

namespace ryzen_llm
{
    namespace avx512
    {

        /**
         * CPU Feature Detection
         *
         * Detects AVX-512 instruction set availability at runtime.
         * Required features:
         * - AVX-512F (Foundation)
         * - AVX-512VNNI (Vector Neural Network Instructions)
         * - AVX-512BW (Byte and Word)
         */
        struct CPUFeatures
        {
            bool has_avx512f;     // AVX-512 Foundation
            bool has_avx512_vnni; // VNNI for INT8 ops
            bool has_avx512bw;    // Byte/Word operations
            bool has_avx512_vbmi; // Vector Bit Manipulation

            CPUFeatures();
            bool supports_optimized_kernel() const;
            std::string to_string() const;
        };

        /**
         * AVX-512 Optimized Ternary×INT8 Matrix Multiplication
         *
         * Y[M × N] = W[M × K] × X[K × N]
         * - W: Ternary weights {-1, 0, +1} with per-element scaling
         * - X: INT8 quantized activations
         * - Y: FP32 output
         *
         * Optimization Strategy:
         * 1. Tile computation into cache-friendly blocks
         * 2. Use AVX-512 VNNI for INT8×INT8 → INT32 accumulation
         * 3. Handle ternary {-1, 0, +1} with masked operations
         * 4. Vectorize dequantization and scaling
         *
         * Performance Target:
         * - Small model (512 hidden): 20-40 tok/s
         * - BitNet 7B (4096 hidden): 25 tok/s
         *
         * @param weights Ternary weight matrix (M × K)
         * @param activations INT8 quantized activations (K × N)
         * @param output FP32 output matrix (M × N)
         * @param M Number of output rows
         * @param N Number of output columns (batch size)
         * @param K Inner dimension
         */
        void optimized_ternary_matmul(
            const bitnet::TernaryWeight &weights,
            const bitnet::QuantizedActivation &activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K);

        /**
         * Kernel Dispatcher
         *
         * Selects optimal implementation based on CPU features:
         * - AVX-512 VNNI if available (best performance)
         * - AVX-512F fallback if VNNI unavailable
         * - Naive scalar fallback if no AVX-512
         *
         * @param weights Ternary weight matrix
         * @param activations INT8 activations
         * @param output FP32 output
         * @param M Output rows
         * @param N Output columns
         * @param K Inner dimension
         */
        void dispatch_ternary_matmul(
            const bitnet::TernaryWeight &weights,
            const bitnet::QuantizedActivation &activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K);

        /**
         * Performance Statistics
         *
         * Tracks kernel performance for benchmarking and optimization.
         */
        struct MatmulStats
        {
            uint64_t total_calls;
            uint64_t total_flops;
            double total_time_ms;
            double peak_gflops;

            MatmulStats() : total_calls(0), total_flops(0), total_time_ms(0.0), peak_gflops(0.0) {}

            void record_call(uint32_t M, uint32_t N, uint32_t K, double time_ms);
            double get_avg_gflops() const;
            void reset();
            std::string to_string() const;
        };

        /**
         * Global statistics instance
         */
        extern MatmulStats g_matmul_stats;

    } // namespace avx512
} // namespace ryzen_llm
