/*
 * Ryzanstein LLM - TL2_0 Parallel Kernel Infrastructure
 * [REF:IP-S5-001] - BitNet January 2026 Parallel Kernel Integration
 *
 * Top-level parallel kernel dispatch layer that:
 * 1. Selects the optimal kernel (LUT, VNNI, Naive) at runtime
 * 2. Provides unified benchmarking infrastructure
 * 3. Supports auto-tuning based on matrix dimensions and CPU capabilities
 * 4. Integrates with existing MatmulStats for performance tracking
 *
 * Kernel Selection Priority:
 *   [1] TL2_0 LUT-GEMV (best for M > 64, K > 128)
 *   [2] AVX-512 VNNI matmul (best for large dense GEMM)
 *   [3] Naive scalar (universal fallback)
 *
 * Copyright (c) 2025-2026 Ryzanstein LLM Project
 * Licensed under MIT License
 */

#pragma once

#include "lut_gemm.h"
#include "matmul.h"
#include "../quantize.h"
#include "../optimization_utils.h"

#include <cstdint>
#include <string>
#include <functional>
#include <vector>
#include <atomic>

namespace ryzanstein_llm
{
    namespace bitnet
    {
        namespace kernels
        {

            // ================================================================
            // Kernel Type Enumeration
            // ================================================================

            /**
             * Available kernel implementations in priority order.
             */
            enum class KernelType : uint8_t
            {
                TL2_LUT_AVX2 = 0,   // TL2_0 LUT + AVX2 + OpenMP (best for GEMV)
                TL2_LUT_OMP = 1,    // TL2_0 LUT + OpenMP (no SIMD)
                TL2_LUT_SCALAR = 2, // TL2_0 LUT scalar (single-thread)
                AVX512_VNNI = 3,    // AVX-512 VNNI tiled matmul
                NAIVE = 4,          // Naive scalar matmul (fallback)
                AUTO = 255          // Auto-select best available
            };

            /**
             * Get human-readable name for a kernel type.
             */
            inline const char *kernel_type_name(KernelType type)
            {
                switch (type)
                {
                case KernelType::TL2_LUT_AVX2:
                    return "TL2_LUT_AVX2";
                case KernelType::TL2_LUT_OMP:
                    return "TL2_LUT_OMP";
                case KernelType::TL2_LUT_SCALAR:
                    return "TL2_LUT_SCALAR";
                case KernelType::AVX512_VNNI:
                    return "AVX512_VNNI";
                case KernelType::NAIVE:
                    return "NAIVE";
                case KernelType::AUTO:
                    return "AUTO";
                default:
                    return "UNKNOWN";
                }
            }

            // ================================================================
            // Parallel Kernel Statistics
            // ================================================================

            /**
             * Extended statistics for TL2_0 parallel kernels.
             * Extends the existing MatmulStats pattern.
             */
            struct TL2KernelStats
            {
                // Call counters per kernel type
                std::atomic<uint64_t> tl2_lut_calls{0};
                std::atomic<uint64_t> avx512_calls{0};
                std::atomic<uint64_t> naive_calls{0};

                // Timing
                std::atomic<uint64_t> total_time_us{0}; // Microseconds
                std::atomic<uint64_t> total_flops{0};

                // Auto-tuning feedback
                std::atomic<uint64_t> lut_faster_count{0};
                std::atomic<uint64_t> vnni_faster_count{0};

                /**
                 * Record a kernel execution.
                 */
                void record(KernelType type, double time_us, uint64_t flops)
                {
                    total_time_us.fetch_add(
                        static_cast<uint64_t>(time_us),
                        std::memory_order_relaxed);
                    total_flops.fetch_add(flops, std::memory_order_relaxed);

                    switch (type)
                    {
                    case KernelType::TL2_LUT_AVX2:
                    case KernelType::TL2_LUT_OMP:
                    case KernelType::TL2_LUT_SCALAR:
                        tl2_lut_calls.fetch_add(1, std::memory_order_relaxed);
                        break;
                    case KernelType::AVX512_VNNI:
                        avx512_calls.fetch_add(1, std::memory_order_relaxed);
                        break;
                    case KernelType::NAIVE:
                        naive_calls.fetch_add(1, std::memory_order_relaxed);
                        break;
                    default:
                        break;
                    }
                }

                /**
                 * Get average GFLOP/s across all calls.
                 */
                double get_avg_gflops() const
                {
                    uint64_t time = total_time_us.load(std::memory_order_relaxed);
                    uint64_t flops = total_flops.load(std::memory_order_relaxed);
                    if (time == 0)
                        return 0.0;
                    return static_cast<double>(flops) / (static_cast<double>(time) * 1000.0);
                }

                /**
                 * Get total call count.
                 */
                uint64_t total_calls() const
                {
                    return tl2_lut_calls.load(std::memory_order_relaxed) +
                           avx512_calls.load(std::memory_order_relaxed) +
                           naive_calls.load(std::memory_order_relaxed);
                }

                /**
                 * Reset all statistics.
                 */
                void reset()
                {
                    tl2_lut_calls.store(0, std::memory_order_relaxed);
                    avx512_calls.store(0, std::memory_order_relaxed);
                    naive_calls.store(0, std::memory_order_relaxed);
                    total_time_us.store(0, std::memory_order_relaxed);
                    total_flops.store(0, std::memory_order_relaxed);
                    lut_faster_count.store(0, std::memory_order_relaxed);
                    vnni_faster_count.store(0, std::memory_order_relaxed);
                }
            };

            // Global statistics instance
            extern TL2KernelStats g_tl2_stats;

            // ================================================================
            // Kernel Selection Logic
            // ================================================================

            /**
             * Determine the optimal kernel type for given problem dimensions.
             *
             * Selection heuristics:
             * - GEMV (N=1): LUT is almost always optimal
             *   - M < 64: Scalar LUT (threading overhead too high)
             *   - M >= 64, K >= 128: Parallel LUT (best throughput)
             * - GEMM (N>1): AVX-512 VNNI can be better for large N
             *   - N > 8 with AVX-512: VNNI tiled matmul
             *   - N <= 8: LUT with batch loop
             *
             * @param M Number of output rows
             * @param N Number of output columns (1 = GEMV)
             * @param K Inner dimension
             * @return Recommended kernel type
             */
            KernelType select_kernel(uint32_t M, uint32_t N, uint32_t K);

            // ================================================================
            // Unified Kernel Interface
            // ================================================================

            /**
             * Unified ternary GEMV with automatic kernel selection.
             *
             * This is the primary entry point for inference code.
             * Selects the optimal kernel, executes it, and records statistics.
             *
             * @param weights Ternary weight matrix [M × K]
             * @param activations Quantized activation vector [K]
             * @param output FP32 output vector [M]
             * @param M Number of output features
             * @param K Number of input features
             * @param kernel Optional: force a specific kernel type (AUTO = auto-select)
             */
            void parallel_ternary_gemv(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t K,
                KernelType kernel = KernelType::AUTO);

            /**
             * Unified ternary GEMM with automatic kernel selection.
             *
             * @param weights Ternary weight matrix [M × K]
             * @param activations Quantized activations [K × N]
             * @param output FP32 output matrix [M × N]
             * @param M Number of output rows
             * @param N Number of output columns (batch)
             * @param K Inner dimension
             * @param kernel Optional: force a specific kernel type (AUTO = auto-select)
             */
            void parallel_ternary_gemm(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t N,
                uint32_t K,
                KernelType kernel = KernelType::AUTO);

            // ================================================================
            // Benchmarking Utilities
            // ================================================================

            /**
             * Benchmark result for a single kernel configuration.
             */
            struct BenchmarkResult
            {
                KernelType kernel;
                uint32_t M, N, K;
                double time_us;          // Average wall-clock microseconds
                double gflops;           // Effective GFLOP/s
                double speedup_vs_naive; // Speedup relative to naive
                double mse;              // Mean squared error vs naive
                int num_threads;
            };

            /**
             * Run benchmarks comparing all available kernels.
             *
             * Creates random ternary weights and INT8 activations,
             * then times each kernel implementation. Reports speedup
             * vs naive and numerical accuracy (MSE).
             *
             * @param M Number of output rows
             * @param K Inner dimension
             * @param iterations Number of timing iterations
             * @return Benchmark results for all tested kernels
             */
            std::vector<BenchmarkResult> benchmark_tl2_kernels(
                uint32_t M,
                uint32_t K,
                int iterations = 100);

            /**
             * Run GEMM benchmarks with varying batch sizes.
             *
             * @param M Number of output rows
             * @param N_values List of batch sizes to test
             * @param K Inner dimension
             * @param iterations Number of timing iterations
             * @return Benchmark results for all (kernel, batch) combinations
             */
            std::vector<BenchmarkResult> benchmark_tl2_gemm(
                uint32_t M,
                const std::vector<uint32_t> &N_values,
                uint32_t K,
                int iterations = 50);

        } // namespace kernels
    } // namespace bitnet
} // namespace ryzanstein_llm
