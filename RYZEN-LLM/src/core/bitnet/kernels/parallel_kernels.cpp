/*
 * Ryzanstein LLM - TL2_0 Parallel Kernel Implementation
 * [REF:IP-S5-001] - BitNet January 2026 Parallel Kernel Integration
 *
 * Implements:
 * - Runtime kernel selection based on dimensions + CPU features
 * - Unified GEMV/GEMM dispatch with statistics tracking
 * - Benchmarking harness comparing all kernel variants
 *
 * Copyright (c) 2025-2026 Ryzanstein LLM Project
 * Licensed under MIT License
 */

#include "parallel_kernels.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward-declare AVX-512 dispatch from optimization layer
// (defined in src/optimization/avx512/matmul.cpp)
namespace ryzanstein_llm
{
    namespace bitnet
    {
        void dispatch_ternary_matmul(
            const TernaryWeight &weights,
            const QuantizedActivation &activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K);
    }
}

namespace ryzanstein_llm
{
    namespace bitnet
    {
        namespace kernels
        {

            // Global statistics instance
            TL2KernelStats g_tl2_stats;

            // ================================================================
            // CPU Feature Detection (cached)
            // ================================================================

            namespace
            {
                struct CPUCapabilities
                {
                    bool has_avx2;
                    bool has_fma;
                    bool has_avx512f;
                    bool has_avx512_vnni;
                    bool has_omp;

                    CPUCapabilities()
                    {
#ifdef __AVX2__
                        has_avx2 = true;
#else
                        has_avx2 = false;
#endif

#ifdef __FMA__
                        has_fma = true;
#else
                        has_fma = false;
#endif

#ifdef __AVX512F__
                        has_avx512f = true;
#else
                        has_avx512f = false;
#endif

#ifdef __AVX512VNNI__
                        has_avx512_vnni = true;
#else
                        has_avx512_vnni = false;
#endif

#ifdef _OPENMP
                        has_omp = true;
#else
                        has_omp = false;
#endif
                    }

                    bool supports_tl2_avx2() const { return has_avx2 && has_omp; }
                    bool supports_tl2_omp() const { return has_omp; }
                    bool supports_avx512_vnni() const { return has_avx512f && has_avx512_vnni; }
                };

                // Static initialization = detected once at startup
                static const CPUCapabilities g_cpu_caps;

            } // anonymous namespace

            // ================================================================
            // Kernel Selection
            // ================================================================

            KernelType select_kernel(uint32_t M, uint32_t N, uint32_t K)
            {
                // GEMV case (N=1): LUT is almost always optimal
                if (N <= 1)
                {
                    if (g_cpu_caps.supports_tl2_avx2() && M >= 32 && K >= 64)
                    {
                        return KernelType::TL2_LUT_AVX2;
                    }
                    if (g_cpu_caps.supports_tl2_omp() && M >= 64)
                    {
                        return KernelType::TL2_LUT_OMP;
                    }
                    if (M >= 16)
                    {
                        return KernelType::TL2_LUT_SCALAR;
                    }
                    return KernelType::NAIVE;
                }

                // GEMM case (N>1)
                // For large batches with AVX-512 VNNI, the tiled matmul can win
                if (N >= 8 && g_cpu_caps.supports_avx512_vnni() && M >= 128 && K >= 128)
                {
                    return KernelType::AVX512_VNNI;
                }

                // For small batches, LUT with batch loop
                if (g_cpu_caps.supports_tl2_avx2())
                {
                    return KernelType::TL2_LUT_AVX2;
                }
                if (g_cpu_caps.supports_tl2_omp())
                {
                    return KernelType::TL2_LUT_OMP;
                }

                return KernelType::NAIVE;
            }

            // ================================================================
            // Unified GEMV Dispatch
            // ================================================================

            void parallel_ternary_gemv(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t K,
                KernelType kernel)
            {
                PerfTimer timer;

                if (kernel == KernelType::AUTO)
                {
                    kernel = select_kernel(M, 1, K);
                }

                switch (kernel)
                {
                case KernelType::TL2_LUT_AVX2:
                    tl2_lut_gemv_avx2(weights, activations, output, M, K);
                    break;

                case KernelType::TL2_LUT_OMP:
                {
                    CacheConfig cache;
                    tl2_lut_gemv_parallel(weights, activations, output, M, K, 0, cache);
                    break;
                }

                case KernelType::TL2_LUT_SCALAR:
                {
                    CacheConfig cache;
                    tl2_lut_gemv_scalar(weights, activations, output, M, K, cache);
                    break;
                }

                case KernelType::AVX512_VNNI:
                    // Use existing AVX-512 dispatch for GEMV
                    dispatch_ternary_matmul(weights, activations, output, M, 1, K);
                    break;

                case KernelType::NAIVE:
                default:
                    naive_ternary_matmul(weights, activations, output, M, 1, K);
                    break;
                }

                double elapsed_us = timer.elapsed_us();
                uint64_t flops = 2ULL * M * K;
                g_tl2_stats.record(kernel, elapsed_us, flops);
            }

            // ================================================================
            // Unified GEMM Dispatch
            // ================================================================

            void parallel_ternary_gemm(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t N,
                uint32_t K,
                KernelType kernel)
            {
                // Delegate N=1 to GEMV path
                if (N == 1)
                {
                    parallel_ternary_gemv(weights, activations, output, M, K, kernel);
                    return;
                }

                PerfTimer timer;

                if (kernel == KernelType::AUTO)
                {
                    kernel = select_kernel(M, N, K);
                }

                switch (kernel)
                {
                case KernelType::TL2_LUT_AVX2:
                case KernelType::TL2_LUT_OMP:
                case KernelType::TL2_LUT_SCALAR:
                {
                    CacheConfig cache;
                    tl2_lut_gemm_tiled(weights, activations, output,
                                       M, N, K, 0, cache);
                    break;
                }

                case KernelType::AVX512_VNNI:
                    dispatch_ternary_matmul(weights, activations, output, M, N, K);
                    break;

                case KernelType::NAIVE:
                default:
                    naive_ternary_matmul(weights, activations, output, M, N, K);
                    break;
                }

                double elapsed_us = timer.elapsed_us();
                uint64_t flops = 2ULL * M * N * K;
                g_tl2_stats.record(kernel, elapsed_us, flops);
            }

            // ================================================================
            // Benchmarking Harness
            // ================================================================

            namespace
            {

                /**
                 * Generate random ternary weights {-1, 0, +1}.
                 */
                TernaryWeight generate_random_weights(uint32_t M, uint32_t K,
                                                      uint32_t group_size = 128)
                {
                    TernaryWeight w;
                    w.rows = M;
                    w.cols = K;
                    w.group_size = group_size;
                    w.values.resize(static_cast<size_t>(M) * K);

                    uint32_t num_groups = (K + group_size - 1) / group_size;
                    w.scales.resize(static_cast<size_t>(M) * num_groups);

                    std::mt19937 rng(42);
                    std::uniform_int_distribution<int> ternary_dist(-1, 1);
                    std::uniform_real_distribution<float> scale_dist(0.01f, 1.0f);

                    for (auto &v : w.values)
                    {
                        v = static_cast<int8_t>(ternary_dist(rng));
                    }
                    for (auto &s : w.scales)
                    {
                        s = scale_dist(rng);
                    }

                    return w;
                }

                /**
                 * Generate random INT8 activations.
                 */
                QuantizedActivation generate_random_activations(uint32_t K)
                {
                    QuantizedActivation a;
                    a.values.resize(K);
                    a.scale = 0.05f;
                    a.zero_point = 0;

                    std::mt19937 rng(123);
                    std::uniform_int_distribution<int> dist(-127, 127);
                    for (auto &v : a.values)
                    {
                        v = static_cast<int8_t>(dist(rng));
                    }

                    return a;
                }

                /**
                 * Compute MSE between two float arrays.
                 */
                double compute_output_mse(const float *a, const float *b, uint32_t n)
                {
                    double sum_sq = 0.0;
                    for (uint32_t i = 0; i < n; ++i)
                    {
                        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
                        sum_sq += diff * diff;
                    }
                    return sum_sq / n;
                }

            } // anonymous namespace

            std::vector<BenchmarkResult> benchmark_tl2_kernels(
                uint32_t M,
                uint32_t K,
                int iterations)
            {
                std::vector<BenchmarkResult> results;

                // Generate test data
                auto weights = generate_random_weights(M, K);
                auto activations = generate_random_activations(K);

                // Reference output (naive)
                std::vector<float> ref_output(M, 0.0f);
                naive_ternary_matmul(weights, activations, ref_output.data(), M, 1, K);

                // Time naive baseline
                double naive_time_us = 0.0;
                {
                    std::vector<float> out(M, 0.0f);
                    PerfTimer timer;
                    for (int i = 0; i < iterations; ++i)
                    {
                        naive_ternary_matmul(weights, activations, out.data(), M, 1, K);
                    }
                    naive_time_us = timer.elapsed_us() / iterations;
                }

                // List of kernels to benchmark
                struct KernelEntry
                {
                    KernelType type;
                    bool available;
                };

                std::vector<KernelEntry> kernels_to_test = {
                    {KernelType::NAIVE, true},
                    {KernelType::TL2_LUT_SCALAR, true},
                    {KernelType::TL2_LUT_OMP, g_cpu_caps.supports_tl2_omp()},
                    {KernelType::TL2_LUT_AVX2, g_cpu_caps.supports_tl2_avx2()},
                    {KernelType::AVX512_VNNI, g_cpu_caps.supports_avx512_vnni()},
                };

                for (const auto &entry : kernels_to_test)
                {
                    if (!entry.available)
                        continue;

                    std::vector<float> out(M, 0.0f);

                    // Warmup
                    parallel_ternary_gemv(weights, activations, out.data(), M, K, entry.type);

                    // Benchmark
                    PerfTimer timer;
                    for (int i = 0; i < iterations; ++i)
                    {
                        parallel_ternary_gemv(weights, activations, out.data(), M, K, entry.type);
                    }
                    double avg_us = timer.elapsed_us() / iterations;

                    // Accuracy check
                    double mse = compute_output_mse(ref_output.data(), out.data(), M);

                    int num_threads = 1;
#ifdef _OPENMP
                    num_threads = omp_get_max_threads();
#endif

                    BenchmarkResult result;
                    result.kernel = entry.type;
                    result.M = M;
                    result.N = 1;
                    result.K = K;
                    result.time_us = avg_us;
                    result.gflops = (2.0 * M * K) / (avg_us * 1000.0);
                    result.speedup_vs_naive = naive_time_us / avg_us;
                    result.mse = mse;
                    result.num_threads = num_threads;

                    results.push_back(result);
                }

                return results;
            }

            std::vector<BenchmarkResult> benchmark_tl2_gemm(
                uint32_t M,
                const std::vector<uint32_t> &N_values,
                uint32_t K,
                int iterations)
            {
                std::vector<BenchmarkResult> results;

                auto weights = generate_random_weights(M, K);

                for (uint32_t N : N_values)
                {
                    // For GEMM, activations are [K × N] stored column-major
                    QuantizedActivation activations;
                    activations.values.resize(static_cast<size_t>(K) * N);
                    activations.scale = 0.05f;
                    activations.zero_point = 0;

                    std::mt19937 rng(456);
                    std::uniform_int_distribution<int> dist(-127, 127);
                    for (auto &v : activations.values)
                    {
                        v = static_cast<int8_t>(dist(rng));
                    }

                    // Reference
                    std::vector<float> ref_output(static_cast<size_t>(M) * N, 0.0f);
                    naive_ternary_matmul(weights, activations, ref_output.data(), M, N, K);

                    // Time naive
                    double naive_time_us = 0.0;
                    {
                        std::vector<float> out(static_cast<size_t>(M) * N, 0.0f);
                        PerfTimer timer;
                        for (int i = 0; i < iterations; ++i)
                        {
                            naive_ternary_matmul(weights, activations,
                                                 out.data(), M, N, K);
                        }
                        naive_time_us = timer.elapsed_us() / iterations;
                    }

                    // Test available kernels
                    std::vector<KernelType> types_to_test = {
                        KernelType::NAIVE,
                        KernelType::TL2_LUT_SCALAR,
                    };

                    if (g_cpu_caps.supports_tl2_omp())
                        types_to_test.push_back(KernelType::TL2_LUT_OMP);
                    if (g_cpu_caps.supports_tl2_avx2())
                        types_to_test.push_back(KernelType::TL2_LUT_AVX2);
                    if (g_cpu_caps.supports_avx512_vnni())
                        types_to_test.push_back(KernelType::AVX512_VNNI);

                    for (KernelType type : types_to_test)
                    {
                        std::vector<float> out(static_cast<size_t>(M) * N, 0.0f);

                        // Warmup
                        parallel_ternary_gemm(weights, activations, out.data(),
                                              M, N, K, type);

                        // Time
                        PerfTimer timer;
                        for (int i = 0; i < iterations; ++i)
                        {
                            parallel_ternary_gemm(weights, activations, out.data(),
                                                  M, N, K, type);
                        }
                        double avg_us = timer.elapsed_us() / iterations;

                        double mse = compute_output_mse(ref_output.data(), out.data(),
                                                        M * N);

                        int num_threads = 1;
#ifdef _OPENMP
                        num_threads = omp_get_max_threads();
#endif

                        BenchmarkResult result;
                        result.kernel = type;
                        result.M = M;
                        result.N = N;
                        result.K = K;
                        result.time_us = avg_us;
                        result.gflops = (2.0 * M * N * K) / (avg_us * 1000.0);
                        result.speedup_vs_naive = naive_time_us / avg_us;
                        result.mse = mse;
                        result.num_threads = num_threads;

                        results.push_back(result);
                    }
                }

                return results;
            }

        } // namespace kernels
    } // namespace bitnet
} // namespace ryzanstein_llm
