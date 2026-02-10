/*
 * Ryzanstein LLM - TL2_0 LUT-Based GEMM Implementation
 * [REF:IP-S5-001] - BitNet January 2026 Parallel Kernel Integration
 *
 * Ternary Lookup 2.0 (TL2_0):
 *   Replaces multiply-accumulate with table lookup + accumulate.
 *   Weights ∈ {-1, 0, +1} → 3-entry LUT per activation value.
 *   Eliminates ALL multiplications from the inner loop.
 *
 * Performance hierarchy:
 *   1. AVX2 + OpenMP parallel (best)
 *   2. OpenMP parallel scalar
 *   3. Scalar sequential (fallback)
 *
 * Copyright (c) 2025-2026 Ryzanstein LLM Project
 * Licensed under MIT License
 */

#include "lut_gemm.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ryzanstein_llm
{
    namespace bitnet
    {
        namespace kernels
        {

            // ================================================================
            // Internal helpers
            // ================================================================

            namespace
            {

                /**
                 * Build LUT for a single weight group.
                 *
                 * Each weight group has its own scale factor. The LUT encodes:
                 *   LUT[act_val + 128][w + 1] = dequant(act_val) * w * weight_scale
                 *
                 * @param lut Output TernaryLUT
                 * @param act_scale Activation quantization scale
                 * @param act_zero_point Activation zero point
                 * @param weight_scale Per-group weight scale
                 */
                inline void build_lut_for_group(
                    TernaryLUT &lut,
                    float act_scale,
                    int8_t act_zero_point,
                    float weight_scale)
                {
                    lut.init(act_scale, act_zero_point, weight_scale);
                }

                /**
                 * Compute the weight group index for a given K position.
                 *
                 * @param k Column index in weight matrix
                 * @param group_size Number of elements per weight group
                 * @return Group index
                 */
                inline uint32_t weight_group_index(uint32_t k, uint32_t group_size)
                {
                    return k / group_size;
                }

                /**
                 * Accumulate one row of the ternary matmul using LUT.
                 *
                 * Inner loop: output += LUT[activations[k]][weights[row * K + k]]
                 * for k in [k_start, k_end).
                 *
                 * @param lut Pre-computed lookup table for this group
                 * @param weights_row Pointer to weights for this row
                 * @param act_values Pointer to activation values
                 * @param k_start Start of K range
                 * @param k_end End of K range (exclusive)
                 * @return Accumulated partial sum
                 */
                inline float accumulate_row_lut(
                    const TernaryLUT &lut,
                    const int8_t *weights_row,
                    const int8_t *act_values,
                    uint32_t k_start,
                    uint32_t k_end)
                {
                    float sum = 0.0f;
                    for (uint32_t k = k_start; k < k_end; ++k)
                    {
                        sum += lut.lookup(act_values[k], weights_row[k]);
                    }
                    return sum;
                }

#ifdef __AVX2__
                /**
                 * AVX2-accelerated LUT accumulation for 8 elements at a time.
                 *
                 * Uses scalar lookups but accumulates with AVX2 vectors.
                 * This avoids gather instruction latency while still benefiting
                 * from vectorized addition.
                 *
                 * @param lut Pre-computed lookup table
                 * @param weights_row Pointer to weight row
                 * @param act_values Pointer to activation values
                 * @param k_start Start index
                 * @param k_end End index
                 * @return Accumulated partial sum
                 */
                inline float accumulate_row_lut_avx2(
                    const TernaryLUT &lut,
                    const int8_t *weights_row,
                    const int8_t *act_values,
                    uint32_t k_start,
                    uint32_t k_end)
                {
                    __m256 vsum = _mm256_setzero_ps();
                    uint32_t k = k_start;

                    // Process 8 elements at a time
                    uint32_t k_vec_end = k_start + ((k_end - k_start) / 8) * 8;

                    for (; k < k_vec_end; k += 8)
                    {
                        // Scalar lookups (LUT access is irregular)
                        float vals[8];
                        vals[0] = lut.lookup(act_values[k + 0], weights_row[k + 0]);
                        vals[1] = lut.lookup(act_values[k + 1], weights_row[k + 1]);
                        vals[2] = lut.lookup(act_values[k + 2], weights_row[k + 2]);
                        vals[3] = lut.lookup(act_values[k + 3], weights_row[k + 3]);
                        vals[4] = lut.lookup(act_values[k + 4], weights_row[k + 4]);
                        vals[5] = lut.lookup(act_values[k + 5], weights_row[k + 5]);
                        vals[6] = lut.lookup(act_values[k + 6], weights_row[k + 6]);
                        vals[7] = lut.lookup(act_values[k + 7], weights_row[k + 7]);

                        __m256 vvals = _mm256_loadu_ps(vals);
                        vsum = _mm256_add_ps(vsum, vvals);
                    }

                    // Horizontal sum of AVX2 vector
                    // [a0 a1 a2 a3 | a4 a5 a6 a7]
                    __m128 hi = _mm256_extractf128_ps(vsum, 1);
                    __m128 lo = _mm256_castps256_ps128(vsum);
                    __m128 sum128 = _mm_add_ps(lo, hi);   // [a0+a4, a1+a5, a2+a6, a3+a7]
                    sum128 = _mm_hadd_ps(sum128, sum128); // [a0+a4+a1+a5, a2+a6+a3+a7, ...]
                    sum128 = _mm_hadd_ps(sum128, sum128); // [total, ...]
                    float result = _mm_cvtss_f32(sum128);

                    // Scalar tail
                    for (; k < k_end; ++k)
                    {
                        result += lut.lookup(act_values[k], weights_row[k]);
                    }

                    return result;
                }
#endif // __AVX2__

            } // anonymous namespace

            // ================================================================
            // TL2_0 Scalar GEMV
            // ================================================================

            void tl2_lut_gemv_scalar(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t K,
                const CacheConfig &cache)
            {
                assert(weights.rows >= M && "Weight matrix rows mismatch");
                assert(weights.cols >= K && "Weight matrix cols mismatch");
                assert(activations.values.size() >= K && "Activation vector size mismatch");

                const int8_t *w_data = weights.values.data();
                const int8_t *a_data = activations.values.data();
                const uint32_t group_size = weights.group_size;

                // Auto-tune K tile from cache config
                TileConfig tiles = TileConfig::from_cache(cache);
                const uint32_t k_tile = tiles.tile_k;

                // Process K in tiles for cache locality
                std::memset(output, 0, M * sizeof(float));

                for (uint32_t k_base = 0; k_base < K; k_base += k_tile)
                {
                    uint32_t k_end = std::min(k_base + k_tile, K);

                    // Determine weight group range for this K tile
                    uint32_t group_start = weight_group_index(k_base, group_size);
                    uint32_t group_end = weight_group_index(k_end - 1, group_size);

                    // For each group in this K tile, build and use LUT
                    for (uint32_t g = group_start; g <= group_end; ++g)
                    {
                        uint32_t g_k_start = std::max(k_base, g * group_size);
                        uint32_t g_k_end = std::min(k_end, (g + 1) * group_size);
                        g_k_end = std::min(g_k_end, K);

                        float weight_scale = weights.get_scale(g);

                        TernaryLUT lut;
                        build_lut_for_group(lut, activations.scale,
                                            activations.zero_point, weight_scale);

                        // Accumulate across all output rows
                        for (uint32_t m = 0; m < M; ++m)
                        {
                            const int8_t *w_row = w_data + static_cast<size_t>(m) * K;
                            output[m] += accumulate_row_lut(
                                lut, w_row, a_data, g_k_start, g_k_end);
                        }
                    }
                }
            }

            // ================================================================
            // TL2_0 Parallel GEMV (OpenMP)
            // ================================================================

            void tl2_lut_gemv_parallel(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t K,
                int num_threads,
                const CacheConfig &cache)
            {
                assert(weights.rows >= M && "Weight matrix rows mismatch");
                assert(weights.cols >= K && "Weight matrix cols mismatch");
                assert(activations.values.size() >= K && "Activation vector size mismatch");

#ifdef _OPENMP
                if (num_threads <= 0)
                {
                    num_threads = omp_get_max_threads();
                }
#else
                num_threads = 1;
#endif

                // For small problems, fall back to scalar
                if (M < 64 || num_threads <= 1)
                {
                    tl2_lut_gemv_scalar(weights, activations, output, M, K, cache);
                    return;
                }

                const int8_t *w_data = weights.values.data();
                const int8_t *a_data = activations.values.data();
                const uint32_t group_size = weights.group_size;

                // Compute number of weight groups along K
                uint32_t num_groups = (K + group_size - 1) / group_size;

                // Pre-build LUTs for all weight groups (shared across threads)
                std::vector<TernaryLUT> luts(num_groups);
                for (uint32_t g = 0; g < num_groups; ++g)
                {
                    float weight_scale = weights.get_scale(g);
                    build_lut_for_group(luts[g], activations.scale,
                                        activations.zero_point, weight_scale);
                }

                std::memset(output, 0, M * sizeof(float));

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
                {
#pragma omp for schedule(dynamic, 16)
                    for (int m = 0; m < static_cast<int>(M); ++m)
                    {
                        const int8_t *w_row = w_data + static_cast<size_t>(m) * K;
                        float row_sum = 0.0f;

                        // Process each weight group
                        for (uint32_t g = 0; g < num_groups; ++g)
                        {
                            uint32_t g_k_start = g * group_size;
                            uint32_t g_k_end = std::min((g + 1) * group_size, K);

                            // Prefetch next weight row segment
                            if (g + 1 < num_groups)
                            {
                                prefetch_l1(w_row + (g + 1) * group_size);
                            }

                            row_sum += accumulate_row_lut(
                                luts[g], w_row, a_data, g_k_start, g_k_end);
                        }

                        output[m] = row_sum;
                    }
                }
#else
                // Sequential fallback (same logic without OpenMP)
                for (uint32_t m = 0; m < M; ++m)
                {
                    const int8_t *w_row = w_data + static_cast<size_t>(m) * K;
                    float row_sum = 0.0f;

                    for (uint32_t g = 0; g < num_groups; ++g)
                    {
                        uint32_t g_k_start = g * group_size;
                        uint32_t g_k_end = std::min((g + 1) * group_size, K);

                        row_sum += accumulate_row_lut(
                            luts[g], w_row, a_data, g_k_start, g_k_end);
                    }

                    output[m] = row_sum;
                }
#endif
            }

            // ================================================================
            // TL2_0 Tiled GEMM
            // ================================================================

            void tl2_lut_gemm_tiled(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t N,
                uint32_t K,
                int num_threads,
                const CacheConfig &cache)
            {
                // For N=1, delegate to GEMV
                if (N == 1)
                {
                    tl2_lut_gemv_parallel(weights, activations, output,
                                          M, K, num_threads, cache);
                    return;
                }

                assert(weights.rows >= M && "Weight matrix rows mismatch");
                assert(weights.cols >= K && "Weight matrix cols mismatch");

#ifdef _OPENMP
                if (num_threads <= 0)
                {
                    num_threads = omp_get_max_threads();
                }
#else
                num_threads = 1;
#endif

                const int8_t *w_data = weights.values.data();
                const int8_t *a_data = activations.values.data();
                const uint32_t group_size = weights.group_size;
                uint32_t num_groups = (K + group_size - 1) / group_size;

                // Auto-tune tile sizes
                TileConfig tiles = TileConfig::from_cache(cache);
                const uint32_t tile_m = tiles.tile_m;
                const uint32_t tile_k = tiles.tile_k;
                // N tile: process all columns within each M×K tile
                const uint32_t tile_n = std::min(N, static_cast<uint32_t>(32));

                // Pre-build per-group LUTs for each batch column
                // For GEMM, each column n may have different activation quantization params
                // For simplicity, assume uniform quantization across batch (common case)
                std::vector<TernaryLUT> luts(num_groups);
                for (uint32_t g = 0; g < num_groups; ++g)
                {
                    float weight_scale = weights.get_scale(g);
                    build_lut_for_group(luts[g], activations.scale,
                                        activations.zero_point, weight_scale);
                }

                // Zero output
                std::memset(output, 0, static_cast<size_t>(M) * N * sizeof(float));

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
                {
                    // Tile over M (rows of output)
                    for (uint32_t m_base = 0; m_base < M; m_base += tile_m)
                    {
                        uint32_t m_end = std::min(m_base + tile_m, M);

                        // Tile over N (columns of output / batch)
                        for (uint32_t n_base = 0; n_base < N; n_base += tile_n)
                        {
                            uint32_t n_end = std::min(n_base + tile_n, N);

                            // Tile over K (inner dimension)
                            for (uint32_t k_base = 0; k_base < K; k_base += tile_k)
                            {
                                uint32_t k_end = std::min(k_base + tile_k, K);

                                // Determine groups for this K tile
                                uint32_t g_start = weight_group_index(k_base, group_size);
                                uint32_t g_end = weight_group_index(k_end - 1, group_size);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8) nowait
#endif
                                for (int m = static_cast<int>(m_base); m < static_cast<int>(m_end); ++m)
                                {
                                    const int8_t *w_row = w_data + static_cast<size_t>(m) * K;

                                    for (uint32_t n = n_base; n < n_end; ++n)
                                    {
                                        const int8_t *a_col = a_data + static_cast<size_t>(n) * K;
                                        float partial = 0.0f;

                                        for (uint32_t g = g_start; g <= g_end; ++g)
                                        {
                                            uint32_t gk_start = std::max(k_base, g * group_size);
                                            uint32_t gk_end = std::min(k_end, (g + 1) * group_size);
                                            gk_end = std::min(gk_end, K);

                                            partial += accumulate_row_lut(
                                                luts[g], w_row, a_col, gk_start, gk_end);
                                        }

                                        output[static_cast<size_t>(m) * N + n] += partial;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ================================================================
            // AVX2-accelerated LUT GEMV
            // ================================================================

            void tl2_lut_gemv_avx2(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t K)
            {
#ifdef __AVX2__
                const int8_t *w_data = weights.values.data();
                const int8_t *a_data = activations.values.data();
                const uint32_t group_size = weights.group_size;
                uint32_t num_groups = (K + group_size - 1) / group_size;

                // Pre-build LUTs
                std::vector<TernaryLUT> luts(num_groups);
                for (uint32_t g = 0; g < num_groups; ++g)
                {
                    float weight_scale = weights.get_scale(g);
                    build_lut_for_group(luts[g], activations.scale,
                                        activations.zero_point, weight_scale);
                }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
                for (int m = 0; m < static_cast<int>(M); ++m)
                {
                    const int8_t *w_row = w_data + static_cast<size_t>(m) * K;
                    float row_sum = 0.0f;

                    for (uint32_t g = 0; g < num_groups; ++g)
                    {
                        uint32_t g_k_start = g * group_size;
                        uint32_t g_k_end = std::min((g + 1) * group_size, K);

                        // Prefetch next weight group
                        if (g + 1 < num_groups)
                        {
                            prefetch_l1(w_row + (g + 1) * group_size);
                        }

                        row_sum += accumulate_row_lut_avx2(
                            luts[g], w_row, a_data, g_k_start, g_k_end);
                    }

                    output[m] = row_sum;
                }
#else
                // No AVX2: fall back to parallel scalar
                CacheConfig cache;
                tl2_lut_gemv_parallel(weights, activations, output, M, K, 0, cache);
#endif
            }

            // ================================================================
            // Runtime Dispatchers
            // ================================================================

            void dispatch_tl2_gemv(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t K)
            {
                PerfTimer timer;

#ifdef __AVX2__
                // Best path: AVX2 + OpenMP
                tl2_lut_gemv_avx2(weights, activations, output, M, K);
#elif defined(_OPENMP)
                // OpenMP parallel scalar
                CacheConfig cache;
                tl2_lut_gemv_parallel(weights, activations, output, M, K, 0, cache);
#else
                // Sequential scalar fallback
                CacheConfig cache;
                tl2_lut_gemv_scalar(weights, activations, output, M, K, cache);
#endif

                double elapsed_us = timer.elapsed_us();
                // M*K multiply-accumulate operations (but we use LUT, so effective)
                double effective_flops = 2.0 * M * K;
                double gflops = effective_flops / (elapsed_us * 1000.0);

                // Could log via MatmulStats here if needed
                (void)gflops;
            }

            void dispatch_tl2_gemm(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t N,
                uint32_t K)
            {
                PerfTimer timer;
                CacheConfig cache;

                tl2_lut_gemm_tiled(weights, activations, output,
                                   M, N, K, 0, cache);

                double elapsed_us = timer.elapsed_us();
                double effective_flops = 2.0 * M * N * K;
                double gflops = effective_flops / (elapsed_us * 1000.0);

                (void)gflops;
            }

        } // namespace kernels
    } // namespace bitnet
} // namespace ryzanstein_llm
