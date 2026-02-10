/*
 * Ryzanstein LLM - TL2_0 LUT-Based GEMM for Ternary Weights
 * [REF:IP-S5-001] - BitNet January 2026 Parallel Kernel Integration
 *
 * Implements Ternary Lookup 2.0 (TL2_0) element-wise LUT method:
 * Since weights are constrained to {-1, 0, +1}, we replace
 * multiply-accumulate with table lookup + accumulate.
 *
 * For each activation value x:
 *   w=-1 → -x
 *   w= 0 →  0
 *   w=+1 → +x
 *
 * This eliminates all multiplications in the inner loop.
 *
 * Performance Target: 1.15-2.1x speedup over naive scalar implementation
 *
 * Copyright (c) 2025-2026 Ryzanstein LLM Project
 * Licensed under MIT License
 */

#pragma once

#include "../quantize.h"
#include "../optimization_utils.h"
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace ryzanstein_llm
{
    namespace bitnet
    {
        namespace kernels
        {

            // ================================================================
            // Cache Topology Configuration
            // ================================================================

            /**
             * Cache hierarchy configuration for tile size auto-tuning.
             *
             * Default values target AMD Ryzen 9 7950X:
             *   L1: 32KB per core
             *   L2: 512KB per core (1MB per CCD)
             *   L3: 32MB per CCD (64MB total)
             */
            struct CacheConfig
            {
                uint32_t l1_size_kb; // L1 data cache per core (KB)
                uint32_t l2_size_kb; // L2 cache per core (KB)
                uint32_t l3_size_kb; // L3 cache per CCD (KB)
                uint32_t cache_line; // Cache line size (bytes)
                uint32_t num_cores;  // Available cores for parallelism

                CacheConfig()
                    : l1_size_kb(32),
                      l2_size_kb(512),
                      l3_size_kb(32768),
                      cache_line(64),
                      num_cores(1)
                {
#ifdef _OPENMP
                    num_cores = static_cast<uint32_t>(omp_get_max_threads());
#endif
                }

                /**
                 * Compute optimal tile size for a given cache level.
                 *
                 * For an M×K tile of int8_t weights + K float activations + M float outputs:
                 *   memory = M*K*1 + K*4 + M*4  bytes
                 *
                 * We solve for tile_dim such that tile_dim^2 + 8*tile_dim ≤ cache_bytes
                 *
                 * @param cache_bytes Target cache budget in bytes
                 * @return Optimal square tile dimension
                 */
                uint32_t compute_tile_dim(uint32_t cache_bytes) const
                {
                    // Quadratic: tile^2 + 8*tile <= budget
                    // tile <= (-8 + sqrt(64 + 4*budget)) / 2
                    float discriminant = 64.0f + 4.0f * static_cast<float>(cache_bytes);
                    float tile_f = (-8.0f + std::sqrt(discriminant)) / 2.0f;
                    uint32_t tile = static_cast<uint32_t>(tile_f);
                    // Round down to cache line alignment
                    tile = (tile / (cache_line)) * (cache_line);
                    return std::max(tile, static_cast<uint32_t>(16)); // Minimum 16
                }
            };

            /**
             * Tiling parameters computed from cache topology.
             */
            struct TileConfig
            {
                uint32_t tile_m; // Row tile for output
                uint32_t tile_n; // Column tile for output (batch)
                uint32_t tile_k; // Inner dimension tile

                TileConfig() : tile_m(32), tile_n(1), tile_k(256) {}

                /**
                 * Auto-tune tile sizes from cache configuration.
                 */
                static TileConfig from_cache(const CacheConfig &cache)
                {
                    TileConfig tc;

                    // L1-resident tile for inner loop
                    uint32_t l1_bytes = cache.l1_size_kb * 1024 / 2; // Use half L1
                    tc.tile_k = cache.compute_tile_dim(l1_bytes);
                    tc.tile_k = std::min(tc.tile_k, static_cast<uint32_t>(512));

                    // L2-resident tile for row blocking
                    uint32_t l2_bytes = cache.l2_size_kb * 1024 / 2;
                    tc.tile_m = cache.compute_tile_dim(l2_bytes);
                    tc.tile_m = std::min(tc.tile_m, static_cast<uint32_t>(256));

                    // Batch tile (N=1 for GEMV, scale with L3 for GEMM)
                    tc.tile_n = 1; // Default GEMV

                    return tc;
                }
            };

            // ================================================================
            // TL2_0 Lookup Table
            // ================================================================

            /**
             * Pre-computed lookup table for ternary × INT8 products.
             *
             * For a given INT8 activation value x ∈ [-128, 127]:
             *   lut[x+128][0] = -x  (w = -1)
             *   lut[x+128][1] =  0  (w =  0)
             *   lut[x+128][2] = +x  (w = +1)
             *
             * Memory: 256 × 3 × 4 = 3072 bytes (fits in L1)
             *
             * Access pattern: result = lut[activation + 128][weight + 1]
             */
            struct TernaryLUT
            {
                // [256 activation values][3 weight values: -1, 0, +1 mapped to 0, 1, 2]
                float table[256][3];

                /**
                 * Initialize the lookup table with dequantized products.
                 *
                 * @param act_scale Activation scale factor
                 * @param act_zero_point Activation zero point
                 * @param weight_scale Weight scale factor for this group
                 */
                void init(float act_scale, int8_t act_zero_point, float weight_scale)
                {
                    for (int i = 0; i < 256; ++i)
                    {
                        int8_t quant_val = static_cast<int8_t>(i - 128);
                        float dequant = (static_cast<float>(quant_val) - act_zero_point) * act_scale;

                        // w = -1 → index 0
                        table[i][0] = -dequant * weight_scale;
                        // w =  0 → index 1
                        table[i][1] = 0.0f;
                        // w = +1 → index 2
                        table[i][2] = dequant * weight_scale;
                    }
                }

                /**
                 * Lookup the product of activation × weight.
                 *
                 * @param activation INT8 activation value [-128, 127]
                 * @param weight Ternary weight value {-1, 0, +1}
                 * @return Dequantized product
                 */
                inline float lookup(int8_t activation, int8_t weight) const
                {
                    uint8_t act_idx = static_cast<uint8_t>(activation + 128);
                    uint8_t w_idx = static_cast<uint8_t>(weight + 1); // Maps {-1,0,+1} → {0,1,2}
                    return table[act_idx][w_idx];
                }
            };

            // ================================================================
            // TL2_0 Scalar GEMV (Matrix-Vector Multiply)
            // ================================================================

            /**
             * TL2_0 LUT-based ternary matrix-vector multiplication (scalar).
             *
             * Computes: output[m] = Σ_k LUT[activations[k]][weights[m,k]]
             *
             * Eliminates all multiplications from the inner loop by
             * using pre-computed lookup tables.
             *
             * @param weights Ternary weight matrix [M × K]
             * @param activations INT8 quantized input vector [K]
             * @param output FP32 output vector [M]
             * @param M Number of output features
             * @param K Number of input features
             * @param cache Cache configuration for tiling
             */
            void tl2_lut_gemv_scalar(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t K,
                const CacheConfig &cache = CacheConfig());

            // ================================================================
            // TL2_0 Parallel GEMV (OpenMP Multi-threaded)
            // ================================================================

            /**
             * TL2_0 LUT-based ternary GEMV with OpenMP parallelism.
             *
             * Distributes rows across available CPU cores with:
             * - Dynamic scheduling for load balancing
             * - Thread-local LUT copies to avoid false sharing
             * - Prefetching for weight rows ahead of computation
             *
             * @param weights Ternary weight matrix [M × K]
             * @param activations INT8 quantized input vector [K]
             * @param output FP32 output vector [M]
             * @param M Number of output features
             * @param K Number of input features
             * @param num_threads Number of threads (0 = auto-detect)
             * @param cache Cache configuration for tiling
             */
            void tl2_lut_gemv_parallel(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t K,
                int num_threads = 0,
                const CacheConfig &cache = CacheConfig());

            // ================================================================
            // TL2_0 GEMM (Matrix-Matrix Multiply)
            // ================================================================

            /**
             * TL2_0 LUT-based ternary matrix multiplication with tiling.
             *
             * Computes: output[M × N] = weights[M × K] × activations[K × N]
             *
             * Uses cache-aware tiling:
             * - L1 tile: Inner K dimension (register-level reuse)
             * - L2 tile: M dimension (row blocking)
             * - L3 tile: N dimension (batch blocking)
             *
             * @param weights Ternary weight matrix [M × K]
             * @param activations INT8 quantized activations [K × N]
             * @param output FP32 output matrix [M × N]
             * @param M Number of output rows
             * @param N Number of output columns (batch size)
             * @param K Inner dimension
             * @param num_threads Number of threads (0 = auto-detect)
             * @param cache Cache configuration for tiling
             */
            void tl2_lut_gemm_tiled(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t N,
                uint32_t K,
                int num_threads = 0,
                const CacheConfig &cache = CacheConfig());

            // ================================================================
            // AVX2-accelerated LUT GEMV
            // ================================================================

            /**
             * TL2_0 LUT-based GEMV with AVX2 acceleration.
             *
             * Uses gather instructions to perform 8 simultaneous LUT lookups,
             * then horizontal sums via AVX2 vector operations.
             *
             * Requirements: AVX2 + FMA
             *
             * @param weights Ternary weight matrix [M × K]
             * @param activations INT8 quantized input vector [K]
             * @param output FP32 output vector [M]
             * @param M Number of output features
             * @param K Number of input features
             */
            void tl2_lut_gemv_avx2(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t K);

            // ================================================================
            // Dispatcher
            // ================================================================

            /**
             * Runtime-dispatched TL2_0 GEMV.
             *
             * Selects the best available implementation:
             * 1. AVX2 + OpenMP parallel (best on modern AMD/Intel)
             * 2. OpenMP parallel scalar (no SIMD)
             * 3. Scalar sequential (fallback)
             *
             * @param weights Ternary weight matrix [M × K]
             * @param activations INT8 quantized input vector [K]
             * @param output FP32 output vector [M]
             * @param M Number of output features
             * @param K Number of input features
             */
            void dispatch_tl2_gemv(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t K);

            /**
             * Runtime-dispatched TL2_0 GEMM.
             *
             * @param weights Ternary weight matrix [M × K]
             * @param activations INT8 quantized activations [K × N]
             * @param output FP32 output matrix [M × N]
             * @param M Number of output rows
             * @param N Number of output columns (batch size)
             * @param K Inner dimension
             */
            void dispatch_tl2_gemm(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M,
                uint32_t N,
                uint32_t K);

        } // namespace kernels
    } // namespace bitnet
} // namespace ryzanstein_llm
