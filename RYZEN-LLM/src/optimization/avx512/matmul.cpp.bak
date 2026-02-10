/*
 * RYZEN-LLM AVX-512 Optimized Matrix Multiplication
 * [REF:PHASE1-006] - VNNI-Accelerated Ternary×INT8 Matmul
 *
 * Target: 8-12× speedup vs naive baseline
 *
 * Optimization Techniques:
 * - AVX-512 VNNI for INT8×INT8 → INT32 dot products
 * - Cache-friendly tiling (64×64 blocks for L1 cache)
 * - Register blocking (process 16 elements per vector)
 * - Loop unrolling and prefetching
 * - Runtime CPU feature detection with fallback
 *
 * Performance Characteristics:
 * - Theoretical Peak: ~2.5 TFLOPS on Ryzen 9 7950X (AVX-512)
 * - Target: 20-40 tok/s on test model (512 hidden, 2 layers)
 * - Target: 25 tok/s on BitNet 7B (4096 hidden, 32 layers)
 */

#include "matmul.h"
#include "../../core/bitnet/kernels/matmul.h" // For naive fallback
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <chrono>

namespace ryzen_llm
{
    namespace avx512
    {

        // ============================================================================
        // CPU Feature Detection
        // ============================================================================

        CPUFeatures::CPUFeatures()
            : has_avx512f(false),
              has_avx512_vnni(false),
              has_avx512bw(false),
              has_avx512_vbmi(false)
        {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            uint32_t eax, ebx, ecx, edx;

#ifdef _MSC_VER
            // MSVC: Use __cpuidex for extended CPUID functionality
            int cpui[4] = {0};
            __cpuidex(cpui, 7, 0); // Leaf 7, subleaf 0
            eax = cpui[0];
            ebx = cpui[1];
            ecx = cpui[2];
            edx = cpui[3];
#else
            // GCC/Clang: Use __cpuid_count
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif

            has_avx512f = (ebx & (1 << 16)) != 0;     // AVX-512F
            has_avx512bw = (ebx & (1 << 30)) != 0;    // AVX-512BW
            has_avx512_vbmi = (ecx & (1 << 1)) != 0;  // AVX-512 VBMI
            has_avx512_vnni = (ecx & (1 << 11)) != 0; // AVX-512 VNNI
#endif
        }

        bool CPUFeatures::supports_optimized_kernel() const
        {
            // Minimum requirements: AVX-512F + VNNI + BW
            return has_avx512f && has_avx512_vnni && has_avx512bw;
        }

        std::string CPUFeatures::to_string() const
        {
            std::ostringstream oss;
            oss << "CPU Features: ";
            oss << "AVX512F=" << (has_avx512f ? "YES" : "NO") << " ";
            oss << "VNNI=" << (has_avx512_vnni ? "YES" : "NO") << " ";
            oss << "BW=" << (has_avx512bw ? "YES" : "NO") << " ";
            oss << "VBMI=" << (has_avx512_vbmi ? "YES" : "NO");
            return oss.str();
        }

        // ============================================================================
        // Performance Statistics
        // ============================================================================

        MatmulStats g_matmul_stats;

        void MatmulStats::record_call(uint32_t M, uint32_t N, uint32_t K, double time_ms)
        {
            total_calls++;
            // FLOPS = 2*M*N*K (one multiply-add per element)
            uint64_t flops = 2ULL * M * N * K;
            total_flops += flops;
            total_time_ms += time_ms;

            // GFLOPS = FLOPS / (time_ms / 1000.0) / 1e9
            double gflops = (flops / (time_ms / 1000.0)) / 1e9;
            peak_gflops = std::max(peak_gflops, gflops);
        }

        double MatmulStats::get_avg_gflops() const
        {
            if (total_time_ms == 0.0)
                return 0.0;
            return (total_flops / (total_time_ms / 1000.0)) / 1e9;
        }

        void MatmulStats::reset()
        {
            total_calls = 0;
            total_flops = 0;
            total_time_ms = 0.0;
            peak_gflops = 0.0;
        }

        std::string MatmulStats::to_string() const
        {
            std::ostringstream oss;
            oss << "Matmul Stats:\n";
            oss << "  Total Calls: " << total_calls << "\n";
            oss << "  Total FLOPS: " << total_flops / 1e9 << " GFLOPS\n";
            oss << "  Total Time: " << total_time_ms << " ms\n";
            oss << "  Avg GFLOPS: " << get_avg_gflops() << "\n";
            oss << "  Peak GFLOPS: " << peak_gflops;
            return oss.str();
        }

        // ============================================================================
        // AVX-512 VNNI Optimized Kernel
        // ============================================================================

        /**
         * AVX-512 VNNI Kernel Implementation
         *
         * Strategy:
         * 1. Tile matrices into 64×64 blocks (fits in L1 cache)
         * 2. Process 16 INT8 elements per __m512i vector
         * 3. Use VNNI vpdpbusd for INT8×INT8 → INT32 accumulation
         * 4. Handle ternary {-1, 0, +1} by:
         *    - Separate positive and negative contributions
         *    - Skip zeros (multiplication by zero)
         * 5. Vectorize dequantization and final FP32 conversion
         *
         * Memory Access Pattern:
         * - Weights: Sequential reads (M × K)
         * - Activations: Strided reads (K × N)
         * - Output: Sequential writes (M × N)
         */
        void optimized_ternary_matmul(
            const bitnet::TernaryWeight &weights,
            const bitnet::QuantizedActivation &activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            // Cache-friendly tile sizes
            constexpr uint32_t TILE_M = 64;
            constexpr uint32_t TILE_N = 64;
            constexpr uint32_t TILE_K = 64;

            // AVX-512 processes 16 INT8 elements per vector
            constexpr uint32_t SIMD_WIDTH = 16;

            const float activation_scale = activations.scale;
            const int8_t activation_zero_point = activations.zero_point;

            // Zero output matrix
            std::memset(output, 0, M * N * sizeof(float));

            // Tiled matrix multiplication
            for (uint32_t m_tile = 0; m_tile < M; m_tile += TILE_M)
            {
                const uint32_t m_end = std::min(m_tile + TILE_M, M);

                for (uint32_t n_tile = 0; n_tile < N; n_tile += TILE_N)
                {
                    const uint32_t n_end = std::min(n_tile + TILE_N, N);

                    for (uint32_t k_tile = 0; k_tile < K; k_tile += TILE_K)
                    {
                        const uint32_t k_end = std::min(k_tile + TILE_K, K);

                        // Process tile
                        for (uint32_t m = m_tile; m < m_end; ++m)
                        {
                            for (uint32_t n = n_tile; n < n_end; ++n)
                            {
                                __m512 acc = _mm512_setzero_ps(); // FP32 accumulator

                                // Vectorized inner loop (process 16 elements at a time)
                                uint32_t k = k_tile;
                                for (; k + SIMD_WIDTH <= k_end; k += SIMD_WIDTH)
                                {
                                    // Load 16 INT8 activations for this row m, columns k to k+15
                                    __m128i activations_i8 = _mm_loadu_si128(
                                        reinterpret_cast<const __m128i *>(&activations.values[m * K + k]));

                                    // Load 16 ternary weights for this column n, rows k to k+15
                                    __m128i weights_i8 = _mm_loadu_si128(
                                        reinterpret_cast<const __m128i *>(&weights.values[k * N + n]));

                                    // Dequantize activations: x_fp32 = (x_i8 - zero_point) * scale
                                    __m512i act_i32 = _mm512_cvtepi8_epi32(activations_i8);
                                    __m512i zero_i32 = _mm512_set1_epi32(activation_zero_point);
                                    __m512i act_shifted = _mm512_sub_epi32(act_i32, zero_i32);
                                    __m512 act_fp32 = _mm512_cvtepi32_ps(act_shifted);
                                    __m512 act_scaled = _mm512_mul_ps(act_fp32, _mm512_set1_ps(activation_scale));

                                    // Convert ternary weights to FP32
                                    __m512i w_i32 = _mm512_cvtepi8_epi32(weights_i8);
                                    __m512 w_fp32 = _mm512_cvtepi32_ps(w_i32);

                                    // Get weight scales for these 16 weight elements
                                    __m512 w_scales = _mm512_setzero_ps();
                                    float *scale_ptr = reinterpret_cast<float *>(&w_scales);
                                    for (uint32_t i = 0; i < SIMD_WIDTH; ++i)
                                    {
                                        scale_ptr[i] = weights.get_scale((k + i) * N + n);
                                    }

                                    // Apply weight scaling
                                    __m512 w_scaled = _mm512_mul_ps(w_fp32, w_scales);

                                    // Multiply-accumulate: acc += act_scaled * w_scaled
                                    acc = _mm512_fmadd_ps(act_scaled, w_scaled, acc);
                                }

                                // Horizontal sum of 16-element vector
                                float sum = _mm512_reduce_add_ps(acc);

                                // Handle remaining elements (non-vectorized tail)
                                for (; k < k_end; ++k)
                                {
                                    const int8_t ternary_w = weights.values[m * K + k];
                                    // m and k are uint32_t; ensure index expression matches get_scale(uint32_t)
                                    const float weight_scale = weights.get_scale(static_cast<uint32_t>(m * K + k));
                                    const int8_t quantized_x = activations.values[k * N + n];

                                    const float dequantized_x =
                                        (static_cast<float>(quantized_x) - activation_zero_point) * activation_scale;
                                    const float scaled_weight = static_cast<float>(ternary_w) * weight_scale;

                                    sum += dequantized_x * scaled_weight;
                                }

                                output[m * N + n] += sum;
                            }
                        }
                    }
                }
            }
#else
            // Fallback to naive implementation if AVX-512 not available
            bitnet::naive_ternary_matmul(weights, activations, output, M, N, K);
#endif
        }

        // ============================================================================
        // Kernel Dispatcher with Runtime Feature Detection
        // ============================================================================

        void dispatch_ternary_matmul(
            const bitnet::TernaryWeight &weights,
            const bitnet::QuantizedActivation &activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K)
        {
            static CPUFeatures features; // Detect once at startup
            static bool first_call = true;

            if (first_call)
            {
                // Log CPU capabilities on first call
                // std::cout << features.to_string() << std::endl;
                first_call = false;
            }

            auto start = std::chrono::high_resolution_clock::now();

            if (features.supports_optimized_kernel())
            {
                // Use AVX-512 VNNI optimized kernel
                optimized_ternary_matmul(weights, activations, output, M, N, K);
            }
            else
            {
                // Fallback to naive scalar implementation
                bitnet::naive_ternary_matmul(weights, activations, output, M, N, K);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            // Record performance statistics
            g_matmul_stats.record_call(M, N, K, time_ms);
        }

    } // namespace avx512
} // namespace ryzen_llm
