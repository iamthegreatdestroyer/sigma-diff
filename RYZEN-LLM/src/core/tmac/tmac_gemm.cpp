/**
 * @file tmac_gemm.cpp
 * @brief Implementation of AVX-512 optimized T-MAC GEMM
 *
 * [REF:TMAC-006] - AVX-512 GEMM Implementation
 */

#include "tmac_gemm.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace ryzanstein_llm
{
    namespace tmac
    {

        using namespace std::chrono;

        // ============================================================================
        // CONSTRUCTOR & INITIALIZATION
        // ============================================================================

        TMACGemm::TMACGemm(
            std::shared_ptr<LUTLookup> lut_engine,
            const GEMMConfig &config)
            : lut_engine_(std::move(lut_engine)), config_(config)
        {
            if (!lut_engine_)
            {
                throw std::invalid_argument("LUT engine cannot be null");
            }

            // Auto-detect AVX-512 support
            if (config_.use_avx512)
            {
                config_.use_avx512 = detect_avx512_support();
            }

            std::cout << "Initialized TMACGemm\n";
            std::cout << "  AVX-512: " << (config_.use_avx512 ? "Enabled" : "Disabled") << "\n";
            std::cout << "  Block sizes: M=" << config_.block_m
                      << ", N=" << config_.block_n
                      << ", K=" << config_.block_k << "\n";
        }

        bool TMACGemm::detect_avx512_support()
        {
#ifdef __AVX512F__
            // Check CPUID for AVX-512 support
            // This is a simplified check - full detection would use CPUID instruction
            return true;
#else
            return false;
#endif
        }

        // ============================================================================
        // MAIN GEMM IMPLEMENTATION
        // ============================================================================

        void TMACGemm::gemm(
            const int8_t *W,
            const int8_t *X,
            int32_t *Y,
            uint32_t M,
            uint32_t K,
            uint32_t N)
        {
            if (K % 16 != 0)
            {
                throw std::invalid_argument("K must be divisible by 16 (T-MAC group size)");
            }

            auto start = high_resolution_clock::now();

            // Zero output buffer
            std::memset(Y, 0, M * N * sizeof(int32_t));

            // Blocked GEMM for cache efficiency
            for (uint32_t m_block = 0; m_block < M; m_block += config_.block_m)
            {
                uint32_t m_end = std::min(m_block + config_.block_m, M);

                for (uint32_t n_block = 0; n_block < N; n_block += config_.block_n)
                {
                    uint32_t n_end = std::min(n_block + config_.block_n, N);

                    // Process this block
                    gemm_block(W, X, Y, m_block, m_end, n_block, n_end, K, N);
                }
            }

            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start).count();

            // Update statistics
            stats_.gemm_count++;
            stats_.total_flops += static_cast<uint64_t>(M) * N * K * 2; // MAC = 2 FLOPs
            stats_.total_time_ms += duration / 1000.0;
        }

        void TMACGemm::gemm_block(
            const int8_t *W,
            const int8_t *X,
            int32_t *Y,
            uint32_t m_start, uint32_t m_end,
            uint32_t n_start, uint32_t n_end,
            uint32_t K,
            uint32_t N)
        {
            // Process rows in this block
            for (uint32_t m = m_start; m < m_end; ++m)
            {
                const int8_t *W_row = W + m * K;
                int32_t *Y_row = Y + m * N;

#ifdef __AVX512F__
                if (config_.use_avx512)
                {
                    // AVX-512 vectorized path
                    gemm_inner_avx512(W_row, X + n_start * K, Y_row + n_start, K, n_end - n_start);
                }
                else
                {
                    // Scalar fallback
                    gemm_inner_scalar(W_row, X + n_start * K, Y_row + n_start, K, n_end - n_start);
                }
#else
                // Scalar fallback
                gemm_inner_scalar(W_row, X + n_start * K, Y_row + n_start, K, n_end - n_start);
#endif
            }
        }

        // ============================================================================
        // SCALAR INNER LOOP (FALLBACK)
        // ============================================================================

        void TMACGemm::gemm_inner_scalar(
            const int8_t *W_row,
            const int8_t *X,
            int32_t *Y_row,
            uint32_t K,
            uint32_t N)
        {
            // For each output column
            for (uint32_t n = 0; n < N; ++n)
            {
                const int8_t *X_col = X + n * K;
                int32_t accumulator = 0;

                // Process K in groups of 16 (T-MAC group size)
                uint32_t num_groups = K / 16;

                for (uint32_t g = 0; g < num_groups; ++g)
                {
                    // Extract pattern for this group
                    TernaryPattern pattern;
                    for (uint32_t i = 0; i < 16; ++i)
                    {
                        pattern[i] = W_row[g * 16 + i];
                    }

                    // Lookup results for all 16 activations in this group
                    for (uint32_t i = 0; i < 16; ++i)
                    {
                        int8_t activation = X_col[g * 16 + i];
                        int32_t result = lut_engine_->lookup(pattern, activation);
                        accumulator += result;
                    }
                }

                Y_row[n] = accumulator;
            }
        }

#ifdef __AVX512F__
        // ============================================================================
        // AVX-512 OPTIMIZED INNER LOOP
        // ============================================================================

        void TMACGemm::gemm_inner_avx512(
            const int8_t *W_row,
            const int8_t *X,
            int32_t *Y_row,
            uint32_t K,
            uint32_t N)
        {
            // Process 16 columns at a time using AVX-512
            uint32_t n_vec = (N / 16) * 16; // Round down to multiple of 16

            // Vectorized loop
            for (uint32_t n = 0; n < n_vec; n += 16)
            {
                // 16× INT32 accumulators
                __m512i acc = _mm512_setzero_si512();

                uint32_t num_groups = K / 16;

                for (uint32_t g = 0; g < num_groups; ++g)
                {
                    // Extract pattern
                    TernaryPattern pattern;
                    for (uint32_t i = 0; i < 16; ++i)
                    {
                        pattern[i] = W_row[g * 16 + i];
                    }

                    // Batch lookup for 16 columns × 16 activations
                    // For now, use scalar lookups (can be optimized further)
                    for (uint32_t i = 0; i < 16; ++i)
                    {
                        // Load 16 activations (one per column)
                        int8_t acts_bytes[16];
                        for (int j = 0; j < 16; ++j)
                        {
                            acts_bytes[j] = X[(n + j) * K + g * 16 + i];
                        }
                        __m128i acts_128 = _mm_loadu_si128((__m128i *)acts_bytes);
                        __m256i activations_lo = _mm256_cvtepi8_epi32(acts_128);

                        // Lookup results (scalar for now - can be vectorized)
                        int32_t results[16];
                        for (int j = 0; j < 16; ++j)
                        {
                            int8_t act = X[(n + j) * K + g * 16 + i];
                            results[j] = lut_engine_->lookup(pattern, act);
                        }

                        // Load results and accumulate
                        __m512i results_vec = _mm512_loadu_si512(results);
                        acc = _mm512_add_epi32(acc, results_vec);
                    }
                }

                // Store accumulated results
                _mm512_storeu_si512(Y_row + n, acc);
            }

            // Scalar remainder
            if (n_vec < N)
            {
                gemm_inner_scalar(W_row, X + n_vec * K, Y_row + n_vec, K, N - n_vec);
            }
        }
#endif

        // ============================================================================
        // BATCHED GEMM
        // ============================================================================

        void TMACGemm::gemm_batched(
            const int8_t **W_batch,
            const int8_t **X_batch,
            int32_t **Y_batch,
            uint32_t batch_size,
            uint32_t M,
            uint32_t K,
            uint32_t N)
        {
            for (uint32_t b = 0; b < batch_size; ++b)
            {
                gemm(W_batch[b], X_batch[b], Y_batch[b], M, K, N);
            }
        }

        // ============================================================================
        // STATISTICS
        // ============================================================================

        void TMACGemm::print_stats() const
        {
            if (stats_.gemm_count == 0)
            {
                std::cout << "No GEMMs performed yet.\n";
                return;
            }

            std::cout << "\nT-MAC GEMM Statistics\n";
            std::cout << "=====================\n";
            std::cout << "Total GEMMs: " << stats_.gemm_count << "\n";
            std::cout << "Total FLOPs: " << (stats_.total_flops / 1e9) << " GFLOPs\n";
            std::cout << "Total time: " << stats_.total_time_ms << " ms\n";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Average time: " << stats_.avg_time_ms() << " ms/GEMM\n";
            std::cout << "Throughput: " << stats_.gflops() << " GFLOPS\n";
            std::cout << "Target: >100 GFLOPS → "
                      << (stats_.gflops() > 100.0 ? "✓ PASS" : "✗ FAIL") << "\n";
        }

        // ============================================================================
        // ALIGNED MEMORY UTILITIES
        // ============================================================================

        void *aligned_alloc(size_t size, size_t alignment)
        {
#ifdef _WIN32
            return _aligned_malloc(size, alignment);
#else
            void *ptr = nullptr;
            if (posix_memalign(&ptr, alignment, size) != 0)
            {
                return nullptr;
            }
            return ptr;
#endif
        }

        void aligned_free(void *ptr)
        {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }

    } // namespace tmac
} // namespace ryzanstein_llm
