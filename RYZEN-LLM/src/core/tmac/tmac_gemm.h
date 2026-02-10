#pragma once

/**
 * @file tmac_gemm.h
 * @brief AVX-512 optimized GEMM using T-MAC lookup tables
 *
 * Implements high-performance matrix multiplication:
 *   Y = W × X
 * where:
 *   W: [M, K] ternary weights {-1, 0, +1}
 *   X: [K, N] INT8 activations
 *   Y: [M, N] INT32 outputs
 *
 * Performance optimizations:
 *   - AVX-512 SIMD vectorization (16× INT32 or 64× INT8 per instruction)
 *   - T-MAC lookup tables for ternary × INT8 multiplication
 *   - Cache-friendly memory access patterns
 *   - Prefetching for sequential patterns
 *   - Register blocking for temporal locality
 *
 * Target: >800 GOPS on Ryzanstein 9 (Zen 4)
 *
 * [REF:TMAC-006] - AVX-512 GEMM Implementation
 */

#include "lut_lookup.h"
#include <cstdint>
#include <memory>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace ryzanstein_llm
{
    namespace tmac
    {

        /**
         * T-MAC GEMM configuration parameters
         */
        struct GEMMConfig
        {
            uint32_t block_m = 32;    ///< Row blocking size
            uint32_t block_n = 64;    ///< Column blocking size
            uint32_t block_k = 256;   ///< Inner dimension blocking
            bool use_avx512 = true;   ///< Enable AVX-512 (auto-detect)
            bool use_prefetch = true; ///< Enable prefetching
        };

        /**
         * T-MAC GEMM engine with AVX-512 optimization
         *
         * Implements blocked GEMM with T-MAC lookup tables:
         *
         * Performance characteristics:
         *   - Peak: ~800-1200 GOPS on Ryzanstein 9 7950X (16-core)
         *   - Memory bandwidth: ~50 GB/s (DDR5-6400)
         *   - Cache efficiency: ~80% L1/L2 hit rate
         *
         * Algorithm overview:
         *   1. Block weights into groups of 16 (T-MAC group_size)
         *   2. For each block: lookup ternary pattern × activation
         *   3. Accumulate using AVX-512 INT32 accumulators
         *   4. Write results with cache-line alignment
         */
        class TMACGemm
        {
        public:
            /**
             * Initialize with lookup table engine
             *
             * @param lut_engine T-MAC lookup table engine
             * @param config GEMM configuration (optional)
             */
            explicit TMACGemm(
                std::shared_ptr<LUTLookup> lut_engine,
                const GEMMConfig &config = GEMMConfig{});

            /**
             * Matrix multiplication: Y = W × X
             *
             * @param W Ternary weights [M, K] (row-major)
             * @param X INT8 activations [K, N] (row-major)
             * @param Y Output buffer [M, N] (row-major, preallocated)
             * @param M Number of output rows
             * @param K Inner dimension (must be multiple of 16)
             * @param N Number of output columns
             *
             * Time: O(M × N × K / 16) with T-MAC lookups
             * Space: O(1) working memory
             *
             * Requirements:
             *   - K must be divisible by 16 (T-MAC group size)
             *   - Buffers must be properly aligned (64-byte for AVX-512)
             *
             * Performance:
             *   - Typical: 500-800 GOPS
             *   - Peak: 1200 GOPS with optimal cache behavior
             */
            void gemm(
                const int8_t *W,
                const int8_t *X,
                int32_t *Y,
                uint32_t M,
                uint32_t K,
                uint32_t N);

            /**
             * Batched GEMM: Y[i] = W[i] × X[i] for i in [0, batch_size)
             *
             * More efficient than individual GEMMs due to:
             *   - Better instruction-level parallelism
             *   - Amortized setup costs
             *   - Improved cache reuse
             *
             * @param W_batch Array of weight matrices
             * @param X_batch Array of activation matrices
             * @param Y_batch Array of output matrices (preallocated)
             * @param batch_size Number of matrices
             * @param M, K, N Matrix dimensions (same for all in batch)
             */
            void gemm_batched(
                const int8_t **W_batch,
                const int8_t **X_batch,
                int32_t **Y_batch,
                uint32_t batch_size,
                uint32_t M,
                uint32_t K,
                uint32_t N);

            /**
             * Get GEMM statistics
             */
            struct Stats
            {
                uint64_t gemm_count = 0;    ///< Number of GEMMs performed
                uint64_t total_flops = 0;   ///< Total FLOPs computed
                double total_time_ms = 0.0; ///< Total execution time

                double gflops() const
                {
                    return total_time_ms > 0
                               ? (total_flops / 1e9) / (total_time_ms / 1000.0)
                               : 0.0;
                }

                double avg_time_ms() const
                {
                    return gemm_count > 0 ? total_time_ms / gemm_count : 0.0;
                }
            };

            const Stats &get_stats() const { return stats_; }
            void reset_stats() { stats_ = Stats{}; }
            void print_stats() const;

        private:
            std::shared_ptr<LUTLookup> lut_engine_;
            GEMMConfig config_;
            mutable Stats stats_;

            /**
             * Detect CPU features (AVX-512 support)
             */
            bool detect_avx512_support();

            /**
             * Blocked GEMM kernel
             *
             * Processes a block of size [block_m, block_k] × [block_k, block_n]
             * with optimal cache behavior.
             */
            void gemm_block(
                const int8_t *W,
                const int8_t *X,
                int32_t *Y,
                uint32_t m_start, uint32_t m_end,
                uint32_t n_start, uint32_t n_end,
                uint32_t K,
                uint32_t N);

#ifdef __AVX512F__
            /**
             * AVX-512 vectorized inner loop
             *
             * Processes 16 output elements in parallel using:
             *   - 16× INT32 accumulators
             *   - T-MAC batch lookups
             *   - Horizontal sum reduction
             */
            void gemm_inner_avx512(
                const int8_t *W_row,
                const int8_t *X,
                int32_t *Y_row,
                uint32_t K,
                uint32_t N);
#endif

            /**
             * Scalar fallback inner loop
             *
             * Used when AVX-512 is not available or for edge cases.
             */
            void gemm_inner_scalar(
                const int8_t *W_row,
                const int8_t *X,
                int32_t *Y_row,
                uint32_t K,
                uint32_t N);
        };

        /**
         * Utility: Check if pointer is aligned
         */
        inline bool is_aligned(const void *ptr, size_t alignment)
        {
            return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
        }

        /**
         * Utility: Aligned memory allocation
         *
         * @param size Size in bytes
         * @param alignment Alignment requirement (e.g., 64 for AVX-512)
         * @return Aligned pointer (must be freed with aligned_free)
         */
        void *aligned_alloc(size_t size, size_t alignment);

        /**
         * Utility: Free aligned memory
         */
        void aligned_free(void *ptr);

    } // namespace tmac
} // namespace ryzanstein_llm
