/*
 * RYZEN-LLM KV Cache Optimized - Ultra-High Performance
 * [REF:OL-005b] - Performance-Critical KV Cache for BitNet Attention
 *
 * Optimizations for 30× speedup (0.42 → 12+ tokens/sec):
 * - Ring buffer on sequence dimension (O(1) append, no copies)
 * - Pre-allocated memory pools (zero mallocs per token)
 * - 64-byte cache-line alignment for L1/L2 efficiency
 * - Batch dimension support with minimal per-batch overhead
 * - Lock-free append for single-producer scenarios
 * - Vectorized copy operations (SIMD where possible)
 *
 * Design Principles:
 * 1. Fixed buffer allocation at sequence start
 * 2. Ring buffer pointer arithmetic only (no reallocation)
 * 3. Contiguous memory for each batch (cache-friendly)
 * 4. Head-local storage for parallel head processing
 * 5. Eviction policy for exceeding max_seq_len
 *
 * Memory Layout:
 * [Batch 0: [Head 0: [K|V], Head 1: [K|V], ...],
 *  Batch 1: [Head 0: [K|V], Head 1: [K|V], ...],
 *  ...]
 * All memory is pre-allocated and aligned to 64 bytes.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <optional>

namespace ryzen_llm
{
    namespace optimization
    {

        // ============================================================================
        // Configuration and Constants
        // ============================================================================

        constexpr size_t CACHE_LINE_SIZE = 64;         // L1 cache line
        constexpr size_t ALIGNMENT = CACHE_LINE_SIZE;  // Memory alignment
        constexpr uint32_t DEFAULT_MAX_SEQ_LEN = 2048; // Max sequence length
        constexpr uint32_t DEFAULT_MAX_BATCH = 8;      // Max batch size
        constexpr uint32_t DEFAULT_NUM_HEADS = 32;     // Number of heads
        constexpr uint32_t DEFAULT_HEAD_DIM = 128;     // Dimension per head
        constexpr bool ENABLE_PREFETCH = true;         // CPU prefetch hints

        // ============================================================================
        // Data Structures
        // ============================================================================

        /**
         * Per-batch cache state for ring buffer management
         * Tracks current write position and total cached tokens
         */
        struct CacheState
        {
            uint32_t seq_len;      // Current sequence length
            uint32_t ring_pos;     // Current write position in ring (0 to max_seq_len-1)
            uint32_t full_count;   // Times ring buffer has wrapped around
            uint32_t total_tokens; // Total tokens ever appended to this batch

            CacheState() : seq_len(0), ring_pos(0), full_count(0), total_tokens(0) {}

            void reset()
            {
                seq_len = 0;
                ring_pos = 0;
                full_count = 0;
                total_tokens = 0;
            }
        };

        /**
         * K/V cache storage with ring buffer semantics
         * Organized as: [num_heads][max_seq_len][head_dim]
         * This layout ensures sequential memory access within a head
         */
        struct BatchKVStorage
        {
            float *K_data;     // [num_heads][max_seq_len][head_dim]
            float *V_data;     // [num_heads][max_seq_len][head_dim]
            size_t total_size; // Total bytes allocated for this batch
            CacheState state;  // Ring buffer state
        };

        /**
         * Performance metrics for profiling
         */
        struct CacheMetrics
        {
            uint64_t append_calls = 0;
            uint64_t total_append_ns = 0;
            uint64_t reset_calls = 0;
            uint64_t total_reset_ns = 0;
            uint64_t get_cache_calls = 0;
            uint64_t total_get_cache_ns = 0;

            // Statistics
            double avg_append_ns() const
            {
                return append_calls > 0 ? static_cast<double>(total_append_ns) / append_calls : 0.0;
            }

            double avg_get_cache_ns() const
            {
                return get_cache_calls > 0 ? static_cast<double>(total_get_cache_ns) / get_cache_calls : 0.0;
            }
        };

        // ============================================================================
        // KVCacheManager - Main Interface
        // ============================================================================

        /**
         * High-performance KV cache manager with ring buffer optimization.
         *
         * Usage pattern:
         *   1. allocate(max_seq_len, batch_size, hidden_dim, num_heads)
         *   2. For each token:
         *      - Compute K,V for current token
         *      - append(K, V, seq_pos, batch_idx)
         *   3. For attention computation:
         *      - get_cache(batch_idx, K_cache, V_cache, cached_len)
         *      - Use K_cache, V_cache for scaled dot-product attention
         *   4. For next sequence:
         *      - reset(batch_idx)
         */
        class KVCacheManager
        {
        public:
            /**
             * Allocate memory for KV cache.
             *
             * @param max_seq_len Maximum sequence length (used for ring buffer size)
             * @param batch_size Number of concurrent sequences
             * @param hidden_dim Total hidden dimension (for reshaping to heads)
             * @param num_heads Number of attention heads
             *
             * Memory allocated per batch: 2 × num_heads × max_seq_len × head_dim × sizeof(float)
             * All memory is 64-byte aligned for cache efficiency.
             */
            void allocate(uint32_t max_seq_len, uint32_t batch_size, uint32_t hidden_dim,
                          uint32_t num_heads);

            /**
             * Append new K,V values from current token computation.
             *
             * Performs ring buffer append:
             * - No allocation (pre-allocated)
             * - O(1) time (just pointer updates + memcpy)
             * - Minimal latency (<100ns per token per head group)
             *
             * @param K Pointer to K values [num_heads, head_dim] for current token
             * @param V Pointer to V values [num_heads, head_dim] for current token
             * @param seq_pos Current sequence position (used for validation)
             * @param batch_idx Batch index (0 to batch_size-1)
             *
             * Note: K and V should be in layout [num_heads * head_dim].
             * The function handles the reshape internally.
             */
            void append(const float *K, const float *V, uint32_t seq_pos, uint32_t batch_idx);

            /**
             * Get pointers to cached K,V for attention computation.
             *
             * Returns pointers to the entire cached KV for the batch:
             * K_cache: [num_heads, cached_len, head_dim]
             * V_cache: [num_heads, cached_len, head_dim]
             *
             * Note: Memory layout is optimized for attention computation.
             * If ring buffer has wrapped, handles the linear reconstruction.
             *
             * @param batch_idx Batch index
             * @param K_cache Output pointer to cached K values
             * @param V_cache Output pointer to cached V values
             * @param cached_len Output: number of tokens cached
             */
            void get_cache(uint32_t batch_idx, float *&K_cache, float *&V_cache,
                           uint32_t &cached_len);

            /**
             * Clear cache for a batch (start new sequence).
             *
             * Resets ring buffer state without deallocating memory.
             * @param batch_idx Batch index
             */
            void reset(uint32_t batch_idx);

            /**
             * Clear all caches.
             * Resets all batches without deallocating memory.
             */
            void reset_all();

            /**
             * Get cache state for a batch (for debugging/diagnostics).
             * @param batch_idx Batch index
             * @return Current CacheState
             */
            CacheState get_state(uint32_t batch_idx) const;

            /**
             * Get performance metrics.
             * @return CacheMetrics with timing information
             */
            CacheMetrics get_metrics() const { return metrics_; }

            /**
             * Get memory usage statistics.
             * @return Total bytes allocated for KV cache
             */
            size_t get_memory_usage() const { return total_memory_bytes_; }

            /**
             * Destructor - frees all allocated memory.
             */
            ~KVCacheManager();

            // Disable copying
            KVCacheManager(const KVCacheManager &) = delete;
            KVCacheManager &operator=(const KVCacheManager &) = delete;

            // Allow moving
            KVCacheManager(KVCacheManager &&other) noexcept;
            KVCacheManager &operator=(KVCacheManager &&other) noexcept;

        private:
            // Configuration
            uint32_t max_seq_len_ = 0;
            uint32_t batch_size_ = 0;
            uint32_t hidden_dim_ = 0;
            uint32_t num_heads_ = 0;
            uint32_t head_dim_ = 0;

            // Memory management
            std::vector<BatchKVStorage> batch_storages_;
            uint8_t *memory_pool_ = nullptr;
            size_t total_memory_bytes_ = 0;

            // Temporary buffer for reconstructed cache (if ring buffer wraps)
            std::vector<float> temp_cache_buffer_;

            // Performance metrics
            mutable CacheMetrics metrics_;
            bool enable_metrics_ = true;

            // Helper functions
            size_t compute_batch_size() const;
            void allocate_batch(uint32_t batch_idx);
            void free_batch(uint32_t batch_idx);
            void reconstruct_linear_cache(uint32_t batch_idx, float *output);
        };

        // ============================================================================
        // Memory Utilities
        // ============================================================================

        /**
         * Aligned memory allocator.
         * Ensures memory is aligned to CACHE_LINE_SIZE for optimal performance.
         */
        inline void *aligned_malloc(size_t size, size_t alignment = ALIGNMENT)
        {
#ifdef _MSC_VER
            return _aligned_malloc(size, alignment);
#else
            return aligned_alloc(alignment, size);
#endif
        }

        /**
         * Aligned memory deallocator.
         */
        inline void aligned_free(void *ptr)
        {
#ifdef _MSC_VER
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }

        /**
         * CPU prefetch hint for memory access.
         * Brings memory into L1 cache.
         */
        inline void prefetch_cache_line(const void *ptr)
        {
#ifdef ENABLE_PREFETCH
#if defined(__x86_64__) || defined(_M_X64)
            // movl $0, (%rdi); clflush: writes-back cache line
            __builtin_prefetch(ptr, 0, 3); // rw=0 (read), locality=3 (L1)
#endif
#endif
        }

    } // namespace optimization
} // namespace ryzen_llm
