/*
 * RYZEN-LLM KV Cache Optimized Implementation
 * [REF:OL-005b] - Performance-Critical KV Cache for BitNet Attention
 *
 * Critical optimizations:
 * - memcpy is highly optimized by modern libc (uses SIMD on modern CPUs)
 * - Minimal pointer arithmetic (cache-friendly)
 * - Prefetch hints for sequential access
 * - Ring buffer with O(1) append
 * - Zero allocations per token (amortized O(1))
 */

#include "kv_cache_optimized.h"
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <sstream>
#include <iomanip>

#ifdef _MSC_VER
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#elif !defined(__APPLE__)
#include <cstdlib>
#endif

namespace ryzen_llm
{
    namespace optimization
    {

        // ============================================================================
        // KVCacheManager Implementation
        // ============================================================================

        void KVCacheManager::allocate(uint32_t max_seq_len, uint32_t batch_size,
                                      uint32_t hidden_dim, uint32_t num_heads)
        {
            if (max_seq_len == 0 || batch_size == 0 || hidden_dim == 0 || num_heads == 0)
            {
                throw std::invalid_argument("All parameters must be > 0");
            }

            if (hidden_dim % num_heads != 0)
            {
                throw std::invalid_argument("hidden_dim must be divisible by num_heads");
            }

            // Store configuration
            max_seq_len_ = max_seq_len;
            batch_size_ = batch_size;
            hidden_dim_ = hidden_dim;
            num_heads_ = num_heads;
            head_dim_ = hidden_dim / num_heads;

            // Allocate storage descriptors for each batch
            batch_storages_.clear();
            batch_storages_.resize(batch_size);

            // Calculate per-batch memory requirement
            // Each batch: [num_heads][max_seq_len][head_dim] for both K and V
            // = 2 * num_heads * max_seq_len * head_dim * sizeof(float)
            size_t per_head_size = max_seq_len * head_dim_ * sizeof(float);

            // Add padding for cache-line alignment per head
            size_t aligned_per_head_size =
                ((per_head_size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
            size_t aligned_per_batch_size = 2 * num_heads * aligned_per_head_size;

            // Total memory for all batches
            total_memory_bytes_ = aligned_per_batch_size * batch_size;

            // Allocate aligned memory pool
            memory_pool_ = static_cast<uint8_t *>(aligned_malloc(total_memory_bytes_, ALIGNMENT));
            if (!memory_pool_)
            {
                throw std::bad_alloc();
            }

            // Initialize each batch storage
            uint8_t *offset = memory_pool_;
            for (uint32_t b = 0; b < batch_size; ++b)
            {
                batch_storages_[b].K_data = reinterpret_cast<float *>(offset);
                offset += num_heads * aligned_per_head_size;

                batch_storages_[b].V_data = reinterpret_cast<float *>(offset);
                offset += num_heads * aligned_per_head_size;

                batch_storages_[b].total_size = aligned_per_batch_size;
                batch_storages_[b].state.reset();
            }

            // Allocate temporary buffer for ring buffer reconstruction
            // Used only when ring buffer wraps (rare path)
            temp_cache_buffer_.resize(2 * num_heads * max_seq_len * head_dim_);

            // Initialize memory to zero
            std::memset(memory_pool_, 0, total_memory_bytes_);
        }

        void KVCacheManager::append(const float *K, const float *V, uint32_t seq_pos,
                                    uint32_t batch_idx)
        {
            if (batch_idx >= batch_size_)
            {
                throw std::out_of_range("batch_idx out of range");
            }

            auto start_time = enable_metrics_ ? std::chrono::high_resolution_clock::now()
                                              : std::chrono::high_resolution_clock::time_point{};

            BatchKVStorage &storage = batch_storages_[batch_idx];
            CacheState &state = storage.state;

            // Validate seq_pos matches expected
            if (seq_pos != state.seq_len)
            {
                throw std::invalid_argument("seq_pos mismatch: expected " + std::to_string(state.seq_len) +
                                            " got " + std::to_string(seq_pos));
            }

            // Get ring buffer position
            uint32_t ring_pos = state.ring_pos;

            // Copy K values for each head
            // Layout: input K is [num_heads * head_dim]
            // Output storage: [num_heads][max_seq_len][head_dim]
            for (uint32_t h = 0; h < num_heads_; ++h)
            {
                float *K_head = storage.K_data + h * max_seq_len_ * head_dim_;
                float *K_ring_pos = K_head + ring_pos * head_dim_;
                const float *K_src = K + h * head_dim_;

                // Memcpy is highly optimized (uses SIMD on modern CPUs)
                // This is typically faster than element-wise copy
                std::memcpy(K_ring_pos, K_src, head_dim_ * sizeof(float));

                // Prefetch next cache line if available
                if (h + 1 < num_heads_)
                {
                    prefetch_cache_line(storage.K_data + (h + 1) * max_seq_len_ * head_dim_);
                }
            }

            // Copy V values for each head (same pattern as K)
            for (uint32_t h = 0; h < num_heads_; ++h)
            {
                float *V_head = storage.V_data + h * max_seq_len_ * head_dim_;
                float *V_ring_pos = V_head + ring_pos * head_dim_;
                const float *V_src = V + h * head_dim_;

                std::memcpy(V_ring_pos, V_src, head_dim_ * sizeof(float));

                if (h + 1 < num_heads_)
                {
                    prefetch_cache_line(storage.V_data + (h + 1) * max_seq_len_ * head_dim_);
                }
            }

            // Update state
            state.seq_len++;
            state.total_tokens++;
            state.ring_pos = (ring_pos + 1) % max_seq_len_;

            // Track wrap-around for reconstruction logic
            if (state.ring_pos == 0 && state.seq_len > 0)
            {
                state.full_count++;
            }

            // Record metrics
            if (enable_metrics_)
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                metrics_.append_calls++;
                metrics_.total_append_ns +=
                    std::chrono::nanoseconds(end_time - start_time).count();
            }
        }

        void KVCacheManager::get_cache(uint32_t batch_idx, float *&K_cache, float *&V_cache,
                                       uint32_t &cached_len)
        {
            if (batch_idx >= batch_size_)
            {
                throw std::out_of_range("batch_idx out of range");
            }

            auto start_time = enable_metrics_ ? std::chrono::high_resolution_clock::now()
                                              : std::chrono::high_resolution_clock::time_point{};

            BatchKVStorage &storage = batch_storages_[batch_idx];
            const CacheState &state = storage.state;

            cached_len = std::min(state.seq_len, max_seq_len_);

            // Fast path: Ring buffer hasn't wrapped yet
            // In this case, K and V are already in sequential order
            if (state.seq_len <= max_seq_len_ && state.ring_pos <= state.seq_len)
            {
                K_cache = storage.K_data;
                V_cache = storage.V_data;
            }
            else
            {
                // Slow path: Ring buffer has wrapped
                // Need to reconstruct linear cache by copying from wrap point
                reconstruct_linear_cache(batch_idx, temp_cache_buffer_.data());
                K_cache = temp_cache_buffer_.data();
                V_cache = K_cache + num_heads_ * max_seq_len_ * head_dim_;
            }

            // Record metrics
            if (enable_metrics_)
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                metrics_.get_cache_calls++;
                metrics_.total_get_cache_ns +=
                    std::chrono::nanoseconds(end_time - start_time).count();
            }
        }

        void KVCacheManager::reset(uint32_t batch_idx)
        {
            if (batch_idx >= batch_size_)
            {
                throw std::out_of_range("batch_idx out of range");
            }

            auto start_time = enable_metrics_ ? std::chrono::high_resolution_clock::now()
                                              : std::chrono::high_resolution_clock::time_point{};

            batch_storages_[batch_idx].state.reset();

            if (enable_metrics_)
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                metrics_.reset_calls++;
                metrics_.total_reset_ns +=
                    std::chrono::nanoseconds(end_time - start_time).count();
            }
        }

        void KVCacheManager::reset_all()
        {
            for (uint32_t b = 0; b < batch_size_; ++b)
            {
                reset(b);
            }
        }

        CacheState KVCacheManager::get_state(uint32_t batch_idx) const
        {
            if (batch_idx >= batch_size_)
            {
                throw std::out_of_range("batch_idx out of range");
            }
            return batch_storages_[batch_idx].state;
        }

        KVCacheManager::~KVCacheManager()
        {
            if (memory_pool_)
            {
                aligned_free(memory_pool_);
                memory_pool_ = nullptr;
            }
            batch_storages_.clear();
            temp_cache_buffer_.clear();
        }

        KVCacheManager::KVCacheManager(KVCacheManager &&other) noexcept
            : max_seq_len_(other.max_seq_len_), batch_size_(other.batch_size_),
              hidden_dim_(other.hidden_dim_), num_heads_(other.num_heads_),
              head_dim_(other.head_dim_), batch_storages_(std::move(other.batch_storages_)),
              memory_pool_(other.memory_pool_), total_memory_bytes_(other.total_memory_bytes_),
              temp_cache_buffer_(std::move(other.temp_cache_buffer_)), metrics_(other.metrics_)
        {
            other.memory_pool_ = nullptr;
            other.total_memory_bytes_ = 0;
        }

        KVCacheManager &KVCacheManager::operator=(KVCacheManager &&other) noexcept
        {
            if (this != &other)
            {
                if (memory_pool_)
                {
                    aligned_free(memory_pool_);
                }
                max_seq_len_ = other.max_seq_len_;
                batch_size_ = other.batch_size_;
                hidden_dim_ = other.hidden_dim_;
                num_heads_ = other.num_heads_;
                head_dim_ = other.head_dim_;
                batch_storages_ = std::move(other.batch_storages_);
                memory_pool_ = other.memory_pool_;
                total_memory_bytes_ = other.total_memory_bytes_;
                temp_cache_buffer_ = std::move(other.temp_cache_buffer_);
                metrics_ = other.metrics_;

                other.memory_pool_ = nullptr;
                other.total_memory_bytes_ = 0;
            }
            return *this;
        }

        // ============================================================================
        // Private Helper Methods
        // ============================================================================

        size_t KVCacheManager::compute_batch_size() const
        {
            // Per-batch memory: 2 * num_heads * max_seq_len * head_dim * sizeof(float)
            // Plus alignment padding
            size_t per_head_size = max_seq_len_ * head_dim_ * sizeof(float);
            size_t aligned_per_head_size =
                ((per_head_size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
            return 2 * num_heads_ * aligned_per_head_size;
        }

        void KVCacheManager::reconstruct_linear_cache(uint32_t batch_idx, float *output)
        {
            BatchKVStorage &storage = batch_storages_[batch_idx];
            const CacheState &state = storage.state;

            // Reconstruct linear cache when ring buffer has wrapped
            // Copy from ring_pos to end, then from start to ring_pos
            uint32_t wrapped_portion = max_seq_len_ - state.ring_pos;

            float *K_out = output;
            float *V_out = K_out + num_heads_ * max_seq_len_ * head_dim_;

            for (uint32_t h = 0; h < num_heads_; ++h)
            {
                float *K_head = storage.K_data + h * max_seq_len_ * head_dim_;
                float *V_head = storage.V_data + h * max_seq_len_ * head_dim_;

                // Copy wrapped portion
                std::memcpy(K_out + h * max_seq_len_ * head_dim_, K_head + state.ring_pos * head_dim_,
                            wrapped_portion * head_dim_ * sizeof(float));

                // Copy wraparound portion
                std::memcpy(K_out + h * max_seq_len_ * head_dim_ + wrapped_portion * head_dim_, K_head,
                            state.ring_pos * head_dim_ * sizeof(float));

                // Same for V
                std::memcpy(V_out + h * max_seq_len_ * head_dim_, V_head + state.ring_pos * head_dim_,
                            wrapped_portion * head_dim_ * sizeof(float));

                std::memcpy(V_out + h * max_seq_len_ * head_dim_ + wrapped_portion * head_dim_, V_head,
                            state.ring_pos * head_dim_ * sizeof(float));
            }
        }

    } // namespace optimization
} // namespace ryzen_llm
