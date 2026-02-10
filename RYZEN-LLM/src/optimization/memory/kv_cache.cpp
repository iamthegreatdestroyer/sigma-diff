/*
 * Ryzanstein LLM KV Cache Optimization
 * [REF:OL-005a] - Optimization Layer: KV Cache Management
 */

#include "kv_cache.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>

#ifdef _MSC_VER
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#elif defined(__MINGW32__) || defined(__MINGW64__)
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

#ifdef __x86_64__
#include <xmmintrin.h> // For prefetch
#endif

namespace ryzanstein_llm
{
    namespace memory
    {

        // Global statistics
        CacheStats g_kv_cache_stats;

        std::string CacheStats::to_string() const
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2);
            oss << "KV Cache Statistics:\n";
            oss << "  Total Allocations: " << total_allocations << "\n";
            oss << "  Total Deallocations: " << total_deallocations << "\n";
            oss << "  Cache Hits: " << cache_hits << "\n";
            oss << "  Cache Misses: " << cache_misses << "\n";
            oss << "  Blocks Allocated: " << blocks_allocated << "\n";
            oss << "  Blocks Free: " << blocks_free << "\n";
            oss << "  Memory Usage: " << memory_usage_mb << " MB\n";
            oss << "  Avg Allocation Time: " << avg_allocation_time_us << " μs\n";

            if (cache_hits + cache_misses > 0)
            {
                double hit_rate = 100.0 * cache_hits / (cache_hits + cache_misses);
                oss << "  Cache Hit Rate: " << hit_rate << "%\n";
            }

            return oss.str();
        }

        KVCacheManager::KVCacheManager(const KVCacheConfig &config)
            : config_(config), memory_pool_(nullptr), pool_size_bytes_(0)
        {
            // Calculate memory requirements
            // Each block stores: block_size tokens × num_heads × head_dim floats
            size_t elements_per_block = config_.block_size * config_.num_heads * config_.head_dim;
            size_t bytes_per_block = elements_per_block * sizeof(float);

            // Total for all blocks × 2 (key + value) × num_layers
            pool_size_bytes_ = bytes_per_block * config_.num_blocks * 2 * config_.num_layers;

            // Allocate aligned memory pool
            memory_pool_ = static_cast<float *>(aligned_alloc(ALIGNMENT, pool_size_bytes_));
            if (!memory_pool_)
            {
                throw std::bad_alloc();
            }

            // Initialize to zero
            std::memset(memory_pool_, 0, pool_size_bytes_);

            // Initialize physical blocks
            physical_blocks_.resize(config_.num_blocks);
            free_blocks_.reserve(config_.num_blocks);

            size_t block_offset = 0;
            for (uint32_t i = 0; i < config_.num_blocks; ++i)
            {
                physical_blocks_[i].data = memory_pool_ + block_offset;
                physical_blocks_[i].ref_count = 0;
                physical_blocks_[i].is_allocated = false;
                free_blocks_.push_back(i);

                block_offset += elements_per_block * 2 * config_.num_layers; // K+V for all layers
            }

            // Allocate sequence buffers (for contiguous access)
            size_t max_seq_elements = config_.block_size * MAX_BLOCKS_PER_SEQUENCE *
                                      config_.num_heads * config_.head_dim;
            key_sequence_buffer_.resize(max_seq_elements);
            value_sequence_buffer_.resize(max_seq_elements);

            // Update statistics
            stats_.blocks_free = config_.num_blocks;
            stats_.memory_usage_mb = pool_size_bytes_ / (1024.0 * 1024.0);
        }

        KVCacheManager::~KVCacheManager()
        {
            if (memory_pool_)
            {
                aligned_free(memory_pool_);
                memory_pool_ = nullptr;
            }
        }

        uint32_t KVCacheManager::allocate_physical_block()
        {
            auto start = std::chrono::high_resolution_clock::now();

            if (free_blocks_.empty())
            {
                stats_.cache_misses++;
                return UINT32_MAX; // Out of memory
            }

            uint32_t block_id = free_blocks_.back();
            free_blocks_.pop_back();

            physical_blocks_[block_id].is_allocated = true;
            physical_blocks_[block_id].ref_count = 1;

            stats_.total_allocations++;
            stats_.blocks_allocated++;
            stats_.blocks_free--;
            stats_.cache_hits++;

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            // Update rolling average allocation time
            stats_.avg_allocation_time_us =
                (stats_.avg_allocation_time_us * (stats_.total_allocations - 1) + duration.count()) /
                stats_.total_allocations;

            return block_id;
        }

        void KVCacheManager::free_physical_block(uint32_t block_id)
        {
            if (block_id >= physical_blocks_.size())
                return;

            physical_blocks_[block_id].is_allocated = false;
            physical_blocks_[block_id].ref_count = 0;
            free_blocks_.push_back(block_id);

            stats_.total_deallocations++;
            stats_.blocks_allocated--;
            stats_.blocks_free++;
        }

        void KVCacheManager::increment_ref_count(uint32_t block_id)
        {
            if (block_id < physical_blocks_.size())
            {
                physical_blocks_[block_id].ref_count++;
            }
        }

        void KVCacheManager::decrement_ref_count(uint32_t block_id)
        {
            if (block_id >= physical_blocks_.size())
                return;

            if (physical_blocks_[block_id].ref_count > 0)
            {
                physical_blocks_[block_id].ref_count--;

                if (physical_blocks_[block_id].ref_count == 0)
                {
                    free_physical_block(block_id);
                }
            }
        }

        size_t KVCacheManager::get_block_offset(uint32_t layer_id, uint32_t block_id, bool is_value) const
        {
            // Memory layout: [block][layer][K/V][block_size, num_heads, head_dim]
            size_t elements_per_block = config_.block_size * config_.num_heads * config_.head_dim;
            size_t elements_per_layer = elements_per_block * 2; // K + V
            size_t elements_per_physical_block = elements_per_layer * config_.num_layers;

            size_t offset = block_id * elements_per_physical_block;
            offset += layer_id * elements_per_layer;
            if (is_value)
            {
                offset += elements_per_block; // Skip K cache
            }

            return offset;
        }

        size_t KVCacheManager::get_element_stride() const
        {
            return config_.num_heads * config_.head_dim;
        }

        bool KVCacheManager::AllocateSequence(uint64_t sequence_id, uint32_t estimated_length)
        {
            // Check if already allocated
            if (key_block_tables_.count(sequence_id) > 0)
            {
                return false;
            }

            // Initialize block tables
            key_block_tables_[sequence_id] = BlockTable();
            value_block_tables_[sequence_id] = BlockTable();

            // Preallocate blocks if estimated length provided
            if (estimated_length > 0)
            {
                uint32_t num_blocks_needed = (estimated_length + config_.block_size - 1) / config_.block_size;
                num_blocks_needed = std::min(num_blocks_needed, (uint32_t)MAX_BLOCKS_PER_SEQUENCE);

                return AppendTokens(sequence_id, estimated_length);
            }

            return true;
        }

        void KVCacheManager::FreeSequence(uint64_t sequence_id)
        {
            // Free key blocks
            auto key_it = key_block_tables_.find(sequence_id);
            if (key_it != key_block_tables_.end())
            {
                for (const auto &logical_block : key_it->second.logical_blocks)
                {
                    if (logical_block.physical_block_id != UINT32_MAX)
                    {
                        decrement_ref_count(logical_block.physical_block_id);
                    }
                }
                key_block_tables_.erase(key_it);
            }

            // Free value blocks
            auto value_it = value_block_tables_.find(sequence_id);
            if (value_it != value_block_tables_.end())
            {
                for (const auto &logical_block : value_it->second.logical_blocks)
                {
                    if (logical_block.physical_block_id != UINT32_MAX)
                    {
                        decrement_ref_count(logical_block.physical_block_id);
                    }
                }
                value_block_tables_.erase(value_it);
            }
        }

        bool KVCacheManager::AppendTokens(uint64_t sequence_id, uint32_t num_tokens)
        {
            auto key_it = key_block_tables_.find(sequence_id);
            auto value_it = value_block_tables_.find(sequence_id);

            if (key_it == key_block_tables_.end() || value_it == value_block_tables_.end())
            {
                return false;
            }

            BlockTable &key_table = key_it->second;
            BlockTable &value_table = value_it->second;

            uint32_t old_length = key_table.sequence_length;
            uint32_t new_length = old_length + num_tokens;

            // Calculate blocks needed
            uint32_t old_blocks = (old_length + config_.block_size - 1) / config_.block_size;
            uint32_t new_blocks = (new_length + config_.block_size - 1) / config_.block_size;

            if (new_blocks > MAX_BLOCKS_PER_SEQUENCE)
            {
                return false; // Sequence too long
            }

            // Allocate additional blocks if needed
            for (uint32_t i = old_blocks; i < new_blocks; ++i)
            {
                // Allocate key block
                uint32_t key_block_id = allocate_physical_block();
                if (key_block_id == UINT32_MAX)
                    return false;

                LogicalBlock key_logical;
                key_logical.physical_block_id = key_block_id;
                key_logical.num_tokens = std::min(config_.block_size, new_length - i * config_.block_size);
                key_table.logical_blocks.push_back(key_logical);

                // Allocate value block
                uint32_t value_block_id = allocate_physical_block();
                if (value_block_id == UINT32_MAX)
                {
                    decrement_ref_count(key_block_id);
                    return false;
                }

                LogicalBlock value_logical;
                value_logical.physical_block_id = value_block_id;
                value_logical.num_tokens = key_logical.num_tokens;
                value_table.logical_blocks.push_back(value_logical);
            }

            // Update last block's token count
            if (!key_table.logical_blocks.empty())
            {
                uint32_t last_block_tokens = new_length % config_.block_size;
                if (last_block_tokens == 0)
                    last_block_tokens = config_.block_size;

                key_table.logical_blocks.back().num_tokens = last_block_tokens;
                value_table.logical_blocks.back().num_tokens = last_block_tokens;
            }

            key_table.sequence_length = new_length;
            value_table.sequence_length = new_length;

            return true;
        }

        float *KVCacheManager::GetKeyCache(uint64_t sequence_id, uint32_t layer_id, uint32_t position)
        {
            auto it = key_block_tables_.find(sequence_id);
            if (it == key_block_tables_.end())
                return nullptr;

            const BlockTable &table = it->second;
            if (position >= table.sequence_length)
                return nullptr;

            // Find logical block
            uint32_t block_idx = position / config_.block_size;
            uint32_t block_offset = position % config_.block_size;

            if (block_idx >= table.logical_blocks.size())
                return nullptr;

            uint32_t physical_block_id = table.logical_blocks[block_idx].physical_block_id;
            if (physical_block_id == UINT32_MAX)
                return nullptr;

            // Calculate memory address
            size_t base_offset = get_block_offset(layer_id, physical_block_id, false);
            size_t token_stride = get_element_stride();
            size_t offset = base_offset + block_offset * token_stride;

            return memory_pool_ + offset;
        }

        float *KVCacheManager::GetValueCache(uint64_t sequence_id, uint32_t layer_id, uint32_t position)
        {
            auto it = value_block_tables_.find(sequence_id);
            if (it == value_block_tables_.end())
                return nullptr;

            const BlockTable &table = it->second;
            if (position >= table.sequence_length)
                return nullptr;

            // Find logical block
            uint32_t block_idx = position / config_.block_size;
            uint32_t block_offset = position % config_.block_size;

            if (block_idx >= table.logical_blocks.size())
                return nullptr;

            uint32_t physical_block_id = table.logical_blocks[block_idx].physical_block_id;
            if (physical_block_id == UINT32_MAX)
                return nullptr;

            // Calculate memory address
            size_t base_offset = get_block_offset(layer_id, physical_block_id, true);
            size_t token_stride = get_element_stride();
            size_t offset = base_offset + block_offset * token_stride;

            return memory_pool_ + offset;
        }

        const float *KVCacheManager::GetKeySequence(uint64_t sequence_id, uint32_t layer_id, uint32_t &out_length)
        {
            auto it = key_block_tables_.find(sequence_id);
            if (it == key_block_tables_.end())
            {
                out_length = 0;
                return nullptr;
            }

            const BlockTable &table = it->second;
            out_length = table.sequence_length;

            // Copy all blocks into contiguous buffer
            size_t token_stride = get_element_stride();
            size_t dest_offset = 0;

            for (size_t i = 0; i < table.logical_blocks.size(); ++i)
            {
                const LogicalBlock &logical = table.logical_blocks[i];
                uint32_t physical_block_id = logical.physical_block_id;

                if (physical_block_id == UINT32_MAX)
                    continue;

                size_t base_offset = get_block_offset(layer_id, physical_block_id, false);
                size_t copy_elements = logical.num_tokens * token_stride;

                std::memcpy(
                    key_sequence_buffer_.data() + dest_offset,
                    memory_pool_ + base_offset,
                    copy_elements * sizeof(float));

                dest_offset += copy_elements;
            }

            return key_sequence_buffer_.data();
        }

        const float *KVCacheManager::GetValueSequence(uint64_t sequence_id, uint32_t layer_id, uint32_t &out_length)
        {
            auto it = value_block_tables_.find(sequence_id);
            if (it == value_block_tables_.end())
            {
                out_length = 0;
                return nullptr;
            }

            const BlockTable &table = it->second;
            out_length = table.sequence_length;

            // Copy all blocks into contiguous buffer
            size_t token_stride = get_element_stride();
            size_t dest_offset = 0;

            for (size_t i = 0; i < table.logical_blocks.size(); ++i)
            {
                const LogicalBlock &logical = table.logical_blocks[i];
                uint32_t physical_block_id = logical.physical_block_id;

                if (physical_block_id == UINT32_MAX)
                    continue;

                size_t base_offset = get_block_offset(layer_id, physical_block_id, true);
                size_t copy_elements = logical.num_tokens * token_stride;

                std::memcpy(
                    value_sequence_buffer_.data() + dest_offset,
                    memory_pool_ + base_offset,
                    copy_elements * sizeof(float));

                dest_offset += copy_elements;
            }

            return value_sequence_buffer_.data();
        }

        void KVCacheManager::WriteKey(uint64_t sequence_id, uint32_t layer_id, uint32_t position, const float *key_data)
        {
            float *cache_ptr = GetKeyCache(sequence_id, layer_id, position);
            if (cache_ptr)
            {
                size_t elements = config_.num_heads * config_.head_dim;
                std::memcpy(cache_ptr, key_data, elements * sizeof(float));
            }
        }

        void KVCacheManager::WriteValue(uint64_t sequence_id, uint32_t layer_id, uint32_t position, const float *value_data)
        {
            float *cache_ptr = GetValueCache(sequence_id, layer_id, position);
            if (cache_ptr)
            {
                size_t elements = config_.num_heads * config_.head_dim;
                std::memcpy(cache_ptr, value_data, elements * sizeof(float));
            }
        }

        void KVCacheManager::Prefetch(uint64_t sequence_id, uint32_t layer_id, uint32_t position, uint32_t lookahead)
        {
            if (!config_.enable_prefetching)
                return;

#ifdef __x86_64__
            for (uint32_t i = 0; i < lookahead; ++i)
            {
                uint32_t prefetch_pos = position + i + 1;

                float *key_ptr = GetKeyCache(sequence_id, layer_id, prefetch_pos);
                if (key_ptr)
                {
                    _mm_prefetch(reinterpret_cast<const char *>(key_ptr), _MM_HINT_T0);
                }

                float *value_ptr = GetValueCache(sequence_id, layer_id, prefetch_pos);
                if (value_ptr)
                {
                    _mm_prefetch(reinterpret_cast<const char *>(value_ptr), _MM_HINT_T0);
                }
            }
#endif
        }

        void KVCacheManager::Compress(uint64_t sequence_id, uint32_t retain_recent)
        {
            // Currently unimplemented: keep parameters to avoid API changes
            (void)sequence_id;   // parameter intentionally unused for now
            (void)retain_recent; // parameter intentionally unused for now

            // TODO: Implement FP16/INT8 quantization for old cache entries
            // This is a future optimization for long-context scenarios
            // For now, we keep everything in FP32
        }

        bool KVCacheManager::ForkSequence(uint64_t parent_sequence_id, uint64_t child_sequence_id, uint32_t fork_position)
        {
            auto parent_key_it = key_block_tables_.find(parent_sequence_id);
            auto parent_value_it = value_block_tables_.find(parent_sequence_id);

            if (parent_key_it == key_block_tables_.end() || parent_value_it == value_block_tables_.end())
            {
                return false;
            }

            // Check if child already exists
            if (key_block_tables_.count(child_sequence_id) > 0)
            {
                return false;
            }

            const BlockTable &parent_key = parent_key_it->second;
            const BlockTable &parent_value = parent_value_it->second;

            if (fork_position > parent_key.sequence_length)
            {
                return false;
            }

            // Create child block tables
            BlockTable &child_key = key_block_tables_[child_sequence_id];
            BlockTable &child_value = value_block_tables_[child_sequence_id];

            // Copy block references up to fork position (copy-on-write)
            uint32_t num_blocks = (fork_position + config_.block_size - 1) / config_.block_size;

            for (uint32_t i = 0; i < num_blocks; ++i)
            {
                if (i < parent_key.logical_blocks.size())
                {
                    child_key.logical_blocks.push_back(parent_key.logical_blocks[i]);
                    increment_ref_count(parent_key.logical_blocks[i].physical_block_id);
                }

                if (i < parent_value.logical_blocks.size())
                {
                    child_value.logical_blocks.push_back(parent_value.logical_blocks[i]);
                    increment_ref_count(parent_value.logical_blocks[i].physical_block_id);
                }
            }

            child_key.sequence_length = fork_position;
            child_value.sequence_length = fork_position;

            return true;
        }

        size_t KVCacheManager::GetMemoryUsage() const
        {
            return pool_size_bytes_;
        }

        // ============================================================================
        // ΣLANG Integration Hooks (Placeholder implementations)
        // These will be replaced with actual ΣLANG logic post-MVP.
        // For now, they provide no-op defaults that don't affect current behavior.
        // ============================================================================

        void KVCacheManager::register_sigma_anchors(
            size_t sequence_id,
            uint64_t semantic_hash,
            const std::vector<int32_t> &anchor_positions,
            const std::string &rsu_reference)
        {
            // FUTURE: Store anchor metadata for this sequence's KV states
            // This enables semantic-aware cache lookup and recycling

            // Placeholder: No-op until ΣLANG integration
            (void)sequence_id;
            (void)semantic_hash;
            (void)anchor_positions;
            (void)rsu_reference;

            // TODO(SIGMA): Implement anchor registration
            // - Store semantic_hash in lookup index
            // - Record anchor_positions for partial match scoring
            // - Link to RSU for tier management
        }

        std::optional<size_t> KVCacheManager::lookup_by_semantic_hash(uint64_t semantic_hash)
        {
            // FUTURE: Content-addressable KV cache lookup
            // Returns sequence_id if exact semantic match found

            // Placeholder: Always return empty (no semantic matching yet)
            (void)semantic_hash;
            return std::nullopt;

            // TODO(SIGMA): Implement hash-based lookup
            // - Check hash index for exact match
            // - If found, update access statistics
            // - Return sequence_id for KV reuse
        }

        std::optional<size_t> KVCacheManager::find_recyclable_by_anchors(
            const std::array<uint8_t, 8> &anchor_pattern,
            float min_overlap_ratio)
        {
            // FUTURE: Approximate KV cache matching via anchor patterns
            // Enables partial reuse when content is similar but not identical

            // Placeholder: Always return empty (no anchor matching yet)
            (void)anchor_pattern;
            (void)min_overlap_ratio;
            return std::nullopt;

            // TODO(SIGMA): Implement anchor-based matching
            // - Scan anchor patterns for sufficient overlap
            // - Score candidates by Jaccard similarity
            // - Return best match above threshold
        }

        void KVCacheManager::update_sigma_access(size_t sequence_id)
        {
            // FUTURE: Track access patterns for hot/warm/cold tier decisions

            // Placeholder: No-op
            (void)sequence_id;

            // TODO(SIGMA): Implement access tracking
            // - Increment access_count
            // - Update last_access_timestamp
            // - Check promotion/demotion thresholds
        }

    } // namespace memory
} // namespace ryzanstein_llm
