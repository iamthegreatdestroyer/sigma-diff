/*
 * RYZEN-LLM KV Cache Optimization
 * [REF:OL-005a] - Optimization Layer: KV Cache Management
 *
 * Efficient key-value cache for transformer attention with:
 * - Paged memory allocation (PagedAttention-style)
 * - Contiguous memory layout for cache-friendly access
 * - Block-level management for efficient allocation/deallocation
 * - Prefetching and memory alignment optimization
 * - Cross-sequence batch processing
 * - Optional quantization/compression for long contexts
 *
 * Performance Targets:
 * - 2× speedup via reduced memory bandwidth
 * - <10% memory overhead for management structures
 * - Sub-microsecond block allocation time
 * - Cache-line aligned access patterns
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <array>
#include <cstring>
#include <algorithm>
#include <string>
#include <optional>

namespace ryzen_llm
{
    namespace memory
    {

        // Configuration constants
        constexpr size_t CACHE_BLOCK_SIZE = 16;         // Tokens per block
        constexpr size_t CACHE_LINE_SIZE = 64;          // CPU cache line size
        constexpr size_t MAX_BLOCKS_PER_SEQUENCE = 128; // Max 2048 tokens
        constexpr size_t ALIGNMENT = 64;                // Memory alignment for SIMD

        /**
         * KV Cache Configuration
         */
        struct KVCacheConfig
        {
            uint32_t num_layers;      // Number of transformer layers
            uint32_t num_heads;       // Number of attention heads
            uint32_t head_dim;        // Dimension per head
            uint32_t max_batch_size;  // Maximum batch size
            uint32_t block_size;      // Tokens per block (default: 16)
            uint32_t num_blocks;      // Total blocks in pool
            bool enable_quantization; // Enable FP16/INT8 quantization
            bool enable_prefetching;  // Enable cache prefetching

            KVCacheConfig()
                : num_layers(32), num_heads(32), head_dim(128), max_batch_size(8), block_size(CACHE_BLOCK_SIZE), num_blocks(1024), enable_quantization(false), enable_prefetching(true)
            {
            }
        };

        /**
         * Physical block in memory pool
         * Stores K or V cache for block_size tokens
         */
        struct PhysicalBlock
        {
            float *data;        // [block_size, num_heads, head_dim]
            uint32_t ref_count; // Reference count for sharing
            bool is_allocated;  // Allocation status

            PhysicalBlock() : data(nullptr), ref_count(0), is_allocated(false) {}
        };

        /**
         * Logical block mapping for a sequence
         * Maps logical position to physical block
         */
        struct LogicalBlock
        {
            uint32_t physical_block_id; // Physical block index
            uint32_t num_tokens;        // Number of valid tokens in block

            LogicalBlock() : physical_block_id(UINT32_MAX), num_tokens(0) {}
        };

        /**
         * Block table for a single sequence
         * Maps logical blocks to physical blocks
         */
        struct BlockTable
        {
            std::vector<LogicalBlock> logical_blocks;
            uint32_t sequence_length; // Total tokens in sequence

            BlockTable() : sequence_length(0) {}

            void reset()
            {
                logical_blocks.clear();
                sequence_length = 0;
            }
        };

        // ============================================================================
        // FUTURE: ΣLANG×RSU Integration Support
        // These fields enable semantic-aware KV cache recycling. Currently unused
        // but will be populated when ΣLANG integration is activated post-MVP.
        // See: docs/SIGMALANG_INTEGRATION.md for architecture details.
        // ============================================================================

        /**
         * Semantic anchor metadata for ΣLANG integration
         *
         * Enables content-addressable KV cache lookup via semantic hashing.
         * Currently placeholder for post-MVP ΣLANG×RSU integration.
         */
        struct SigmaAnchorMetadata
        {
            // Semantic hash for content-addressable lookup (O(1) exact match)
            // Computed from ΣLANG glyph codebook indices
            uint64_t semantic_hash = 0;

            // Token positions marked as semantically important by ΣLANG
            // High-confidence glyphs map to these positions for cache alignment
            std::vector<int32_t> anchor_positions;

            // Glyph pattern signature for approximate matching
            // First N codebook indices form the pattern key
            std::array<uint8_t, 8> anchor_pattern = {0};

            // Reference to source RSU (Recyclable Semantic Unit) if applicable
            // Empty string means this entry was not created via ΣLANG
            std::string rsu_reference;

            // Timestamp for tier management (hot/warm/cold storage)
            int64_t creation_timestamp = 0;
            int64_t last_access_timestamp = 0;
            uint32_t access_count = 0;

            // Check if this entry has ΣLANG metadata
            bool has_sigma_metadata() const
            {
                return semantic_hash != 0 || !anchor_positions.empty();
            }
        };

        /**
         * Cache statistics for monitoring
         */
        struct CacheStats
        {
            uint64_t total_allocations;
            uint64_t total_deallocations;
            uint64_t cache_hits;
            uint64_t cache_misses;
            uint64_t blocks_allocated;
            uint64_t blocks_free;
            double memory_usage_mb;
            double avg_allocation_time_us;

            CacheStats()
                : total_allocations(0), total_deallocations(0), cache_hits(0), cache_misses(0), blocks_allocated(0), blocks_free(0), memory_usage_mb(0.0), avg_allocation_time_us(0.0)
            {
            }

            void reset()
            {
                *this = CacheStats();
            }

            std::string to_string() const;
        };

        /**
         * KV Cache Manager
         *
         * Implements paged memory management for transformer KV cache:
         * 1. Allocates physical blocks from memory pool
         * 2. Maps logical positions to physical blocks
         * 3. Supports block sharing across sequences (prefix caching)
         * 4. Optimizes memory layout for cache-line access
         * 5. Optional quantization for long contexts
         */
        class KVCacheManager
        {
        public:
            explicit KVCacheManager(const KVCacheConfig &config);
            ~KVCacheManager();

            // Sequence management

            /**
             * Allocate cache for a new sequence
             *
             * @param sequence_id Unique sequence identifier
             * @param estimated_length Estimated sequence length (for preallocation)
             * @return true if successful
             */
            bool AllocateSequence(uint64_t sequence_id, uint32_t estimated_length = 0);

            /**
             * Free cache for a sequence
             *
             * @param sequence_id Sequence identifier
             */
            void FreeSequence(uint64_t sequence_id);

            /**
             * Append tokens to sequence cache
             *
             * @param sequence_id Sequence identifier
             * @param num_tokens Number of tokens to append
             * @return true if successful
             */
            bool AppendTokens(uint64_t sequence_id, uint32_t num_tokens);

            // Cache access

            /**
             * Get key cache pointer for layer and position
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param position Token position
             * @return Pointer to key cache [num_heads, head_dim]
             */
            float *GetKeyCache(uint64_t sequence_id, uint32_t layer_id, uint32_t position);

            /**
             * Get value cache pointer for layer and position
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param position Token position
             * @return Pointer to value cache [num_heads, head_dim]
             */
            float *GetValueCache(uint64_t sequence_id, uint32_t layer_id, uint32_t position);

            /**
             * Get contiguous key cache for layer (all positions)
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param out_length Output sequence length
             * @return Pointer to key cache [seq_len, num_heads, head_dim]
             */
            const float *GetKeySequence(uint64_t sequence_id, uint32_t layer_id, uint32_t &out_length);

            /**
             * Get contiguous value cache for layer (all positions)
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param out_length Output sequence length
             * @return Pointer to value cache [seq_len, num_heads, head_dim]
             */
            const float *GetValueSequence(uint64_t sequence_id, uint32_t layer_id, uint32_t &out_length);

            /**
             * Write key cache for layer and position
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param position Token position
             * @param key_data Key data [num_heads, head_dim]
             */
            void WriteKey(uint64_t sequence_id, uint32_t layer_id, uint32_t position, const float *key_data);

            /**
             * Write value cache for layer and position
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param position Token position
             * @param value_data Value data [num_heads, head_dim]
             */
            void WriteValue(uint64_t sequence_id, uint32_t layer_id, uint32_t position, const float *value_data);

            // Optimization features

            /**
             * Prefetch cache blocks for upcoming positions
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param position Current position
             * @param lookahead Number of positions to prefetch
             */
            void Prefetch(uint64_t sequence_id, uint32_t layer_id, uint32_t position, uint32_t lookahead = 4);

            /**
             * Compress old cache entries (quantize to FP16/INT8)
             *
             * @param sequence_id Sequence identifier
             * @param retain_recent Number of recent tokens to keep uncompressed
             */
            void Compress(uint64_t sequence_id, [[maybe_unused]] uint32_t retain_recent);

            /**
             * Fork sequence (copy-on-write for prefix sharing)
             *
             * @param parent_sequence_id Parent sequence
             * @param child_sequence_id Child sequence
             * @param fork_position Position to fork at
             * @return true if successful
             */
            bool ForkSequence(uint64_t parent_sequence_id, uint64_t child_sequence_id, uint32_t fork_position);

            // Statistics and monitoring

            /**
             * Get current cache statistics
             */
            const CacheStats &GetStats() const { return stats_; }

            /**
             * Reset statistics
             */
            void ResetStats() { stats_.reset(); }

            /**
             * Get memory usage in bytes
             */
            size_t GetMemoryUsage() const;

            /**
             * Get configuration
             */
            const KVCacheConfig &GetConfig() const { return config_; }

            // ============================================================================
            // ΣLANG Integration Hooks (Phase 3 - Post-MVP)
            // These methods are placeholders for future ΣLANG×RSU integration.
            // Currently no-ops that don't affect MVP behavior.
            // ============================================================================

            /**
             * Register ΣLANG anchor metadata for semantic-aware cache recycling
             *
             * @param sequence_id Sequence identifier
             * @param semantic_hash Content hash from ΣLANG glyph codebook
             * @param anchor_positions Token positions marked as semantic anchors
             * @param rsu_reference Optional RSU reference ID
             *
             * FUTURE: Enables O(1) semantic cache lookup and intelligent recycling
             * Currently placeholder (no-op until ΣLANG integration)
             */
            void register_sigma_anchors(
                size_t sequence_id,
                uint64_t semantic_hash,
                const std::vector<int32_t> &anchor_positions,
                const std::string &rsu_reference = "");

            /**
             * Lookup KV cache by semantic hash (O(1) when ΣLANG active)
             *
             * @param semantic_hash Semantic hash from ΣLANG encoding
             * @return Optional sequence_id if exact semantic match found
             *
             * FUTURE: Content-addressable KV cache lookup
             * Currently returns empty (no semantic matching yet)
             */
            std::optional<size_t> lookup_by_semantic_hash(uint64_t semantic_hash);

            /**
             * Find recyclable cache with partial anchor overlap
             *
             * @param anchor_pattern 8-byte anchor pattern signature
             * @param min_overlap_ratio Minimum Jaccard similarity threshold
             * @return Optional sequence_id of best match above threshold
             *
             * FUTURE: Approximate KV cache matching via anchor patterns
             * Currently returns empty (no anchor matching yet)
             */
            std::optional<size_t> find_recyclable_by_anchors(
                const std::array<uint8_t, 8> &anchor_pattern,
                float min_overlap_ratio = 0.5f);

            /**
             * Update access statistics for tier management
             *
             * @param sequence_id Sequence identifier
             *
             * FUTURE: Track access patterns for hot/warm/cold tier decisions
             * Currently no-op
             */
            void update_sigma_access(size_t sequence_id);

        private:
            // Block management
            uint32_t allocate_physical_block();
            void free_physical_block(uint32_t block_id);
            void increment_ref_count(uint32_t block_id);
            void decrement_ref_count(uint32_t block_id);

            // Memory layout helpers
            size_t get_block_offset(uint32_t layer_id, uint32_t block_id, bool is_value) const;
            size_t get_element_stride() const;

            // Configuration
            KVCacheConfig config_;

            // Physical memory pool
            std::vector<PhysicalBlock> physical_blocks_;
            std::vector<uint32_t> free_blocks_;
            float *memory_pool_; // Contiguous memory allocation
            size_t pool_size_bytes_;

            // Sequence block tables
            std::unordered_map<uint64_t, BlockTable> key_block_tables_;
            std::unordered_map<uint64_t, BlockTable> value_block_tables_;

            // Contiguous buffers for sequence access (optimization)
            std::vector<float> key_sequence_buffer_;
            std::vector<float> value_sequence_buffer_;

            // Statistics
            mutable CacheStats stats_;

            // FUTURE: ΣLANG integration data structures
            // std::unordered_map<uint64_t, size_t> sigma_hash_index_;
            // std::unordered_map<size_t, SigmaAnchorMetadata> sigma_metadata_;
        };

        /**
         * Global KV cache statistics
         */
        extern CacheStats g_kv_cache_stats;

    } // namespace memory
} // namespace ryzen_llm
