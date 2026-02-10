#pragma once

/**
 * @file recycler_interface.h
 * @brief Abstract interface for token recycling systems
 *
 * This interface supports multiple recycling implementations:
 * - BasicTokenRecycler: Simple caching (MVP placeholder)
 * - SigmaRSUEngine: Full ΣLANG×RSU compression (post-MVP)
 *
 * The SigmaRSUEngine achieves 30-250x token efficiency through:
 * 1. ΣLANG semantic encoding (10-50x compression)
 * 2. RSU temporal recycling (3-5x efficiency)
 * 3. Delta chain encoding (1.5-2x additional)
 * 4. KV cache recycling with semantic anchors
 *
 * Architecture Reference: docs/SIGMALANG_INTEGRATION.md
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <memory>

namespace ryzanstein_llm
{
    namespace recycler
    {

        /**
         * @brief Processing mode indicating how input was handled
         */
        enum class ProcessingMode
        {
            FAST_PATH,       // Below threshold - bypassed full processing
            EXACT_HIT,       // Identical content found - maximum efficiency
            APPROXIMATE_HIT, // Similar content found - delta encoding used
            DELTA_CHAIN,     // Part of conversation chain - chain compression
            FRESH_ENCODE     // New content - full encoding required
        };

        /**
         * @brief Result of processing input through the recycler
         */
        struct ProcessedContext
        {
            // Tokens to use for inference (may be reduced from original)
            std::vector<int32_t> tokens;

            // Reference to stored RSU (empty if not cached)
            std::string rsu_reference;

            // Reference to conversation chain (empty if standalone)
            std::string chain_reference;

            // How the input was processed
            ProcessingMode processing_mode = ProcessingMode::FRESH_ENCODE;

            // Compression metrics
            float compression_ratio = 1.0f;
            size_t effective_tokens = 0;
            size_t original_token_count = 0;

            // KV cache reference if recyclable states available
            std::optional<size_t> recycled_kv_sequence_id;

            // Parent RSU if delta-encoded
            std::optional<std::string> delta_from;

            // Convenience methods
            size_t tokens_saved() const
            {
                return original_token_count > effective_tokens
                           ? original_token_count - effective_tokens
                           : 0;
            }

            float efficiency_gain_percent() const
            {
                return original_token_count > 0
                           ? (static_cast<float>(tokens_saved()) / original_token_count) * 100.0f
                           : 0.0f;
            }

            bool has_recycled_kv() const
            {
                return recycled_kv_sequence_id.has_value();
            }
        };

        /**
         * @brief Context injection result for inference preparation
         */
        struct InjectionResult
        {
            // Combined tokens for inference (history + current)
            std::vector<int32_t> tokens;

            // KV cache sequence ID if recyclable (nullopt = compute fresh)
            std::optional<size_t> recycled_kv_sequence_id;

            // Number of tokens that were recycled vs computed fresh
            size_t recycled_token_count = 0;
            size_t fresh_token_count = 0;
        };

        /**
         * @brief Statistics for monitoring recycler performance
         */
        struct RecyclerStatistics
        {
            // Processing counts
            uint64_t inputs_processed = 0;
            uint64_t exact_hits = 0;
            uint64_t approximate_hits = 0;
            uint64_t delta_encodes = 0;
            uint64_t fresh_encodes = 0;
            uint64_t fast_path_bypasses = 0;

            // Token metrics
            uint64_t total_tokens_input = 0;
            uint64_t total_tokens_output = 0;
            uint64_t total_kv_cache_hits = 0;

            // Storage metrics
            uint64_t rsus_created = 0;
            uint64_t rsus_promoted = 0;
            uint64_t rsus_demoted = 0;
            uint64_t chains_created = 0;

            // Derived metrics
            float hit_rate() const
            {
                return inputs_processed > 0
                           ? static_cast<float>(exact_hits + approximate_hits) / inputs_processed
                           : 0.0f;
            }

            float average_compression() const
            {
                return total_tokens_output > 0
                           ? static_cast<float>(total_tokens_input) / total_tokens_output
                           : 1.0f;
            }

            uint64_t total_tokens_saved() const
            {
                return total_tokens_input > total_tokens_output
                           ? total_tokens_input - total_tokens_output
                           : 0;
            }
        };

        /**
         * @brief Abstract interface for token recycling implementations
         *
         * Implementations:
         * - BasicTokenRecycler: Simple LRU caching (MVP)
         * - SigmaRSUEngine: Full ΣLANG×RSU compression (post-MVP)
         */
        class ITokenRecycler
        {
        public:
            virtual ~ITokenRecycler() = default;

            /**
             * @brief Process input tokens through the recycling pipeline
             *
             * This is the main entry point called by ContextManager before inference.
             * Analyzes input, checks for existing content, returns optimized context.
             *
             * @param tokens Raw input tokens from tokenizer
             * @param conversation_id Optional conversation ID for chain tracking
             * @return ProcessedContext with optimized tokens and metadata
             */
            virtual ProcessedContext process_input(
                const std::vector<int32_t> &tokens,
                const std::string &conversation_id = "") = 0;

            /**
             * @brief Prepare optimized context for injection into inference
             *
             * Called by ContextManager when assembling the full context window.
             * Reconstructs historical content and merges with current tokens.
             *
             * @param base_tokens Current turn's tokens
             * @param rsu_references RSU IDs to include from conversation history
             * @param max_tokens Maximum context window size
             * @return InjectionResult with combined tokens and recycled KV info
             */
            virtual InjectionResult prepare_context_injection(
                const std::vector<int32_t> &base_tokens,
                const std::vector<std::string> &rsu_references,
                size_t max_tokens) = 0;

            /**
             * @brief Register KV cache states for future recycling
             *
             * Called by CacheManager after inference completes.
             * Links KV states to RSU for semantic-aware recycling.
             *
             * @param rsu_reference RSU ID
             * @param kv_sequence_id KV cache sequence identifier
             * @param anchor_positions Token positions marked as anchors
             */
            virtual void register_kv_cache(
                const std::string &rsu_reference,
                size_t kv_sequence_id,
                const std::vector<int32_t> &anchor_positions) = 0;

            /**
             * @brief Get performance statistics
             * @return Current statistics snapshot
             */
            virtual RecyclerStatistics get_statistics() const = 0;

            /**
             * @brief Reset all statistics counters
             */
            virtual void reset_statistics() = 0;

            /**
             * @brief Check if recycler is using ΣLANG compression
             * @return true if ΣLANG is active, false for basic recycling
             */
            virtual bool is_sigma_enabled() const = 0;

            /**
             * @brief Get the name/version of this recycler implementation
             * @return Implementation identifier string
             */
            virtual std::string get_implementation_name() const = 0;
        };

        /**
         * @brief Factory for creating token recycler instances
         */
        class TokenRecyclerFactory
        {
        public:
            /**
             * @brief Create a token recycler based on configuration
             *
             * @param use_sigma If true, creates SigmaRSUEngine (requires trained codebook)
             *                  If false, creates BasicTokenRecycler (MVP default)
             * @param config_path Path to configuration file (optional)
             * @return Unique pointer to recycler instance
             */
            static std::unique_ptr<ITokenRecycler> create(
                bool use_sigma = false,
                const std::string &config_path = "");
        };

    } // namespace recycler
} // namespace ryzanstein_llm
