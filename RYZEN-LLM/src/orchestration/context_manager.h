#pragma once

/**
 * @file context_manager.h
 * @brief Context management with token recycler integration
 *
 * Manages context preparation for inference, integrating with token
 * recycling systems to enable compression and KV cache reuse.
 */

#include "../recycler/recycler_interface.h"
#include <vector>
#include <string>
#include <memory>
#include <optional>

namespace ryzanstein_llm
{
    namespace orchestration
    {

        /**
         * @brief Prepared context ready for inference
         */
        struct PreparedContext
        {
            // Optimized tokens for inference (potentially compressed)
            std::vector<int32_t> tokens;

            // RSU reference if recycled (empty if not)
            std::string rsu_reference;

            // Optional recycled KV cache sequence ID
            std::optional<size_t> recycled_kv_sequence_id;

            // Compression metrics for monitoring
            float compression_ratio = 1.0f;
            size_t effective_tokens = 0;
        };

        /**
         * @brief Context manager with token recycling support
         *
         * Coordinates between token recycler and inference engine.
         * Handles context preparation, compression, and KV cache registration.
         */
        class ContextManager
        {
        public:
            ContextManager();
            ~ContextManager() = default;

            /**
             * @brief Set the token recycler implementation
             *
             * Allows hot-swapping between BasicTokenRecycler (MVP) and
             * SigmaRSUEngine (post-MVP) without code changes.
             *
             * @param recycler Unique pointer to recycler instance
             */
            void set_recycler(std::unique_ptr<recycler::ITokenRecycler> recycler);

            /**
             * @brief Prepare context for inference
             *
             * Uses token recycler (if set) for potential compression and caching.
             * Returns optimized context ready for model inference.
             *
             * @param input_tokens Raw input tokens from tokenizer
             * @param conversation_id Optional conversation ID for chain tracking
             * @return PreparedContext with optimized tokens and metadata
             */
            PreparedContext prepare_for_inference(
                const std::vector<int32_t> &input_tokens,
                const std::string &conversation_id = "");

            /**
             * @brief Post-inference hook for KV cache registration
             *
             * Called after inference completes to link KV cache states
             * with RSU for future recycling.
             *
             * @param rsu_reference RSU ID from prepared context
             * @param kv_sequence_id KV cache sequence identifier
             * @param anchor_positions Token positions marked as semantic anchors
             */
            void post_inference_hook(
                const std::string &rsu_reference,
                size_t kv_sequence_id,
                const std::vector<int32_t> &anchor_positions);

            /**
             * @brief Get recycler statistics
             * @return Current statistics if recycler is set, empty otherwise
             */
            std::optional<recycler::RecyclerStatistics> get_recycler_stats() const;

            /**
             * @brief Check if ΣLANG compression is active
             * @return true if using SigmaRSUEngine, false otherwise
             */
            bool is_sigma_active() const;

        private:
            std::unique_ptr<recycler::ITokenRecycler> recycler_;
        };

    } // namespace orchestration
} // namespace ryzanstein_llm
