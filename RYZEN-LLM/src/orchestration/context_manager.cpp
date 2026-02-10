#include "context_manager.h"
#include "../recycler/basic_recycler.h"

namespace ryzanstein_llm
{
    namespace orchestration
    {

        ContextManager::ContextManager()
        {
            // Default to basic recycler (MVP)
            // Post-MVP: Switch to SigmaRSUEngine via config
            recycler_ = recycler::TokenRecyclerFactory::create(
                false, // use_sigma = false for MVP
                ""     // config_path
            );
        }

        void ContextManager::set_recycler(
            std::unique_ptr<recycler::ITokenRecycler> recycler)
        {
            recycler_ = std::move(recycler);
        }

        PreparedContext ContextManager::prepare_for_inference(
            const std::vector<int32_t> &input_tokens,
            const std::string &conversation_id)
        {
            PreparedContext result;

            if (recycler_)
            {
                // Use recycler for potential compression/caching
                auto processed = recycler_->process_input(input_tokens, conversation_id);

                result.tokens = std::move(processed.tokens);
                result.rsu_reference = processed.rsu_reference;
                result.recycled_kv_sequence_id = processed.recycled_kv_sequence_id;
                result.compression_ratio = processed.compression_ratio;
                result.effective_tokens = processed.effective_tokens;

                // Log compression metrics (useful for debugging)
                if (processed.processing_mode != recycler::ProcessingMode::FAST_PATH)
                {
                    // FUTURE: Emit metrics for monitoring
                    // - Log compression_ratio
                    // - Log processing_mode
                    // - Track performance over time
                }
            }
            else
            {
                // Fallback: no recycling
                result.tokens = input_tokens;
                result.compression_ratio = 1.0f;
                result.effective_tokens = input_tokens.size();
            }

            return result;
        }

        void ContextManager::post_inference_hook(
            const std::string &rsu_reference,
            size_t kv_sequence_id,
            const std::vector<int32_t> &anchor_positions)
        {
            if (recycler_ && !rsu_reference.empty())
            {
                recycler_->register_kv_cache(rsu_reference, kv_sequence_id, anchor_positions);
            }
        }

        std::optional<recycler::RecyclerStatistics> ContextManager::get_recycler_stats() const
        {
            if (recycler_)
            {
                return recycler_->get_statistics();
            }
            return std::nullopt;
        }

        bool ContextManager::is_sigma_active() const
        {
            return recycler_ && recycler_->is_sigma_enabled();
        }

    } // namespace orchestration
} // namespace ryzanstein_llm
