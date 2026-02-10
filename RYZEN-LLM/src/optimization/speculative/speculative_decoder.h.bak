#pragma once

#include "draft_model.h"
#include "verifier.h"

#include <vector>
#include <cstdint>
#include <memory>
#include <string>

namespace ryzen_llm
{
    namespace speculative
    {

        /**
         * @brief Configuration for speculative decoding system
         *
         * Combines configuration for both draft model and verifier, with additional
         * parameters for overall system behavior.
         */
        struct SpeculativeConfig
        {
            // Draft model configuration
            DraftModelConfig draft_config;

            // Verifier configuration
            VerifierConfig verifier_config;

            // System behavior
            bool enable_speculative_decoding; // True: use speculative, False: direct inference
            uint32_t batch_size;              // Batch processing (typically 1 for generation)
            bool enable_logging;              // Log detailed execution trace
        };

        /**
         * @brief Unified statistics for speculative decoding system
         *
         * Combines metrics from draft model and verifier to give overall picture
         * of speculative decoding performance.
         */
        struct SpeculativeStats
        {
            DraftModelStats draft_stats;
            VerifierStats verifier_stats;
            uint64_t total_tokens_generated; // Total output tokens
            uint64_t total_forward_passes;   // Target model forward passes

            // Constructor
            SpeculativeStats()
                : total_tokens_generated(0), total_forward_passes(0) {}

            /**
             * @brief Calculate overall speedup vs non-speculative
             *
             * Speedup = tokens_generated / target_forward_passes
             * (Baseline: 1 forward pass = 1 token, so speedup > 1 means faster)
             *
             * @return Speedup factor (typical: 2.0-3.5x)
             */
            float get_overall_speedup() const
            {
                if (total_forward_passes == 0)
                    return 1.0f;
                return static_cast<float>(total_tokens_generated) /
                       static_cast<float>(total_forward_passes);
            }

            /**
             * @brief Calculate theoretical speedup without rejections
             *
             * If all draft tokens were accepted (0% rejection rate):
             * speedup = avg_K (number of tokens per batch)
             *
             * @return Theoretical maximum speedup
             */
            float get_theoretical_max_speedup() const
            {
                if (draft_stats.num_inferences == 0)
                    return 1.0f;
                return draft_stats.get_avg_k();
            }

            /**
             * @brief Calculate efficiency vs theoretical maximum
             *
             * efficiency = actual_speedup / theoretical_speedup
             * (100% = perfect, < 100% = rejections reducing speedup)
             *
             * @return Efficiency as percentage (0-100)
             */
            float get_efficiency_percent() const
            {
                float theoretical = get_theoretical_max_speedup();
                if (theoretical < 1.0f)
                    return 100.0f;
                return 100.0f * get_overall_speedup() / theoretical;
            }

            // Reset all statistics
            void reset()
            {
                draft_stats.reset();
                verifier_stats.reset();
                total_tokens_generated = 0;
                total_forward_passes = 0;
            }
        };

        /**
         * @brief Main speculative decoding orchestrator
         *
         * High-level interface combining draft model and verifier for fast token
         * generation. Handles state management, acceptance feedback loops, and
         * dynamic K tuning.
         *
         * Key responsibilities:
         * 1. Call draft model to get K candidates
         * 2. Call verifier to validate against target
         * 3. Return final token sequence
         * 4. Update statistics and feedback to draft model
         * 5. Tune K based on acceptance rates
         *
         * Performance characteristics:
         * - With 0% rejections: K× speedup
         * - With 50% rejections: ~0.7-1.5× speedup
         * - With 100% rejections: ~0.3× speedup (worse than baseline)
         *
         * Example usage:
         * @code
         *   SpeculativeConfig config = {...};
         *   SpeculativeDecoder speculative(config);
         *
         *   std::vector<int> prefix = {101, 2054, 2003};
         *   std::vector<int> next_tokens = speculative.decode_next_tokens(prefix, 4);
         *   // next_tokens has 1-4 tokens depending on acceptance
         * @endcode
         */
        class SpeculativeDecoder
        {
        public:
            /**
             * @brief Initialize speculative decoder
             * @param config Configuration for system behavior
             * @throws std::invalid_argument if config is invalid
             */
            explicit SpeculativeDecoder(const SpeculativeConfig &config);

            /**
             * @brief Decode next tokens using speculative decoding
             *
             * Main entry point for speculative decoding. Generates K candidate tokens
             * from draft model, verifies them against target model, and returns final
             * tokens. Automatically updates statistics and feedback.
             *
             * Algorithm:
             * 1. Call draft_model.generate_candidates(prefix, K)
             * 2. Call verifier.verify(prefix, candidates, draft_probs)
             * 3. Update statistics
             * 4. Return verified tokens
             * 5. Optionally adjust K based on acceptance rate
             *
             * @param prefix Current context (token IDs)
             * @param max_tokens Maximum tokens to generate in this batch (≤ max_K)
             * @return Vector of accepted tokens (1 to max_tokens elements)
             *
             * @note Returns empty vector only on error (invalid prefix, etc.)
             * @note If speculative decoding disabled, falls back to single-token inference
             */
            std::vector<int> decode_next_tokens(
                const std::vector<int> &prefix,
                uint32_t max_tokens);

            /**
             * @brief Single-token fallback when speculative decoding is disabled
             *
             * Direct inference through target model without draft model acceleration.
             * Used as fallback for edge cases or when speculative disabled.
             *
             * @param prefix Current context
             * @return Single next token
             */
            int decode_single_token(const std::vector<int> &prefix);

            /**
             * @brief Get current statistics
             * @return Combined statistics from draft and verifier
             */
            SpeculativeStats get_stats() const { return stats_; }

            /**
             * @brief Reset all statistics
             */
            void reset_stats()
            {
                stats_.reset();
                draft_model_->reset_stats();
                verifier_->reset_stats();
            }

            /**
             * @brief Get draft model
             * @return Pointer to draft model (non-owning)
             */
            DraftModel *get_draft_model() { return draft_model_.get(); }

            /**
             * @brief Get verifier
             * @return Pointer to verifier (non-owning)
             */
            Verifier *get_verifier() { return verifier_.get(); }

            /**
             * @brief Get configuration
             * @return Reference to current configuration
             */
            const SpeculativeConfig &get_config() const { return config_; }

            /**
             * @brief Set draft length (K)
             * @param K New draft length
             */
            void set_K(uint32_t K)
            {
                if (draft_model_)
                {
                    draft_model_->set_K(K);
                }
            }

            /**
             * @brief Get current draft length
             * @return Current K value
             */
            uint32_t get_K() const
            {
                if (draft_model_)
                    return draft_model_->get_current_K();
                return 0;
            }

        private:
            SpeculativeConfig config_;
            std::unique_ptr<DraftModel> draft_model_;
            std::unique_ptr<Verifier> verifier_;
            SpeculativeStats stats_;

            /**
             * @brief Internal: Log execution trace
             *
             * If enable_logging is true, writes detailed information about
             * draft generation and verification to logger.
             *
             * @param message Message to log
             */
            void log(const std::string &message) const;

            /**
             * @brief Internal: Update statistics after decode
             *
             * Records tokens generated, forward passes, and acceptance metrics
             * for performance analysis.
             *
             * @param tokens_generated Number of output tokens
             * @param num_target_passes Number of target forward passes
             */
            void update_stats(uint32_t tokens_generated, uint32_t num_target_passes);
        };

    } // namespace speculative
} // namespace ryzen_llm
