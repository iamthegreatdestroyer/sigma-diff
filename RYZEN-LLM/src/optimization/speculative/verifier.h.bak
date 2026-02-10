#pragma once

#include <cstdint>
#include <vector>

namespace ryzen_llm
{
    namespace speculative
    {

        // ============================================================================
        // Configuration Structure
        // ============================================================================

        struct VerifierConfig
        {
            uint32_t vocab_size;       // Size of vocabulary
            uint32_t hidden_size;      // Hidden dimension of model
            uint32_t num_layers;       // Number of transformer layers
            uint32_t num_heads;        // Number of attention heads
            float temperature;         // Sampling temperature (>0)
            float rejection_threshold; // Acceptance probability threshold (0-1)
            bool enable_statistics;    // Enable statistics tracking

            // Default constructor
            VerifierConfig()
                : vocab_size(0), hidden_size(0), num_layers(0), num_heads(0), temperature(1.0f), rejection_threshold(0.5f), enable_statistics(false)
            {
            }
        };

        // ============================================================================
        // Result Structure
        // ============================================================================

        struct VerifierResult
        {
            std::vector<int> accepted_tokens; // Tokens accepted by verifier
            uint32_t num_accepted = 0;        // Number of accepted tokens
            float acceptance_rate = 0.0f;     // Acceptance rate for this batch
        };

        // ============================================================================
        // Statistics Structure
        // ============================================================================

        struct VerifierStats
        {
            uint64_t num_verifications = 0; // Total verifications performed
            uint64_t num_rejections = 0;    // Total rejected tokens
            uint64_t total_tokens = 0;      // Total tokens verified

            // Get rejection rate
            float get_rejection_rate() const
            {
                if (total_tokens == 0)
                    return 0.0f;
                return static_cast<float>(num_rejections) / static_cast<float>(total_tokens);
            }

            // Get acceptance rate
            float get_acceptance_rate() const
            {
                return 1.0f - get_rejection_rate();
            }

            // Reset statistics
            void reset()
            {
                num_verifications = 0;
                num_rejections = 0;
                total_tokens = 0;
            }
        };

        // ============================================================================
        // Verifier Class
        // ============================================================================

        class Verifier
        {
        public:
            // Constructor
            explicit Verifier(const VerifierConfig &config);

            // Destructor
            ~Verifier() = default;

            // Delete copy operations (non-copyable)
            Verifier(const Verifier &) = delete;
            Verifier &operator=(const Verifier &) = delete;

            // Allow move operations (movable)
            Verifier(Verifier &&) noexcept = default;
            Verifier &operator=(Verifier &&) noexcept = default;

            // ========================================================================
            // Public API Methods
            // ========================================================================

            /**
             * Verify draft tokens against target model distribution.
             *
             * @param prefix Context token sequence
             * @param draft_tokens Candidate tokens from draft model
             * @param target_logits Logits from target model for each position
             * @return VerifierResult with accepted tokens and statistics
             *
             * @note target_logits should have size = draft_tokens.size()
             *       each element in target_logits should have size = vocab_size
             */
            VerifierResult verify(
                const std::vector<int> &prefix,
                const std::vector<int> &draft_tokens,
                const std::vector<std::vector<float>> &target_logits);

            /**
             * Sample a single token from target model distribution.
             *
             * @param target_logits Raw logits from target model (size vocab_size)
             * @return Sampled token ID, or -1 on error
             */
            int sample_token(const std::vector<float> &target_logits);

            // ========================================================================
            // Getter Methods
            // ========================================================================

            /// Get current configuration
            const VerifierConfig &get_config() const { return config_; }

            /// Get number of verifications performed
            uint64_t get_num_verifications() const { return num_verifications_; }

            /// Get number of rejections
            uint64_t get_num_rejections() const { return num_rejections_; }

            /// Get rejection rate
            float get_rejection_rate() const
            {
                if (num_verifications_ == 0)
                {
                    return 0.0f;
                }
                return static_cast<float>(num_rejections_) / static_cast<float>(num_verifications_);
            }

            /// Reset verification statistics
            void reset_stats()
            {
                num_verifications_ = 0;
                num_rejections_ = 0;
            }

        private:
            // ========================================================================
            // Private Helper Methods
            // ========================================================================

            /**
             * Check if draft token should be accepted.
             *
             * @param draft_token Token ID from draft model
             * @param target_probs Probability distribution from target model
             * @return true if token is acceptable, false if should be rejected
             */
            bool check_acceptance_criteria(
                int draft_token,
                const std::vector<float> &target_probs);

            /**
             * Rejection sample a token from target distribution.
             *
             * @param target_probs Target probability distribution
             * @param rejected_token Token that was rejected (avoid resampling this)
             * @return Sampled token ID, or -1 if failed
             */
            int rejection_sample(
                const std::vector<float> &target_probs,
                int rejected_token);

            /**
             * Softmax with numerical stability.
             *
             * @param logits Input logits
             * @return Probability distribution (sums to ~1.0)
             */
            std::vector<float> softmax(const std::vector<float> &logits);

            /**
             * Apply temperature scaling to logits.
             *
             * @param logits Input logits
             * @param temperature Scaling factor (>0)
             * @return Scaled logits
             */
            std::vector<float> apply_temperature(
                const std::vector<float> &logits,
                float temperature);

            // ========================================================================
            // Member Variables
            // ========================================================================

            VerifierConfig config_;      // Configuration
            uint64_t num_verifications_; // Total verifications performed
            uint64_t num_rejections_;    // Total rejections
        };

    } // namespace speculative
} // namespace ryzen_llm