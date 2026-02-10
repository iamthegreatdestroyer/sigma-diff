#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace ryzen_llm
{
    namespace speculative
    {

        // ============================================================================
        // Configuration Structure
        // ============================================================================

        struct DraftModelConfig
        {
            // Model architecture parameters
            uint32_t vocab_size;  // Size of vocabulary
            uint32_t hidden_dim;  // Hidden dimension size
            uint32_t hidden_size; // Alias for hidden_dim (compatibility)
            uint32_t num_layers;  // Number of model layers
            uint32_t num_heads;   // Number of attention heads
            uint32_t max_seq_len; // Maximum sequence length

            // Speculative decoding parameters
            uint32_t min_K;              // Minimum number of candidates
            uint32_t max_K;              // Maximum number of candidates
            uint32_t K_adjust_frequency; // How often to adjust K (in inferences)

            // Sampling parameters
            float temperature; // Sampling temperature (>0)
            uint32_t top_k;    // Top-k sampling (0 = disabled)
            float top_p;       // Top-p (nucleus) sampling (0-1)

            // Adaptive parameters
            float acceptance_rate_target; // Target acceptance rate for K adjustment (0-1)
            bool enable_statistics;       // Enable statistics tracking

            // Default constructor
            DraftModelConfig()
                : vocab_size(32000), hidden_dim(512), hidden_size(512),
                  num_layers(6), num_heads(8), max_seq_len(2048),
                  min_K(1), max_K(5), K_adjust_frequency(10),
                  temperature(1.0f), top_k(0), top_p(0.9f),
                  acceptance_rate_target(0.7f), enable_statistics(true) {}
        };

        // ============================================================================
        // Statistics Structure
        // ============================================================================

        struct DraftModelStats
        {
            uint64_t num_inferences = 0;     // Total number of inferences
            uint64_t num_accepted = 0;       // Total accepted tokens
            uint64_t total_draft_tokens = 0; // Total draft tokens generated
            uint64_t total_K = 0;            // Sum of K values used

            // Get current acceptance rate
            float get_acceptance_rate() const
            {
                if (total_draft_tokens == 0)
                {
                    return 0.0f;
                }
                return static_cast<float>(num_accepted) / static_cast<float>(total_draft_tokens);
            }

            // Get average K value
            float get_avg_k() const
            {
                if (num_inferences == 0)
                {
                    return 0.0f;
                }
                return static_cast<float>(total_K) / static_cast<float>(num_inferences);
            }

            // Reset statistics
            void reset()
            {
                num_inferences = 0;
                num_accepted = 0;
                total_draft_tokens = 0;
                total_K = 0;
            }
        };

        // ============================================================================
        // DraftModel Class
        // ============================================================================

        class DraftModel
        {
        public:
            // Constructor
            explicit DraftModel(const DraftModelConfig &config);

            // Destructor
            ~DraftModel() = default;

            // Delete copy operations (non-copyable)
            DraftModel(const DraftModel &) = delete;
            DraftModel &operator=(const DraftModel &) = delete;

            // Allow move operations (movable)
            DraftModel(DraftModel &&) noexcept = default;
            DraftModel &operator=(DraftModel &&) noexcept = default;

            // ========================================================================
            // Public API Methods
            // ========================================================================

            /**
             * Generate K candidate tokens for speculative decoding.
             *
             * @param prefix Input token sequence
             * @param K Number of candidates to generate (uses current_K_ if 0)
             * @return Vector of K candidate tokens, empty vector on error
             *
             * @note Validates:
             *       - prefix is not empty
             *       - K is within [min_K, max_K]
             *       - prefix size is within max_seq_len
             *       - all tokens are valid (< vocab_size)
             *
             * @performance Time: O(K * forward_pass_time)
             *             Space: O(K * hidden_dim)
             */
            std::vector<int> generate_candidates(
                const std::vector<int> &prefix,
                uint32_t K = 0);

            /**
             * Record whether a draft token was accepted by verifier.
             *
             * @param token_id The token ID that was verified
             * @param was_accepted Whether the token was accepted
             *
             * @note Used for statistics and adaptive K adjustment
             */
            void record_acceptance(int token_id, bool was_accepted);

            // ========================================================================
            // Getter Methods
            // ========================================================================

            /// Get current configuration
            const DraftModelConfig &get_config() const { return config_; }

            /// Get current K value (number of candidates)
            uint32_t get_current_K() const { return current_K_; }

            /// Get statistics
            const DraftModelStats &get_stats() const { return stats_; }

            /// Get statistics (mutable)
            DraftModelStats &get_stats_mut() { return stats_; }

            // ========================================================================
            // Setter Methods
            // ========================================================================

            /// Set current K value (number of candidates)
            void set_current_K(uint32_t K)
            {
                if (K >= config_.min_K && K <= config_.max_K)
                {
                    current_K_ = K;
                }
                // Otherwise, remain unchanged
            }

            /// Alias for set_current_K (for compatibility)
            void set_K(uint32_t K) { set_current_K(K); }

            /// Reset statistics
            void reset_stats()
            {
                stats_.reset();
            }

        private:
            // ========================================================================
            // Private Helper Methods
            // ========================================================================

            /**
             * Forward pass through draft model.
             *
             * @param prefix Input token sequence
             * @return Logits vector of size vocab_size, empty on error
             *
             * @note Validates:
             *       - prefix is not empty
             *       - prefix size is within max_seq_len
             *       - all tokens are valid (< vocab_size)
             *
             * @performance Time: O(seq_len * hidden_dim)
             *             Space: O(hidden_dim)
             */
            std::vector<float> forward(const std::vector<int> &prefix);

            /**
             * Convert logits to probability distribution.
             *
             * Applies:
             * 1. Temperature scaling
             * 2. Softmax normalization
             * 3. Top-k filtering (if enabled)
             * 4. Top-p filtering (if enabled)
             *
             * @param logits Raw logits from model (size vocab_size)
             * @return Probability distribution (size vocab_size), sums to ~1.0
             *
             * @performance Time: O(vocab_size * log(vocab_size)) if top-k/p enabled
             */
            std::vector<float> sample_distribution(const std::vector<float> &logits);

            /**
             * Sample a single token from probability distribution.
             *
             * Uses inverse transform sampling (cumulative probability method):
             * - Generate uniform random u in [0, 1)
             * - Find first token i where cumsum(probs[0..i]) >= u
             *
             * @param probs Probability distribution (must sum to ~1.0)
             * @return Sampled token ID, or -1 on error
             *
             * @note Distribution validation:
             *       - sum must be in [0.9, 1.1] (numerical tolerance)
             *       - size must equal vocab_size
             *
             * @performance Time: O(vocab_size) worst case, O(k) average
             *             Space: O(1)
             */
            int sample_token(const std::vector<float> &probs);

            /**
             * Adaptively adjust K based on acceptance rate.
             *
             * Strategy:
             * - If acceptance_rate < target - 0.05: decrease K
             * - If acceptance_rate > target + 0.05: increase K
             * - Otherwise: keep current K
             *
             * @note Only called if enable_statistics is true and
             *       inferences % K_adjust_frequency == 0
             *
             * @performance Time: O(1)
             */
            void adjust_K_adaptive();

            /**
             * Softmax with numerical stability.
             *
             * Implementation:
             * 1. Find max logit (for stability)
             * 2. Compute exp(logit[i] - max)
             * 3. Normalize by sum of exponentials
             *
             * @param logits Input logits
             * @return Probabilities that sum to 1.0
             *
             * @performance Time: O(vocab_size)
             *             Space: O(vocab_size)
             */
            std::vector<float> softmax(const std::vector<float> &logits);

            /**
             * Apply temperature scaling to logits.
             *
             * Formula: scaled[i] = logits[i] / temperature
             *
             * Effect:
             * - temperature > 1: flattens distribution (more uniform)
             * - temperature = 1: no effect
             * - temperature < 1: sharpens distribution (more peaked)
             *
             * @param logits Input logits
             * @param temperature Scaling factor (must be > 0)
             * @return Scaled logits
             */
            std::vector<float> apply_temperature(
                const std::vector<float> &logits,
                float temperature);

            /**
             * Apply top-k filtering to probability distribution.
             *
             * Strategy:
             * 1. Sort tokens by probability (descending)
             * 2. Zero out all but top-k tokens
             * 3. Renormalize probabilities to sum to 1.0
             *
             * @param probs Input probability distribution
             * @param k Number of top tokens to keep (0 = no filtering)
             * @return Filtered and renormalized distribution
             *
             * @performance Time: O(vocab_size * log(vocab_size))
             *             Space: O(vocab_size)
             */
            std::vector<float> apply_top_k(
                const std::vector<float> &probs,
                uint32_t k);

            /**
             * Apply top-p (nucleus) filtering to probability distribution.
             *
             * Strategy:
             * 1. Sort tokens by probability (descending)
             * 2. Accumulate probabilities until exceeding threshold p
             * 3. Zero out tokens beyond threshold
             * 4. Renormalize probabilities to sum to 1.0
             *
             * @param probs Input probability distribution
             * @param p Cumulative probability threshold (0-1)
             * @return Filtered and renormalized distribution
             *
             * @performance Time: O(vocab_size * log(vocab_size))
             *             Space: O(vocab_size)
             */
            std::vector<float> apply_top_p(
                const std::vector<float> &probs,
                float p);

            // ========================================================================
            // Member Variables
            // ========================================================================

            DraftModelConfig config_; // Configuration
            uint32_t current_K_;      // Current number of candidates to generate
            DraftModelStats stats_;   // Performance statistics
        };

    } // namespace speculative
} // namespace ryzen_llm