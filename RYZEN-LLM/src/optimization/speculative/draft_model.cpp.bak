#include "draft_model.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <stdexcept>

namespace ryzen_llm
{
    namespace speculative
    {

        // ============================================================================
        // DraftModel Constructor & Public Methods
        // ============================================================================

        DraftModel::DraftModel(const DraftModelConfig &config)
            : config_(config), current_K_(config.min_K)
        {
            // Validate configuration
            if (config_.vocab_size == 0)
            {
                throw std::invalid_argument("vocab_size must be > 0");
            }
            if (config_.hidden_dim == 0)
            {
                throw std::invalid_argument("hidden_dim must be > 0");
            }
            if (config_.max_seq_len == 0)
            {
                throw std::invalid_argument("max_seq_len must be > 0");
            }
            if (config_.min_K == 0 || config_.max_K == 0 || config_.min_K > config_.max_K)
            {
                throw std::invalid_argument("Invalid K range: min_K and max_K must be > 0 and min_K <= max_K");
            }
            if (config_.K_adjust_frequency == 0)
            {
                throw std::invalid_argument("K_adjust_frequency must be > 0");
            }
            if (config_.temperature <= 0.0f)
            {
                throw std::invalid_argument("temperature must be > 0");
            }
            if (config_.acceptance_rate_target < 0.0f || config_.acceptance_rate_target > 1.0f)
            {
                throw std::invalid_argument("acceptance_rate_target must be in [0, 1]");
            }
            if (config_.top_p < 0.0f || config_.top_p > 1.0f)
            {
                throw std::invalid_argument("top_p must be in [0, 1]");
            }
        }

        std::vector<int> DraftModel::generate_candidates(
            const std::vector<int> &prefix,
            uint32_t K)
        {

            // Validate inputs
            if (prefix.empty())
            {
                return {};
            }
            if (K == 0 || K > config_.max_K)
            {
                return {};
            }
            if (prefix.size() > config_.max_seq_len)
            {
                return {};
            }

            // Validate all tokens are valid
            for (int token : prefix)
            {
                if (token < 0 || token >= static_cast<int>(config_.vocab_size))
                {
                    return {};
                }
            }

            std::vector<int> candidates;
            candidates.reserve(K);

            // Generate K candidates sequentially
            // Each candidate is conditioned on the prefix + previously generated tokens
            std::vector<int> working_prefix = prefix;

            for (uint32_t i = 0; i < K; ++i)
            {
                // Get logits from draft model
                std::vector<float> logits = forward(working_prefix);
                if (logits.size() != config_.vocab_size)
                {
                    return {}; // Forward pass failed
                }

                // Apply sampling strategy
                std::vector<float> probs = sample_distribution(logits);

                // Sample a token
                int token = sample_token(probs);
                if (token < 0)
                {
                    break; // Sampling failed, stop here
                }

                candidates.push_back(token);
                working_prefix.push_back(token);

                // Prevent prefix from growing beyond max_seq_len
                if (working_prefix.size() > config_.max_seq_len)
                {
                    break;
                }
            }

            // Update statistics
            if (config_.enable_statistics)
            {
                stats_.num_inferences++;
                stats_.total_draft_tokens += candidates.size();

                // Check if we should adjust K
                if (stats_.num_inferences % config_.K_adjust_frequency == 0)
                {
                    adjust_K_adaptive();
                }
            }

            return candidates;
        }

        void DraftModel::record_acceptance(int token_id, bool was_accepted)
        {
            if (!config_.enable_statistics)
            {
                return;
            }

            // Validate token_id
            if (token_id < 0 || token_id >= static_cast<int>(config_.vocab_size))
            {
                return;
            }

            // Increment total inferences count
            stats_.num_inferences++;

            // Increment total draft tokens (for acceptance rate calculation)
            stats_.total_draft_tokens++;

            if (was_accepted)
            {
                stats_.num_accepted++;
            }
        }

        // ============================================================================
        // Private Helper Methods
        // ============================================================================

        std::vector<float> DraftModel::forward(const std::vector<int> &prefix)
        {
            // TODO: Implement actual draft model inference
            // For now, return uniform distribution as placeholder

            if (prefix.empty() || prefix.size() > config_.max_seq_len)
            {
                return {};
            }

            // Validate all tokens
            for (int token : prefix)
            {
                if (token < 0 || token >= static_cast<int>(config_.vocab_size))
                {
                    return {};
                }
            }

            // Placeholder: uniform logits (all tokens equally likely)
            std::vector<float> logits(config_.vocab_size, 0.0f);

            // In actual implementation, this would:
            // 1. Embed the prefix tokens
            // 2. Pass through draft model layers
            // 3. Project to vocab_size logits
            // 4. Return logits for next token position

            return logits;
        }

        std::vector<float> DraftModel::sample_distribution(const std::vector<float> &logits)
        {
            if (logits.size() != config_.vocab_size)
            {
                return {};
            }

            // Apply temperature scaling
            std::vector<float> scaled_logits = apply_temperature(logits, config_.temperature);

            // Apply softmax to get probabilities
            std::vector<float> probs = softmax(scaled_logits);
            if (probs.empty())
            {
                return {};
            }

            // Apply top-k filtering if enabled (top_k > 0)
            if (config_.top_k > 0 && config_.top_k < config_.vocab_size)
            {
                probs = apply_top_k(probs, config_.top_k);
            }

            // Apply top-p filtering if enabled (top_p < 1.0)
            if (config_.top_p < 1.0f && config_.top_p > 0.0f)
            {
                probs = apply_top_p(probs, config_.top_p);
            }

            return probs;
        }

        int DraftModel::sample_token(const std::vector<float> &probs)
        {
            if (probs.empty() || probs.size() != config_.vocab_size)
            {
                return -1;
            }

            // Validate probabilities sum to approximately 1.0
            float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
            if (sum < 0.9f || sum > 1.1f)
            {
                return -1; // Invalid distribution
            }

            // Use inverse transform sampling (cumulative probability)
            static thread_local std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            float u = dist(rng);
            float cumulative = 0.0f;

            for (uint32_t i = 0; i < probs.size(); ++i)
            {
                cumulative += probs[i];
                if (u <= cumulative)
                {
                    return static_cast<int>(i);
                }
            }

            // Fallback: return last token (shouldn't reach here with valid probs)
            return static_cast<int>(probs.size() - 1);
        }

        void DraftModel::adjust_K_adaptive()
        {
            if (!config_.enable_statistics)
            {
                return;
            }

            float current_acceptance = stats_.get_acceptance_rate();
            float target_acceptance = config_.acceptance_rate_target;

            // If acceptance rate is below target, decrease K (generate fewer candidates)
            // If acceptance rate is above target, increase K (generate more candidates)

            const float tolerance = 0.05f; // Allow 5% deviation from target

            if (current_acceptance < target_acceptance - tolerance)
            {
                // Too many rejections, decrease K
                uint32_t new_K = std::max(config_.min_K, current_K_ - 1);
                current_K_ = new_K;
            }
            else if (current_acceptance > target_acceptance + tolerance)
            {
                // High acceptance rate, can increase K
                uint32_t new_K = std::min(config_.max_K, current_K_ + 1);
                current_K_ = new_K;
            }
            // Otherwise keep current K if within tolerance
        }

        std::vector<float> DraftModel::softmax(const std::vector<float> &logits)
        {
            if (logits.empty())
            {
                return {};
            }

            // Numerical stability: subtract max to prevent overflow
            float max_logit = *std::max_element(logits.begin(), logits.end());

            std::vector<float> probs(logits.size());
            float sum_exp = 0.0f;

            for (size_t i = 0; i < logits.size(); ++i)
            {
                float exp_val = std::exp(logits[i] - max_logit);
                probs[i] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            if (sum_exp > 0.0f)
            {
                for (float &prob : probs)
                {
                    prob /= sum_exp;
                }
            }
            else
            {
                // Uniform fallback if sum is zero (shouldn't happen)
                float uniform_prob = 1.0f / logits.size();
                std::fill(probs.begin(), probs.end(), uniform_prob);
            }

            return probs;
        }

        std::vector<float> DraftModel::apply_temperature(
            const std::vector<float> &logits,
            float temperature)
        {

            if (temperature <= 0.0f)
            {
                return {};
            }

            std::vector<float> scaled(logits.size());
            for (size_t i = 0; i < logits.size(); ++i)
            {
                scaled[i] = logits[i] / temperature;
            }
            return scaled;
        }

        std::vector<float> DraftModel::apply_top_k(
            const std::vector<float> &probs,
            uint32_t k)
        {

            if (k == 0 || probs.empty() || k >= probs.size())
            {
                return probs; // No filtering needed
            }

            // Create indices sorted by probability (descending)
            std::vector<size_t> indices(probs.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                      [&probs](size_t a, size_t b)
                      { return probs[a] > probs[b]; });

            // Zero out tokens below top-k
            std::vector<float> filtered = probs;
            for (size_t i = k; i < filtered.size(); ++i)
            {
                filtered[indices[i]] = 0.0f;
            }

            // Renormalize
            float sum = std::accumulate(filtered.begin(), filtered.end(), 0.0f);
            if (sum > 0.0f)
            {
                for (float &prob : filtered)
                {
                    prob /= sum;
                }
            }

            return filtered;
        }

        std::vector<float> DraftModel::apply_top_p(
            const std::vector<float> &probs,
            float p)
        {

            if (p <= 0.0f || p >= 1.0f || probs.empty())
            {
                return probs; // No filtering needed
            }

            // Create indices sorted by probability (descending)
            std::vector<size_t> indices(probs.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                      [&probs](size_t a, size_t b)
                      { return probs[a] > probs[b]; });

            // Accumulate until threshold, with epsilon tolerance
            std::vector<size_t> selected;
            float cumsum = 0.0f;
            const float EPSILON = 1e-7f; // For floating-point comparisons

            for (size_t idx : indices)
            {
                cumsum += probs[idx];
                selected.push_back(idx);
                if (cumsum >= p - EPSILON)
                {
                    break;
                }
            }

            // Ensure we selected at least 1 token
            if (selected.empty())
            {
                selected.push_back(indices[0]);
            }

            // Create filtered probabilities
            std::vector<float> filtered(probs.size(), 0.0f);
            float selected_sum = 0.0f;
            for (size_t idx : selected)
            {
                selected_sum += probs[idx];
            }

            // Renormalize selected probabilities
            for (size_t idx : selected)
            {
                filtered[idx] = probs[idx] / selected_sum;
            }

            return filtered;
        }

    } // namespace speculative
} // namespace ryzen_llm
