/*
 * RWKV WKV (Weighted Key-Value) Operator
 * [REF:CC-004d] - Core Components: RWKV Attention-Free Runtime
 *
 * This module implements the WKV operator, which is the core of RWKV's
 * linear-complexity attention mechanism.
 *
 * Algorithm:
 * - Maintains recurrent state: (numerator, denominator) accumulators
 * - Updates state: num += exp(decay) * key * value, denom += exp(decay) * key
 * - Computes output: output = num / denom (receptance-weighted sum)
 * - Time complexity: O(N) vs O(N²) for traditional attention
 *
 * State Structure:
 * - numerator [hidden_dim]: accumulates weighted values
 * - denominator [hidden_dim]: accumulates weights for normalization
 * - Both are maintained across time steps for streaming inference
 *
 * References:
 * - RWKV: "Reinventing RNNs for the Transformer Era" (2023)
 * - https://arxiv.org/abs/2305.13048
 */

#ifndef RYZEN_LLM_RWKV_WKV_H
#define RYZEN_LLM_RWKV_WKV_H

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace ryzanstein_llm
{
    namespace rwkv
    {

        /**
         * WKV Operator Configuration
         *
         * Controls numerical stability and computation precision.
         */
        struct WKVConfig
        {
            // Stability epsilon to prevent division by zero
            float epsilon;

            // Whether to use low-precision intermediate computations for speed
            bool use_fp16_accumulation;

            // Normalization type: "softmax" (traditional), "exp" (linear with exp decay)
            std::string norm_type;

            // Default configuration
            WKVConfig()
                : epsilon(1e-6f),
                  use_fp16_accumulation(false),
                  norm_type("exp") {}
        };

        /**
         * WKV Operator
         *
         * Implements the recurrent weighted key-value mechanism that replaces
         * traditional attention in RWKV models.
         *
         * State Layout:
         * - numerator [hidden_dim]: accumulated weighted values
         * - denominator [hidden_dim]: accumulated weights
         * - Both shared across heads for efficiency
         *
         * Computation Per Token:
         * 1. Compute time decay: decay_t = exp(-exp(decay_weights_t))
         * 2. Update numerator: num += decay_t * key_t * value_t
         * 3. Update denominator: denom += decay_t * key_t
         * 4. Compute output: out = num / (denom + eps)
         * 5. Save state for next token
         */
        class WKVOperator
        {
        public:
            /**
             * Construct WKV operator
             *
             * @param hidden_dim Model hidden dimension
             * @param config WKV configuration
             */
            explicit WKVOperator(
                uint32_t hidden_dim,
                const WKVConfig &config = WKVConfig());

            ~WKVOperator() = default;

            /**
             * Initialize WKV state
             *
             * Resets numerator and denominator accumulators to zero.
             * Call this before processing a new sequence.
             */
            void initialize();

            /**
             * Forward pass: compute WKV output for single token
             *
             * Computation:
             * 1. Apply time decay to previous state: decay_t = exp(-exp(weight_t))
             * 2. Blend previous states with new values:
             *    num_t = num_{t-1} * decay_t + key_t * value_t
             *    denom_t = denom_{t-1} * decay_t + key_t
             * 3. Normalize output: out_t = num_t / (denom_t + eps)
             * 4. Update state for next token
             *
             * @param key Input key [hidden_dim]
             * @param value Input value [hidden_dim]
             * @param weight Time-decay weight [hidden_dim]
             * @param receptance Attention weights [hidden_dim]
             * @param output Output values [hidden_dim]
             * @return True if successful
             */
            bool forward(
                const float *key,
                const float *value,
                const float *weight,
                const float *receptance,
                float *output);

            /**
             * Forward pass for entire sequence
             *
             * Processes sequence token-by-token, maintaining state across tokens.
             * Use this for training or greedy decoding of entire sequences.
             *
             * @param keys Sequence of keys [seq_len * hidden_dim]
             * @param values Sequence of values [seq_len * hidden_dim]
             * @param weights Time-decay weights [seq_len * hidden_dim]
             * @param receptances Receptance values [seq_len * hidden_dim]
             * @param seq_len Number of tokens
             * @param output Output sequence [seq_len * hidden_dim]
             * @return True if successful
             */
            bool forward_sequence(
                const float *keys,
                const float *values,
                const float *weights,
                const float *receptances,
                uint32_t seq_len,
                float *output);

            /**
             * Reset internal state
             *
             * Clears numerator and denominator accumulators.
             * Call before processing new conversation or sequence.
             */
            void reset_state();

            /**
             * Save state for stateful inference
             *
             * Saves numerator and denominator for resuming inference later.
             *
             * @param state_buffer Buffer for state [2 * hidden_dim]
             */
            void save_state(float *state_buffer) const;

            /**
             * Load state for resuming inference
             *
             * Restores numerator and denominator from saved state.
             *
             * @param state_buffer Previously saved state [2 * hidden_dim]
             */
            void load_state(const float *state_buffer);

            /**
             * Get hidden dimension
             */
            uint32_t get_hidden_dim() const { return hidden_dim_; }

            /**
             * Get configuration
             */
            const WKVConfig &get_config() const { return config_; }

            /**
             * Get internal state size
             */
            size_t get_state_size() const { return 2 * hidden_dim_; }

        private:
            /**
             * Compute time decay factor: exp(-exp(weight))
             *
             * The nested exponential ensures:
             * - Numerically stable: never overflows
             * - Output ∈ (0, 1): appropriate for decay
             * - Trainable: weight ∈ ℝ, smooth gradients
             *
             * @param weight Input weight (unbounded)
             * @return Decay factor ∈ (0, 1)
             */
            float compute_time_decay(float weight) const;

            /**
             * Elementwise multiplication: out = a * b
             *
             * @param a First vector [hidden_dim]
             * @param b Second vector [hidden_dim]
             * @param out Output [hidden_dim]
             */
            void elementwise_multiply(const float *a, const float *b, float *out);

            /**
             * Elementwise addition: out = a + b
             *
             * @param a First vector [hidden_dim]
             * @param b Second vector [hidden_dim]
             * @param out Output [hidden_dim]
             */
            void elementwise_add(const float *a, const float *b, float *out);

            /**
             * Elementwise division: out = a / (b + eps)
             *
             * @param a Numerator [hidden_dim]
             * @param b Denominator [hidden_dim]
             * @param out Output [hidden_dim]
             */
            void elementwise_divide(const float *a, const float *b, float *out);

            // ===== Member Variables =====

            uint32_t hidden_dim_;
            WKVConfig config_;

            // State accumulators (maintained across forward passes)
            std::vector<float> numerator_;   // Σ exp(decay) * K * V
            std::vector<float> denominator_; // Σ exp(decay) * K

            // Temporary buffers for intermediate computations
            std::vector<float> buffer_decay_;          // Computed decay factors
            std::vector<float> buffer_decayed_keys_;   // K * decay
            std::vector<float> buffer_decayed_values_; // V * decay
            std::vector<float> buffer_kv_;             // K * V
            std::vector<float> buffer_numerator_;      // Updated numerator
            std::vector<float> buffer_denominator_;    // Updated denominator
            std::vector<float> buffer_output_;         // Final output before weighting

            // Initialization flag
            bool initialized_;
        };

    } // namespace rwkv
} // namespace ryzanstein_llm

#endif // RYZEN_LLM_RWKV_WKV_H
