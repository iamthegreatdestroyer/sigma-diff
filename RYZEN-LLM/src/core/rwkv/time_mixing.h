/*
 * RWKV Time Mixing Layer
 * [REF:CC-004d] - Core Components: RWKV Attention-Free Runtime
 *
 * This module implements the time-mixing layer from RWKV, which combines
 * information across time steps without quadratic attention complexity.
 *
 * Algorithm Overview:
 * - Time-shift mixing: blends current token with previous token representations
 * - Receptance projection: computes attention-like weights without softmax
 * - Weight, Key, Value projections: linear transformations with learnable parameters
 * - Linear RNN mechanism: recurrent computation replaces transformer attention
 *
 * Performance:
 * - O(N) complexity: linear in sequence length (vs O(N²) for attention)
 * - Cache-friendly: minimal state required for inference
 * - AVX-512 optimizable: vectorizable projections
 *
 * References:
 * - RWKV: "Reinventing RNNs for the Transformer Era" (2023)
 * - https://arxiv.org/abs/2305.13048
 * - https://github.com/BlinkDL/RWKV-LM
 */

#ifndef RYZEN_LLM_RWKV_TIME_MIXING_H
#define RYZEN_LLM_RWKV_TIME_MIXING_H

#include <cstdint>
#include <vector>
#include <memory>
#include <cstring>

namespace ryzanstein_llm
{
    namespace rwkv
    {

        /**
         * Time-shift mixing configuration
         *
         * Controls the balance between using current and previous token information.
         * Time-decay (decay_rate) is critical for preventing gradient saturation.
         */
        struct TimeMixingConfig
        {
            // Blend factor for time-shift: t = decay * prev + (1 - decay) * curr
            // Higher decay → more weight on previous states, better long-range
            float time_decay_rate;

            // First token initialization factor
            // Values: ∈ [0.0, 1.0] - controls how much initial state influences generation
            float first_token_factor;

            // Number of attention heads (for head-wise mixing)
            uint32_t num_heads;

            // Whether to use head-wise time decay (different decay per head)
            bool per_head_decay;

            // AVX-512 optimization flags
            bool use_avx512_projection;
            bool use_avx512_shift;

            // Initialize with default values
            TimeMixingConfig()
                : time_decay_rate(0.9f),
                  first_token_factor(0.1f),
                  num_heads(32),
                  per_head_decay(true),
                  use_avx512_projection(true),
                  use_avx512_shift(true) {}
        };

        /**
         * RWKV Time Mixing Layer
         *
         * Implements the time-mixing mechanism for a single transformer layer.
         * Maintains state for the previous token to enable streaming inference.
         *
         * Layout:
         * - Input: [batch_size, seq_len, hidden_dim]
         * - Output: [batch_size, seq_len, hidden_dim]
         *
         * State:
         * - prev_x: Previous layer input [batch_size, hidden_dim]
         * - prev_xx: Previous receptance output [batch_size, hidden_dim]
         */
        class TimeMixingLayer
        {
        public:
            /**
             * Construct a time mixing layer
             *
             * @param hidden_dim Dimension of hidden state
             * @param layer_id Layer index (0 to num_layers-1)
             * @param config Configuration parameters
             */
            explicit TimeMixingLayer(
                uint32_t hidden_dim,
                uint32_t layer_id = 0,
                const TimeMixingConfig &config = TimeMixingConfig());

            ~TimeMixingLayer() = default;

            /**
             * Initialize layer parameters (weights and biases)
             *
             * Uses Xavier initialization for linear projections.
             * Creates per-head decay parameters if per_head_decay is enabled.
             */
            void initialize();

            /**
             * Forward pass for time mixing layer
             *
             * Computation:
             * 1. Time-shift: blends current input with previous states
             * 2. Receptance: compute R = tanh(input @ W_r)
             * 3. Weight: compute W = log(exp(input @ W_w) + 1)
             * 4. Key/Value: compute K = input @ W_k, V = input @ W_v
             * 5. Output: compute receptance-weighted sum with exponential decay
             *
             * @param input Current token input [hidden_dim]
             * @param seq_position Position in sequence
             * @param output Resulting mixed representation [hidden_dim]
             * @return True if successful
             */
            bool forward(
                const float *input,
                uint32_t seq_position,
                float *output);

            /**
             * Forward pass for entire sequence (training or greedy decoding)
             *
             * @param input Sequence of tokens [seq_len * hidden_dim]
             * @param seq_len Number of tokens in sequence
             * @param output Output sequence [seq_len * hidden_dim]
             * @return True if successful
             */
            bool forward_sequence(
                const float *input,
                uint32_t seq_len,
                float *output);

            /**
             * Time-shift operation: blend current with previous token
             *
             * Computation:
             * - x_shifted = decay_rate * prev_x + (1 - decay_rate) * current_x
             * - xx_shifted = decay_rate * prev_xx + (1 - decay_rate) * current_xx
             *
             * This allows the model to implicitly "attend" to the previous token
             * without explicit attention computation.
             *
             * @param current Current input [hidden_dim]
             * @param output Time-shifted representation [hidden_dim]
             * @return True if successful
             */
            bool time_shift(const float *current, float *output);

            /**
             * Reset internal state (prev_x, prev_xx)
             *
             * Call this before processing a new sequence or conversation.
             */
            void reset_state();

            /**
             * Save state for stateful inference (multi-turn conversations)
             *
             * @param state_buffer Buffer to save state [2 * hidden_dim]
             */
            void save_state(float *state_buffer) const;

            /**
             * Load state for resuming inference
             *
             * @param state_buffer Previously saved state [2 * hidden_dim]
             */
            void load_state(const float *state_buffer);

            /**
             * Get hidden dimension
             */
            uint32_t get_hidden_dim() const { return hidden_dim_; }

            /**
             * Get layer ID
             */
            uint32_t get_layer_id() const { return layer_id_; }

            /**
             * Get configuration
             */
            const TimeMixingConfig &get_config() const { return config_; }

            /**
             * Get internal state for debugging/analysis
             * @param state_size Size of state in elements (should be 2 * hidden_dim)
             * @return Pointer to internal state (valid until next forward pass)
             */
            const float *get_state(size_t &state_size) const
            {
                state_size = 2 * hidden_dim_;
                return prev_x_.data();
            }

        private:
            /**
             * Receptance projection: compute importance weights
             *
             * R = tanh(X @ W_r + b_r)
             *
             * Uses activation function to bound output to [-1, 1] range.
             * Acts as attention-like mechanism without softmax complexity.
             *
             * @param input [hidden_dim]
             * @param output [hidden_dim]
             */
            void compute_receptance(const float *input, float *output);

            /**
             * Weight projection: compute time-decay weights per token
             *
             * W = log(exp(X @ W_w + b_w) + 1)
             *
             * Ensures non-negativity and smooth gradient flow.
             * Each token can have different decay factor.
             *
             * @param input [hidden_dim]
             * @param output [hidden_dim]
             */
            void compute_weight(const float *input, float *output);

            /**
             * Key projection: compute query-like vectors
             *
             * K = X @ W_k + b_k
             *
             * @param input [hidden_dim]
             * @param output [hidden_dim]
             */
            void compute_key(const float *input, float *output);

            /**
             * Value projection: compute value-like vectors
             *
             * V = X @ W_v + b_v
             *
             * @param input [hidden_dim]
             * @param output [hidden_dim]
             */
            void compute_value(const float *input, float *output);

            /**
             * Output projection: map to output dimension
             *
             * O = X @ W_out + b_out
             *
             * @param input [hidden_dim]
             * @param output [hidden_dim]
             */
            void compute_output_projection(const float *input, float *output);

            // ===== Member Variables =====

            // Model dimensions
            uint32_t hidden_dim_;
            uint32_t layer_id_;

            // Configuration
            TimeMixingConfig config_;

            // Learnable parameters (per-head if num_heads > 1)

            // Time decay rates (one per head if per_head_decay)
            std::vector<float> time_decay_rates_;

            // Projection matrices [hidden_dim x hidden_dim]
            std::vector<float> weight_r_;   // Receptance projection
            std::vector<float> weight_w_;   // Weight projection
            std::vector<float> weight_k_;   // Key projection
            std::vector<float> weight_v_;   // Value projection
            std::vector<float> weight_out_; // Output projection

            // Bias terms [hidden_dim]
            std::vector<float> bias_r_;
            std::vector<float> bias_w_;
            std::vector<float> bias_k_;
            std::vector<float> bias_v_;
            std::vector<float> bias_out_;

            // State buffers (maintained across forward passes)
            std::vector<float> prev_x_;  // Previous input [hidden_dim]
            std::vector<float> prev_xx_; // Previous receptance [hidden_dim]

            // Temporary buffers for intermediate computations
            std::vector<float> buffer_r_;     // Receptance computation
            std::vector<float> buffer_w_;     // Weight computation
            std::vector<float> buffer_k_;     // Key computation
            std::vector<float> buffer_v_;     // Value computation
            std::vector<float> buffer_shift_; // Time-shift output

            // Initialization flag
            bool initialized_;
        };

    } // namespace rwkv
} // namespace ryzanstein_llm

#endif // RYZEN_LLM_RWKV_TIME_MIXING_H
