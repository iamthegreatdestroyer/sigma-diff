/*
 * RWKV WKV Operator Implementation
 * [REF:CC-004d] - Core Components: RWKV Attention-Free Runtime
 *
 * This file implements the core WKV (Weighted Key-Value) operator that
 * replaces traditional attention in RWKV models with a linear-complexity
 * recurrent mechanism.
 *
 * Key Features:
 * - WKV recurrent state updates
 * - Time-decay mechanisms
 * - Linear complexity in sequence length
 * - Cache-efficient state management
 */

#include "wkv.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace ryzen_llm
{
    namespace rwkv
    {

        // ===== Constructor and Initialization =====

        WKVOperator::WKVOperator(
            uint32_t hidden_dim,
            const WKVConfig &config)
            : hidden_dim_(hidden_dim),
              config_(config),
              initialized_(false)
        {

            // Initialize state accumulators
            numerator_.resize(hidden_dim, 0.0f);
            denominator_.resize(hidden_dim, 0.0f);

            // Initialize temporary buffers
            buffer_decay_.resize(hidden_dim);
            buffer_decayed_keys_.resize(hidden_dim);
            buffer_decayed_values_.resize(hidden_dim);
            buffer_kv_.resize(hidden_dim);
            buffer_numerator_.resize(hidden_dim);
            buffer_denominator_.resize(hidden_dim);
            buffer_output_.resize(hidden_dim);
        }

        void WKVOperator::initialize()
        {
            if (initialized_)
                return;

            std::fill(numerator_.begin(), numerator_.end(), 0.0f);
            std::fill(denominator_.begin(), denominator_.end(), 0.0f);

            initialized_ = true;
        }

        // ===== Helper Operations =====

        float WKVOperator::compute_time_decay(float weight) const
        {
            // Compute exp(-exp(weight)) for numerical stability
            // This ensures:
            // - Output always in (0, 1)
            // - No overflow: exp(weight) clamped
            // - Smooth gradients for training

            // Clamp weight to prevent overflow in nested exponentials
            // exp(x) reaches max float (~88) when x ≈ 88, so exp(exp(x)) overflows
            // We clamp weight to ensure exp(weight) is reasonable
            const float MAX_WEIGHT = 10.0f; // exp(10) ≈ 22000, manageable
            const float MIN_WEIGHT = -10.0f;

            float clamped_weight = std::clamp(weight, MIN_WEIGHT, MAX_WEIGHT);
            float exp_weight = std::exp(clamped_weight);

            // Now compute exp(-exp_weight), careful about range
            if (exp_weight > 100.0f)
            {
                // exp(-exp_weight) ≈ 0 for large exp_weight
                return 0.0f;
            }
            else
            {
                return std::exp(-exp_weight);
            }
        }

        void WKVOperator::elementwise_multiply(const float *a, const float *b, float *out)
        {
            // out[i] = a[i] * b[i]
            for (uint32_t i = 0; i < hidden_dim_; ++i)
            {
                out[i] = a[i] * b[i];
            }
        }

        void WKVOperator::elementwise_add(const float *a, const float *b, float *out)
        {
            // out[i] = a[i] + b[i]
            for (uint32_t i = 0; i < hidden_dim_; ++i)
            {
                out[i] = a[i] + b[i];
            }
        }

        void WKVOperator::elementwise_divide(const float *a, const float *b, float *out)
        {
            // out[i] = a[i] / (b[i] + epsilon)
            for (uint32_t i = 0; i < hidden_dim_; ++i)
            {
                out[i] = a[i] / (b[i] + config_.epsilon);
            }
        }

        // ===== Main Forward Pass =====

        bool WKVOperator::forward(
            const float *key,
            const float *value,
            const float *weight,
            const float *receptance,
            float *output)
        {

            if (!initialized_)
                initialize();

            // Step 1: Compute time decay factors for each element
            // decay[i] = exp(-exp(weight[i]))
            for (uint32_t i = 0; i < hidden_dim_; ++i)
            {
                buffer_decay_[i] = compute_time_decay(weight[i]);
            }

            // Step 2: Decay previous state
            // numerator_new = numerator_old * decay
            // denominator_new = denominator_old * decay
            elementwise_multiply(numerator_.data(), buffer_decay_.data(), buffer_numerator_.data());
            elementwise_multiply(denominator_.data(), buffer_decay_.data(), buffer_denominator_.data());

            // Step 3: Add new key-value pair to state
            // Compute key * value (elementwise)
            elementwise_multiply(key, value, buffer_kv_.data());

            // numerator += key * value
            elementwise_add(buffer_numerator_.data(), buffer_kv_.data(), buffer_numerator_.data());

            // denominator += key
            elementwise_add(buffer_denominator_.data(), key, buffer_denominator_.data());

            // Step 4: Normalize to compute output
            // output = numerator / (denominator + epsilon)
            elementwise_divide(buffer_numerator_.data(), buffer_denominator_.data(), buffer_output_.data());

            // Step 5: Apply receptance weighting
            // output *= receptance (attention-like weighting)
            elementwise_multiply(buffer_output_.data(), receptance, output);

            // Step 6: Update state for next token
            std::memcpy(numerator_.data(), buffer_numerator_.data(), hidden_dim_ * sizeof(float));
            std::memcpy(denominator_.data(), buffer_denominator_.data(), hidden_dim_ * sizeof(float));

            return true;
        }

        bool WKVOperator::forward_sequence(
            const float *keys,
            const float *values,
            const float *weights,
            const float *receptances,
            uint32_t seq_len,
            float *output)
        {

            if (!initialized_)
                initialize();

            // Reset state for new sequence
            reset_state();

            // Process each token sequentially
            for (uint32_t t = 0; t < seq_len; ++t)
            {
                const float *token_key = keys + t * hidden_dim_;
                const float *token_value = values + t * hidden_dim_;
                const float *token_weight = weights + t * hidden_dim_;
                const float *token_receptance = receptances + t * hidden_dim_;
                float *token_output = output + t * hidden_dim_;

                if (!forward(token_key, token_value, token_weight, token_receptance, token_output))
                {
                    return false;
                }
            }

            return true;
        }

        // ===== State Management =====

        void WKVOperator::reset_state()
        {
            std::fill(numerator_.begin(), numerator_.end(), 0.0f);
            std::fill(denominator_.begin(), denominator_.end(), 0.0f);
        }

        void WKVOperator::save_state(float *state_buffer) const
        {
            // Save: [numerator | denominator]
            std::memcpy(state_buffer, numerator_.data(), hidden_dim_ * sizeof(float));
            std::memcpy(state_buffer + hidden_dim_, denominator_.data(), hidden_dim_ * sizeof(float));
        }

        void WKVOperator::load_state(const float *state_buffer)
        {
            // Load: [numerator | denominator]
            std::memcpy(numerator_.data(), state_buffer, hidden_dim_ * sizeof(float));
            std::memcpy(denominator_.data(), state_buffer + hidden_dim_, hidden_dim_ * sizeof(float));
        }

    } // namespace rwkv
} // namespace ryzen_llm
