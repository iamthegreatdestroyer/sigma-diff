/*
 * RWKV Time Mixing and WKV Operator Unit Tests
 * [REF:CC-004d] - Core Components: RWKV Attention-Free Runtime
 *
 * Comprehensive test suite for RWKV time mixing layer and WKV operator.
 * Tests cover:
 * - Time-shift blending correctness
 * - Projection operations (R, W, K, V)
 * - WKV state management and accumulation
 * - Numerical stability
 * - Streaming inference with state preservation
 */

#include <gtest/gtest.h>
#include "../../src/core/rwkv/time_mixing.h"
#include "../../src/core/rwkv/wkv.h"
#include <cmath>
#include <random>
#include <algorithm>

namespace ryzen_llm::rwkv
{

    // ===== Test Fixtures =====

    class RWKVTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            gen_ = std::mt19937(42);
            dist_ = std::uniform_real_distribution<float>(-1.0f, 1.0f);
        }

        void generate_random_vector(std::vector<float> &vec)
        {
            for (auto &v : vec)
            {
                v = dist_(gen_);
            }
        }

        float compute_mse(const float *a, const float *b, size_t size)
        {
            float mse = 0.0f;
            for (size_t i = 0; i < size; ++i)
            {
                float diff = a[i] - b[i];
                mse += diff * diff;
            }
            return mse / size;
        }

        float compute_max_error(const float *a, const float *b, size_t size)
        {
            float max_err = 0.0f;
            for (size_t i = 0; i < size; ++i)
            {
                float err = std::abs(a[i] - b[i]);
                max_err = std::max(max_err, err);
            }
            return max_err;
        }

        std::mt19937 gen_;
        std::uniform_real_distribution<float> dist_;
    };

    // ===== Time Mixing Layer Tests =====

    TEST_F(RWKVTest, TimeMixingInitialization)
    {
        // Test: Layer initializes with correct dimensions and default config
        uint32_t hidden_dim = 64;
        TimeMixingLayer layer(hidden_dim);

        EXPECT_EQ(layer.get_hidden_dim(), hidden_dim);
        EXPECT_EQ(layer.get_layer_id(), 0);

        const auto &config = layer.get_config();
        EXPECT_FLOAT_EQ(config.time_decay_rate, 0.9f);
        EXPECT_FLOAT_EQ(config.first_token_factor, 0.1f);
    }

    TEST_F(RWKVTest, TimeMixingForwardPass)
    {
        // Test: Forward pass produces valid output
        uint32_t hidden_dim = 32;
        TimeMixingLayer layer(hidden_dim);

        std::vector<float> input(hidden_dim);
        std::vector<float> output(hidden_dim);
        generate_random_vector(input);

        EXPECT_TRUE(layer.forward(input.data(), 0, output.data()));

        // Output should be finite
        for (uint32_t i = 0; i < hidden_dim; ++i)
        {
            EXPECT_TRUE(std::isfinite(output[i])) << "Output[" << i << "] is not finite";
        }
    }

    TEST_F(RWKVTest, TimeMixingSequenceProcessing)
    {
        // Test: Process entire sequence with state persistence
        uint32_t hidden_dim = 32;
        uint32_t seq_len = 10;
        TimeMixingLayer layer(hidden_dim);

        std::vector<float> input(seq_len * hidden_dim);
        std::vector<float> output(seq_len * hidden_dim);
        generate_random_vector(input);

        EXPECT_TRUE(layer.forward_sequence(input.data(), seq_len, output.data()));

        // All outputs should be finite
        for (uint32_t i = 0; i < seq_len * hidden_dim; ++i)
        {
            EXPECT_TRUE(std::isfinite(output[i])) << "Output[" << i << "] is not finite";
        }
    }

    TEST_F(RWKVTest, TimeMixingStateManagement)
    {
        // Test: Save and load state correctly
        uint32_t hidden_dim = 32;
        TimeMixingLayer layer(hidden_dim);

        // Process initial token
        std::vector<float> input(hidden_dim);
        std::vector<float> output1(hidden_dim);
        generate_random_vector(input);
        layer.forward(input.data(), 0, output1.data());

        // Save state
        std::vector<float> state(2 * hidden_dim);
        layer.save_state(state.data());

        // Process another token
        generate_random_vector(input);
        std::vector<float> output2(hidden_dim);
        layer.forward(input.data(), 1, output2.data());

        // Reset and load state
        layer.reset_state();
        layer.load_state(state.data());

        // Process same input again - should get similar output
        std::vector<float> output3(hidden_dim);
        layer.forward(input.data(), 1, output3.data());

        float mse = compute_mse(output2.data(), output3.data(), hidden_dim);
        EXPECT_LT(mse, 1e-4f) << "State save/load causes MSE > 1e-4";
    }

    TEST_F(RWKVTest, TimeMixingMultiHeadConfig)
    {
        // Test: Multi-head configuration with per-head decay
        uint32_t hidden_dim = 64;
        TimeMixingConfig config;
        config.num_heads = 8;
        config.per_head_decay = true;

        TimeMixingLayer layer(hidden_dim, 0, config);

        std::vector<float> input(hidden_dim);
        std::vector<float> output(hidden_dim);
        generate_random_vector(input);

        EXPECT_TRUE(layer.forward(input.data(), 0, output.data()));

        for (uint32_t i = 0; i < hidden_dim; ++i)
        {
            EXPECT_TRUE(std::isfinite(output[i]));
        }
    }

    // ===== WKV Operator Tests =====

    TEST_F(RWKVTest, WKVInitialization)
    {
        // Test: WKV operator initializes correctly
        uint32_t hidden_dim = 64;
        WKVOperator wkv(hidden_dim);

        EXPECT_EQ(wkv.get_hidden_dim(), hidden_dim);
        EXPECT_EQ(wkv.get_state_size(), 2 * hidden_dim);
    }

    TEST_F(RWKVTest, WKVForwardPass)
    {
        // Test: WKV forward pass computes output
        uint32_t hidden_dim = 32;
        WKVOperator wkv(hidden_dim);

        std::vector<float> key(hidden_dim);
        std::vector<float> value(hidden_dim);
        std::vector<float> weight(hidden_dim);
        std::vector<float> receptance(hidden_dim);
        std::vector<float> output(hidden_dim);

        generate_random_vector(key);
        generate_random_vector(value);
        for (auto &w : weight)
            w = dist_(gen_) * 2.0f; // Weights unbounded
        for (auto &r : receptance)
            r = std::abs(dist_(gen_)); // Receptance [0, 1]

        EXPECT_TRUE(wkv.forward(key.data(), value.data(), weight.data(),
                                receptance.data(), output.data()));

        // Output should be finite and reasonable magnitude
        for (uint32_t i = 0; i < hidden_dim; ++i)
        {
            EXPECT_TRUE(std::isfinite(output[i])) << "Output[" << i << "] is not finite";
        }
    }

    TEST_F(RWKVTest, WKVStateAccumulation)
    {
        // Test: WKV correctly accumulates state across tokens
        uint32_t hidden_dim = 16;
        WKVOperator wkv(hidden_dim);

        std::vector<float> key(hidden_dim, 1.0f);
        std::vector<float> value(hidden_dim, 1.0f);
        std::vector<float> weight(hidden_dim, 0.5f);
        std::vector<float> receptance(hidden_dim, 0.5f);
        std::vector<float> output(hidden_dim);

        // First token
        wkv.forward(key.data(), value.data(), weight.data(), receptance.data(), output.data());
        std::vector<float> output1 = std::vector<float>(output.begin(), output.end());

        // Second token with same inputs
        wkv.forward(key.data(), value.data(), weight.data(), receptance.data(), output.data());
        std::vector<float> output2 = std::vector<float>(output.begin(), output.end());

        // Outputs should be different due to accumulated state
        float diff = compute_max_error(output1.data(), output2.data(), hidden_dim);
        EXPECT_GT(diff, 1e-6f) << "State accumulation not detected";
    }

    TEST_F(RWKVTest, WKVSequenceProcessing)
    {
        // Test: Process entire sequence with WKV
        uint32_t hidden_dim = 32;
        uint32_t seq_len = 5;
        WKVOperator wkv(hidden_dim);

        std::vector<float> keys(seq_len * hidden_dim);
        std::vector<float> values(seq_len * hidden_dim);
        std::vector<float> weights(seq_len * hidden_dim);
        std::vector<float> receptances(seq_len * hidden_dim);
        std::vector<float> output(seq_len * hidden_dim);

        generate_random_vector(keys);
        generate_random_vector(values);
        for (auto &w : weights)
            w = dist_(gen_) * 2.0f;
        for (auto &r : receptances)
            r = std::abs(dist_(gen_));

        EXPECT_TRUE(wkv.forward_sequence(keys.data(), values.data(), weights.data(),
                                         receptances.data(), seq_len, output.data()));

        for (uint32_t i = 0; i < seq_len * hidden_dim; ++i)
        {
            EXPECT_TRUE(std::isfinite(output[i])) << "Output[" << i << "] is not finite";
        }
    }

    TEST_F(RWKVTest, WKVStateManagement)
    {
        // Test: Save and load WKV state
        uint32_t hidden_dim = 32;
        WKVOperator wkv(hidden_dim);

        std::vector<float> key(hidden_dim, 1.0f);
        std::vector<float> value(hidden_dim, 1.0f);
        std::vector<float> weight(hidden_dim, 0.5f);
        std::vector<float> receptance(hidden_dim, 0.5f);
        std::vector<float> output1(hidden_dim);

        // Process first token and save state
        wkv.forward(key.data(), value.data(), weight.data(), receptance.data(), output1.data());
        std::vector<float> saved_state(2 * hidden_dim);
        wkv.save_state(saved_state.data());

        // Process more tokens
        for (int i = 0; i < 3; ++i)
        {
            wkv.forward(key.data(), value.data(), weight.data(), receptance.data(), output1.data());
        }

        // Reset and load saved state
        wkv.reset_state();
        wkv.load_state(saved_state.data());

        // Process same token - output should match saved state behavior
        std::vector<float> output2(hidden_dim);
        wkv.forward(key.data(), value.data(), weight.data(), receptance.data(), output2.data());

        // After loading and processing same token again, output should be consistent
        std::vector<float> output3(hidden_dim);
        wkv.forward(key.data(), value.data(), weight.data(), receptance.data(), output3.data());

        float mse = compute_mse(output2.data(), output3.data(), hidden_dim);
        EXPECT_LT(mse, 1e-4f) << "State save/load inconsistent";
    }

    TEST_F(RWKVTest, WKVNumericalStability)
    {
        // Test: WKV handles edge cases (very large/small values)
        uint32_t hidden_dim = 16;
        WKVConfig config;
        config.epsilon = 1e-6f;
        WKVOperator wkv(hidden_dim, config);

        std::vector<float> key(hidden_dim, 1e-8f);  // Very small
        std::vector<float> value(hidden_dim, 1e8f); // Very large
        std::vector<float> weight(hidden_dim, 5.0f);
        std::vector<float> receptance(hidden_dim, 0.5f);
        std::vector<float> output(hidden_dim);

        EXPECT_TRUE(wkv.forward(key.data(), value.data(), weight.data(),
                                receptance.data(), output.data()));

        for (uint32_t i = 0; i < hidden_dim; ++i)
        {
            EXPECT_TRUE(std::isfinite(output[i])) << "NaN/Inf in output[" << i << "]";
        }
    }

    TEST_F(RWKVTest, TimeMixingAndWKVIntegration)
    {
        // Test: Time mixing output compatible with WKV input
        uint32_t hidden_dim = 32;
        TimeMixingLayer time_mixing(hidden_dim);
        WKVOperator wkv(hidden_dim);

        std::vector<float> input(hidden_dim);
        std::vector<float> tm_output(hidden_dim);
        std::vector<float> wkv_output(hidden_dim);

        generate_random_vector(input);

        // Time mixing produces receptance, weight, key, value
        EXPECT_TRUE(time_mixing.forward(input.data(), 0, tm_output.data()));

        // Can use outputs for WKV computation
        std::vector<float> key = tm_output;
        std::vector<float> value = tm_output;
        std::vector<float> weight(hidden_dim, 0.5f);
        std::vector<float> receptance(hidden_dim, 0.5f);

        EXPECT_TRUE(wkv.forward(key.data(), value.data(), weight.data(),
                                receptance.data(), wkv_output.data()));

        for (uint32_t i = 0; i < hidden_dim; ++i)
        {
            EXPECT_TRUE(std::isfinite(wkv_output[i]));
        }
    }

    TEST_F(RWKVTest, LargeSequenceProcessing)
    {
        // Test: Handle large sequences without memory issues
        uint32_t hidden_dim = 64;
        uint32_t seq_len = 100;
        TimeMixingLayer layer(hidden_dim);

        std::vector<float> input(seq_len * hidden_dim);
        std::vector<float> output(seq_len * hidden_dim);
        generate_random_vector(input);

        EXPECT_TRUE(layer.forward_sequence(input.data(), seq_len, output.data()));

        // Check statistics
        float sum = 0.0f, min_val = output[0], max_val = output[0];
        for (uint32_t i = 0; i < seq_len * hidden_dim; ++i)
        {
            EXPECT_TRUE(std::isfinite(output[i]));
            sum += output[i];
            min_val = std::min(min_val, output[i]);
            max_val = std::max(max_val, output[i]);
        }

        // Values should be in reasonable range (not exploding/vanishing)
        EXPECT_LT(std::abs(sum), seq_len * hidden_dim * 100.0f) << "Output sum is too large";
    }

} // namespace ryzen_llm::rwkv

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
