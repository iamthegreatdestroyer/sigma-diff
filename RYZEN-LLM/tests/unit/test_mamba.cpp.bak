// Copyright (c) 2024 RYZEN-LLM Project
// Licensed under MIT License
//
// Unit Tests for Mamba SSM Implementation

#include <gtest/gtest.h>
#include "mamba/ssm.h"
#include "mamba/scan.h"
#include <random>
#include <cmath>

using namespace ryzen_llm::mamba;

class MambaSSMTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gen_.seed(42);
    }

    // Helper: generate random SSM parameters
    SSMParameters generate_test_parameters(
        size_t d_model,
        size_t d_inner,
        size_t d_state,
        size_t d_conv)
    {
        SSMParameters params;
        params.d_model = d_model;
        params.d_inner = d_inner;
        params.d_state = d_state;
        params.d_conv = d_conv;

        // Initialize with small random values
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

        params.W_x.resize(d_model * d_inner);
        params.W_z.resize(d_model * d_inner);
        params.W_B.resize(d_inner * d_state);
        params.W_C.resize(d_inner * d_state);
        params.W_dt.resize(d_inner);
        params.A_log.resize(d_inner * d_state);
        params.D.resize(d_inner);
        params.conv_weight.resize(d_inner * d_conv);
        params.conv_bias.resize(d_inner);
        params.W_out.resize(d_inner * d_model);

        for (auto &val : params.W_x)
            val = dist(gen_);
        for (auto &val : params.W_z)
            val = dist(gen_);
        for (auto &val : params.W_B)
            val = dist(gen_);
        for (auto &val : params.W_C)
            val = dist(gen_);
        for (auto &val : params.W_dt)
            val = dist(gen_);

        // A should be negative for stability (stored in log scale)
        std::uniform_real_distribution<float> a_dist(-5.0f, -1.0f);
        for (auto &val : params.A_log)
            val = a_dist(gen_);

        for (auto &val : params.D)
            val = dist(gen_);
        for (auto &val : params.conv_weight)
            val = dist(gen_);
        for (auto &val : params.conv_bias)
            val = 0.0f;
        for (auto &val : params.W_out)
            val = dist(gen_);

        return params;
    }

    // Helper: generate random input
    std::vector<float> generate_random_input(size_t size)
    {
        std::vector<float> input(size);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &val : input)
        {
            val = dist(gen_);
        }
        return input;
    }

    // Helper: check output is finite and reasonable
    bool check_output_valid(const float *output, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            if (!std::isfinite(output[i]))
                return false;
            if (std::abs(output[i]) > 1000.0f)
                return false; // Sanity check
        }
        return true;
    }

    std::mt19937 gen_;
};

// Test 1: SSMOperator composition
TEST_F(MambaSSMTest, OperatorComposition)
{
    size_t d_inner = 4;
    size_t d_state = 2;

    SSMOperator op1(d_inner, d_state);
    SSMOperator op2(d_inner, d_state);
    SSMOperator result(d_inner, d_state);

    // Initialize with simple values
    for (size_t i = 0; i < d_inner * d_state; ++i)
    {
        op1.A[i] = 0.9f;
        op1.Bx[i] = 1.0f;
        op2.A[i] = 0.8f;
        op2.Bx[i] = 2.0f;
    }

    result.Compose(op1, op2);

    // Check composition: A = 0.8 * 0.9 = 0.72, Bx = 0.8 * 1.0 + 2.0 = 2.8
    for (size_t i = 0; i < d_inner * d_state; ++i)
    {
        EXPECT_NEAR(result.A[i], 0.72f, 1e-5f);
        EXPECT_NEAR(result.Bx[i], 2.8f, 1e-5f);
    }
}

// Test 2: SSM initialization
TEST_F(MambaSSMTest, Initialization)
{
    SSMConfig config;
    config.d_model = 64;
    config.d_state = 8;
    config.d_conv = 4;

    SelectiveSSM ssm(config);
    EXPECT_FALSE(ssm.IsInitialized());

    auto params = generate_test_parameters(64, 128, 8, 4);
    bool success = ssm.Initialize(params);

    EXPECT_TRUE(success);
    EXPECT_TRUE(ssm.IsInitialized());
    EXPECT_EQ(ssm.GetConfig().d_model, 64);
    EXPECT_EQ(ssm.GetConfig().d_state, 8);
}

// Test 3: State creation and reset
TEST_F(MambaSSMTest, StateManagement)
{
    SSMConfig config;
    config.d_model = 32;
    config.d_state = 4;

    SelectiveSSM ssm(config);
    auto params = generate_test_parameters(32, 64, 4, 4);
    ssm.Initialize(params);

    auto state = ssm.CreateState();

    // Check state dimensions
    EXPECT_EQ(state.h.size(), 64 * 4);          // d_inner * d_state
    EXPECT_EQ(state.conv_state.size(), 64 * 3); // d_inner * (d_conv - 1)
    EXPECT_EQ(state.seq_pos, 0);

    // Modify state
    state.h[0] = 1.5f;
    state.seq_pos = 10;

    // Reset and verify
    ssm.ResetState(state);
    EXPECT_EQ(state.h[0], 0.0f);
    EXPECT_EQ(state.seq_pos, 0);
}

// Test 4: Single token generation
TEST_F(MambaSSMTest, SingleTokenGeneration)
{
    SSMConfig config;
    config.d_model = 16;
    config.d_state = 4;
    config.use_avx512 = false; // Test without SIMD

    SelectiveSSM ssm(config);
    auto params = generate_test_parameters(16, 32, 4, 4);
    ssm.Initialize(params);

    auto state = ssm.CreateState();
    auto input = generate_random_input(1 * 16);
    std::vector<float> output(1 * 16);

    // Generate single token
    ssm.ForwardGeneration(input.data(), output.data(), 1, state);

    // Check output validity
    EXPECT_TRUE(check_output_valid(output.data(), output.size()));
    EXPECT_EQ(state.seq_pos, 1);

    // Generate another token (state should accumulate)
    ssm.ForwardGeneration(input.data(), output.data(), 1, state);
    EXPECT_EQ(state.seq_pos, 2);
}

// Test 5: Prefill mode (short sequence)
TEST_F(MambaSSMTest, PrefillShortSequence)
{
    SSMConfig config;
    config.d_model = 32;
    config.d_state = 8;
    config.use_avx512 = false;

    SelectiveSSM ssm(config);
    auto params = generate_test_parameters(32, 64, 8, 4);
    ssm.Initialize(params);

    size_t batch_size = 1;
    size_t seq_len = 16;
    auto input = generate_random_input(batch_size * seq_len * 32);
    std::vector<float> output(batch_size * seq_len * 32);

    // Process sequence
    ssm.ForwardPrefill(input.data(), output.data(), batch_size, seq_len);

    // Check output validity
    EXPECT_TRUE(check_output_valid(output.data(), output.size()));

    // Check statistics
    const auto &stats = ssm.GetStats();
    EXPECT_EQ(stats.total_forward_calls, 1);
    EXPECT_EQ(stats.total_tokens_processed, seq_len);
}

// Test 6: Prefill mode (longer sequence)
TEST_F(MambaSSMTest, PrefillLongSequence)
{
    SSMConfig config;
    config.d_model = 64;
    config.d_state = 16;

    SelectiveSSM ssm(config);
    auto params = generate_test_parameters(64, 128, 16, 4);
    ssm.Initialize(params);

    size_t batch_size = 2;
    size_t seq_len = 128;
    auto input = generate_random_input(batch_size * seq_len * 64);
    std::vector<float> output(batch_size * seq_len * 64);

    // Process sequence
    ssm.ForwardPrefill(input.data(), output.data(), batch_size, seq_len);

    // Check output validity
    EXPECT_TRUE(check_output_valid(output.data(), output.size()));
}

// Test 7: Consistency between prefill and generation
TEST_F(MambaSSMTest, PrefillGenerationConsistency)
{
    SSMConfig config;
    config.d_model = 16;
    config.d_state = 4;
    config.use_avx512 = false;

    SelectiveSSM ssm(config);
    auto params = generate_test_parameters(16, 32, 4, 4);
    ssm.Initialize(params);

    size_t seq_len = 8;
    auto input = generate_random_input(seq_len * 16);

    // Prefill mode
    std::vector<float> output_prefill(seq_len * 16);
    ssm.ForwardPrefill(input.data(), output_prefill.data(), 1, seq_len);

    // Generation mode (process tokens one by one)
    std::vector<float> output_generation(seq_len * 16);
    auto state = ssm.CreateState();
    for (size_t t = 0; t < seq_len; ++t)
    {
        ssm.ForwardGeneration(input.data() + t * 16,
                              output_generation.data() + t * 16,
                              1, state);
    }

    // Outputs should be similar (not exact due to numerical differences)
    float max_diff = 0.0f;
    for (size_t i = 0; i < seq_len * 16; ++i)
    {
        float diff = std::abs(output_prefill[i] - output_generation[i]);
        max_diff = std::max(max_diff, diff);
    }

    // Allow for some numerical difference
    EXPECT_LT(max_diff, 0.5f) << "Max difference: " << max_diff;
}

// Test 8: Parallel scan basic functionality
TEST_F(MambaSSMTest, ParallelScanBasic)
{
    ParallelScan scanner(4);

    size_t d_inner = 8;
    size_t d_state = 4;
    size_t seq_len = 16;

    // Create test operators
    std::vector<float> A_bar(seq_len * d_inner * d_state);
    std::vector<float> B_bar(seq_len * d_inner * d_state);
    std::vector<float> x(seq_len * d_inner);
    std::vector<float> h(seq_len * d_inner * d_state);

    // Initialize with simple values
    std::fill(A_bar.begin(), A_bar.end(), 0.9f);
    std::fill(B_bar.begin(), B_bar.end(), 0.1f);
    std::fill(x.begin(), x.end(), 1.0f);

    // Run scan
    scanner.ScanSSM(A_bar.data(), B_bar.data(), x.data(), h.data(),
                    1, seq_len, d_inner, d_state);

    // Check that output is valid and increasing
    EXPECT_TRUE(check_output_valid(h.data(), h.size()));

    // First state should be smallest (just B*x)
    float first_sum = 0.0f;
    float last_sum = 0.0f;
    for (size_t i = 0; i < d_inner * d_state; ++i)
    {
        first_sum += h[i];
        last_sum += h[(seq_len - 1) * d_inner * d_state + i];
    }

    EXPECT_LT(first_sum, last_sum) << "States should accumulate over time";
}

// Test 9: Statistics tracking
TEST_F(MambaSSMTest, StatisticsTracking)
{
    SSMConfig config;
    config.d_model = 32;
    config.d_state = 8;

    SelectiveSSM ssm(config);
    auto params = generate_test_parameters(32, 64, 8, 4);
    ssm.Initialize(params);

    ssm.ResetStats();
    EXPECT_EQ(ssm.GetStats().total_forward_calls, 0);

    // Run a few forward passes
    auto input = generate_random_input(16 * 32);
    std::vector<float> output(16 * 32);

    for (int i = 0; i < 5; ++i)
    {
        ssm.ForwardPrefill(input.data(), output.data(), 1, 16);
    }

    const auto &stats = ssm.GetStats();
    EXPECT_EQ(stats.total_forward_calls, 5);
    EXPECT_EQ(stats.total_tokens_processed, 5 * 16);
    EXPECT_GT(stats.total_time_ms, 0.0);
    EXPECT_GT(stats.get_throughput_tokens_per_sec(), 0.0);

    // Test to_string doesn't crash
    std::string stats_str = stats.to_string();
    EXPECT_GT(stats_str.length(), 0);
}

// Test 10: Memory usage calculation
TEST_F(MambaSSMTest, MemoryUsage)
{
    SSMConfig config;
    config.d_model = 64;
    config.d_state = 16;

    SelectiveSSM ssm(config);
    auto params = generate_test_parameters(64, 128, 16, 4);
    ssm.Initialize(params);

    size_t memory = ssm.GetMemoryUsage();
    EXPECT_GT(memory, 0);

    // Memory should be reasonable (< 100MB for this small model)
    EXPECT_LT(memory, 100 * 1024 * 1024);
}

// Test 11: Batched processing
TEST_F(MambaSSMTest, BatchedProcessing)
{
    SSMConfig config;
    config.d_model = 32;
    config.d_state = 8;

    SelectiveSSM ssm(config);
    auto params = generate_test_parameters(32, 64, 8, 4);
    ssm.Initialize(params);

    size_t batch_size = 4;
    size_t seq_len = 32;
    auto input = generate_random_input(batch_size * seq_len * 32);
    std::vector<float> output(batch_size * seq_len * 32);

    // Process batch
    ssm.ForwardPrefill(input.data(), output.data(), batch_size, seq_len);

    // Check all batch outputs are valid
    EXPECT_TRUE(check_output_valid(output.data(), output.size()));
}

// Test 12: Edge case - zero input
TEST_F(MambaSSMTest, ZeroInput)
{
    SSMConfig config;
    config.d_model = 16;
    config.d_state = 4;

    SelectiveSSM ssm(config);
    auto params = generate_test_parameters(16, 32, 4, 4);
    ssm.Initialize(params);

    std::vector<float> input(8 * 16, 0.0f);
    std::vector<float> output(8 * 16);

    // Process zeros
    ssm.ForwardPrefill(input.data(), output.data(), 1, 8);

    // Output should be valid (likely small but non-zero due to biases)
    EXPECT_TRUE(check_output_valid(output.data(), output.size()));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
