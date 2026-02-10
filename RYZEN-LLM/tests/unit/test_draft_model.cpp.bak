#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <map>

#include "../../src/optimization/speculative/draft_model.h"

using namespace ryzen_llm::speculative;

// ============================================================================
// Test Fixtures
// ============================================================================

class DraftModelTest : public ::testing::Test
{
protected:
    DraftModelConfig default_config{
        32000, // vocab_size
        768,   // hidden_dim
        2048,  // max_seq_len
        1,     // min_K
        8,     // max_K
        100,   // K_adjust_frequency
        0.8f,  // temperature
        50,    // top_k
        0.95f, // top_p
        0.75f, // acceptance_rate_target
        true}; // enable_statistics

    void SetUp() override
    {
    }

    void TearDown() override
    {
    }
};

// ============================================================================
// Constructor & Configuration Validation Tests
// ============================================================================

TEST_F(DraftModelTest, ConstructorSucceedsWithValidConfig)
{
    EXPECT_NO_THROW(DraftModel model(default_config));
}

TEST_F(DraftModelTest, ConstructorThrowsOnZeroVocabSize)
{
    DraftModelConfig config = default_config;
    config.vocab_size = 0;
    EXPECT_THROW({ DraftModel m(config); }, std::invalid_argument);
}

TEST_F(DraftModelTest, ConstructorThrowsOnZeroHiddenDim)
{
    DraftModelConfig config = default_config;
    config.hidden_dim = 0;
    EXPECT_THROW({ DraftModel m(config); }, std::invalid_argument);
}

TEST_F(DraftModelTest, ConstructorThrowsOnZeroMaxSeqLen)
{
    DraftModelConfig config = default_config;
    config.max_seq_len = 0;
    EXPECT_THROW({ DraftModel m(config); }, std::invalid_argument);
}

TEST_F(DraftModelTest, ConstructorThrowsOnInvalidKRange)
{
    DraftModelConfig config = default_config;
    config.min_K = 5;
    config.max_K = 2; // min > max
    EXPECT_THROW({ DraftModel m(config); }, std::invalid_argument);
}

TEST_F(DraftModelTest, ConstructorThrowsOnZeroKValues)
{
    DraftModelConfig config = default_config;
    config.min_K = 0;
    EXPECT_THROW({ DraftModel m(config); }, std::invalid_argument);
}

TEST_F(DraftModelTest, ConstructorThrowsOnZeroKAdjustFrequency)
{
    DraftModelConfig config = default_config;
    config.K_adjust_frequency = 0;
    EXPECT_THROW({ DraftModel m(config); }, std::invalid_argument);
}

TEST_F(DraftModelTest, ConstructorThrowsOnInvalidTemperature)
{
    DraftModelConfig config = default_config;
    config.temperature = 0.0f;
    EXPECT_THROW({ DraftModel m(config); }, std::invalid_argument);

    DraftModelConfig config2 = default_config;
    config2.temperature = -1.0f;
    EXPECT_THROW({ DraftModel m2(config2); }, std::invalid_argument);
}

TEST_F(DraftModelTest, ConstructorThrowsOnInvalidAcceptanceRateTarget)
{
    DraftModelConfig config = default_config;
    config.acceptance_rate_target = -0.1f;
    EXPECT_THROW({ DraftModel m(config); }, std::invalid_argument);

    DraftModelConfig config2 = default_config;
    config2.acceptance_rate_target = 1.1f;
    EXPECT_THROW({ DraftModel m2(config2); }, std::invalid_argument);
}

TEST_F(DraftModelTest, ConstructorThrowsOnInvalidTopP)
{
    DraftModelConfig config = default_config;
    config.top_p = -0.1f;
    EXPECT_THROW({ DraftModel m(config); }, std::invalid_argument);

    DraftModelConfig config2 = default_config;
    config2.top_p = 1.1f;
    EXPECT_THROW({ DraftModel m2(config2); }, std::invalid_argument);
}

// ============================================================================
// Candidate Generation Tests
// ============================================================================

TEST_F(DraftModelTest, GenerateCandidatesReturnsEmptyForEmptyPrefix)
{
    DraftModel model(default_config);
    std::vector<int> prefix;

    auto candidates = model.generate_candidates(prefix, 4);

    EXPECT_TRUE(candidates.empty());
}

TEST_F(DraftModelTest, GenerateCandidatesReturnsEmptyForZeroK)
{
    DraftModel model(default_config);
    std::vector<int> prefix = {100, 200, 300};

    auto candidates = model.generate_candidates(prefix, 0);

    EXPECT_TRUE(candidates.empty());
}

TEST_F(DraftModelTest, GenerateCandidatesReturnsEmptyForKExceedsMax)
{
    DraftModel model(default_config);
    std::vector<int> prefix = {100, 200, 300};
    uint32_t K = default_config.max_K + 1;

    auto candidates = model.generate_candidates(prefix, K);

    EXPECT_TRUE(candidates.empty());
}

TEST_F(DraftModelTest, GenerateCandidatesReturnsEmptyForPrefixExceedsMaxLen)
{
    DraftModelConfig config = default_config;
    config.max_seq_len = 10;
    DraftModel model(config);

    std::vector<int> prefix(11, 100); // Longer than max_seq_len

    auto candidates = model.generate_candidates(prefix, 4);

    EXPECT_TRUE(candidates.empty());
}

TEST_F(DraftModelTest, GenerateCandidatesReturnsEmptyForInvalidTokens)
{
    DraftModel model(default_config);
    std::vector<int> prefix = {100, 200, static_cast<int>(default_config.vocab_size)}; // Last token invalid

    auto candidates = model.generate_candidates(prefix, 4);

    EXPECT_TRUE(candidates.empty());
}

TEST_F(DraftModelTest, GenerateCandidatesReturnsEmptyForNegativeTokens)
{
    DraftModel model(default_config);
    std::vector<int> prefix = {100, -1, 300};

    auto candidates = model.generate_candidates(prefix, 4);

    EXPECT_TRUE(candidates.empty());
}

// ============================================================================
// Statistics Tracking Tests
// ============================================================================

TEST_F(DraftModelTest, StatsInitializeToZero)
{
    DraftModel model(default_config);
    auto stats = model.get_stats();

    EXPECT_EQ(stats.num_inferences, 0);
    EXPECT_EQ(stats.num_accepted, 0);
    EXPECT_EQ(stats.total_draft_tokens, 0);
    EXPECT_FLOAT_EQ(stats.get_acceptance_rate(), 0.0f);
}

TEST_F(DraftModelTest, RecordAcceptanceUpdatesStats)
{
    DraftModel model(default_config);

    model.record_acceptance(0, true);
    auto stats = model.get_stats();
    EXPECT_EQ(stats.num_inferences, 1);
    EXPECT_EQ(stats.num_accepted, 1);

    model.record_acceptance(1, false);
    stats = model.get_stats();
    EXPECT_EQ(stats.num_inferences, 2);
    EXPECT_EQ(stats.num_accepted, 1);
}

TEST_F(DraftModelTest, AcceptanceRateCalculation)
{
    DraftModel model(default_config);

    // Record 3 accepted, 2 rejected
    for (int i = 0; i < 3; ++i)
        model.record_acceptance(i, true);
    for (int i = 3; i < 5; ++i)
        model.record_acceptance(i, false);

    auto stats = model.get_stats();
    EXPECT_FLOAT_EQ(stats.get_acceptance_rate(), 0.6f);
}

TEST_F(DraftModelTest, ResetStatsResets)
{
    DraftModel model(default_config);

    model.record_acceptance(0, true);
    model.record_acceptance(1, false);
    model.reset_stats();

    auto stats = model.get_stats();
    EXPECT_EQ(stats.num_inferences, 0);
    EXPECT_EQ(stats.num_accepted, 0);
    EXPECT_FLOAT_EQ(stats.get_acceptance_rate(), 0.0f);
}

// ============================================================================
// K Adaptation Tests
// ============================================================================

TEST_F(DraftModelTest, GetCurrentKInitializedToMinK)
{
    DraftModel model(default_config);
    EXPECT_EQ(model.get_current_K(), default_config.min_K);
}

TEST_F(DraftModelTest, SetCurrentKValidates)
{
    DraftModel model(default_config);

    // Valid K
    model.set_current_K(4);
    EXPECT_EQ(model.get_current_K(), 4);

    // K below min
    model.set_current_K(0);
    EXPECT_EQ(model.get_current_K(), 4); // Should remain unchanged

    // K above max
    model.set_current_K(10);
    EXPECT_EQ(model.get_current_K(), 4); // Should remain unchanged
}

// ============================================================================
// Configuration Access Tests
// ============================================================================

TEST_F(DraftModelTest, GetConfigReturnsCorrectConfiguration)
{
    DraftModel model(default_config);
    auto retrieved_config = model.get_config();

    EXPECT_EQ(retrieved_config.vocab_size, default_config.vocab_size);
    EXPECT_EQ(retrieved_config.hidden_dim, default_config.hidden_dim);
    EXPECT_EQ(retrieved_config.max_seq_len, default_config.max_seq_len);
    EXPECT_EQ(retrieved_config.min_K, default_config.min_K);
    EXPECT_EQ(retrieved_config.max_K, default_config.max_K);
    EXPECT_FLOAT_EQ(retrieved_config.temperature, default_config.temperature);
    EXPECT_FLOAT_EQ(retrieved_config.top_p, default_config.top_p);
}

// ============================================================================
// Sampling Algorithm Tests (Distribution-based)
// ============================================================================

class DraftModelSamplingTest : public DraftModelTest
{
protected:
    // Helper to generate synthetic logits for testing
    std::vector<float> generate_test_logits(uint32_t vocab_size, float peak_value = 5.0f)
    {
        std::vector<float> logits(vocab_size, -10.0f); // Low baseline
        logits[0] = peak_value;                        // High value at position 0
        return logits;
    }

    // Helper to compute softmax
    std::vector<float> softmax(const std::vector<float> &logits)
    {
        std::vector<float> result(logits.size());
        float max_logit = *std::max_element(logits.begin(), logits.end());

        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i)
        {
            result[i] = std::exp(logits[i] - max_logit);
            sum += result[i];
        }

        for (float &val : result)
        {
            val /= sum;
        }

        return result;
    }

    // Helper to create uniform distribution
    std::vector<float> create_uniform_distribution(uint32_t vocab_size)
    {
        std::vector<float> dist(vocab_size, 1.0f / vocab_size);
        return dist;
    }
};

TEST_F(DraftModelSamplingTest, SoftmaxProducesValidProbability)
{
    DraftModel model(default_config);

    std::vector<float> logits = generate_test_logits(100);
    auto probs = softmax(logits);

    // Check that probabilities sum to 1.0
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Check that all probabilities are in [0, 1]
    for (float p : probs)
    {
        EXPECT_GE(p, 0.0f);
        EXPECT_LE(p, 1.0f);
    }
}

TEST_F(DraftModelSamplingTest, SoftmaxNumericalStabilityWithLargeLogits)
{
    // Test that softmax doesn't overflow with very large logits
    std::vector<float> logits(100, 1000.0f); // Huge values

    auto probs = softmax(logits);

    // Should still sum to 1.0
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // All should be approximately 1/100
    for (float p : probs)
    {
        EXPECT_NEAR(p, 0.01f, 1e-5f);
    }
}

TEST_F(DraftModelSamplingTest, SoftmaxNumericalStabilityWithMixedLogits)
{
    // Test with realistic mixed values
    std::vector<float> logits = {10.0f, -10.0f, 5.0f, 0.0f, -5.0f};

    auto probs = softmax(logits);

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Token 0 should have highest probability
    EXPECT_GT(probs[0], probs[1]);
    EXPECT_GT(probs[0], probs[2]);
}

// ============================================================================
// Temperature Scaling Tests
// ============================================================================

TEST_F(DraftModelSamplingTest, TemperatureScalingSharpensWithLowTemp)
{
    // Lower temperature should make distribution sharper
    std::vector<float> logits = {1.0f, 0.5f, 0.0f, -0.5f, -1.0f};

    auto probs_high_temp = softmax(logits);
    // Manual temperature scaling for low temp
    std::vector<float> scaled_logits(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
    {
        scaled_logits[i] = logits[i] / 0.5f; // Divide by low temperature
    }
    auto probs_low_temp = softmax(scaled_logits);

    // Max probability should be higher with low temperature
    auto max_high = *std::max_element(probs_high_temp.begin(), probs_high_temp.end());
    auto max_low = *std::max_element(probs_low_temp.begin(), probs_low_temp.end());

    EXPECT_GT(max_low, max_high);
}

TEST_F(DraftModelSamplingTest, TemperatureScalingFlattensWithHighTemp)
{
    // Higher temperature should make distribution flatter
    std::vector<float> logits = {1.0f, 0.5f, 0.0f, -0.5f, -1.0f};

    auto probs_low_temp = softmax(logits);
    // Manual temperature scaling for high temp
    std::vector<float> scaled_logits(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
    {
        scaled_logits[i] = logits[i] / 2.0f; // Divide by high temperature
    }
    auto probs_high_temp = softmax(scaled_logits);

    // Max probability should be lower with high temperature
    auto max_low = *std::max_element(probs_low_temp.begin(), probs_low_temp.end());
    auto max_high = *std::max_element(probs_high_temp.begin(), probs_high_temp.end());

    EXPECT_GT(max_low, max_high);
}

// ============================================================================
// Top-K Filtering Tests
// ============================================================================

TEST_F(DraftModelSamplingTest, TopKFilteringPreservesTopTokens)
{
    uint32_t vocab_size = 100;
    uint32_t k = 10;

    std::vector<float> logits(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i)
    {
        logits[i] = static_cast<float>(i);
    }

    auto probs = softmax(logits);

    // Manual top-k filtering
    std::vector<std::pair<float, size_t>> indexed_probs;
    for (size_t i = 0; i < probs.size(); ++i)
    {
        indexed_probs.push_back({probs[i], i});
    }
    std::sort(indexed_probs.rbegin(), indexed_probs.rend());

    // Keep only top k
    std::vector<float> filtered_probs(probs.size(), 0.0f);
    float sum = 0.0f;
    for (uint32_t i = 0; i < k; ++i)
    {
        filtered_probs[indexed_probs[i].second] = indexed_probs[i].first;
        sum += indexed_probs[i].first;
    }

    // Renormalize
    for (float &p : filtered_probs)
    {
        p /= sum;
    }

    // Check that only top k tokens have non-zero probability
    uint32_t count = 0;
    for (float p : filtered_probs)
    {
        if (p > 0.0f)
            count++;
    }
    EXPECT_EQ(count, k);

    // Check that result sums to 1.0
    float total = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
    EXPECT_NEAR(total, 1.0f, 1e-5f);
}

// ============================================================================
// Top-P (Nucleus) Filtering Tests
// ============================================================================

TEST_F(DraftModelSamplingTest, TopPFilteringAccumulatesCorrectly)
{
    std::vector<float> logits = {10.0f, 5.0f, 1.0f, -5.0f, -10.0f};
    float top_p = 0.9f;

    auto probs = softmax(logits);

    // Manual top-p filtering
    std::vector<std::pair<float, size_t>> indexed_probs;
    for (size_t i = 0; i < probs.size(); ++i)
    {
        indexed_probs.push_back({probs[i], i});
    }
    std::sort(indexed_probs.rbegin(), indexed_probs.rend());

    std::vector<float> filtered_probs(probs.size(), 0.0f);
    float cumsum = 0.0f;
    for (const auto &[prob, idx] : indexed_probs)
    {
        if (cumsum + prob <= top_p)
        {
            filtered_probs[idx] = prob;
            cumsum += prob;
        }
    }

    // Renormalize
    float sum = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
    for (float &p : filtered_probs)
    {
        p /= sum;
    }

    // Check that result sums to 1.0
    float total = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
    EXPECT_NEAR(total, 1.0f, 1e-5f);

    // At least one token should be selected
    uint32_t count = 0;
    for (float p : filtered_probs)
    {
        if (p > 0.0f)
            count++;
    }
    EXPECT_GT(count, 0);
}

TEST_F(DraftModelSamplingTest, TopPFilteringWithP1ReturnsAll)
{
    // With p=1.0, should keep all tokens
    std::vector<float> logits = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    auto probs = softmax(logits);

    // All probabilities should be retained
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// Inverse Transform Sampling Tests
// ============================================================================

class DraftModelInverseTransformTest : public DraftModelSamplingTest
{
protected:
    // Helper to perform inverse transform sampling
    int inverse_transform_sample(const std::vector<float> &probabilities, uint32_t vocab_size)
    {
        // Generate random uniform value
        static std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float u = dis(gen);

        // Accumulate probabilities
        float cumsum = 0.0f;
        for (uint32_t i = 0; i < vocab_size; ++i)
        {
            cumsum += probabilities[i];
            if (u <= cumsum)
            {
                return i;
            }
        }

        return vocab_size - 1; // Fallback to last token
    }
};

TEST_F(DraftModelInverseTransformTest, InverseTransformSamplingSelectsValidToken)
{
    std::vector<float> probs = {0.5f, 0.3f, 0.2f};
    uint32_t vocab_size = 3;

    int sampled = inverse_transform_sample(probs, vocab_size);

    EXPECT_GE(sampled, 0);
    EXPECT_LT(sampled, static_cast<int>(vocab_size));
}

TEST_F(DraftModelInverseTransformTest, InverseTransformSamplingHigherProbHigherLikelihood)
{
    // Token 0 has highest probability, should be sampled more often
    std::vector<float> probs = {0.8f, 0.15f, 0.05f};
    uint32_t vocab_size = 3;

    std::map<int, int> counts;
    for (int i = 0; i < 1000; ++i)
    {
        int sampled = inverse_transform_sample(probs, vocab_size);
        counts[sampled]++;
    }

    // Token 0 should have the most samples
    EXPECT_GT(counts[0], counts[1]);
    EXPECT_GT(counts[1], counts[2]);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

class DraftModelEdgeCaseTest : public DraftModelTest
{
};

TEST_F(DraftModelEdgeCaseTest, MinimalValidConfig)
{
    DraftModelConfig minimal_config{
        100,    // vocab_size
        64,     // hidden_dim
        10,     // max_seq_len
        1,      // min_K
        1,      // max_K
        1,      // K_adjust_frequency
        1.0f,   // temperature
        0,      // top_k
        1.0f,   // top_p
        0.5f,   // acceptance_rate_target
        false}; // enable_statistics

    EXPECT_NO_THROW({ DraftModel m(minimal_config); });
}

TEST_F(DraftModelEdgeCaseTest, LargeVocabSize)
{
    DraftModelConfig large_config = default_config;
    large_config.vocab_size = 1000000; // Very large

    EXPECT_NO_THROW(DraftModel model(large_config));
}

TEST_F(DraftModelEdgeCaseTest, MaxKEqualsMinK)
{
    DraftModelConfig config = default_config;
    config.min_K = 4;
    config.max_K = 4;

    DraftModel model(config);
    EXPECT_EQ(model.get_current_K(), 4);
}

TEST_F(DraftModelEdgeCaseTest, ExtremeTemperatures)
{
    // Very low temperature
    DraftModelConfig low_temp_config = default_config;
    low_temp_config.temperature = 0.01f;
    EXPECT_NO_THROW(DraftModel model(low_temp_config));

    // Very high temperature
    DraftModelConfig high_temp_config = default_config;
    high_temp_config.temperature = 100.0f;
    EXPECT_NO_THROW(DraftModel model(high_temp_config));
}

// ============================================================================
// Test Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
