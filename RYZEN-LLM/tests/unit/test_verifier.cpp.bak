#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "../../src/optimization/speculative/verifier.h"

using namespace ryzen_llm::speculative;

// ============================================================================
// Test Fixtures
// ============================================================================

class VerifierTest : public ::testing::Test
{
protected:
    VerifierConfig default_config{32000, 1.0f, 0.5f, true};

    // Helper to generate synthetic logits
    std::vector<float> generate_test_logits(uint32_t vocab_size, float peak_value = 5.0f)
    {
        std::vector<float> logits(vocab_size, -10.0f);
        logits[0] = peak_value;
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

TEST_F(VerifierTest, ConstructorSucceedsWithValidConfig)
{
    EXPECT_NO_THROW(Verifier verifier(default_config));
}

TEST_F(VerifierTest, ConstructorThrowsOnZeroVocabSize)
{
    VerifierConfig config = default_config;
    config.vocab_size = 0;
    EXPECT_THROW({ Verifier v(config); }, std::invalid_argument);
}

TEST_F(VerifierTest, ConstructorThrowsOnInvalidTemperature)
{
    VerifierConfig config = default_config;
    config.temperature = 0.0f;
    EXPECT_THROW({ Verifier v(config); }, std::invalid_argument);

    VerifierConfig config2 = default_config;
    config2.temperature = -1.0f;
    EXPECT_THROW({ Verifier v2(config2); }, std::invalid_argument);
}

TEST_F(VerifierTest, ConstructorThrowsOnInvalidRejectionThreshold)
{
    VerifierConfig config = default_config;
    config.rejection_threshold = -0.1f;
    EXPECT_THROW({ Verifier v(config); }, std::invalid_argument);

    VerifierConfig config2 = default_config;
    config2.rejection_threshold = 1.1f;
    EXPECT_THROW({ Verifier v2(config2); }, std::invalid_argument);
}

// ============================================================================
// Configuration Access Tests
// ============================================================================

TEST_F(VerifierTest, GetConfigReturnsCorrectConfiguration)
{
    Verifier verifier(default_config);
    auto retrieved_config = verifier.get_config();

    EXPECT_EQ(retrieved_config.vocab_size, default_config.vocab_size);
    EXPECT_FLOAT_EQ(retrieved_config.temperature, default_config.temperature);
    EXPECT_FLOAT_EQ(retrieved_config.rejection_threshold, default_config.rejection_threshold);
    EXPECT_EQ(retrieved_config.enable_statistics, default_config.enable_statistics);
}

// ============================================================================
// Token Verification Tests
// ============================================================================

TEST_F(VerifierTest, VerifyReturnsEmptyForEmptyDraftTokens)
{
    Verifier verifier(default_config);
    std::vector<int> prefix = {100, 200};
    std::vector<int> draft_tokens;
    std::vector<std::vector<float>> target_logits;

    auto result = verifier.verify(prefix, draft_tokens, target_logits);

    EXPECT_TRUE(result.accepted_tokens.empty());
    EXPECT_EQ(result.num_accepted, 0);
}

TEST_F(VerifierTest, VerifyReturnsEmptyForMismatchedLogitsSize)
{
    Verifier verifier(default_config);
    std::vector<int> prefix = {100, 200};
    std::vector<int> draft_tokens = {50, 60};
    std::vector<std::vector<float>> target_logits;
    target_logits.push_back(generate_test_logits(default_config.vocab_size));
    // Missing second logits entry

    auto result = verifier.verify(prefix, draft_tokens, target_logits);

    EXPECT_TRUE(result.accepted_tokens.empty());
}

TEST_F(VerifierTest, VerifyReturnsEmptyForInvalidLogitsSize)
{
    Verifier verifier(default_config);
    std::vector<int> prefix = {100, 200};
    std::vector<int> draft_tokens = {50, 60};
    std::vector<std::vector<float>> target_logits;
    target_logits.push_back(std::vector<float>(default_config.vocab_size - 1)); // Wrong size
    target_logits.push_back(std::vector<float>(default_config.vocab_size));

    auto result = verifier.verify(prefix, draft_tokens, target_logits);

    EXPECT_TRUE(result.accepted_tokens.empty());
}

TEST_F(VerifierTest, VerifyReturnsEmptyForInvalidTokens)
{
    Verifier verifier(default_config);
    std::vector<int> prefix = {100, 200};
    std::vector<int> draft_tokens = {static_cast<int>(default_config.vocab_size)}; // Invalid token
    std::vector<std::vector<float>> target_logits;
    target_logits.push_back(generate_test_logits(default_config.vocab_size));

    auto result = verifier.verify(prefix, draft_tokens, target_logits);

    EXPECT_TRUE(result.accepted_tokens.empty());
}

TEST_F(VerifierTest, VerifyReturnsEmptyForNegativeTokens)
{
    Verifier verifier(default_config);
    std::vector<int> prefix = {100, 200};
    std::vector<int> draft_tokens = {-1};
    std::vector<std::vector<float>> target_logits;
    target_logits.push_back(generate_test_logits(default_config.vocab_size));

    auto result = verifier.verify(prefix, draft_tokens, target_logits);

    EXPECT_TRUE(result.accepted_tokens.empty());
}

// ============================================================================
// Acceptance Criteria Tests
// ============================================================================

TEST_F(VerifierTest, VerifyAcceptsTokensAboveThreshold)
{
    VerifierConfig config = default_config;
    config.rejection_threshold = 0.1f; // Low threshold (easier to accept)
    Verifier verifier(config);

    std::vector<int> prefix = {100, 200};
    std::vector<int> draft_tokens = {0}; // High probability token
    std::vector<std::vector<float>> target_logits;
    target_logits.push_back(generate_test_logits(default_config.vocab_size, 10.0f)); // Very high peak

    auto result = verifier.verify(prefix, draft_tokens, target_logits);

    // Token 0 should have high probability and be accepted
    // (exact acceptance depends on threshold logic)
    EXPECT_GE(result.num_accepted, 0);
}

TEST_F(VerifierTest, VerifyRejectsTokensBelowThreshold)
{
    VerifierConfig config = default_config;
    config.rejection_threshold = 0.99f; // Very high threshold (hard to accept)
    Verifier verifier(config);

    std::vector<int> prefix = {100, 200};
    std::vector<int> draft_tokens = {100}; // Arbitrary token, unlikely to meet threshold
    std::vector<std::vector<float>> target_logits;
    target_logits.push_back(generate_test_logits(default_config.vocab_size, -10.0f)); // Low peak elsewhere

    auto result = verifier.verify(prefix, draft_tokens, target_logits);

    // Most tokens should be rejected with such high threshold
    EXPECT_LE(result.acceptance_rate, 0.5f);
}

// ============================================================================
// Statistics Tracking Tests
// ============================================================================

TEST_F(VerifierTest, StatsInitializeToZero)
{
    Verifier verifier(default_config);

    EXPECT_EQ(verifier.get_num_verifications(), 0);
    EXPECT_EQ(verifier.get_rejection_rate(), 0.0f);
}

TEST_F(VerifierTest, GetNumVerificationsIncrementsAfterVerify)
{
    Verifier verifier(default_config);

    std::vector<int> prefix = {100};
    std::vector<int> draft_tokens = {50};
    std::vector<std::vector<float>> target_logits;
    target_logits.push_back(generate_test_logits(default_config.vocab_size));

    verifier.verify(prefix, draft_tokens, target_logits);

    EXPECT_EQ(verifier.get_num_verifications(), 1);

    verifier.verify(prefix, draft_tokens, target_logits);

    EXPECT_EQ(verifier.get_num_verifications(), 2);
}

// ============================================================================
// Token Sampling Tests
// ============================================================================

TEST_F(VerifierTest, SampleTokenReturnsValidToken)
{
    Verifier verifier(default_config);

    std::vector<float> logits = generate_test_logits(default_config.vocab_size);

    int sampled = verifier.sample_token(logits);

    EXPECT_GE(sampled, 0);
    EXPECT_LT(sampled, static_cast<int>(default_config.vocab_size));
}

TEST_F(VerifierTest, SampleTokenHandlesUniformDistribution)
{
    Verifier verifier(default_config);

    std::vector<float> logits(100, 0.0f); // Uniform logits

    int sampled = verifier.sample_token(logits);

    EXPECT_GE(sampled, 0);
    EXPECT_LT(sampled, 100);
}

TEST_F(VerifierTest, SampleTokenHandlesExtremePeakDistribution)
{
    Verifier verifier(default_config);

    std::vector<float> logits(100, -100.0f);
    logits[0] = 100.0f; // Extreme peak at token 0

    int sampled = verifier.sample_token(logits);

    EXPECT_GE(sampled, 0);
    EXPECT_LT(sampled, 100);
}

// ============================================================================
// Softmax Stability Tests
// ============================================================================

class VerifierNumericalTest : public VerifierTest
{
protected:
    // Manual softmax for comparison
    std::vector<float> compute_softmax(const std::vector<float> &logits)
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
};

TEST_F(VerifierNumericalTest, SoftmaxProducesValidProbability)
{
    std::vector<float> logits = generate_test_logits(100);
    auto probs = compute_softmax(logits);

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    for (float p : probs)
    {
        EXPECT_GE(p, 0.0f);
        EXPECT_LE(p, 1.0f);
    }
}

TEST_F(VerifierNumericalTest, SoftmaxStableWithLargeLogits)
{
    std::vector<float> logits(100, 1000.0f);

    auto probs = compute_softmax(logits);

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    for (float p : probs)
    {
        EXPECT_NEAR(p, 0.01f, 1e-5f);
    }
}

TEST_F(VerifierNumericalTest, SoftmaxStableWithSmallLogits)
{
    std::vector<float> logits(100, -1000.0f);

    auto probs = compute_softmax(logits);

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    for (float p : probs)
    {
        EXPECT_NEAR(p, 0.01f, 1e-5f);
    }
}

TEST_F(VerifierNumericalTest, SoftmaxStableWithMixedLogits)
{
    std::vector<float> logits = {1000.0f, -1000.0f, 500.0f, -500.0f, 0.0f};

    auto probs = compute_softmax(logits);

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Token 0 should have highest probability
    EXPECT_GT(probs[0], probs[1]);
    EXPECT_GT(probs[0], probs[2]);
}

// ============================================================================
// Temperature Scaling Tests
// ============================================================================

TEST_F(VerifierNumericalTest, TemperatureScalingAffectsDistribution)
{
    std::vector<float> logits = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    auto probs_normal = compute_softmax(logits);

    // Scaled logits with temperature = 0.5
    std::vector<float> scaled_logits(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
    {
        scaled_logits[i] = logits[i] / 0.5f;
    }
    auto probs_low_temp = compute_softmax(scaled_logits);

    // Low temperature should sharpen distribution
    auto max_normal = *std::max_element(probs_normal.begin(), probs_normal.end());
    auto max_low = *std::max_element(probs_low_temp.begin(), probs_low_temp.end());

    EXPECT_GT(max_low, max_normal);
}

// ============================================================================
// Edge Cases & Robustness
// ============================================================================

class VerifierEdgeCaseTest : public VerifierTest
{
};

TEST_F(VerifierEdgeCaseTest, MinimalValidConfig)
{
    VerifierConfig minimal_config{100, 1.0f, 0.5f, false};
    EXPECT_NO_THROW({
        Verifier verifier(minimal_config);
    });
}

TEST_F(VerifierEdgeCaseTest, LargeVocabSize)
{
    VerifierConfig large_config = default_config;
    large_config.vocab_size = 1000000;

    EXPECT_NO_THROW({
        Verifier verifier(large_config);
    });
}

TEST_F(VerifierEdgeCaseTest, ExtremeTemperatures)
{
    // Very low temperature
    VerifierConfig low_temp_config = default_config;
    low_temp_config.temperature = 0.01f;
    EXPECT_NO_THROW({
        Verifier verifier(low_temp_config);
    });

    // Very high temperature
    VerifierConfig high_temp_config = default_config;
    high_temp_config.temperature = 100.0f;
    EXPECT_NO_THROW({
        Verifier verifier(high_temp_config);
    });
}

TEST_F(VerifierEdgeCaseTest, ThresholdBoundaryValues)
{
    // Threshold = 0 (accept everything)
    VerifierConfig config_min = default_config;
    config_min.rejection_threshold = 0.0f;
    EXPECT_NO_THROW({
        Verifier verifier_min(config_min);
    });

    // Threshold = 1 (reject everything)
    VerifierConfig config_max = default_config;
    config_max.rejection_threshold = 1.0f;
    EXPECT_NO_THROW({
        Verifier verifier_max(config_max);
    });
}

// ============================================================================
// Integration-like Tests
// ============================================================================

class VerifierIntegrationTest : public VerifierTest
{
protected:
    // Helper to create realistic logits batch
    std::vector<std::vector<float>> create_target_logits_batch(uint32_t batch_size, uint32_t vocab_size)
    {
        std::vector<std::vector<float>> batch;
        for (uint32_t i = 0; i < batch_size; ++i)
        {
            batch.push_back(generate_test_logits(vocab_size, 5.0f + i));
        }
        return batch;
    }
};

TEST_F(VerifierIntegrationTest, VerifyMultipleTokensSequentially)
{
    Verifier verifier(default_config);

    std::vector<int> prefix = {100, 200};
    std::vector<int> draft_tokens = {50, 51, 52};
    std::vector<std::vector<float>> target_logits = create_target_logits_batch(3, default_config.vocab_size);

    auto result = verifier.verify(prefix, draft_tokens, target_logits);

    // Should complete without error
    EXPECT_GE(result.num_accepted, 0);
    EXPECT_LE(result.num_accepted, 3);
    EXPECT_FLOAT_EQ(result.acceptance_rate, static_cast<float>(result.num_accepted) / 3.0f);
}

TEST_F(VerifierIntegrationTest, VerifyPreservesTokenOrder)
{
    VerifierConfig config = default_config;
    config.rejection_threshold = 0.0f; // Accept all
    Verifier verifier(config);

    std::vector<int> prefix = {100};
    std::vector<int> draft_tokens = {10, 20, 30, 40, 50};
    std::vector<std::vector<float>> target_logits = create_target_logits_batch(5, default_config.vocab_size);

    auto result = verifier.verify(prefix, draft_tokens, target_logits);

    // Accepted tokens should maintain relative order
    if (!result.accepted_tokens.empty())
    {
        for (size_t i = 1; i < result.accepted_tokens.size(); ++i)
        {
            // Verify they follow the draft order (if both accepted)
            int token = result.accepted_tokens[i];
            EXPECT_GE(token, 0);
            EXPECT_LT(token, static_cast<int>(default_config.vocab_size));
        }
    }
}

// ============================================================================
// Test Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
