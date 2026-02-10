#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <random>

// Test focus: Sampling algorithms and their mathematical properties
// These tests validate the core probability distributions and sampling methods

namespace sampling_algorithms_test
{

    // ============================================================================
    // Softmax Implementation for Testing
    // ============================================================================

    class SoftmaxTest : public ::testing::Test
    {
    protected:
        // Reference softmax implementation with max subtraction for stability
        std::vector<float> softmax_stable(const std::vector<float> &logits)
        {
            if (logits.empty())
                return {};

            std::vector<float> result(logits.size());
            float max_logit = *std::max_element(logits.begin(), logits.end());

            // Compute exp(logit - max)
            float sum = 0.0f;
            for (size_t i = 0; i < logits.size(); ++i)
            {
                result[i] = std::exp(logits[i] - max_logit);
                sum += result[i];
            }

            // Normalize
            for (float &val : result)
            {
                val /= sum;
            }

            return result;
        }

        // Alternative softmax for comparison (not stable)
        std::vector<float> softmax_naive(const std::vector<float> &logits)
        {
            if (logits.empty())
                return {};

            std::vector<float> result(logits.size());
            float sum = 0.0f;

            for (size_t i = 0; i < logits.size(); ++i)
            {
                result[i] = std::exp(logits[i]);
                sum += result[i];
            }

            for (float &val : result)
            {
                val /= sum;
            }

            return result;
        }
    };

    // ============================================================================
    // Softmax Correctness Tests
    // ============================================================================

    TEST_F(SoftmaxTest, SoftmaxSumToOne)
    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        auto probs = softmax_stable(logits);

        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        EXPECT_NEAR(sum, 1.0f, 1e-6f);
    }

    TEST_F(SoftmaxTest, SoftmaxAllPositive)
    {
        std::vector<float> logits = {-5.0f, 0.0f, 5.0f};
        auto probs = softmax_stable(logits);

        for (float p : probs)
        {
            EXPECT_GE(p, 0.0f);
            EXPECT_LE(p, 1.0f);
        }
    }

    TEST_F(SoftmaxTest, SoftmaxMonotonicity)
    {
        // If logits are ordered, probabilities should follow same order
        std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        auto probs = softmax_stable(logits);

        for (size_t i = 1; i < probs.size(); ++i)
        {
            EXPECT_LE(probs[i - 1], probs[i]);
        }
    }

    TEST_F(SoftmaxTest, SoftmaxUniformLogits)
    {
        std::vector<float> logits(100, 0.0f); // All equal
        auto probs = softmax_stable(logits);

        for (float p : probs)
        {
            EXPECT_NEAR(p, 0.01f, 1e-6f); // Should be uniform
        }
    }

    // ============================================================================
    // Softmax Numerical Stability Tests
    // ============================================================================

    TEST_F(SoftmaxTest, StabilityWithLargePositiveLogits)
    {
        std::vector<float> logits = {1000.0f, 1001.0f, 1002.0f};
        auto probs = softmax_stable(logits);

        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        EXPECT_NEAR(sum, 1.0f, 1e-5f);

        // Should not contain NaN or Inf
        for (float p : probs)
        {
            EXPECT_TRUE(std::isfinite(p));
        }
    }

    TEST_F(SoftmaxTest, StabilityWithLargeNegativeLogits)
    {
        std::vector<float> logits = {-1000.0f, -999.0f, -998.0f};
        auto probs = softmax_stable(logits);

        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        EXPECT_NEAR(sum, 1.0f, 1e-5f);

        for (float p : probs)
        {
            EXPECT_TRUE(std::isfinite(p));
        }
    }

    TEST_F(SoftmaxTest, StabilityWithMixedLogits)
    {
        std::vector<float> logits = {1000.0f, -1000.0f, 0.0f, 500.0f, -500.0f};
        auto probs = softmax_stable(logits);

        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        EXPECT_NEAR(sum, 1.0f, 1e-5f);

        for (float p : probs)
        {
            EXPECT_TRUE(std::isfinite(p));
        }
    }

    TEST_F(SoftmaxTest, StableVsNaiveComparison)
    {
        // On reasonable logits, both should match
        std::vector<float> logits = {0.5f, 1.5f, 2.5f};

        auto stable = softmax_stable(logits);
        auto naive = softmax_naive(logits);

        for (size_t i = 0; i < stable.size(); ++i)
        {
            EXPECT_NEAR(stable[i], naive[i], 1e-6f);
        }
    }

    TEST_F(SoftmaxTest, StableHandlesLargeLogitsWhereNaiveFails)
    {
        std::vector<float> logits = {700.0f, 701.0f, 702.0f};

        auto stable = softmax_stable(logits);

        // Stable version should work
        float sum = std::accumulate(stable.begin(), stable.end(), 0.0f);
        EXPECT_NEAR(sum, 1.0f, 1e-5f);

        // Naive version might overflow (exp(700) is huge)
        // We don't test it to avoid actual overflow, but that's the point
    }

    // ============================================================================
    // Temperature Scaling Tests
    // ============================================================================

    class TemperatureTest : public SoftmaxTest
    {
    protected:
        std::vector<float> apply_temperature(const std::vector<float> &logits, float temperature)
        {
            std::vector<float> scaled(logits.size());
            for (size_t i = 0; i < logits.size(); ++i)
            {
                scaled[i] = logits[i] / temperature;
            }
            return scaled;
        }
    };

    TEST_F(TemperatureTest, TemperatureSharpensDistribution)
    {
        std::vector<float> logits = {2.0f, 1.0f, 0.0f, -1.0f, -2.0f};

        // Normal
        auto probs_normal = softmax_stable(logits);
        auto max_normal = *std::max_element(probs_normal.begin(), probs_normal.end());

        // Low temperature (sharper)
        auto scaled_logits = apply_temperature(logits, 0.5f);
        auto probs_sharp = softmax_stable(scaled_logits);
        auto max_sharp = *std::max_element(probs_sharp.begin(), probs_sharp.end());

        EXPECT_GT(max_sharp, max_normal);
    }

    TEST_F(TemperatureTest, TemperatureFlattenedDistribution)
    {
        std::vector<float> logits = {2.0f, 1.0f, 0.0f, -1.0f, -2.0f};

        // Normal
        auto probs_normal = softmax_stable(logits);
        auto max_normal = *std::max_element(probs_normal.begin(), probs_normal.end());
        auto min_normal = *std::min_element(probs_normal.begin(), probs_normal.end());
        float variance_normal = max_normal - min_normal;

        // High temperature (flatter)
        auto scaled_logits = apply_temperature(logits, 2.0f);
        auto probs_flat = softmax_stable(scaled_logits);
        auto max_flat = *std::max_element(probs_flat.begin(), probs_flat.end());
        auto min_flat = *std::min_element(probs_flat.begin(), probs_flat.end());
        float variance_flat = max_flat - min_flat;

        EXPECT_LT(variance_flat, variance_normal);
    }

    TEST_F(TemperatureTest, TemperatureOneIsIdentity)
    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};

        auto probs_original = softmax_stable(logits);
        auto scaled_logits = apply_temperature(logits, 1.0f);
        auto probs_scaled = softmax_stable(scaled_logits);

        for (size_t i = 0; i < probs_original.size(); ++i)
        {
            EXPECT_NEAR(probs_original[i], probs_scaled[i], 1e-6f);
        }
    }

    TEST_F(TemperatureTest, TemperatureExtremeValues)
    {
        std::vector<float> logits = {5.0f, 0.0f, -5.0f};

        // Very low temperature
        auto scaled_low = apply_temperature(logits, 0.01f);
        auto probs_low = softmax_stable(scaled_low);

        // Very high temperature
        auto scaled_high = apply_temperature(logits, 100.0f);
        auto probs_high = softmax_stable(scaled_high);

        // Low temp should have sharp peak
        auto max_low = *std::max_element(probs_low.begin(), probs_low.end());
        EXPECT_GT(max_low, 0.9f);

        // High temp should be more uniform
        auto max_high = *std::max_element(probs_high.begin(), probs_high.end());
        auto min_high = *std::min_element(probs_high.begin(), probs_high.end());
        EXPECT_LT(max_high - min_high, 0.3f); // Closer to uniform
    }

    // ============================================================================
    // Top-K Filtering Tests
    // ============================================================================

    class TopKTest : public SoftmaxTest
    {
    protected:
        std::vector<float> apply_top_k(const std::vector<float> &probabilities, uint32_t k)
        {
            if (probabilities.empty() || k == 0)
                return probabilities;

            std::vector<float> result(probabilities.size(), 0.0f);

            // Create indexed pairs
            std::vector<std::pair<float, size_t>> indexed;
            for (size_t i = 0; i < probabilities.size(); ++i)
            {
                indexed.push_back({probabilities[i], i});
            }

            // Sort by probability descending
            std::sort(indexed.rbegin(), indexed.rend());

            // Keep top k
            float sum = 0.0f;
            for (uint32_t i = 0; i < std::min(static_cast<uint32_t>(indexed.size()), k); ++i)
            {
                result[indexed[i].second] = indexed[i].first;
                sum += indexed[i].first;
            }

            // Renormalize
            if (sum > 0.0f)
            {
                for (float &p : result)
                {
                    p /= sum;
                }
            }

            return result;
        }
    };

    TEST_F(TopKTest, TopKPreservesTopTokens)
    {
        std::vector<float> logits(100);
        for (uint32_t i = 0; i < 100; ++i)
        {
            logits[i] = static_cast<float>(i);
        }

        auto probs = softmax_stable(logits);
        auto filtered = apply_top_k(probs, 10);

        // Count non-zero entries
        uint32_t count = 0;
        for (float p : filtered)
        {
            if (p > 1e-6f)
                count++;
        }

        EXPECT_EQ(count, 10);
    }

    TEST_F(TopKTest, TopKRemovesBelowKTokens)
    {
        std::vector<float> logits = {10.0f, 9.0f, 8.0f, 7.0f, 6.0f,
                                     -10.0f, -11.0f, -12.0f, -13.0f, -14.0f};

        auto probs = softmax_stable(logits);
        auto filtered = apply_top_k(probs, 5);

        // The bottom 5 logits should have ~zero probability
        float bottom_sum = 0.0f;
        for (size_t i = 5; i < filtered.size(); ++i)
        {
            bottom_sum += filtered[i];
        }

        EXPECT_LT(bottom_sum, 1e-5f);
    }

    TEST_F(TopKTest, TopKRenormalizes)
    {
        std::vector<float> logits = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

        auto probs = softmax_stable(logits);
        auto filtered = apply_top_k(probs, 3);

        float sum = std::accumulate(filtered.begin(), filtered.end(), 0.0f);
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }

    TEST_F(TopKTest, TopKLargerThanVocab)
    {
        std::vector<float> logits = {3.0f, 2.0f, 1.0f};
        auto probs = softmax_stable(logits);

        auto filtered = apply_top_k(probs, 1000); // k > vocab_size

        // Should be identical to original
        for (size_t i = 0; i < probs.size(); ++i)
        {
            EXPECT_NEAR(filtered[i], probs[i], 1e-6f);
        }
    }

    TEST_F(TopKTest, TopKEqualOne)
    {
        std::vector<float> logits = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        auto probs = softmax_stable(logits);

        auto filtered = apply_top_k(probs, 1);

        // Only one token should have probability
        uint32_t count = 0;
        for (float p : filtered)
        {
            if (p > 1e-6f)
                count++;
        }

        EXPECT_EQ(count, 1);

        // It should be the token with highest probability
        auto max_idx = std::distance(probs.begin(),
                                     std::max_element(probs.begin(), probs.end()));
        EXPECT_GT(filtered[max_idx], 0.999f);
    }

    // ============================================================================
    // Top-P (Nucleus) Filtering Tests
    // ============================================================================

    class TopPTest : public SoftmaxTest
    {
    protected:
        std::vector<float> apply_top_p(const std::vector<float> &probabilities, float p)
        {
            if (probabilities.empty() || p >= 1.0f)
                return probabilities;

            std::vector<float> result(probabilities.size(), 0.0f);

            // Create indexed pairs
            std::vector<std::pair<float, size_t>> indexed;
            for (size_t i = 0; i < probabilities.size(); ++i)
            {
                indexed.push_back({probabilities[i], i});
            }

            // Sort by probability descending
            std::sort(indexed.rbegin(), indexed.rend());

            // Accumulate until exceeding p
            float cumsum = 0.0f;
            float sum = 0.0f;
            const float EPSILON = 1e-7f;

            for (const auto &[prob, idx] : indexed)
            {
                if (cumsum >= p - EPSILON)
                    break;

                result[idx] = prob;
                cumsum += prob;
                sum += prob;
            }

            // Ensure at least one token is selected
            if (sum == 0.0f && !probabilities.empty())
            {
                size_t max_idx = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
                result[max_idx] = probabilities[max_idx];
                sum = probabilities[max_idx];
            }

            // Renormalize
            if (sum > 0.0f)
            {
                for (float &prob : result)
                {
                    prob /= sum;
                }
            }

            return result;
        }
    };

    TEST_F(TopPTest, TopPAccumulatesCorrectly)
    {
        std::vector<float> logits = {10.0f, 5.0f, 1.0f, -5.0f, -10.0f};
        auto probs = softmax_stable(logits);

        auto filtered = apply_top_p(probs, 0.9f);

        // Verify cumulative sum is close to 1.0 after renormalization
        float cumsum = 0.0f;
        for (float p : filtered)
        {
            cumsum += p;
        }

        EXPECT_NEAR(cumsum, 1.0f, 1e-5f); // Should sum to 1.0 after renormalization
    }

    TEST_F(TopPTest, TopPRenormalizes)
    {
        std::vector<float> logits = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        auto probs = softmax_stable(logits);

        auto filtered = apply_top_p(probs, 0.5f);

        float sum = std::accumulate(filtered.begin(), filtered.end(), 0.0f);
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }

    TEST_F(TopPTest, TopPWithP1ReturnsAll)
    {
        std::vector<float> logits = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        auto probs = softmax_stable(logits);

        auto filtered = apply_top_p(probs, 1.0f);

        // Should return all
        float sum = std::accumulate(filtered.begin(), filtered.end(), 0.0f);
        EXPECT_NEAR(sum, 1.0f, 1e-5f);

        for (size_t i = 0; i < probs.size(); ++i)
        {
            EXPECT_NEAR(filtered[i], probs[i], 1e-6f);
        }
    }

    TEST_F(TopPTest, TopPWithSmallPRemovesMostTokens)
    {
        std::vector<float> logits = {10.0f, 5.0f, 1.0f, -5.0f, -10.0f};
        auto probs = softmax_stable(logits);

        auto filtered = apply_top_p(probs, 0.1f); // Very small

        // Most tokens should be removed
        uint32_t kept = 0;
        for (float p : filtered)
        {
            if (p > 1e-6f)
                kept++;
        }

        EXPECT_LT(kept, 3); // Should keep only 1-2 top tokens
    }

    TEST_F(TopPTest, TopPMonotonicityPreservation)
    {
        std::vector<float> logits = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        auto probs = softmax_stable(logits);

        auto filtered = apply_top_p(probs, 0.7f);

        // Collect kept and rejected tokens
        std::vector<float> kept_probs, rejected_probs;
        for (size_t i = 0; i < filtered.size(); ++i)
        {
            if (filtered[i] > 1e-6f)
            {
                kept_probs.push_back(probs[i]);
            }
            else
            {
                rejected_probs.push_back(probs[i]);
            }
        }

        // All kept tokens should have higher probability than all rejected tokens
        if (!kept_probs.empty() && !rejected_probs.empty())
        {
            float min_kept = *std::min_element(kept_probs.begin(), kept_probs.end());
            float max_rejected = *std::max_element(rejected_probs.begin(), rejected_probs.end());
            EXPECT_GE(min_kept, max_rejected);
        }
    }

    // ============================================================================
    // Inverse Transform Sampling Tests
    // ============================================================================

    class InverseTransformTest : public SoftmaxTest
    {
    protected:
        int inverse_transform_sample(const std::vector<float> &probabilities)
        {
            if (probabilities.empty())
                return -1;

            static std::mt19937 gen(42);
            std::uniform_real_distribution<float> dis(0.0f, 1.0f);
            float u = dis(gen);

            float cumsum = 0.0f;
            for (size_t i = 0; i < probabilities.size(); ++i)
            {
                cumsum += probabilities[i];
                if (u <= cumsum)
                {
                    return static_cast<int>(i);
                }
            }

            return static_cast<int>(probabilities.size() - 1);
        }

        // Distribution empirical test: sample many times and check distribution
        std::map<int, int> sample_distribution(const std::vector<float> &probabilities,
                                               uint32_t num_samples = 10000)
        {
            std::map<int, int> counts;
            for (uint32_t i = 0; i < num_samples; ++i)
            {
                int sample = inverse_transform_sample(probabilities);
                counts[sample]++;
            }
            return counts;
        }
    };

    TEST_F(InverseTransformTest, InverseTransformSelectsValidToken)
    {
        std::vector<float> probs = {0.5f, 0.3f, 0.2f};

        int sampled = inverse_transform_sample(probs);

        EXPECT_GE(sampled, 0);
        EXPECT_LT(sampled, 3);
    }

    TEST_F(InverseTransformTest, InverseTransformUniformDistribution)
    {
        std::vector<float> probs = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

        auto counts = sample_distribution(probs, 50000);

        // Each token should be sampled roughly equally
        for (int i = 0; i < 5; ++i)
        {
            float observed_prob = static_cast<float>(counts[i]) / 50000.0f;
            EXPECT_NEAR(observed_prob, 0.2f, 0.02f); // 2% tolerance
        }
    }

    TEST_F(InverseTransformTest, InverseTransformSkewedDistribution)
    {
        // Token 0 has 80% probability
        std::vector<float> probs = {0.8f, 0.15f, 0.05f};

        auto counts = sample_distribution(probs, 50000);

        float observed_prob_0 = static_cast<float>(counts[0]) / 50000.0f;
        float observed_prob_1 = static_cast<float>(counts[1]) / 50000.0f;

        EXPECT_NEAR(observed_prob_0, 0.8f, 0.02f);
        EXPECT_NEAR(observed_prob_1, 0.15f, 0.02f);
        EXPECT_GT(counts[0], counts[1]);
        EXPECT_GT(counts[1], counts[2]);
    }

    TEST_F(InverseTransformTest, InverseTransformCoverageAllTokens)
    {
        std::vector<float> probs(1000);
        std::fill(probs.begin(), probs.end(), 1.0f / 1000.0f);

        auto counts = sample_distribution(probs, 100000);

        // All tokens should be sampled at least once
        for (int i = 0; i < 1000; ++i)
        {
            EXPECT_GT(counts[i], 0);
        }
    }

} // namespace sampling_algorithms_test

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
