/**
 * @file test_tmac_basic.cpp
 * @brief Basic unit tests for T-MAC implementation
 *
 * Tests:
 *   1. Pattern canonicalization correctness
 *   2. Lookup table generation
 *   3. Lookup correctness vs naive computation
 *   4. Performance benchmarks
 */

#include "../pattern_generator.h"
#include "../frequency_analyzer.h"
#include "../delta_encoder.h"
#include "../table_builder.h"
#include "../lut_lookup.h"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>

using namespace ryzanstein_llm::tmac;
using namespace std::chrono;

// ============================================================================
// TEST UTILITIES
// ============================================================================

std::vector<int8_t> generate_random_ternary_weights(uint32_t M, uint32_t K)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1, 1);

    std::vector<int8_t> weights(M * K);
    for (auto &w : weights)
    {
        w = dist(gen);
    }
    return weights;
}

int32_t naive_dot_product(const TernaryPattern &pattern, int8_t activation)
{
    int32_t result = 0;
    for (int8_t w : pattern)
    {
        result += w * activation;
    }
    return result;
}

// ============================================================================
// TEST 1: Pattern Canonicalization
// ============================================================================

void test_pattern_canonicalization()
{
    std::cout << "\n[TEST 1] Pattern Canonicalization\n";
    std::cout << "==================================\n";

    PatternGenerator gen;

    // Test 1.1: Zero pattern (self-symmetric)
    TernaryPattern zero;
    zero.fill(0);

    auto canonical_zero = gen.canonicalize(zero);
    assert(gen.is_self_symmetric(zero));
    std::cout << "✓ Zero pattern is self-symmetric\n";

    // Test 1.2: Symmetry property (w ~ -w)
    TernaryPattern pos = {1, -1, 0, 1, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0};
    TernaryPattern neg;
    for (size_t i = 0; i < 16; ++i)
    {
        neg[i] = -pos[i];
    }

    auto canonical_pos = gen.canonicalize(pos);
    auto canonical_neg = gen.canonicalize(neg);

    // Canonical forms should be the same
    assert(canonical_pos.pattern == canonical_neg.pattern);
    // But flip flags should be opposite
    assert(canonical_pos.flip != canonical_neg.flip);

    std::cout << "✓ Symmetry property verified (w ~ -w)\n";

    // Test 1.3: Tie-breaking (equal +1/-1 counts)
    TernaryPattern tie = {1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto canonical_tie = gen.canonicalize(tie);

    // First non-zero should be +1 in canonical form
    assert(canonical_tie.pattern[0] == 1);
    std::cout << "✓ Tie-breaking rule works correctly\n";

    std::cout << "✓ ALL CANONICALIZATION TESTS PASSED\n";
}

// ============================================================================
// TEST 2: Frequency Analysis
// ============================================================================

void test_frequency_analysis()
{
    std::cout << "\n[TEST 2] Frequency Analysis\n";
    std::cout << "============================\n";

    // Generate small test weights
    uint32_t M = 128;
    uint32_t K = 256;
    auto weights = generate_random_ternary_weights(M, K);

    FrequencyAnalyzer analyzer;
    auto frequencies = analyzer.analyze_weights(weights, M, K, 16);

    assert(!frequencies.empty());
    std::cout << "✓ Found " << frequencies.size() << " unique patterns\n";

    // Check that frequencies are sorted
    for (size_t i = 1; i < frequencies.size(); ++i)
    {
        assert(frequencies[i - 1].count >= frequencies[i].count);
    }
    std::cout << "✓ Frequencies sorted correctly (descending)\n";

    // Check probability sum ≈ 1.0
    double prob_sum = 0.0;
    for (const auto &freq : frequencies)
    {
        prob_sum += freq.probability;
    }
    assert(std::abs(prob_sum - 1.0) < 0.001);
    std::cout << "✓ Probability sum = " << prob_sum << " (≈1.0)\n";

    // Test coverage computation
    double coverage_10 = analyzer.compute_coverage(frequencies, 10);
    double coverage_all = analyzer.compute_coverage(frequencies, frequencies.size());

    assert(coverage_10 <= 1.0);
    assert(coverage_all >= 0.99); // Should be ~1.0
    std::cout << "✓ Coverage: top-10 = " << (coverage_10 * 100.0) << "%\n";

    std::cout << "✓ ALL FREQUENCY ANALYSIS TESTS PASSED\n";
}

// ============================================================================
// TEST 3: Delta Encoding
// ============================================================================

void test_delta_encoding()
{
    std::cout << "\n[TEST 3] Delta Encoding\n";
    std::cout << "========================\n";

    PatternGenerator gen;
    DeltaEncoder encoder;

    // Create base and similar target
    TernaryPattern base = {1, -1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1};
    TernaryPattern target = {1, -1, 1, 1, 0, -1, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1};
    // Difference at position 2: 0 → 1 (Hamming distance = 1)

    uint32_t hamming = encoder.hamming_distance(base, target);
    assert(hamming == 1);
    std::cout << "✓ Hamming distance computed correctly: " << hamming << "\n";

    // Canonicalize
    auto base_can = gen.canonicalize(base);
    auto target_can = gen.canonicalize(target);

    // Encode delta
    auto delta = encoder.encode_delta(target_can, base_can);
    assert(delta.size() == 256);
    std::cout << "✓ Delta encoded successfully (256 values)\n";

    // Verify reconstruction for some activation values
    for (int8_t act : {-128, -50, 0, 50, 127})
    {
        int32_t base_result = naive_dot_product(base, act);
        int32_t target_result = naive_dot_product(target, act);

        uint8_t act_idx = static_cast<uint8_t>(act + 128);
        int32_t reconstructed = base_result + delta[act_idx];

        assert(reconstructed == target_result);
    }
    std::cout << "✓ Delta reconstruction verified (correct results)\n";

    std::cout << "✓ ALL DELTA ENCODING TESTS PASSED\n";
}

// ============================================================================
// TEST 4: Lookup Correctness
// ============================================================================

void test_lookup_correctness()
{
    std::cout << "\n[TEST 4] Lookup Correctness\n";
    std::cout << "============================\n";

    // Generate small test weights
    uint32_t M = 64;
    uint32_t K = 256; // Must be multiple of 16
    auto weights = generate_random_ternary_weights(M, K);

    // Build lookup tables
    TableBuilder builder(16);
    builder.set_tier_sizes(100, 1000); // Small tiers for testing
    auto lut = builder.build(weights, M, K);

    // Create lookup engine
    auto lookup = std::make_shared<CompressedLUT>(std::move(lut));
    LUTLookup engine(lookup);

    // Test random patterns
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> weight_dist(-1, 1);
    std::uniform_int_distribution<int> act_dist(-128, 127);

    constexpr int NUM_TESTS = 1000;
    int correct = 0;

    for (int i = 0; i < NUM_TESTS; ++i)
    {
        // Generate random pattern
        TernaryPattern pattern;
        for (auto &w : pattern)
        {
            w = weight_dist(gen);
        }

        int8_t activation = act_dist(gen);

        // Lookup vs naive
        int32_t lut_result = engine.lookup(pattern, activation);
        int32_t naive_result = naive_dot_product(pattern, activation);

        if (lut_result == naive_result)
        {
            correct++;
        }
        else
        {
            std::cerr << "Mismatch: LUT=" << lut_result
                      << ", Naive=" << naive_result << "\n";
        }
    }

    double accuracy = static_cast<double>(correct) / NUM_TESTS;
    std::cout << "✓ Accuracy: " << (accuracy * 100.0) << "% ("
              << correct << "/" << NUM_TESTS << ")\n";

    assert(accuracy == 1.0); // Must be 100% correct

    // Print statistics
    engine.print_stats();

    std::cout << "✓ ALL LOOKUP CORRECTNESS TESTS PASSED\n";
}

// ============================================================================
// TEST 5: Performance Benchmark
// ============================================================================

void test_performance()
{
    std::cout << "\n[TEST 5] Performance Benchmark\n";
    std::cout << "===============================\n";

    // Generate realistic weights
    uint32_t M = 512;
    uint32_t K = 2048;
    auto weights = generate_random_ternary_weights(M, K);

    // Build lookup tables
    TableBuilder builder(16);
    auto lut_struct = builder.build(weights, M, K);
    auto lut = std::make_shared<CompressedLUT>(std::move(lut_struct));
    LUTLookup engine(lut);

    // Generate test patterns
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> weight_dist(-1, 1);
    std::uniform_int_distribution<int> act_dist(-128, 127);

    constexpr int WARMUP = 1000;
    constexpr int BENCHMARK = 100000;

    std::vector<TernaryPattern> test_patterns(BENCHMARK);
    std::vector<int8_t> test_activations(BENCHMARK);

    for (int i = 0; i < BENCHMARK; ++i)
    {
        for (auto &w : test_patterns[i])
        {
            w = weight_dist(gen);
        }
        test_activations[i] = act_dist(gen);
    }

    // Warmup
    for (int i = 0; i < WARMUP; ++i)
    {
        engine.lookup(test_patterns[i % BENCHMARK], test_activations[i % BENCHMARK]);
    }
    engine.reset_stats();

    // Benchmark
    auto start = high_resolution_clock::now();

    for (int i = 0; i < BENCHMARK; ++i)
    {
        volatile int32_t result = engine.lookup(test_patterns[i], test_activations[i]);
        (void)result; // Prevent optimization
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();

    double avg_ns = static_cast<double>(duration) / BENCHMARK;
    double avg_us = avg_ns / 1000.0;

    std::cout << "Results:\n";
    std::cout << "  Total lookups: " << BENCHMARK << "\n";
    std::cout << "  Total time: " << (duration / 1e6) << " ms\n";
    std::cout << "  Average latency: " << avg_ns << " ns/lookup\n";
    std::cout << "  Average latency: " << avg_us << " μs/lookup\n";
    std::cout << "  Target: <50,000 ns (50 μs) → "
              << (avg_ns < 50000 ? "✓ PASS" : "✗ FAIL") << "\n";

    engine.print_stats();

    std::cout << "✓ PERFORMANCE BENCHMARK COMPLETE\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main()
{
    std::cout << "====================================\n";
    std::cout << "T-MAC COMPREHENSIVE TEST SUITE\n";
    std::cout << "====================================\n";

    try
    {
        test_pattern_canonicalization();
        test_frequency_analysis();
        test_delta_encoding();
        test_lookup_correctness();
        test_performance();

        std::cout << "\n====================================\n";
        std::cout << "✓ ALL TESTS PASSED!\n";
        std::cout << "====================================\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n✗ TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
