/*
 * Ryzanstein LLM T-MAC Lookup Table GEMM Unit Tests
 * [REF:PHASE1-007] - Comprehensive Testing for T-MAC Implementation
 *
 * Test Coverage:
 * 1. Table generation correctness and memory usage
 * 2. Lookup-based matmul correctness vs AVX-512 baseline
 * 3. Performance benchmarking (target 2-4× speedup over AVX-512)
 * 4. Hybrid kernel with tail handling
 * 5. Edge cases (small matrices, non-aligned dimensions)
 * 6. Statistics tracking
 */

#include <gtest/gtest.h>
#include "../../src/core/tmac/lut_gemm.h"
#include "../../src/optimization/avx512/matmul.h"
#include "../../src/core/bitnet/quantize.h"
#include <random>
#include <cmath>
#include <chrono>

using namespace ryzanstein_llm;

class TMACTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Fixed seed for reproducibility
        gen_.seed(42);
    }

    /**
     * Generate random ternary weights for testing
     */
    bitnet::TernaryWeight generate_ternary_weights(uint32_t M, uint32_t K, uint32_t group_size = 0)
    {
        bitnet::TernaryWeight weights(M, K, group_size);

        // Generate random FP32 weights
        std::vector<float> fp32_weights(M * K);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < fp32_weights.size(); ++i)
        {
            fp32_weights[i] = dist(gen_);
        }

        // Quantize to ternary
        bitnet::QuantConfig config;
        config.per_group_scaling = (group_size > 0);
        config.weight_group_size = group_size;

        bitnet::quantize_ternary_weights(fp32_weights.data(), M, K, weights, config);

        return weights;
    }

    /**
     * Generate random INT8 activations for testing
     */
    bitnet::QuantizedActivation generate_int8_activations(uint32_t K, uint32_t N)
    {
        bitnet::QuantizedActivation acts(K * N);

        // Generate random FP32 activations
        std::vector<float> fp32_acts(K * N);
        std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
        for (size_t i = 0; i < fp32_acts.size(); ++i)
        {
            fp32_acts[i] = dist(gen_);
        }

        // Quantize to INT8
        bitnet::quantize_activations(fp32_acts.data(), fp32_acts.size(), acts);

        return acts;
    }

    /**
     * Compute mean squared error between two matrices
     */
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

    /**
     * Compute max absolute error
     */
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
};

/**
 * Test 1: Table Generation
 */
TEST_F(TMACTest, TableGeneration)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 64;

    auto weights = generate_ternary_weights(M, K);

    tmac::TMACConfig config;
    config.lookup_width = 8;

    tmac::LookupTableGEMM engine(config);

    // Generate tables
    bool success = engine.GenerateTables(weights);
    EXPECT_TRUE(success);
    EXPECT_TRUE(engine.IsReady());

    // Check memory usage (should be reasonable)
    size_t mem_bytes = engine.GetMemoryUsage();
    size_t expected_bytes = M * ((K + 7) / 8) * 256 * sizeof(float);
    EXPECT_GT(mem_bytes, 0);
    EXPECT_LT(mem_bytes, expected_bytes * 2); // Allow some overhead

    std::cout << "Table Memory Usage: " << mem_bytes / 1024.0 << " KB" << std::endl;
    std::cout << "Table Gen Time: " << engine.GetStats().table_gen_time_ms << " ms" << std::endl;
}

/**
 * Test 2: Small Matrix Correctness
 */
TEST_F(TMACTest, SmallMatrixCorrectness)
{
    constexpr uint32_t M = 16;
    constexpr uint32_t N = 8;
    constexpr uint32_t K = 16;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    // Compute with T-MAC
    tmac::TMACConfig config;
    config.lookup_width = 8;
    tmac::LookupTableGEMM tmac_engine(config);
    tmac_engine.GenerateTables(weights);

    std::vector<float> tmac_output(M * N);
    tmac_engine.Compute(activations, tmac_output.data(), M, N, K);

    // Compute with AVX-512 baseline
    std::vector<float> avx512_output(M * N);
    avx512::optimized_ternary_matmul(weights, activations, avx512_output.data(), M, N, K);

    // Compare results
    float mse = compute_mse(tmac_output.data(), avx512_output.data(), M * N);
    float max_err = compute_max_error(tmac_output.data(), avx512_output.data(), M * N);

    std::cout << "Small Matrix MSE: " << mse << ", Max Error: " << max_err << std::endl;

    // Allow some numerical error due to different computation order
    EXPECT_LT(mse, 1e-2f);
    EXPECT_LT(max_err, 1.0f);
}

/**
 * Test 3: Medium Matrix Correctness
 */
TEST_F(TMACTest, MediumMatrixCorrectness)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 128;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    // Compute with T-MAC
    tmac::TMACConfig config;
    config.lookup_width = 8;
    tmac::LookupTableGEMM tmac_engine(config);
    tmac_engine.GenerateTables(weights);

    std::vector<float> tmac_output(M * N);
    tmac_engine.Compute(activations, tmac_output.data(), M, N, K);

    // Compute with AVX-512 baseline
    std::vector<float> avx512_output(M * N);
    avx512::optimized_ternary_matmul(weights, activations, avx512_output.data(), M, N, K);

    // Compare results
    float mse = compute_mse(tmac_output.data(), avx512_output.data(), M * N);
    float max_err = compute_max_error(tmac_output.data(), avx512_output.data(), M * N);

    std::cout << "Medium Matrix MSE: " << mse << ", Max Error: " << max_err << std::endl;

    EXPECT_LT(mse, 1e-1f);
    EXPECT_LT(max_err, 2.0f);
}

/**
 * Test 4: Large Matrix Correctness
 */
TEST_F(TMACTest, LargeMatrixCorrectness)
{
    constexpr uint32_t M = 512;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 512;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    // Compute with T-MAC
    tmac::TMACConfig config;
    config.lookup_width = 8;
    tmac::LookupTableGEMM tmac_engine(config);
    tmac_engine.GenerateTables(weights);

    std::vector<float> tmac_output(M * N);
    tmac_engine.Compute(activations, tmac_output.data(), M, N, K);

    // Compute with AVX-512 baseline
    std::vector<float> avx512_output(M * N);
    avx512::optimized_ternary_matmul(weights, activations, avx512_output.data(), M, N, K);

    // Compare results
    float mse = compute_mse(tmac_output.data(), avx512_output.data(), M * N);
    float max_err = compute_max_error(tmac_output.data(), avx512_output.data(), M * N);

    std::cout << "Large Matrix MSE: " << mse << ", Max Error: " << max_err << std::endl;

    EXPECT_LT(mse, 1.0f);
    EXPECT_LT(max_err, 5.0f);
}

/**
 * Test 5: Performance Benchmark
 */
TEST_F(TMACTest, PerformanceBenchmark)
{
    constexpr uint32_t M = 4096;
    constexpr uint32_t N = 1; // Single token
    constexpr uint32_t K = 4096;
    constexpr int NUM_ITERATIONS = 100;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    // Setup T-MAC
    tmac::TMACConfig config;
    config.lookup_width = 8;
    tmac::LookupTableGEMM tmac_engine(config);
    tmac_engine.GenerateTables(weights);

    std::vector<float> tmac_output(M * N);
    std::vector<float> avx512_output(M * N);

    // Warmup
    tmac_engine.Compute(activations, tmac_output.data(), M, N, K);
    avx512::optimized_ternary_matmul(weights, activations, avx512_output.data(), M, N, K);

    // Benchmark T-MAC
    auto tmac_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        tmac_engine.Compute(activations, tmac_output.data(), M, N, K);
    }
    auto tmac_end = std::chrono::high_resolution_clock::now();
    double tmac_time = std::chrono::duration<double, std::milli>(tmac_end - tmac_start).count();

    // Benchmark AVX-512
    auto avx512_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        avx512::optimized_ternary_matmul(weights, activations, avx512_output.data(), M, N, K);
    }
    auto avx512_end = std::chrono::high_resolution_clock::now();
    double avx512_time = std::chrono::duration<double, std::milli>(avx512_end - avx512_start).count();

    double tmac_avg = tmac_time / NUM_ITERATIONS;
    double avx512_avg = avx512_time / NUM_ITERATIONS;
    double speedup = avx512_avg / tmac_avg;

    // Compute GFLOPS
    double gflops_tmac = (2.0 * M * N * K) / (tmac_avg / 1000.0) / 1e9;
    double gflops_avx512 = (2.0 * M * N * K) / (avx512_avg / 1000.0) / 1e9;

    std::cout << "\nPerformance Benchmark (" << M << "x" << N << "x" << K << ", " << NUM_ITERATIONS << " iterations):" << std::endl;
    std::cout << "  T-MAC Time: " << tmac_avg << " ms" << std::endl;
    std::cout << "  T-MAC GFLOPS: " << gflops_tmac << std::endl;
    std::cout << "  AVX-512 Time: " << avx512_avg << " ms" << std::endl;
    std::cout << "  AVX-512 GFLOPS: " << gflops_avx512 << std::endl;
    std::cout << "  Speedup: " << speedup << "×" << std::endl;

    // T-MAC should be faster than AVX-512 (target 2-4×)
    // May not achieve on all hardware, so we check if it's at least competitive
    EXPECT_GT(speedup, 0.5); // At least 50% of AVX-512 performance
    std::cout << "  Note: Target speedup is 2-4× on hardware with sufficient cache" << std::endl;
}

/**
 * Test 6: Hybrid Kernel with Tail Handling
 */
TEST_F(TMACTest, HybridKernelTail)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 8;
    constexpr uint32_t K = 71; // Non-aligned (71 = 8*8 + 7 tail)

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    // Compute with hybrid kernel
    tmac::TMACConfig config;
    config.lookup_width = 8;
    tmac::LookupTableGEMM tmac_engine(config);
    tmac_engine.GenerateTables(weights);

    std::vector<float> hybrid_output(M * N);
    tmac_engine.ComputeHybrid(weights, activations, hybrid_output.data(), M, N, K);

    // Compute with AVX-512 baseline
    std::vector<float> avx512_output(M * N);
    avx512::optimized_ternary_matmul(weights, activations, avx512_output.data(), M, N, K);

    // Compare results
    float mse = compute_mse(hybrid_output.data(), avx512_output.data(), M * N);
    float max_err = compute_max_error(hybrid_output.data(), avx512_output.data(), M * N);

    std::cout << "Hybrid Kernel (K=" << K << ") MSE: " << mse << ", Max Error: " << max_err << std::endl;

    EXPECT_LT(mse, 1e-1f);
    EXPECT_LT(max_err, 2.0f);
}

/**
 * Test 7: Non-Divisible Dimensions
 */
TEST_F(TMACTest, NonDivisibleDimensions)
{
    constexpr uint32_t M = 17;
    constexpr uint32_t N = 5;
    constexpr uint32_t K = 23;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    // Compute with hybrid kernel (handles non-aligned automatically)
    tmac::TMACConfig config;
    config.lookup_width = 8;
    tmac::LookupTableGEMM tmac_engine(config);
    tmac_engine.GenerateTables(weights);

    std::vector<float> hybrid_output(M * N);
    tmac_engine.ComputeHybrid(weights, activations, hybrid_output.data(), M, N, K);

    // Compute with AVX-512 baseline
    std::vector<float> avx512_output(M * N);
    avx512::optimized_ternary_matmul(weights, activations, avx512_output.data(), M, N, K);

    // Compare results
    float mse = compute_mse(hybrid_output.data(), avx512_output.data(), M * N);

    std::cout << "Non-Divisible Dims MSE: " << mse << std::endl;

    EXPECT_LT(mse, 1e-1f);
}

/**
 * Test 8: All-Zero Weights Edge Case
 */
TEST_F(TMACTest, AllZeroWeights)
{
    constexpr uint32_t M = 32;
    constexpr uint32_t N = 4;
    constexpr uint32_t K = 32;

    bitnet::TernaryWeight weights(M, K, 0);
    // All weights are zero (default initialization)
    std::fill(weights.values.begin(), weights.values.end(), 0);
    weights.scales[0] = 1.0f;

    auto activations = generate_int8_activations(K, N);

    // Compute with T-MAC
    tmac::TMACConfig config;
    tmac::LookupTableGEMM tmac_engine(config);
    tmac_engine.GenerateTables(weights);

    std::vector<float> output(M * N);
    tmac_engine.Compute(activations, output.data(), M, N, K);

    // All outputs should be close to zero
    for (size_t i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(output[i], 0.0f, 1e-3f);
    }
}

/**
 * Test 9: Statistics Tracking
 */
TEST_F(TMACTest, StatisticsTracking)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 8;
    constexpr uint32_t K = 64;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    tmac::TMACConfig config;
    tmac::LookupTableGEMM tmac_engine(config);
    tmac_engine.GenerateTables(weights);

    // Reset stats
    tmac_engine.GetStats().reset();

    // Run multiple computations
    std::vector<float> output(M * N);
    for (int i = 0; i < 10; ++i)
    {
        tmac_engine.Compute(activations, output.data(), M, N, K);
    }

    // Check statistics
    const auto &stats = tmac_engine.GetStats();
    EXPECT_EQ(stats.total_calls, 10);
    EXPECT_GT(stats.total_lookups, 0);
    EXPECT_GT(stats.total_time_ms, 0);
    EXPECT_GT(stats.table_gen_time_ms, 0);
    EXPECT_GT(stats.get_avg_time_ms(), 0);
    EXPECT_GT(stats.get_avg_throughput_gops(), 0);

    std::cout << "\nStatistics Tracking:\n"
              << stats.to_string() << std::endl;
}

/**
 * Test 10: Different Lookup Widths
 */
TEST_F(TMACTest, DifferentLookupWidths)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 8;
    constexpr uint32_t K = 64;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    std::vector<uint32_t> lookup_widths = {4, 8};

    for (uint32_t lw : lookup_widths)
    {
        tmac::TMACConfig config;
        config.lookup_width = lw;
        tmac::LookupTableGEMM tmac_engine(config);
        tmac_engine.GenerateTables(weights);

        std::vector<float> tmac_output(M * N);
        tmac_engine.Compute(activations, tmac_output.data(), M, N, K);

        // Compute baseline
        std::vector<float> avx512_output(M * N);
        avx512::optimized_ternary_matmul(weights, activations, avx512_output.data(), M, N, K);

        float mse = compute_mse(tmac_output.data(), avx512_output.data(), M * N);

        std::cout << "Lookup Width " << lw << " MSE: " << mse << std::endl;

        EXPECT_LT(mse, 1.0f);
    }
}

/**
 * Main Test Entry Point
 */
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
