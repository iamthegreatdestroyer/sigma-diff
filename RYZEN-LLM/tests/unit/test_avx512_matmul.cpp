/*
 * Ryzanstein LLM AVX-512 Matmul Unit Tests
 * [REF:PHASE1-006] - Validation for AVX-512 VNNI Optimization
 *
 * Test Coverage:
 * 1. CPU feature detection
 * 2. Numerical correctness vs naive baseline
 * 3. Performance benchmarking
 * 4. Edge cases (small/large matrices, corner cases)
 */

#include <gtest/gtest.h>
#include "../../src/optimization/avx512/matmul.h"
#include "../../src/core/bitnet/kernels/matmul.h"
#include "../../src/core/bitnet/quantize.h"
#include <random>
#include <chrono>
#include <cmath>

using namespace ryzanstein_llm;

class AVX512MatmulTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize random number generator with fixed seed for reproducibility
        gen_ = std::mt19937(42);
        dist_ = std::normal_distribution<float>(0.0f, 1.0f);
    }

    // Generate random ternary weight matrix
    bitnet::TernaryWeight generate_ternary_weights(uint32_t M, uint32_t K)
    {
        std::vector<float> fp32_weights(M * K);
        for (auto &w : fp32_weights)
        {
            w = dist_(gen_);
        }

        bitnet::QuantizationConfig config;
        config.absmax_percentile = 95.0f;
        config.clip_threshold = 3.0f;

        return bitnet::quantize_weights_ternary(fp32_weights.data(), M, K, config);
    }

    // Generate random INT8 quantized activations
    bitnet::QuantizedActivation generate_int8_activations(uint32_t K, uint32_t N)
    {
        std::vector<float> fp32_activations(K * N);
        for (auto &a : fp32_activations)
        {
            a = dist_(gen_);
        }

        bitnet::QuantizationConfig config;
        config.absmax_percentile = 95.0f;

        return bitnet::quantize_activations_int8(fp32_activations.data(), K * N, config);
    }

    // Compute mean squared error between two matrices
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

    std::mt19937 gen_;
    std::normal_distribution<float> dist_;
};

// ============================================================================
// CPU Feature Detection Tests
// ============================================================================

TEST_F(AVX512MatmulTest, CPUFeatureDetection)
{
    avx512::CPUFeatures features;

    // Test feature detection doesn't crash
    EXPECT_NO_THROW({
        bool has_f = features.has_avx512f;
        bool has_vnni = features.has_avx512_vnni;
        bool has_bw = features.has_avx512bw;
        bool supports = features.supports_optimized_kernel();

        // Log features for debugging
        std::cout << features.to_string() << std::endl;
        std::cout << "Supports optimized kernel: " << (supports ? "YES" : "NO") << std::endl;
    });

    // If AVX-512 is available, VNNI and BW should also be checked
    if (features.has_avx512f)
    {
        std::cout << "AVX-512F detected - checking VNNI and BW support" << std::endl;
    }
}

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_F(AVX512MatmulTest, SmallMatrixCorrectness)
{
    // Small test case: 8×8 @ 8×1
    constexpr uint32_t M = 8, N = 1, K = 8;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    std::vector<float> output_naive(M * N, 0.0f);
    std::vector<float> output_optimized(M * N, 0.0f);

    // Compute with naive kernel
    bitnet::naive_ternary_matmul(
        weights,
        activations,
        output_naive.data(),
        M, N, K);

    // Compute with optimized kernel
    avx512::dispatch_ternary_matmul(
        weights,
        activations,
        output_optimized.data(),
        M, N, K);

    // Check MSE < 1e-4 (should be identical for small matrices)
    float mse = compute_mse(output_naive.data(), output_optimized.data(), M * N);
    EXPECT_LT(mse, 1e-4f) << "MSE too high for small matrix: " << mse;

    // Check element-wise similarity
    for (uint32_t i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(output_naive[i], output_optimized[i], 1e-3f)
            << "Element " << i << " differs: naive=" << output_naive[i]
            << " optimized=" << output_optimized[i];
    }
}

TEST_F(AVX512MatmulTest, MediumMatrixCorrectness)
{
    // Medium test case: 64×64 @ 64×1
    constexpr uint32_t M = 64, N = 1, K = 64;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    std::vector<float> output_naive(M * N, 0.0f);
    std::vector<float> output_optimized(M * N, 0.0f);

    bitnet::naive_ternary_matmul(weights, activations, output_naive.data(), M, N, K);
    avx512::dispatch_ternary_matmul(weights, activations, output_optimized.data(), M, N, K);

    float mse = compute_mse(output_naive.data(), output_optimized.data(), M * N);
    EXPECT_LT(mse, 1e-3f) << "MSE too high for medium matrix: " << mse;
}

TEST_F(AVX512MatmulTest, LargeMatrixCorrectness)
{
    // Large test case: 512×512 @ 512×1 (typical hidden size)
    constexpr uint32_t M = 512, N = 1, K = 512;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    std::vector<float> output_naive(M * N, 0.0f);
    std::vector<float> output_optimized(M * N, 0.0f);

    bitnet::naive_ternary_matmul(weights, activations, output_naive.data(), M, N, K);
    avx512::dispatch_ternary_matmul(weights, activations, output_optimized.data(), M, N, K);

    float mse = compute_mse(output_naive.data(), output_optimized.data(), M * N);
    EXPECT_LT(mse, 1e-2f) << "MSE too high for large matrix: " << mse;
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(AVX512MatmulTest, PerformanceBenchmark)
{
    // Benchmark configuration: 4096×4096 @ 4096×1 (BitNet 7B hidden size)
    constexpr uint32_t M = 4096, N = 1, K = 4096;
    constexpr int NUM_ITERATIONS = 100;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    std::vector<float> output_naive(M * N, 0.0f);
    std::vector<float> output_optimized(M * N, 0.0f);

    // Warm up
    bitnet::naive_ternary_matmul(weights, activations, output_naive.data(), M, N, K);
    avx512::dispatch_ternary_matmul(weights, activations, output_optimized.data(), M, N, K);

    // Benchmark naive implementation
    auto start_naive = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        bitnet::naive_ternary_matmul(weights, activations, output_naive.data(), M, N, K);
    }
    auto end_naive = std::chrono::high_resolution_clock::now();
    double time_naive_ms = std::chrono::duration<double, std::milli>(end_naive - start_naive).count();

    // Benchmark optimized implementation
    auto start_opt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        avx512::dispatch_ternary_matmul(weights, activations, output_optimized.data(), M, N, K);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    double time_opt_ms = std::chrono::duration<double, std::milli>(end_opt - start_opt).count();

    // Compute speedup
    double speedup = time_naive_ms / time_opt_ms;

    std::cout << "\n=== Performance Benchmark (4096×4096 @ 4096×1) ===" << std::endl;
    std::cout << "Naive implementation: " << time_naive_ms / NUM_ITERATIONS << " ms/iteration" << std::endl;
    std::cout << "Optimized implementation: " << time_opt_ms / NUM_ITERATIONS << " ms/iteration" << std::endl;
    std::cout << "Speedup: " << speedup << "×" << std::endl;

    // Compute GFLOPS
    uint64_t flops = 2ULL * M * N * K; // 2 ops per multiply-add
    double gflops_naive = (flops * NUM_ITERATIONS) / (time_naive_ms / 1000.0) / 1e9;
    double gflops_opt = (flops * NUM_ITERATIONS) / (time_opt_ms / 1000.0) / 1e9;

    std::cout << "Naive GFLOPS: " << gflops_naive << std::endl;
    std::cout << "Optimized GFLOPS: " << gflops_opt << std::endl;

    // Verify we get at least some speedup (target 8-12×)
    avx512::CPUFeatures features;
    if (features.supports_optimized_kernel())
    {
        EXPECT_GT(speedup, 2.0) << "Expected at least 2× speedup on AVX-512 capable CPU";
        // Ideally we want 8-12×, but this depends on CPU capabilities
    }
    else
    {
        // Fallback to naive, so speedup should be ~1.0
        EXPECT_NEAR(speedup, 1.0, 0.5);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(AVX512MatmulTest, NonDivisibleByVectorWidth)
{
    // Test matrix dimensions not divisible by 16 (AVX-512 width)
    constexpr uint32_t M = 17, N = 1, K = 23;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);

    std::vector<float> output_naive(M * N, 0.0f);
    std::vector<float> output_optimized(M * N, 0.0f);

    bitnet::naive_ternary_matmul(weights, activations, output_naive.data(), M, N, K);
    avx512::dispatch_ternary_matmul(weights, activations, output_optimized.data(), M, N, K);

    float mse = compute_mse(output_naive.data(), output_optimized.data(), M * N);
    EXPECT_LT(mse, 1e-3f) << "MSE too high for non-aligned matrix: " << mse;
}

TEST_F(AVX512MatmulTest, AllZeroTernaryWeights)
{
    // Test case: all weights are zero
    constexpr uint32_t M = 32, N = 1, K = 32;

    bitnet::TernaryWeight weights(M, K);
    std::fill(weights.values.begin(), weights.values.end(), 0);
    std::fill(weights.scales.begin(), weights.scales.end(), 1.0f);

    auto activations = generate_int8_activations(K, N);

    std::vector<float> output_naive(M * N, 0.0f);
    std::vector<float> output_optimized(M * N, 0.0f);

    bitnet::naive_ternary_matmul(weights, activations, output_naive.data(), M, N, K);
    avx512::dispatch_ternary_matmul(weights, activations, output_optimized.data(), M, N, K);

    // All outputs should be zero
    for (uint32_t i = 0; i < M * N; ++i)
    {
        EXPECT_FLOAT_EQ(output_naive[i], 0.0f);
        EXPECT_FLOAT_EQ(output_optimized[i], 0.0f);
    }
}

// ============================================================================
// Statistics Test
// ============================================================================

TEST_F(AVX512MatmulTest, StatisticsTracking)
{
    // Reset statistics
    avx512::g_matmul_stats.reset();

    constexpr uint32_t M = 128, N = 1, K = 128;

    auto weights = generate_ternary_weights(M, K);
    auto activations = generate_int8_activations(K, N);
    std::vector<float> output(M * N, 0.0f);

    // Perform several calls
    for (int i = 0; i < 10; ++i)
    {
        avx512::dispatch_ternary_matmul(weights, activations, output.data(), M, N, K);
    }

    // Check statistics were recorded
    EXPECT_EQ(avx512::g_matmul_stats.total_calls, 10u);
    EXPECT_GT(avx512::g_matmul_stats.total_flops, 0u);
    EXPECT_GT(avx512::g_matmul_stats.total_time_ms, 0.0);
    EXPECT_GT(avx512::g_matmul_stats.get_avg_gflops(), 0.0);

    std::cout << "\n"
              << avx512::g_matmul_stats.to_string() << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
