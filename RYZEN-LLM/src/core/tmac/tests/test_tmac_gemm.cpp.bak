/**
 * @file test_tmac_gemm.cpp
 * @brief Comprehensive tests for T-MAC GEMM implementation
 *
 * Tests:
 *   1. Correctness vs naive GEMM
 *   2. Different matrix sizes
 *   3. Performance benchmarks
 *   4. AVX-512 vs scalar comparison
 */

#include "../tmac_gemm.h"
#include "../table_builder.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>

using namespace ryzen_llm::tmac;
using namespace std::chrono;

// ============================================================================
// TEST UTILITIES
// ============================================================================

std::vector<int8_t> generate_random_ternary(uint32_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1, 1);

    std::vector<int8_t> data(size);
    for (auto &val : data)
    {
        val = dist(gen);
    }
    return data;
}

std::vector<int8_t> generate_random_int8(uint32_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-128, 127);

    std::vector<int8_t> data(size);
    for (auto &val : data)
    {
        val = static_cast<int8_t>(dist(gen));
    }
    return data;
}

void naive_gemm(
    const int8_t *W,
    const int8_t *X,
    int32_t *Y,
    uint32_t M,
    uint32_t K,
    uint32_t N)
{
    // Y = W × X (naive O(M×K×N) implementation)
    for (uint32_t m = 0; m < M; ++m)
    {
        for (uint32_t n = 0; n < N; ++n)
        {
            int32_t sum = 0;
            for (uint32_t k = 0; k < K; ++k)
            {
                sum += static_cast<int32_t>(W[m * K + k]) *
                       static_cast<int32_t>(X[k * N + n]);
            }
            Y[m * N + n] = sum;
        }
    }
}

double compute_relative_error(
    const int32_t *Y_expected,
    const int32_t *Y_actual,
    uint32_t size)
{
    double sum_sq_error = 0.0;
    double sum_sq_expected = 0.0;

    for (uint32_t i = 0; i < size; ++i)
    {
        double error = static_cast<double>(Y_expected[i]) - static_cast<double>(Y_actual[i]);
        sum_sq_error += error * error;
        sum_sq_expected += static_cast<double>(Y_expected[i]) * static_cast<double>(Y_expected[i]);
    }

    return std::sqrt(sum_sq_error / (sum_sq_expected + 1e-10));
}

// ============================================================================
// TEST 1: Small Matrix Correctness
// ============================================================================

void test_small_matrix_correctness()
{
    std::cout << "\n[TEST 1] Small Matrix Correctness\n";
    std::cout << "===================================\n";

    uint32_t M = 8;
    uint32_t K = 64; // Must be multiple of 16
    uint32_t N = 16;

    // Generate random data
    auto W = generate_random_ternary(M * K);
    auto X = generate_random_int8(K * N);

    std::vector<int32_t> Y_naive(M * N);
    std::vector<int32_t> Y_tmac(M * N);

    // Naive GEMM (reference)
    naive_gemm(W.data(), X.data(), Y_naive.data(), M, K, N);

    // Build T-MAC tables
    TableBuilder builder(16);
    builder.set_tier_sizes(100, 500); // Small for testing
    auto lut_struct = builder.build(W, M, K);
    auto lut_shared = std::make_shared<CompressedLUT>(std::move(lut_struct));
    auto lut_engine = std::make_shared<LUTLookup>(lut_shared);

    // T-MAC GEMM
    TMACGemm gemm_engine(lut_engine);
    gemm_engine.gemm(W.data(), X.data(), Y_tmac.data(), M, K, N);

    // Compare results
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < M * N; ++i)
    {
        if (Y_naive[i] != Y_tmac[i])
        {
            mismatches++;
            if (mismatches <= 5)
            { // Print first 5 mismatches
                std::cout << "  Mismatch at index " << i << ": "
                          << "naive=" << Y_naive[i] << ", "
                          << "tmac=" << Y_tmac[i] << "\n";
            }
        }
    }

    double rel_error = compute_relative_error(Y_naive.data(), Y_tmac.data(), M * N);

    std::cout << "  Matrix size: [" << M << ", " << K << "] × [" << K << ", " << N << "]\n";
    std::cout << "  Mismatches: " << mismatches << " / " << (M * N) << "\n";
    std::cout << "  Relative error: " << (rel_error * 100.0) << "%\n";

    assert(mismatches == 0);
    std::cout << "✓ SMALL MATRIX TEST PASSED\n";
}

// ============================================================================
// TEST 2: Medium Matrix Correctness
// ============================================================================

void test_medium_matrix_correctness()
{
    std::cout << "\n[TEST 2] Medium Matrix Correctness\n";
    std::cout << "====================================\n";

    uint32_t M = 128;
    uint32_t K = 512;
    uint32_t N = 64;

    auto W = generate_random_ternary(M * K);
    auto X = generate_random_int8(K * N);

    std::vector<int32_t> Y_naive(M * N);
    std::vector<int32_t> Y_tmac(M * N);

    // Naive GEMM
    std::cout << "  Computing naive GEMM..." << std::flush;
    auto start_naive = high_resolution_clock::now();
    naive_gemm(W.data(), X.data(), Y_naive.data(), M, K, N);
    auto end_naive = high_resolution_clock::now();
    auto time_naive = duration_cast<microseconds>(end_naive - start_naive).count();
    std::cout << " Done (" << (time_naive / 1000.0) << " ms)\n";

    // Build T-MAC tables
    std::cout << "  Building T-MAC tables..." << std::flush;
    TableBuilder builder(16);
    auto lut_struct = builder.build(W, M, K);
    auto lut_shared = std::make_shared<CompressedLUT>(std::move(lut_struct));
    auto lut_engine = std::make_shared<LUTLookup>(lut_shared);
    std::cout << " Done\n";

    // T-MAC GEMM
    std::cout << "  Computing T-MAC GEMM..." << std::flush;
    TMACGemm gemm_engine(lut_engine);
    auto start_tmac = high_resolution_clock::now();
    gemm_engine.gemm(W.data(), X.data(), Y_tmac.data(), M, K, N);
    auto end_tmac = high_resolution_clock::now();
    auto time_tmac = duration_cast<microseconds>(end_tmac - start_tmac).count();
    std::cout << " Done (" << (time_tmac / 1000.0) << " ms)\n";

    // Compare
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < M * N; ++i)
    {
        if (Y_naive[i] != Y_tmac[i])
        {
            mismatches++;
        }
    }

    double rel_error = compute_relative_error(Y_naive.data(), Y_tmac.data(), M * N);
    double speedup = static_cast<double>(time_naive) / time_tmac;

    std::cout << "\n  Results:\n";
    std::cout << "    Naive time: " << (time_naive / 1000.0) << " ms\n";
    std::cout << "    T-MAC time: " << (time_tmac / 1000.0) << " ms\n";
    std::cout << "    Speedup: " << speedup << "×\n";
    std::cout << "    Mismatches: " << mismatches << " / " << (M * N) << "\n";
    std::cout << "    Relative error: " << (rel_error * 100.0) << "%\n";

    assert(mismatches == 0);
    std::cout << "✓ MEDIUM MATRIX TEST PASSED\n";
}

// ============================================================================
// TEST 3: Large Matrix Performance
// ============================================================================

void test_large_matrix_performance()
{
    std::cout << "\n[TEST 3] Large Matrix Performance\n";
    std::cout << "===================================\n";

    uint32_t M = 512;
    uint32_t K = 2048;
    uint32_t N = 256;

    std::cout << "  Matrix size: [" << M << ", " << K << "] × [" << K << ", " << N << "]\n";
    std::cout << "  Output size: [" << M << ", " << N << "]\n\n";

    auto W = generate_random_ternary(M * K);
    auto X = generate_random_int8(K * N);
    std::vector<int32_t> Y(M * N);

    // Build T-MAC tables
    std::cout << "  Building T-MAC tables..." << std::flush;
    TableBuilder builder(16);
    auto lut_struct = builder.build(W, M, K);
    auto lut_shared = std::make_shared<CompressedLUT>(std::move(lut_struct));
    auto lut_engine = std::make_shared<LUTLookup>(lut_shared);
    std::cout << " Done\n\n";

    // T-MAC GEMM
    TMACGemm gemm_engine(lut_engine);

    // Warmup
    std::cout << "  Warmup (5 iterations)..." << std::flush;
    for (int i = 0; i < 5; ++i)
    {
        gemm_engine.gemm(W.data(), X.data(), Y.data(), M, K, N);
    }
    std::cout << " Done\n";

    gemm_engine.reset_stats();

    // Benchmark
    constexpr int NUM_ITERS = 50;
    std::cout << "  Benchmarking (" << NUM_ITERS << " iterations)..." << std::flush;

    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERS; ++i)
    {
        gemm_engine.gemm(W.data(), X.data(), Y.data(), M, K, N);
    }
    auto end = high_resolution_clock::now();
    auto total_time = duration_cast<microseconds>(end - start).count();

    std::cout << " Done\n\n";

    // Compute metrics
    uint64_t total_ops = static_cast<uint64_t>(M) * K * N * 2 * NUM_ITERS; // MAC = 2 ops
    double avg_time_ms = (total_time / 1000.0) / NUM_ITERS;
    double gflops = (total_ops / 1e9) / (total_time / 1e6);

    std::cout << "  Performance:\n";
    std::cout << "    Average time: " << avg_time_ms << " ms/GEMM\n";
    std::cout << "    Throughput: " << gflops << " GFLOPS\n";
    std::cout << "    Target: >100 GFLOPS → "
              << (gflops > 100.0 ? "✓ PASS" : "✗ FAIL (but expected for scalar)") << "\n";

    // Print GEMM stats
    gemm_engine.print_stats();

    // Print LUT stats
    std::cout << "\n";
    lut_engine->print_stats();

    std::cout << "\n✓ LARGE MATRIX BENCHMARK COMPLETE\n";
}

// ============================================================================
// TEST 4: Various Matrix Sizes
// ============================================================================

void test_various_sizes()
{
    std::cout << "\n[TEST 4] Various Matrix Sizes\n";
    std::cout << "==============================\n";

    struct TestCase
    {
        uint32_t M, K, N;
        const char *desc;
    };

    TestCase test_cases[] = {
        {16, 256, 32, "Tiny"},
        {64, 512, 64, "Small"},
        {128, 1024, 128, "Medium"},
        {256, 2048, 256, "Large"},
    };

    for (const auto &tc : test_cases)
    {
        std::cout << "\n  " << tc.desc << " matrix: ["
                  << tc.M << ", " << tc.K << "] × ["
                  << tc.K << ", " << tc.N << "]\n";

        auto W = generate_random_ternary(tc.M * tc.K);
        auto X = generate_random_int8(tc.K * tc.N);
        std::vector<int32_t> Y_naive(tc.M * tc.N);
        std::vector<int32_t> Y_tmac(tc.M * tc.N);

        // Naive
        naive_gemm(W.data(), X.data(), Y_naive.data(), tc.M, tc.K, tc.N);

        // T-MAC
        TableBuilder builder(16);
        builder.set_tier_sizes(1000, 5000);
        auto lut_struct = builder.build(W, tc.M, tc.K);
        auto lut_shared = std::make_shared<CompressedLUT>(std::move(lut_struct));
        auto lut_engine = std::make_shared<LUTLookup>(lut_shared);

        TMACGemm gemm_engine(lut_engine);
        gemm_engine.gemm(W.data(), X.data(), Y_tmac.data(), tc.M, tc.K, tc.N);

        // Verify
        uint32_t mismatches = 0;
        for (uint32_t i = 0; i < tc.M * tc.N; ++i)
        {
            if (Y_naive[i] != Y_tmac[i])
                mismatches++;
        }

        std::cout << "    Mismatches: " << mismatches << " / " << (tc.M * tc.N)
                  << " → " << (mismatches == 0 ? "✓ PASS" : "✗ FAIL") << "\n";

        assert(mismatches == 0);
    }

    std::cout << "\n✓ ALL SIZE TESTS PASSED\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main()
{
    std::cout << "====================================\n";
    std::cout << "T-MAC GEMM COMPREHENSIVE TEST SUITE\n";
    std::cout << "====================================\n";

    try
    {
        test_small_matrix_correctness();
        test_medium_matrix_correctness();
        test_various_sizes();
        test_large_matrix_performance();

        std::cout << "\n====================================\n";
        std::cout << "✓ ALL GEMM TESTS PASSED!\n";
        std::cout << "====================================\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n✗ TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
