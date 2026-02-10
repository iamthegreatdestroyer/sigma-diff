/**
 * @file benchmark_gemm_performance.cpp
 * @brief Comprehensive performance benchmark for T-MAC GEMM optimizations
 *
 * Measures and compares:
 *   - Baseline scalar implementation (50-100 GFLOPS)
 *   - Current AVX-512 implementation (100-300 GFLOPS)
 *   - Optimized AVX-512 implementation (500-800 GFLOPS target)
 *
 * Benchmark matrices:
 *   - Small:  128 × 512  × 512   (~67M FLOPs)
 *   - Medium: 512 × 2048 × 2048  (~2.1B FLOPs)
 *   - Large:  1024 × 4096 × 4096 (~34B FLOPs)
 *
 * [REF:VELOCITY-002] - Performance Benchmarking
 */

#include "../src/core/tmac/tmac_gemm.h"
#include "../src/core/tmac/lut_lookup.h"
#include "../src/core/tmac/table_builder.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <cstring>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

using namespace ryzen_llm::tmac;
using namespace std::chrono;

// Use the public header for the optimized GEMM API
#include "../src/core/tmac/tmac_gemm_optimized.h"

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void *aligned_alloc_wrapper(size_t size, size_t alignment)
{
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
    {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_free_wrapper(void *ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * Generate random ternary weight matrix {-1, 0, +1}
 */
void generate_ternary_weights(int8_t *W, uint32_t M, uint32_t K, float sparsity = 0.3f)
{
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = 0; i < M * K; ++i)
    {
        float val = dist(gen);
        if (val < sparsity)
        {
            W[i] = 0;
        }
        else if (val < 0.5f + sparsity / 2.0f)
        {
            W[i] = -1;
        }
        else
        {
            W[i] = 1;
        }
    }
}

/**
 * Generate random INT8 activation matrix
 */
void generate_activations(int8_t *X, uint32_t K, uint32_t N)
{
    std::mt19937 gen(67890);
    std::uniform_int_distribution<int> dist(-128, 127);

    for (uint32_t i = 0; i < K * N; ++i)
    {
        X[i] = static_cast<int8_t>(dist(gen));
    }
}

/**
 * Verify correctness: compare two result matrices
 */
bool verify_correctness(
    const int32_t *Y_ref,
    const int32_t *Y_test,
    uint32_t M,
    uint32_t N,
    double tolerance = 1e-6)
{
    uint32_t errors = 0;
    const uint32_t max_errors_to_print = 10;

    for (uint32_t i = 0; i < M * N; ++i)
    {
        if (Y_ref[i] != Y_test[i])
        {
            if (errors < max_errors_to_print)
            {
                std::cout << "  Mismatch at [" << (i / N) << ", " << (i % N) << "]: "
                          << "ref=" << Y_ref[i] << ", test=" << Y_test[i] << "\n";
            }
            errors++;
        }
    }

    if (errors > 0)
    {
        std::cout << "  Total mismatches: " << errors << " / " << (M * N)
                  << " (" << (100.0 * errors / (M * N)) << "%)\n";
        return false;
    }

    return true;
}

/**
 * Calculate GFLOPS for GEMM operation
 */
double calculate_gflops(uint32_t M, uint32_t K, uint32_t N, double time_ms)
{
    // GEMM: M×N outputs, each requires K MACs (2 FLOPs per MAC)
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (time_ms / 1000.0);
    return gflops;
}

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

struct BenchmarkConfig
{
    uint32_t M, K, N;
    std::string name;
    uint32_t warmup_iterations = 3;
    uint32_t benchmark_iterations = 10;
};

const std::vector<BenchmarkConfig> BENCHMARK_CONFIGS = {
    {128, 512, 512, "Small (128×512×512)", 3, 20},
    {512, 2048, 2048, "Medium (512×2K×2K)", 3, 10},
    {1024, 4096, 4096, "Large (1024×4K×4K)", 2, 5},
    {256, 1024, 1024, "Square (256×1K×1K)", 3, 10},
    {64, 256, 8192, "Tall (64×256×8K)", 3, 15},
    {2048, 4096, 64, "Wide (2K×4K×64)", 3, 10},
};

// ============================================================================
// BENCHMARK IMPLEMENTATIONS
// ============================================================================

struct BenchmarkResult
{
    std::string name;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double gflops;
    bool correctness_verified;
};

/**
 * Benchmark baseline TMACGemm implementation
 */
BenchmarkResult benchmark_baseline(
    const BenchmarkConfig &config,
    std::shared_ptr<LUTLookup> lut_engine,
    const int8_t *W,
    const int8_t *X,
    int32_t *Y_ref)
{
    TMACGemm gemm_engine(lut_engine);

    // Warmup
    for (uint32_t i = 0; i < config.warmup_iterations; ++i)
    {
        gemm_engine.gemm(W, X, Y_ref, config.M, config.K, config.N);
    }

    // Benchmark
    std::vector<double> times;
    for (uint32_t i = 0; i < config.benchmark_iterations; ++i)
    {
        auto start = high_resolution_clock::now();
        gemm_engine.gemm(W, X, Y_ref, config.M, config.K, config.N);
        auto end = high_resolution_clock::now();

        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(time_ms);
    }

    // Calculate statistics
    double sum = 0.0, min_time = times[0], max_time = times[0];
    for (double t : times)
    {
        sum += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    double avg_time = sum / times.size();

    BenchmarkResult result;
    result.name = "Baseline (Current)";
    result.avg_time_ms = avg_time;
    result.min_time_ms = min_time;
    result.max_time_ms = max_time;
    result.gflops = calculate_gflops(config.M, config.K, config.N, avg_time);
    result.correctness_verified = true; // Reference implementation

    return result;
}

/**
 * Benchmark optimized implementation
 */
BenchmarkResult benchmark_optimized(
    const BenchmarkConfig &config,
    LUTLookup *lut_engine,
    const int8_t *W,
    const int8_t *X,
    int32_t *Y_test,
    const int32_t *Y_ref)
{
    // Warmup
    for (uint32_t i = 0; i < config.warmup_iterations; ++i)
    {
        gemm_optimized(lut_engine, W, X, Y_test, config.M, config.K, config.N);
    }

    // Benchmark
    std::vector<double> times;
    for (uint32_t i = 0; i < config.benchmark_iterations; ++i)
    {
        auto start = high_resolution_clock::now();
        gemm_optimized(lut_engine, W, X, Y_test, config.M, config.K, config.N);
        auto end = high_resolution_clock::now();

        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(time_ms);
    }

    // Verify correctness
    bool correct = verify_correctness(Y_ref, Y_test, config.M, config.N);

    // Calculate statistics
    double sum = 0.0, min_time = times[0], max_time = times[0];
    for (double t : times)
    {
        sum += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    double avg_time = sum / times.size();

    BenchmarkResult result;
    result.name = "Optimized (New)";
    result.avg_time_ms = avg_time;
    result.min_time_ms = min_time;
    result.max_time_ms = max_time;
    result.gflops = calculate_gflops(config.M, config.K, config.N, avg_time);
    result.correctness_verified = correct;

    return result;
}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================

int main(int argc, char **argv)
{
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       T-MAC GEMM PERFORMANCE BENCHMARK - AVX-512 OPTIMIZED     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    // Initialize LUT engine
    std::cout << "Initializing T-MAC lookup tables...\n";
    TableBuilder builder;

    // Generate random ternary weights for benchmark
    std::vector<int8_t> weights(512 * 2048);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-1, 1);
    for (auto &w : weights)
    {
        w = static_cast<int8_t>(dist(gen));
    }

    auto lut = builder.build(weights, 512, 2048);
    auto lut_shared = std::make_shared<CompressedLUT>(std::move(lut));
    auto lut_engine = std::make_shared<LUTLookup>(lut_shared);
    std::cout << "✓ Lookup tables ready\n\n";

    // Run benchmarks for each configuration
    for (const auto &config : BENCHMARK_CONFIGS)
    {
        std::cout << "═══════════════════════════════════════════════════════════\n";
        std::cout << "Benchmark: " << config.name << "\n";
        std::cout << "Matrix dimensions: M=" << config.M << ", K=" << config.K << ", N=" << config.N << "\n";
        std::cout << "───────────────────────────────────────────────────────────\n";

        // Allocate aligned memory
        int8_t *W = (int8_t *)aligned_alloc_wrapper(config.M * config.K, 64);
        int8_t *X = (int8_t *)aligned_alloc_wrapper(config.K * config.N, 64);
        int32_t *Y_ref = (int32_t *)aligned_alloc_wrapper(config.M * config.N * sizeof(int32_t), 64);
        int32_t *Y_test = (int32_t *)aligned_alloc_wrapper(config.M * config.N * sizeof(int32_t), 64);

        if (!W || !X || !Y_ref || !Y_test)
        {
            std::cerr << "ERROR: Memory allocation failed\n";
            return 1;
        }

        // Generate test data
        std::cout << "Generating test data...\n";
        generate_ternary_weights(W, config.M, config.K);
        generate_activations(X, config.K, config.N);

        // Benchmark baseline
        std::cout << "Running baseline benchmark...\n";
        auto result_baseline = benchmark_baseline(config, lut_engine, W, X, Y_ref);

        // Benchmark optimized
        std::cout << "Running optimized benchmark...\n";
        auto result_optimized = benchmark_optimized(config, lut_engine.get(), W, X, Y_test, Y_ref);

        // Print results
        std::cout << "\n┌─────────────────────────────────────────────────────────┐\n";
        std::cout << "│                    RESULTS                              │\n";
        std::cout << "├─────────────────────────────────────────────────────────┤\n";

        auto print_result = [](const BenchmarkResult &r)
        {
            std::cout << "│ " << std::left << std::setw(20) << r.name;
            std::cout << std::right << std::fixed << std::setprecision(2);
            std::cout << std::setw(10) << r.avg_time_ms << " ms";
            std::cout << std::setw(12) << r.gflops << " GFLOPS";
            std::cout << "   " << (r.correctness_verified ? "✓" : "✗") << " │\n";
        };

        print_result(result_baseline);
        print_result(result_optimized);

        // Calculate speedup
        double speedup = result_baseline.avg_time_ms / result_optimized.avg_time_ms;
        double gflops_improvement = result_optimized.gflops / result_baseline.gflops;

        std::cout << "├─────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Speedup:          " << std::setw(6) << std::fixed << std::setprecision(2)
                  << speedup << "×                                   │\n";
        std::cout << "│ GFLOPS Increase:  " << std::setw(6) << std::fixed << std::setprecision(2)
                  << gflops_improvement << "×                                   │\n";

        // Performance targets
        std::cout << "├─────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Target: 300 GFLOPS (3× speedup)   "
                  << (result_optimized.gflops >= 300.0 ? "✓ PASS" : "✗ FAIL") << "            │\n";
        std::cout << "│ Target: 500 GFLOPS (5× speedup)   "
                  << (result_optimized.gflops >= 500.0 ? "✓ PASS" : "✗ FAIL") << "            │\n";
        std::cout << "│ Stretch: 1000 GFLOPS (10× speedup) "
                  << (result_optimized.gflops >= 1000.0 ? "✓ PASS" : "✗ FAIL") << "           │\n";
        std::cout << "└─────────────────────────────────────────────────────────┘\n\n";

        // Cleanup
        aligned_free_wrapper(W);
        aligned_free_wrapper(X);
        aligned_free_wrapper(Y_ref);
        aligned_free_wrapper(Y_test);
    }

    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "Benchmark complete!\n\n";

    return 0;
}
