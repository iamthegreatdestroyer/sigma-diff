
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <omp.h>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

// SIMD detection
#if defined(__AVX512F__) && defined(__AVX512BW__)
#define HAS_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define HAS_AVX2 1
#include <immintrin.h>
#endif

namespace benchmark {

// Simple ternary matrix multiplication benchmark
// Simulates the core computation in BitNet layers
void benchmark_ternary_matmul_scalar(
    const int8_t* weights, const float* activations,
    float* output, int M, int N, int K)
{
    #pragma omp parallel for schedule(dynamic, 4) if (N > 16)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int8_t w = weights[m * K + k];
                float act = activations[n * K + k];
                if (w == 1) sum += act;
                else if (w == -1) sum -= act;
            }
            output[m * N + n] = sum;
        }
    }
}

#if HAS_AVX512
void benchmark_ternary_matmul_avx512(
    const int8_t* weights, const float* activations,
    float* output, int M, int N, int K)
{
    for (int m = 0; m < M; ++m) {
        int n = 0;
        for (; n + 16 <= N; n += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int k = 0; k < K; ++k) {
                __m128i acts_128 = _mm_loadu_si128((__m128i*)&activations[n * K + k]);
                __m512i acts_vec = _mm512_cvtepi8_epi32(acts_128);
                __m512 acts_f = _mm512_cvtepi32_ps(acts_vec);

                int8_t w = weights[m * K + k];
                __m512 w_vec = _mm512_set1_ps((float)w);
                acc = _mm512_fmadd_ps(w_vec, acts_f, acc);
            }
            _mm512_storeu_ps(&output[m * N + n], acc);
        }
        // Handle remainder with scalar
        for (; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int8_t w = weights[m * K + k];
                float act = activations[n * K + k];
                if (w == 1) sum += act;
                else if (w == -1) sum -= act;
            }
            output[m * N + n] = sum;
        }
    }
}
#endif

#if HAS_AVX2
void benchmark_ternary_matmul_avx2(
    const int8_t* weights, const float* activations,
    float* output, int M, int N, int K)
{
    for (int m = 0; m < M; ++m) {
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; ++k) {
                __m128i acts_128 = _mm_loadl_epi64((__m128i*)&activations[n * K + k]);
                __m256i acts_vec = _mm256_cvtepi8_epi32(acts_128);
                __m256 acts_f = _mm256_cvtepi32_ps(acts_vec);

                int8_t w = weights[m * K + k];
                __m256 w_vec = _mm256_set1_ps((float)w);
                acc = _mm256_fmadd_ps(w_vec, acts_f, acc);
            }
            _mm256_storeu_ps(&output[m * N + n], acc);
        }
        // Handle remainder with scalar
        for (; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int8_t w = weights[m * K + k];
                float act = activations[n * K + k];
                if (w == 1) sum += act;
                else if (w == -1) sum -= act;
            }
            output[m * N + n] = sum;
        }
    }
}
#endif

} // namespace benchmark

int main(int argc, char* argv[]) {
    std::cout << "PHASE 3 SIMD MICRO-BENCHMARK\n";
    std::cout << "==============================\n";

    // Test parameters - similar to BitNet layer sizes
    const int M = 4096;  // Output features
    const int N = 32;    // Batch size
    const int K = 4096;  // Input features

    std::cout << "Test configuration: " << M << "×" << N << " output, " << K << " inner\n";
    std::cout << "Memory footprint: "
              << (M*N + M*K + N*K) * sizeof(float) / (1024*1024) << " MB\n\n";

    // Allocate memory
    std::vector<int8_t> weights(M * K);
    std::vector<float> activations(N * K);
    std::vector<float> output(M * N);

    // Initialize with realistic ternary weights and activations
    for (size_t i = 0; i < weights.size(); ++i) {
        int r = rand() % 3;
        weights[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
    }
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = (rand() % 201 - 100) / 100.0f; // -1.0 to 1.0
    }

    // Benchmark scalar version
    std::cout << "Running scalar benchmark...\n";
    auto start = std::chrono::high_resolution_clock::now();
    benchmark::benchmark_ternary_matmul_scalar(weights.data(), activations.data(), output.data(), M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    double scalar_time = std::chrono::duration<double>(end - start).count();
    double scalar_gflops = (2.0 * M * N * K) / scalar_time / 1e9;

    std::cout << "Scalar time: " << scalar_time << " s\n";
    std::cout << "Scalar throughput: " << scalar_gflops << " GFLOPS\n\n";

    double best_time = scalar_time;
    double best_gflops = scalar_gflops;
    std::string best_method = "scalar";

#if HAS_AVX512
    std::cout << "Running AVX-512 benchmark...\n";
    start = std::chrono::high_resolution_clock::now();
    benchmark::benchmark_ternary_matmul_avx512(weights.data(), activations.data(), output.data(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    double avx512_time = std::chrono::duration<double>(end - start).count();
    double avx512_gflops = (2.0 * M * N * K) / avx512_time / 1e9;

    std::cout << "AVX-512 time: " << avx512_time << " s\n";
    std::cout << "AVX-512 throughput: " << avx512_gflops << " GFLOPS\n";
    std::cout << "AVX-512 speedup: " << scalar_time / avx512_time << "×\n\n";

    if (avx512_gflops > best_gflops) {
        best_time = avx512_time;
        best_gflops = avx512_gflops;
        best_method = "AVX-512";
    }
#endif

#if HAS_AVX2
    std::cout << "Running AVX2 benchmark...\n";
    start = std::chrono::high_resolution_clock::now();
    benchmark::benchmark_ternary_matmul_avx2(weights.data(), activations.data(), output.data(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    double avx2_time = std::chrono::duration<double>(end - start).count();
    double avx2_gflops = (2.0 * M * N * K) / avx2_time / 1e9;

    std::cout << "AVX2 time: " << avx2_time << " s\n";
    std::cout << "AVX2 throughput: " << avx2_gflops << " GFLOPS\n";
    std::cout << "AVX2 speedup: " << scalar_time / avx2_time << "×\n\n";

    if (avx2_gflops > best_gflops) {
        best_time = avx2_time;
        best_gflops = avx2_gflops;
        best_method = "AVX2";
    }
#endif

    std::cout << "SUMMARY\n";
    std::cout << "=======\n";
    std::cout << "Best method: " << best_method << "\n";
    std::cout << "Best throughput: " << best_gflops << " GFLOPS\n";
    std::cout << "Best speedup: " << scalar_time / best_time << "×\n";

    // Estimate token throughput (rough approximation)
    // Assuming 4096 tokens per forward pass, 4096 hidden size
    double tokens_per_sec = (N * 4096) / best_time;  // Rough estimate
    std::cout << "Estimated token throughput: " << tokens_per_sec << " tok/s\n";

    return 0;
}
