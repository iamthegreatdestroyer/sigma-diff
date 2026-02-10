#pragma once

/**
 * @file optimization_utils.h
 * @brief Performance optimization utilities for BitNet inference
 *
 * Provides:
 * - Memory prefetching helpers
 * - Vectorization utilities
 * - SIMD intrinsics wrappers
 * - Performance timing macros
 *
 * [REF:VELOCITY-002] - Optimization Utilities
 */

#include <cstdint>
#include <chrono>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
#warning "OpenMP not available - multi-threading disabled"
#endif

namespace ryzanstein_llm
{
    namespace bitnet
    {
        // ============================================================================
        // MEMORY PREFETCHING UTILITIES
        // ============================================================================

        /**
         * Prefetch memory to L1 cache (temporal locality)
         *
         * @param ptr Pointer to memory to prefetch
         * @param bytes Number of bytes to prefetch (recommendation: 64-128)
         */
        inline void prefetch_l1(const void *ptr, size_t bytes = 64)
        {
#ifdef __GNUC__
            for (size_t i = 0; i < bytes; i += 64)
            {
                __builtin_prefetch((const char *)ptr + i, 0, 3); // Read, high temporal
            }
#elif defined(_MSC_VER)
            _mm_prefetch((const char *)ptr, _MM_HINT_T0);
#endif
        }

        /**
         * Prefetch memory to L2 cache (moderate temporal locality)
         *
         * @param ptr Pointer to memory to prefetch
         * @param bytes Number of bytes to prefetch
         */
        inline void prefetch_l2(const void *ptr, size_t bytes = 128)
        {
#ifdef __GNUC__
            for (size_t i = 0; i < bytes; i += 64)
            {
                __builtin_prefetch((const char *)ptr + i, 0, 2); // Read, moderate temporal
            }
#elif defined(_MSC_VER)
            _mm_prefetch((const char *)ptr, _MM_HINT_T1);
#endif
        }

        /**
         * Prefetch memory to L3 cache (low temporal locality)
         *
         * @param ptr Pointer to memory to prefetch
         * @param bytes Number of bytes to prefetch
         */
        inline void prefetch_l3(const void *ptr, size_t bytes = 256)
        {
#ifdef __GNUC__
            for (size_t i = 0; i < bytes; i += 64)
            {
                __builtin_prefetch((const char *)ptr + i, 0, 1); // Read, low temporal
            }
#elif defined(_MSC_VER)
            _mm_prefetch((const char *)ptr, _MM_HINT_T2);
#endif
        }

        // ============================================================================
        // VECTORIZED OPERATIONS
        // ============================================================================

        /**
         * Compute horizontal sum of float array using SIMD
         *
         * For 256-bit vectors (AVX/AVX2):
         * - Process 8 floats at a time
         * - Expected speedup: 6-8× vs scalar
         *
         * @param data Array of floats
         * @param N Number of elements
         * @return Sum of all elements
         */
        inline float horizontal_sum_simd(const float *data, uint32_t N)
        {
#ifdef __AVX2__
            __m256 sum_vec = _mm256_setzero_ps();
            uint32_t i = 0;

            // Process 8 floats at a time
            for (; i + 8 <= N; i += 8)
            {
                __m256 v = _mm256_loadu_ps(data + i);
                sum_vec = _mm256_add_ps(sum_vec, v);
            }

            // Horizontal sum of 8 floats
            __m256 temp = _mm256_permute2f128_ps(sum_vec, sum_vec, 1);
            sum_vec = _mm256_add_ps(sum_vec, temp);
            temp = _mm256_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1));
            sum_vec = _mm256_add_ps(sum_vec, temp);
            temp = _mm256_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(1, 0, 3, 2));
            sum_vec = _mm256_add_ps(sum_vec, temp);

            float result = _mm256_cvtss_f32(sum_vec);

            // Handle remaining elements
            for (; i < N; ++i)
            {
                result += data[i];
            }

            return result;
#else
            float result = 0.0f;
            for (uint32_t i = 0; i < N; ++i)
            {
                result += data[i];
            }
            return result;
#endif
        }

        /**
         * Subtract a scalar from all elements in place using SIMD
         *
         * @param data Array to modify
         * @param N Number of elements
         * @param value Value to subtract
         */
        inline void subtract_scalar_simd(float *data, uint32_t N, float value)
        {
#ifdef __AVX2__
            __m256 v_scalar = _mm256_set1_ps(value);
            uint32_t i = 0;

            for (; i + 8 <= N; i += 8)
            {
                __m256 v = _mm256_loadu_ps(data + i);
                v = _mm256_sub_ps(v, v_scalar);
                _mm256_storeu_ps(data + i, v);
            }

            // Handle remaining elements
            for (; i < N; ++i)
            {
                data[i] -= value;
            }
#else
            for (uint32_t i = 0; i < N; ++i)
            {
                data[i] -= value;
            }
#endif
        }

        /**
         * Divide all elements by a scalar in place using SIMD
         *
         * @param data Array to modify
         * @param N Number of elements
         * @param divisor Value to divide by
         */
        inline void divide_scalar_simd(float *data, uint32_t N, float divisor)
        {
#ifdef __AVX2__
            __m256 v_divisor = _mm256_set1_ps(divisor);
            uint32_t i = 0;

            for (; i + 8 <= N; i += 8)
            {
                __m256 v = _mm256_loadu_ps(data + i);
                v = _mm256_div_ps(v, v_divisor);
                _mm256_storeu_ps(data + i, v);
            }

            // Handle remaining elements
            for (; i < N; ++i)
            {
                data[i] /= divisor;
            }
#else
            for (uint32_t i = 0; i < N; ++i)
            {
                data[i] /= divisor;
            }
#endif
        }

        /**
         * Element-wise multiplication with accumulation
         *
         * output[i] += scale * data[i]
         *
         * @param data Input array
         * @param output Output array
         * @param N Number of elements
         * @param scale Scaling factor
         */
        inline void multiply_accumulate_simd(
            const float *data,
            float *output,
            uint32_t N,
            float scale)
        {
#ifdef __AVX2__
            __m256 v_scale = _mm256_set1_ps(scale);
            uint32_t i = 0;

            for (; i + 8 <= N; i += 8)
            {
                __m256 v_data = _mm256_loadu_ps(data + i);
                __m256 v_out = _mm256_loadu_ps(output + i);
                v_out = _mm256_add_ps(v_out, _mm256_mul_ps(v_data, v_scale));
                _mm256_storeu_ps(output + i, v_out);
            }

            // Handle remaining
            for (; i < N; ++i)
            {
                output[i] += data[i] * scale;
            }
#else
            for (uint32_t i = 0; i < N; ++i)
            {
                output[i] += data[i] * scale;
            }
#endif
        }

        // ============================================================================
        // PERFORMANCE MONITORING
        // ============================================================================

        /**
         * Simple timer for performance measurement
         */
        class PerfTimer
        {
        public:
            using Clock = std::chrono::high_resolution_clock;

            void start()
            {
                start_time_ = Clock::now();
            }

            double elapsed_ms() const
            {
                auto end = Clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_);
                return duration.count() / 1000.0;
            }

            double elapsed_us() const
            {
                auto end = Clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_);
                return duration.count();
            }

        private:
            Clock::time_point start_time_;
        };

        // ============================================================================
        // OPENMP UTILITIES
        // ============================================================================

        /**
         * Get effective number of threads to use
         *
         * @return Number of threads available (1 if OpenMP disabled)
         */
        inline int get_num_threads()
        {
#ifdef _OPENMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        }

        /**
         * Set number of threads for parallel regions
         *
         * @param num_threads Number of threads to use
         */
        inline void set_num_threads(int num_threads)
        {
#ifdef _OPENMP
            omp_set_num_threads(num_threads);
#endif
        }

    } // namespace bitnet
} // namespace ryzanstein_llm
