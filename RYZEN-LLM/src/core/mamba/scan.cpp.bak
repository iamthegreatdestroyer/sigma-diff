// Copyright (c) 2024 RYZEN-LLM Project
// Licensed under MIT License
//
// Parallel Scan Operations for Mamba
// [REF:CC-004c] - Core Components: Mamba SSM Runtime

#include "mamba/scan.h"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <thread>
#include <omp.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace ryzen_llm
{
    namespace mamba
    {

        // SSMOperator implementation
        void SSMOperator::Compose(const SSMOperator &left, const SSMOperator &right)
        {
            // Compose: this = right ∘ left
            // A_result = A_right * A_left
            // Bx_result = A_right * Bx_left + Bx_right

            for (size_t i = 0; i < d_inner; ++i)
            {
                for (size_t j = 0; j < d_state; ++j)
                {
                    // A_result[i,j] = A_right[i,j] * A_left[i,j] (element-wise for diagonal)
                    A[i * d_state + j] = right.A[i * d_state + j] * left.A[i * d_state + j];

                    // Bx_result[i,j] = A_right[i,j] * Bx_left[i,j] + Bx_right[i,j]
                    Bx[i * d_state + j] = right.A[i * d_state + j] * left.Bx[i * d_state + j] +
                                          right.Bx[i * d_state + j];
                }
            }
        }

        // ParallelScan implementation
        ParallelScan::ParallelScan(size_t num_threads)
            : thread_pool_initialized_(false)
        {
            config_.num_threads = num_threads;
            InitThreadPool();
        }

        ParallelScan::~ParallelScan()
        {
            ShutdownThreadPool();
        }

        void ParallelScan::InitThreadPool()
        {
            // In a real implementation, create thread pool here
            // For now, use std::thread directly
            thread_pool_initialized_ = true;
        }

        void ParallelScan::ShutdownThreadPool()
        {
            thread_pool_initialized_ = false;
        }

        void ParallelScan::ScanSSM(
            const float *A_bar,
            const float *B_bar,
            const float *x,
            float *h,
            size_t batch_size,
            size_t seq_len,
            size_t d_inner,
            size_t d_state)
        {
            auto start = std::chrono::high_resolution_clock::now();

            // Process each batch independently
            for (size_t b = 0; b < batch_size; ++b)
            {
                // Create operators for this sequence
                operators_.resize(seq_len, SSMOperator(d_inner, d_state));
                results_.resize(seq_len, SSMOperator(d_inner, d_state));

                // Initialize operators: op[t] = (A_bar[t], B_bar[t] * x[t])
                for (size_t t = 0; t < seq_len; ++t)
                {
                    for (size_t i = 0; i < d_inner; ++i)
                    {
                        for (size_t j = 0; j < d_state; ++j)
                        {
                            size_t idx = b * seq_len * d_inner * d_state +
                                         t * d_inner * d_state + i * d_state + j;

                            operators_[t].A[i * d_state + j] = A_bar[idx];
                            operators_[t].Bx[i * d_state + j] = B_bar[idx] *
                                                                x[b * seq_len * d_inner +
                                                                  t * d_inner + i];
                        }
                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();

                // Perform parallel scan
                if (seq_len < config_.grain_size)
                {
                    // Sequential scan for small sequences
                    results_[0] = operators_[0];
                    for (size_t t = 1; t < seq_len; ++t)
                    {
                        results_[t].Compose(results_[t - 1], operators_[t]);
                    }
                }
                else
                {
                    // Blelloch parallel scan
                    SSMOperator identity(d_inner, d_state);
                    for (size_t i = 0; i < d_inner * d_state; ++i)
                    {
                        identity.A[i] = 1.0f;  // Identity for multiplication
                        identity.Bx[i] = 0.0f; // Identity for addition
                    }

                    Upsweep(operators_.data(), seq_len, d_inner, d_state);
                    Downsweep(operators_.data(), seq_len, d_inner, d_state, identity);

                    // Copy results
                    for (size_t t = 0; t < seq_len; ++t)
                    {
                        results_[t] = operators_[t];
                    }
                }

                auto t2 = std::chrono::high_resolution_clock::now();

                // Extract states h[t] = results[t].Bx
                for (size_t t = 0; t < seq_len; ++t)
                {
                    for (size_t i = 0; i < d_inner; ++i)
                    {
                        for (size_t j = 0; j < d_state; ++j)
                        {
                            size_t h_idx = b * seq_len * d_inner * d_state +
                                           t * d_inner * d_state + i * d_state + j;
                            h[h_idx] = results_[t].Bx[i * d_state + j];
                        }
                    }
                }

                auto end = std::chrono::high_resolution_clock::now();
                (void)end; // intentionally unused timing marker

                stats_.upsweep_time_ms +=
                    std::chrono::duration<double, std::milli>(t1 - start).count();
                stats_.downsweep_time_ms +=
                    std::chrono::duration<double, std::milli>(t2 - t1).count();
            }

            auto total_end = std::chrono::high_resolution_clock::now();

            stats_.total_scans++;
            stats_.total_elements += batch_size * seq_len;
            stats_.total_time_ms +=
                std::chrono::duration<double, std::milli>(total_end - start).count();
        }

        void ParallelScan::Upsweep(
            SSMOperator *data,
            size_t length,
            size_t d_inner,
            size_t d_state)
        {
            // Build reduction tree bottom-up
            // After upsweep, data[length-1] contains total reduction

            for (size_t stride = 1; stride < length; stride *= 2)
            {
// Parallelize inner loop - iterations are independent within each stride level
// Each thread gets its own temp SSMOperator constructed in the loop body
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < length; i += stride * 2)
                {
                    size_t left_idx = i + stride - 1;
                    size_t right_idx = i + stride * 2 - 1;

                    if (right_idx < length)
                    {
                        SSMOperator temp(d_inner, d_state);
                        temp.Compose(data[left_idx], data[right_idx]);
                        data[right_idx] = temp;
                    }
                }
            }
        }

        void ParallelScan::Downsweep(
            SSMOperator *data,
            size_t length,
            size_t d_inner,
            size_t d_state,
            const SSMOperator &identity)
        {
            // Set root to identity
            data[length - 1] = identity;

            // Traverse tree top-down
            for (size_t stride = length / 2; stride > 0; stride /= 2)
            {
// Parallelize inner loop - iterations are independent within each stride level
// Each thread gets its own temp/composed SSMOperators constructed in the loop body
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < length; i += stride * 2)
                {
                    size_t left_idx = i + stride - 1;
                    size_t right_idx = i + stride * 2 - 1;

                    if (right_idx < length)
                    {
                        SSMOperator temp = data[left_idx];
                        data[left_idx] = data[right_idx];

                        SSMOperator composed(d_inner, d_state);
                        composed.Compose(data[right_idx], temp);
                        data[right_idx] = composed;
                    }
                }
            }
        }

        void ParallelScan::SegmentedScan(
            const float *input,
            float *output,
            const size_t *segment_lengths,
            size_t num_segments,
            size_t d_inner,
            size_t d_state)
        {
            // Each segment is scanned independently
            size_t offset = 0;

            for (size_t seg = 0; seg < num_segments; ++seg)
            {
                size_t seg_len = segment_lengths[seg];

                // Create operators for this segment
                operators_.resize(seg_len, SSMOperator(d_inner, d_state));

                // Initialize from input
                for (size_t t = 0; t < seg_len; ++t)
                {
                    for (size_t i = 0; i < d_inner * d_state; ++i)
                    {
                        operators_[t].A[i] = input[offset + t * d_inner * d_state * 2 + i];
                        operators_[t].Bx[i] = input[offset + t * d_inner * d_state * 2 +
                                                    d_inner * d_state + i];
                    }
                }

                // Sequential scan for this segment
                results_.resize(seg_len, SSMOperator(d_inner, d_state));
                results_[0] = operators_[0];
                for (size_t t = 1; t < seg_len; ++t)
                {
                    results_[t].Compose(results_[t - 1], operators_[t]);
                }

                // Write output
                for (size_t t = 0; t < seg_len; ++t)
                {
                    for (size_t i = 0; i < d_inner * d_state; ++i)
                    {
                        output[offset + t * d_inner * d_state + i] = results_[t].Bx[i];
                    }
                }

                offset += seg_len * d_inner * d_state * 2;
            }
        }

#ifdef __AVX512F__
        void ParallelScan::ComposeOperatorsAVX512(
            const SSMOperator &left,
            const SSMOperator &right,
            SSMOperator &result)
        {
            size_t size = result.d_inner * result.d_state;

            size_t i = 0;
            for (; i + 16 <= size; i += 16)
            {
                // Load 16 values at a time
                __m512 a_right = _mm512_loadu_ps(&right.A[i]);
                __m512 a_left = _mm512_loadu_ps(&left.A[i]);
                __m512 bx_right = _mm512_loadu_ps(&right.Bx[i]);
                __m512 bx_left = _mm512_loadu_ps(&left.Bx[i]);

                // A_result = A_right * A_left
                __m512 a_result = _mm512_mul_ps(a_right, a_left);

                // Bx_result = A_right * Bx_left + Bx_right
                __m512 bx_result = _mm512_fmadd_ps(a_right, bx_left, bx_right);

                // Store results
                _mm512_storeu_ps(&result.A[i], a_result);
                _mm512_storeu_ps(&result.Bx[i], bx_result);
            }

            // Handle remaining elements
            for (; i < size; ++i)
            {
                result.A[i] = right.A[i] * left.A[i];
                result.Bx[i] = right.A[i] * left.Bx[i] + right.Bx[i];
            }
        }
#else
        void ParallelScan::ComposeOperatorsAVX512(
            const SSMOperator &left,
            const SSMOperator &right,
            SSMOperator &result)
        {
            // Fallback to SSMOperator::Compose
            result.Compose(left, right);
        }
#endif

    } // namespace mamba
} // namespace ryzen_llm
