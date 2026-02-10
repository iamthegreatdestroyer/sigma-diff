// Copyright (c) 2024 RYZEN-LLM Project
// Licensed under MIT License
//
// Parallel Scan Operations for Mamba - Public API
// [REF:CC-004c] - Core Components: Mamba SSM Runtime
//
// This header defines efficient parallel scan algorithms for Mamba's
// state space operations, enabling parallelization across sequence length.
//
// Paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//        Section 3.3 - Efficient Implementation
//
// Key Features:
// - Work-efficient parallel scan (Blelloch algorithm)
// - Associative operator for SSM state updates
// - Multi-threaded execution (OpenMP-style)
// - SIMD-optimized operations (AVX-512)
//
// Algorithm:
//   Given sequence of (A, B*x) pairs, compute cumulative state updates:
//   h[0] = B[0] * x[0]
//   h[1] = A[1] * h[0] + B[1] * x[1]
//   h[2] = A[2] * h[1] + B[2] * x[2]
//   ...
//
//   Parallel scan computes this in O(log N) parallel depth vs O(N) sequential

#pragma once

#include <cstdint>
#include <vector>
#include <functional>

namespace ryzen_llm
{
    namespace mamba
    {

        // Associative operator for SSM state updates
        // Combines two consecutive state updates: (A2, B2) ∘ (A1, B1)
        // Result: (A2 * A1, A2 * B1 + B2)
        struct SSMOperator
        {
            std::vector<float> A;  // State transition matrices [d_inner, d_state]
            std::vector<float> Bx; // Input contributions [d_inner, d_state]

            size_t d_inner;
            size_t d_state;

            SSMOperator() : d_inner(0), d_state(0) {}

            SSMOperator(size_t di, size_t ds)
                : d_inner(di), d_state(ds)
            {
                A.resize(di * ds);
                Bx.resize(di * ds);
            }

            // Compose two operators: this = right ∘ left
            void Compose(const SSMOperator &left, const SSMOperator &right);
        };

        // Parallel scan configuration
        struct ScanConfig
        {
            size_t num_threads; // Number of parallel threads
            size_t grain_size;  // Minimum work per thread (default 256)
            bool use_avx512;    // Use AVX-512 SIMD

            ScanConfig()
                : num_threads(8), grain_size(256), use_avx512(true)
            {
            }
        };

        // Statistics for parallel scan
        struct ScanStats
        {
            uint64_t total_scans;
            uint64_t total_elements;
            double total_time_ms;
            double upsweep_time_ms;
            double downsweep_time_ms;

            ScanStats() { reset(); }

            void reset()
            {
                total_scans = 0;
                total_elements = 0;
                total_time_ms = 0.0;
                upsweep_time_ms = 0.0;
                downsweep_time_ms = 0.0;
            }

            double get_avg_time_per_scan_ms() const
            {
                return total_scans > 0 ? total_time_ms / total_scans : 0.0;
            }

            double get_throughput_elements_per_sec() const
            {
                return total_time_ms > 0 ? (total_elements * 1000.0) / total_time_ms : 0.0;
            }
        };

        // Parallel scan implementation
        class ParallelScan
        {
        public:
            explicit ParallelScan(size_t num_threads = 8);
            ~ParallelScan();

            // Parallel scan for SSM state updates
            // Inputs:
            //   A_bar: [batch, seq_len, d_inner, d_state] - discretized A matrices
            //   B_bar: [batch, seq_len, d_inner, d_state] - discretized B matrices
            //   x: [batch, seq_len, d_inner] - input sequence
            // Output:
            //   h: [batch, seq_len, d_inner, d_state] - cumulative states
            void ScanSSM(
                const float *A_bar,
                const float *B_bar,
                const float *x,
                float *h,
                size_t batch_size,
                size_t seq_len,
                size_t d_inner,
                size_t d_state);

            // Generic parallel prefix scan
            // Given sequence [a0, a1, a2, ...] and associative operator ⊕,
            // compute [a0, a0⊕a1, a0⊕a1⊕a2, ...]
            template <typename T, typename BinaryOp>
            void Scan(
                const T *input,
                T *output,
                size_t length,
                BinaryOp op,
                const T &identity);

            // Segmented scan for batched sequences
            // Each segment is scanned independently
            void SegmentedScan(
                const float *input,
                float *output,
                const size_t *segment_lengths,
                size_t num_segments,
                size_t d_inner,
                size_t d_state);

            // Configuration
            void SetNumThreads(size_t n) { config_.num_threads = n; }
            void SetGrainSize(size_t g) { config_.grain_size = g; }

            const ScanStats &GetStats() const { return stats_; }
            void ResetStats() { stats_.reset(); }

        private:
            // Blelloch parallel scan phases
            void Upsweep(
                SSMOperator *data,
                size_t length,
                size_t d_inner,
                size_t d_state);

            void Downsweep(
                SSMOperator *data,
                size_t length,
                size_t d_inner,
                size_t d_state,
                const SSMOperator &identity);

            // Helper: compose operators with SIMD
            void ComposeOperatorsAVX512(
                const SSMOperator &left,
                const SSMOperator &right,
                SSMOperator &result);

            // Thread pool management
            void InitThreadPool();
            void ShutdownThreadPool();

            ScanConfig config_;
            ScanStats stats_;

            // Work buffers
            std::vector<SSMOperator> operators_;
            std::vector<SSMOperator> results_;

            // Thread pool (simplified - use std::thread in real implementation)
            bool thread_pool_initialized_;
        };

        // Template implementation for generic scan
        template <typename T, typename BinaryOp>
        void ParallelScan::Scan(
            const T *input,
            T *output,
            size_t length,
            BinaryOp op,
            const T &identity)
        {
            if (length == 0)
                return;
            if (length == 1)
            {
                output[0] = input[0];
                return;
            }

            // For small sequences, use sequential scan
            if (length < config_.grain_size)
            {
                output[0] = input[0];
                for (size_t i = 1; i < length; ++i)
                {
                    output[i] = op(output[i - 1], input[i]);
                }
                return;
            }

            // Blelloch parallel scan (work-efficient)
            // 1. Upsweep: build partial sums in tree
            // 2. Downsweep: propagate sums down tree

            std::vector<T> tree(length * 2);

            // Copy input to leaves
            for (size_t i = 0; i < length; ++i)
            {
                tree[length + i] = input[i];
            }

            // Upsweep phase
            for (size_t stride = 1; stride < length; stride *= 2)
            {
                for (size_t i = 0; i < length; i += stride * 2)
                {
                    if (i + stride < length)
                    {
                        size_t left = length + i + stride - 1;
                        size_t right = length + i + stride * 2 - 1;
                        if (right < tree.size())
                        {
                            tree[right] = op(tree[left], tree[right]);
                        }
                    }
                }
            }

            // Set root to identity
            tree[tree.size() - 1] = identity;

            // Downsweep phase
            for (size_t stride = length; stride > 0; stride /= 2)
            {
                for (size_t i = 0; i < length; i += stride * 2)
                {
                    if (i + stride < length)
                    {
                        size_t left = length + i + stride - 1;
                        size_t right = length + i + stride * 2 - 1;
                        if (right < tree.size())
                        {
                            T temp = tree[left];
                            tree[left] = tree[right];
                            tree[right] = op(tree[right], temp);
                        }
                    }
                }
            }

            // Copy results
            for (size_t i = 0; i < length; ++i)
            {
                output[i] = tree[length + i];
            }
        }

    } // namespace mamba
} // namespace ryzen_llm
