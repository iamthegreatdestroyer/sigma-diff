// Copyright (c) 2024 Ryzanstein LLM Project
// Licensed under MIT License
//
// Mamba Selective State Space Model - Public API
// [REF:CC-004c] - Core Components: Mamba SSM Runtime
//
// This header defines the selective state space mechanism that enables
// Mamba to achieve linear-time sequence modeling without attention.
//
// Paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//        Gu & Dao, 2023 (https://arxiv.org/abs/2312.00752)
//
// Key Features:
// - O(N) complexity vs O(N²) for attention
// - Selective mechanism (input-dependent state transitions)
// - Efficient recurrent inference mode
// - Parallel scan for training/prefill
// - Hardware-aware SIMD optimizations
//
// Architecture:
//   Input (D)
//     ↓
//   Linear Projection → [B, C, Δ] (selective parameters)
//     ↓
//   SSM State Update:
//     A_bar = exp(Δ ⊙ A)           (discretization)
//     B_bar = (Δ ⊙ B)
//     h[t] = A_bar ⊙ h[t-1] + B_bar ⊙ x[t]   (state update)
//     y[t] = C ⊙ h[t]               (output)
//     ↓
//   Output (D)

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace ryzanstein_llm
{
    namespace mamba
    {

        // Forward declarations
        class ParallelScan;

        // Configuration for SSM layer
        struct SSMConfig
        {
            size_t d_model; // Model dimension (e.g., 2560 for Mamba-2.8B)
            size_t d_state; // State dimension (typically 16)
            size_t d_conv;  // Convolution kernel size (typically 4)
            float dt_min;   // Min delta value (typically 0.001)
            float dt_max;   // Max delta value (typically 0.1)
            bool use_cuda;  // Use CUDA if available (false for CPU-only)

            // CPU-specific optimizations
            bool use_avx512;    // Use AVX-512 SIMD instructions
            size_t num_threads; // Number of threads for parallel operations

            SSMConfig()
                : d_model(2560), d_state(16), d_conv(4), dt_min(0.001f), dt_max(0.1f), use_cuda(false), use_avx512(true), num_threads(8)
            {
            }
        };

        // SSM state for recurrent inference
        struct SSMState
        {
            std::vector<float> h;          // Hidden state [d_model, d_state]
            std::vector<float> conv_state; // Convolution state [d_model, d_conv-1]
            size_t seq_pos;                // Current sequence position

            SSMState() : seq_pos(0) {}

            void reset()
            {
                std::fill(h.begin(), h.end(), 0.0f);
                std::fill(conv_state.begin(), conv_state.end(), 0.0f);
                seq_pos = 0;
            }

            size_t memory_bytes() const
            {
                return h.size() * sizeof(float) +
                       conv_state.size() * sizeof(float) +
                       sizeof(seq_pos);
            }
        };

        // SSM parameters (learned during training)
        struct SSMParameters
        {
            // Linear projections for selective parameters
            std::vector<float> W_x;  // [d_model, d_inner] - input projection
            std::vector<float> W_z;  // [d_model, d_inner] - gate projection
            std::vector<float> W_B;  // [d_inner, d_state] - B parameter projection
            std::vector<float> W_C;  // [d_inner, d_state] - C parameter projection
            std::vector<float> W_dt; // [d_inner] - delta projection

            // SSM core parameters
            std::vector<float> A_log; // [d_inner, d_state] - A matrix (log scale)
            std::vector<float> D;     // [d_inner] - skip connection weights

            // Convolution parameters
            std::vector<float> conv_weight; // [d_inner, d_conv] - 1D conv kernel
            std::vector<float> conv_bias;   // [d_inner] - conv bias

            // Output projection
            std::vector<float> W_out; // [d_inner, d_model] - output projection

            // Dimensions
            size_t d_model;
            size_t d_inner; // Expansion ratio (typically 2× d_model)
            size_t d_state;
            size_t d_conv;

            SSMParameters() : d_model(0), d_inner(0), d_state(0), d_conv(0) {}

            size_t memory_bytes() const
            {
                return (W_x.size() + W_z.size() + W_B.size() + W_C.size() +
                        W_dt.size() + A_log.size() + D.size() +
                        conv_weight.size() + conv_bias.size() + W_out.size()) *
                       sizeof(float);
            }
        };

        // Performance statistics
        struct SSMStats
        {
            uint64_t total_forward_calls;
            uint64_t total_tokens_processed;
            double total_time_ms;
            double conv_time_ms;
            double ssm_time_ms;
            double proj_time_ms;

            SSMStats() { reset(); }

            void reset()
            {
                total_forward_calls = 0;
                total_tokens_processed = 0;
                total_time_ms = 0.0;
                conv_time_ms = 0.0;
                ssm_time_ms = 0.0;
                proj_time_ms = 0.0;
            }

            double get_avg_time_per_token_ms() const
            {
                return total_tokens_processed > 0 ? total_time_ms / total_tokens_processed : 0.0;
            }

            double get_throughput_tokens_per_sec() const
            {
                return total_time_ms > 0 ? (total_tokens_processed * 1000.0) / total_time_ms : 0.0;
            }

            std::string to_string() const;
        };

        // Main Selective SSM class
        class SelectiveSSM
        {
        public:
            explicit SelectiveSSM(const SSMConfig &config = SSMConfig());
            ~SelectiveSSM();

            // Initialize with parameters (load from trained model)
            bool Initialize(const SSMParameters &params);

            // Forward pass - prefill mode (parallel scan)
            // Input: [batch, seq_len, d_model]
            // Output: [batch, seq_len, d_model]
            void ForwardPrefill(
                const float *input,
                float *output,
                size_t batch_size,
                size_t seq_len);

            // Forward pass - generation mode (recurrent, single token)
            // Input: [batch, 1, d_model]
            // Output: [batch, 1, d_model]
            void ForwardGeneration(
                const float *input,
                float *output,
                size_t batch_size,
                SSMState &state);

            // State management
            SSMState CreateState() const;
            void ResetState(SSMState &state) const;

            // Accessors
            const SSMConfig &GetConfig() const { return config_; }
            const SSMParameters &GetParameters() const { return params_; }
            const SSMStats &GetStats() const { return stats_; }
            void ResetStats() { stats_.reset(); }

            bool IsInitialized() const { return initialized_; }
            size_t GetMemoryUsage() const;

        private:
            // Core SSM operations
            void ApplyConv1D(
                const float *input,
                float *output,
                float *conv_state,
                size_t batch_size,
                size_t seq_len);

            void ComputeSelectiveParameters(
                const float *x,
                float *B,
                float *C,
                float *delta,
                size_t batch_size,
                size_t seq_len);

            void SSMStep(
                const float *x,
                const float *B,
                const float *C,
                const float *delta,
                float *h,
                float *y,
                size_t batch_size,
                size_t seq_len);

            void SSMStepRecurrent(
                const float *x,
                const float *B,
                const float *C,
                const float *delta,
                float *h,
                float *y,
                size_t batch_size);

            // Helper functions
            void Discretize(
                const float *delta,
                float *A_bar,
                float *B_bar,
                const float *B,
                size_t batch_size,
                size_t seq_len);

            void MatMul(
                const float *A,
                const float *B,
                float *C,
                size_t M,
                size_t N,
                size_t K,
                bool transpose_B = false);

            void SiLU(float *x, size_t size);

            // SIMD optimizations
            void MatMulAVX512(
                const float *A,
                const float *B,
                float *C,
                size_t M,
                size_t N,
                size_t K);

            SSMConfig config_;
            SSMParameters params_;
            SSMStats stats_;
            bool initialized_;

            // Work buffers (avoid repeated allocation)
            std::vector<float> x_proj_;   // [batch, seq_len, d_inner]
            std::vector<float> z_proj_;   // [batch, seq_len, d_inner]
            std::vector<float> B_sel_;    // [batch, seq_len, d_state]
            std::vector<float> C_sel_;    // [batch, seq_len, d_state]
            std::vector<float> delta_;    // [batch, seq_len, d_inner]
            std::vector<float> conv_out_; // [batch, seq_len, d_inner]
            std::vector<float> ssm_out_;  // [batch, seq_len, d_inner]
            std::vector<float> A_bar_;    // [batch, seq_len, d_inner, d_state]
            std::vector<float> B_bar_;    // [batch, seq_len, d_inner, d_state]

            // Parallel scan helper
            std::unique_ptr<ParallelScan> parallel_scan_;
        };

        // Global statistics
        extern SSMStats g_ssm_stats;

    } // namespace mamba
} // namespace ryzanstein_llm
