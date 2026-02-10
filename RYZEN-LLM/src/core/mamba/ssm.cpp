// Copyright (c) 2024 Ryzanstein LLM Project
// Licensed under MIT License
//
// Mamba Selective State Space Model Implementation
// [REF:CC-004c] - Core Components: Mamba SSM Runtime

#include "mamba/ssm.h"
#include "mamba/scan.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <sstream>
#include <iomanip>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace ryzanstein_llm
{
    namespace mamba
    {

        // Global statistics
        SSMStats g_ssm_stats;

        std::string SSMStats::to_string() const
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2);
            oss << "SSM Stats:\n";
            oss << "  Forward calls: " << total_forward_calls << "\n";
            oss << "  Tokens processed: " << total_tokens_processed << "\n";
            oss << "  Total time: " << total_time_ms << " ms\n";
            oss << "    Conv: " << conv_time_ms << " ms ("
                << (total_time_ms > 0 ? 100.0 * conv_time_ms / total_time_ms : 0.0) << "%)\n";
            oss << "    SSM: " << ssm_time_ms << " ms ("
                << (total_time_ms > 0 ? 100.0 * ssm_time_ms / total_time_ms : 0.0) << "%)\n";
            oss << "    Proj: " << proj_time_ms << " ms ("
                << (total_time_ms > 0 ? 100.0 * proj_time_ms / total_time_ms : 0.0) << "%)\n";
            oss << "  Avg time/token: " << get_avg_time_per_token_ms() << " ms\n";
            oss << "  Throughput: " << get_throughput_tokens_per_sec() << " tokens/sec";
            return oss.str();
        }

        SelectiveSSM::SelectiveSSM(const SSMConfig &config)
            : config_(config), initialized_(false)
        {
            parallel_scan_ = std::make_unique<ParallelScan>(config.num_threads);
        }

        SelectiveSSM::~SelectiveSSM() = default;

        bool SelectiveSSM::Initialize(const SSMParameters &params)
        {
            params_ = params;

            // Validate parameters
            if (params_.d_model == 0 || params_.d_inner == 0 ||
                params_.d_state == 0 || params_.d_conv == 0)
            {
                return false;
            }

            // Update config dimensions
            config_.d_model = params_.d_model;
            config_.d_state = params_.d_state;
            config_.d_conv = params_.d_conv;

            initialized_ = true;
            return true;
        }

        SSMState SelectiveSSM::CreateState() const
        {
            SSMState state;
            state.h.resize(params_.d_inner * params_.d_state, 0.0f);
            state.conv_state.resize(params_.d_inner * (params_.d_conv - 1), 0.0f);
            state.seq_pos = 0;
            return state;
        }

        void SelectiveSSM::ResetState(SSMState &state) const
        {
            state.reset();
        }

        size_t SelectiveSSM::GetMemoryUsage() const
        {
            return params_.memory_bytes() +
                   (x_proj_.size() + z_proj_.size() + B_sel_.size() +
                    C_sel_.size() + delta_.size() + conv_out_.size() +
                    ssm_out_.size() + A_bar_.size() + B_bar_.size()) *
                       sizeof(float);
        }

        void SelectiveSSM::ForwardPrefill(
            const float *input,
            float *output,
            size_t batch_size,
            size_t seq_len)
        {
            auto start = std::chrono::high_resolution_clock::now();

            const size_t d_model = params_.d_model;
            const size_t d_inner = params_.d_inner;
            const size_t d_state = params_.d_state;

            // Allocate work buffers
            size_t batch_seq = batch_size * seq_len;
            x_proj_.resize(batch_seq * d_inner);
            z_proj_.resize(batch_seq * d_inner);
            B_sel_.resize(batch_seq * d_state);
            C_sel_.resize(batch_seq * d_state);
            delta_.resize(batch_seq * d_inner);
            conv_out_.resize(batch_seq * d_inner);
            ssm_out_.resize(batch_seq * d_inner);

            auto t1 = std::chrono::high_resolution_clock::now();

            // Step 1: Linear projections (x and z for gating)
            MatMul(input, params_.W_x.data(), x_proj_.data(),
                   batch_seq, d_inner, d_model);
            MatMul(input, params_.W_z.data(), z_proj_.data(),
                   batch_seq, d_inner, d_model);

            auto t2 = std::chrono::high_resolution_clock::now();

            // Step 2: 1D Convolution
            std::vector<float> conv_state(batch_size * d_inner * (params_.d_conv - 1), 0.0f);
            ApplyConv1D(x_proj_.data(), conv_out_.data(), conv_state.data(),
                        batch_size, seq_len);

            auto t3 = std::chrono::high_resolution_clock::now();

            // Step 3: Compute selective parameters (B, C, delta)
            ComputeSelectiveParameters(conv_out_.data(), B_sel_.data(), C_sel_.data(),
                                       delta_.data(), batch_size, seq_len);

            // Step 4: SSM step with parallel scan
            SSMStep(conv_out_.data(), B_sel_.data(), C_sel_.data(), delta_.data(),
                    nullptr, ssm_out_.data(), batch_size, seq_len);

            auto t4 = std::chrono::high_resolution_clock::now();

            // Step 5: Gated activation (element-wise multiply with SiLU(z))
            SiLU(z_proj_.data(), z_proj_.size());
            for (size_t i = 0; i < ssm_out_.size(); ++i)
            {
                ssm_out_[i] *= z_proj_[i];
            }

            // Step 6: Output projection
            MatMul(ssm_out_.data(), params_.W_out.data(), output,
                   batch_seq, d_model, d_inner);

            auto end = std::chrono::high_resolution_clock::now();

            // Update statistics
            stats_.total_forward_calls++;
            stats_.total_tokens_processed += batch_seq;

            double proj_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double conv_time = std::chrono::duration<double, std::milli>(t3 - t2).count();
            double ssm_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
            double total_time = std::chrono::duration<double, std::milli>(end - start).count();

            stats_.proj_time_ms += proj_time;
            stats_.conv_time_ms += conv_time;
            stats_.ssm_time_ms += ssm_time;
            stats_.total_time_ms += total_time;

            g_ssm_stats = stats_;
        }

        void SelectiveSSM::ForwardGeneration(
            const float *input,
            float *output,
            size_t batch_size,
            SSMState &state)
        {
            auto start = std::chrono::high_resolution_clock::now();

            const size_t d_model = params_.d_model;
            const size_t d_inner = params_.d_inner;
            const size_t d_state = params_.d_state;

            // Allocate work buffers for single token
            x_proj_.resize(batch_size * d_inner);
            z_proj_.resize(batch_size * d_inner);
            B_sel_.resize(batch_size * d_state);
            C_sel_.resize(batch_size * d_state);
            delta_.resize(batch_size * d_inner);
            conv_out_.resize(batch_size * d_inner);
            ssm_out_.resize(batch_size * d_inner);

            auto t1 = std::chrono::high_resolution_clock::now();

            // Step 1: Linear projections
            MatMul(input, params_.W_x.data(), x_proj_.data(),
                   batch_size, d_inner, d_model);
            MatMul(input, params_.W_z.data(), z_proj_.data(),
                   batch_size, d_inner, d_model);

            auto t2 = std::chrono::high_resolution_clock::now();

            // Step 2: 1D Convolution (recurrent mode - single step)
            ApplyConv1D(x_proj_.data(), conv_out_.data(), state.conv_state.data(),
                        batch_size, 1);

            auto t3 = std::chrono::high_resolution_clock::now();

            // Step 3: Compute selective parameters
            ComputeSelectiveParameters(conv_out_.data(), B_sel_.data(), C_sel_.data(),
                                       delta_.data(), batch_size, 1);

            // Step 4: SSM step (recurrent mode)
            SSMStepRecurrent(conv_out_.data(), B_sel_.data(), C_sel_.data(), delta_.data(),
                             state.h.data(), ssm_out_.data(), batch_size);

            auto t4 = std::chrono::high_resolution_clock::now();

            // Step 5: Gated activation
            SiLU(z_proj_.data(), z_proj_.size());
            for (size_t i = 0; i < ssm_out_.size(); ++i)
            {
                ssm_out_[i] *= z_proj_[i];
            }

            // Step 6: Output projection
            MatMul(ssm_out_.data(), params_.W_out.data(), output,
                   batch_size, d_model, d_inner);

            auto end = std::chrono::high_resolution_clock::now();

            state.seq_pos++;

            // Update statistics
            stats_.total_forward_calls++;
            stats_.total_tokens_processed += batch_size;

            double proj_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double conv_time = std::chrono::duration<double, std::milli>(t3 - t2).count();
            double ssm_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
            double total_time = std::chrono::duration<double, std::milli>(end - start).count();

            stats_.proj_time_ms += proj_time;
            stats_.conv_time_ms += conv_time;
            stats_.ssm_time_ms += ssm_time;
            stats_.total_time_ms += total_time;

            g_ssm_stats = stats_;
        }

        void SelectiveSSM::ApplyConv1D(
            const float *input,
            float *output,
            float *conv_state,
            size_t batch_size,
            size_t seq_len)
        {
            const size_t d_inner = params_.d_inner;
            const size_t d_conv = params_.d_conv;

            for (size_t b = 0; b < batch_size; ++b)
            {
                for (size_t i = 0; i < d_inner; ++i)
                {
                    for (size_t t = 0; t < seq_len; ++t)
                    {
                        float sum = params_.conv_bias[i];

                        // Shift conv state and add new input
                        for (size_t k = 0; k < d_conv - 1; ++k)
                        {
                            size_t state_idx = b * d_inner * (d_conv - 1) + i * (d_conv - 1) + k;
                            float state_val = (k == 0) ? input[b * seq_len * d_inner + t * d_inner + i]
                                                       : conv_state[state_idx - 1];
                            if (k < d_conv - 1)
                            {
                                conv_state[state_idx] = state_val;
                            }
                            sum += params_.conv_weight[i * d_conv + k] * state_val;
                        }

                        // Last weight is for current input
                        sum += params_.conv_weight[i * d_conv + (d_conv - 1)] *
                               input[b * seq_len * d_inner + t * d_inner + i];

                        output[b * seq_len * d_inner + t * d_inner + i] = sum;
                    }
                }
            }
        }

        void SelectiveSSM::ComputeSelectiveParameters(
            const float *x,
            float *B,
            float *C,
            float *delta,
            size_t batch_size,
            size_t seq_len)
        {
            const size_t d_inner = params_.d_inner;
            const size_t d_state = params_.d_state;
            size_t batch_seq = batch_size * seq_len;

            // B = softplus(Linear(x))
            MatMul(x, params_.W_B.data(), B, batch_seq, d_state, d_inner);

            // C = Linear(x)
            MatMul(x, params_.W_C.data(), C, batch_seq, d_state, d_inner);

            // delta = softplus(Linear(x)) + bias
            for (size_t i = 0; i < batch_seq; ++i)
            {
                for (size_t j = 0; j < d_inner; ++j)
                {
                    float sum = 0.0f;
                    for (size_t k = 0; k < d_inner; ++k)
                    {
                        sum += x[i * d_inner + k] * params_.W_dt[k];
                    }
                    // Softplus(x) = log(1 + exp(x))
                    delta[i * d_inner + j] = std::log1p(std::exp(sum));
                    // Clamp to reasonable range
                    delta[i * d_inner + j] = std::max(config_.dt_min,
                                                      std::min(config_.dt_max, delta[i * d_inner + j]));
                }
            }
        }

        void SelectiveSSM::SSMStep(
            const float *x,
            const float *B,
            const float *C,
            const float *delta,
            float *h,
            float *y,
            size_t batch_size,
            size_t seq_len)
        {
            (void)h; // Suppress unused parameter warning
            const size_t d_inner = params_.d_inner;
            const size_t d_state = params_.d_state;
            size_t batch_seq = batch_size * seq_len;
            (void)h; // parameter intentionally unused: using internal h_states buffer

            // Discretize continuous parameters
            A_bar_.resize(batch_seq * d_inner * d_state);
            B_bar_.resize(batch_seq * d_inner * d_state);

            Discretize(delta, A_bar_.data(), B_bar_.data(), B, batch_size, seq_len);

            // Use parallel scan for prefill
            std::vector<float> h_states(batch_seq * d_inner * d_state, 0.0f);

            parallel_scan_->ScanSSM(A_bar_.data(), B_bar_.data(), x, h_states.data(),
                                    batch_size, seq_len, d_inner, d_state);

            // Compute output: y = C * h + D * x
            for (size_t b = 0; b < batch_size; ++b)
            {
                for (size_t t = 0; t < seq_len; ++t)
                {
                    for (size_t i = 0; i < d_inner; ++i)
                    {
                        float sum = params_.D[i] * x[b * seq_len * d_inner + t * d_inner + i];

                        for (size_t j = 0; j < d_state; ++j)
                        {
                            sum += C[b * seq_len * d_state + t * d_state + j] *
                                   h_states[b * seq_len * d_inner * d_state +
                                            t * d_inner * d_state + i * d_state + j];
                        }

                        y[b * seq_len * d_inner + t * d_inner + i] = sum;
                    }
                }
            }
        }

        void SelectiveSSM::SSMStepRecurrent(
            const float *x,
            const float *B,
            const float *C,
            const float *delta,
            float *h,
            float *y,
            size_t batch_size)
        {
            const size_t d_inner = params_.d_inner;
            const size_t d_state = params_.d_state;

            // Discretize for single step
            std::vector<float> A_bar(d_inner * d_state);
            std::vector<float> B_bar(d_inner * d_state);

            for (size_t i = 0; i < d_inner; ++i)
            {
                for (size_t j = 0; j < d_state; ++j)
                {
                    float A_val = std::exp(params_.A_log[i * d_state + j]);
                    A_bar[i * d_state + j] = std::exp(delta[i] * A_val);
                    B_bar[i * d_state + j] = delta[i] * B[j];
                }
            }

            // State update: h[t] = A_bar * h[t-1] + B_bar * x[t]
            for (size_t b = 0; b < batch_size; ++b)
            {
                for (size_t i = 0; i < d_inner; ++i)
                {
                    for (size_t j = 0; j < d_state; ++j)
                    {
                        size_t h_idx = b * d_inner * d_state + i * d_state + j;
                        h[h_idx] = A_bar[i * d_state + j] * h[h_idx] +
                                   B_bar[i * d_state + j] * x[b * d_inner + i];
                    }
                }
            }

            // Output: y = C * h + D * x
            for (size_t b = 0; b < batch_size; ++b)
            {
                for (size_t i = 0; i < d_inner; ++i)
                {
                    float sum = params_.D[i] * x[b * d_inner + i];

                    for (size_t j = 0; j < d_state; ++j)
                    {
                        sum += C[b * d_state + j] * h[b * d_inner * d_state + i * d_state + j];
                    }

                    y[b * d_inner + i] = sum;
                }
            }
        }

        void SelectiveSSM::Discretize(
            const float *delta,
            float *A_bar,
            float *B_bar,
            const float *B,
            size_t batch_size,
            size_t seq_len)
        {
            const size_t d_inner = params_.d_inner;
            const size_t d_state = params_.d_state;

            for (size_t b = 0; b < batch_size; ++b)
            {
                for (size_t t = 0; t < seq_len; ++t)
                {
                    for (size_t i = 0; i < d_inner; ++i)
                    {
                        float dt = delta[b * seq_len * d_inner + t * d_inner + i];

                        for (size_t j = 0; j < d_state; ++j)
                        {
                            // A is stored in log scale for numerical stability
                            float A_val = std::exp(params_.A_log[i * d_state + j]);

                            // Zero-order hold discretization
                            A_bar[b * seq_len * d_inner * d_state + t * d_inner * d_state +
                                  i * d_state + j] = std::exp(dt * A_val);
                            B_bar[b * seq_len * d_inner * d_state + t * d_inner * d_state +
                                  i * d_state + j] = dt * B[b * seq_len * d_state + t * d_state + j];
                        }
                    }
                }
            }
        }

        void SelectiveSSM::MatMul(
            const float *A,
            const float *B,
            float *C,
            size_t M,
            size_t N,
            size_t K,
            bool transpose_B)
        {
#ifdef __AVX512F__
            if (config_.use_avx512 && N >= 16)
            {
                MatMulAVX512(A, B, C, M, N, K);
                return;
            }
#endif

            // Naive implementation
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k)
                    {
                        float b_val = transpose_B ? B[j * K + k] : B[k * N + j];
                        sum += A[i * K + k] * b_val;
                    }
                    C[i * N + j] = sum;
                }
            }
        }

        void SelectiveSSM::SiLU(float *x, size_t size)
        {
            for (size_t i = 0; i < size; ++i)
            {
                x[i] = x[i] / (1.0f + std::exp(-x[i]));
            }
        }

#ifdef __AVX512F__
        void SelectiveSSM::MatMulAVX512(
            const float *A,
            const float *B,
            float *C,
            size_t M,
            size_t N,
            size_t K)
        {
            // Process 16 columns at a time
            for (size_t i = 0; i < M; ++i)
            {
                size_t j = 0;
                for (; j + 16 <= N; j += 16)
                {
                    __m512 acc = _mm512_setzero_ps();

                    for (size_t k = 0; k < K; ++k)
                    {
                        __m512 a_val = _mm512_set1_ps(A[i * K + k]);
                        __m512 b_vals = _mm512_loadu_ps(&B[k * N + j]);
                        acc = _mm512_fmadd_ps(a_val, b_vals, acc);
                    }

                    _mm512_storeu_ps(&C[i * N + j], acc);
                }

                // Handle remaining columns
                for (; j < N; ++j)
                {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k)
                    {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
#else
        void SelectiveSSM::MatMulAVX512(
            const float *A,
            const float *B,
            float *C,
            size_t M,
            size_t N,
            size_t K)
        {
            // Fallback to naive implementation
            MatMul(A, B, C, M, N, K);
        }
#endif

    } // namespace mamba
} // namespace ryzanstein_llm
