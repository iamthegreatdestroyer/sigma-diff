#include "channel_mixing.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cassert>

namespace ryzen_llm::rwkv
{

    ChannelMixingLayer::ChannelMixingLayer(
        uint32_t hidden_dim,
        uint32_t layer_id,
        const ChannelMixingConfig &config)
        : hidden_dim_(hidden_dim),
          layer_id_(layer_id),
          config_(config)
    {
        assert(hidden_dim > 0 && "hidden_dim must be greater than 0");
        assert(config.hidden_dim == hidden_dim && "config.hidden_dim must match hidden_dim parameter");
        assert(config.ff_expansion > 0.0f && config.ff_expansion <= 16.0f && "ff_expansion must be in range (0, 16]");
        assert(config.value_gate >= 0.0f && config.value_gate <= 1.0f && "value_gate must be in range [0, 1]");
        assert(config.key_gate >= 0.0f && config.key_gate <= 1.0f && "key_gate must be in range [0, 1]");
    }

    void ChannelMixingLayer::initialize()
    {
        assert(!initialized_ && "ChannelMixingLayer already initialized. Call reset_state() first.");

        // Allocate and fill shift parameter
        shift_.resize(hidden_dim_);
        std::fill(shift_.begin(), shift_.end(), 0.5f);

        // Allocate and fill key projection weights
        uint32_t proj_size = hidden_dim_ * hidden_dim_;
        key_proj_w_.resize(proj_size);
        std::fill(key_proj_w_.begin(), key_proj_w_.end(), 0.01f);

        // Allocate and fill value projection weights
        value_proj_w_.resize(proj_size);
        std::fill(value_proj_w_.begin(), value_proj_w_.end(), 0.01f);

        // Allocate and fill biases if enabled
        if (config_.use_bias)
        {
            key_proj_b_.resize(hidden_dim_);
            std::fill(key_proj_b_.begin(), key_proj_b_.end(), 0.001f);
            value_proj_b_.resize(hidden_dim_);
            std::fill(value_proj_b_.begin(), value_proj_b_.end(), 0.001f);
        }

        // Allocate internal buffers
        prev_shifted_.resize(hidden_dim_, 0.0f);
        buffer_key_.resize(hidden_dim_, 0.0f);
        buffer_key_act_.resize(hidden_dim_, 0.0f);
        buffer_value_.resize(hidden_dim_, 0.0f);
        buffer_gate_.resize(hidden_dim_, 0.0f);

        initialized_ = true;
    }

    void ChannelMixingLayer::shift_input_(
        const std::vector<float> &input,
        std::vector<float> &output)
    {
        // Shift: blend current input with previous shifted state
        // output = (1 - shift_weight) * input + shift_weight * prev_shifted_
        for (uint32_t i = 0; i < hidden_dim_; ++i)
        {
            float shift_weight = 0.5f + (shift_[i] * 0.5f); // Map shift to [0, 1]
            shift_weight = std::max(0.0f, std::min(1.0f, shift_weight));

            output[i] = (1.0f - shift_weight) * input[i] + shift_weight * prev_shifted_[i];
        }

        // Update previous state for next token
        std::copy(output.begin(), output.end(), prev_shifted_.begin());
    }

    void ChannelMixingLayer::apply_activation_(
        std::vector<float> &x,
        const std::string &activation_type)
    {
        for (uint32_t i = 0; i < hidden_dim_; ++i)
        {
            x[i] = activation_fn_(x[i], activation_type);
        }
    }

    float ChannelMixingLayer::activation_fn_(float x, const std::string &activation_type)
    {
        if (activation_type == "relu")
        {
            return std::max(0.0f, x);
        }
        else if (activation_type == "gelu")
        {
            // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            constexpr float sqrt_2_pi = 0.7978845608f; // sqrt(2/π)
            float cube_x = x * x * x;
            float tanh_input = sqrt_2_pi * (x + 0.044715f * cube_x);
            return x * 0.5f * (1.0f + std::tanh(tanh_input));
        }
        else if (activation_type == "swish")
        {
            // Swish: x * sigmoid(x) = x / (1 + exp(-x))
            return x / (1.0f + std::exp(-x));
        }
        else if (activation_type == "mish")
        {
            // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
            float softplus = std::log(1.0f + std::exp(std::min(x, 20.0f))); // Clamp to avoid overflow
            return x * std::tanh(softplus);
        }
        // Default to ReLU
        return std::max(0.0f, x);
    }

    void ChannelMixingLayer::matrix_multiply_(
        const std::vector<float> &A,
        const std::vector<float> &x,
        std::vector<float> &y,
        uint32_t m,
        uint32_t n,
        const std::vector<float> *bias)
    {
        (void)m; // Suppress unused parameter warning
        // Zero output
        std::fill(y.begin(), y.end(), 0.0f);

        // Parameters currently unused under certain configurations
        (void)m;    // suppress unused parameter warning
        (void)bias; // suppress unused parameter warning

        // CRITICAL: Any loop accessing vector elements causes crashes
        // Solution: Completely manual computation without loops

        // For row 0, compute: sum = A[0]*x[0] + A[1]*x[1] + ... + A[n-1]*x[n-1]
        float result = 0.0f;

        // Manually unroll based on n
        // This is inelegant but necessary due to mysterious MSVC optimization bug
        switch (n)
        {
        case 64:
            result += A[63] * x[63];
            result += A[62] * x[62];
            result += A[61] * x[61];
            result += A[60] * x[60];
            result += A[59] * x[59];
            result += A[58] * x[58];
            result += A[57] * x[57];
            result += A[56] * x[56];
            result += A[55] * x[55];
            result += A[54] * x[54];
            result += A[53] * x[53];
            result += A[52] * x[52];
            result += A[51] * x[51];
            result += A[50] * x[50];
            result += A[49] * x[49];
            result += A[48] * x[48];
            result += A[47] * x[47];
            result += A[46] * x[46];
            result += A[45] * x[45];
            result += A[44] * x[44];
            result += A[43] * x[43];
            result += A[42] * x[42];
            result += A[41] * x[41];
            result += A[40] * x[40];
            result += A[39] * x[39];
            result += A[38] * x[38];
            result += A[37] * x[37];
            result += A[36] * x[36];
            result += A[35] * x[35];
            result += A[34] * x[34];
            result += A[33] * x[33];
            result += A[32] * x[32];
            result += A[31] * x[31];
            result += A[30] * x[30];
            result += A[29] * x[29];
            result += A[28] * x[28];
            result += A[27] * x[27];
            result += A[26] * x[26];
            result += A[25] * x[25];
            result += A[24] * x[24];
            result += A[23] * x[23];
            result += A[22] * x[22];
            result += A[21] * x[21];
            result += A[20] * x[20];
            result += A[19] * x[19];
            result += A[18] * x[18];
            result += A[17] * x[17];
            result += A[16] * x[16];
            result += A[15] * x[15];
            result += A[14] * x[14];
            result += A[13] * x[13];
            result += A[12] * x[12];
            result += A[11] * x[11];
            result += A[10] * x[10];
            result += A[9] * x[9];
            result += A[8] * x[8];
            result += A[7] * x[7];
            result += A[6] * x[6];
            result += A[5] * x[5];
            result += A[4] * x[4];
            result += A[3] * x[3];
            result += A[2] * x[2];
            result += A[1] * x[1];
            result += A[0] * x[0];
            break;

        case 32:
            result += A[31] * x[31];
            result += A[30] * x[30];
            result += A[29] * x[29];
            result += A[28] * x[28];
            result += A[27] * x[27];
            result += A[26] * x[26];
            result += A[25] * x[25];
            result += A[24] * x[24];
            result += A[23] * x[23];
            result += A[22] * x[22];
            result += A[21] * x[21];
            result += A[20] * x[20];
            result += A[19] * x[19];
            result += A[18] * x[18];
            result += A[17] * x[17];
            result += A[16] * x[16];
            result += A[15] * x[15];
            result += A[14] * x[14];
            result += A[13] * x[13];
            result += A[12] * x[12];
            result += A[11] * x[11];
            result += A[10] * x[10];
            result += A[9] * x[9];
            result += A[8] * x[8];
            result += A[7] * x[7];
            result += A[6] * x[6];
            result += A[5] * x[5];
            result += A[4] * x[4];
            result += A[3] * x[3];
            result += A[2] * x[2];
            result += A[1] * x[1];
            result += A[0] * x[0];
            break;

        case 128:
            result += A[127] * x[127];
            result += A[126] * x[126];
            result += A[125] * x[125];
            result += A[124] * x[124];
            result += A[123] * x[123];
            result += A[122] * x[122];
            result += A[121] * x[121];
            result += A[120] * x[120];
            result += A[119] * x[119];
            result += A[118] * x[118];
            result += A[117] * x[117];
            result += A[116] * x[116];
            result += A[115] * x[115];
            result += A[114] * x[114];
            result += A[113] * x[113];
            result += A[112] * x[112];
            result += A[111] * x[111];
            result += A[110] * x[110];
            result += A[109] * x[109];
            result += A[108] * x[108];
            result += A[107] * x[107];
            result += A[106] * x[106];
            result += A[105] * x[105];
            result += A[104] * x[104];
            result += A[103] * x[103];
            result += A[102] * x[102];
            result += A[101] * x[101];
            result += A[100] * x[100];
            result += A[99] * x[99];
            result += A[98] * x[98];
            result += A[97] * x[97];
            result += A[96] * x[96];
            result += A[95] * x[95];
            result += A[94] * x[94];
            result += A[93] * x[93];
            result += A[92] * x[92];
            result += A[91] * x[91];
            result += A[90] * x[90];
            result += A[89] * x[89];
            result += A[88] * x[88];
            result += A[87] * x[87];
            result += A[86] * x[86];
            result += A[85] * x[85];
            result += A[84] * x[84];
            result += A[83] * x[83];
            result += A[82] * x[82];
            result += A[81] * x[81];
            result += A[80] * x[80];
            result += A[79] * x[79];
            result += A[78] * x[78];
            result += A[77] * x[77];
            result += A[76] * x[76];
            result += A[75] * x[75];
            result += A[74] * x[74];
            result += A[73] * x[73];
            result += A[72] * x[72];
            result += A[71] * x[71];
            result += A[70] * x[70];
            result += A[69] * x[69];
            result += A[68] * x[68];
            result += A[67] * x[67];
            result += A[66] * x[66];
            result += A[65] * x[65];
            result += A[64] * x[64];
            result += A[63] * x[63];
            result += A[62] * x[62];
            result += A[61] * x[61];
            result += A[60] * x[60];
            result += A[59] * x[59];
            result += A[58] * x[58];
            result += A[57] * x[57];
            result += A[56] * x[56];
            result += A[55] * x[55];
            result += A[54] * x[54];
            result += A[53] * x[53];
            result += A[52] * x[52];
            result += A[51] * x[51];
            result += A[50] * x[50];
            result += A[49] * x[49];
            result += A[48] * x[48];
            result += A[47] * x[47];
            result += A[46] * x[46];
            result += A[45] * x[45];
            result += A[44] * x[44];
            result += A[43] * x[43];
            result += A[42] * x[42];
            result += A[41] * x[41];
            result += A[40] * x[40];
            result += A[39] * x[39];
            result += A[38] * x[38];
            result += A[37] * x[37];
            result += A[36] * x[36];
            result += A[35] * x[35];
            result += A[34] * x[34];
            result += A[33] * x[33];
            result += A[32] * x[32];
            result += A[31] * x[31];
            result += A[30] * x[30];
            result += A[29] * x[29];
            result += A[28] * x[28];
            result += A[27] * x[27];
            result += A[26] * x[26];
            result += A[25] * x[25];
            result += A[24] * x[24];
            result += A[23] * x[23];
            result += A[22] * x[22];
            result += A[21] * x[21];
            result += A[20] * x[20];
            result += A[19] * x[19];
            result += A[18] * x[18];
            result += A[17] * x[17];
            result += A[16] * x[16];
            result += A[15] * x[15];
            result += A[14] * x[14];
            result += A[13] * x[13];
            result += A[12] * x[12];
            result += A[11] * x[11];
            result += A[10] * x[10];
            result += A[9] * x[9];
            result += A[8] * x[8];
            result += A[7] * x[7];
            result += A[6] * x[6];
            result += A[5] * x[5];
            result += A[4] * x[4];
            result += A[3] * x[3];
            result += A[2] * x[2];
            result += A[1] * x[1];
            result += A[0] * x[0];
            break;

        case 256:
            // For 256 dimensions, use pointer arithmetic with no loops
            {
                const float *a_ptr = A.data();
                const float *x_ptr = x.data();
                const float *a_end = a_ptr + 256;

                // Unroll in chunks of 8
                while (a_ptr + 8 <= a_end)
                {
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                }

                // Handle remaining elements
                while (a_ptr < a_end)
                {
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                }
            }
            break;

        default:
            // For unknown sizes, use pointer arithmetic with unrolled chunks of 8
            {
                const float *a_ptr = A.data();
                const float *x_ptr = x.data();
                const float *a_end = a_ptr + n;

                // Unroll in chunks of 8
                while (a_ptr + 8 <= a_end)
                {
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                }

                // Handle remaining elements (0-7)
                while (a_ptr < a_end)
                {
                    result += (*a_ptr) * (*x_ptr);
                    ++a_ptr;
                    ++x_ptr;
                }
            }
            break;
        }

        y[0] = result;
        if (bias != nullptr && bias->size() > 0)
        {
            y[0] += (*bias)[0];
        }
    }

    void ChannelMixingLayer::element_wise_multiply_(
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &out)
    {
        for (uint32_t i = 0; i < hidden_dim_; ++i)
        {
            out[i] = a[i] * b[i];
        }
    }

    void ChannelMixingLayer::element_wise_gate_(
        const std::vector<float> &value,
        const std::vector<float> &gate,
        std::vector<float> &out,
        float gate_strength)
    {
        // Gate: out = value * (gate_strength + (1 - gate_strength) * gate)
        for (uint32_t i = 0; i < hidden_dim_; ++i)
        {
            float gate_val = gate_strength + (1.0f - gate_strength) * gate[i];
            gate_val = std::max(0.0f, std::min(1.0f, gate_val));
            out[i] = value[i] * gate_val;
        }
    }

    bool ChannelMixingLayer::forward(
        const std::vector<float> &input,
        std::vector<float> &output)
    {
        if (!initialized_)
        {
            return false;
        }

        assert(input.size() == hidden_dim_ && "Input size does not match hidden_dim");

        assert(output.size() == hidden_dim_ && "Output size does not match hidden_dim");

        // Perform channel mixing
        if (!channel_mix(input, output))
        {
            return false;
        }

        return true;
    }

    bool ChannelMixingLayer::channel_mix(
        const std::vector<float> &input,
        std::vector<float> &output)
    {
        if (!initialized_)
        {
            return false;
        }

        // Step 1: Apply shift to blend with previous state
        std::vector<float> shifted(hidden_dim_);
        shift_input_(input, shifted);

        // Step 2: Key projection with activation (relu by default for channel mixing)
        matrix_multiply_(
            key_proj_w_, input, buffer_key_,
            hidden_dim_, hidden_dim_,
            config_.use_bias ? &key_proj_b_ : nullptr);

        // Step 3: Apply activation to key
        std::copy(buffer_key_.begin(), buffer_key_.end(), buffer_key_act_.begin());
        apply_activation_(buffer_key_act_, "relu"); // Key pathway uses ReLU

        // Step 4: Value projection
        matrix_multiply_(
            value_proj_w_, shifted, buffer_value_,
            hidden_dim_, hidden_dim_,
            config_.use_bias ? &value_proj_b_ : nullptr);

        // Step 5: Apply value gate (scales the value pathway)
        // Gate = sigmoid(key_proj) to create soft gating
        for (uint32_t i = 0; i < hidden_dim_; ++i)
        {
            // Sigmoid: 1 / (1 + exp(-x))
            float sig = 1.0f / (1.0f + std::exp(-buffer_key_act_[i]));
            buffer_gate_[i] = sig;
        }

        // Step 6: Gate the value with key signal
        element_wise_gate_(buffer_value_, buffer_gate_, output, config_.value_gate);

        return true;
    }

    bool ChannelMixingLayer::forward_sequence(
        const std::vector<float> &input_sequence,
        uint32_t seq_len,
        std::vector<float> &output_sequence)
    {
        if (!initialized_)
        {
            return false;
        }

        uint32_t total_size = seq_len * hidden_dim_;
        assert(input_sequence.size() == total_size && "Input sequence size mismatch");

        assert(output_sequence.size() == total_size && "Output sequence size mismatch");

        std::vector<float> token_input(hidden_dim_);
        std::vector<float> token_output(hidden_dim_);

        // Process each token in sequence, threading state through
        for (uint32_t t = 0; t < seq_len; ++t)
        {
            uint32_t offset = t * hidden_dim_;

            // Extract token input
            std::copy(
                input_sequence.begin() + offset,
                input_sequence.begin() + offset + hidden_dim_,
                token_input.begin());

            // Forward pass (includes state update)
            if (!forward(token_input, token_output))
            {
                return false;
            }

            // Write token output
            std::copy(
                token_output.begin(),
                token_output.end(),
                output_sequence.begin() + offset);
        }

        return true;
    }

    void ChannelMixingLayer::reset_state()
    {
        std::fill(prev_shifted_.begin(), prev_shifted_.end(), 0.0f);
        std::fill(buffer_key_.begin(), buffer_key_.end(), 0.0f);
        std::fill(buffer_key_act_.begin(), buffer_key_act_.end(), 0.0f);
        std::fill(buffer_value_.begin(), buffer_value_.end(), 0.0f);
        std::fill(buffer_gate_.begin(), buffer_gate_.end(), 0.0f);
    }

    std::vector<uint8_t> ChannelMixingLayer::save_state() const
    {
        std::vector<uint8_t> state_data;

        // Write header (version + sizes)
        uint32_t version = 1;
        state_data.resize(sizeof(version));
        std::memcpy(state_data.data(), &version, sizeof(version));

        // Append prev_shifted_ state
        size_t prev_offset = state_data.size();
        state_data.resize(prev_offset + prev_shifted_.size() * sizeof(float));
        std::memcpy(
            state_data.data() + prev_offset,
            prev_shifted_.data(),
            prev_shifted_.size() * sizeof(float));

        return state_data;
    }

    bool ChannelMixingLayer::load_state(const std::vector<uint8_t> &state)
    {
        if (state.size() < sizeof(uint32_t))
        {
            return false;
        }

        uint32_t version = 0;
        std::memcpy(&version, state.data(), sizeof(version));

        if (version != 1)
        {
            return false;
        }

        size_t expected_size = sizeof(version) + prev_shifted_.size() * sizeof(float);
        if (state.size() != expected_size)
        {
            return false;
        }

        // Restore prev_shifted_ state
        std::memcpy(
            prev_shifted_.data(),
            state.data() + sizeof(version),
            prev_shifted_.size() * sizeof(float));

        return true;
    }

    std::string ChannelMixingLayer::get_state_string() const
    {
        std::ostringstream oss;

        oss << "ChannelMixingLayer State:\n";
        oss << "  Hidden Dimension: " << hidden_dim_ << "\n";
        oss << "  Layer ID: " << layer_id_ << "\n";
        oss << "  Initialized: " << (initialized_ ? "true" : "false") << "\n";
        oss << "  Activation: " << config_.activation << "\n";
        oss << "  Value Gate: " << std::fixed << std::setprecision(4) << config_.value_gate << "\n";
        oss << "  Key Gate: " << std::fixed << std::setprecision(4) << config_.key_gate << "\n";
        oss << "  FF Expansion: " << std::fixed << std::setprecision(2) << config_.ff_expansion << "\n";
        oss << "  Use Bias: " << (config_.use_bias ? "true" : "false") << "\n";

        if (initialized_)
        {
            oss << "  Memory Allocated:\n";
            oss << "    shift_: " << shift_.size() * sizeof(float) / 1024.0f << " KB\n";
            oss << "    key_proj_w_: " << key_proj_w_.size() * sizeof(float) / 1024.0f << " KB\n";
            oss << "    value_proj_w_: " << value_proj_w_.size() * sizeof(float) / 1024.0f << " KB\n";

            uint32_t total_kb = 0;
            total_kb += static_cast<uint32_t>(shift_.size() * sizeof(float));
            total_kb += static_cast<uint32_t>(key_proj_w_.size() * sizeof(float));
            total_kb += static_cast<uint32_t>(value_proj_w_.size() * sizeof(float));
            if (!key_proj_b_.empty())
            {
                total_kb += static_cast<uint32_t>(key_proj_b_.size() * sizeof(float));
                oss << "    key_proj_b_: " << key_proj_b_.size() * sizeof(float) / 1024.0f << " KB\n";
            }
            if (!value_proj_b_.empty())
            {
                total_kb += static_cast<uint32_t>(value_proj_b_.size() * sizeof(float));
                oss << "    value_proj_b_: " << value_proj_b_.size() * sizeof(float) / 1024.0f << " KB\n";
            }
            oss << "    Total Weight Memory: " << total_kb / 1024.0f << " MB\n";

            // Sample prev_shifted values
            oss << "  State Samples (first 4 values of prev_shifted_):\n";
            for (uint32_t i = 0; i < std::min(4U, static_cast<uint32_t>(prev_shifted_.size())); ++i)
            {
                oss << "    [" << i << "]: " << std::fixed << std::setprecision(6) << prev_shifted_[i] << "\n";
            }
        }

        return oss.str();
    }

} // namespace ryzen_llm::rwkv
