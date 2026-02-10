/**
 * @file bitnet_model.cpp
 * @brief Implementation of complete BitNet model
 *
 * [REF:BITNET-002] - Model Implementation
 */

#include "bitnet_model.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>

namespace ryzen_llm
{
    namespace bitnet
    {

        using namespace std::chrono;

        // ============================================================================
        // CONSTRUCTOR
        // ============================================================================

        BitNetModel::BitNetModel(
            const ModelConfig &config,
            const ModelWeights &weights,
            std::shared_ptr<tmac::TMACGemmOptimized> gemm_engine)
            : config_(config), weights_(weights), gemm_engine_(std::move(gemm_engine))
        {
            std::cout << "Initializing BitNet Model: " << config_.name << "\n";
            std::cout << "  Layers: " << config_.num_layers << "\n";
            std::cout << "  Hidden dim: " << config_.hidden_dim << "\n";
            std::cout << "  Vocab size: " << config_.vocab_size << "\n";

            // Initialize transformer layers
            layers_.reserve(config_.num_layers);
            for (uint32_t i = 0; i < config_.num_layers; ++i)
            {
                layers_.push_back(std::make_unique<BitNetLayer>(
                    weights_.layers[i],
                    gemm_engine_));
            }

            // Allocate working buffers
            size_t max_hidden_size = config_.max_seq_len * config_.hidden_dim;
            hidden_states_.resize(max_hidden_size);
            layer_output_.resize(max_hidden_size);

            std::cout << "Model initialized successfully!\n\n";
        }

        // ============================================================================
        // GENERATION
        // ============================================================================

        std::vector<uint32_t> BitNetModel::generate(
            const std::vector<uint32_t> &prompt,
            const GenerationConfig &gen_config)
        {
            auto start_time = high_resolution_clock::now();

            std::vector<uint32_t> tokens = prompt;

            std::cout << "Generating " << gen_config.max_new_tokens << " tokens...\n";
            std::cout << "  Temperature: " << gen_config.temperature << "\n";
            std::cout << "  Top-p: " << gen_config.top_p << "\n";
            std::cout << "  Top-k: " << gen_config.top_k << "\n\n";

            // Generate tokens autoregressively
            for (uint32_t step = 0; step < gen_config.max_new_tokens; ++step)
            {
                // Prepare input
                uint32_t seq_len = tokens.size();
                if (seq_len > config_.max_seq_len)
                {
                    std::cerr << "Sequence length exceeds maximum. Truncating.\n";
                    seq_len = config_.max_seq_len;
                }

                // Allocate logits buffer
                std::vector<float> logits(config_.vocab_size);

                // Forward pass (only last token in cache mode)
                uint32_t input_len = gen_config.use_cache ? 1 : seq_len;
                const uint32_t *input_tokens = gen_config.use_cache
                                                   ? &tokens.back()
                                                   : tokens.data();

                std::vector<float> output_logits(input_len * config_.vocab_size);
                forward(input_tokens, output_logits.data(), 1, input_len);

                // Get logits for last position
                const float *last_logits = output_logits.data() + (input_len - 1) * config_.vocab_size;
                std::copy(last_logits, last_logits + config_.vocab_size, logits.data());

                // Sample next token
                uint32_t next_token = sample_token(logits.data(), gen_config);
                tokens.push_back(next_token);

                // Progress indicator
                if ((step + 1) % 10 == 0)
                {
                    std::cout << "  Generated " << (step + 1) << " tokens...\r" << std::flush;
                }
            }

            std::cout << "\n";

            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end_time - start_time).count();

            // Update statistics
            stats_.total_tokens_generated += gen_config.max_new_tokens;
            stats_.total_time_ms += duration;

            double tokens_per_sec = (gen_config.max_new_tokens * 1000.0) / duration;
            std::cout << "Generation complete!\n";
            std::cout << "  Time: " << duration << " ms\n";
            std::cout << "  Speed: " << tokens_per_sec << " tokens/sec\n";

            return tokens;
        }

        // ============================================================================
        // FORWARD PASS
        // ============================================================================

        void BitNetModel::forward(
            const uint32_t *tokens,
            float *logits,
            uint32_t batch_size,
            uint32_t seq_len)
        {
            size_t hidden_size = batch_size * seq_len * config_.hidden_dim;

            // Step 1: Embed tokens
            embed_tokens(tokens, hidden_states_.data(), batch_size, seq_len);

            // Step 2: Add positional encoding
            add_positional_encoding(hidden_states_.data(), seq_len, config_.hidden_dim);

            // Step 3: Pass through transformer layers
            float *current_input = hidden_states_.data();
            float *current_output = layer_output_.data();

            for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx)
            {
                layers_[layer_idx]->forward(
                    current_input,
                    current_output,
                    batch_size,
                    seq_len);

                // Swap buffers for next layer
                std::swap(current_input, current_output);
            }

            // Step 4: Final layer norm
            // (current_input now contains the output from the last layer)
            std::vector<float> normalized(hidden_size);

            for (uint32_t i = 0; i < batch_size * seq_len; ++i)
            {
                const float *in_vec = current_input + i * config_.hidden_dim;
                float *out_vec = normalized.data() + i * config_.hidden_dim;

                // Compute mean
                float mean = 0.0f;
                for (uint32_t j = 0; j < config_.hidden_dim; ++j)
                {
                    mean += in_vec[j];
                }
                mean /= config_.hidden_dim;

                // Compute variance
                float variance = 0.0f;
                for (uint32_t j = 0; j < config_.hidden_dim; ++j)
                {
                    float diff = in_vec[j] - mean;
                    variance += diff * diff;
                }
                variance /= config_.hidden_dim;

                // Normalize
                float inv_std = 1.0f / std::sqrt(variance + weights_.final_ln.eps);
                for (uint32_t j = 0; j < config_.hidden_dim; ++j)
                {
                    float norm_val = (in_vec[j] - mean) * inv_std;
                    out_vec[j] = weights_.final_ln.gamma[j] * norm_val + weights_.final_ln.beta[j];
                }
            }

            // Step 5: Compute logits
            compute_logits(normalized.data(), logits, batch_size, seq_len);
        }

        // ============================================================================
        // EMBEDDING & POSITIONAL ENCODING
        // ============================================================================

        void BitNetModel::embed_tokens(
            const uint32_t *tokens,
            float *output,
            uint32_t batch_size,
            uint32_t seq_len)
        {
            for (uint32_t b = 0; b < batch_size; ++b)
            {
                for (uint32_t s = 0; s < seq_len; ++s)
                {
                    uint32_t token_id = tokens[b * seq_len + s];

                    if (token_id >= config_.vocab_size)
                    {
                        std::cerr << "Warning: Token ID " << token_id
                                  << " exceeds vocab size. Clamping.\n";
                        token_id = config_.vocab_size - 1;
                    }

                    const float *embedding = weights_.token_embedding.data() +
                                             token_id * config_.hidden_dim;
                    float *output_vec = output + (b * seq_len + s) * config_.hidden_dim;

                    std::copy(embedding, embedding + config_.hidden_dim, output_vec);
                }
            }
        }

        void BitNetModel::add_positional_encoding(
            float *hidden_states,
            uint32_t seq_len,
            uint32_t hidden_dim,
            uint32_t start_pos)
        {
            for (uint32_t s = 0; s < seq_len; ++s)
            {
                uint32_t pos = start_pos + s;

                if (pos >= config_.max_seq_len)
                {
                    std::cerr << "Warning: Position " << pos
                              << " exceeds max sequence length.\n";
                    pos = config_.max_seq_len - 1;
                }

                const float *pos_encoding = weights_.position_encoding.data() +
                                            pos * hidden_dim;
                float *hidden_vec = hidden_states + s * hidden_dim;

                for (uint32_t d = 0; d < hidden_dim; ++d)
                {
                    hidden_vec[d] += pos_encoding[d];
                }
            }
        }

        // ============================================================================
        // OUTPUT PROJECTION & LOGITS
        // ============================================================================

        void BitNetModel::compute_logits(
            const float *hidden_states,
            float *logits,
            uint32_t batch_size,
            uint32_t seq_len)
        {
            size_t hidden_size = batch_size * seq_len * config_.hidden_dim;

            // Quantize hidden states
            std::vector<int8_t> hidden_int8(hidden_size);
            float max_abs = 0.0f;
            for (size_t i = 0; i < hidden_size; ++i)
            {
                max_abs = std::max(max_abs, std::abs(hidden_states[i]));
            }
            float scale = max_abs / 127.0f;
            if (scale == 0.0f)
                scale = 1.0f;
            float inv_scale = 1.0f / scale;

            for (size_t i = 0; i < hidden_size; ++i)
            {
                float scaled = hidden_states[i] * inv_scale;
                hidden_int8[i] = static_cast<int8_t>(std::round(std::clamp(scaled, -127.0f, 127.0f)));
            }

            // Compute logits: hidden × W_output^T
            size_t logits_size = batch_size * seq_len * config_.vocab_size;
            std::vector<int32_t> logits_int32(logits_size);

            gemm_engine_->gemm(
                weights_.output_projection.data(),
                hidden_int8.data(),
                logits_int32.data(),
                config_.vocab_size,
                config_.hidden_dim,
                batch_size * seq_len);

            // Dequantize
            for (size_t i = 0; i < logits_size; ++i)
            {
                logits[i] = static_cast<float>(logits_int32[i]) * scale;
            }
        }

        // ============================================================================
        // SAMPLING
        // ============================================================================

        uint32_t BitNetModel::sample_token(
            const float *logits,
            const GenerationConfig &gen_config)
        {
            std::vector<float> logits_copy(config_.vocab_size);
            std::copy(logits, logits + config_.vocab_size, logits_copy.data());

            // Apply temperature
            if (gen_config.temperature != 1.0f)
            {
                apply_temperature(logits_copy.data(), config_.vocab_size, gen_config.temperature);
            }

            // Apply top-k
            if (gen_config.top_k > 0 && gen_config.top_k < config_.vocab_size)
            {
                apply_top_k(logits_copy.data(), config_.vocab_size, gen_config.top_k);
            }

            // Apply top-p
            if (gen_config.top_p < 1.0f)
            {
                apply_top_p(logits_copy.data(), config_.vocab_size, gen_config.top_p);
            }

            // Convert to probabilities
            softmax(logits_copy.data(), config_.vocab_size);

            // Sample from distribution
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<uint32_t> dist(
                logits_copy.begin(),
                logits_copy.end());

            return dist(gen);
        }

        void BitNetModel::apply_temperature(float *logits, uint32_t vocab_size, float temperature)
        {
            float inv_temp = 1.0f / temperature;
            for (uint32_t i = 0; i < vocab_size; ++i)
            {
                logits[i] *= inv_temp;
            }
        }

        void BitNetModel::apply_top_k(float *logits, uint32_t vocab_size, uint32_t k)
        {
            std::vector<std::pair<float, uint32_t>> scored_tokens;
            scored_tokens.reserve(vocab_size);

            for (uint32_t i = 0; i < vocab_size; ++i)
            {
                scored_tokens.push_back({logits[i], i});
            }

            std::partial_sort(
                scored_tokens.begin(),
                scored_tokens.begin() + k,
                scored_tokens.end(),
                [](const auto &a, const auto &b)
                { return a.first > b.first; });

            // Zero out tokens outside top-k
            float min_top_k = scored_tokens[k - 1].first;
            for (uint32_t i = 0; i < vocab_size; ++i)
            {
                if (logits[i] < min_top_k)
                {
                    logits[i] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        void BitNetModel::apply_top_p(float *logits, uint32_t vocab_size, float p)
        {
            std::vector<std::pair<float, uint32_t>> scored_tokens;
            scored_tokens.reserve(vocab_size);

            for (uint32_t i = 0; i < vocab_size; ++i)
            {
                scored_tokens.push_back({logits[i], i});
            }

            std::sort(
                scored_tokens.begin(),
                scored_tokens.end(),
                [](const auto &a, const auto &b)
                { return a.first > b.first; });

            // Compute softmax for cumulative probability
            std::vector<float> probs(vocab_size);
            float max_logit = scored_tokens[0].first;
            float sum = 0.0f;
            for (uint32_t i = 0; i < vocab_size; ++i)
            {
                probs[i] = std::exp(scored_tokens[i].first - max_logit);
                sum += probs[i];
            }

            // Find cutoff
            float cumsum = 0.0f;
            uint32_t cutoff_idx = vocab_size;
            for (uint32_t i = 0; i < vocab_size; ++i)
            {
                cumsum += probs[i] / sum;
                if (cumsum >= p)
                {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Zero out tokens outside top-p
            for (uint32_t i = cutoff_idx; i < vocab_size; ++i)
            {
                uint32_t token_idx = scored_tokens[i].second;
                logits[token_idx] = -std::numeric_limits<float>::infinity();
            }
        }

        void BitNetModel::softmax(float *logits, uint32_t size)
        {
            // Find max for numerical stability
            float max_val = logits[0];
            for (uint32_t i = 1; i < size; ++i)
            {
                max_val = std::max(max_val, logits[i]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (uint32_t i = 0; i < size; ++i)
            {
                logits[i] = std::exp(logits[i] - max_val);
                sum += logits[i];
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            for (uint32_t i = 0; i < size; ++i)
            {
                logits[i] *= inv_sum;
            }
        }

        // ============================================================================
        // STATISTICS
        // ============================================================================

        void BitNetModel::print_stats() const
        {
            if (stats_.total_tokens_generated == 0)
            {
                std::cout << "No tokens generated yet.\n";
                return;
            }

            std::cout << "\nBitNet Model Statistics\n";
            std::cout << "=======================\n";
            std::cout << "Total tokens generated: " << stats_.total_tokens_generated << "\n";
            std::cout << "Total time: " << stats_.total_time_ms << " ms\n";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Average speed: " << stats_.tokens_per_second() << " tokens/sec\n";
        }

        // ============================================================================
        // WEIGHT LOADING (STUB)
        // ============================================================================

        ModelWeights load_model_weights(
            const std::string &checkpoint_path,
            const ModelConfig &config)
        {
            // TODO: Implement actual weight loading from checkpoint
            // For now, return empty weights structure
            std::cout << "Loading weights from: " << checkpoint_path << "\n";
            std::cout << "Warning: Weight loading not yet implemented. Using dummy weights.\n";

            ModelWeights weights;
            // Initialize with dummy data
            return weights;
        }

    } // namespace bitnet
} // namespace ryzen_llm
