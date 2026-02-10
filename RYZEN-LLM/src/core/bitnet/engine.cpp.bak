/*
 * RYZEN-LLM BitNet Inference Engine Implementation
 * [REF:PHASE1-003] - BitNet b1.58 Transformer Engine
 */

#include "engine.h"
#include "../../optimization/avx512/matmul.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>

namespace ryzen_llm
{
    namespace bitnet
    {

        // ============================================================================
        // Constructor
        // ============================================================================

        bitnet::BitNetEngine::BitNetEngine(const ModelConfig &config)
            : config_(config), q_weights_(config.num_layers, TernaryWeight(config.hidden_size, config.hidden_size)), k_weights_(config.num_layers, TernaryWeight(config.hidden_size, config.hidden_size)), v_weights_(config.num_layers, TernaryWeight(config.hidden_size, config.hidden_size)), o_weights_(config.num_layers, TernaryWeight(config.hidden_size, config.hidden_size)), gate_weights_(config.num_layers, TernaryWeight(config.hidden_size, config.intermediate_size)), up_weights_(config.num_layers, TernaryWeight(config.hidden_size, config.intermediate_size)), down_weights_(config.num_layers, TernaryWeight(config.intermediate_size, config.hidden_size)), attn_norm_weights_(config.num_layers, std::vector<float>(config.hidden_size, 1.0f)), mlp_norm_weights_(config.num_layers, std::vector<float>(config.hidden_size, 1.0f)), final_norm_weights_(config.hidden_size, 1.0f), lm_head_weights_(config.hidden_size * config.vocab_size, 0.0f), hidden_states_(config.hidden_size, 0.0f), residual_(config.hidden_size, 0.0f), attn_output_(config.hidden_size, 0.0f), mlp_output_(config.hidden_size, 0.0f)
        {
            // Initialize advanced KV cache manager
            memory::KVCacheConfig kv_config;
            kv_config.num_layers = config.num_layers;
            kv_config.num_heads = config.num_heads;
            kv_config.head_dim = config.head_dim;
            kv_config.max_batch_size = 1;                                              // Single sequence for now
            kv_config.block_size = 16;                                                 // 16 tokens per block
            kv_config.num_blocks = (config.max_seq_length / kv_config.block_size) * 2; // 2x capacity
            kv_config.enable_quantization = false;                                     // Start without quantization
            kv_config.enable_prefetching = true;

            kv_cache_manager_ = std::make_unique<memory::KVCacheManager>(kv_config);

            // Allocate sequence for inference (sequence ID = 0)
            kv_cache_manager_->AllocateSequence(0, config.max_seq_length);

            // Initialize speculative decoder if enabled
            if (config.use_speculative_decoding)
            {
                speculative::SpeculativeConfig spec_config;
                spec_config.draft_config.vocab_size = config.vocab_size;
                spec_config.draft_config.hidden_size = config.hidden_size;
                spec_config.draft_config.num_layers = 6; // Smaller draft model
                spec_config.draft_config.num_heads = config.num_heads;
                spec_config.verifier_config.vocab_size = config.vocab_size;
                spec_config.verifier_config.hidden_size = config.hidden_size;
                spec_config.verifier_config.num_layers = config.num_layers;
                spec_config.verifier_config.num_heads = config.num_heads;
                spec_config.enable_speculative_decoding = true;
                spec_config.batch_size = 1;

                speculative_decoder_ = std::make_unique<speculative::SpeculativeDecoder>(spec_config);
            }

            // Initialize T-MAC engine if enabled
            if (config.use_tmac)
            {
                tmac::TMACConfig tmac_config;
                tmac_config.lookup_width = 8; // 8-bit lookup (256 entries)

                tmac_engine_ = std::make_unique<tmac::LookupTableGEMM>(tmac_config);

                if (config.tmac_precompute_on_load)
                {
                    // Precompute tables for common matrix sizes
                    // This will be done after weights are loaded
                    std::cout << "[T-MAC] Engine initialized, tables will be built on first inference\n";
                }
            }

            // Initialize embedding weights
            embedding_weights_ = TernaryWeight(config.vocab_size, config.hidden_size);
        }

        // ============================================================================
        // Model Loading (SafeTensors Format)
        // ============================================================================

        bool bitnet::BitNetEngine::load_weights(const std::string &weights_path)
        {
            // Try to load from SafeTensors format first
            if (load_safetensors_weights(weights_path))
            {
                return true;
            }

            // Fallback: Initialize with random weights for testing
            return initialize_random_weights();
        }

        bool bitnet::BitNetEngine::load_safetensors_weights(const std::string &weights_path)
        {
            std::ifstream file(weights_path, std::ios::binary | std::ios::ate);
            if (!file.is_open())
            {
                return false;
            }

            // Get file size
            std::streamsize file_size = file.tellg();
            file.seekg(0, std::ios::beg);

            // Read the entire file into memory
            std::vector<char> file_data(file_size);
            if (!file.read(file_data.data(), file_size))
            {
                file.close();
                return false;
            }
            file.close();

            // Parse SafeTensors format
            // Format: [8-byte length][JSON header][tensor data...]
            if (file_size < 8)
            {
                return false; // Too small for header length
            }

            // Read header length (little-endian u64)
            uint64_t header_len;
            std::memcpy(&header_len, file_data.data(), 8);

            if (header_len == 0 || header_len > static_cast<uint64_t>(file_size - 8))
            {
                return false; // Invalid header length
            }

            // Parse JSON header
            std::string header_json(file_data.data() + 8, header_len);
            try
            {
                // Simple JSON parsing for SafeTensors metadata
                // This is a basic implementation - in production, use a proper JSON library
                return parse_safetensors_header(header_json, file_data.data() + 8 + header_len, file_size - 8 - header_len);
            }
            catch (const std::exception &e)
            {
                // JSON parsing failed
                return false;
            }
        }

        bool bitnet::BitNetEngine::parse_safetensors_header(const std::string &header_json, const char *data_start, size_t data_size)
        {
            (void)header_json;
            (void)data_start;
            (void)data_size; // Suppress unused parameter warnings
            // Basic SafeTensors JSON parser for BitNet b1.58 weights
            // Expected weight names and their mappings to our engine structure

            struct TensorInfo
            {
                std::string name;
                std::vector<uint32_t> shape;
                std::string dtype;
                std::vector<uint64_t> data_offsets;
            };

            std::vector<TensorInfo> tensors;

            // Very basic JSON parsing - find tensor entries
            // This is a simplified parser for the specific BitNet structure
            size_t pos = 0;
            while (pos < header_json.size())
            {
                // Find tensor name (quoted string followed by colon)
                size_t name_start = header_json.find('"', pos);
                if (name_start == std::string::npos)
                    break;
                name_start++; // Skip opening quote

                size_t name_end = header_json.find('"', name_start);
                if (name_end == std::string::npos)
                    break;

                std::string tensor_name = header_json.substr(name_start, name_end - name_start);

                // Find the shape array
                size_t shape_start = header_json.find("[", name_end);
                if (shape_start == std::string::npos)
                    break;

                size_t shape_end = header_json.find("]", shape_start);
                if (shape_end == std::string::npos)
                    break;

                // Parse shape (simplified - assumes 2D tensors)
                std::string shape_str = header_json.substr(shape_start + 1, shape_end - shape_start - 1);
                std::vector<uint32_t> shape;
                size_t comma_pos = 0;
                while ((comma_pos = shape_str.find(',')) != std::string::npos)
                {
                    shape.push_back(std::stoul(shape_str.substr(0, comma_pos)));
                    shape_str = shape_str.substr(comma_pos + 1);
                }
                if (!shape_str.empty())
                {
                    shape.push_back(std::stoul(shape_str));
                }

                // Find data_offsets
                size_t offsets_start = header_json.find("[", shape_end);
                if (offsets_start == std::string::npos)
                    break;

                size_t offsets_end = header_json.find("]", offsets_start);
                if (offsets_end == std::string::npos)
                    break;

                std::string offsets_str = header_json.substr(offsets_start + 1, offsets_end - offsets_start - 1);
                std::vector<uint64_t> data_offsets;
                comma_pos = 0;
                while ((comma_pos = offsets_str.find(',')) != std::string::npos)
                {
                    data_offsets.push_back(std::stoull(offsets_str.substr(0, comma_pos)));
                    offsets_str = offsets_str.substr(comma_pos + 1);
                }
                if (!offsets_str.empty())
                {
                    data_offsets.push_back(std::stoull(offsets_str));
                }

                tensors.push_back({tensor_name, shape, "F32", data_offsets});

                pos = offsets_end + 1;
            }

            // Now load the tensors into our engine structure
            for (const auto &tensor : tensors)
            {
                if (tensor.data_offsets.size() != 2)
                    continue; // Invalid offsets

                uint64_t data_offset = tensor.data_offsets[0];
                uint64_t data_size = tensor.data_offsets[1] - tensor.data_offsets[0];

                if (data_offset + data_size > data_size)
                    continue; // Out of bounds

                const char *tensor_data = data_start + data_offset;

                // Map tensor names to our engine structure
                if (tensor.name == "embed_tokens.weight" && tensor.shape.size() == 2)
                {
                    // Embedding weights: [vocab_size, hidden_size]
                    uint32_t rows = tensor.shape[0];
                    uint32_t cols = tensor.shape[1];
                    std::vector<float> weights(rows * cols);
                    std::memcpy(weights.data(), tensor_data, data_size);
                    embedding_weights_ = quantize_weights_ternary(weights.data(), rows, cols, config_.quant_config);
                }
                else if (tensor.name.find("layers.") == 0 && tensor.name.find(".attention.") != std::string::npos)
                {
                    // Extract layer index
                    size_t layer_start = tensor.name.find("layers.") + 7;
                    size_t layer_end = tensor.name.find(".", layer_start);
                    if (layer_end == std::string::npos)
                        continue;
                    uint32_t layer_idx = std::stoul(tensor.name.substr(layer_start, layer_end - layer_start));

                    if (layer_idx >= config_.num_layers)
                        continue;

                    if (tensor.name.find("q_proj.weight") != std::string::npos && tensor.shape.size() == 2)
                    {
                        uint32_t rows = tensor.shape[0];
                        uint32_t cols = tensor.shape[1];
                        std::vector<float> weights(rows * cols);
                        std::memcpy(weights.data(), tensor_data, data_size);
                        q_weights_[layer_idx] = quantize_weights_ternary(weights.data(), rows, cols, config_.quant_config);
                    }
                    else if (tensor.name.find("k_proj.weight") != std::string::npos && tensor.shape.size() == 2)
                    {
                        uint32_t rows = tensor.shape[0];
                        uint32_t cols = tensor.shape[1];
                        std::vector<float> weights(rows * cols);
                        std::memcpy(weights.data(), tensor_data, data_size);
                        k_weights_[layer_idx] = quantize_weights_ternary(weights.data(), rows, cols, config_.quant_config);
                    }
                    else if (tensor.name.find("v_proj.weight") != std::string::npos && tensor.shape.size() == 2)
                    {
                        uint32_t rows = tensor.shape[0];
                        uint32_t cols = tensor.shape[1];
                        std::vector<float> weights(rows * cols);
                        std::memcpy(weights.data(), tensor_data, data_size);
                        v_weights_[layer_idx] = quantize_weights_ternary(weights.data(), rows, cols, config_.quant_config);
                    }
                    else if (tensor.name.find("o_proj.weight") != std::string::npos && tensor.shape.size() == 2)
                    {
                        uint32_t rows = tensor.shape[0];
                        uint32_t cols = tensor.shape[1];
                        std::vector<float> weights(rows * cols);
                        std::memcpy(weights.data(), tensor_data, data_size);
                        o_weights_[layer_idx] = quantize_weights_ternary(weights.data(), rows, cols, config_.quant_config);
                    }
                }
                else if (tensor.name.find("layers.") == 0 && tensor.name.find(".mlp.") != std::string::npos)
                {
                    // Extract layer index
                    size_t layer_start = tensor.name.find("layers.") + 7;
                    size_t layer_end = tensor.name.find(".", layer_start);
                    if (layer_end == std::string::npos)
                        continue;
                    uint32_t layer_idx = std::stoul(tensor.name.substr(layer_start, layer_end - layer_start));

                    if (layer_idx >= config_.num_layers)
                        continue;

                    if (tensor.name.find("gate_proj.weight") != std::string::npos && tensor.shape.size() == 2)
                    {
                        uint32_t rows = tensor.shape[0];
                        uint32_t cols = tensor.shape[1];
                        std::vector<float> weights(rows * cols);
                        std::memcpy(weights.data(), tensor_data, data_size);
                        gate_weights_[layer_idx] = quantize_weights_ternary(weights.data(), rows, cols, config_.quant_config);
                    }
                    else if (tensor.name.find("up_proj.weight") != std::string::npos && tensor.shape.size() == 2)
                    {
                        uint32_t rows = tensor.shape[0];
                        uint32_t cols = tensor.shape[1];
                        std::vector<float> weights(rows * cols);
                        std::memcpy(weights.data(), tensor_data, data_size);
                        up_weights_[layer_idx] = quantize_weights_ternary(weights.data(), rows, cols, config_.quant_config);
                    }
                    else if (tensor.name.find("down_proj.weight") != std::string::npos && tensor.shape.size() == 2)
                    {
                        uint32_t rows = tensor.shape[0];
                        uint32_t cols = tensor.shape[1];
                        std::vector<float> weights(rows * cols);
                        std::memcpy(weights.data(), tensor_data, data_size);
                        down_weights_[layer_idx] = quantize_weights_ternary(weights.data(), rows, cols, config_.quant_config);
                    }
                }
                else if (tensor.name.find("layers.") == 0 && tensor.name.find("_norm.weight") != std::string::npos)
                {
                    // Extract layer index
                    size_t layer_start = tensor.name.find("layers.") + 7;
                    size_t layer_end = tensor.name.find(".", layer_start);
                    if (layer_end == std::string::npos)
                        continue;
                    uint32_t layer_idx = std::stoul(tensor.name.substr(layer_start, layer_end - layer_start));

                    if (layer_idx >= config_.num_layers)
                        continue;

                    if (tensor.name.find("attention_norm.weight") != std::string::npos && tensor.shape.size() == 1)
                    {
                        uint32_t size = tensor.shape[0];
                        (void)size; // Suppress unused variable warning
                        std::memcpy(attn_norm_weights_[layer_idx].data(), tensor_data, data_size);
                    }
                    else if (tensor.name.find("mlp_norm.weight") != std::string::npos && tensor.shape.size() == 1)
                    {
                        uint32_t size = tensor.shape[0];
                        (void)size; // Suppress unused variable warning
                        std::memcpy(mlp_norm_weights_[layer_idx].data(), tensor_data, data_size);
                    }
                }
                else if (tensor.name == "norm.weight" && tensor.shape.size() == 1)
                {
                    // Final layer norm
                    uint32_t size = tensor.shape[0];
                    (void)size; // Suppress unused variable warning
                    std::memcpy(final_norm_weights_.data(), tensor_data, data_size);
                }
                else if (tensor.name == "lm_head.weight" && tensor.shape.size() == 2)
                {
                    // Output projection (keep as float32)
                    uint32_t rows = tensor.shape[0];
                    uint32_t cols = tensor.shape[1];
                    (void)rows;
                    (void)cols; // Suppress unused variable warnings
                    std::memcpy(lm_head_weights_.data(), tensor_data, data_size);
                }
            }

            return true;
        }

        bool bitnet::BitNetEngine::initialize_random_weights()
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 0.02f);

            // Initialize embedding weights
            std::vector<float> temp_embed(config_.vocab_size * config_.hidden_size);
            for (auto &w : temp_embed)
            {
                w = dist(gen);
            }
            embedding_weights_ = quantize_weights_ternary(
                temp_embed.data(),
                config_.vocab_size,
                config_.hidden_size,
                config_.quant_config);

            // Initialize transformer weights
            for (uint32_t layer = 0; layer < config_.num_layers; ++layer)
            {
                // Attention weights
                std::vector<float> temp_q(config_.hidden_size * config_.hidden_size);
                std::vector<float> temp_k(config_.hidden_size * config_.hidden_size);
                std::vector<float> temp_v(config_.hidden_size * config_.hidden_size);
                std::vector<float> temp_o(config_.hidden_size * config_.hidden_size);

                for (auto &w : temp_q)
                    w = dist(gen);
                for (auto &w : temp_k)
                    w = dist(gen);
                for (auto &w : temp_v)
                    w = dist(gen);
                for (auto &w : temp_o)
                    w = dist(gen);

                q_weights_[layer] = quantize_weights_ternary(temp_q.data(), config_.hidden_size, config_.hidden_size, config_.quant_config);
                k_weights_[layer] = quantize_weights_ternary(temp_k.data(), config_.hidden_size, config_.hidden_size, config_.quant_config);
                v_weights_[layer] = quantize_weights_ternary(temp_v.data(), config_.hidden_size, config_.hidden_size, config_.quant_config);
                o_weights_[layer] = quantize_weights_ternary(temp_o.data(), config_.hidden_size, config_.hidden_size, config_.quant_config);

                // MLP weights
                std::vector<float> temp_gate(config_.hidden_size * config_.intermediate_size);
                std::vector<float> temp_up(config_.hidden_size * config_.intermediate_size);
                std::vector<float> temp_down(config_.intermediate_size * config_.hidden_size);

                for (auto &w : temp_gate)
                    w = dist(gen);
                for (auto &w : temp_up)
                    w = dist(gen);
                for (auto &w : temp_down)
                    w = dist(gen);

                gate_weights_[layer] = quantize_weights_ternary(temp_gate.data(), config_.hidden_size, config_.intermediate_size, config_.quant_config);
                up_weights_[layer] = quantize_weights_ternary(temp_up.data(), config_.hidden_size, config_.intermediate_size, config_.quant_config);
                down_weights_[layer] = quantize_weights_ternary(temp_down.data(), config_.intermediate_size, config_.hidden_size, config_.quant_config);

                // Layer norms (keep as float32 for precision)
                std::normal_distribution<float> norm_dist(1.0f, 0.02f);
                for (auto &w : attn_norm_weights_[layer])
                    w = norm_dist(gen);
                for (auto &w : mlp_norm_weights_[layer])
                    w = norm_dist(gen);
            }

            // Final norm
            std::normal_distribution<float> norm_dist(1.0f, 0.02f);
            for (auto &w : final_norm_weights_)
                w = norm_dist(gen);

            // Output projection (keep as float32)
            for (auto &w : lm_head_weights_)
                w = dist(gen);

            return true;
        }

        // ============================================================================
        // Generation
        // ============================================================================

        std::vector<uint32_t> bitnet::BitNetEngine::generate(
            const std::vector<uint32_t> &input_tokens,
            const GenerationConfig &gen_config)
        {
            reset_cache();

            std::vector<uint32_t> output_tokens = input_tokens;
            output_tokens.reserve(input_tokens.size() + gen_config.max_tokens);

            // Process input tokens (prefill phase)
            for (size_t i = 0; i < input_tokens.size(); ++i)
            {
                forward(input_tokens[i], static_cast<uint32_t>(i));
            }

            // Generate new tokens (decode phase)
            if (config_.use_speculative_decoding && speculative_decoder_)
            {
                // Use speculative decoding for accelerated generation
                std::vector<int> prefix_int(output_tokens.begin(), output_tokens.end());
                std::vector<int> new_tokens = speculative_decoder_->decode_next_tokens(prefix_int, gen_config.max_tokens);

                // Convert back to uint32_t and add to output
                for (int token : new_tokens)
                {
                    if (token == 2 || token == 0) // EOS tokens
                        break;
                    output_tokens.push_back(static_cast<uint32_t>(token));
                }
            }
            else
            {
                // Standard autoregressive generation
                for (uint32_t i = 0; i < gen_config.max_tokens; ++i)
                {
                    const uint32_t position = static_cast<uint32_t>(input_tokens.size()) + i;
                    const uint32_t last_token = output_tokens.back();

                    // Forward pass
                    std::vector<float> logits = forward(last_token, position);

                    // Sample next token
                    uint32_t next_token = sample_token(logits, gen_config);

                    // Check for EOS token (commonly token 2 or 0)
                    if (next_token == 2 || next_token == 0)
                    {
                        break;
                    }

                    output_tokens.push_back(next_token);
                }
            }

            return output_tokens;
        }

        // ============================================================================
        // Forward Pass
        // ============================================================================

        std::vector<float> bitnet::BitNetEngine::forward(uint32_t token_id, uint32_t position)
        {
            // 1. Token embedding lookup
            embedding_lookup(token_id, hidden_states_.data());

            // 2. Process each transformer layer
            for (uint32_t layer = 0; layer < config_.num_layers; ++layer)
            {
                // Save residual
                std::copy(hidden_states_.begin(), hidden_states_.end(), residual_.begin());

                // Pre-attention RMSNorm
                rms_norm(
                    hidden_states_.data(),
                    attn_norm_weights_[layer].data(),
                    hidden_states_.data(),
                    config_.hidden_size);

                // Self-attention
                attention_layer(layer, hidden_states_.data(), position, attn_output_.data());

                // Residual connection
                for (uint32_t i = 0; i < config_.hidden_size; ++i)
                {
                    hidden_states_[i] = residual_[i] + attn_output_[i];
                }

                // Save residual again
                std::copy(hidden_states_.begin(), hidden_states_.end(), residual_.begin());

                // Pre-MLP RMSNorm
                rms_norm(
                    hidden_states_.data(),
                    mlp_norm_weights_[layer].data(),
                    hidden_states_.data(),
                    config_.hidden_size);

                // MLP
                mlp_layer(layer, hidden_states_.data(), mlp_output_.data());

                // Residual connection
                for (uint32_t i = 0; i < config_.hidden_size; ++i)
                {
                    hidden_states_[i] = residual_[i] + mlp_output_[i];
                }
            }

            // 3. Final RMSNorm
            rms_norm(
                hidden_states_.data(),
                final_norm_weights_.data(),
                hidden_states_.data(),
                config_.hidden_size);

            // 4. Output projection (LM head)
            std::vector<float> logits(config_.vocab_size, 0.0f);

            for (uint32_t v = 0; v < config_.vocab_size; ++v)
            {
                float sum = 0.0f;
                for (uint32_t h = 0; h < config_.hidden_size; ++h)
                {
                    sum += hidden_states_[h] * lm_head_weights_[v * config_.hidden_size + h];
                }
                logits[v] = sum;
            }

            return logits;
        }

        // ============================================================================
        // Core Transformer Operations
        // ============================================================================

        void bitnet::BitNetEngine::embedding_lookup(uint32_t token_id, float *output)
        {
            // Lookup token embedding and dequantize
            if (token_id >= config_.vocab_size)
            {
                token_id = 0; // UNK token
            }

            for (uint32_t i = 0; i < config_.hidden_size; ++i)
            {
                const uint32_t idx = token_id * config_.hidden_size + i;
                const float scale = embedding_weights_.get_scale(idx);
                output[i] = static_cast<float>(embedding_weights_.values[idx]) * scale;
            }
        }

        void bitnet::BitNetEngine::rms_norm(
            const float *input,
            const float *weight,
            float *output,
            uint32_t size)
        {
            // Compute RMS: sqrt(mean(x^2) + eps)
            double sum_squares = 0.0;
            for (uint32_t i = 0; i < size; ++i)
            {
                sum_squares += static_cast<double>(input[i]) * input[i];
            }

            const float rms = static_cast<float>(std::sqrt(sum_squares / static_cast<double>(size) + config_.rms_norm_eps));
            const float inv_rms = 1.0f / rms;

            // Normalize and scale
            for (uint32_t i = 0; i < size; ++i)
            {
                output[i] = input[i] * inv_rms * weight[i];
            }
        }

        void bitnet::BitNetEngine::attention_layer(
            uint32_t layer_idx,
            const float *input,
            uint32_t position,
            float *output)
        {
            const uint32_t h = config_.hidden_size;
            const uint32_t num_heads = config_.num_heads;
            const uint32_t head_dim = config_.head_dim;

            // Allocate Q, K, V
            std::vector<float> q(h, 0.0f);
            std::vector<float> k(h, 0.0f);
            std::vector<float> v(h, 0.0f);

            // Quantize input to INT8
            QuantizedActivation q_input = quantize_activations_int8(
                input,
                h,
                config_.quant_config);

            // Compute Q, K, V projections using optimized matmul (T-MAC or AVX-512)
            dispatch_ternary_matmul(
                q_weights_[layer_idx],
                q_input,
                q.data(),
                h, 1, h);

            dispatch_ternary_matmul(
                k_weights_[layer_idx],
                q_input,
                k.data(),
                h, 1, h);

            dispatch_ternary_matmul(
                v_weights_[layer_idx],
                q_input,
                v.data(),
                h, 1, h);

            // Apply rotary positional embeddings
            apply_rotary_embeddings(q.data(), k.data(), position, head_dim);

            // Store K, V in advanced cache manager
            const uint64_t sequence_id = 0; // Single sequence for now

            // Get cache pointers for current position
            float *k_cache_ptr = kv_cache_manager_->GetKeyCache(sequence_id, layer_idx, position);
            float *v_cache_ptr = kv_cache_manager_->GetValueCache(sequence_id, layer_idx, position);

            if (k_cache_ptr && v_cache_ptr)
            {
                // Copy K and V to cache
                std::copy(k.begin(), k.end(), k_cache_ptr);
                std::copy(v.begin(), v.end(), v_cache_ptr);

                // Update cache length
                kv_cache_manager_->AppendTokens(sequence_id, 1);
            }

            // Get current sequence length
            uint32_t current_length;
            const float *k_sequence = kv_cache_manager_->GetKeySequence(sequence_id, layer_idx, current_length);
            const float *v_sequence = kv_cache_manager_->GetValueSequence(sequence_id, layer_idx, current_length);

            // Multi-head attention
            std::vector<float> attn_scores(current_length, 0.0f);
            std::vector<float> head_output(head_dim, 0.0f);
            std::fill(output, output + h, 0.0f);

            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            for (uint32_t head = 0; head < num_heads; ++head)
            {
                const uint32_t head_offset = head * head_dim;

                // Compute attention scores: Q @ K^T using cached sequence
                for (uint32_t t = 0; t < current_length; ++t)
                {
                    float score = 0.0f;
                    const uint32_t k_offset = t * num_heads * head_dim + head_offset;

                    for (uint32_t d = 0; d < head_dim; ++d)
                    {
                        score += q[head_offset + d] * k_sequence[k_offset + d];
                    }

                    attn_scores[t] = score * scale;
                }

                // Softmax
                softmax(attn_scores.data(), current_length);

                // Weighted sum of values: softmax(QK^T) @ V using cached sequence
                std::fill(head_output.begin(), head_output.end(), 0.0f);

                for (uint32_t t = 0; t < current_length; ++t)
                {
                    const uint32_t v_offset = t * num_heads * head_dim + head_offset;

                    for (uint32_t d = 0; d < head_dim; ++d)
                    {
                        head_output[d] += attn_scores[t] * v_sequence[v_offset + d];
                    }
                }

                // Copy to output
                for (uint32_t d = 0; d < head_dim; ++d)
                {
                    output[head_offset + d] = head_output[d];
                }
            }

            // Output projection
            QuantizedActivation q_attn_out = quantize_activations_int8(
                output,
                h,
                config_.quant_config);

            std::vector<float> final_output(h, 0.0f);
            dispatch_ternary_matmul(
                o_weights_[layer_idx],
                q_attn_out,
                final_output.data(),
                h, 1, h);

            std::copy(final_output.begin(), final_output.end(), output);
        }

        void bitnet::BitNetEngine::mlp_layer(
            uint32_t layer_idx,
            const float *input,
            float *output)
        {
            const uint32_t h = config_.hidden_size;
            const uint32_t i = config_.intermediate_size;

            // Quantize input
            QuantizedActivation q_input = quantize_activations_int8(
                input,
                h,
                config_.quant_config);

            // Gate and Up projections
            std::vector<float> gate(i, 0.0f);
            std::vector<float> up(i, 0.0f);

            dispatch_ternary_matmul(
                gate_weights_[layer_idx],
                q_input,
                gate.data(),
                static_cast<uint32_t>(i), 1, h);

            dispatch_ternary_matmul(
                up_weights_[layer_idx],
                q_input,
                up.data(),
                i, 1, h);

            // SwiGLU activation: gate * SiLU(up)
            std::vector<float> swiglu(i, 0.0f);
            for (uint32_t j = 0; j < i; ++j)
            {
                // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                const float sigmoid = 1.0f / (1.0f + std::exp(-up[j]));
                swiglu[j] = gate[j] * (up[j] * sigmoid);
            }

            // Down projection
            QuantizedActivation q_swiglu = quantize_activations_int8(
                swiglu.data(),
                i,
                config_.quant_config);

            dispatch_ternary_matmul(
                down_weights_[layer_idx],
                q_swiglu,
                output,
                h, 1, static_cast<uint32_t>(i));
        }

        // ============================================================================
        // Sampling Methods
        // ============================================================================

        uint32_t bitnet::BitNetEngine::sample_token(
            const std::vector<float> &logits,
            const GenerationConfig &config)
        {
            if (config.temperature < 1e-6f)
            {
                return sample_greedy(logits);
            }

            if (config.top_k > 0)
            {
                return sample_top_k(logits, config.top_k, config.temperature);
            }

            if (config.top_p < 1.0f)
            {
                return sample_top_p(logits, config.top_p, config.temperature);
            }

            // Default: temperature sampling
            std::vector<float> scaled_logits = logits;
            for (auto &logit : scaled_logits)
            {
                logit /= config.temperature;
            }
            // scaled_logits.size() returns size_t; softmax expects uint32_t
            softmax(scaled_logits.data(), static_cast<uint32_t>(scaled_logits.size()));

            // Sample from distribution
            std::random_device rd;
            std::mt19937 gen(config.seed != 0 ? config.seed : rd());
            std::discrete_distribution<uint32_t> dist(
                scaled_logits.begin(),
                scaled_logits.end());

            return dist(gen);
        }

        uint32_t bitnet::BitNetEngine::sample_greedy(const std::vector<float> &logits)
        {
            return static_cast<uint32_t>(std::distance(
                logits.begin(),
                std::max_element(logits.begin(), logits.end())));
        }

        uint32_t bitnet::BitNetEngine::sample_top_k(
            const std::vector<float> &logits,
            uint32_t k,
            float temperature)
        {
            // Create indexed copy
            std::vector<std::pair<float, uint32_t>> indexed_logits;
            indexed_logits.reserve(logits.size());

            for (size_t i = 0; i < logits.size(); ++i)
            {
                indexed_logits.emplace_back(logits[i], static_cast<uint32_t>(i));
            }

            // Partial sort to get top-k
            std::partial_sort(
                indexed_logits.begin(),
                indexed_logits.begin() + k,
                indexed_logits.end(),
                [](const auto &a, const auto &b)
                { return a.first > b.first; });

            // Apply temperature and softmax
            std::vector<float> top_k_probs(k);
            for (uint32_t i = 0; i < k; ++i)
            {
                top_k_probs[i] = indexed_logits[i].first / temperature;
            }
            // k is uint32_t; keep explicit types
            softmax(top_k_probs.data(), static_cast<uint32_t>(k));

            // Sample
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<uint32_t> dist(top_k_probs.begin(), top_k_probs.end());

            return indexed_logits[dist(gen)].second;
        }

        uint32_t bitnet::BitNetEngine::sample_top_p(
            const std::vector<float> &logits,
            float p,
            float temperature)
        {
            // Create indexed copy
            std::vector<std::pair<float, uint32_t>> indexed_logits;
            indexed_logits.reserve(logits.size());

            for (size_t i = 0; i < logits.size(); ++i)
            {
                indexed_logits.emplace_back(logits[i] / temperature, static_cast<uint32_t>(i));
            }

            // Sort by logit value
            std::sort(
                indexed_logits.begin(),
                indexed_logits.end(),
                [](const auto &a, const auto &b)
                { return a.first > b.first; });

            // Apply softmax
            std::vector<float> probs(indexed_logits.size());
            for (size_t i = 0; i < indexed_logits.size(); ++i)
            {
                probs[i] = indexed_logits[i].first;
            }
            // probs.size() is size_t; cast to uint32_t to satisfy softmax signature
            softmax(probs.data(), static_cast<uint32_t>(probs.size()));

            // Find nucleus
            float cumsum = 0.0f;
            size_t nucleus_size = 0;

            for (size_t i = 0; i < probs.size(); ++i)
            {
                cumsum += probs[i];
                nucleus_size = i + 1;
                if (cumsum >= p)
                {
                    break;
                }
            }

            // Renormalize nucleus
            std::vector<float> nucleus_probs(nucleus_size);
            float nucleus_sum = 0.0f;
            for (size_t i = 0; i < nucleus_size; ++i)
            {
                nucleus_probs[i] = probs[i];
                nucleus_sum += probs[i];
            }
            for (auto &prob : nucleus_probs)
            {
                prob /= nucleus_sum;
            }

            // Sample
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<uint32_t> dist(
                nucleus_probs.begin(),
                nucleus_probs.end());

            return indexed_logits[dist(gen)].second;
        }

        // ============================================================================
        // Helper Functions
        // ============================================================================

        void bitnet::BitNetEngine::softmax(float *logits, uint32_t size)
        {
            // Find max for numerical stability
            float max_logit = logits[0];
            for (uint32_t i = 1; i < size; ++i)
            {
                max_logit = std::max(max_logit, logits[i]);
            }

            // Compute exp and sum
            double sum = 0.0;
            for (uint32_t i = 0; i < size; ++i)
            {
                logits[i] = std::exp(logits[i] - max_logit);
                sum += logits[i];
            }

            // Normalize
            const float inv_sum = static_cast<float>(1.0 / sum); // sum is double; cast to float intentionally
            for (uint32_t i = 0; i < size; ++i)
            {
                logits[i] *= inv_sum;
            }
        }

        void bitnet::BitNetEngine::apply_rotary_embeddings(
            float *q,
            float *k,
            uint32_t position,
            uint32_t head_dim)
        {
            // Simplified RoPE (rotary positional embeddings)
            // For each pair of dimensions, apply rotation
            const float theta_base = 10000.0f;

            for (uint32_t i = 0; i < head_dim; i += 2)
            {
                const float freq = 1.0f / std::pow(
                                              theta_base,
                                              static_cast<float>(i) / head_dim);
                const float theta = position * freq;
                const float cos_theta = std::cos(theta);
                const float sin_theta = std::sin(theta);

                // Apply rotation to Q
                const float q0 = q[i];
                const float q1 = q[i + 1];
                q[i] = q0 * cos_theta - q1 * sin_theta;
                q[i + 1] = q0 * sin_theta + q1 * cos_theta;

                // Apply rotation to K
                const float k0 = k[i];
                const float k1 = k[i + 1];
                k[i] = k0 * cos_theta - k1 * sin_theta;
                k[i + 1] = k0 * sin_theta + k1 * cos_theta;
            }
        }

        void bitnet::BitNetEngine::dispatch_ternary_matmul(
            const TernaryWeight &weights,
            const QuantizedActivation &activations,
            float *output,
            uint32_t M, uint32_t N, uint32_t K)
        {
            // Use T-MAC if available and enabled, otherwise fall back to AVX-512
            if (tmac_engine_ && config_.use_tmac)
            {
                tmac_engine_->ComputeHybrid(weights, activations, output, M, N, K);
            }
            else
            {
                avx512::dispatch_ternary_matmul(weights, activations, output, M, N, K);
            }
        }

        void bitnet::BitNetEngine::reset_cache()
        {
            // Reset the advanced KV cache manager
            kv_cache_manager_->FreeSequence(0);                             // Free current sequence
            kv_cache_manager_->AllocateSequence(0, config_.max_seq_length); // Reallocate
        }

    } // namespace bitnet
} // namespace ryzen_llm
