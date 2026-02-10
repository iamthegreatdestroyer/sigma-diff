/*
 * RYZEN-LLM BitNet Inference Engine
 * [REF:PHASE1-003] - BitNet b1.58 Transformer Engine
 */

#pragma once

#include "quantize.h"
#include "kernels/matmul.h"
#include "../tmac/lut_gemm.h"
#include "../../optimization/memory/kv_cache.h"
#include "../../optimization/speculative/speculative_decoder.h"
#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <iostream>

namespace ryzen_llm
{
    namespace bitnet
    {

        // Forward declarations
        struct ModelConfig;
        struct GenerationConfig;
        struct KVCache;

        /**
         * BitNet Model Configuration
         */
        struct ModelConfig
        {
            uint32_t vocab_size;        // Vocabulary size
            uint32_t hidden_size;       // Hidden dimension (d_model)
            uint32_t intermediate_size; // MLP intermediate dimension
            uint32_t num_layers;        // Number of transformer layers
            uint32_t num_heads;         // Number of attention heads
            uint32_t head_dim;          // Dimension per attention head
            uint32_t max_seq_length;    // Maximum sequence length
            float rms_norm_eps;         // RMSNorm epsilon

            QuantConfig quant_config; // Quantization configuration

            // T-MAC optimization (use lookup tables for matmul)
            bool use_tmac;                // Default: true if available
            bool tmac_precompute_on_load; // Precompute tables when loading weights

            // Speculative decoding (Phase 2 optimization)
            bool use_speculative_decoding; // Enable speculative decoding
            uint32_t speculative_k;        // Number of draft tokens to generate

            ModelConfig()
                : vocab_size(32000), hidden_size(4096), intermediate_size(11008), num_layers(32), num_heads(32), head_dim(128), max_seq_length(2048), rms_norm_eps(1e-6f), quant_config(), use_tmac(true), tmac_precompute_on_load(true), use_speculative_decoding(false), speculative_k(4)
            {
            }
        };

        /**
         * Generation Configuration
         */
        struct GenerationConfig
        {
            uint32_t max_tokens;      // Maximum tokens to generate
            float temperature;        // Sampling temperature (0.0 = greedy)
            uint32_t top_k;           // Top-K sampling (0 = disabled)
            float top_p;              // Nucleus sampling (1.0 = disabled)
            float repetition_penalty; // Repetition penalty (1.0 = disabled)
            uint32_t seed;            // Random seed (0 = random)

            GenerationConfig()
                : max_tokens(100), temperature(0.7f), top_k(50), top_p(0.9f), repetition_penalty(1.1f), seed(0)
            {
            }
        };

        /**
         * KV Cache for Attention (per layer)
         */
        struct KVCache
        {
            std::vector<float> k_cache; // Key cache [seq_len, num_heads, head_dim]
            std::vector<float> v_cache; // Value cache [seq_len, num_heads, head_dim]
            uint32_t current_length;    // Current cached sequence length

            KVCache(uint32_t max_seq_len, uint32_t num_heads, uint32_t head_dim)
                : k_cache(max_seq_len * num_heads * head_dim, 0.0f), v_cache(max_seq_len * num_heads * head_dim, 0.0f), current_length(0)
            {
            }

            void reset()
            {
                current_length = 0;
            }
        };

        /**
         * BitNet Inference Engine
         *
         * Implements BitNet b1.58 transformer with ternary weights.
         * Key features:
         * - Token embedding lookup
         * - RMSNorm layer normalization
         * - Multi-head self-attention with ternary QKV projections
         * - SwiGLU MLP blocks with ternary weights
         * - KV caching for efficient generation
         * - Top-K/Top-P/Temperature sampling
         */
        class BitNetEngine
        {
        public:
            explicit BitNetEngine(const ModelConfig &config);
            ~BitNetEngine() = default;

            // Delete copy/move constructors (weights are large)
            BitNetEngine(const BitNetEngine &) = delete;
            BitNetEngine &operator=(const BitNetEngine &) = delete;

            /**
             * Load model weights from file
             *
             * @param weights_path Path to model weights file
             * @return true if successful
             */
            bool load_weights(const std::string &weights_path);

            /**
             * Generate text from input tokens
             *
             * @param input_tokens Input token IDs
             * @param gen_config Generation configuration
             * @return Generated token IDs
             */
            std::vector<uint32_t> generate(
                const std::vector<uint32_t> &input_tokens,
                const GenerationConfig &gen_config);

            /**
             * Forward pass (single token, with KV cache)
             *
             * @param token_id Input token ID
             * @param position Position in sequence
             * @return Logits for next token [vocab_size]
             */
            std::vector<float> forward(uint32_t token_id, uint32_t position);

            /**
             * Reset KV cache (for new sequence)
             */
            void reset_cache();

            /**
             * Get model configuration
             */
            const ModelConfig &get_config() const { return config_; }

        private:
        private:
            // Core transformer operations
            void embedding_lookup(uint32_t token_id, float *output);
            void rms_norm(const float *input, const float *weight, float *output, uint32_t size);
            void attention_layer(
                uint32_t layer_idx,
                const float *input,
                uint32_t position,
                float *output);
            void mlp_layer(
                uint32_t layer_idx,
                const float *input,
                float *output);

            // Sampling methods
            uint32_t sample_token(const std::vector<float> &logits, const GenerationConfig &config);
            uint32_t sample_greedy(const std::vector<float> &logits);
            uint32_t sample_top_k(const std::vector<float> &logits, uint32_t k, float temperature);
            uint32_t sample_top_p(const std::vector<float> &logits, float p, float temperature);

            // Helper functions
            void softmax(float *logits, uint32_t size);
            void apply_rotary_embeddings(float *q, float *k, uint32_t position, uint32_t head_dim);

            // Optimized matmul dispatch (T-MAC or AVX-512)
            void dispatch_ternary_matmul(
                const TernaryWeight &weights,
                const QuantizedActivation &activations,
                float *output,
                uint32_t M, uint32_t N, uint32_t K);

            // Weight loading methods
            bool load_safetensors_weights(const std::string &weights_path);
            bool parse_safetensors_header(const std::string &header_json, const char *data_start, size_t data_size);
            bool initialize_random_weights();

            ModelConfig config_;

            // Model weights (ternary quantized)
            TernaryWeight embedding_weights_;         // [vocab_size, hidden_size]
            std::vector<TernaryWeight> q_weights_;    // [num_layers, hidden_size, hidden_size]
            std::vector<TernaryWeight> k_weights_;    // [num_layers, hidden_size, hidden_size]
            std::vector<TernaryWeight> v_weights_;    // [num_layers, hidden_size, hidden_size]
            std::vector<TernaryWeight> o_weights_;    // [num_layers, hidden_size, hidden_size]
            std::vector<TernaryWeight> gate_weights_; // [num_layers, hidden_size, intermediate_size]
            std::vector<TernaryWeight> up_weights_;   // [num_layers, hidden_size, intermediate_size]
            std::vector<TernaryWeight> down_weights_; // [num_layers, intermediate_size, hidden_size]

            // RMSNorm weights (FP32)
            std::vector<std::vector<float>> attn_norm_weights_; // [num_layers, hidden_size]
            std::vector<std::vector<float>> mlp_norm_weights_;  // [num_layers, hidden_size]
            std::vector<float> final_norm_weights_;             // [hidden_size]

            // Output projection (unquantized for numerical stability)
            std::vector<float> lm_head_weights_; // [hidden_size, vocab_size]

            // T-MAC lookup table engine for optimized matmul
            std::unique_ptr<tmac::LookupTableGEMM> tmac_engine_;

            // KV Cache
            // Advanced KV Cache Manager (from optimization layer)
            std::unique_ptr<memory::KVCacheManager> kv_cache_manager_;

            // Speculative decoding (from optimization layer)
            std::unique_ptr<speculative::SpeculativeDecoder> speculative_decoder_;

            // Intermediate buffers (preallocated)
            std::vector<float> hidden_states_;
            std::vector<float> residual_;
            std::vector<float> attn_output_;
            std::vector<float> mlp_output_;
        };

    } // namespace bitnet
} // namespace ryzen_llm
