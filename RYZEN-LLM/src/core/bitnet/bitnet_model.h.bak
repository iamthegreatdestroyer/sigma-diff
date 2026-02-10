#pragma once

/**
 * @file bitnet_model.h
 * @brief Complete BitNet transformer model for inference
 *
 * Implements a full BitNet model with:
 *   - Token embedding
 *   - Positional encoding
 *   - N transformer layers
 *   - Output projection & sampling
 *
 * [REF:BITNET-002] - Model Implementation
 */

#include "bitnet_layer.h"
#include <string>
#include <vector>

namespace ryzen_llm
{
    namespace bitnet
    {

        /**
         * Model configuration
         */
        struct ModelConfig
        {
            uint32_t vocab_size = 32000; ///< Vocabulary size
            uint32_t hidden_dim = 4096;  ///< Hidden dimension
            uint32_t num_layers = 32;    ///< Number of transformer layers
            uint32_t num_heads = 32;     ///< Number of attention heads
            uint32_t ffn_dim = 11008;    ///< FFN intermediate dimension
            uint32_t max_seq_len = 2048; ///< Maximum sequence length

            std::string name = "BitNet-7B"; ///< Model name
        };

        /**
         * Model weights (all components)
         */
        struct ModelWeights
        {
            // Embedding
            std::vector<float> token_embedding;   ///< [vocab_size, hidden_dim]
            std::vector<float> position_encoding; ///< [max_seq_len, hidden_dim]

            // Transformer layers
            std::vector<BitNetLayerParams> layers; ///< [num_layers]

            // Output
            std::vector<int8_t> output_projection; ///< [hidden_dim, vocab_size] (ternary)
            LayerNormParams final_ln;              ///< Final layer norm
        };

        /**
         * Generation configuration
         */
        struct GenerationConfig
        {
            uint32_t max_new_tokens = 256; ///< Maximum tokens to generate
            float temperature = 0.8f;      ///< Sampling temperature
            float top_p = 0.9f;            ///< Nucleus sampling threshold
            uint32_t top_k = 50;           ///< Top-k sampling
            bool use_cache = true;         ///< Enable KV caching
        };

        /**
         * Complete BitNet model for inference
         *
         * Usage:
         *   BitNetModel model(config, weights, gemm_engine);
         *   auto tokens = model.generate(prompt, gen_config);
         */
        class BitNetModel
        {
        public:
            /**
             * Initialize model
             *
             * @param config Model configuration
             * @param weights Model weights (loaded from checkpoint)
             * @param gemm_engine T-MAC GEMM engine
             */
            BitNetModel(
                const ModelConfig &config,
                const ModelWeights &weights,
                std::shared_ptr<tmac::TMACGemmOptimized> gemm_engine);

            /**
             * Generate tokens from prompt
             *
             * @param prompt Input token IDs
             * @param gen_config Generation configuration
             * @return Generated token IDs
             *
             * Example:
             *   std::vector<uint32_t> prompt = {1, 450, 22172, ...};  // "The quick"
             *   auto output = model.generate(prompt, gen_config);
             *   // output: [1, 450, 22172, 17354, ...]  // "The quick brown fox"
             */
            std::vector<uint32_t> generate(
                const std::vector<uint32_t> &prompt,
                const GenerationConfig &gen_config);

            /**
             * Forward pass through model (single step)
             *
             * @param tokens Input token IDs [batch_size, seq_len]
             * @param logits Output logits [batch_size, seq_len, vocab_size]
             * @param batch_size Batch size
             * @param seq_len Sequence length
             */
            void forward(
                const uint32_t *tokens,
                float *logits,
                uint32_t batch_size,
                uint32_t seq_len);

            /**
             * Get model statistics
             */
            struct Stats
            {
                uint64_t total_tokens_generated = 0;
                double total_time_ms = 0.0;
                double tokens_per_second() const
                {
                    return total_time_ms > 0
                               ? (total_tokens_generated * 1000.0) / total_time_ms
                               : 0.0;
                }
            };

            const Stats &get_stats() const { return stats_; }
            void reset_stats() { stats_ = Stats{}; }
            void print_stats() const;

        private:
            ModelConfig config_;
            ModelWeights weights_;
            std::shared_ptr<tmac::TMACGemmOptimized> gemm_engine_;

            // Transformer layers
            std::vector<std::unique_ptr<BitNetLayer>> layers_;

            // Working buffers
            std::vector<float> hidden_states_;
            std::vector<float> layer_output_;

            mutable Stats stats_;

            /**
             * Embed tokens to hidden states
             *
             * @param tokens Input token IDs
             * @param output Hidden states [batch×seq×hidden]
             * @param batch_size Batch size
             * @param seq_len Sequence length
             */
            void embed_tokens(
                const uint32_t *tokens,
                float *output,
                uint32_t batch_size,
                uint32_t seq_len);

            /**
             * Add positional encoding
             *
             * @param hidden_states Hidden states to modify (in-place)
             * @param seq_len Sequence length
             * @param hidden_dim Hidden dimension
             * @param start_pos Starting position (for KV cache)
             */
            void add_positional_encoding(
                float *hidden_states,
                uint32_t seq_len,
                uint32_t hidden_dim,
                uint32_t start_pos = 0);

            /**
             * Compute logits from final hidden state
             *
             * @param hidden_states Final layer output [batch×seq×hidden]
             * @param logits Output logits [batch×seq×vocab]
             * @param batch_size Batch size
             * @param seq_len Sequence length
             */
            void compute_logits(
                const float *hidden_states,
                float *logits,
                uint32_t batch_size,
                uint32_t seq_len);

            /**
             * Sample next token from logits
             *
             * @param logits Logits for last position [vocab_size]
             * @param gen_config Generation configuration
             * @return Sampled token ID
             */
            uint32_t sample_token(
                const float *logits,
                const GenerationConfig &gen_config);

            /**
             * Apply temperature scaling
             */
            void apply_temperature(float *logits, uint32_t vocab_size, float temperature);

            /**
             * Top-k filtering
             */
            void apply_top_k(float *logits, uint32_t vocab_size, uint32_t k);

            /**
             * Top-p (nucleus) sampling
             */
            void apply_top_p(float *logits, uint32_t vocab_size, float p);

            /**
             * Softmax for probability distribution
             */
            void softmax(float *logits, uint32_t size);
        };

        /**
         * Load model weights from checkpoint file
         *
         * @param checkpoint_path Path to model checkpoint
         * @param config Model configuration
         * @return Loaded weights
         */
        ModelWeights load_model_weights(
            const std::string &checkpoint_path,
            const ModelConfig &config);

    } // namespace bitnet
} // namespace ryzen_llm
