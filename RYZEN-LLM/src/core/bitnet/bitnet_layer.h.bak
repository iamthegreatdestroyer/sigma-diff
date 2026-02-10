#pragma once

/**
 * @file bitnet_layer.h
 * @brief BitNet transformer layer implementation with T-MAC acceleration
 *
 * Implements a complete BitNet transformer layer:
 *   - Multi-head self-attention with ternary weights
 *   - Feed-forward network (FFN) with ternary weights
 *   - Layer normalization
 *   - Residual connections
 *
 * Architecture:
 *   x_norm = LayerNorm(x)
 *   attn_out = MultiHeadAttention(x_norm)
 *   x = x + attn_out
 *   x_norm = LayerNorm(x)
 *   ffn_out = FFN(x_norm)
 *   x = x + ffn_out
 *
 * [REF:BITNET-001] - Forward Pass Implementation
 */

#include "tmac_gemm_optimized.h"
#include <memory>
#include <vector>
#include <cstdint>

namespace ryzen_llm
{
    namespace bitnet
    {

        /**
         * Layer normalization parameters
         */
        struct LayerNormParams
        {
            std::vector<float> gamma; ///< Scale parameters [hidden_dim]
            std::vector<float> beta;  ///< Shift parameters [hidden_dim]
            float eps = 1e-5f;        ///< Epsilon for numerical stability
        };

        /**
         * Multi-head attention parameters
         */
        struct AttentionParams
        {
            // Weight matrices (ternary)
            std::vector<int8_t> W_q; ///< Query projection [hidden_dim, hidden_dim]
            std::vector<int8_t> W_k; ///< Key projection [hidden_dim, hidden_dim]
            std::vector<int8_t> W_v; ///< Value projection [hidden_dim, hidden_dim]
            std::vector<int8_t> W_o; ///< Output projection [hidden_dim, hidden_dim]

            // Configuration
            uint32_t num_heads;  ///< Number of attention heads
            uint32_t head_dim;   ///< Dimension per head
            uint32_t hidden_dim; ///< Total hidden dimension
            float scale_factor;  ///< Attention scaling (1/sqrt(head_dim))
        };

        /**
         * Feed-forward network parameters
         */
        struct FFNParams
        {
            // Weight matrices (ternary)
            std::vector<int8_t> W_up;   ///< Up projection [hidden_dim, ffn_dim]
            std::vector<int8_t> W_down; ///< Down projection [ffn_dim, hidden_dim]

            // Configuration
            uint32_t hidden_dim; ///< Hidden dimension
            uint32_t ffn_dim;    ///< FFN intermediate dimension (typically 4× hidden_dim)
        };

        /**
         * Complete BitNet transformer layer
         */
        struct BitNetLayerParams
        {
            LayerNormParams ln1;  ///< Pre-attention layer norm
            AttentionParams attn; ///< Multi-head self-attention
            LayerNormParams ln2;  ///< Pre-FFN layer norm
            FFNParams ffn;        ///< Feed-forward network
        };

        /**
         * BitNet transformer layer implementation
         *
         * Executes one complete transformer layer with:
         *   - T-MAC accelerated matrix operations
         *   - Optimized memory layout
         *   - In-place operations where possible
         */
        class BitNetLayer
        {
        public:
            /**
             * Initialize layer with parameters
             *
             * @param params Layer parameters (weights, config)
             * @param gemm_engine T-MAC GEMM engine
             */
            BitNetLayer(
                const BitNetLayerParams &params,
                std::shared_ptr<tmac::TMACGemmOptimized> gemm_engine);

            /**
             * Forward pass through transformer layer
             *
             * @param input Input activations [batch_size, seq_len, hidden_dim] (flattened)
             * @param output Output activations [batch_size, seq_len, hidden_dim] (preallocated)
             * @param batch_size Batch size
             * @param seq_len Sequence length
             * @param cache Optional KV cache for inference (nullptr for training)
             *
             * Time: O(batch_size × seq_len × hidden_dim²) with T-MAC
             * Space: O(batch_size × seq_len × hidden_dim) for intermediate activations
             */
            void forward(
                const float *input,
                float *output,
                uint32_t batch_size,
                uint32_t seq_len,
                void *cache = nullptr);

            /**
             * Get memory requirements for intermediate buffers
             *
             * @param batch_size Maximum batch size
             * @param seq_len Maximum sequence length
             * @return Required bytes for workspace
             */
            size_t get_workspace_size(uint32_t batch_size, uint32_t seq_len) const;

        private:
            BitNetLayerParams params_;
            std::shared_ptr<tmac::TMACGemmOptimized> gemm_engine_;

            // Workspace buffers (allocated once, reused)
            std::vector<float> workspace_;

            /**
             * Layer normalization
             *
             * output = gamma * (input - mean) / sqrt(variance + eps) + beta
             *
             * @param input Input tensor
             * @param output Output tensor (can be same as input)
             * @param ln_params LayerNorm parameters
             * @param size Number of elements
             * @param hidden_dim Hidden dimension for normalization
             */
            void layer_norm(
                const float *input,
                float *output,
                const LayerNormParams &ln_params,
                uint32_t size,
                uint32_t hidden_dim);

            /**
             * Multi-head self-attention
             *
             * Implements scaled dot-product attention:
             *   Q = x × W_q, K = x × W_k, V = x × W_v
             *   Attention = softmax(Q × K^T / sqrt(d_k)) × V
             *   Output = Attention × W_o
             *
             * @param input Input tensor [batch×seq×hidden]
             * @param output Output tensor [batch×seq×hidden]
             * @param batch_size Batch size
             * @param seq_len Sequence length
             */
            void multi_head_attention(
                const float *input,
                float *output,
                uint32_t batch_size,
                uint32_t seq_len);

            /**
             * Feed-forward network
             *
             * Implements: FFN(x) = GELU(x × W_up) × W_down
             *
             * @param input Input tensor [batch×seq×hidden]
             * @param output Output tensor [batch×seq×hidden]
             * @param batch_size Batch size
             * @param seq_len Sequence length
             */
            void feed_forward(
                const float *input,
                float *output,
                uint32_t batch_size,
                uint32_t seq_len);

            /**
             * GELU activation function
             *
             * GELU(x) = x * Φ(x) where Φ is Gaussian CDF
             * Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
             *
             * @param x Input/output tensor (in-place)
             * @param size Number of elements
             */
            void gelu_activation(float *x, size_t size);

            /**
             * Softmax along last dimension
             *
             * @param x Input/output tensor
             * @param rows Number of rows
             * @param cols Number of columns (softmax dimension)
             */
            void softmax(float *x, uint32_t rows, uint32_t cols);

            /**
             * Quantize float32 to INT8 for T-MAC input
             *
             * Uses symmetric quantization:
             *   scale = max(abs(x)) / 127
             *   x_int8 = round(x / scale)
             *
             * @param input FP32 tensor
             * @param output INT8 tensor
             * @param size Number of elements
             * @return Quantization scale factor
             */
            float quantize_to_int8(
                const float *input,
                int8_t *output,
                size_t size);

            /**
             * Dequantize INT32 GEMM output to FP32
             *
             * @param input INT32 tensor
             * @param output FP32 tensor
             * @param size Number of elements
             * @param scale Scale factor from quantization
             */
            void dequantize_from_int32(
                const int32_t *input,
                float *output,
                size_t size,
                float scale);
        };

    } // namespace bitnet
} // namespace ryzen_llm
