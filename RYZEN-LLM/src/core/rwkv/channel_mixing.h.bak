#ifndef RYZEN_LLM_RWKV_CHANNEL_MIXING_H
#define RYZEN_LLM_RWKV_CHANNEL_MIXING_H

#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ryzen_llm::rwkv
{

    /**
     * Configuration parameters for the Channel Mixing layer.
     *
     * Channel mixing in RWKV is the complementary mechanism to time mixing.
     * While time mixing blends information across the temporal/sequential dimension,
     * channel mixing blends information across the feature/channel dimension.
     *
     * This layer applies learned shifts and transformations to each feature dimension,
     * effectively implementing a position-wise feedforward-like operation with gating.
     */
    struct ChannelMixingConfig
    {
        /// Number of hidden dimensions (feature channels to mix)
        uint32_t hidden_dim = 768;

        /// Value gating activation - gates the value pathway (0.5 ~ 1.0)
        float value_gate = 0.99f;

        /// Key gating activation - gates the key pathway (0.0 ~ 0.5)
        float key_gate = 0.05f;

        /// Ratio for the expansion of the feedforward intermediate dimension
        float ff_expansion = 4.0f;

        /// Whether to use bias terms in channel mixing projections
        bool use_bias = true;

        /// Type of activation function: "relu", "gelu", "swish", "mish"
        std::string activation = "mish";

        /// Epsilon for numerical stability in normalization (default 1e-5)
        float eps = 1e-5f;
    };

    /**
     * Channel Mixing Layer for RWKV Attention-Free RNN.
     *
     * The channel mixing layer implements cross-channel information mixing through
     * learned projections. This is the complementary mechanism to time mixing:
     * - Time mixing: blends information across sequence positions
     * - Channel mixing: blends information across feature dimensions
     *
     * Architecture:
     *   x_shifted = shift(x, position_offset)
     *   key_proj = W_k @ relu(x)
     *   value_proj = W_v @ x_shifted
     *   output = value_proj * gate(key_proj)
     *
     * Key features:
     * - Learned position-wise shift for each channel
     * - Parallel key and value pathways with independent gating
     * - Efficient channel-wise computation without attention mechanism
     * - State management for stateful inference
     *
     * Performance characteristics:
     * - Computation: O(hidden_dim^2) per token
     * - Memory: O(hidden_dim) per layer per batch
     * - Vectorizable: Yes, with AVX-512 optimizations available
     * - Bottleneck: Matrix multiplications (W_k, W_v projections)
     */
    class ChannelMixingLayer
    {
    public:
        /**
         * Construct a channel mixing layer.
         *
         * @param hidden_dim Number of feature channels
         * @param layer_id   Unique identifier for this layer (used for initialization)
         * @param config     Configuration parameters for mixing behavior
         * @throws std::invalid_argument if hidden_dim is 0 or config is invalid
         */
        explicit ChannelMixingLayer(
            uint32_t hidden_dim,
            uint32_t layer_id,
            const ChannelMixingConfig &config);

        /**
         * Initialize weight matrices with Xavier initialization.
         *
         * Allocates and initializes:
         * - shift_: learnable channel shift for temporal blending (hidden_dim,)
         * - key_proj_w_: key projection weight matrix (hidden_dim, hidden_dim)
         * - key_proj_b_: key projection bias (hidden_dim,) if use_bias=true
         * - value_proj_w_: value projection weight matrix (hidden_dim, hidden_dim)
         * - value_proj_b_: value projection bias (hidden_dim,) if use_bias=true
         *
         * @throws std::runtime_error if called more than once without reset_state()
         */
        void initialize();

        /**
         * Compute single-token forward pass through channel mixing.
         *
         * Processes a single token's hidden state through the mixing layer:
         * 1. Apply learned shift to previous state (temporal context)
         * 2. Compute key projection with activation
         * 3. Compute value projection on shifted input
         * 4. Gate value pathway with key projection
         * 5. Cache shifted input for next token
         *
         * @param input      Input hidden state (hidden_dim,)
         * @param output     Output hidden state (hidden_dim,) - pre-allocated
         * @return true if forward pass succeeded, false if not initialized
         * @throws std::invalid_argument if input size != hidden_dim
         */
        bool forward(
            const std::vector<float> &input,
            std::vector<float> &output);

        /**
         * Compute forward pass for a sequence of tokens.
         *
         * Processes a full sequence with state threading:
         * - Input: (seq_len, hidden_dim)
         * - Output: (seq_len, hidden_dim)
         * - State is maintained and threaded through sequence
         *
         * @param input_sequence   Sequence of hidden states (seq_len * hidden_dim,)
         * @param seq_len          Number of tokens in sequence
         * @param output_sequence  Output hidden states (seq_len * hidden_dim,) - pre-allocated
         * @return true if successful, false if not initialized or size mismatch
         * @throws std::invalid_argument if sizes don't match
         */
        bool forward_sequence(
            const std::vector<float> &input_sequence,
            uint32_t seq_len,
            std::vector<float> &output_sequence);

        /**
         * Perform pure channel mixing computation on input.
         *
         * Core mixing algorithm without state management:
         * - Pure feedforward operation on single token
         * - Used internally by forward() and forward_sequence()
         * - Can be used for stateless processing if needed
         *
         * @param input      Input hidden state (hidden_dim,)
         * @param output     Output hidden state (hidden_dim,) - pre-allocated
         * @return true if successful, false if invalid state
         */
        bool channel_mix(
            const std::vector<float> &input,
            std::vector<float> &output);

        /**
         * Reset internal state to initial condition.
         *
         * Clears:
         * - prev_shifted_: previous shifted state
         * - All internal buffers
         *
         * Used between independent sequences (e.g., new conversation turn).
         */
        void reset_state();

        /**
         * Save current internal state for later restoration.
         *
         * Captures:
         * - prev_shifted_: previous shifted hidden state
         * - Position information
         * - Internal buffer states
         *
         * Used for multi-turn conversations or checkpointing.
         *
         * @return Serialized state data (binary blob)
         */
        std::vector<uint8_t> save_state() const;

        /**
         * Restore internal state from previously saved checkpoint.
         *
         * Restores:
         * - prev_shifted_: previous shifted hidden state
         * - Position information
         * - Internal buffer states
         *
         * @param state      Serialized state from save_state()
         * @return true if restoration successful, false if state invalid
         * @throws std::runtime_error if state format invalid
         */
        bool load_state(const std::vector<uint8_t> &state);

        /**
         * Get current state of the layer for debugging.
         *
         * @return String representation of layer state (initialization, buffers, stats)
         */
        std::string get_state_string() const;

        /**
         * Get configuration parameters.
         *
         * @return Reference to configuration used at initialization
         */
        const ChannelMixingConfig &get_config() const { return config_; }

        /**
         * Get layer identifier.
         *
         * @return Layer ID used for initialization seeding
         */
        uint32_t get_layer_id() const { return layer_id_; }

        /**
         * Get hidden dimension.
         *
         * @return Number of feature channels
         */
        uint32_t get_hidden_dim() const { return hidden_dim_; }

        /**
         * Check if layer has been initialized.
         *
         * @return true if initialize() was called successfully, false otherwise
         */
        bool is_initialized() const { return initialized_; }

        /// Default destructor
        ~ChannelMixingLayer() = default;

        // Prevent copying to avoid state confusion
        ChannelMixingLayer(const ChannelMixingLayer &) = delete;
        ChannelMixingLayer &operator=(const ChannelMixingLayer &) = delete;

        // Allow move semantics
        ChannelMixingLayer(ChannelMixingLayer &&) = default;
        ChannelMixingLayer &operator=(ChannelMixingLayer &&) = default;

    private:
        // Initialization state
        bool initialized_ = false;
        uint32_t hidden_dim_ = 0;
        uint32_t layer_id_ = 0;
        ChannelMixingConfig config_;

        // Weight matrices and parameters
        std::vector<float> shift_;        ///< Channel shift parameter (hidden_dim,)
        std::vector<float> key_proj_w_;   ///< Key projection weights (hidden_dim x hidden_dim)
        std::vector<float> key_proj_b_;   ///< Key projection bias (hidden_dim,)
        std::vector<float> value_proj_w_; ///< Value projection weights (hidden_dim x hidden_dim)
        std::vector<float> value_proj_b_; ///< Value projection bias (hidden_dim,)

        // Internal buffers and state
        std::vector<float> prev_shifted_;   ///< Previous shifted state for temporal context
        std::vector<float> buffer_key_;     ///< Temporary buffer for key computation
        std::vector<float> buffer_key_act_; ///< Temporary buffer for key activation
        std::vector<float> buffer_value_;   ///< Temporary buffer for value computation
        std::vector<float> buffer_gate_;    ///< Temporary buffer for gating computation

        // Helper methods
        void apply_activation_(
            std::vector<float> &x,
            const std::string &activation_type);

        float activation_fn_(float x, const std::string &activation_type);

        void matrix_multiply_(
            const std::vector<float> &A, // Matrix (m x n)
            const std::vector<float> &x, // Vector (n,)
            std::vector<float> &y,       // Output (m,) - pre-allocated
            uint32_t m,
            uint32_t n,
            const std::vector<float> *bias = nullptr);

        void element_wise_multiply_(
            const std::vector<float> &a,
            const std::vector<float> &b,
            std::vector<float> &out);

        void element_wise_gate_(
            const std::vector<float> &value,
            const std::vector<float> &gate,
            std::vector<float> &out,
            float gate_strength);

        void shift_input_(
            const std::vector<float> &input,
            std::vector<float> &output);
    };

} // namespace ryzen_llm::rwkv

#endif // RYZEN_LLM_RWKV_CHANNEL_MIXING_H
