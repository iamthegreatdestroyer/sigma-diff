#ifndef WEIGHT_VALIDATOR_H
#define WEIGHT_VALIDATOR_H

#include "safetensors_loader.h"
#include <map>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>

namespace ryzanstein_llm
{
    namespace io
    {

        /**
         * @struct BitNetConfig
         * @brief Configuration for BitNet-7B model
         */
        struct BitNetConfig
        {
            // Model dimensions
            uint64_t hidden_size = 4096;
            uint64_t num_heads = 32;
            uint64_t num_layers = 32;
            uint64_t intermediate_size = 11008;
            uint64_t vocab_size = 32000;
            uint64_t max_seq_length = 2048;

            // Quantization settings
            bool use_ternary_quantization = true;
            float weight_scale = 1.0f;

            // Expected layer structure
            std::vector<std::string> expected_layer_types = {
                "embed_tokens",    // Embedding layer
                "input_layernorm", // Layer normalization
                "self_attn",       // Self-attention
                "mlp",             // MLP layer
                "post_attention_layernorm"};
        };

        /**
         * @struct ValidationResult
         * @brief Detailed validation results
         */
        struct ValidationResult
        {
            bool is_valid = false;
            std::vector<std::string> errors;
            std::vector<std::string> warnings;
            std::vector<std::string> info;

            struct TensorValidation
            {
                std::string name;
                bool shape_valid = false;
                bool dtype_valid = false;
                bool data_valid = false;
                bool quantization_valid = false;
                float min_value = 0.0f;
                float max_value = 0.0f;
                float mean_value = 0.0f;
                float std_dev = 0.0f;
            };

            std::map<std::string, TensorValidation> tensor_validations;
            uint64_t total_parameters = 0;
            uint64_t total_bytes = 0;
            double validation_time_seconds = 0.0;

            std::string report() const;
        };

        /**
         * @class WeightValidator
         * @brief Validates BitNet-7B weights and model structure
         *
         * Performs:
         * - Shape validation against expected dimensions
         * - Data type verification
         * - Quantization integrity checks
         * - Statistical validation (NaN, Inf detection)
         * - Memory consistency checks
         * - Layer completeness verification
         */
        class WeightValidator
        {
        public:
            WeightValidator(const BitNetConfig &config = BitNetConfig());

            /**
             * @brief Validate a loaded weight map
             * @param weights Map of tensor names to Tensor objects
             * @return Detailed validation results
             */
            ValidationResult validate_bitnet_weights(
                const std::map<std::string, Tensor> &weights);

            /**
             * @brief Validate a single tensor
             * @param tensor The tensor to validate
             * @param expected_shape Expected shape (empty = no check)
             * @return Validation errors (empty = valid)
             */
            std::vector<std::string> validate_tensor(
                const Tensor &tensor,
                const std::vector<uint64_t> &expected_shape = {});

            /**
             * @brief Check for data type consistency
             */
            bool validate_dtype_consistency(const std::map<std::string, Tensor> &weights);

            /**
             * @brief Check quantization integrity for int8 weights
             */
            bool validate_quantization(
                const Tensor &tensor,
                float expected_scale = 1.0f);

            /**
             * @brief Detect NaN and Inf values in float tensors
             */
            std::pair<bool, std::string> check_numerical_stability(
                const Tensor &tensor);

            /**
             * @brief Verify layer structure matches expected layout
             */
            bool verify_layer_structure(
                const std::map<std::string, Tensor> &weights);

            /**
             * @brief Get expected shape for a given weight name
             */
            std::vector<uint64_t> get_expected_shape(const std::string &weight_name);

            /**
             * @brief Set custom configuration
             */
            void set_config(const BitNetConfig &config) { config_ = config; }

            /**
             * @brief Enable verbose validation logging
             */
            void set_verbose(bool verbose) { verbose_ = verbose; }

        private:
            BitNetConfig config_;
            bool verbose_;

            /**
             * @brief Extract layer name and type from full weight name
             * e.g., "model.layers.0.self_attn.q_proj.weight" -> ("self_attn", "q_proj")
             */
            std::pair<std::string, std::string> parse_weight_name_(
                const std::string &full_name);

            /**
             * @brief Validate attention head configuration
             */
            bool validate_attention_heads_(
                const Tensor &q_proj,
                const Tensor &k_proj,
                const Tensor &v_proj);

            /**
             * @brief Compute statistics for float32 tensor
             */
            struct FloatStats
            {
                float min_val;
                float max_val;
                float mean;
                float std_dev;
                bool has_nan;
                bool has_inf;
            };

            FloatStats compute_float_stats_(const Tensor &tensor);

            /**
             * @brief Validate embedding dimensions
             */
            bool validate_embeddings_(const std::map<std::string, Tensor> &weights);

            /**
             * @brief Check parameter count matches expected
             */
            bool validate_param_count_(const std::map<std::string, Tensor> &weights);
        };

        /**
         * @brief Pretty-print validation results
         */
        std::ostream &operator<<(
            std::ostream &os,
            const ValidationResult &result);

    } // namespace io
} // namespace ryzanstein_llm

#endif // WEIGHT_VALIDATOR_H
