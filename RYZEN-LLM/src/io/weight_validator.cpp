#include "weight_validator.h"
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <sstream>

namespace ryzanstein_llm
{
    namespace io
    {

        std::string ValidationResult::report() const
        {
            std::ostringstream oss;
            oss << "\n=== Weight Validation Report ===\n";
            oss << "Status: " << (is_valid ? "VALID" : "INVALID") << "\n";
            oss << "Total Parameters: " << total_parameters << " ("
                << (total_parameters / 1e9) << "B)\n";
            oss << "Total Bytes: " << total_bytes << " ("
                << (total_bytes / 1e9) << "GB)\n";
            oss << "Validation Time: " << validation_time_seconds << "s\n";

            if (!errors.empty())
            {
                oss << "\nErrors (" << errors.size() << "):\n";
                for (const auto &err : errors)
                {
                    oss << "  ✗ " << err << "\n";
                }
            }

            if (!warnings.empty())
            {
                oss << "\nWarnings (" << warnings.size() << "):\n";
                for (const auto &warn : warnings)
                {
                    oss << "  ⚠ " << warn << "\n";
                }
            }

            if (!info.empty())
            {
                oss << "\nInfo:\n";
                for (const auto &inf : info)
                {
                    oss << "  ℹ " << inf << "\n";
                }
            }

            oss << "\nTensor Validation Summary:\n";
            for (const auto &[name, tv] : tensor_validations)
            {
                oss << "  " << name << ": ";
                oss << (tv.shape_valid ? "✓shape " : "✗shape ");
                oss << (tv.dtype_valid ? "✓dtype " : "✗dtype ");
                oss << (tv.data_valid ? "✓data " : "✗data ");
                if (tv.quantization_valid)
                {
                    oss << "✓quant ";
                }
                oss << "\n";
                oss << "    [";
                oss << std::scientific << std::setprecision(3);
                oss << tv.min_value << ", " << tv.max_value << "] ";
                oss << "μ=" << tv.mean_value << " σ=" << tv.std_dev << "\n";
            }

            oss << "================================\n";
            return oss.str();
        }

        std::ostream &operator<<(
            std::ostream &os,
            const ValidationResult &result)
        {
            os << result.report();
            return os;
        }

        WeightValidator::WeightValidator(const BitNetConfig &config)
            : config_(config), verbose_(false) {}

        ValidationResult WeightValidator::validate_bitnet_weights(
            const std::map<std::string, Tensor> &weights)
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            ValidationResult result;

            if (weights.empty())
            {
                result.errors.push_back("Weight map is empty");
                return result;
            }

            // 1. Validate individual tensors
            for (const auto &[name, tensor] : weights)
            {
                if (verbose_)
                {
                    std::cerr << "Validating tensor: " << name << "\n";
                }

                auto errors = validate_tensor(tensor);
                auto [stable, msg] = check_numerical_stability(tensor);

                ValidationResult::TensorValidation tv;
                tv.name = name;
                tv.shape_valid = errors.empty();
                tv.dtype_valid = tensor.dtype != DataType::UNKNOWN;
                tv.data_valid = !errors.empty() ? false : true;
                tv.quantization_valid = validate_quantization(tensor, config_.weight_scale);

                result.tensor_validations[name] = tv;
                result.total_parameters += tensor.num_elements();
                result.total_bytes += tensor.total_bytes();

                // Compute statistics for float tensors
                if (tensor.dtype == DataType::FLOAT32)
                {
                    auto stats = compute_float_stats_(tensor);
                    tv.min_value = stats.min_val;
                    tv.max_value = stats.max_val;
                    tv.mean_value = stats.mean;
                    tv.std_dev = stats.std_dev;

                    if (stats.has_nan)
                    {
                        result.errors.push_back("Tensor " + name + " contains NaN");
                    }
                    if (stats.has_inf)
                    {
                        result.errors.push_back("Tensor " + name + " contains Inf");
                    }
                }
            }

            // 2. Validate layer structure
            if (!verify_layer_structure_(weights))
            {
                result.warnings.push_back("Some expected layers are missing");
            }

            // 3. Validate parameter count
            if (!validate_param_count_(weights))
            {
                result.warnings.push_back(
                    "Parameter count differs from expected BitNet-7B (7B params)");
            }

            // 4. Check data type consistency
            if (!validate_dtype_consistency(weights))
            {
                result.warnings.push_back("Mixed data types detected");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;

            result.validation_time_seconds = elapsed.count();
            result.is_valid = result.errors.empty();

            return result;
        }

        std::vector<std::string> WeightValidator::validate_tensor(
            const Tensor &tensor,
            const std::vector<uint64_t> &expected_shape)
        {
            std::vector<std::string> errors;

            if (tensor.name.empty())
            {
                errors.push_back("Tensor has empty name");
            }

            if (tensor.shape.empty())
            {
                errors.push_back("Tensor " + tensor.name + " has empty shape");
                return errors;
            }

            if (tensor.data.empty())
            {
                errors.push_back("Tensor " + tensor.name + " has no data");
                return errors;
            }

            // Check shape dimensions
            uint64_t expected_elements = 1;
            for (auto dim : tensor.shape)
            {
                if (dim == 0)
                {
                    errors.push_back("Tensor " + tensor.name + " has zero dimension");
                    break;
                }
                expected_elements *= dim;
            }

            // Validate expected shape if provided
            if (!expected_shape.empty())
            {
                if (tensor.shape != expected_shape)
                {
                    errors.push_back("Tensor " + tensor.name + " shape mismatch");
                }
            }

            // Validate data size matches shape
            size_t expected_bytes = expected_elements * TensorMetadata::dtype_size(tensor.dtype);
            if (tensor.data.size() != expected_bytes)
            {
                errors.push_back(
                    "Tensor " + tensor.name + " data size mismatch: " +
                    std::to_string(tensor.data.size()) + " bytes, expected " +
                    std::to_string(expected_bytes));
            }

            return errors;
        }

        bool WeightValidator::validate_dtype_consistency(
            const std::map<std::string, Tensor> &weights)
        {
            if (weights.empty())
                return true;

            std::map<DataType, int> dtype_counts;
            for (const auto &[name, tensor] : weights)
            {
                dtype_counts[tensor.dtype]++;
            }

            // BitNet should primarily use float32, int8, or bfloat16
            for (const auto &[dtype, count] : dtype_counts)
            {
                if (dtype == DataType::UNKNOWN)
                {
                    return false;
                }
            }

            return true;
        }

        bool WeightValidator::validate_quantization(
            const Tensor &tensor,
            float expected_scale)
        {
            if (tensor.dtype != DataType::INT8)
            {
                return true; // Not quantized
            }

            auto *data = tensor.data_ptr<int8_t>();
            int32_t min_val = std::numeric_limits<int8_t>::max();
            int32_t max_val = std::numeric_limits<int8_t>::min();

            for (uint64_t i = 0; i < tensor.num_elements(); ++i)
            {
                min_val = std::min(min_val, static_cast<int32_t>(data[i]));
                max_val = std::max(max_val, static_cast<int32_t>(data[i]));
            }

            // For int8 quantization, we expect values to span most of [-128, 127] range
            // unless it's a small parameter tensor
            if (tensor.num_elements() > 1000)
            {
                if (min_val == max_val)
                {
                    return false; // All same value - likely corrupt
                }
            }

            return true;
        }

        std::pair<bool, std::string> WeightValidator::check_numerical_stability(
            const Tensor &tensor)
        {
            if (tensor.dtype != DataType::FLOAT32)
            {
                return {true, ""};
            }

            auto *data = tensor.data_ptr<float>();

            for (uint64_t i = 0; i < tensor.num_elements(); ++i)
            {
                if (std::isnan(data[i]))
                {
                    return {false, "Found NaN in " + tensor.name};
                }
                if (std::isinf(data[i]))
                {
                    return {false, "Found Inf in " + tensor.name};
                }
            }

            return {true, ""};
        }

        bool WeightValidator::verify_layer_structure_(
            const std::map<std::string, Tensor> &weights)
        {
            // Check for essential components
            bool has_embeddings = false;
            bool has_layers = false;
            bool has_lm_head = false;

            for (const auto &[name, tensor] : weights)
            {
                if (name.find("embed") != std::string::npos)
                {
                    has_embeddings = true;
                }
                if (name.find("layers") != std::string::npos && name.find("weight") != std::string::npos)
                {
                    has_layers = true;
                }
                if (name.find("lm_head") != std::string::npos)
                {
                    has_lm_head = true;
                }
            }

            return has_embeddings && has_layers;
        }

        std::pair<std::string, std::string> WeightValidator::parse_weight_name_(
            const std::string &full_name)
        {
            // Extract layer type and component from full name
            // e.g., "model.layers.0.self_attn.q_proj.weight" -> ("self_attn", "q_proj")

            std::string layer_type = "";
            std::string component = "";

            size_t pos = 0;
            while ((pos = full_name.find('.', pos)) != std::string::npos)
            {
                pos++;
                // Look for known layer types
                if (full_name.substr(pos).find("self_attn") == 0)
                {
                    layer_type = "self_attn";
                }
                else if (full_name.substr(pos).find("mlp") == 0)
                {
                    layer_type = "mlp";
                }
                else if (full_name.substr(pos).find("norm") != std::string::npos)
                {
                    layer_type = "norm";
                }
            }

            return {layer_type, component};
        }

        WeightValidator::FloatStats WeightValidator::compute_float_stats_(
            const Tensor &tensor)
        {
            FloatStats stats = {
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::lowest(),
                0.0f, 0.0f, false, false};

            if (tensor.dtype != DataType::FLOAT32)
            {
                return stats;
            }

            auto *data = tensor.data_ptr<float>();
            uint64_t count = 0;

            // First pass: min, max, mean
            for (uint64_t i = 0; i < tensor.num_elements(); ++i)
            {
                if (std::isnan(data[i]))
                {
                    stats.has_nan = true;
                }
                else if (std::isinf(data[i]))
                {
                    stats.has_inf = true;
                }
                else
                {
                    stats.min_val = std::min(stats.min_val, data[i]);
                    stats.max_val = std::max(stats.max_val, data[i]);
                    stats.mean += data[i];
                    count++;
                }
            }

            if (count > 0)
            {
                stats.mean /= count;
            }

            // Second pass: std dev
            float variance = 0.0f;
            count = 0;
            for (uint64_t i = 0; i < tensor.num_elements(); ++i)
            {
                if (std::isfinite(data[i]))
                {
                    float diff = data[i] - stats.mean;
                    variance += diff * diff;
                    count++;
                }
            }

            if (count > 0)
            {
                stats.std_dev = std::sqrt(variance / count);
            }

            return stats;
        }

        bool WeightValidator::validate_embeddings_(
            const std::map<std::string, Tensor> &weights)
        {
            // Check embedding layer exists and has correct size
            for (const auto &[name, tensor] : weights)
            {
                if (name.find("embed_tokens") != std::string::npos)
                {
                    if (tensor.shape.size() == 2)
                    {
                        if (tensor.shape[0] == config_.vocab_size &&
                            tensor.shape[1] == config_.hidden_size)
                        {
                            return true;
                        }
                    }
                }
            }

            return false;
        }

        bool WeightValidator::validate_param_count_(
            const std::map<std::string, Tensor> &weights)
        {
            uint64_t total = 0;
            for (const auto &[name, tensor] : weights)
            {
                total += tensor.num_elements();
            }

            // BitNet-7B has approximately 7B parameters
            // Allow ±10% tolerance
            uint64_t expected = 7000000000ULL;
            uint64_t tolerance = expected / 10;

            return total >= (expected - tolerance) && total <= (expected + tolerance);
        }

        std::vector<uint64_t> WeightValidator::get_expected_shape(
            const std::string &weight_name)
        {
            // Determine expected shape based on weight name

            if (weight_name.find("embed_tokens") != std::string::npos)
            {
                return {config_.vocab_size, config_.hidden_size};
            }

            if (weight_name.find("q_proj") != std::string::npos ||
                weight_name.find("k_proj") != std::string::npos ||
                weight_name.find("v_proj") != std::string::npos)
            {
                return {config_.hidden_size, config_.hidden_size};
            }

            if (weight_name.find("o_proj") != std::string::npos)
            {
                return {config_.hidden_size, config_.hidden_size};
            }

            if (weight_name.find("up_proj") != std::string::npos ||
                weight_name.find("down_proj") != std::string::npos)
            {
                return {config_.intermediate_size, config_.hidden_size};
            }

            if (weight_name.find("gate_proj") != std::string::npos)
            {
                return {config_.intermediate_size, config_.hidden_size};
            }

            if (weight_name.find("norm") != std::string::npos ||
                weight_name.find("layer_norm") != std::string::npos)
            {
                return {config_.hidden_size};
            }

            if (weight_name.find("lm_head") != std::string::npos)
            {
                return {config_.vocab_size, config_.hidden_size};
            }

            return {}; // Unknown shape
        }

        bool WeightValidator::validate_attention_heads_(
            const Tensor &q_proj,
            const Tensor &k_proj,
            const Tensor &v_proj)
        {
            // Check that projections have compatible shapes for multi-head attention
            if (q_proj.shape.size() != 2 || k_proj.shape.size() != 2 || v_proj.shape.size() != 2)
            {
                return false;
            }

            // All should have same hidden dimension
            if (q_proj.shape[0] != k_proj.shape[0] || q_proj.shape[0] != v_proj.shape[0])
            {
                return false;
            }

            // Check head dimensions divide evenly
            uint64_t head_dim = config_.hidden_size / config_.num_heads;
            if (config_.hidden_size % config_.num_heads != 0)
            {
                return false;
            }

            return true;
        }

    } // namespace io
} // namespace ryzanstein_llm
