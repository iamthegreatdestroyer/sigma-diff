#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "bitnet/quantize.h"
#include "bitnet/engine.h" // Add BitNetEngine

// Cross-platform export macro
#ifdef _WIN32
#define RYZEN_EXPORT __declspec(dllexport)
#else
#define RYZEN_EXPORT __attribute__((visibility("default")))
#endif

// C interface for ctypes
extern "C"
{
    RYZEN_EXPORT int test_function()
    {
        return 42;
    }

    // Test function that returns a constant
    RYZEN_EXPORT int test_quantize_scalar()
    {
        return 12345;
    }

    // Test function that just creates a TernaryWeight object
    RYZEN_EXPORT int test_simple_loop()
    {
        try
        {
            // Test fabs function
            float weights[16] = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16};
            float sum = 0.0f;
            for (int i = 0; i < 16; ++i)
            {
                sum += fabs(weights[i]); // Use fabs instead of std::abs
            }
            return (int)sum; // Should return 136
        }
        catch (...)
        {
            return -1;
        }
    }

    // Test nested loops like in quantization
    RYZEN_EXPORT int test_nested_loops()
    {
        try
        {
            // Test nested loops for element counting
            size_t rows = 4;
            size_t cols = 4;
            size_t total_size = 0;
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    total_size++;
                }
            }
            return (int)total_size; // Should return 16
        }
        catch (...)
        {
            return -1;
        }
    }

    // Test quantization-like operations step by step
    RYZEN_EXPORT int test_quantization_steps()
    {
        try
        {
            // Step 1: Basic setup like quantization function
            size_t rows = 4;
            size_t cols = 4;
            float weights[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

            // Step 2: Count elements with nested loops
            size_t total_size = 0;
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    total_size++;
                }
            }

            // Step 3: Allocate vectors (like in quantization)
            std::vector<float> weights_vec(total_size);
            std::vector<int8_t> ternary_weights(total_size);

            // Step 4: Copy weights to vector
            for (size_t i = 0; i < total_size; ++i)
            {
                weights_vec[i] = weights[i];
            }

            // Step 5: Compute mean_abs (like in quantization)
            float mean_abs = 0.0f;
            for (size_t i = 0; i < total_size; ++i)
            {
                mean_abs += fabs(weights_vec[i]);
            }
            mean_abs /= total_size;

            return (int)(mean_abs * 1000); // Should return 8*1000 = 8000
        }
        catch (...)
        {
            return -1;
        }
    }

    // Test just vector allocation
    RYZEN_EXPORT int test_vector_allocation()
    {
        try
        {
            size_t total_size = 16;
            std::vector<float> weights_vec(total_size);
            std::vector<int8_t> ternary_weights(total_size);
            return (int)total_size; // Should return 16
        }
        catch (...)
        {
            return -1;
        }
    }

    // Test vector access
    RYZEN_EXPORT int test_vector_access()
    {
        try
        {
            size_t total_size = 16;
            std::vector<float> weights_vec(total_size);
            for (size_t i = 0; i < total_size; ++i)
            {
                weights_vec[i] = (float)i;
            }
            return (int)weights_vec[15]; // Should return 15
        }
        catch (...)
        {
            return -1;
        }
    }

    RYZEN_EXPORT int test_weight_quantize_only()
    {
        try
        {

            // Create test data
            float weights[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

            ryzanstein_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;

            auto ternary_weight = ryzanstein_llm::bitnet::quantize_weights_ternary_scalar(weights, 4, 4, config);

            return 1;
        }
        catch (const std::exception &e)
        {
            return -1;
        }
        catch (...)
        {
            return -2;
        }
    }

    RYZEN_EXPORT int test_activation_quantize_only()
    {
        try
        {

            // Create test data
            float activations[4] = {1, -2, 3, -4};

            ryzanstein_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;

            auto quantized_activation = ryzanstein_llm::bitnet::quantize_activations_int8_scalar(activations, 4, config);

            return 1;
        }
        catch (const std::exception &e)
        {
            return -1;
        }
        catch (...)
        {
            return -2;
        }
    }

    RYZEN_EXPORT int test_scalar_quantize_direct()
    {
        try
        {

            // Create test data
            float weights[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            float activations[4] = {1, -2, 3, -4};

            ryzanstein_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;

            auto ternary_weight = ryzanstein_llm::bitnet::quantize_weights_ternary_scalar(weights, 4, 4, config);

            auto quantized_activation = ryzanstein_llm::bitnet::quantize_activations_int8_scalar(activations, 4, config);

            return 1;
        }
        catch (const std::exception &e)
        {
            return -1;
        }
        catch (...)
        {
            return -2;
        }
    }

    // Test function that calls quantize_weights_ternary_scalar without creating objects
    RYZEN_EXPORT int test_quantize_weights_only_scalar()
    {
        try
        {
            std::cout << "DEBUG: test_quantize_weights_only_scalar called" << std::endl;
            // Create dummy data
            float weights[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
            ryzanstein_llm::bitnet::QuantConfig config;

            // Call the function but don't use the result
            ryzanstein_llm::bitnet::quantize_weights_ternary_scalar(weights, 4, 4, config);

            std::cout << "DEBUG: quantize_weights_ternary_scalar completed successfully" << std::endl;
            return 1; // Success
        }
        catch (const std::exception &e)
        {
            std::cout << "DEBUG: Exception in quantize_weights_ternary_scalar: " << e.what() << std::endl;
            return -1; // Failure
        }
        catch (...)
        {
            std::cout << "DEBUG: Unknown exception in quantize_weights_ternary_scalar" << std::endl;
            return -2; // Failure
        }
    }

    // Test function that replicates test_basic_quantize_ops but with larger data
    RYZEN_EXPORT int test_basic_quantize_large()
    {
        try
        {
            // Same data as test_quantize_no_vector
            const int size = 16;
            float weights[size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                   9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
            float mean_abs = 0.0f;

            // Compute mean absolute value
            for (int i = 0; i < size; ++i)
            {
                mean_abs += std::abs(weights[i]);
            }
            mean_abs /= size;

            // Threshold
            const float threshold = 0.7f * mean_abs;

            // Quantize
            int8_t result[size];
            for (int i = 0; i < size; ++i)
            {
                const float w = weights[i];
                if (std::abs(w) > threshold)
                {
                    result[i] = (w > 0) ? 1 : -1;
                }
                else
                {
                    result[i] = 0;
                }
            }

            return 1; // Success
        }
        catch (const std::exception &e)
        {
            return -1; // Failure
        }
        catch (...)
        {
            return -2; // Failure
        }
    }

    // Quantization functions
    RYZEN_EXPORT int test_quantize_weights_only(const float *weights, uint32_t rows, uint32_t cols)
    {
        std::cout << "DEBUG: test_quantize_weights_only called" << std::endl;
        // Just do some basic computation without creating objects
        float sum = 0.0f;
        for (uint32_t i = 0; i < rows * cols; ++i)
        {
            sum += weights[i];
        }
        return static_cast<int>(sum * 1000); // Return scaled sum
    }
    RYZEN_EXPORT void *quantize_weights_ternary_c(
        const float *weights, uint32_t rows, uint32_t cols)
    {
        ryzanstein_llm::bitnet::QuantConfig config;
        ryzanstein_llm::bitnet::TernaryWeightCPU result =
            ryzanstein_llm::bitnet::quantize_weights_ternary_scalar(weights, rows, cols, config);
        ryzanstein_llm::bitnet::TernaryWeightCPU *result_ptr =
            new ryzanstein_llm::bitnet::TernaryWeightCPU(result);
        return result_ptr;
    }

    RYZEN_EXPORT void *quantize_activations_int8_c(
        const float *activations, size_t size)
    {
        ryzanstein_llm::bitnet::QuantConfig config;
        ryzanstein_llm::bitnet::QuantizedActivationCPU result =
            ryzanstein_llm::bitnet::quantize_activations_int8_scalar(activations, size, config);
        ryzanstein_llm::bitnet::QuantizedActivationCPU *result_ptr =
            new ryzanstein_llm::bitnet::QuantizedActivationCPU(result);
        return result_ptr;
    }

    RYZEN_EXPORT void dequantize_weights_c(
        void *ternary_weight_ptr, float *output, uint32_t rows, uint32_t cols)
    {
        auto *ternary_weight = static_cast<ryzanstein_llm::bitnet::TernaryWeightCPU *>(ternary_weight_ptr);
        ryzanstein_llm::bitnet::dequantize_weights_scalar(*ternary_weight, output, rows, cols);
    }

    RYZEN_EXPORT void dequantize_activations_c(
        void *quantized_ptr, float *output, size_t size)
    {
        auto *quantized = static_cast<ryzanstein_llm::bitnet::QuantizedActivationCPU *>(quantized_ptr);
        ryzanstein_llm::bitnet::dequantize_activations_scalar(*quantized, output, size);
    }

    RYZEN_EXPORT float compute_quantization_error_c(
        const float *original, const float *quantized, size_t size)
    {
        return ryzanstein_llm::bitnet::compute_quantization_error_scalar(original, quantized, size);
    }

    // Memory management
    RYZEN_EXPORT void free_ternary_weight(void *ptr)
    {
        delete static_cast<ryzanstein_llm::bitnet::TernaryWeight *>(ptr);
    }

    RYZEN_EXPORT void free_quantized_activation(void *ptr)
    {
        delete static_cast<ryzanstein_llm::bitnet::QuantizedActivation *>(ptr);
    }

    // Test floating-point accumulation operations
    RYZEN_EXPORT int test_floating_point_accumulation()
    {
        // Test basic floating-point operations
        float a = 1.0f;
        float b = 2.0f;
        float sum = a + b;

        // Test std::abs
        float val = -3.5f;
        float abs_val = std::abs(val);

        // Test accumulation loop (this is where it fails)
        float sum_abs = 0.0f;
        float weights[] = {1.0f, -2.0f, 3.0f, -4.0f};
        int num_weights = 4;

        for (int i = 0; i < num_weights; ++i)
        {
            float abs_weight = std::abs(weights[i]);
            sum_abs += abs_weight; // This line likely triggers the error
        }

        return static_cast<int>(sum_abs * 1000); // Return scaled integer
    }

    // Test division operations
    RYZEN_EXPORT int test_division_operations()
    {
        // Test basic division
        float numerator = 10.0f;
        float denominator = 4.0f;
        float result = numerator / denominator;

        // Test division by zero protection
        float safe_div = (denominator > 0) ? (numerator / denominator) : 1.0f;

        // Test the exact division from quantization
        float sum_abs_original = 10.0f;
        float sum_abs_quantized = 5.0f;
        float scale = (sum_abs_quantized > 0) ? (sum_abs_original / sum_abs_quantized) : 1.0f;

        // Test mean calculation
        float total = 0.0f;
        float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
        int count = 4;
        for (int i = 0; i < count; ++i)
        {
            total += values[i];
        }
        float mean = total / count;

        return static_cast<int>(scale * 1000);
    }

    // Test std::vector operations
    RYZEN_EXPORT int test_vector_operations()
    {
        // Test basic vector creation
        std::vector<int8_t> values;

        // Test resize operation (this might trigger AVX-512)
        size_t size = 16;
        values.resize(size);

        // Test filling vector
        for (size_t i = 0; i < size; ++i)
        {
            values[i] = static_cast<int8_t>(i % 3 - 1); // -1, 0, 1 pattern
        }

        // Test float vector
        std::vector<float> scales;
        scales.resize(1);
        scales[0] = 2.5f;

        // Test larger resize
        size_t large_size = 64;
        values.resize(large_size);

        // Test multiple groups resize (like TernaryWeight)
        uint32_t r = 4, c = 4, gs = 8;
        uint32_t total_size = r * c;
        uint32_t num_groups = (gs > 0) ? ((total_size + gs - 1) / gs) : 1;

        std::vector<int8_t> test_values;
        std::vector<float> test_scales;

        test_values.resize(total_size);
        test_scales.resize(num_groups);

        return static_cast<int>(values.size() + scales.size() + test_values.size() + test_scales.size());
    }

    // Test avoiding std::min_element and std::max_element
    RYZEN_EXPORT int test_min_max_avoidance()
    {
        // Test data - same as used in quantize_activations_int8_scalar
        float activations[4] = {1.0f, -2.0f, 3.0f, -4.0f};
        size_t size = 4;

        // Manual min/max calculation instead of std::min_element/max_element
        float min_val = activations[0];
        float max_val = activations[0];

        for (size_t i = 1; i < size; ++i)
        {
            if (activations[i] < min_val)
                min_val = activations[i];
            if (activations[i] > max_val)
                max_val = activations[i];
        }

        // Continue with quantization logic (simplified)
        float abs_max = (fabs(min_val) > fabs(max_val)) ? fabs(min_val) : fabs(max_val);
        float scale = (abs_max > 0.0f) ? (127.0f / abs_max) : 1.0f;

        // Test quantization
        int8_t quantized[4];
        for (size_t i = 0; i < size; ++i)
        {
            float clamped = (activations[i] < -abs_max) ? -abs_max : ((activations[i] > abs_max) ? abs_max : activations[i]);
            quantized[i] = static_cast<int8_t>(round(clamped * scale));
        }

        return 42; // Success
    }

    RYZEN_EXPORT int test_int8_vector_operations()
    {
        try
        {
            // Test std::vector<int8_t> operations
            std::vector<int8_t> values;

            values.resize(16);

            // Fill with test data
            for (size_t i = 0; i < 16; ++i)
            {
                values[i] = static_cast<int8_t>(i - 8); // -8 to 7
            }

            // Test reading back
            int sum = 0;
            for (size_t i = 0; i < 16; ++i)
            {
                sum += values[i];
            }

            return sum;
        }
        catch (const std::exception &e)
        {
            return -1;
        }
        catch (...)
        {
            return -2;
        }
    }

    // CPU-compatible quantization functions (no std::vector)

    RYZEN_EXPORT void *quantize_weights_ternary_cpu_c(
        const float *weights, uint32_t rows, uint32_t cols)
    {
        ryzanstein_llm::bitnet::QuantConfig config;
        ryzanstein_llm::bitnet::TernaryWeightCPU *result =
            new ryzanstein_llm::bitnet::TernaryWeightCPU(
                ryzanstein_llm::bitnet::quantize_weights_ternary_cpu(weights, rows, cols, config));
        return result;
    }

    RYZEN_EXPORT void *quantize_activations_int8_cpu_c(
        const float *activations, size_t size)
    {
        ryzanstein_llm::bitnet::QuantConfig config;
        ryzanstein_llm::bitnet::QuantizedActivationCPU *result =
            new ryzanstein_llm::bitnet::QuantizedActivationCPU(
                ryzanstein_llm::bitnet::quantize_activations_int8_cpu(activations, size, config));
        return result;
    }

    RYZEN_EXPORT void dequantize_weights_cpu_c(
        void *ternary_weight_ptr, float *output)
    {
        auto *ternary_weight = static_cast<ryzanstein_llm::bitnet::TernaryWeightCPU *>(ternary_weight_ptr);
        ryzanstein_llm::bitnet::dequantize_weights_cpu(*ternary_weight, output);
    }

    RYZEN_EXPORT void dequantize_activations_cpu_c(
        void *quantized_ptr, float *output)
    {
        auto *quantized = static_cast<ryzanstein_llm::bitnet::QuantizedActivationCPU *>(quantized_ptr);
        ryzanstein_llm::bitnet::dequantize_activations_cpu(*quantized, output);
    }

    RYZEN_EXPORT float compute_quantization_error_cpu_c(
        const float *original, const float *quantized, size_t size)
    {
        return ryzanstein_llm::bitnet::compute_quantization_error_cpu(original, quantized, size);
    }

    // Memory management for CPU-compatible structs
    RYZEN_EXPORT void free_ternary_weight_cpu(void *ptr)
    {
        delete static_cast<ryzanstein_llm::bitnet::TernaryWeightCPU *>(ptr);
    }

    RYZEN_EXPORT void free_quantized_activation_cpu(void *ptr)
    {
        delete static_cast<ryzanstein_llm::bitnet::QuantizedActivationCPU *>(ptr);
    }

    // Test functions for CPU-compatible quantization

    RYZEN_EXPORT int test_ternary_weight_cpu_constructor()
    {
        try
        {
            // Just create the object and return success
            ryzanstein_llm::bitnet::TernaryWeightCPU weight(4, 4, 0);
            // Clean up
            delete[] weight.values;
            delete[] weight.scales;
            return 42;
        }
        catch (const std::exception &e)
        {
            return -1;
        }
        catch (...)
        {
            return -2;
        }
    }

    RYZEN_EXPORT int test_quantize_weights_cpu_simple()
    {
        try
        {
            // Create test data
            float weights[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

            ryzanstein_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;

            auto ternary_weight = ryzanstein_llm::bitnet::quantize_weights_ternary_cpu(weights, 4, 4, config);

            // Clean up
            delete[] ternary_weight.values;
            delete[] ternary_weight.scales;

            return 1;
        }
        catch (const std::exception &e)
        {
            return -1;
        }
        catch (...)
        {
            return -2;
        }
    }

    RYZEN_EXPORT int test_quantize_activations_cpu()
    {
        try
        {
            // Create test data
            float activations[4] = {1, -2, 3, -4};

            ryzanstein_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;

            auto quantized_activation = ryzanstein_llm::bitnet::quantize_activations_int8_cpu(activations, 4, config);

            // Clean up
            delete[] quantized_activation.values;

            return 1;
        }
        catch (const std::exception &e)
        {
            return -1;
        }
        catch (...)
        {
            return -2;
        }
    }

    RYZEN_EXPORT int test_full_quantization_cpu()
    {
        try
        {
            // Create test data
            float weights[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            float activations[4] = {1, -2, 3, -4};

            ryzanstein_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;

            auto ternary_weight = ryzanstein_llm::bitnet::quantize_weights_ternary_cpu(weights, 4, 4, config);
            auto quantized_activation = ryzanstein_llm::bitnet::quantize_activations_int8_cpu(activations, 4, config);

            // Test dequantization
            float dequantized_weights[16];
            float dequantized_activations[4];

            ryzanstein_llm::bitnet::dequantize_weights_cpu(ternary_weight, dequantized_weights);
            ryzanstein_llm::bitnet::dequantize_activations_cpu(quantized_activation, dequantized_activations);

            // Test error computation
            float error_weights = ryzanstein_llm::bitnet::compute_quantization_error_cpu(
                weights, dequantized_weights, 16);
            float error_activations = ryzanstein_llm::bitnet::compute_quantization_error_cpu(
                activations, dequantized_activations, 4);

            // Clean up
            delete[] ternary_weight.values;
            delete[] ternary_weight.scales;
            delete[] quantized_activation.values;

            return 1;
        }
        catch (const std::exception &e)
        {
            return -1;
        }
        catch (...)
        {
            return -2;
        }
    }
} // Close extern "C"

// ============================================================================
// Python Bindings (pybind11)
// ============================================================================

namespace py = pybind11;

PYBIND11_MODULE(ryzen_llm_bindings, m)
{
    m.doc() = "Ryzanstein LLM C++ Bindings: BitNet Quantization and Inference";

    // ========================================================================
    // Quantization Configuration
    // ========================================================================

    py::class_<ryzanstein_llm::bitnet::QuantConfig>(m, "QuantConfig")
        .def(py::init<>())
        .def_readwrite("per_group_scaling", &ryzanstein_llm::bitnet::QuantConfig::per_group_scaling)
        .def_readwrite("weight_group_size", &ryzanstein_llm::bitnet::QuantConfig::weight_group_size)
        .def_readwrite("activation_clip_value", &ryzanstein_llm::bitnet::QuantConfig::activation_clip_value)
        .def_readwrite("symmetric_activations", &ryzanstein_llm::bitnet::QuantConfig::symmetric_activations)
        .def("__repr__", [](const ryzanstein_llm::bitnet::QuantConfig &c)
             { return "<QuantConfig per_group_scaling=" + std::to_string(c.per_group_scaling) +
                      " weight_group_size=" + std::to_string(c.weight_group_size) +
                      " activation_clip_value=" + std::to_string(c.activation_clip_value) + ">"; });

    // ========================================================================
    // Ternary Weight Representation
    // ========================================================================

    py::class_<ryzanstein_llm::bitnet::TernaryWeight>(m, "TernaryWeight")
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t, uint32_t>())
        .def_readonly("values", &ryzanstein_llm::bitnet::TernaryWeight::values)
        .def_readonly("scales", &ryzanstein_llm::bitnet::TernaryWeight::scales)
        .def_readonly("rows", &ryzanstein_llm::bitnet::TernaryWeight::rows)
        .def_readonly("cols", &ryzanstein_llm::bitnet::TernaryWeight::cols)
        .def_readonly("group_size", &ryzanstein_llm::bitnet::TernaryWeight::group_size)
        .def("get_scale", &ryzanstein_llm::bitnet::TernaryWeight::get_scale)
        .def("size", [](const ryzanstein_llm::bitnet::TernaryWeight &w)
             { return w.rows * w.cols; })
        .def("num_scales", [](const ryzanstein_llm::bitnet::TernaryWeight &w)
             { return w.scales.size(); })
        .def("__repr__", [](const ryzanstein_llm::bitnet::TernaryWeight &w)
             { return "<TernaryWeight rows=" + std::to_string(w.rows) +
                      " cols=" + std::to_string(w.cols) +
                      " scales=" + std::to_string(w.scales.size()) + ">"; });

    // ========================================================================
    // Quantized Activation Representation
    // ========================================================================

    py::class_<ryzanstein_llm::bitnet::QuantizedActivation>(m, "QuantizedActivation")
        .def(py::init<>())
        .def(py::init<size_t>())
        .def_readonly("values", &ryzanstein_llm::bitnet::QuantizedActivation::values)
        .def_readwrite("scale", &ryzanstein_llm::bitnet::QuantizedActivation::scale)
        .def_readwrite("zero_point", &ryzanstein_llm::bitnet::QuantizedActivation::zero_point)
        .def("size", [](const ryzanstein_llm::bitnet::QuantizedActivation &a)
             { return a.values.size(); })
        .def("__repr__", [](const ryzanstein_llm::bitnet::QuantizedActivation &a)
             { return "<QuantizedActivation size=" + std::to_string(a.values.size()) +
                      " scale=" + std::to_string(a.scale) + ">"; });

    // ========================================================================
    // Quantization Functions
    // ========================================================================

    m.def("quantize_weights_ternary", [](py::array_t<float> weights, uint32_t rows, uint32_t cols, const ryzanstein_llm::bitnet::QuantConfig &config) -> ryzanstein_llm::bitnet::TernaryWeight
          {
              auto buf = weights.request();
              if (buf.size != rows * cols) {
                  throw std::runtime_error("weights size mismatch: expected " + 
                                         std::to_string(rows * cols) + ", got " + 
                                         std::to_string(buf.size));
              }
              return ryzanstein_llm::bitnet::quantize_weights_ternary(
                  static_cast<float*>(buf.ptr), rows, cols, config); }, py::arg("weights"), py::arg("rows"), py::arg("cols"), py::arg("config") = ryzanstein_llm::bitnet::QuantConfig(), "Quantize FP32 weights to ternary {-1, 0, +1}\n\n"
                                                                                                                                "Args:\n"
                                                                                                                                "  weights: FP32 weight array [rows x cols]\n"
                                                                                                                                "  rows: Number of rows\n"
                                                                                                                                "  cols: Number of columns\n"
                                                                                                                                "  config: QuantConfig instance\n\n"
                                                                                                                                "Returns:\n"
                                                                                                                                "  TernaryWeight with quantized values and scales");

    m.def("quantize_activations_int8", [](py::array_t<float> activations, const ryzanstein_llm::bitnet::QuantConfig &config) -> ryzanstein_llm::bitnet::QuantizedActivation
          {
              auto buf = activations.request();
              return ryzanstein_llm::bitnet::quantize_activations_int8(
                  static_cast<float*>(buf.ptr), buf.size, config); }, py::arg("activations"), py::arg("config") = ryzanstein_llm::bitnet::QuantConfig(), "Quantize FP32 activations to INT8\n\n"
                                                                                                  "Args:\n"
                                                                                                  "  activations: FP32 activation array\n"
                                                                                                  "  config: QuantConfig instance\n\n"
                                                                                                  "Returns:\n"
                                                                                                  "  QuantizedActivation with quantized values and scale");

    m.def("dequantize_weights", [](const ryzanstein_llm::bitnet::TernaryWeight &weights) -> py::array_t<float>
          {
              auto output = new float[weights.rows * weights.cols];
              ryzanstein_llm::bitnet::dequantize_weights(weights, output);
              
              // Create numpy array that takes ownership
              py::capsule free_when_done(output, [](void *f) noexcept { delete[](float*)f; });
              return py::array_t<float>(
                  {weights.rows, weights.cols},
                  {weights.cols * sizeof(float), sizeof(float)},
                  output,
                  free_when_done); }, "Dequantize ternary weights back to FP32\n\n"
               "Args:\n"
               "  weights: TernaryWeight instance\n\n"
               "Returns:\n"
               "  FP32 array of shape [rows, cols]");

    m.def("dequantize_activations", [](const ryzanstein_llm::bitnet::QuantizedActivation &activations) -> py::array_t<float>
          {
              auto output = new float[activations.values.size()];
              ryzanstein_llm::bitnet::dequantize_activations(activations, output);
              
              py::capsule free_when_done(output, [](void *f) noexcept { delete[](float*)f; });
              return py::array_t<float>(
                  {activations.values.size()},
                  {sizeof(float)},
                  output,
                  free_when_done); }, "Dequantize INT8 activations back to FP32\n\n"
               "Args:\n"
               "  activations: QuantizedActivation instance\n\n"
               "Returns:\n"
               "  FP32 array");

    m.def("compute_quantization_error", [](py::array_t<float> original, py::array_t<float> quantized) -> float
          {
              auto orig_buf = original.request();
              auto quant_buf = quantized.request();
              if (orig_buf.size != quant_buf.size) {
                  throw std::runtime_error("array size mismatch");
              }
              return ryzanstein_llm::bitnet::compute_quantization_error(
                  static_cast<float*>(orig_buf.ptr),
                  static_cast<float*>(quant_buf.ptr),
                  orig_buf.size); }, "Compute mean squared error between original and quantized values\n\n"
               "Args:\n"
               "  original: Original FP32 values\n"
               "  quantized: Quantized FP32 values\n\n"
               "Returns:\n"
               "  Mean squared error");

    // ========================================================================
    // Legacy Test Functions
    // ========================================================================

    m.def("test_function", []()
          { return 42; }, "Legacy test function");

    m.def("test_quantize_scalar", [](py::array_t<float> weights, uint32_t rows, uint32_t cols)
          {
        auto buf = weights.request();
        if (buf.size != rows * cols) {
            throw std::runtime_error("weights size mismatch");
        }
        
        auto ternary = ryzanstein_llm::bitnet::quantize_weights_ternary(
            static_cast<float*>(buf.ptr), rows, cols);
        
        return py::dict(
            py::arg("num_values") = (int32_t)ternary.values.size(),
            py::arg("num_scales") = (int32_t)ternary.scales.size(),
            py::arg("shape") = py::tuple(py::cast(std::vector<uint32_t>{ternary.rows, ternary.cols}))
        ); }, "Test ternary quantization and return metadata");

    // ========================================================================
    // Model Configuration
    // ========================================================================

    py::class_<ryzanstein_llm::bitnet::ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &ryzanstein_llm::bitnet::ModelConfig::vocab_size)
        .def_readwrite("hidden_size", &ryzanstein_llm::bitnet::ModelConfig::hidden_size)
        .def_readwrite("intermediate_size", &ryzanstein_llm::bitnet::ModelConfig::intermediate_size)
        .def_readwrite("num_layers", &ryzanstein_llm::bitnet::ModelConfig::num_layers)
        .def_readwrite("num_heads", &ryzanstein_llm::bitnet::ModelConfig::num_heads)
        .def_readwrite("head_dim", &ryzanstein_llm::bitnet::ModelConfig::head_dim)
        .def_readwrite("max_seq_length", &ryzanstein_llm::bitnet::ModelConfig::max_seq_length)
        .def_readwrite("rms_norm_eps", &ryzanstein_llm::bitnet::ModelConfig::rms_norm_eps)
        .def_readwrite("use_tmac", &ryzanstein_llm::bitnet::ModelConfig::use_tmac)
        .def_readwrite("use_speculative_decoding", &ryzanstein_llm::bitnet::ModelConfig::use_speculative_decoding)
        .def_readwrite("speculative_k", &ryzanstein_llm::bitnet::ModelConfig::speculative_k)
        .def("__repr__", [](const ryzanstein_llm::bitnet::ModelConfig &c)
             { return "<ModelConfig vocab=" + std::to_string(c.vocab_size) +
                      " hidden=" + std::to_string(c.hidden_size) +
                      " layers=" + std::to_string(c.num_layers) + ">"; });

    // ========================================================================
    // Generation Configuration
    // ========================================================================

    py::class_<ryzanstein_llm::bitnet::GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("max_tokens", &ryzanstein_llm::bitnet::GenerationConfig::max_tokens)
        .def_readwrite("temperature", &ryzanstein_llm::bitnet::GenerationConfig::temperature)
        .def_readwrite("top_k", &ryzanstein_llm::bitnet::GenerationConfig::top_k)
        .def_readwrite("top_p", &ryzanstein_llm::bitnet::GenerationConfig::top_p)
        .def_readwrite("repetition_penalty", &ryzanstein_llm::bitnet::GenerationConfig::repetition_penalty)
        .def_readwrite("seed", &ryzanstein_llm::bitnet::GenerationConfig::seed)
        .def("__repr__", [](const ryzanstein_llm::bitnet::GenerationConfig &c)
             { return "<GenerationConfig max_tokens=" + std::to_string(c.max_tokens) +
                      " temp=" + std::to_string(c.temperature) +
                      " top_k=" + std::to_string(c.top_k) + ">"; });

    // ========================================================================
    // BitNet Inference Engine
    // ========================================================================

    py::class_<ryzanstein_llm::bitnet::BitNetEngine>(m, "BitNetEngine")
        .def(py::init<const ryzanstein_llm::bitnet::ModelConfig &>(), py::arg("config"))
        .def("load_weights", &ryzanstein_llm::bitnet::BitNetEngine::load_weights,
             py::arg("weights_path"),
             "Load model weights from file")
        .def("generate", &ryzanstein_llm::bitnet::BitNetEngine::generate,
             py::arg("input_tokens"),
             py::arg("gen_config") = ryzanstein_llm::bitnet::GenerationConfig(),
             "Generate tokens from input")
        .def("forward", &ryzanstein_llm::bitnet::BitNetEngine::forward,
             py::arg("token_id"),
             py::arg("position"),
             "Single forward pass, returns logits")
        .def("reset_cache", &ryzanstein_llm::bitnet::BitNetEngine::reset_cache,
             "Reset the KV cache for new sequence")
        .def("get_config", &ryzanstein_llm::bitnet::BitNetEngine::get_config,
             py::return_value_policy::reference,
             "Get model configuration");
}
