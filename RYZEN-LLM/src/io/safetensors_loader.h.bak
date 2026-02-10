#ifndef SAFETENSORS_LOADER_H
#define SAFETENSORS_LOADER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <iostream>

namespace ryzen_llm
{
    namespace io
    {

        /**
         * @enum DataType
         * @brief Supported data types in SafeTensors format
         */
        enum class DataType
        {
            FLOAT32,  // 32-bit floating point
            FLOAT16,  // 16-bit floating point
            INT8,     // 8-bit signed integer (quantized)
            INT32,    // 32-bit signed integer
            UINT8,    // 8-bit unsigned integer
            BFLOAT16, // Brain floating point
            UNKNOWN   // Unknown type
        };

        /**
         * @struct TensorMetadata
         * @brief Metadata describing a single tensor
         */
        struct TensorMetadata
        {
            std::string name;
            std::vector<uint64_t> shape;
            DataType dtype;
            uint64_t data_offset;
            uint64_t data_length;

            TensorMetadata() : dtype(DataType::UNKNOWN), data_offset(0), data_length(0) {}

            // Get total number of elements
            uint64_t num_elements() const
            {
                uint64_t count = 1;
                for (auto dim : shape)
                {
                    count *= dim;
                }
                return count;
            }

            // Get size in bytes for a given dtype
            static size_t dtype_size(DataType dtype)
            {
                switch (dtype)
                {
                case DataType::FLOAT32:
                case DataType::INT32:
                    return 4;
                case DataType::FLOAT16:
                case DataType::BFLOAT16:
                    return 2;
                case DataType::INT8:
                case DataType::UINT8:
                    return 1;
                default:
                    throw std::invalid_argument("Unknown data type");
                }
            }

            // Convert dtype string to enum
            static DataType parse_dtype(const std::string &dtype_str)
            {
                if (dtype_str == "F32" || dtype_str == "float32")
                    return DataType::FLOAT32;
                if (dtype_str == "F16" || dtype_str == "float16")
                    return DataType::FLOAT16;
                if (dtype_str == "I8" || dtype_str == "int8")
                    return DataType::INT8;
                if (dtype_str == "I32" || dtype_str == "int32")
                    return DataType::INT32;
                if (dtype_str == "U8" || dtype_str == "uint8")
                    return DataType::UINT8;
                if (dtype_str == "BF16" || dtype_str == "bfloat16")
                    return DataType::BFLOAT16;
                return DataType::UNKNOWN;
            }

            // Convert dtype enum to string
            static std::string dtype_to_string(DataType dtype)
            {
                switch (dtype)
                {
                case DataType::FLOAT32:
                    return "float32";
                case DataType::FLOAT16:
                    return "float16";
                case DataType::INT8:
                    return "int8";
                case DataType::INT32:
                    return "int32";
                case DataType::UINT8:
                    return "uint8";
                case DataType::BFLOAT16:
                    return "bfloat16";
                default:
                    return "unknown";
                }
            }
        };

        /**
         * @struct Tensor
         * @brief In-memory tensor representation with data ownership
         */
        struct Tensor
        {
            std::string name;
            std::vector<uint64_t> shape;
            DataType dtype;
            std::vector<uint8_t> data;

            Tensor() : dtype(DataType::UNKNOWN) {}

            Tensor(const std::string &n, const std::vector<uint64_t> &s, DataType dt)
                : name(n), shape(s), dtype(dt) {}

            uint64_t num_elements() const
            {
                uint64_t count = 1;
                for (auto dim : shape)
                {
                    count *= dim;
                }
                return count;
            }

            // Get data as typed pointer
            template <typename T>
            const T *data_ptr() const
            {
                return reinterpret_cast<const T *>(data.data());
            }

            template <typename T>
            T *data_ptr()
            {
                return reinterpret_cast<T *>(data.data());
            }

            size_t total_bytes() const
            {
                return data.size();
            }
        };

        /**
         * @class SafeTensorsLoader
         * @brief Production-grade parser for SafeTensors format files
         *
         * Handles:
         * - Binary deserialization from SafeTensors files
         * - Tensor metadata extraction
         * - Data type conversion and quantization
         * - Memory-mapped loading for large files
         * - Comprehensive error handling
         *
         * SafeTensors format structure:
         * [8 bytes: header_size (little-endian u64)]
         * [header_size bytes: JSON metadata]
         * [remaining bytes: tensor data]
         */
        class SafeTensorsLoader
        {
        public:
            SafeTensorsLoader();
            ~SafeTensorsLoader() = default;

            /**
             * @brief Load and parse a SafeTensors file
             * @param filename Path to the .safetensors file
             * @param use_mmap Enable memory mapping for large files (default: true)
             * @return Map of tensor names to Tensor objects
             * @throws std::runtime_error on parsing/IO errors
             */
            std::map<std::string, Tensor> load(
                const std::string &filename,
                bool use_mmap = true);

            /**
             * @brief Load a SafeTensors file with quantization to INT8
             * @param filename Path to the .safetensors file
             * @param quantize_to_int8 If true, convert float weights to int8
             * @return Map of tensor names to Tensor objects
             */
            std::map<std::string, Tensor> load_quantized(
                const std::string &filename,
                bool quantize_to_int8 = true);

            /**
             * @brief Extract metadata without loading tensor data
             * @param filename Path to the .safetensors file
             * @return Map of tensor names to metadata
             */
            std::map<std::string, TensorMetadata> load_metadata(
                const std::string &filename);

            /**
             * @brief Get the file size in bytes
             */
            uint64_t get_file_size(const std::string &filename);

            /**
             * @brief Get statistics about loaded tensors
             */
            struct LoaderStats
            {
                uint64_t total_tensors = 0;
                uint64_t total_parameters = 0;
                uint64_t total_bytes = 0;
                double load_time_seconds = 0.0;
                std::string report() const;
            };

            LoaderStats get_last_stats() const { return last_stats_; }

            /**
             * @brief Enable verbose logging
             */
            void set_verbose(bool verbose) { verbose_ = verbose; }

        private:
            LoaderStats last_stats_;
            bool verbose_;

            /**
             * @brief Read header from file and parse JSON metadata
             * @return Map of tensor names to their metadata
             */
            std::map<std::string, TensorMetadata> read_header_(
                const std::string &filename,
                std::ifstream &file);

            /**
             * @brief Parse JSON header string
             * Minimal JSON parser for SafeTensors format
             */
            std::map<std::string, TensorMetadata> parse_json_header_(
                const std::string &json_str,
                uint64_t data_offset);

            /**
             * @brief Load tensor data from file
             */
            std::vector<uint8_t> load_tensor_data_(
                std::ifstream &file,
                uint64_t offset,
                uint64_t length);

            /**
             * @brief Convert float32 to int8 quantized values
             */
            int8_t float_to_int8_(float value, float scale);

            /**
             * @brief Helper to read little-endian uint64
             */
            static uint64_t read_u64_le_(const uint8_t *bytes);

            /**
             * @brief Helper to read little-endian uint32
             */
            static uint32_t read_u32_le_(const uint8_t *bytes);

            /**
             * @brief Validate file format and structure
             */
            void validate_file_structure_(
                const std::string &filename,
                const std::map<std::string, TensorMetadata> &metadata);
        };

        /**
         * @brief Helper function to convert raw bytes to int32 (little-endian safe)
         */
        inline int32_t bytes_to_int32(const uint8_t *bytes)
        {
            return (bytes[0]) |
                   (bytes[1] << 8) |
                   (bytes[2] << 16) |
                   (bytes[3] << 24);
        }

        /**
         * @brief Helper function to convert raw bytes to uint64 (little-endian safe)
         */
        inline uint64_t bytes_to_uint64(const uint8_t *bytes)
        {
            return static_cast<uint64_t>(bytes[0]) |
                   (static_cast<uint64_t>(bytes[1]) << 8) |
                   (static_cast<uint64_t>(bytes[2]) << 16) |
                   (static_cast<uint64_t>(bytes[3]) << 24) |
                   (static_cast<uint64_t>(bytes[4]) << 32) |
                   (static_cast<uint64_t>(bytes[5]) << 40) |
                   (static_cast<uint64_t>(bytes[6]) << 48) |
                   (static_cast<uint64_t>(bytes[7]) << 56);
        }

    } // namespace io
} // namespace ryzen_llm

#endif // SAFETENSORS_LOADER_H
