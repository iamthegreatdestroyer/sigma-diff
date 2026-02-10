#include "safetensors_loader.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <limits>
#include <cmath>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define OPEN_FILE(fname) _open(fname, _O_RDONLY | _O_BINARY)
#define CLOSE_FILE(fd) _close(fd)
#define FILE_HANDLE int
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#define OPEN_FILE(fname) open(fname, O_RDONLY)
#define CLOSE_FILE(fd) close(fd)
#define FILE_HANDLE int
#endif

namespace ryzanstein_llm
{
    namespace io
    {

        SafeTensorsLoader::SafeTensorsLoader() : verbose_(false) {}

        uint64_t SafeTensorsLoader::read_u64_le_(const uint8_t *bytes)
        {
            return bytes_to_uint64(bytes);
        }

        uint32_t SafeTensorsLoader::read_u32_le_(const uint8_t *bytes)
        {
            return bytes_to_int32(bytes);
        }

        std::string SafeTensorsLoader::LoaderStats::report() const
        {
            std::ostringstream oss;
            oss << "\n=== SafeTensors Loader Statistics ===\n";
            oss << "Total Tensors: " << total_tensors << "\n";
            oss << "Total Parameters: " << total_parameters << " ("
                << (total_parameters / 1e9) << "B)\n";
            oss << "Total Bytes: " << total_bytes << " ("
                << (total_bytes / 1e9) << "GB)\n";
            oss << "Load Time: " << load_time_seconds << " seconds\n";
            if (load_time_seconds > 0)
            {
                oss << "Throughput: "
                    << (total_bytes / 1e9 / load_time_seconds) << " GB/s\n";
            }
            oss << "=====================================\n";
            return oss.str();
        }

        std::map<std::string, Tensor> SafeTensorsLoader::load(
            const std::string &filename,
            bool use_mmap)
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            last_stats_ = LoaderStats();

            if (verbose_)
            {
                std::cerr << "Loading SafeTensors file: " << filename << "\n";
            }

            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            try
            {
                // Read metadata
                auto metadata = read_header_(filename, file);
                validate_file_structure_(filename, metadata);

                std::map<std::string, Tensor> tensors;
                uint64_t param_count = 0;
                uint64_t byte_count = 0;

                // Load each tensor
                for (const auto &[name, meta] : metadata)
                {
                    if (verbose_)
                    {
                        std::cerr << "Loading tensor: " << name << " [";
                        for (size_t i = 0; i < meta.shape.size(); ++i)
                        {
                            if (i > 0)
                                std::cerr << ", ";
                            std::cerr << meta.shape[i];
                        }
                        std::cerr << "]\n";
                    }

                    // Read tensor data
                    auto data = load_tensor_data_(file, meta.data_offset, meta.data_length);

                    Tensor tensor(name, meta.shape, meta.dtype);
                    tensor.data = std::move(data);

                    param_count += meta.num_elements();
                    byte_count += tensor.total_bytes();

                    tensors[name] = std::move(tensor);
                }

                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end_time - start_time;

                last_stats_.total_tensors = tensors.size();
                last_stats_.total_parameters = param_count;
                last_stats_.total_bytes = byte_count;
                last_stats_.load_time_seconds = elapsed.count();

                if (verbose_)
                {
                    std::cerr << last_stats_.report();
                }

                return tensors;
            }
            catch (const std::exception &e)
            {
                file.close();
                throw;
            }
        }

        std::map<std::string, Tensor> SafeTensorsLoader::load_quantized(
            const std::string &filename,
            bool quantize_to_int8)
        {
            auto tensors = load(filename, true);

            if (!quantize_to_int8)
            {
                return tensors;
            }

            // Quantize float32 tensors to int8
            std::map<std::string, Tensor> quantized;

            for (auto &[name, tensor] : tensors)
            {
                if (tensor.dtype == DataType::FLOAT32)
                {
                    if (verbose_)
                    {
                        std::cerr << "Quantizing " << name << " to int8\n";
                    }

                    // Compute scale factor from max absolute value
                    auto *data = tensor.data_ptr<float>();
                    float max_abs = 0.0f;
                    for (uint64_t i = 0; i < tensor.num_elements(); ++i)
                    {
                        float val = std::abs(data[i]);
                        if (std::isfinite(val))
                        {
                            max_abs = std::max(max_abs, val);
                        }
                    }

                    float scale = (max_abs > 0) ? (127.0f / max_abs) : 1.0f;

                    // Create int8 tensor
                    Tensor int8_tensor(name, tensor.shape, DataType::INT8);
                    int8_tensor.data.resize(tensor.num_elements());

                    auto *int8_data = int8_tensor.data_ptr<int8_t>();
                    for (uint64_t i = 0; i < tensor.num_elements(); ++i)
                    {
                        int8_data[i] = float_to_int8_(data[i], scale);
                    }

                    quantized[name] = std::move(int8_tensor);
                }
                else
                {
                    quantized[name] = std::move(tensor);
                }
            }

            return quantized;
        }

        std::map<std::string, TensorMetadata> SafeTensorsLoader::load_metadata(
            const std::string &filename)
        {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            return read_header_(filename, file);
        }

        uint64_t SafeTensorsLoader::get_file_size(const std::string &filename)
        {
            std::ifstream file(filename, std::ios::binary | std::ios::ate);
            if (!file.is_open())
            {
                throw std::runtime_error("Cannot open file: " + filename);
            }
            return file.tellg();
        }

        std::map<std::string, TensorMetadata> SafeTensorsLoader::read_header_(
            const std::string &filename,
            std::ifstream &file)
        {
            // Read header size (8 bytes, little-endian u64)
            uint8_t header_size_bytes[8];
            file.read(reinterpret_cast<char *>(header_size_bytes), 8);

            if (file.gcount() != 8)
            {
                throw std::runtime_error("Failed to read header size from " + filename);
            }

            uint64_t header_size = read_u64_le_(header_size_bytes);

            if (header_size == 0 || header_size > 100 * 1024 * 1024)
            {
                throw std::runtime_error(
                    "Invalid header size: " + std::to_string(header_size) +
                    " (expected < 100MB)");
            }

            // Read header JSON
            std::vector<char> header_buffer(header_size);
            file.read(header_buffer.data(), header_size);

            if (file.gcount() != static_cast<std::streamsize>(header_size))
            {
                throw std::runtime_error("Failed to read header from " + filename);
            }

            std::string json_str(header_buffer.begin(), header_buffer.end());
            uint64_t data_offset = 8 + header_size;

            return parse_json_header_(json_str, data_offset);
        }

        std::map<std::string, TensorMetadata> SafeTensorsLoader::parse_json_header_(
            const std::string &json_str,
            uint64_t data_offset)
        {
            std::map<std::string, TensorMetadata> metadata;

            // Simple JSON parser for SafeTensors header
            // Expected format: {"tensor_name": {"dtype": "F32", "shape": [d0, d1, ...], "data_offsets": [start, end]}, ...}

            size_t pos = 0;
            while (pos < json_str.size())
            {
                // Find next tensor definition
                size_t quote_pos = json_str.find('"', pos);
                if (quote_pos == std::string::npos)
                    break;

                size_t name_end = json_str.find('"', quote_pos + 1);
                if (name_end == std::string::npos)
                    break;

                std::string tensor_name = json_str.substr(quote_pos + 1, name_end - quote_pos - 1);

                // Skip metadata key
                if (tensor_name == "__metadata__")
                {
                    pos = name_end + 1;
                    continue;
                }

                // Find dtype
                size_t dtype_pos = json_str.find("\"dtype\"", name_end);
                if (dtype_pos == std::string::npos)
                {
                    pos = name_end + 1;
                    continue;
                }

                size_t dtype_colon = json_str.find(':', dtype_pos);
                size_t dtype_quote1 = json_str.find('"', dtype_colon);
                size_t dtype_quote2 = json_str.find('"', dtype_quote1 + 1);

                std::string dtype_str = json_str.substr(dtype_quote1 + 1, dtype_quote2 - dtype_quote1 - 1);

                // Find shape
                size_t shape_pos = json_str.find("\"shape\"", dtype_pos);
                if (shape_pos == std::string::npos)
                {
                    pos = name_end + 1;
                    continue;
                }

                size_t shape_bracket1 = json_str.find('[', shape_pos);
                size_t shape_bracket2 = json_str.find(']', shape_bracket1);

                std::string shape_str = json_str.substr(shape_bracket1 + 1, shape_bracket2 - shape_bracket1 - 1);

                std::vector<uint64_t> shape;
                std::istringstream shape_stream(shape_str);
                std::string dim_str;
                while (std::getline(shape_stream, dim_str, ','))
                {
                    // Trim whitespace
                    dim_str.erase(0, dim_str.find_first_not_of(" \t"));
                    dim_str.erase(dim_str.find_last_not_of(" \t") + 1);

                    if (!dim_str.empty())
                    {
                        shape.push_back(std::stoull(dim_str));
                    }
                }

                // Find data_offsets
                size_t offset_pos = json_str.find("\"data_offsets\"", shape_pos);
                if (offset_pos == std::string::npos)
                {
                    pos = name_end + 1;
                    continue;
                }

                size_t offset_bracket1 = json_str.find('[', offset_pos);
                size_t offset_bracket2 = json_str.find(']', offset_bracket1);

                std::string offset_str = json_str.substr(offset_bracket1 + 1, offset_bracket2 - offset_bracket1 - 1);

                std::istringstream offset_stream(offset_str);
                std::vector<uint64_t> offsets;
                std::string off_str;
                while (std::getline(offset_stream, off_str, ','))
                {
                    off_str.erase(0, off_str.find_first_not_of(" \t"));
                    off_str.erase(off_str.find_last_not_of(" \t") + 1);

                    if (!off_str.empty())
                    {
                        offsets.push_back(std::stoull(off_str));
                    }
                }

                // Create metadata
                TensorMetadata meta;
                meta.name = tensor_name;
                meta.shape = shape;
                meta.dtype = TensorMetadata::parse_dtype(dtype_str);

                if (offsets.size() >= 2)
                {
                    meta.data_offset = data_offset + offsets[0];
                    meta.data_length = offsets[1] - offsets[0];
                }

                if (meta.dtype != DataType::UNKNOWN && meta.shape.size() > 0)
                {
                    metadata[tensor_name] = meta;
                }

                pos = offset_bracket2 + 1;
            }

            return metadata;
        }

        std::vector<uint8_t> SafeTensorsLoader::load_tensor_data_(
            std::ifstream &file,
            uint64_t offset,
            uint64_t length)
        {
            file.seekg(offset);

            std::vector<uint8_t> data(length);
            file.read(reinterpret_cast<char *>(data.data()), length);

            if (file.gcount() != static_cast<std::streamsize>(length))
            {
                throw std::runtime_error(
                    "Failed to read tensor data at offset " + std::to_string(offset) +
                    ", expected " + std::to_string(length) + " bytes");
            }

            return data;
        }

        int8_t SafeTensorsLoader::float_to_int8_(float value, float scale)
        {
            // Clamp to [-1, 1] range then scale
            if (!std::isfinite(value))
            {
                return 0;
            }

            float clamped = std::max(-1.0f, std::min(1.0f, value));
            float scaled = clamped * scale;

            // Round to nearest integer
            int32_t int_val = static_cast<int32_t>(std::round(scaled));

            // Clamp to int8 range
            return static_cast<int8_t>(
                std::max(-128, std::min(127, int_val)));
        }

        void SafeTensorsLoader::validate_file_structure_(
            const std::string &filename,
            const std::map<std::string, TensorMetadata> &metadata)
        {
            if (metadata.empty())
            {
                throw std::runtime_error("No tensors found in " + filename);
            }

            // Verify no overlapping tensor regions
            std::vector<std::pair<uint64_t, uint64_t>> regions;
            for (const auto &[name, meta] : metadata)
            {
                regions.push_back({meta.data_offset, meta.data_offset + meta.data_length});
            }

            // Simple check: regions should be non-overlapping
            std::sort(regions.begin(), regions.end());
            for (size_t i = 1; i < regions.size(); ++i)
            {
                if (regions[i].first < regions[i - 1].second)
                {
                    throw std::runtime_error(
                        "Overlapping tensor regions detected in " + filename);
                }
            }
        }

    } // namespace io
} // namespace ryzanstein_llm
