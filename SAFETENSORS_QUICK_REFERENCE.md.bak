# SafeTensors Loader - Quick Reference Card

## Core Classes

### SafeTensorsLoader

```cpp
// Load entire file
std::map<std::string, Tensor> load(const std::string& filename,
                                    bool use_mmap = true);

// Load with quantization
std::map<std::string, Tensor> load_quantized(const std::string& filename,
                                            bool quantize_to_int8 = true);

// Get metadata only
std::map<std::string, TensorMetadata> load_metadata(const std::string& filename);

// Get file size
uint64_t get_file_size(const std::string& filename);

// Get load statistics
LoaderStats get_last_stats() const;

// Enable verbose output
void set_verbose(bool verbose);
```

### WeightValidator

```cpp
// Validate all weights
ValidationResult validate_bitnet_weights(
    const std::map<std::string, Tensor>& weights);

// Validate single tensor
std::vector<std::string> validate_tensor(
    const Tensor& tensor,
    const std::vector<uint64_t>& expected_shape = {});

// Check quantization
bool validate_quantization(const Tensor& tensor, float scale = 1.0f);

// Check for NaN/Inf
std::pair<bool, std::string> check_numerical_stability(
    const Tensor& tensor);

// Verify layer structure
bool verify_layer_structure(
    const std::map<std::string, Tensor>& weights);
```

## Data Types

| Type     | Enum                 | Size | Use             |
| -------- | -------------------- | ---- | --------------- |
| float32  | `DataType::FLOAT32`  | 4B   | Full precision  |
| float16  | `DataType::FLOAT16`  | 2B   | Mixed precision |
| int8     | `DataType::INT8`     | 1B   | Quantized       |
| bfloat16 | `DataType::BFLOAT16` | 2B   | Brain float     |
| int32    | `DataType::INT32`    | 4B   | Indices         |
| uint8    | `DataType::UINT8`    | 1B   | Unsigned        |

## Tensor Structure

```cpp
struct Tensor {
    std::string name;           // Tensor name
    std::vector<uint64_t> shape; // Dimensions
    DataType dtype;             // Data type
    std::vector<uint8_t> data;  // Raw data

    uint64_t num_elements() const;

    template<typename T>
    T* data_ptr();              // Get typed pointer
};
```

## Common Patterns

### Pattern 1: Load & Use

```cpp
SafeTensorsLoader loader;
auto tensors = loader.load("model.safetensors");

for (auto& [name, tensor] : tensors) {
    float* data = tensor.data_ptr<float>();
    // Use data...
}
```

### Pattern 2: Load & Validate

```cpp
SafeTensorsLoader loader;
auto tensors = loader.load("model.safetensors");

WeightValidator validator;
auto result = validator.validate_bitnet_weights(tensors);

if (result.is_valid) {
    std::cout << "✓ Weights valid\n";
}
```

### Pattern 3: Load with Quantization

```cpp
SafeTensorsLoader loader;
auto tensors = loader.load_quantized("model.safetensors", true);
// 28GB → 7GB, 4x memory savings
```

### Pattern 4: Fast Metadata Check

```cpp
SafeTensorsLoader loader;
auto metadata = loader.load_metadata("model.safetensors");

for (auto& [name, meta] : metadata) {
    std::cout << name << " [";
    for (auto dim : meta.shape) std::cout << dim << " ";
    std::cout << "]\n";
}
```

### Pattern 5: Error Handling

```cpp
try {
    auto tensors = loader.load("model.safetensors");
} catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << "\n";
}
```

## Performance Targets

```
Model: BitNet-7B (70B parameters, 28GB file)

Load Time:              3.2 seconds
Throughput:            8.75 GB/s
Quantized Load:        2.1 seconds
Metadata Only:        0.45 seconds

Memory:
  Float32:            28 GB
  Quantized (int8):    7 GB (75% savings)
```

## BitNet-7B Configuration

```cpp
BitNetConfig config;
config.hidden_size = 4096;      // d_model
config.num_heads = 32;          // n_heads
config.num_layers = 32;         // n_layers
config.intermediate_size = 11008; // d_ff
config.vocab_size = 32000;      // vocab
config.max_seq_length = 2048;   // max_seq
```

## Error Messages

| Error                         | Cause                      | Fix             |
| ----------------------------- | -------------------------- | --------------- |
| `Cannot open file`            | File missing/no permission | Check path      |
| `Invalid header size`         | Corrupted file             | Re-download     |
| `Failed to read header`       | I/O error                  | Check disk      |
| `Overlapping tensor regions`  | Corrupted metadata         | Verify checksum |
| `Tensor X data size mismatch` | Incomplete data            | Re-download     |

## CMake Integration

```cmake
add_library(safetensors_io
    src/io/safetensors_loader.cpp
    src/io/weight_validator.cpp
)

target_include_directories(safetensors_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(safetensors_io PUBLIC cxx_std_17)

target_link_libraries(my_target PRIVATE safetensors_io)
```

## Include Headers

```cpp
#include "src/io/safetensors_loader.h"
#include "src/io/weight_validator.h"

using namespace ryzen_llm::io;
```

## Verbose Output

```cpp
SafeTensorsLoader loader;
loader.set_verbose(true);

WeightValidator validator;
validator.set_verbose(true);

auto tensors = loader.load("model.safetensors");
auto result = validator.validate_bitnet_weights(tensors);

std::cout << loader.get_last_stats().report();
std::cout << result.report();
```

## Statistics

```cpp
// Get load statistics
auto stats = loader.get_last_stats();
std::cout << "Tensors: " << stats.total_tensors << "\n";
std::cout << "Params: " << stats.total_parameters << "\n";
std::cout << "Bytes: " << stats.total_bytes << "\n";
std::cout << "Time: " << stats.load_time_seconds << "s\n";
std::cout << stats.report();

// Get validation statistics
auto result = validator.validate_bitnet_weights(tensors);
std::cout << "Valid: " << result.is_valid << "\n";
std::cout << "Errors: " << result.errors.size() << "\n";
std::cout << "Parameters: " << result.total_parameters << "\n";
std::cout << result.report();
```

## Accessing Tensor Data

```cpp
// Float tensor
float* data = tensor.data_ptr<float>();
for (uint64_t i = 0; i < tensor.num_elements(); ++i) {
    float val = data[i];
}

// Int8 tensor
int8_t* data = tensor.data_ptr<int8_t>();

// Int32 tensor
int32_t* data = tensor.data_ptr<int32_t>();
```

## Validation Details

```cpp
auto result = validator.validate_bitnet_weights(tensors);

// Check overall validity
if (!result.is_valid) {
    // Print errors
    for (auto& err : result.errors) {
        std::cerr << "✗ " << err << "\n";
    }
}

// Check warnings
for (auto& warn : result.warnings) {
    std::cerr << "⚠ " << warn << "\n";
}

// Per-tensor validation
for (auto& [name, tv] : result.tensor_validations) {
    std::cout << name << ": ";
    std::cout << (tv.shape_valid ? "✓shape " : "✗shape ");
    std::cout << (tv.dtype_valid ? "✓dtype " : "✗dtype ");
    std::cout << (tv.data_valid ? "✓data " : "✗data ");
    std::cout << "\n";
}
```

## Files Created

```
src/io/
├── safetensors_loader.h         # Main header
├── safetensors_loader.cpp       # Parser implementation
├── weight_validator.h            # Validator header
├── weight_validator.cpp          # Validator implementation
├── CMakeLists.txt                # Build config
└── README_SAFETENSORS.md         # Full documentation

tests/
└── test_safetensors_loader.cpp   # Examples & tests

docs/
└── SAFETENSORS_INTEGRATION_GUIDE.md # Integration guide
```

## Quick Start

```cpp
// 1. Include
#include "src/io/safetensors_loader.h"
#include "src/io/weight_validator.h"
using namespace ryzen_llm::io;

// 2. Load
SafeTensorsLoader loader;
auto tensors = loader.load("bitnet-7b.safetensors");

// 3. Validate
WeightValidator validator;
auto result = validator.validate_bitnet_weights(tensors);

// 4. Use
if (result.is_valid) {
    for (auto& [name, tensor] : tensors) {
        float* data = tensor.data_ptr<float>();
        // Inference code here...
    }
}

// 5. Statistics
std::cout << loader.get_last_stats().report();
std::cout << result.report();
```

---

For complete documentation, see:

- `README_SAFETENSORS.md` - Full API reference
- `SAFETENSORS_INTEGRATION_GUIDE.md` - Integration guide
- `tests/test_safetensors_loader.cpp` - 8 working examples
