# T-MAC GEMM Optimization - Integration Guide

## Quick Start

### 1. Build with Optimizations

```bash
cd Ryzanstein LLM

# Configure with AVX-512 enabled
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=native -mavx512f -O3 -funroll-loops" \
      -DENABLE_AVX512=ON

# Build
cmake --build build --config Release -j16

# Run benchmarks
./build/Release/benchmark_gemm_performance
```

### 2. Using the Optimized API

```cpp
#include "core/tmac/tmac_gemm_optimized.h"

// Initialize LUT engine
auto lut = std::make_shared<CompressedLUT>();
TableBuilder builder;
builder.build(lut.get());
auto lut_engine = std::make_shared<LUTLookup>(lut);

// Allocate aligned memory (64-byte alignment for AVX-512)
int8_t* W = (int8_t*)_aligned_malloc(M * K, 64);
int8_t* X = (int8_t*)_aligned_malloc(K * N, 64);
int32_t* Y = (int32_t*)_aligned_malloc(M * N * sizeof(int32_t), 64);

// Run optimized GEMM
gemm_optimized(lut_engine.get(), W, X, Y, M, K, N);

// Cleanup
_aligned_free(W);
_aligned_free(X);
_aligned_free(Y);
```

### 3. Integration with Existing Code

**Option A: Direct replacement**

```cpp
// Old code:
TMACGemm gemm_engine(lut_engine);
gemm_engine.gemm(W, X, Y, M, K, N);

// New code:
gemm_optimized(lut_engine.get(), W, X, Y, M, K, N);
```

**Option B: Modify TMACGemm class**

```cpp
// In tmac_gemm.cpp, replace gemm_inner_avx512 with:
#include "tmac_gemm_optimized.h"

void TMACGemm::gemm_inner_avx512(...) {
    // Call optimized implementation
    gemm_optimized(lut_engine_.get(), W_row, X, Y_row, 1, K, N);
}
```

## Performance Targets

### Expected Results

| Matrix Size | Baseline   | Optimized      | Speedup |
| ----------- | ---------- | -------------- | ------- |
| 128×512×512 | 80 GFLOPS  | 400-600 GFLOPS | 5-8×    |
| 512×2K×2K   | 100 GFLOPS | 500-700 GFLOPS | 5-7×    |
| 1024×4K×4K  | 120 GFLOPS | 600-800 GFLOPS | 5-7×    |

### Acceptance Criteria

- ✅ **Correctness:** 100% match with baseline
- ✅ **Minimum target:** 300 GFLOPS (3× speedup)
- 🎯 **Primary target:** 500-800 GFLOPS (5-8× speedup)
- 🔬 **Stretch goal:** 1000+ GFLOPS (10× speedup)

## Compilation Requirements

### Compiler Flags

**Required:**

```bash
-march=native      # Enable all CPU features
-mavx512f          # Enable AVX-512 foundation
-O3                # Maximum optimization
```

**Recommended:**

```bash
-funroll-loops     # Loop unrolling
-ffast-math        # Aggressive math optimizations (use with caution)
-flto              # Link-time optimization
```

**Debug Build:**

```bash
-g -O2 -march=native -mavx512f
```

### CMake Configuration

Add to `CMakeLists.txt`:

```cmake
# Check for AVX-512 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx512f" COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx512f")
    message(STATUS "AVX-512 support enabled")
else()
    message(WARNING "AVX-512 not supported by compiler")
endif()

# Add optimized sources
add_library(tmac_optimized
    src/core/tmac/tmac_gemm_optimized.cpp
)

target_compile_options(tmac_optimized PRIVATE
    -O3
    -funroll-loops
    $<$<CONFIG:Release>:-flto>
)

# Add benchmark
add_executable(benchmark_gemm_performance
    tests/benchmark_gemm_performance.cpp
)

target_link_libraries(benchmark_gemm_performance
    tmac_core
    tmac_optimized
)
```

## Hardware Requirements

### Minimum Requirements

- **CPU:** x86-64 with AVX-512F support
  - Intel: Ice Lake (10th gen) or newer
  - AMD: Zen 4 (Ryzanstein 7000 series) or newer
- **RAM:** 8 GB minimum (16 GB recommended for large matrices)
- **Compiler:** GCC 9+, Clang 10+, or MSVC 2019+

### Optimal Hardware (Tested)

- **CPU:** AMD Ryzanstein 9 7950X (Zen 4)
  - 16 cores, 32 threads
  - Base: 4.5 GHz, Boost: 5.7 GHz
  - L1: 32 KB/core, L2: 512 KB/core, L3: 64 MB shared
- **RAM:** DDR5-6400 (50 GB/s bandwidth)
- **Compiler:** GCC 12.2 with `-march=znver4`

### Detecting CPU Features

```cpp
#include <cpuid.h>

bool has_avx512() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
        return (ebx & bit_AVX512F) != 0;
    }
    return false;
}
```

## Troubleshooting

### Issue: Illegal Instruction Error

**Symptom:** Program crashes with "Illegal instruction"

**Cause:** Running on CPU without AVX-512 support

**Solution:**

```cpp
#ifdef __AVX512F__
    gemm_optimized(...);
#else
    gemm_baseline(...); // Fallback
#endif
```

### Issue: Low Performance (< 300 GFLOPS)

**Possible causes:**

1. **Not using aligned memory**
   - Solution: Use `_aligned_malloc(size, 64)`
2. **Power throttling**
   - Solution: Check CPU frequency, disable power saving
3. **Memory bandwidth saturation**
   - Solution: Reduce matrix size or use blocking
4. **Cache thrashing**
   - Solution: Tune block sizes for your CPU

**Debug performance:**

```bash
# Check CPU frequency
watch -n1 "grep MHz /proc/cpuinfo"

# Monitor cache misses (Linux)
perf stat -e cache-references,cache-misses ./benchmark

# Profile with VTune (Intel)
vtune -collect hotspots ./benchmark
```

### Issue: Correctness Mismatch

**Symptom:** Results differ from baseline

**Debug steps:**

1. Print intermediate values
2. Test with small matrices (16×16×16)
3. Check for uninitialized memory
4. Verify alignment

```cpp
// Add debug output
#define DEBUG_GEMM
#ifdef DEBUG_GEMM
    std::cout << "Pattern: ";
    for (int i = 0; i < 16; ++i) std::cout << (int)pattern[i] << " ";
    std::cout << "\nResults: ";
    for (int i = 0; i < 16; ++i) std::cout << results[i] << " ";
    std::cout << "\n";
#endif
```

## Advanced Tuning

### 1. Block Size Optimization

Test different block sizes:

```cpp
const uint32_t block_configs[][3] = {
    {32, 64, 256},   // Default (Zen 4)
    {64, 64, 128},   // Larger M-blocks
    {32, 128, 256},  // Larger N-blocks
    {16, 32, 512},   // Smaller blocks (Ice Lake)
};

for (auto [m, n, k] : block_configs) {
    // Test and benchmark
}
```

### 2. Prefetch Distance Tuning

Adjust prefetch distance based on latency:

```cpp
// For high-latency memory:
if (g + 4 < num_groups) {  // Prefetch 4 iterations ahead
    _mm_prefetch(..., _MM_HINT_T0);
}

// For low-latency (L1/L2):
if (g + 2 < num_groups) {  // Prefetch 2 iterations ahead
    _mm_prefetch(..., _MM_HINT_T0);
}
```

### 3. Gather Optimization

Replace scalar gather with SIMD gather (if beneficial):

```cpp
// Scalar gather (current)
for (int j = 0; j < 16; ++j) {
    activations[j] = X[(n + j) * K + g * 16 + i];
}

// SIMD gather (experimental)
__m512i indices = _mm512_setr_epi32(
    (n+0)*K + g*16 + i, (n+1)*K + g*16 + i, ...
);
__m512i gathered = _mm512_i32gather_epi32(indices, X, 1);
```

## Future Enhancements

### 1. True Vectorized Lookup (Highest Impact)

Modify `LUTLookup` to return SIMD results:

```cpp
__m512i LUTLookup::lookup_batch_simd(
    const TernaryPattern& pattern,
    __m512i activations);
```

**Expected:** Additional 2-3× speedup (1500-2000 GFLOPS)

### 2. Multi-Threading

Parallelize across M-dimension:

```cpp
#pragma omp parallel for schedule(dynamic)
for (uint32_t m_block = 0; m_block < M; m_block += BLOCK_M) {
    // Process block
}
```

**Expected:** 8-16× on 16-core CPU (4000-8000 GFLOPS)

### 3. VNNI Instructions

Use INT8×INT8 accumulation for dense patterns:

```cpp
#ifdef __AVX512VNNI__
__m512i acc = _mm512_dpbusd_epi32(acc, weights, activations);
#endif
```

**Expected:** 10-20% improvement on certain workloads

### 4. GPU Acceleration

Port to CUDA/ROCm:

- **Expected:** 50-100× speedup (30,000-50,000 GFLOPS on RTX 4090)
- **Effort:** New implementation required

## Validation Checklist

Before deploying to production:

- [ ] Compile with AVX-512 flags
- [ ] Run benchmark suite and verify GFLOPS targets
- [ ] Verify 100% correctness on all test cases
- [ ] Test on target hardware (not just development machine)
- [ ] Profile cache hit rates and memory bandwidth
- [ ] Stress test with large matrices (> L3 cache size)
- [ ] Verify thread safety if using multi-threading
- [ ] Document any CPU-specific optimizations

## Support and Contact

For issues or questions:

1. Check this guide and optimization report
2. Review benchmark results
3. File issue with:
   - CPU model and specifications
   - Compiler version and flags
   - Actual vs expected performance
   - Output of `cat /proc/cpuinfo | grep flags`

---

**Files in this optimization package:**

- `src/core/tmac/tmac_gemm_optimized.{h,cpp}` - Implementation
- `tests/benchmark_gemm_performance.cpp` - Benchmarks
- `docs/algorithms/AVX512_OPTIMIZATION_REPORT.md` - Technical report
- `docs/algorithms/INTEGRATION_GUIDE.md` - This guide

**Status:** ✅ Ready for integration and testing
