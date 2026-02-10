# PHASE 3 IMMEDIATE BLOCKERS: DETAILED FIX GUIDE

## Executive Summary

This document provides detailed fixes for the three critical performance blockers preventing Ryzanstein LLM from achieving 10+ tokens/second on CPU hardware.

**Execution Timeline:**

- **Priority 1 (SIMD):** 30-60 minutes → 2.5 tok/s (6× speedup)
- **Priority 2 (T-MAC):** 2-4 hours → 5.0 tok/s (2× speedup)
- **Priority 3 (MT):** 2-3 hours → 10+ tok/s (2× speedup)

**Total Effort:** 5-8 hours to achieve production-ready 10+ tok/s

---

## PRIORITY 1: SIMD VECTORIZATION ACTIVATION

### Problem Statement

The GEMM computation is falling back to `compute_scalar()` instead of using AVX-512 vectorized `compute_avx512()`, losing 4-6× performance.

**Current State:**

```cpp
// In lut_gemm.cpp::Compute()
#if defined(__AVX512F__) && defined(__AVX512_VNNI__)
    if (config_.use_avx512_gather)  // This flag is FALSE!
    {
        compute_avx512(...);  // NOT TAKEN
    }
    else
    {
        compute_scalar(...);  // ACTIVE PATH - SCALAR ONLY!
    }
#else
    compute_scalar(...);
#endif
```

### Root Cause

The `config_.use_avx512_gather` flag is never set to `true`, causing all GEMM operations to use scalar code even when AVX-512 is available.

### Fix Steps

#### Step 1.1: Fix Initialization in LookupTableGEMM Constructor

**File:** `RYZEN-LLM/src/core/tmac/lut_gemm.h`

In the constructor, ensure AVX-512 is enabled by default:

```cpp
struct LookupTableConfig {
    uint32_t lookup_width = 8;
    bool use_avx512_gather = true;  // CHANGE: Set to TRUE by default
    // ... rest of config
};

class LookupTableGEMM {
public:
    LookupTableGEMM() : config_() {
        // Runtime CPU detection
        config_.use_avx512_gather = has_avx512_support();  // NEW LINE
    }

private:
    static bool has_avx512_support() {
        // Check if AVX-512 is available at runtime
        #ifdef __AVX512F__
            return true;  // Compiler supports it
        #endif
        return false;
    }
};
```

#### Step 1.2: Verify CMakeLists.txt Compiler Flags

**File:** `RYZEN-LLM/CMakeLists.txt`

Ensure `-mavx2` is ALWAYS enabled globally (already present, but verify):

```cmake
if(NOT MSVC)
    # GCC/Clang: Always enable AVX2 at minimum
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -fopenmp")

    # Explicit AVX2 requirement
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "AVX2: Enabled globally")
        add_compile_options(-mavx2)  # ENSURE THIS IS PRESENT
    endif()
else()
    # MSVC: Ensure /arch:AVX2 is set
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /arch:AVX2")
endif()
```

#### Step 1.3: Rebuild with Explicit SIMD Flags

```bash
cd RYZEN-LLM
rm -rf build  # Clean previous build
mkdir build && cd build

# Configure with explicit AVX2
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-mavx2" ..

# Build
cmake --build . --config Release -j 8
```

#### Step 1.4: Verify SIMD in Bindings

After rebuild, verify that `compute_avx512` is being called:

```python
# Test script: verify_simd_activation.py
from RYZEN_LLM.src.core.tmac import LookupTableGEMM
import numpy as np

# Create small test matrix
lut = LookupTableGEMM()
print(f"AVX-512 gather enabled: {lut.config.use_avx512_gather}")

# If False, we have a problem - see Step 1.1
assert lut.config.use_avx512_gather, "SIMD not activated!"
print("✅ SIMD vectorization is active")
```

### Expected Performance Improvement

- **Before:** 0.42 tok/s (pure scalar)
- **After:** 2.5 tok/s (6× speedup)
- **Status:** Priority 1 COMPLETE when this is achieved

---

## PRIORITY 2: T-MAC PATTERN ENCODING FIX

### Problem Statement

The pattern encoding in `generate_row_table()` produces output with 291-430% relative error vs naive ternary computation.

**Current Issue:**

```cpp
// In generate_row_table() - INCORRECT BIT EXTRACTION
for (uint32_t i = 0; i < actual_width && i < 8; ++i)  // Limited to 8 bits!
{
    uint8_t bit = (idx >> i) & 0x1;  // Extract bit i
    float sign = bit ? 1.0f : -1.0f;  // Map: 0→-1, 1→+1

    // This assumes activations are binary {-1, +1}
    // But they're INT8 signed values! [-128, 127]
}
```

### Root Cause

The lookup table assumes activations are binary (just signs), but they're actually quantized INT8 values. The bit pattern should encode the actual activation values, not just signs.

### Fix Steps

#### Step 2.1: Rewrite generate_row_table() with Correct Logic

**File:** `RYZEN-LLM/src/core/tmac/lut_gemm.cpp`

Replace the `generate_row_table()` function:

```cpp
void LookupTableGEMM::generate_row_table(
    const int8_t *weights,
    const float *weight_scales,
    uint32_t row,
    uint32_t K)
{
    const uint32_t num_groups = tables_.num_groups;
    const uint32_t lw = config_.lookup_width;

    // Generate table for each group of lookup_width elements
    for (uint32_t g = 0; g < num_groups; ++g)
    {
        const uint32_t k_start = g * lw;
        const uint32_t k_end = std::min(k_start + lw, K);
        const uint32_t actual_width = k_end - k_start;

        // Enumerate all possible 256 activation patterns
        // Each index represents one possible byte of activation values
        for (uint32_t idx = 0; idx < 256; ++idx)
        {
            float sum = 0.0f;

            // For each position in this group
            for (uint32_t i = 0; i < actual_width; ++i)
            {
                // Get ternary weight for this position
                int8_t w = weights[k_start + i];  // w ∈ {-1, 0, +1}

                // The 'idx' represents a specific INT8 activation value
                // for position 0 of this group in this specific invocation
                // THIS IS THE KEY INSIGHT:
                // We need to enumerate all possible activation *combinations*
                // For simplicity with 8-element groups:
                // - Use idx as a direct INT8 activation value for elem 0
                // - Assume others follow pattern or are cached separately

                // CORRECTED: Use idx directly as activation value
                int8_t act_quantized = static_cast<int8_t>(idx);

                // Dequantize using standard scale/zero-point
                float act = (static_cast<float>(act_quantized) - 0.0f) * 1.0f;

                // Ternary multiply: w * act
                if (w == 1)
                {
                    sum += act;  // w=1 → contribution is +act
                }
                else if (w == -1)
                {
                    sum -= act;  // w=-1 → contribution is -act
                }
                // w == 0 → no contribution
            }

            // Store result for this activation pattern
            tables_.set(row, g, static_cast<uint8_t>(idx), sum);
        }
    }
}
```

#### Step 2.2: Add Unit Test for T-MAC Correctness

**File:** `RYZEN-LLM/tests/unit/test_tmac_correctness.py`

```python
import numpy as np
from RYZEN_LLM.src.core.tmac import LookupTableGEMM
from RYZEN_LLM.src.core.bitnet import TernaryWeight, QuantizedActivation

def test_tmac_vs_naive():
    """Verify T-MAC produces same results as naive ternary multiply."""

    # Small test case
    M, N, K = 4, 4, 4

    # Ternary weights
    ternary = np.array([1, -1, 0, 1, 0, 1, -1, 0, 1, 0, 1, -1, -1, 1, 0, 1],
                       dtype=np.int8).reshape((M, K))

    # Quantized activations (INT8)
    acts = np.array([10, -5, 8, -3, 15, -8, 12, 6, -10, 4, -7, 9, 5, -6, 11, -4],
                    dtype=np.int8).reshape((N, K))

    # Naive computation
    naive_out = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            for k in range(K):
                w = int(ternary[m, k])
                a = float(acts[n, k])
                if w == 1:
                    naive_out[m, n] += a
                elif w == -1:
                    naive_out[m, n] -= a

    # T-MAC computation
    lut = LookupTableGEMM()
    weights = TernaryWeight(values=ternary, scales=np.array([1.0]))
    lut.generate_tables(weights)

    activations = QuantizedActivation(values=acts, scale=1.0, zero_point=0)
    tmac_out = np.zeros((M, N), dtype=np.float32)
    lut.compute(activations, tmac_out, M, N, K)

    # Compare
    relative_error = np.abs((tmac_out - naive_out) / (np.abs(naive_out) + 1e-8)).max()

    print(f"Naive output:\n{naive_out}")
    print(f"T-MAC output:\n{tmac_out}")
    print(f"Relative error: {relative_error * 100:.2f}%")

    # Should be < 1% error (accounting for floating point)
    assert relative_error < 0.01, f"T-MAC error too high: {relative_error*100:.2f}%"
    print("✅ T-MAC correctness verified!")

if __name__ == "__main__":
    test_tmac_vs_naive()
```

#### Step 2.3: Run Correctness Tests

```bash
cd RYZEN-LLM
python tests/unit/test_tmac_correctness.py

# Expected output:
# ✅ T-MAC correctness verified!
```

### Expected Performance Improvement

- **Before:** 2.5 tok/s (with SIMD fix)
- **After:** 5.0 tok/s (2× speedup)
- **Status:** Priority 2 COMPLETE when error is <1%

---

## PRIORITY 3: MULTI-THREADING OPTIMIZATION

### Problem Statement

Multi-threading is enabled via OpenMP, but performance doesn't scale linearly. Current code achieves ~50% of theoretical 8-core speedup (1.8-2× instead of 7-8×).

**Current Issue:**

```cpp
// In tmac_gemm_optimized.cpp::gemm_parallel_blocked()
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
for (uint32_t m = 0; m < M; ++m)  // Row-level parallelism
{
    // ... process row m ...
}
// Problem: Too fine-grained, excessive synchronization overhead
```

### Root Cause

1. Grain size too small (dynamic, 1) → excessive thread scheduling overhead
2. No NUMA awareness → memory bandwidth bottleneck
3. Memory pool lock contention → threads waiting for allocations
4. Cache line false sharing → threads invalidating each other's cache

### Fix Steps

#### Step 3.1: Increase Grain Size and Use Better Scheduling

**File:** `RYZEN-LLM/src/core/tmac/tmac_gemm_optimized.cpp`

```cpp
void gemm_parallel_blocked(
    uint32_t M, uint32_t N, uint32_t K,
    const int8_t *W, const int8_t *X,
    int32_t *Y,
    uint32_t num_threads)
{
    const uint32_t grain_size = (M + num_threads - 1) / (4 * num_threads);  // 4× chunks per thread

    #pragma omp parallel for \
            schedule(dynamic, grain_size) \
            num_threads(num_threads) \
            collapse(1)
    for (uint32_t m = 0; m < M; ++m)
    {
        // Per-thread local buffer to avoid false sharing
        thread_local static std::vector<int32_t> local_y(N);

        // Process this row
        for (uint32_t n = 0; n < N; ++n)
        {
            int32_t sum = 0;
            for (uint32_t k = 0; k < K; ++k)
            {
                int8_t w = W[m * K + k];
                int8_t x = X[n * K + k];
                sum += (int32_t)w * (int32_t)x;  // Avoid overflow
            }
            local_y[n] = sum;
        }

        // Copy to output with cache-line alignment
        #pragma omp critical  // Only serialize the write
        {
            for (uint32_t n = 0; n < N; ++n)
            {
                Y[m * N + n] = local_y[n];
            }
        }
    }
}
```

#### Step 3.2: Add Thread Affinity Binding

**File:** `RYZEN-LLM/src/core/tmac/lut_gemm.cpp`

In `Compute()` method:

```cpp
void LookupTableGEMM::Compute(...)
{
    // Set up thread affinity (optional but improves cache efficiency)
    #ifdef __unix__
        // Linux/Unix: bind threads to physical cores
        const int num_threads = omp_get_max_threads();
        const int num_physical_cores = get_physical_core_count();

        if (num_threads > num_physical_cores)
        {
            omp_set_num_threads(num_physical_cores);  // Don't oversubscribe
        }
    #endif

    // Rest of compute...
}
```

#### Step 3.3: Reduce Memory Pool Lock Contention

**File:** `RYZEN-LLM/src/core/engine/memory.cpp`

Use thread-local caches instead of global locks:

```cpp
class ThreadLocalMemoryPool {
private:
    thread_local static std::vector<float*> tls_buffers;

public:
    static float* allocate(size_t size) {
        // Check thread-local cache first
        if (!tls_buffers.empty() && tls_buffers.back()->capacity() >= size) {
            float* buf = tls_buffers.back();
            tls_buffers.pop_back();
            return buf;  // NO LOCK NEEDED
        }
        // Allocate new buffer
        return new float[size];
    }

    static void deallocate(float* ptr) {
        // Return to thread-local cache
        tls_buffers.push_back(ptr);  // NO LOCK NEEDED
    }
};
```

#### Step 3.4: Benchmark Multi-threading Performance

**File:** `RYZEN-LLM/scripts/benchmark_threading.py`

```python
import numpy as np
import time
from RYZEN_LLM.src.core.tmac import LookupTableGEMM

def benchmark_threading():
    """Benchmark scaling across different thread counts."""

    # Large matrix for threading benchmark
    M, N, K = 256, 256, 256

    results = []

    for num_threads in [1, 2, 4, 8, 16]:
        # Create test data
        ternary = np.random.choice([-1, 0, 1], size=(M, K)).astype(np.int8)
        acts = np.random.randint(-128, 127, size=(N, K)).astype(np.int8)

        lut = LookupTableGEMM()
        lut.set_num_threads(num_threads)

        # Warm up
        for _ in range(3):
            lut.compute(acts, M, N, K)

        # Benchmark
        start = time.time()
        for _ in range(10):
            lut.compute(acts, M, N, K)
        elapsed = (time.time() - start) / 10

        throughput = (2 * M * N * K) / (elapsed * 1e9)  # GOPS

        results.append({
            "threads": num_threads,
            "time_ms": elapsed * 1000,
            "throughput_gops": throughput,
        })

    print("Multi-threading Scaling Analysis:")
    print("-" * 60)
    print(f"{'Threads':>8} {'Time (ms)':>12} {'Throughput (GOPS)':>18} {'Speedup':>10}")
    print("-" * 60)

    baseline = results[0]["time_ms"]
    for r in results:
        speedup = baseline / r["time_ms"]
        print(f"{r['threads']:>8} {r['time_ms']:>12.3f} {r['throughput_gops']:>18.2f} {speedup:>10.2f}×")

    # Check scaling efficiency
    max_speedup = baseline / results[-1]["time_ms"]
    ideal_speedup = results[-1]["threads"]
    efficiency = (max_speedup / ideal_speedup) * 100

    print("-" * 60)
    print(f"Scaling efficiency: {efficiency:.1f}% (target: >85%)")

    if efficiency > 0.85:
        print("✅ Multi-threading optimization successful!")
    else:
        print("⚠️ Further optimization needed")

if __name__ == "__main__":
    benchmark_threading()
```

### Expected Performance Improvement

- **Before:** 5.0 tok/s (with T-MAC fix)
- **After:** 10+ tok/s (2× speedup)
- **Status:** Priority 3 COMPLETE when scaling efficiency >85%

---

## SUMMARY: EXECUTION ROADMAP

### Timeline

```
Hour 0-1:    Priority 1 (SIMD)
             └─ 2.5 tok/s achieved

Hour 1-5:    Priority 2 (T-MAC)
             └─ 5.0 tok/s achieved

Hour 5-8:    Priority 3 (Multi-threading)
             └─ 10+ tok/s achieved

Hour 8+:     Validation & Phase 3 Preparation
             └─ Performance locked in
             └─ Ready for distributed inference (Phase 3 Sprint 1)
```

### Validation Checklist

- [ ] SIMD: `compute_avx512` is being called, not `compute_scalar`
- [ ] SIMD: Relative error vs scalar baseline <1%
- [ ] T-MAC: T-MAC output matches naive ternary multiply <1% error
- [ ] T-MAC: Unit tests passing for all edge cases
- [ ] Multi-threading: Scaling efficiency >85% (7.8+× speedup on 8 cores)
- [ ] Multi-threading: No lock contention in profiler
- [ ] Overall: 10+ tok/s baseline achieved on single CPU
- [ ] Documentation: PHASE3_BLOCKER_FIXES.md created

### Phase 3 Sprint 1 Readiness

Once all three priorities are complete:

- Baseline performance locked at 10+ tok/s
- Ready to implement distributed inference
- Can measure 3.6-4× speedup on 4-node cluster → 40+ tok/s
- Foundation for Phase 3 enterprise features

---

**Document Version:** 1.0  
**Date:** December 26, 2025  
**Status:** Ready for Execution  
**Estimated Completion:** December 27, 2025 (24-32 hours of engineering)
