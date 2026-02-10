# PRIORITY 1: SIMD ACTIVATION - QUICK REFERENCE & STATUS

**Status:** ✅ IN PROGRESS  
**Started:** December 26, 2025  
**Target:** Achieve 2.5+ tokens/second (6× speedup)  
**Timeline:** 30-60 minutes

---

## CHANGES MADE

### ✅ File 1: `src/core/tmac/lut_gemm.h` - SIMD Initialization

**Change:** Added SIMD activation code to LookupTableGEMM constructor

```cpp
// Constructor now initializes use_avx512_gather based on compiler support
#ifdef __AVX512F__
config_.use_avx512_gather = true;  // Enable vectorized path
#else
config_.use_avx512_gather = false; // Scalar fallback only
#endif
```

**Impact:** Activates AVX-512 gather code path at runtime

---

### ✅ File 2: `CMakeLists.txt` - SIMD Compilation Flags

**Status:** Already contains required flags:

- GCC/Clang: `-march=native -mtune=native`
- MSVC: `/arch:AVX2`
- OpenMP: Enabled globally

**No changes needed** - flags already present

---

## NEXT STEPS (Execute in Order)

### Step 1: Run Execution Script (5 minutes)

```bash
cd c:\Users\sgbil\Ryot
python PRIORITY1_EXECUTION.py
```

This script will:

- ✅ Verify code changes are in place
- ✅ Clean and rebuild project with proper flags
- ✅ Run SIMD activation diagnostics
- ✅ Verify AVX-512 path is active

### Step 2: Manual Build (if needed)

```bash
cd RYZEN-LLM
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j 8
```

### Step 3: Verify SIMD Active

```bash
python diagnostic_simd_activation.py
# Look for: "compute_avx512 function exists: True"
# Look for: "use_avx512_gather_config: True"
```

### Step 4: Test Performance

```bash
python benchmark.py --test simd --iterations 10
# Expected: ≥2.5 tok/s
# Current baseline: 0.42 tok/s
# Expected improvement: 6×
```

---

## VERIFICATION CHECKLIST

- [ ] Code changes applied to lut_gemm.h
- [ ] CMakeLists.txt has `-march=native` (GCC) or `/arch:AVX2` (MSVC)
- [ ] Build completes without errors
- [ ] Diagnostic script shows `compute_avx512` is active
- [ ] Diagnostic shows `use_avx512_gather = true`
- [ ] No scalar fallback warnings
- [ ] Performance ≥2.5 tok/s achieved
- [ ] 6× speedup confirmed

---

## DEBUGGING IF NEEDED

### Issue: SIMD not activating

**Cause:** `__AVX512F__` not defined at compile time  
**Fix:** Ensure CMakeLists.txt has `-march=native`

```bash
# Check if AVX2 is available
gcc -Q --help=warning | grep avx
# Should see: -mavx2 is enabled by default
```

### Issue: Build fails

**Cause:** Stale CMake cache or compiler issues  
**Fix:**

```bash
cd RYZEN-LLM
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-march=native -O3" ..
cmake --build . -v 2>&1 | tee build.log
```

### Issue: Performance still 0.42 tok/s

**Cause:** Code path not switching to AVX-512  
**Debug:**

```bash
# Add debug output to lut_gemm.cpp::Compute()
# Look for: "Using compute_avx512" vs "Using compute_scalar"
```

---

## PERFORMANCE EXPECTATIONS

```
BEFORE: 0.42 tok/s (scalar GEMM only)
         ↓ (SIMD activation)
AFTER:  2.5 tok/s (AVX-512 vectorized)
         ↓ (6× speedup)

Metrics to track:
- Throughput (tok/s)
- GOPS (billion operations per second)
- Memory bandwidth utilization
- CPU core utilization
```

---

## SUCCESS CRITERIA

✅ **Priority 1 COMPLETE when:**

1. `diagnostic_simd_activation.py` shows `compute_avx512` active
2. No "scalar fallback" messages in logs
3. Performance ≥2.5 tok/s achieved
4. 6× speedup confirmed (0.42 → 2.5)
5. All tests passing
6. Zero regressions vs baseline

---

## FILES INVOLVED

| File                                 | Change                     |
| ------------------------------------ | -------------------------- |
| `RYZEN-LLM/src/core/tmac/lut_gemm.h` | Added SIMD initialization  |
| `RYZEN-LLM/CMakeLists.txt`           | Already has required flags |
| `PRIORITY1_EXECUTION.py`             | Execution script (NEW)     |
| `PRIORITY1_STATUS.md`                | This status file (NEW)     |

---

## PROGRESS TRACKING

**Phase:** Priority 1 of 3  
**Estimated Duration:** 30-60 minutes  
**Status:** IN PROGRESS ✅

```
Tasks:
  [✅] Code changes implemented
  [ ] Build executed
  [ ] SIMD verified active
  [ ] Performance tested (target: 2.5+ tok/s)
  [ ] Move to Priority 2 (T-MAC fix)
```

---

## NEXT MILESTONE

**After Priority 1 Complete:**

- ✅ 0.42 → 2.5 tok/s achieved
- ✅ SIMD vectorization confirmed
- ⏭️ Begin Priority 2: T-MAC Pattern Encoding

**Goal:** Achieve 5.0 tok/s total (2× more improvement)

---

## REFERENCE DOCUMENTS

- `PHASE3_BLOCKER_FIXES_DETAILED.md` - Full technical guide
- `PHASE3_EXECUTION_PLAN_SUMMARY.md` - Detailed roadmap
- `PHASE3_HANDOFF_PACKAGE.md` - Executive overview
- `diagnostic_simd_activation.py` - Diagnostic tool
- `PRIORITY1_EXECUTION.py` - This execution script

---

**Last Updated:** December 26, 2025  
**Status:** EXECUTING PRIORITY 1  
**Next Review:** After build completes
