# PRIORITY 1 EXECUTION GUIDE: SIMD ACTIVATION

**Status:** ✅ READY TO EXECUTE  
**Estimated Duration:** 30-60 minutes  
**Target Performance:** 0.42 → 2.5 tok/s (6× speedup)

---

## ✅ PHASE 1: VERIFICATION (Code changes are complete)

### What Was Changed

**File 1: `src/core/tmac/lut_gemm.h`**

- ✅ Modified LookupTableGEMM constructor
- ✅ Added SIMD initialization code with #ifdef **AVX512F**
- ✅ Set config\_.use_avx512_gather = true when available

**File 2: `CMakeLists.txt`**

- ✅ Already contains `-march=native -mtune=native` for GCC/Clang
- ✅ Already contains `/arch:AVX2` for MSVC
- ✅ OpenMP enabled globally
- ⏭️ No changes needed

---

## 🔨 PHASE 2: REBUILD PROJECT

### Option A: Automatic Rebuild (Recommended)

```bash
cd c:\Users\sgbil\Ryot
python PRIORITY1_EXECUTION.py
```

This script will:

1. ✅ Verify code changes are in place
2. ✅ Clean old build directory
3. ✅ Run CMake configuration
4. ✅ Rebuild entire project
5. ✅ Run diagnostics
6. ✅ Generate report

**Expected output:**

```
[HH:MM:SS] [INFO] STEP 1: Verify Code Changes
[HH:MM:SS] [OK]   ✅ Config initialization: True
[HH:MM:SS] [OK]   ✅ AVX512F check: True
[HH:MM:SS] [OK]   ✅ SIMD activation comment: True
[HH:MM:SS] [INFO] STEP 2: Clean Build
[HH:MM:SS] [OK]   ✅ Build succeeded
[HH:MM:SS] [FINAL] Overall Status: COMPLETED
```

### Option B: Manual Rebuild (If automation fails)

```bash
cd c:\Users\sgbil\Ryot\RYZEN-LLM

# Step 1: Clean old build
rm -rf build
mkdir build
cd build

# Step 2: Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Expected CMake output:
# -- AVX-512 compiler support detected
# -- AVX-512 VNNI compiler support detected
# -- FMA3 support detected
# -- Configure done

# Step 3: Build project
cmake --build . --config Release -j 8

# Expected build output:
# [100%] Built target ryzen_llm_core
# [100%] Built target ryzen_llm_optimization
# [100%] Built target ryzen_llm_bindings
```

---

## 🔍 PHASE 3: VERIFICATION (SIMD is active)

### Run Diagnostic Script

```bash
cd c:\Users\sgbil\Ryot
python diagnostic_simd_activation.py
```

**Expected output (SUCCESS case):**

```
[1] SYSTEM & BUILD ENVIRONMENT ANALYSIS
  system: Windows
  processor: x86_64
  cpu_count: 8
  avx2_supported: True
  avx512f_supported: True
  fma_supported: True

[2] SIMD ACTIVATION DIAGNOSTIC
  ✅ avx2_enabled_in_cmake: True
  ✅ avx512_enabled_in_cmake: True
  ✅ compute_avx512_function_exists: True
  ✅ compute_scalar_as_fallback: True

[5] PERFORMANCE TARGETS & ROADMAP
  → Current Baseline:    0.42 tok/s
  → Fix SIMD (P1):       2.5 tok/s (6× speedup)
  → Fix T-MAC (P2):      5.0 tok/s (2× speedup)
  → Fix Threading (P3):  10+ tok/s (2× speedup)

ANALYSIS:
  ✅ SIMD vectorization appears to be properly enabled
```

### Check Diagnostic Report

```bash
# View the generated JSON report
cat PHASE3_DIAGNOSTIC_REPORT.json | python -m json.tool

# Look for these key indicators:
# - "avx2_supported": true
# - "avx512f_supported": true
# - "compute_avx512_function_exists": true
```

---

## ⚡ PHASE 4: PERFORMANCE TESTING

### Quick Performance Check

```bash
cd c:\Users\sgbil\Ryot\RYZEN-LLM

# Create quick test
python -c "
import time
import sys
sys.path.insert(0, '.')

from src.core.tmac import LookupTableGEMM

print('Testing SIMD performance...')
lut = LookupTableGEMM()

# Check config
print(f'SIMD enabled: {lut.config.use_avx512_gather}')

if lut.config.use_avx512_gather:
    print('✅ SIMD path is ACTIVE')
else:
    print('❌ SIMD path is INACTIVE - using scalar fallback')
"
```

**Expected output:**

```
Testing SIMD performance...
SIMD enabled: True
✅ SIMD path is ACTIVE
```

### Full Benchmark

```bash
# Run comprehensive benchmark
python benchmark.py --test simd --iterations 10

# Expected output shows:
# - Throughput: ≥2.5 GOPS
# - Time per operation: <1ms
# - Status: PASS (for 2.5+ tok/s)
```

---

## ✅ SUCCESS CRITERIA

Priority 1 is COMPLETE when ALL of these are met:

- [x] Code changes applied to lut_gemm.h
- [x] CMakeLists.txt has correct SIMD flags
- [ ] **Build completes without errors**
- [ ] **Diagnostic shows `use_avx512_gather = True`**
- [ ] **No scalar fallback warnings in output**
- [ ] **Performance ≥2.5 tok/s achieved**
- [ ] **6× speedup verified (0.42 → 2.5)**

---

## 🐛 TROUBLESHOOTING

### Issue: Build fails with CMake error

**Solution:**

```bash
# Clean everything
rm -rf RYZEN-LLM/build
rm -rf RYZEN-LLM/CMakeCache.txt
rm -rf RYZEN-LLM/CMakeFiles

# Try again with verbose output
cd RYZEN-LLM/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON
cmake --build . -v
```

### Issue: SIMD still not activating

**Debug:**

1. Check if `-march=native` is in CMAKE_CXX_FLAGS_RELEASE:

   ```bash
   grep "CMAKE_CXX_FLAGS_RELEASE" RYZEN-LLM/CMakeLists.txt
   # Should show: -march=native -mtune=native
   ```

2. Check if compiler supports AVX-512:

   ```bash
   gcc -Q --help=warning | grep avx
   # Should show: -mavx2 -mavx512f enabled by default
   ```

3. Add debug output to lut_gemm.cpp:
   ```cpp
   void LookupTableGEMM::Compute(...) {
       #ifdef __AVX512F__
       std::cerr << "AVX-512 is available at compile time\n";
       if (config_.use_avx512_gather) {
           std::cerr << "Using AVX-512 vectorized path\n";
           compute_avx512(...);
       }
       #else
       std::cerr << "AVX-512 NOT available at compile time\n";
       #endif
   }
   ```

### Issue: Performance still 0.42 tok/s

**Debug:**

1. Check what code path is actually executing:

   ```bash
   # Add logging to compute_scalar() and compute_avx512()
   # Compare log output to see which is being called
   ```

2. Profile with VTune:

   ```bash
   vtune -collect general-exploration -r vtune-result -- python benchmark.py
   # Look for: compute_scalar vs compute_avx512 in call stack
   ```

3. Check memory bandwidth utilization:
   ```bash
   # If <50% bandwidth utilized, threading or layout issue
   # If >80% bandwidth utilized, approaching limits
   ```

---

## 📊 EXPECTED PERFORMANCE CURVE

```
BEFORE Priority 1:
  0.42 tok/s  ══════════════════════════════════════════════════
              │
              ├─ Why: Pure scalar GEMM, no vectorization
              │
              └─ Bottleneck: CPU compute (no SIMD)

AFTER Priority 1:
              │
              └─ Enable AVX-512 gather paths

  2.5 tok/s   ════════════════════════════════════════════════════
              │
              ├─ Why: 6× speedup from vectorization
              │
              └─ Bottleneck: Memory bandwidth or T-MAC correctness
```

---

## 🎯 NEXT STEPS AFTER PRIORITY 1

### If SUCCESS (2.5+ tok/s achieved):

1. ✅ Commit changes: `git commit -m "Priority 1: SIMD activation"`
2. ✅ Tag: `git tag -a v2.1-p1-simd`
3. ⏭️ **Begin Priority 2: T-MAC Pattern Encoding**
4. ⏭️ Target: Achieve 5.0 tok/s (2× more improvement)

### If PARTIAL SUCCESS (1.5-2.5 tok/s):

1. ✅ Investigation needed - some SIMD path is active
2. ⏭️ Continue to Priority 2 anyway
3. ⏭️ Revisit if Priority 2 doesn't improve performance

### If FAILURE (Still 0.42 tok/s):

1. ❌ Debug thoroughly before continuing
2. ❌ Check CMake configuration explicitly
3. ❌ Profile with system tools to see what's happening
4. ❌ Don't move to Priority 2 until SIMD is confirmed active

---

## 📋 EXECUTION CHECKLIST

**Pre-Build:**

- [ ] Read this entire guide
- [ ] Verify lut_gemm.h has SIMD initialization code
- [ ] Verify CMakeLists.txt has -march=native
- [ ] Create git branch: `git checkout -b phase3/p1-simd`

**Build Phase:**

- [ ] Run PRIORITY1_EXECUTION.py or manual rebuild
- [ ] Build completes without errors
- [ ] No linker warnings about missing symbols

**Verification Phase:**

- [ ] Run diagnostic_simd_activation.py
- [ ] Verify `use_avx512_gather = True` in output
- [ ] Check GOPS >0.5 (baseline for SIMD)

**Performance Testing:**

- [ ] Run benchmark.py with SIMD test
- [ ] Confirm throughput ≥2.5 tok/s
- [ ] Measure memory bandwidth utilization
- [ ] Verify 6× speedup from baseline

**Documentation:**

- [ ] Create PRIORITY1_COMPLETION_REPORT.md
- [ ] Document performance improvements
- [ ] Note any issues encountered
- [ ] Commit all changes

---

## 🔗 REFERENCE LINKS

- **Full Guide:** `PHASE3_BLOCKER_FIXES_DETAILED.md` (Section: Priority 1)
- **Status Tracking:** `PRIORITY1_STATUS.md`
- **Execution Script:** `PRIORITY1_EXECUTION.py`
- **Diagnostics:** `diagnostic_simd_activation.py`

---

## ⏱️ TIMELINE

```
Minute  0: Start rebuild
Minute 10: Build should complete
Minute 15: Run diagnostics
Minute 20: Confirm SIMD active
Minute 30: Performance testing
Minute 45: Documentation & next steps
Minute 60: Begin Priority 2 (if successful)
```

---

## 🎬 START EXECUTION NOW

### Execute immediately:

```bash
cd c:\Users\sgbil\Ryot

# Run the automated execution script
python PRIORITY1_EXECUTION.py

# Monitor output for:
# ✅ All verification steps pass
# ✅ Build completes successfully
# ✅ SIMD diagnostic shows active
# ✅ Performance ≥2.5 tok/s
```

---

**PRIORITY 1 IS READY FOR EXECUTION**  
**Estimated completion: 45 minutes**  
**Next: Priority 2 - T-MAC Pattern Encoding**
