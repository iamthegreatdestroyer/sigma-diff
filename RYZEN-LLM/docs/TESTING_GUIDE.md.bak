# RYZEN-LLM Testing Quick Reference

**Task 5: Integration Testing Guide**

## Quick Start

### Run All Tests (Recommended First Time)

```powershell
# Windows
.\scripts\run_integration_tests.ps1 -Build -All

# Linux/Mac
python scripts/run_integration_tests.py --build --all
```

### Run Integration Tests Only (Fast Iteration)

```powershell
# Windows
.\scripts\run_integration_tests.ps1 -Integration

# Linux/Mac
python scripts/run_integration_tests.py --integration
```

---

## Test Organization

```
tests/
├── unit/                          # C++ unit tests (pytest)
│   └── test_bitnet_matmul.py     # Quantization & matmul tests
│
├── integration/                   # End-to-end tests
│   └── test_bitnet_generation.py # Full generation pipeline (Task 5)
│
└── e2e/                           # Cross-model tests (Phase 1 complete)
    └── test_phase1_complete.py   # BitNet + Mamba + RWKV (Task 17)
```

---

## Test Execution Options

### PowerShell (Windows)

```powershell
# Build and run all tests
.\scripts\run_integration_tests.ps1 -Build -All

# Run with Debug build
.\scripts\run_integration_tests.ps1 -Build -Config Debug -Integration

# Run only benchmarks
.\scripts\run_integration_tests.ps1 -Benchmark

# Run specific marker
.\scripts\run_integration_tests.ps1 -Integration -Markers "performance"

# Skip build (use existing)
.\scripts\run_integration_tests.ps1 -Integration
```

### Python (Cross-Platform)

```bash
# Build and run all tests
python scripts/run_integration_tests.py --build --all

# Run with Release build (default)
python scripts/run_integration_tests.py --build --config Release --integration

# Run only integration tests
python scripts/run_integration_tests.py --integration

# Run benchmarks
python scripts/run_integration_tests.py --benchmark

# Filter by markers
python scripts/run_integration_tests.py --integration --markers "not slow"
```

### Direct pytest (Advanced)

```bash
# Run all integration tests
pytest tests/integration -v

# Run with coverage
pytest tests/integration -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/integration/test_bitnet_generation.py -v

# Run specific test class
pytest tests/integration/test_bitnet_generation.py::TestGeneration -v

# Run specific test method
pytest tests/integration/test_bitnet_generation.py::TestGeneration::test_greedy_generation_deterministic -v

# Run with markers
pytest tests/integration -m "not benchmark" -v

# Run benchmarks only
pytest tests/integration -m benchmark -v

# Show print statements
pytest tests/integration -v -s

# Stop on first failure
pytest tests/integration -v -x
```

---

## Test Markers

```python
@pytest.mark.benchmark      # Performance benchmarks
@pytest.mark.integration    # Integration tests
@pytest.mark.skipif()       # Conditional skip
```

**Usage:**

```bash
# Run all except benchmarks
pytest -m "not benchmark"

# Run only benchmarks
pytest -m benchmark

# Run integration tests
pytest -m integration
```

---

## Common Workflows

### 1. First-Time Setup

```powershell
# Install dependencies
python -m pip install pytest pytest-cov pytest-benchmark numpy

# Build project
.\scripts\run_integration_tests.ps1 -Build

# Run all tests
.\scripts\run_integration_tests.ps1 -All
```

### 2. Development Iteration (After Code Changes)

```powershell
# Rebuild and test
.\scripts\run_integration_tests.ps1 -Build -Integration

# Or just test (if no C++ changes)
.\scripts\run_integration_tests.ps1 -Integration
```

### 3. Performance Benchmarking

```powershell
# Run benchmarks only
.\scripts\run_integration_tests.ps1 -Benchmark

# Or with pytest directly (more details)
pytest tests/integration -m benchmark -v --benchmark-autosave
```

### 4. Continuous Integration (CI/CD)

```bash
# Full test suite with coverage
python scripts/run_integration_tests.py --build --all

# Check exit code
if [ $? -eq 0 ]; then
    echo "Tests passed"
else
    echo "Tests failed"
    exit 1
fi
```

### 5. Debugging Test Failures

```bash
# Run with full traceback
pytest tests/integration -v --tb=long

# Run with pdb on failure
pytest tests/integration -v --pdb

# Run single failing test
pytest tests/integration/test_bitnet_generation.py::TestClass::test_method -v -s
```

---

## Test Configuration

### Small Test Model (Default)

```python
# tests/integration/test_bitnet_generation.py
class TestConfig:
    VOCAB_SIZE = 1000
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    NUM_HEADS = 8
    MAX_SEQ_LENGTH = 128
```

**Rationale:** Fast iteration (seconds vs minutes)

### Custom Configuration (Advanced)

```python
# Override in test
@pytest.fixture
def custom_config():
    config = TestConfig()
    config.NUM_LAYERS = 4  # More layers
    config.HIDDEN_SIZE = 1024  # Larger model
    return config
```

---

## Expected Output

### Successful Run

```
==================== RYZEN-LLM Integration Tests ====================

Checking Python...
  Python 3.13.9

Checking pytest...
  pytest installed

==================== Running Integration Tests ====================

tests/integration/test_bitnet_generation.py::TestBitNetEngine::test_engine_initialization PASSED
tests/integration/test_bitnet_generation.py::TestBitNetEngine::test_synthetic_weight_shapes PASSED
tests/integration/test_bitnet_generation.py::TestBitNetEngine::test_weight_value_ranges PASSED
tests/integration/test_bitnet_generation.py::TestGeneration::test_greedy_generation_deterministic PASSED
tests/integration/test_bitnet_generation.py::TestGeneration::test_temperature_sampling_diversity PASSED
...

==================== 23 passed in 5.42s ====================

SUCCESS - All tests passed
```

### Test Failure Example

```
FAILED tests/integration/test_bitnet_generation.py::TestPerplexity::test_perplexity_perfect_prediction

Expected perplexity ~1.0, got 3.5

AssertionError: Perfect prediction should have perplexity ~1.0, got 3.5
```

**Action:** Debug the perplexity computation or check model predictions

---

## Coverage Reports

### Terminal Coverage (Default)

```
---------- coverage: platform win32, python 3.13.9 -----------
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/core/bitnet/engine.py                 150     10    93%   45-47, 120-125
tests/integration/test_bitnet_generation  400      5    99%   350-352
---------------------------------------------------------------------
TOTAL                                     550     15    97%
```

### HTML Coverage Report

```bash
pytest tests/integration --cov=src --cov-report=html

# Open htmlcov/index.html in browser
```

### Coverage Threshold

```bash
# Fail if coverage < 90%
pytest tests/integration --cov=src --cov-fail-under=90
```

---

## Performance Targets (Task 5)

### Baseline (Before AVX-512)

| Metric     | Target    | Status       |
| ---------- | --------- | ------------ |
| Throughput | ≥10 tok/s | ⏳ Simulated |
| TTFT       | <400ms    | ⏳ Simulated |
| Perplexity | <15.0     | ✅ Validated |
| Memory     | <500 MB   | ✅ Estimated |

### After Task 6 (AVX-512)

| Metric     | Target      | Status     |
| ---------- | ----------- | ---------- |
| Throughput | 20-40 tok/s | ⏳ Pending |
| TTFT       | <200ms      | ⏳ Pending |

---

## Troubleshooting

### Issue: Library Not Found

```
Warning: Could not load BitNet library
```

**Solution:**

```powershell
# Build the C++ library
.\scripts\run_integration_tests.ps1 -Build
```

### Issue: Import Error

```
ModuleNotFoundError: No module named 'pytest'
```

**Solution:**

```bash
pip install pytest pytest-cov pytest-benchmark numpy
```

### Issue: Test Timeout

```
FAILED (timeout)
```

**Solution:**

```bash
# Increase timeout (pytest.ini or command line)
pytest tests/integration --timeout=300
```

### Issue: CMake Not Found

```
cmake: command not found
```

**Solution:**

```powershell
# Install CMake
choco install cmake ninja  # Windows (Chocolatey)
# or download from https://cmake.org/download/
```

---

## Next Steps

### Task 6: AVX-512 Optimization

```powershell
# After implementing AVX-512 kernels, benchmark real throughput
.\scripts\run_integration_tests.ps1 -Build -Benchmark

# Compare naive vs optimized
pytest tests/integration -m benchmark -v --benchmark-compare
```

### Task 17: Phase 1 Complete

```bash
# Run full Phase 1 integration tests (all models)
pytest tests/e2e/test_phase1_complete.py -v
```

---

## References

- **Test Suite:** `tests/integration/test_bitnet_generation.py` (750 lines, 23 tests)
- **Python Runner:** `scripts/run_integration_tests.py`
- **PowerShell Runner:** `scripts/run_integration_tests.ps1`
- **Progress Report:** `docs/phase1_progress_task5.md`

---

**Status:** ✅ Task 5 Complete - Testing Infrastructure Ready

**Next:** Task 6 - AVX-512 Optimized Matmul Kernels (8-12× speedup target)
