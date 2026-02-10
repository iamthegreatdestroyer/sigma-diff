# IMPLEMENTATION FIXES REQUIRED - Ryzanstein LLM

Based on test execution results, the following implementation issues need to be addressed.

---

## PRIORITY 1: DraftModel Statistics Tracking (3 failures - 33% of total failures)

### Files Affected

- **Source:** `src/optimization/speculative/draft_model.cpp`
- **Tests:** `tests/unit/test_draft_model.cpp`

### Failing Tests

1. `RecordAcceptanceUpdatesStats` (line 215)
2. `AcceptanceRateCalculation` (line 221)
3. `VerifyRejectsTokensBelowThreshold` (in test_verifier.cpp line 217)

### Issue Analysis

**Test 1: RecordAcceptanceUpdatesStats**

```cpp
// Test expectation
stats.num_inferences = 0;
model.record_acceptance(1, 1.0f);
EXPECT_EQ(stats.num_inferences, 1);  // FAILS: actual = 0

// Root cause
DraftModel::record_acceptance() is not incrementing stats.num_inferences
```

**Test 2: AcceptanceRateCalculation**

```cpp
// Test expectation
model.record_acceptance(1, 1.0f);
EXPECT_GT(model.get_acceptance_rate(), 0.0f);  // FAILS: actual = 0

// Root cause
Statistics not being updated or calculated properly
```

**Test 3: VerifyRejectsTokensBelowThreshold**

```cpp
// Verifier expects rejection of low-probability tokens
// But acceptance_rate = 1.0 (all tokens accepted)
// This cascades from DraftModel stats issue
```

### Recommended Fix

**In `draft_model.cpp`, locate `DraftModel::record_acceptance()` method:**

```cpp
void DraftModel::record_acceptance(uint32_t token_id, float score) {
    // CURRENT (BROKEN): Does nothing

    // REQUIRED FIX:
    // 1. Validate token_id and score
    if (token_id >= config_.vocab_size) {
        return;
    }

    // 2. Increment total inferences count
    stats_.num_inferences++;  // <-- KEY: This line is missing

    // 3. Increment accepted count if above threshold
    if (score >= config_.min_acceptance_threshold) {
        stats_.num_accepted++;
    }

    // 4. Update cumulative score for averaging
    stats_.total_score += score;
}
```

**Also verify statistics getter:**

```cpp
float DraftModel::get_acceptance_rate() const {
    // REQUIRED LOGIC:
    if (stats_.num_inferences == 0) {
        return 0.0f;
    }
    return static_cast<float>(stats_.num_accepted) / stats_.num_inferences;
}
```

### Validation

Run test after fix:

```bash
cd build && cmake --build . --config Release --target test_draft_model
.\tests\unit\Release\test_draft_model.exe
# Should show: [  PASSED  ] RecordAcceptanceUpdatesStats
# Should show: [  PASSED  ] AcceptanceRateCalculation
```

---

## PRIORITY 2: DraftModel K Adjustment (1 failure - 11% of total failures)

### Files Affected

- **Source:** `src/optimization/speculative/draft_model.cpp`
- **Tests:** `tests/unit/test_draft_model.cpp:244`

### Failing Test

`SetCurrentKValidates`

### Issue Analysis

```cpp
// Test expectation
DraftModelConfig config = default_config;
config.initial_k = 4;
DraftModel model(config);

EXPECT_EQ(model.get_current_K(), 4);
model.set_current_K(8);
EXPECT_EQ(model.get_current_K(), 8);  // FAILS: actual != 8
```

### Root Cause

- `set_current_K()` method not implemented or not storing value correctly
- `get_current_K()` returning default value instead of current value

### Recommended Fix

**In `draft_model.cpp`:**

```cpp
// Add member variable to DraftModel class (in header)
// private:
//   uint32_t current_k_;

void DraftModel::set_current_K(uint32_t k) {
    // Validate k is within reasonable bounds
    if (k < 1 || k > config_.vocab_size) {
        throw std::invalid_argument("K must be between 1 and vocab_size");
    }

    // Store the new K value
    current_k_ = k;  // <-- KEY: Actually store the value
}

uint32_t DraftModel::get_current_K() const {
    return current_k_;  // <-- Return stored value, not config_.initial_k
}
```

**In constructor, initialize:**

```cpp
DraftModel::DraftModel(const DraftModelConfig& config) : config_(config) {
    current_k_ = config.initial_k;  // Initialize to config value
    // ... rest of constructor
}
```

### Validation

```bash
cd build && cmake --build . --config Release --target test_draft_model
.\tests\unit\Release\test_draft_model.exe
# Should show: [  PASSED  ] SetCurrentKValidates
```

---

## PRIORITY 3: TopP Sampling Algorithm (3 failures - 33% of total failures)

### Files Affected

- **Source:** `src/api/sampling.cpp` (or appropriate sampling module)
- **Tests:** `tests/unit/test_sampling_algorithms.cpp`

### Failing Tests

1. `TopPAccumulatesCorrectly` (line 482)
2. `TopPWithSmallPRemovesMostTokens` (line 514)
3. `TopPMonotonicityPreservation` (line 523)

### Issue Analysis

**Test 1: TopPAccumulatesCorrectly**

```cpp
// Floating-point precision error
// cumsum accumulation exceeded 1.0f
EXPECT_LE(cumsum, 1.0f);  // FAILS: actual = 1.00000012
```

**Test 2: TopPWithSmallPRemovesMostTokens**

```cpp
// Filter is not aggressive enough
// Expected < 3 tokens, got 3
float p = 0.3f;
// Should filter out most low-probability tokens
EXPECT_LT(filtered_tokens.size(), 3);  // FAILS: size = 3
```

**Test 3: TopPMonotonicityPreservation**

```cpp
// Ordering is violated after TopP filtering
EXPECT_GE(sorted_tokens[i].probability, sorted_tokens[i+1].probability);
// FAILS: tokens[3]=4 (prob=0.1) > tokens[2]=3 (prob=0.05)
```

### Root Causes

1. **Floating-point accumulation error:** Cumsum can exceed 1.0 due to rounding
2. **Threshold logic issue:** TopP cutoff not tight enough
3. **Sorting issue:** Results not properly sorted after filtering

### Recommended Fix

**In sampling module, locate TopP implementation:**

```cpp
std::vector<Token> apply_top_p(
    const std::vector<float>& logits,
    float p,
    float temperature
) {
    // Step 1: Apply temperature scaling
    std::vector<float> scaled = apply_temperature(logits, temperature);

    // Step 2: Convert to probabilities via softmax
    std::vector<float> probs = softmax(scaled);

    // Step 3: Sort by probability (descending)
    std::vector<int> indices = argsort(probs, /* descending */ true);

    // Step 4: Accumulate until threshold, with epsilon tolerance
    std::vector<int> selected;
    float cumsum = 0.0f;
    const float EPSILON = 1e-7f;  // <-- Add epsilon for floating-point

    for (int idx : indices) {
        cumsum += probs[idx];
        selected.push_back(idx);

        // Stop when cumsum >= p (with tolerance)
        if (cumsum >= p - EPSILON) {  // <-- Use epsilon comparison
            break;
        }
    }

    // Step 5: Ensure we selected at least 1 token
    if (selected.empty()) {
        selected.push_back(indices[0]);
    }

    // Step 6: Sort selected indices back to original order to preserve monotonicity
    std::sort(selected.begin(), selected.end());  // <-- Maintain ordering

    // Step 7: Normalize probabilities of selected tokens
    float selected_sum = 0.0f;
    for (int idx : selected) {
        selected_sum += probs[idx];
    }

    std::vector<Token> result;
    for (int idx : selected) {
        result.push_back({
            idx,
            probs[idx] / selected_sum  // <-- Renormalize
        });
    }

    return result;
}
```

### Key Points

- **Epsilon tolerance:** Use `1e-7f` for floating-point comparisons
- **Maintain monotonicity:** Sort results by original index after filtering
- **Renormalize:** Scale filtered probabilities to sum to 1.0
- **Minimum tokens:** Always return at least 1 token

### Validation

```bash
cd build && cmake --build . --config Release --target test_sampling_algorithms
.\tests\unit\Release\test_sampling_algorithms.exe
# Should show: [  PASSED  ] TopPAccumulatesCorrectly
# Should show: [  PASSED  ] TopPWithSmallPRemovesMostTokens
# Should show: [  PASSED  ] TopPMonotonicityPreservation
```

---

## PRIORITY 4: Verifier Token Sampling (2 failures - 22% of total failures)

### Files Affected

- **Source:** `src/optimization/speculative/verifier.cpp`
- **Tests:** `tests/unit/test_verifier.cpp`

### Failing Tests

1. `SampleTokenHandlesUniformDistribution` (line 274)
2. `SampleTokenHandlesExtremePeakDistribution` (line 287)

### Issue Analysis

```cpp
// Both tests fail with same symptom
Verifier verifier(config);
std::vector<float> logits(vocab_size, 1.0f);  // Uniform or extreme
int sampled = verifier.sample_token(logits);
EXPECT_GE(sampled, 0);  // FAILS: actual = -1 (invalid token)
```

### Root Cause

- `Verifier::sample_token()` returning -1 (error/invalid token)
- Sampling algorithm failing on certain distributions

### Recommended Fix

**In `verifier.cpp`, locate `Verifier::sample_token()` method:**

```cpp
int Verifier::sample_token(const std::vector<float>& logits) {
    // Validate input
    if (logits.empty() || logits.size() != config_.vocab_size) {
        return -1;  // Invalid
    }

    // Step 1: Apply temperature scaling
    std::vector<float> scaled_logits;
    for (float logit : logits) {
        scaled_logits.push_back(logit / config_.temperature);
    }

    // Step 2: Convert to probabilities with stability
    std::vector<float> probs = softmax_stable(scaled_logits);

    // Step 3: Use inverse transform sampling (categorical distribution)
    float u = random_uniform(0.0f, 1.0f);
    float cumsum = 0.0f;

    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (u <= cumsum) {
            return static_cast<int>(i);  // <-- Valid token
        }
    }

    // Step 4: Fallback to highest probability token (numerical stability)
    int max_idx = 0;
    float max_prob = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = i;
        }
    }

    return max_idx;  // <-- Always return valid token
}
```

### Key Points

- **Input validation:** Check logits size matches vocab_size
- **Temperature scaling:** Properly scale logits before softmax
- **Softmax stability:** Use log-sum-exp trick for numerical stability
- **Fallback:** Always return valid token (highest probability as fallback)
- **Range:** Return valid index [0, vocab_size)

### Validation

```bash
cd build && cmake --build . --config Release --target test_verifier
.\tests\unit\Release\test_verifier.exe
# Should show: [  PASSED  ] SampleTokenHandlesUniformDistribution
# Should show: [  PASSED  ] SampleTokenHandlesExtremePeakDistribution
```

---

## SUMMARY OF FIXES

| Priority | Module        | Issue                     | Tests Failed | Est. Difficulty |
| -------- | ------------- | ------------------------- | ------------ | --------------- |
| 1        | DraftModel    | Stats not tracked         | 3            | Low             |
| 2        | DraftModel    | K adjustment broken       | 1            | Low             |
| 3        | TopP Sampling | Algorithm edge cases      | 3            | Medium          |
| 4        | Verifier      | Token sampling returns -1 | 2            | Medium          |

**Total Tests to Fix:** 9/92 (9.8%)  
**Estimated Implementation Time:** 2-4 hours  
**Expected Final Pass Rate:** 100% (if all fixes applied correctly)

---

## VALIDATION CHECKLIST

After implementing each fix:

- [ ] Fix DraftModel stats tracking

  - [ ] RecordAcceptanceUpdatesStats passes
  - [ ] AcceptanceRateCalculation passes
  - [ ] VerifyRejectsTokensBelowThreshold passes

- [ ] Fix DraftModel K adjustment

  - [ ] SetCurrentKValidates passes

- [ ] Fix TopP sampling

  - [ ] TopPAccumulatesCorrectly passes
  - [ ] TopPWithSmallPRemovesMostTokens passes
  - [ ] TopPMonotonicityPreservation passes

- [ ] Fix Verifier token sampling
  - [ ] SampleTokenHandlesUniformDistribution passes
  - [ ] SampleTokenHandlesExtremePeakDistribution passes

### Full Test Run Command

```powershell
cd c:\Users\sgbil\Ryzanstein\Ryzanstein LLM\build

# Build and test
cmake --build . --config Release

# Run all tests
cd tests\unit\Release
.\test_draft_model.exe
.\test_sampling_algorithms.exe
.\test_verifier.exe

# Expected: All tests pass (92/92)
```

---

## NEXT STEPS

1. **Immediate:** Apply Priority 1 and 2 fixes (DraftModel stats/K adjustment) - straightforward
2. **Short term:** Apply Priority 3 fix (TopP sampling) - requires algorithm understanding
3. **Validation:** Run all 3 test executables and verify all 92 tests pass
4. **Coverage:** Generate code coverage report with gcov/lcov
5. **Integration:** Add tests to CI/CD pipeline

---

**Generated:** 2025 (Session conclusion)
**Test Framework:** Google Test v1.14.0
**Build System:** CMake 3.20+, MSVC 19.44, C++17
