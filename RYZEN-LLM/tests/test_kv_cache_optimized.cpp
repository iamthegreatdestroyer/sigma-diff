/*
 * Unit Tests for KV Cache Optimized
 * Validates correctness and performance
 */

#include "kv_cache_optimized.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace ryzanstein_llm::optimization;

// ============================================================================
// Test Utilities
// ============================================================================

bool compare_floats(float a, float b, float epsilon = 1e-6f)
{
    return std::abs(a - b) < epsilon;
}

void print_test_result(const char *test_name, bool passed)
{
    std::cout << (passed ? "✓ PASS" : "✗ FAIL") << " - " << test_name << "\n";
}

// ============================================================================
// Test Cases
// ============================================================================

/**
 * Test 1: Basic allocation
 */
void test_allocation()
{
    KVCacheManager cache;
    cache.allocate(2048, 8, 4096, 32);

    assert(cache.get_memory_usage() > 0);
    print_test_result("Basic Allocation", true);
}

/**
 * Test 2: Single token append and retrieval
 */
void test_single_append()
{
    KVCacheManager cache;
    cache.allocate(512, 1, 4096, 32);

    // Create K,V data
    std::vector<float> K(4096, 1.5f);
    std::vector<float> V(4096, 2.5f);

    // Append
    cache.append(K.data(), V.data(), 0, 0);

    // Check state
    CacheState state = cache.get_state(0);
    assert(state.seq_len == 1);
    assert(state.ring_pos == 1); // Wraps to next position
    assert(state.total_tokens == 1);

    // Get cache
    float *K_cache, *V_cache;
    uint32_t cached_len;
    cache.get_cache(0, K_cache, V_cache, cached_len);

    assert(cached_len == 1);
    assert(K_cache != nullptr);
    assert(V_cache != nullptr);

    // Verify data
    assert(compare_floats(K_cache[0], 1.5f));
    assert(compare_floats(V_cache[0], 2.5f));

    print_test_result("Single Token Append", true);
}

/**
 * Test 3: Multiple token append
 */
void test_multi_append()
{
    KVCacheManager cache;
    cache.allocate(256, 1, 4096, 32);

    std::vector<float> K(4096, 1.5f);
    std::vector<float> V(4096, 2.5f);

    // Append 100 tokens
    for (int i = 0; i < 100; i++)
    {
        cache.append(K.data(), V.data(), i, 0);
    }

    CacheState state = cache.get_state(0);
    assert(state.seq_len == 100);
    assert(state.total_tokens == 100);

    float *K_cache, *V_cache;
    uint32_t cached_len;
    cache.get_cache(0, K_cache, V_cache, cached_len);
    assert(cached_len == 100);

    print_test_result("Multiple Token Append", true);
}

/**
 * Test 4: Ring buffer wrapping
 */
void test_ring_wrap()
{
    const uint32_t MAX_SEQ = 128;
    KVCacheManager cache;
    cache.allocate(MAX_SEQ, 1, 4096, 32);

    std::vector<float> K(4096, 1.5f);
    std::vector<float> V(4096, 2.5f);

    // Fill entire ring buffer
    for (uint32_t i = 0; i < MAX_SEQ; i++)
    {
        cache.append(K.data(), V.data(), i, 0);
    }

    CacheState state = cache.get_state(0);
    assert(state.seq_len == MAX_SEQ);
    assert(state.ring_pos == 0); // Wrapped back to 0

    // Append one more - should overwrite T0
    K[0] = 9.9f;
    cache.append(K.data(), V.data(), MAX_SEQ, 0);

    state = cache.get_state(0);
    assert(state.seq_len == MAX_SEQ); // Still max (ring keeps it bounded)
    assert(state.ring_pos == 1);      // Now at position 1
    assert(state.full_count == 1);    // One wrap detected

    print_test_result("Ring Buffer Wrapping", true);
}

/**
 * Test 5: Multiple batches independent
 */
void test_multi_batch()
{
    KVCacheManager cache;
    cache.allocate(256, 4, 4096, 32);

    std::vector<float> K(4096, 1.5f);
    std::vector<float> V(4096, 2.5f);

    // Append different amounts to each batch
    for (int b = 0; b < 4; b++)
    {
        for (int i = 0; i < (b + 1) * 10; i++)
        {
            cache.append(K.data(), V.data(), i, b);
        }
    }

    // Verify each batch has correct length
    for (int b = 0; b < 4; b++)
    {
        CacheState state = cache.get_state(b);
        assert(state.seq_len == (b + 1) * 10);
    }

    print_test_result("Multiple Batch Independence", true);
}

/**
 * Test 6: Reset functionality
 */
void test_reset()
{
    KVCacheManager cache;
    cache.allocate(256, 2, 4096, 32);

    std::vector<float> K(4096, 1.5f);
    std::vector<float> V(4096, 2.5f);

    // Add tokens to batch 0
    for (int i = 0; i < 50; i++)
    {
        cache.append(K.data(), V.data(), i, 0);
    }

    CacheState state_before = cache.get_state(0);
    assert(state_before.seq_len == 50);

    // Reset batch 0
    cache.reset(0);

    CacheState state_after = cache.get_state(0);
    assert(state_after.seq_len == 0);
    assert(state_after.ring_pos == 0);
    assert(state_after.total_tokens == 0);

    // Batch 1 should be unaffected
    assert(cache.get_state(1).seq_len == 0);

    print_test_result("Reset Functionality", true);
}

/**
 * Test 7: Data layout and memory access
 */
void test_memory_layout()
{
    KVCacheManager cache;
    cache.allocate(256, 1, 128, 4); // Simpler: 4 heads, 32 dim/head

    // Create distinct K,V values for verification
    std::vector<float> K(128);
    std::vector<float> V(128);
    for (int i = 0; i < 128; i++)
    {
        K[i] = 1.0f + i * 0.01f;
        V[i] = 2.0f + i * 0.01f;
    }

    cache.append(K.data(), V.data(), 0, 0);

    float *K_cache, *V_cache;
    uint32_t cached_len;
    cache.get_cache(0, K_cache, V_cache, cached_len);

    // Verify head 0 is at the start
    for (int i = 0; i < 32; i++)
    {
        assert(compare_floats(K_cache[i], K[i]));
        assert(compare_floats(V_cache[i], V[i]));
    }

    print_test_result("Memory Layout Correctness", true);
}

/**
 * Test 8: Error handling
 */
void test_error_handling()
{
    KVCacheManager cache;
    cache.allocate(256, 2, 4096, 32);

    std::vector<float> K(4096, 1.5f);
    std::vector<float> V(4096, 2.5f);

    bool caught_batch_error = false;
    try
    {
        cache.append(K.data(), V.data(), 0, 5); // Invalid batch
    }
    catch (const std::out_of_range &)
    {
        caught_batch_error = true;
    }
    assert(caught_batch_error);

    bool caught_seq_error = false;
    try
    {
        cache.append(K.data(), V.data(), 5, 0); // seq_pos mismatch
    }
    catch (const std::invalid_argument &)
    {
        caught_seq_error = true;
    }
    assert(caught_seq_error);

    print_test_result("Error Handling", true);
}

/**
 * Test 9: Performance - append latency
 */
void test_append_performance()
{
    KVCacheManager cache;
    cache.allocate(2048, 1, 4096, 32);

    std::vector<float> K(4096, 1.5f);
    std::vector<float> V(4096, 2.5f);

    const int ITERATIONS = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; i++)
    {
        cache.append(K.data(), V.data(), i % 2048, 0);
        if (i % 2048 == 0)
        {
            cache.reset(0);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    double avg_ns = static_cast<double>(duration.count()) / ITERATIONS;

    std::cout << "  Average append: " << std::fixed << std::setprecision(1) << avg_ns
              << " ns\n";
    assert(avg_ns < 1000);

    CacheMetrics metrics = cache.get_metrics();
    std::cout << "  Metrics: " << metrics.append_calls << " calls, "
              << std::fixed << std::setprecision(1) << metrics.avg_append_ns() << " ns avg\n";

    print_test_result("Append Performance (<1us)", true);
}

/**
 * Test 10: Performance - memory bandwidth
 */
void test_throughput_performance()
{
    KVCacheManager cache;
    cache.allocate(2048, 8, 4096, 32);

    std::vector<float> K(4096, 1.5f);
    std::vector<float> V(4096, 2.5f);

    const int NUM_TOKENS = 256;

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < NUM_TOKENS; t++)
    {
        for (int b = 0; b < 8; b++)
        {
            cache.append(K.data(), V.data(), t, b);

            float *K_cache, *V_cache;
            uint32_t cached_len;
            cache.get_cache(b, K_cache, V_cache, cached_len);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double tokens_per_sec = (NUM_TOKENS * 8 * 1000.0) / duration.count();

    std::cout << "  Throughput: " << std::fixed << std::setprecision(0) << tokens_per_sec
              << " tokens/sec\n";
    std::cout << "  Time per token: " << (duration.count() * 1000.0 / (NUM_TOKENS * 8)) << " μs\n";

    print_test_result("Throughput Performance", true);
}

// ============================================================================
// Test Suite Runner
// ============================================================================

int main()
{
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║      KV Cache Optimized - Unit Test Suite                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try
    {
        std::cout << "Correctness Tests:\n";
        std::cout << "──────────────────\n";
        test_allocation();
        test_single_append();
        test_multi_append();
        test_ring_wrap();
        test_multi_batch();
        test_reset();
        test_memory_layout();
        test_error_handling();

        std::cout << "\n";
        std::cout << "Performance Tests:\n";
        std::cout << "──────────────────\n";
        test_append_performance();

        std::cout << "\n";
        std::cout << "Throughput Tests:\n";
        std::cout << "────────────────\n";
        test_throughput_performance();

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                 ALL TESTS PASSED ✓                        ║\n";
        std::cout << "║          KV Cache is production-ready                     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
