/*
 * Ryzanstein LLM KV Cache Unit Tests
 * [REF:PHASE1-TEST] - Unit tests for KV cache optimization
 */

#include <gtest/gtest.h>
#include "memory/kv_cache.h"
#include <random>
#include <cmath>
#include <algorithm>

using namespace ryzanstein_llm::memory;

class KVCacheTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Reset random seed for reproducibility
        gen.seed(42);

        // Default config for tests
        config = KVCacheConfig();
        config.num_layers = 4;
        config.num_heads = 8;
        config.head_dim = 64;
        config.max_batch_size = 4;
        config.block_size = 16;
        config.num_blocks = 64;
        config.enable_quantization = false;
        config.enable_prefetching = true;
    }

    std::vector<float> generate_random_data(size_t size)
    {
        std::vector<float> data(size);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = dist(gen);
        }
        return data;
    }

    bool compare_arrays(const float *a, const float *b, size_t size, float tolerance = 1e-5f)
    {
        for (size_t i = 0; i < size; ++i)
        {
            if (std::abs(a[i] - b[i]) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    KVCacheConfig config;
    std::mt19937 gen;
};

TEST_F(KVCacheTest, Initialization)
{
    KVCacheManager cache(config);

    EXPECT_EQ(cache.GetConfig().num_layers, config.num_layers);
    EXPECT_EQ(cache.GetConfig().num_heads, config.num_heads);
    EXPECT_EQ(cache.GetConfig().head_dim, config.head_dim);
    EXPECT_GT(cache.GetMemoryUsage(), 0u);

    const auto &stats = cache.GetStats();
    EXPECT_EQ(stats.total_allocations, 0u);
    EXPECT_EQ(stats.blocks_free, config.num_blocks);

    std::cout << "KV Cache initialized with "
              << (cache.GetMemoryUsage() / (1024.0 * 1024.0)) << " MB\n";
}

TEST_F(KVCacheTest, SequenceAllocation)
{
    KVCacheManager cache(config);

    uint64_t seq_id = 1;
    uint32_t estimated_length = 32;

    EXPECT_TRUE(cache.AllocateSequence(seq_id, estimated_length));
    EXPECT_FALSE(cache.AllocateSequence(seq_id, estimated_length)); // Duplicate

    const auto &stats = cache.GetStats();
    EXPECT_GT(stats.total_allocations, 0u);
    EXPECT_LT(stats.blocks_free, config.num_blocks);

    cache.FreeSequence(seq_id);

    const auto &stats2 = cache.GetStats();
    EXPECT_GT(stats2.total_deallocations, 0u);
}

TEST_F(KVCacheTest, WriteAndReadKey)
{
    KVCacheManager cache(config);

    uint64_t seq_id = 1;
    ASSERT_TRUE(cache.AllocateSequence(seq_id, 16));

    uint32_t layer_id = 0;
    uint32_t position = 0;

    // Generate random key data
    size_t key_size = config.num_heads * config.head_dim;
    auto key_data = generate_random_data(key_size);

    // Write key
    cache.WriteKey(seq_id, layer_id, position, key_data.data());

    // Read key
    float *retrieved_key = cache.GetKeyCache(seq_id, layer_id, position);
    ASSERT_NE(retrieved_key, nullptr);

    // Verify data matches
    EXPECT_TRUE(compare_arrays(key_data.data(), retrieved_key, key_size));
}

TEST_F(KVCacheTest, WriteAndReadValue)
{
    KVCacheManager cache(config);

    uint64_t seq_id = 2;
    ASSERT_TRUE(cache.AllocateSequence(seq_id, 16));

    uint32_t layer_id = 1;
    uint32_t position = 5;

    // Generate random value data
    size_t value_size = config.num_heads * config.head_dim;
    auto value_data = generate_random_data(value_size);

    // Write value
    cache.WriteValue(seq_id, layer_id, position, value_data.data());

    // Read value
    float *retrieved_value = cache.GetValueCache(seq_id, layer_id, position);
    ASSERT_NE(retrieved_value, nullptr);

    // Verify data matches
    EXPECT_TRUE(compare_arrays(value_data.data(), retrieved_value, value_size));
}

TEST_F(KVCacheTest, MultiplePositions)
{
    KVCacheManager cache(config);

    uint64_t seq_id = 3;
    uint32_t seq_length = 48; // 3 blocks
    ASSERT_TRUE(cache.AllocateSequence(seq_id, seq_length));

    uint32_t layer_id = 0;
    size_t kv_size = config.num_heads * config.head_dim;

    // Write keys and values at multiple positions
    std::vector<std::vector<float>> keys(seq_length);
    std::vector<std::vector<float>> values(seq_length);

    for (uint32_t pos = 0; pos < seq_length; ++pos)
    {
        keys[pos] = generate_random_data(kv_size);
        values[pos] = generate_random_data(kv_size);

        cache.WriteKey(seq_id, layer_id, pos, keys[pos].data());
        cache.WriteValue(seq_id, layer_id, pos, values[pos].data());
    }

    // Verify all positions
    for (uint32_t pos = 0; pos < seq_length; ++pos)
    {
        float *key_ptr = cache.GetKeyCache(seq_id, layer_id, pos);
        float *value_ptr = cache.GetValueCache(seq_id, layer_id, pos);

        ASSERT_NE(key_ptr, nullptr);
        ASSERT_NE(value_ptr, nullptr);

        EXPECT_TRUE(compare_arrays(keys[pos].data(), key_ptr, kv_size));
        EXPECT_TRUE(compare_arrays(values[pos].data(), value_ptr, kv_size));
    }
}

TEST_F(KVCacheTest, MultipleLayersAndSequences)
{
    KVCacheManager cache(config);

    uint32_t num_sequences = 3;
    uint32_t seq_length = 32;
    size_t kv_size = config.num_heads * config.head_dim;

    // Allocate multiple sequences
    for (uint64_t seq_id = 0; seq_id < num_sequences; ++seq_id)
    {
        ASSERT_TRUE(cache.AllocateSequence(seq_id, seq_length));
    }

    // Write data to all layers and sequences
    for (uint64_t seq_id = 0; seq_id < num_sequences; ++seq_id)
    {
        for (uint32_t layer_id = 0; layer_id < config.num_layers; ++layer_id)
        {
            auto key_data = generate_random_data(kv_size);
            auto value_data = generate_random_data(kv_size);

            uint32_t pos = layer_id * 4; // Different position per layer
            cache.WriteKey(seq_id, layer_id, pos, key_data.data());
            cache.WriteValue(seq_id, layer_id, pos, value_data.data());
        }
    }

    // Verify cache statistics
    const auto &stats = cache.GetStats();
    EXPECT_GT(stats.total_allocations, 0u);
    EXPECT_LT(stats.blocks_free, config.num_blocks);

    std::cout << "Multi-sequence test: " << stats.blocks_allocated
              << " blocks allocated\n";
}

TEST_F(KVCacheTest, SequenceGrowth)
{
    KVCacheManager cache(config);

    uint64_t seq_id = 4;

    // Start with small sequence
    ASSERT_TRUE(cache.AllocateSequence(seq_id, 8));

    // Grow sequence incrementally
    EXPECT_TRUE(cache.AppendTokens(seq_id, 8));  // 16 total
    EXPECT_TRUE(cache.AppendTokens(seq_id, 16)); // 32 total
    EXPECT_TRUE(cache.AppendTokens(seq_id, 32)); // 64 total

    // Write data at various positions
    uint32_t layer_id = 0;
    size_t kv_size = config.num_heads * config.head_dim;

    std::vector<uint32_t> test_positions = {0, 15, 31, 63};
    for (uint32_t pos : test_positions)
    {
        auto key_data = generate_random_data(kv_size);
        cache.WriteKey(seq_id, layer_id, pos, key_data.data());

        float *retrieved = cache.GetKeyCache(seq_id, layer_id, pos);
        ASSERT_NE(retrieved, nullptr);
        EXPECT_TRUE(compare_arrays(key_data.data(), retrieved, kv_size));
    }
}

TEST_F(KVCacheTest, GetKeySequence)
{
    KVCacheManager cache(config);

    uint64_t seq_id = 5;
    uint32_t seq_length = 48;
    ASSERT_TRUE(cache.AllocateSequence(seq_id, seq_length));

    uint32_t layer_id = 0;
    size_t kv_size = config.num_heads * config.head_dim;

    // Write keys at all positions
    std::vector<std::vector<float>> keys(seq_length);
    for (uint32_t pos = 0; pos < seq_length; ++pos)
    {
        keys[pos] = generate_random_data(kv_size);
        cache.WriteKey(seq_id, layer_id, pos, keys[pos].data());
    }

    // Get contiguous key sequence
    uint32_t out_length = 0;
    const float *key_seq = cache.GetKeySequence(seq_id, layer_id, out_length);

    ASSERT_NE(key_seq, nullptr);
    EXPECT_EQ(out_length, seq_length);

    // Verify all keys in sequence
    for (uint32_t pos = 0; pos < seq_length; ++pos)
    {
        const float *key_at_pos = key_seq + pos * kv_size;
        EXPECT_TRUE(compare_arrays(keys[pos].data(), key_at_pos, kv_size));
    }
}

TEST_F(KVCacheTest, GetValueSequence)
{
    KVCacheManager cache(config);

    uint64_t seq_id = 6;
    uint32_t seq_length = 32;
    ASSERT_TRUE(cache.AllocateSequence(seq_id, seq_length));

    uint32_t layer_id = 1;
    size_t kv_size = config.num_heads * config.head_dim;

    // Write values at all positions
    std::vector<std::vector<float>> values(seq_length);
    for (uint32_t pos = 0; pos < seq_length; ++pos)
    {
        values[pos] = generate_random_data(kv_size);
        cache.WriteValue(seq_id, layer_id, pos, values[pos].data());
    }

    // Get contiguous value sequence
    uint32_t out_length = 0;
    const float *value_seq = cache.GetValueSequence(seq_id, layer_id, out_length);

    ASSERT_NE(value_seq, nullptr);
    EXPECT_EQ(out_length, seq_length);

    // Verify all values in sequence
    for (uint32_t pos = 0; pos < seq_length; ++pos)
    {
        const float *value_at_pos = value_seq + pos * kv_size;
        EXPECT_TRUE(compare_arrays(values[pos].data(), value_at_pos, kv_size));
    }
}

TEST_F(KVCacheTest, ForkSequence)
{
    KVCacheManager cache(config);

    uint64_t parent_id = 7;
    uint64_t child_id = 8;
    uint32_t seq_length = 32;
    uint32_t fork_position = 16;

    // Create parent sequence
    ASSERT_TRUE(cache.AllocateSequence(parent_id, seq_length));

    uint32_t layer_id = 0;
    size_t kv_size = config.num_heads * config.head_dim;

    // Write data to parent
    std::vector<std::vector<float>> parent_keys(seq_length);
    for (uint32_t pos = 0; pos < seq_length; ++pos)
    {
        parent_keys[pos] = generate_random_data(kv_size);
        cache.WriteKey(parent_id, layer_id, pos, parent_keys[pos].data());
    }

    // Fork at position 16
    ASSERT_TRUE(cache.ForkSequence(parent_id, child_id, fork_position));

    // Child should have parent's data up to fork position
    for (uint32_t pos = 0; pos < fork_position; ++pos)
    {
        float *child_key = cache.GetKeyCache(child_id, layer_id, pos);
        ASSERT_NE(child_key, nullptr);
        EXPECT_TRUE(compare_arrays(parent_keys[pos].data(), child_key, kv_size));
    }

    // Child can be extended independently
    EXPECT_TRUE(cache.AppendTokens(child_id, 16));

    auto child_key_new = generate_random_data(kv_size);
    cache.WriteKey(child_id, layer_id, fork_position, child_key_new.data());

    float *retrieved_child = cache.GetKeyCache(child_id, layer_id, fork_position);
    ASSERT_NE(retrieved_child, nullptr);
    EXPECT_TRUE(compare_arrays(child_key_new.data(), retrieved_child, kv_size));

    // Parent should be unchanged
    float *parent_key = cache.GetKeyCache(parent_id, layer_id, fork_position);
    ASSERT_NE(parent_key, nullptr);
    EXPECT_TRUE(compare_arrays(parent_keys[fork_position].data(), parent_key, kv_size));
}

TEST_F(KVCacheTest, OutOfMemory)
{
    // Small cache for testing OOM
    KVCacheConfig small_config = config;
    small_config.num_blocks = 4; // Very small

    KVCacheManager cache(small_config);

    uint32_t seq_length = 16;
    uint32_t num_sequences = 10; // Try to allocate more than we have blocks

    uint32_t successful_allocations = 0;
    for (uint64_t seq_id = 0; seq_id < num_sequences; ++seq_id)
    {
        if (cache.AllocateSequence(seq_id, seq_length))
        {
            successful_allocations++;
        }
    }

    // Should run out of blocks
    EXPECT_LT(successful_allocations, num_sequences);

    const auto &stats = cache.GetStats();
    EXPECT_EQ(stats.blocks_free, 0u);

    std::cout << "OOM test: allocated " << successful_allocations
              << " sequences before running out of blocks\n";
}

TEST_F(KVCacheTest, Statistics)
{
    KVCacheManager cache(config);

    cache.ResetStats();

    uint64_t seq_id = 9;
    ASSERT_TRUE(cache.AllocateSequence(seq_id, 32));

    const auto &stats = cache.GetStats();
    EXPECT_GT(stats.total_allocations, 0u);
    EXPECT_GT(stats.blocks_allocated, 0u);
    EXPECT_LT(stats.blocks_free, config.num_blocks);
    EXPECT_GT(stats.memory_usage_mb, 0.0);
    EXPECT_GE(stats.avg_allocation_time_us, 0.0);

    std::cout << stats.to_string() << "\n";
}

TEST_F(KVCacheTest, BoundaryConditions)
{
    KVCacheManager cache(config);

    uint64_t seq_id = 10;
    ASSERT_TRUE(cache.AllocateSequence(seq_id, 16));

    uint32_t layer_id = 0;
    size_t kv_size = config.num_heads * config.head_dim;

    // Test position at block boundary
    uint32_t block_boundary = config.block_size;
    auto key_data = generate_random_data(kv_size);

    cache.WriteKey(seq_id, layer_id, block_boundary - 1, key_data.data());

    float *retrieved = cache.GetKeyCache(seq_id, layer_id, block_boundary - 1);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_TRUE(compare_arrays(key_data.data(), retrieved, kv_size));

    // Test invalid position
    float *invalid = cache.GetKeyCache(seq_id, layer_id, 100);
    EXPECT_EQ(invalid, nullptr);

    // Test invalid sequence
    float *invalid_seq = cache.GetKeyCache(999, layer_id, 0);
    EXPECT_EQ(invalid_seq, nullptr);
}

TEST_F(KVCacheTest, MemoryAlignment)
{
    KVCacheManager cache(config);

    uint64_t seq_id = 11;
    ASSERT_TRUE(cache.AllocateSequence(seq_id, 32));

    uint32_t layer_id = 0;

    // Check that cache pointers are properly aligned
    for (uint32_t pos = 0; pos < 32; ++pos)
    {
        float *key_ptr = cache.GetKeyCache(seq_id, layer_id, pos);
        float *value_ptr = cache.GetValueCache(seq_id, layer_id, pos);

        ASSERT_NE(key_ptr, nullptr);
        ASSERT_NE(value_ptr, nullptr);

        // Check alignment (should be aligned to at least 16 bytes for SIMD)
        uintptr_t key_addr = reinterpret_cast<uintptr_t>(key_ptr);
        uintptr_t value_addr = reinterpret_cast<uintptr_t>(value_ptr);

        EXPECT_EQ(key_addr % 16, 0u) << "Key cache misaligned at position " << pos;
        EXPECT_EQ(value_addr % 16, 0u) << "Value cache misaligned at position " << pos;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
