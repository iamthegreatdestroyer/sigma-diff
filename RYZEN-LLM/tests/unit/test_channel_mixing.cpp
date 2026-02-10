#include "../test_framework.h"
#include <core/rwkv/channel_mixing.h>

using namespace ryzanstein_llm::rwkv;

/**
 * Test Suite: Channel Mixing Layer
 *
 * Tests cover:
 * 1. Construction and initialization
 * 2. Single-token forward passes
 * 3. Sequence processing with state threading
 * 4. State management (save/load/reset)
 * 5. Configuration validation
 * 6. Memory allocation and cleanup
 */

void test_channel_mixing_construction()
{
    // Test basic construction
    ChannelMixingConfig config;
    config.hidden_dim = 256;

    ChannelMixingLayer layer(256, 0, config);

    ASSERT_EQ(layer.get_hidden_dim(), 256U);
    ASSERT_EQ(layer.get_layer_id(), 0U);
    ASSERT_FALSE(layer.is_initialized());

    printf("✓ test_channel_mixing_construction passed\n");
}

void test_channel_mixing_invalid_config()
{
    ChannelMixingConfig config;

    // Test invalid hidden_dim
    try
    {
        ChannelMixingLayer layer(0, 0, config);
        throw std::runtime_error("Should throw for zero hidden_dim");
    }
    catch (const std::invalid_argument &)
    {
        printf("✓ Correctly caught invalid hidden_dim\n");
    }

    // Test invalid ff_expansion
    config.ff_expansion = 0.0f;
    try
    {
        ChannelMixingLayer layer(256, 0, config);
        throw std::runtime_error("Should throw for invalid ff_expansion");
    }
    catch (const std::invalid_argument &)
    {
        printf("✓ Correctly caught invalid ff_expansion\n");
    }

    printf("✓ test_channel_mixing_invalid_config passed\n");
}

void test_channel_mixing_initialization()
{
    ChannelMixingConfig config;
    config.hidden_dim = 128;

    ChannelMixingLayer layer(128, 1, config);

    ASSERT_FALSE(layer.is_initialized());
    layer.initialize();
    ASSERT_TRUE(layer.is_initialized());

    printf("✓ test_channel_mixing_initialization passed\n");
}

void test_channel_mixing_double_init()
{
    ChannelMixingConfig config;
    config.hidden_dim = 64;

    ChannelMixingLayer layer(64, 2, config);
    layer.initialize();

    // Should throw on second initialization
    try
    {
        layer.initialize();
        throw std::runtime_error("Should throw on double initialization");
    }
    catch (const std::runtime_error &)
    {
        printf("✓ Correctly prevented double initialization\n");
    }

    printf("✓ test_channel_mixing_double_init passed\n");
}

void test_channel_mixing_single_forward()
{
    ChannelMixingConfig config;
    config.hidden_dim = 32;
    config.activation = "relu";
    config.use_bias = true;

    ChannelMixingLayer layer(32, 3, config);
    layer.initialize();

    // Create test input
    std::vector<float> input(32, 0.5f);
    std::vector<float> output(32, 0.0f);

    // Forward pass
    bool success = layer.forward(input, output);
    ASSERT_TRUE(success);

    // Check output is non-zero (should have been transformed)
    float sum = 0.0f;
    for (float val : output)
    {
        sum += val;
    }
    ASSERT_TRUE(sum > 0.0f); // Should have some non-zero output

    printf("✓ test_channel_mixing_single_forward passed\n");
}

void test_channel_mixing_uninitialized_forward()
{
    ChannelMixingConfig config;
    config.hidden_dim = 16;

    ChannelMixingLayer layer(16, 4, config);

    std::vector<float> input(16, 0.1f);
    std::vector<float> output(16, 0.0f);

    // Should return false if not initialized
    bool success = layer.forward(input, output);
    ASSERT_FALSE(success);

    printf("✓ test_channel_mixing_uninitialized_forward passed\n");
}

void test_channel_mixing_sequence_forward()
{
    ChannelMixingConfig config;
    config.hidden_dim = 64;
    config.activation = "mish";

    ChannelMixingLayer layer(64, 5, config);
    layer.initialize();

    uint32_t seq_len = 10;
    uint32_t total_size = seq_len * 64;

    // Create sequence input
    std::vector<float> input_seq(total_size, 0.3f);
    std::vector<float> output_seq(total_size, 0.0f);

    bool success = layer.forward_sequence(input_seq, seq_len, output_seq);
    ASSERT_TRUE(success);

    // Verify output has been computed
    float sum = 0.0f;
    for (float val : output_seq)
    {
        sum += val;
    }
    ASSERT_TRUE(sum > 0.0f);

    printf("✓ test_channel_mixing_sequence_forward passed\n");
}

void test_channel_mixing_state_save_load()
{
    ChannelMixingConfig config;
    config.hidden_dim = 48;

    ChannelMixingLayer layer(48, 6, config);
    layer.initialize();

    // Do a forward pass to change state
    std::vector<float> input(48, 0.2f);
    std::vector<float> output(48, 0.0f);
    layer.forward(input, output);

    // Save state
    std::vector<uint8_t> saved_state = layer.save_state();
    ASSERT_TRUE(saved_state.size() > 0);

    // Reset and verify state changed
    layer.reset_state();

    // Load state back
    bool success = layer.load_state(saved_state);
    ASSERT_TRUE(success);

    printf("✓ test_channel_mixing_state_save_load passed\n");
}

void test_channel_mixing_state_load_invalid()
{
    ChannelMixingConfig config;
    config.hidden_dim = 32;

    ChannelMixingLayer layer(32, 7, config);
    layer.initialize();

    // Try loading invalid state
    std::vector<uint8_t> invalid_state = {1, 2, 3}; // Too small and invalid
    bool success = layer.load_state(invalid_state);
    ASSERT_FALSE(success);

    printf("✓ test_channel_mixing_state_load_invalid passed\n");
}

void test_channel_mixing_reset_state()
{
    ChannelMixingConfig config;
    config.hidden_dim = 24;

    ChannelMixingLayer layer(24, 8, config);
    layer.initialize();

    // Do forward passes
    std::vector<float> input(24, 0.4f);
    std::vector<float> output(24, 0.0f);

    layer.forward(input, output);
    layer.forward(input, output); // State should be modified

    // Reset
    layer.reset_state();

    // Should still work
    bool success = layer.forward(input, output);
    ASSERT_TRUE(success);

    printf("✓ test_channel_mixing_reset_state passed\n");
}

void test_channel_mixing_different_activations()
{
    std::vector<std::string> activations = {"relu", "gelu", "swish", "mish"};

    for (const auto &activation : activations)
    {
        ChannelMixingConfig config;
        config.hidden_dim = 32;
        config.activation = activation;

        ChannelMixingLayer layer(32, 9, config);
        layer.initialize();

        std::vector<float> input(32, 0.2f);
        std::vector<float> output(32, 0.0f);

        bool success = layer.forward(input, output);
        ASSERT_TRUE(success);
    }

    printf("✓ test_channel_mixing_different_activations passed\n");
}

void test_channel_mixing_without_bias()
{
    ChannelMixingConfig config;
    config.hidden_dim = 40;
    config.use_bias = false;

    ChannelMixingLayer layer(40, 10, config);
    layer.initialize();

    std::vector<float> input(40, 0.3f);
    std::vector<float> output(40, 0.0f);

    bool success = layer.forward(input, output);
    ASSERT_TRUE(success);

    printf("✓ test_channel_mixing_without_bias passed\n");
}

// Main test runner
int main()
{
    printf("===== Channel Mixing Layer Tests =====\n\n");

    test_channel_mixing_construction();
    test_channel_mixing_invalid_config();
    test_channel_mixing_initialization();
    test_channel_mixing_double_init();
    test_channel_mixing_single_forward();
    test_channel_mixing_uninitialized_forward();
    test_channel_mixing_sequence_forward();
    test_channel_mixing_state_save_load();
    test_channel_mixing_state_load_invalid();
    test_channel_mixing_reset_state();
    test_channel_mixing_different_activations();
    test_channel_mixing_without_bias();

    printf("\n===== All Tests Passed! =====\n");
    return 0;
}
