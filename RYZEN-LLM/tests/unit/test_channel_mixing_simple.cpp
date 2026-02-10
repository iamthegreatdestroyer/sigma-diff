#include "../test_framework.h"
#include <core/rwkv/channel_mixing.h>

using namespace ryzanstein_llm::rwkv;

/**
 * Simplified Channel Mixing Layer Tests
 * Focus on basic functionality without complex state management
 */

void test_cm_construction()
{
    ChannelMixingConfig config;
    config.hidden_dim = 256;

    ChannelMixingLayer layer(256, 0, config);
    ASSERT_EQ(layer.get_hidden_dim(), 256U);
    ASSERT_FALSE(layer.is_initialized());

    printf("✓ test_cm_construction passed\n");
}

void test_cm_initialization()
{
    printf("  [TEST] Initializing ChannelMixingConfig...\n");
    fflush(stdout);

    ChannelMixingConfig config;
    config.hidden_dim = 64; // Reduced size for testing
    printf("  [TEST] Config created, hidden_dim=%u\n", config.hidden_dim);
    fflush(stdout);

    printf("  [TEST] Creating ChannelMixingLayer...\n");
    fflush(stdout);
    ChannelMixingLayer layer(64, 1, config);
    printf("  [TEST] ChannelMixingLayer created\n");
    fflush(stdout);

    printf("  [TEST] About to call initialize()...\n");
    fflush(stdout);
    layer.initialize();
    printf("  [TEST] initialize() returned successfully\n");
    fflush(stdout);

    printf("  [TEST] About to check is_initialized()...\n");
    fflush(stdout);
    bool initialized = layer.is_initialized();
    printf("  [TEST] is_initialized() returned: %d\n", initialized ? 1 : 0);
    fflush(stdout);

    ASSERT_TRUE(initialized);
    printf("  [TEST] ASSERT_TRUE passed\n");
    fflush(stdout);

    printf("✓ test_cm_initialization passed\n");
}

void test_cm_single_forward()
{
    printf("[TEST] test_cm_single_forward starting\n");
    fflush(stdout);

    ChannelMixingConfig config;
    config.hidden_dim = 64;
    config.activation = "relu";
    config.use_bias = true;
    printf("[TEST] config created\n");
    fflush(stdout);

    ChannelMixingLayer layer(64, 2, config);
    printf("[TEST] layer created\n");
    fflush(stdout);

    layer.initialize();
    printf("[TEST] layer initialized\n");
    fflush(stdout);

    std::vector<float> input(64, 0.5f);
    std::vector<float> output(64, 0.0f);
    printf("[TEST] input/output vectors created\n");
    fflush(stdout);

    printf("[TEST] About to call layer.forward()\n");
    fflush(stdout);
    bool success = layer.forward(input, output);
    printf("[TEST] forward() returned: %d\n", success ? 1 : 0);
    fflush(stdout);

    ASSERT_TRUE(success);

    printf("✓ test_cm_single_forward passed\n");
}

void test_cm_uninitialized_forward()
{
    ChannelMixingConfig config;
    config.hidden_dim = 32;

    ChannelMixingLayer layer(32, 3, config);

    std::vector<float> input(32, 0.1f);
    std::vector<float> output(32, 0.0f);

    bool success = layer.forward(input, output);
    ASSERT_FALSE(success);

    printf("✓ test_cm_uninitialized_forward passed\n");
}

void test_cm_sequence()
{
    ChannelMixingConfig config;
    config.hidden_dim = 48;

    ChannelMixingLayer layer(48, 4, config);
    layer.initialize();

    uint32_t seq_len = 5;
    uint32_t total_size = seq_len * 48;

    std::vector<float> input_seq(total_size, 0.3f);
    std::vector<float> output_seq(total_size, 0.0f);

    bool success = layer.forward_sequence(input_seq, seq_len, output_seq);
    ASSERT_TRUE(success);

    printf("✓ test_cm_sequence passed\n");
}

void test_cm_reset()
{
    ChannelMixingConfig config;
    config.hidden_dim = 40;

    ChannelMixingLayer layer(40, 5, config);
    layer.initialize();

    std::vector<float> input(40, 0.2f);
    std::vector<float> output(40, 0.0f);

    layer.forward(input, output);
    layer.reset_state();

    bool success = layer.forward(input, output);
    ASSERT_TRUE(success);

    printf("✓ test_cm_reset passed\n");
}

void test_cm_gelu()
{
    ChannelMixingConfig config;
    config.hidden_dim = 32;
    config.activation = "gelu";

    ChannelMixingLayer layer(32, 6, config);
    layer.initialize();

    std::vector<float> input(32, 0.2f);
    std::vector<float> output(32, 0.0f);

    bool success = layer.forward(input, output);
    ASSERT_TRUE(success);

    printf("✓ test_cm_gelu passed\n");
}

void test_cm_swish()
{
    ChannelMixingConfig config;
    config.hidden_dim = 32;
    config.activation = "swish";

    ChannelMixingLayer layer(32, 7, config);
    layer.initialize();

    std::vector<float> input(32, 0.2f);
    std::vector<float> output(32, 0.0f);

    bool success = layer.forward(input, output);
    ASSERT_TRUE(success);

    printf("✓ test_cm_swish passed\n");
}

int main()
{
    printf("===== Channel Mixing Layer - Simplified Tests =====\n\n");

    test_cm_construction();
    test_cm_initialization();
    test_cm_single_forward();
    test_cm_uninitialized_forward();
    test_cm_sequence();
    test_cm_reset();
    test_cm_gelu();
    test_cm_swish();

    printf("\n===== All Tests Passed! =====\n");
    return 0;
}
