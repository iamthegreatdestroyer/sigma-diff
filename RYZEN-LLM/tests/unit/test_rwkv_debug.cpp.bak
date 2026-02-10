#include <iostream>

// Include actual RWKV headers
#include "time_mixing.h"

using namespace ryzen_llm::rwkv;

int main()
{
    std::cout << "Test 1: Include successful\n";

    try
    {
        std::cout << "Test 2: Creating TimeMixingConfig...\n";
        TimeMixingConfig config;
        std::cout << "  config.time_decay_rate = " << config.time_decay_rate << "\n";

        std::cout << "Test 3: Creating TimeMixingLayer...\n";
        TimeMixingLayer layer(128, 0, config);
        std::cout << "  hidden_dim = " << layer.get_hidden_dim() << "\n";
        std::cout << "  layer_id = " << layer.get_layer_id() << "\n";

        std::cout << "Test 4: Calling initialize()...\n";
        layer.initialize();
        std::cout << "  initialize() completed successfully\n";

        std::cout << "\n✓ All tests passed!\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n✗ Exception: " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "\n✗ Unknown exception\n";
        return 1;
    }
}
