#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

// Include actual RWKV headers
#include "time_mixing.h"
#include "wkv.h"

using namespace ryzanstein_llm::rwkv;

int main()
{
    std::cout << "\n========== RWKV Minimal Smoke Test ==========\n\n";

    int passed = 0;
    int failed = 0;

    // Test 1: TimeMixingLayer Construction Only
    std::cout << "[1] TimeMixingLayer Construction... ";
    try
    {
        TimeMixingConfig config;
        TimeMixingLayer layer(128, 0, config);
        std::cout << "PASS\n";
        passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        failed++;
    }

    // Test 2: TimeMixingLayer Initialize
    std::cout << "[2] TimeMixingLayer Initialize... ";
    try
    {
        TimeMixingConfig config;
        TimeMixingLayer layer(128, 0, config);
        layer.initialize();
        std::cout << "PASS\n";
        passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        failed++;
    }

    // Test 3: WKVOperator Construction Only
    std::cout << "[3] WKVOperator Construction... ";
    try
    {
        WKVConfig config;
        WKVOperator wkv(128, config);
        std::cout << "PASS\n";
        passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        failed++;
    }

    // Test 4: WKVOperator Initialize
    std::cout << "[4] WKVOperator Initialize... ";
    try
    {
        WKVConfig config;
        WKVOperator wkv(128, config);
        wkv.initialize();
        std::cout << "PASS\n";
        passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        failed++;
    }

    // Test 5: Reset State
    std::cout << "[5] Reset State... ";
    try
    {
        WKVConfig config;
        WKVOperator wkv(128, config);
        wkv.initialize();
        wkv.reset_state();
        std::cout << "PASS\n";
        passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        failed++;
    }

    // Test 6: Get Hidden Dim
    std::cout << "[6] Get Hidden Dimension... ";
    try
    {
        TimeMixingConfig config;
        TimeMixingLayer layer(256, 0, config);
        if (layer.get_hidden_dim() != 256)
        {
            throw std::runtime_error("Hidden dim mismatch");
        }
        std::cout << "PASS\n";
        passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        failed++;
    }

    // Test 7: Get Layer ID
    std::cout << "[7] Get Layer ID... ";
    try
    {
        TimeMixingConfig config;
        TimeMixingLayer layer(128, 5, config);
        if (layer.get_layer_id() != 5)
        {
            throw std::runtime_error("Layer ID mismatch");
        }
        std::cout << "PASS\n";
        passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        failed++;
    }

    // Test 8: Multiple Layers
    std::cout << "[8] Multiple Layers... ";
    try
    {
        for (int i = 0; i < 5; ++i)
        {
            TimeMixingConfig config;
            TimeMixingLayer layer(64, i, config);
            layer.initialize();
        }
        std::cout << "PASS\n";
        passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        failed++;
    }

    // Test 9: Config Access
    std::cout << "[9] Config Access... ";
    try
    {
        TimeMixingConfig config;
        config.time_decay_rate = 0.95f;
        TimeMixingLayer layer(128, 0, config);
        auto cfg = layer.get_config();
        if (cfg.time_decay_rate != 0.95f)
        {
            throw std::runtime_error("Config not preserved");
        }
        std::cout << "PASS\n";
        passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        failed++;
    }

    // Test 10: WKV Multiple Operations
    std::cout << "[10] WKV Multiple Operations... ";
    try
    {
        WKVConfig config;
        WKVOperator wkv(64, config);
        wkv.initialize();
        wkv.reset_state();
        wkv.initialize();
        wkv.reset_state();
        std::cout << "PASS\n";
        passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        failed++;
    }

    // Summary
    std::cout << "\n========== Test Results ==========\n";
    std::cout << "Total:  " << (passed + failed) << "\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";
    std::cout << "=================================\n\n";

    return (failed == 0) ? 0 : 1;
}
