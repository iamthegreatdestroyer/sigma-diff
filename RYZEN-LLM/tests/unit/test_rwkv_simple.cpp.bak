/*
 * RWKV Task 11 - Hardware Compilation Smoke Test
 * Verifies that compiled RWKV code runs without crashing
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <random>

#include "../../src/core/rwkv/time_mixing.h"
#include "../../src/core/rwkv/wkv.h"

using namespace ryzen_llm::rwkv;

// Simple random vector generator
std::vector<float> generate_random(size_t size, int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vec(size);
    for (auto &v : vec)
    {
        v = dist(gen);
    }
    return vec;
}

int main()
{
    std::cout << "\n========== RWKV Task 11: Smoke Test ==========\n\n";

    int tests_passed = 0;
    int tests_failed = 0;

    // Test 1: TimeMixingLayer Construction
    std::cout << "[1] TimeMixingLayer Construction... ";
    try
    {
        uint32_t hidden_dim = 2048;
        TimeMixingConfig config;
        TimeMixingLayer layer(hidden_dim, 0, config);
        std::cout << "PASS\n";
        tests_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        tests_failed++;
    }

    // Test 2: Forward Pass (Single Token)
    std::cout << "[2] Forward Pass (Single Token)... ";
    try
    {
        TimeMixingConfig config;
        TimeMixingLayer layer(512, 0, config);
        layer.initialize(); // Initialize layer parameters first

        std::vector<float> input = generate_random(512);
        std::vector<float> output(512);

        bool success = layer.forward(input.data(), 0, output.data());
        if (!success)
            throw std::runtime_error("Forward pass returned false");

        // Verify output has non-zero values
        float sum = 0.0f;
        for (auto v : output)
        {
            if (std::isnan(v) || std::isinf(v))
            {
                throw std::runtime_error("Output contains NaN or Inf");
            }
            sum += std::abs(v);
        }

        if (sum == 0.0f)
        {
            throw std::runtime_error("Output is all zeros");
        }

        std::cout << "PASS\n";
        tests_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        tests_failed++;
    }

    // Test 3: Forward Sequence
    std::cout << "[3] Forward Sequence... ";
    try
    {
        TimeMixingConfig config;
        TimeMixingLayer layer(256, 0, config);
        layer.initialize(); // Initialize layer parameters first

        uint32_t seq_len = 4;
        std::vector<float> input = generate_random(seq_len * 256);
        std::vector<float> output(seq_len * 256);

        bool success = layer.forward_sequence(input.data(), seq_len, output.data());
        if (!success)
            throw std::runtime_error("Forward sequence returned false");

        // Verify output validity
        for (auto v : output)
        {
            if (std::isnan(v) || std::isinf(v))
            {
                throw std::runtime_error("Output contains NaN or Inf");
            }
        }

        std::cout << "PASS\n";
        tests_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        tests_failed++;
    }

    // Test 4: WKVOperator Construction
    std::cout << "[4] WKVOperator Construction... ";
    try
    {
        WKVConfig config;
        WKVOperator wkv(512, config);
        wkv.initialize();
        std::cout << "PASS\n";
        tests_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        tests_failed++;
    }

    // Test 5: WKV Forward Pass
    std::cout << "[5] WKV Forward Pass (Single Token)... ";
    try
    {
        WKVConfig config;
        WKVOperator wkv(256, config);
        wkv.initialize();

        std::vector<float> k = generate_random(256, 1);
        std::vector<float> v = generate_random(256, 2);
        std::vector<float> w = generate_random(256, 3);
        std::vector<float> r = generate_random(256, 4);
        std::vector<float> output(256);

        bool success = wkv.forward(k.data(), v.data(), w.data(), r.data(), output.data());
        if (!success)
            throw std::runtime_error("Forward pass returned false");

        // Verify output validity
        for (auto v : output)
        {
            if (std::isnan(v) || std::isinf(v))
            {
                throw std::runtime_error("Output contains NaN or Inf");
            }
        }

        std::cout << "PASS\n";
        tests_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        tests_failed++;
    }

    // Test 6: WKV Sequence Forward
    std::cout << "[6] WKV Forward Sequence... ";
    try
    {
        WKVConfig config;
        WKVOperator wkv(128, config);
        wkv.initialize();

        uint32_t seq_len = 3;
        std::vector<float> keys = generate_random(seq_len * 128, 10);
        std::vector<float> values = generate_random(seq_len * 128, 11);
        std::vector<float> weights = generate_random(seq_len * 128, 12);
        std::vector<float> receptances = generate_random(seq_len * 128, 13);
        std::vector<float> output(seq_len * 128);

        bool success = wkv.forward_sequence(
            keys.data(), values.data(), weights.data(), receptances.data(),
            seq_len, output.data());
        if (!success)
            throw std::runtime_error("Forward sequence returned false");

        // Verify output validity
        for (auto v : output)
        {
            if (std::isnan(v) || std::isinf(v))
            {
                throw std::runtime_error("Output contains NaN or Inf");
            }
        }

        std::cout << "PASS\n";
        tests_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        tests_failed++;
    }

    // Test 7: State Management
    std::cout << "[7] State Reset... ";
    try
    {
        WKVConfig config;
        WKVOperator wkv(128, config);

        wkv.initialize();
        wkv.reset_state();
        wkv.initialize(); // Should work after reset

        std::cout << "PASS\n";
        tests_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        tests_failed++;
    }

    // Test 8: Multi-Layer Processing
    std::cout << "[8] Multi-Layer Processing... ";
    try
    {
        std::vector<TimeMixingLayer> layers;
        for (int i = 0; i < 3; ++i)
        {
            TimeMixingConfig config;
            TimeMixingLayer layer(128, i, config);
            layer.initialize(); // Initialize before using
            layers.push_back(std::move(layer));
        }

        std::vector<float> x = generate_random(128);
        for (auto &layer : layers)
        {
            std::vector<float> output(128);
            bool success = layer.forward(x.data(), 0, output.data());
            if (!success)
                throw std::runtime_error("Layer forward failed");
            x = output;
        }

        std::cout << "PASS\n";
        tests_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        tests_failed++;
    }

    // Test 9: Time Shift Operation
    std::cout << "[9] Time Shift Operation... ";
    try
    {
        TimeMixingConfig config;
        TimeMixingLayer layer(256, 0, config);
        layer.initialize(); // Initialize layer parameters first

        std::vector<float> input = generate_random(256);
        std::vector<float> output(256);

        bool success = layer.time_shift(input.data(), output.data());
        if (!success)
            throw std::runtime_error("Time shift returned false");

        std::cout << "PASS\n";
        tests_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        tests_failed++;
    }

    // Test 10: State Save/Load
    std::cout << "[10] State Save/Load... ";
    try
    {
        WKVConfig config;
        WKVOperator wkv(128, config);
        wkv.initialize();

        std::vector<float> state_buffer(256); // 2 * hidden_dim
        wkv.save_state(state_buffer.data());
        wkv.load_state(state_buffer.data());

        std::cout << "PASS\n";
        tests_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        tests_failed++;
    }

    // Summary
    std::cout << "\n========== Test Results ==========\n";
    std::cout << "Total:  " << (tests_passed + tests_failed) << "\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    std::cout << "=================================\n\n";

    if (tests_failed == 0)
    {
        std::cout << "✓ All tests passed!\n\n";
        return 0;
    }
    else
    {
        std::cout << "✗ Some tests failed!\n\n";
        return 1;
    }
}
