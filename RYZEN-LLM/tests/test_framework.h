/*
 * Simple Assert-Based Testing Framework
 * No external dependencies (no GoogleTest required)
 */

#pragma once

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>

namespace test_framework
{

    class TestRunner
    {
    public:
        static TestRunner &instance()
        {
            static TestRunner runner;
            return runner;
        }

        void add_test(const std::string &name, void (*test_func)())
        {
            tests_.push_back({name, test_func});
        }

        int run_all()
        {
            int passed = 0;
            int failed = 0;

            std::cout << "\n========== Running Unit Tests ==========\n\n";

            for (const auto &test : tests_)
            {
                try
                {
                    std::cout << "[ RUN      ] " << test.name << std::endl;
                    test.func();
                    std::cout << "[       OK ] " << test.name << std::endl;
                    passed++;
                }
                catch (const std::exception &e)
                {
                    std::cout << "[   FAILED ] " << test.name << std::endl;
                    std::cout << "  Error: " << e.what() << std::endl;
                    failed++;
                }
                catch (...)
                {
                    std::cout << "[   FAILED ] " << test.name << std::endl;
                    std::cout << "  Error: Unknown exception" << std::endl;
                    failed++;
                }
            }

            std::cout << "\n========== Test Results ==========\n";
            std::cout << "Total: " << (passed + failed) << ", ";
            std::cout << "Passed: " << passed << ", ";
            std::cout << "Failed: " << failed << "\n\n";

            return (failed == 0) ? 0 : 1;
        }

    private:
        struct Test
        {
            std::string name;
            void (*func)();
        };

        std::vector<Test> tests_;
    };

// Assertion macros
#define ASSERT_TRUE(condition)                                                    \
    if (!(condition))                                                             \
    {                                                                             \
        throw std::runtime_error(std::string("Assertion failed: ") + #condition); \
    }

#define ASSERT_FALSE(condition)                                                    \
    if ((condition))                                                               \
    {                                                                              \
        throw std::runtime_error(std::string("Assertion failed: !") + #condition); \
    }

#define ASSERT_EQ(a, b)                                                                 \
    if ((a) != (b))                                                                     \
    {                                                                                   \
        throw std::runtime_error(std::string("Assertion failed: ") + #a + " == " + #b); \
    }

#define ASSERT_NEQ(a, b)                                                                \
    if ((a) == (b))                                                                     \
    {                                                                                   \
        throw std::runtime_error(std::string("Assertion failed: ") + #a + " != " + #b); \
    }

#define ASSERT_FLOAT_EQ(a, b, tol)                                                                            \
    if (std::fabs((a) - (b)) > (tol))                                                                         \
    {                                                                                                         \
        throw std::runtime_error(std::string("Float assertion failed: |") + #a + " - " + #b + "| > " + #tol); \
    }

#define REGISTER_TEST(name, func)                                         \
    struct TestRegistrar_##name                                           \
    {                                                                     \
        TestRegistrar_##name()                                            \
        {                                                                 \
            test_framework::TestRunner::instance().add_test(#name, func); \
        }                                                                 \
    } registrar_##name;

} // namespace test_framework
