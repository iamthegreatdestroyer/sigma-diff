# RYZEN-LLM Unit Tests CMakeLists.txt
# Speculative Decoding Component Tests

cmake_minimum_required(VERSION 3.20)

# Note: GoogleTest v1.14.0 requires CMake 3.20, which we satisfy

# ============================================================================
# Google Test Configuration
# ============================================================================

# Fetch GoogleTest from GitHub
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: prevent overriding parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ============================================================================
# Test Executable Definitions
# ============================================================================

# Enable testing
enable_testing()
include(GoogleTest)

# Option: control whether to use gtest_discover_tests (some MSVC/MSBuild setups invoke
# CMake at build time for test discovery which can fail on certain environments
# due to quoting/Path issues with CMake's GoogleTestAddTests.cmake.
# Default ON, set to OFF in CI or developer machines if discovery causes build-time failures.
option(ENABLE_GTEST_DISCOVERY "Use gtest_discover_tests for test registration (may invoke CMake at build time)" ON)


# Test 1: Draft Model Tests
add_executable(test_draft_model
    test_draft_model.cpp
)

target_link_libraries(test_draft_model
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        ryzen_llm_optimization
)

# Include paths for headers
target_include_directories(test_draft_model
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/optimization/speculative
        ${CMAKE_SOURCE_DIR}
)

# Compiler flags
target_compile_features(test_draft_model PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_draft_model PRIVATE /W4 /WX)
else()
    target_compile_options(test_draft_model PRIVATE -Wall -Wextra -Werror)
endif()

# Register test
if(ENABLE_GTEST_DISCOVERY)
    gtest_discover_tests(test_draft_model
        PROPERTIES
            LABELS "unit;draft_model"
    )
else()
    add_test(NAME test_draft_model COMMAND test_draft_model)
    set_tests_properties(test_draft_model PROPERTIES LABELS "unit;draft_model")
endif()

# ============================================================================
# Test 2: Verifier Tests
# ============================================================================

add_executable(test_verifier
    test_verifier.cpp
)

target_link_libraries(test_verifier
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        ryzen_llm_optimization
)

target_include_directories(test_verifier
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/optimization/speculative
        ${CMAKE_SOURCE_DIR}
)

target_compile_features(test_verifier PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_verifier PRIVATE /W4 /WX)
else()
    target_compile_options(test_verifier PRIVATE -Wall -Wextra -Werror)
endif()

if(ENABLE_GTEST_DISCOVERY)
    gtest_discover_tests(test_verifier
        PROPERTIES
            LABELS "unit;verifier"
    )
else()
    add_test(NAME test_verifier COMMAND test_verifier)
    set_tests_properties(test_verifier PROPERTIES LABELS "unit;verifier")
endif()

# ============================================================================
# Test 3: Sampling Algorithms Tests
# ============================================================================

add_executable(test_sampling_algorithms
    test_sampling_algorithms.cpp
)

target_link_libraries(test_sampling_algorithms
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        ryzen_llm_optimization
)

target_include_directories(test_sampling_algorithms
    PRIVATE
        ${CMAKE_SOURCE_DIR}
)

target_compile_features(test_sampling_algorithms PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_sampling_algorithms PRIVATE /W4 /WX)
else()
    target_compile_options(test_sampling_algorithms PRIVATE -Wall -Wextra -Werror)
endif()

if(ENABLE_GTEST_DISCOVERY)
    gtest_discover_tests(test_sampling_algorithms
        PROPERTIES
            LABELS "unit;algorithms"
    )
else()
    add_test(NAME test_sampling_algorithms COMMAND test_sampling_algorithms)
    set_tests_properties(test_sampling_algorithms PROPERTIES LABELS "unit;algorithms")
endif()

# ============================================================================
# Test Summary
# ============================================================================

message(STATUS "Speculative Decoding Unit Tests Configuration")
message(STATUS "  - test_draft_model (58 test cases)")
message(STATUS "  - test_verifier (48 test cases)")
message(STATUS "  - test_sampling_algorithms (42 test cases)")
message(STATUS "  Total: 148 test cases")
message(STATUS "")
message(STATUS "To run all tests:")
message(STATUS "  cmake --build . --target test")
message(STATUS "")
message(STATUS "To run specific test:")
message(STATUS "  ctest -L draft_model")
message(STATUS "  ctest -L verifier")
message(STATUS "  ctest -L algorithms")
