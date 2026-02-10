# T-MAC Tests CMakeLists
# Comprehensive test suite for T-MAC lookup tables and GEMM

# Test: Basic T-MAC functionality
add_executable(test_tmac_basic
    test_tmac_basic.cpp
)

target_link_libraries(test_tmac_basic
    ryzen_llm_tmac
)

add_test(NAME TMACBasicTest COMMAND test_tmac_basic)

# Test: T-MAC GEMM
add_executable(test_tmac_gemm
    test_tmac_gemm.cpp
)

target_link_libraries(test_tmac_gemm
    ryzen_llm_tmac
)

add_test(NAME TMACGemmTest COMMAND test_tmac_gemm)

# Benchmark: GEMM Performance
add_executable(benchmark_gemm_performance
    ${PROJECT_SOURCE_DIR}/tests/benchmark_gemm_performance.cpp
)

target_link_libraries(benchmark_gemm_performance
    ryzen_llm_tmac
)

# Don't add benchmark as a test (it's for manual execution)
# add_test(NAME TMACGemmBenchmark COMMAND benchmark_gemm_performance)

message(STATUS "T-MAC tests configured:")
message(STATUS "  - test_tmac_basic")
message(STATUS "  - test_tmac_gemm")
message(STATUS "  - benchmark_gemm_performance (manual)")
