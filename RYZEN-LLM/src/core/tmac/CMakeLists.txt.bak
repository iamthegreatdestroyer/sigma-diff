
# T-MAC Lookup Table GEMM
# [REF:PHASE1-007] - Table-Based Matrix Multiplication
# [REF:TMAC-001] - T-MAC Lookup Tables
# [REF:TMAC-006] - AVX-512 GEMM Kernels

# Source files
set(TMAC_SOURCES
    lut_gemm.cpp
    pattern_generator.cpp
    frequency_analyzer.cpp
    delta_encoder.cpp
    table_builder.cpp
    lut_lookup.cpp
    tmac_gemm.cpp
    tmac_gemm_optimized.cpp
)

# Create T-MAC library
add_library(ryzen_llm_tmac STATIC ${TMAC_SOURCES})

# Include directories
target_include_directories(ryzen_llm_tmac PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Link dependencies
target_link_libraries(ryzen_llm_tmac PUBLIC
    OpenMP::OpenMP_CXX
)

# Enable AVX-512 for T-MAC (vectorized table lookups)
# Disabled for compatibility - Ryzen 7 7730U doesn't support AVX-512
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     target_compile_options(ryzen_llm_tmac PRIVATE
#         -mavx512f
#         -mavx512bw
#         -mavx512vnni
#         -march=native
#     )
# elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
#     target_compile_options(ryzen_llm_tmac PRIVATE
#         /arch:AVX512
#     )
# endif()

# Use conservative CPU flags for compatibility
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ryzen_llm_tmac PRIVATE
        -mavx512f
        -mavx512bw
        -mavx512vnni
        -march=native
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(ryzen_llm_tmac PRIVATE
        /arch:AVX2
    )
endif()

# C++20 features required

# T-MAC Tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
target_compile_features(ryzen_llm_tmac PUBLIC cxx_std_20)
