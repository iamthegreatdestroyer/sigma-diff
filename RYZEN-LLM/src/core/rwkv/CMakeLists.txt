# RWKV Core Module - Time Mixing and WKV Operator
# [REF:CM-003d] - CMake Build Configuration for RWKV Components

cmake_minimum_required(VERSION 3.20)
project(ryzen_llm_rwkv)

# Source files
set(RWKV_SOURCES
    time_mixing.cpp
    channel_mixing.cpp
    wkv.cpp
)

set(RWKV_HEADERS
    time_mixing.h
    channel_mixing.h
    wkv.h
)

# RWKV Core Library
add_library(ryzen_llm_rwkv STATIC ${RWKV_SOURCES})

# Include directories
target_include_directories(ryzen_llm_rwkv PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/src
)

# Compiler flags - AVX-512 support (like Mamba)
# Disabled for compatibility - Ryzanstein 7 7730U doesn't support AVX-512
# if(MSVC)
#     target_compile_options(ryzen_llm_rwkv PRIVATE
#         /arch:AVX512 /O2 /fp:fast /GR- /EHsc-
#     )
# else()
#     target_compile_options(ryzen_llm_rwkv PRIVATE
#         -march=native -mavx512f -mavx512cd -mavx512bw -mavx512dq
#         -O3 -ffast-math -fno-exceptions -fno-rtti
#     )
# endif()

# Use conservative CPU flags for compatibility
if(MSVC)
    target_compile_options(ryzen_llm_rwkv PRIVATE
        /arch:AVX2 /O2 /fp:fast /GR- /EHsc-
    )
else()
    target_compile_options(ryzen_llm_rwkv PRIVATE
        -march=haswell -mtune=generic
        -O3 -ffast-math -fno-exceptions -fno-rtti
    )
endif()

# C++ Standard
set_target_properties(ryzen_llm_rwkv PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Optimizations
target_compile_definitions(ryzen_llm_rwkv PRIVATE
    RWKV_ENABLE_AVX512=1
    RWKV_FAST_MATH=1
)

message(STATUS "RWKV Core module configured")
