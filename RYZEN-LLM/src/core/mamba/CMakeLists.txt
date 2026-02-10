# Mamba SSM Core Library

cmake_minimum_required(VERSION 3.20)

# Library sources
set(MAMBA_SOURCES
    ssm.cpp
    scan.cpp
)

set(MAMBA_HEADERS
    ssm.h
    scan.h
)

# Create library
add_library(ryzen_llm_mamba STATIC ${MAMBA_SOURCES})

# Include directories
target_include_directories(ryzen_llm_mamba PUBLIC
    ${CMAKE_SOURCE_DIR}/src/core
)

# C++20 required
target_compile_features(ryzen_llm_mamba PUBLIC cxx_std_20)

# Compiler flags
if(MSVC)
    target_compile_options(ryzen_llm_mamba PRIVATE
        /W4 /WX
        /arch:AVX2  # Use AVX2 instead of AVX512 for compatibility
    )
else()
    target_compile_options(ryzen_llm_mamba PRIVATE
        -Wall -Wextra -fexceptions
        -march=native
        -mavx512f
        -mavx512bw
        -mavx512vl
    )
endif()

# AVX-512 detection
include(CheckCXXSourceCompiles)
set(AVX512_TEST_CODE "
#include <immintrin.h>
int main() {
    __m512 a = _mm512_setzero_ps();
    return 0;
}
")

check_cxx_source_compiles("${AVX512_TEST_CODE}" HAVE_AVX512)
if(HAVE_AVX512)
    target_compile_definitions(ryzen_llm_mamba PUBLIC HAVE_AVX512)
    message(STATUS "AVX-512 support: YES")
else()
    message(STATUS "AVX-512 support: NO (will use fallback)")
endif()

# Installation
install(TARGETS ryzen_llm_mamba
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${MAMBA_HEADERS}
    DESTINATION include/mamba
)
