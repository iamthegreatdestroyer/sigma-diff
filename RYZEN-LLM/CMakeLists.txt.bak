# RYZEN-LLM CMake Build Configuration
# [REF:AP-009] - Appendix: Technical Stack
# Requires CMake 3.20+ for C++17 support

cmake_minimum_required(VERSION 3.20)
project(RYZEN-LLM VERSION 2.0.0 LANGUAGES CXX)

# C++17 Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compiler-specific flags
if(MSVC)
    # MSVC flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /wd4100")  # Disable C4100 (unreferenced parameter)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /arch:AVX2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")
    
    # Enable OpenMP on MSVC
    find_package(OpenMP REQUIRED)
else()
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fexceptions")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -fopenmp")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fopenmp")
    
    # Enable OpenMP on GCC/Clang
    find_package(OpenMP REQUIRED)
endif()

# AVX-512 Detection and Flags
# NOTE: We detect compiler support but DO NOT enable /arch:AVX512 globally
# because the runtime CPU may not support it. AVX-512 code should be in
# separate translation units with runtime CPU detection.
include(CheckCXXCompilerFlag)

if(MSVC)
    # MSVC AVX-512 detection - only define HAVE_AVX512, don't set arch globally
    check_cxx_compiler_flag("/arch:AVX512" MSVC_SUPPORTS_AVX512)
    if(MSVC_SUPPORTS_AVX512)
        message(STATUS "MSVC AVX-512 compiler support detected (NOT enabling globally - use runtime detection)")
        # Do NOT add /arch:AVX512 to global flags - causes crash on CPUs without AVX-512!
        # Individual files that need AVX-512 should use set_source_files_properties()
        add_definitions(-DHAVE_AVX512_COMPILER)  # Indicates compiler can generate AVX-512
    else()
        message(WARNING "MSVC AVX-512 not supported by compiler")
    endif()
else()
    # Check for AVX-512 COMPILER support (GCC/Clang)
    # NOTE: Do NOT add -mavx512* to global flags - the runtime CPU may not support it!
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
    check_cxx_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512VL)
    check_cxx_compiler_flag("-mavx512vnni" COMPILER_SUPPORTS_AVX512VNNI)

    if(COMPILER_SUPPORTS_AVX512F AND COMPILER_SUPPORTS_AVX512DQ AND COMPILER_SUPPORTS_AVX512VL)
        message(STATUS "AVX-512 compiler support detected (NOT enabling globally - use runtime detection)")
        # Do NOT add -mavx512* to global flags - causes crash on CPUs without AVX-512!
        # Individual source files that need AVX-512 should use set_source_files_properties()
        add_definitions(-DHAVE_AVX512_COMPILER)  # Indicates compiler can generate AVX-512
        
        if(COMPILER_SUPPORTS_AVX512VNNI)
            message(STATUS "AVX-512 VNNI compiler support detected")
            add_definitions(-DHAVE_AVX512VNNI_COMPILER)
        endif()
    else()
        message(WARNING "AVX-512 not supported by compiler")
    endif()

    # FMA3 support
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    if(COMPILER_SUPPORTS_FMA)
        message(STATUS "FMA3 support detected")
        add_compile_options(-mfma)
        add_definitions(-DHAVE_FMA)
    endif()
    
    # AVX2 support (required for fallback optimizations)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "AVX2 support detected")
        add_compile_options(-mavx2)
        add_definitions(-DHAVE_AVX2)
    endif()
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/core
    ${PROJECT_SOURCE_DIR}/src/optimization
)

# Find Python and pybind11 for bindings
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Core inference engines
add_subdirectory(src/core)

# Optimization layer
add_subdirectory(src/optimization)

# Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings with pybind11" ON)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(src/api/bindings)
endif()

# Libraries
set(CORE_LIBRARIES
    ryzen_llm_core
    ryzen_llm_optimization
)

# Main executable (if needed)
# add_executable(ryzen-llm-server src/main.cpp)
# target_link_libraries(ryzen-llm-server ${CORE_LIBRARIES})

# Testing
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
install(TARGETS ${CORE_LIBRARIES}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Print configuration summary
message(STATUS "")
message(STATUS "========== RYZEN-LLM Configuration ==========")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(MSVC)
    message(STATUS "AVX-512: ${MSVC_SUPPORTS_AVX512}")
    message(STATUS "AVX2: Enabled via /arch:AVX2")
else()
    message(STATUS "AVX-512: ${COMPILER_SUPPORTS_AVX512F}")
    message(STATUS "AVX-512 VNNI: ${COMPILER_SUPPORTS_AVX512VNNI}")
    message(STATUS "FMA3: ${COMPILER_SUPPORTS_FMA}")
endif()
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "=============================================")
message(STATUS "")
