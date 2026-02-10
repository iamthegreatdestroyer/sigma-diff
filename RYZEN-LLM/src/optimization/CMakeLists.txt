# Optimization layer
# - KV-Cache Management
# - Speculative Decoding
# - Advanced AVX-512 SIMD Kernels
# - Token Recycling System

set(OPTIMIZATION_SOURCES
    cache_manager.cpp
    avx512/matmul.cpp
    avx512/activation.cpp
    avx512/vnni.cpp
    speculative/draft_model.cpp
    speculative/verifier.cpp
    speculative/speculative_decoder.cpp
    memory/kv_cache.cpp
    memory/kv_cache_optimized.cpp
    memory/pool.cpp
    ../recycler/basic_recycler.cpp
)

add_library(ryzen_llm_optimization STATIC ${OPTIMIZATION_SOURCES})

target_include_directories(ryzen_llm_optimization PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../recycler
)

target_link_libraries(ryzen_llm_optimization
    ryzen_llm_core
)

# AVX-512 compilation flags DISABLED - Ryzanstein 7 7730U (Zen 3) only has AVX2
# The matmul.cpp has runtime CPU detection that falls back to naive kernels
# DO NOT enable /arch:AVX512 as it makes ALL code in the file use AVX-512 instructions
# 
# For systems with AVX-512 support (Zen 4+), enable with:
#   cmake .. -DENABLE_AVX512=ON
option(ENABLE_AVX512 "Enable AVX-512 optimizations (requires Zen4+ or Intel)" OFF)

if(ENABLE_AVX512)
    message(STATUS "AVX-512 optimizations ENABLED")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            avx512/matmul.cpp
            avx512/activation.cpp
            avx512/vnni.cpp
            PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vnni"
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set_source_files_properties(
            avx512/matmul.cpp
            avx512/activation.cpp
            avx512/vnni.cpp
            PROPERTIES COMPILE_FLAGS "/arch:AVX512"
        )
    endif()
else()
    message(STATUS "AVX-512 optimizations DISABLED (using AVX2 fallback)")
    # Use AVX2 for better compatibility (Zen 3+ and most modern Intel)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            avx512/matmul.cpp
            avx512/activation.cpp
            avx512/vnni.cpp
            PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set_source_files_properties(
            avx512/matmul.cpp
            avx512/activation.cpp
            avx512/vnni.cpp
            PROPERTIES COMPILE_FLAGS "/arch:AVX2"
        )
    endif()
endif()
