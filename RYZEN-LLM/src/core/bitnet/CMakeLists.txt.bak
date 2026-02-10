# RYZEN-LLM BitNet Core Module
# [REF:PHASE1-002] - BitNet b1.58 Ternary Inference

cmake_minimum_required(VERSION 3.20)

# BitNet core library
add_library(ryzen_llm_bitnet STATIC
    quantize.cpp
    engine.cpp
    kernels/matmul.cpp
    bitnet_layer.cpp
    bitnet_model.cpp
)

target_include_directories(ryzen_llm_bitnet PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../tmac
)

# Link with T-MAC library and OpenMP
target_link_libraries(ryzen_llm_bitnet PUBLIC
    ryzen_llm_tmac
    ryzen_llm_optimization
    OpenMP::OpenMP_CXX
)

# C++17 standard
set_target_properties(ryzen_llm_bitnet PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(ryzen_llm_bitnet PRIVATE
        /O2          # Optimize for speed
        /arch:AVX2   # Use AVX2 instead of AVX512 for compatibility
        /fp:fast     # Fast floating-point
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ryzen_llm_bitnet PRIVATE
        -O3                # Maximum optimization
        -march=haswell     # Use AVX2-capable architecture instead of native
        -mtune=generic     # Generic tuning for compatibility
        -ffast-math        # Fast floating-point
    )
endif()

# AVX-512 specific flags (disabled for compatibility - Ryzen 7 7730U doesn't support AVX-512)
# if(RYZEN_LLM_AVX512_FOUND)
#     if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
#         target_compile_options(ryzen_llm_bitnet PRIVATE /arch:AVX512)
#     else()
#         target_compile_options(ryzen_llm_bitnet PRIVATE
#             -mavx512f -mavx512dq -mavx512vl -mavx512vnni
#         )
#     endif()
#     target_compile_definitions(ryzen_llm_bitnet PRIVATE RYZEN_LLM_AVX512_ENABLED)
# endif()

# Install library
install(TARGETS ryzen_llm_bitnet
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# BitNet Tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Install headers
install(FILES
    quantize.h
    kernels/matmul.h
    bitnet_layer.h
    bitnet_model.h
    DESTINATION include/ryzen_llm/bitnet
)
