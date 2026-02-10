# SafeTensors Loader and Weight Validator - CMake Configuration
# Add this to your main CMakeLists.txt or create as standalone

cmake_minimum_required(VERSION 3.15)

# SafeTensors I/O Library
add_library(safetensors_io
    src/io/safetensors_loader.cpp
    src/io/weight_validator.cpp
)

target_include_directories(safetensors_io
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(safetensors_io
    PUBLIC
        cxx_std_17
)

# Enable optimizations
if(MSVC)
    target_compile_options(safetensors_io PRIVATE /O2 /W4)
else()
    target_compile_options(safetensors_io PRIVATE -O3 -Wall -Wextra)
endif()

# Test executable
if(BUILD_TESTS)
    add_executable(test_safetensors_loader
        tests/test_safetensors_loader.cpp
    )
    
    target_link_libraries(test_safetensors_loader
        PRIVATE
            safetensors_io
    )
    
    target_compile_features(test_safetensors_loader
        PRIVATE
            cxx_std_17
    )
    
    if(MSVC)
        target_compile_options(test_safetensors_loader PRIVATE /O2)
    else()
        target_compile_options(test_safetensors_loader PRIVATE -O3)
    endif()
    
    add_test(NAME SafeTensorsLoader
        COMMAND test_safetensors_loader
    )
endif()

# Export targets
export(TARGETS safetensors_io
    FILE ${CMAKE_BINARY_DIR}/SafeTensorsIOTargets.cmake
)
