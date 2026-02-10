# Python Bindings CMakeLists.txt
# [REF:API-008c] - API Layer: Python-C++ Bindings

# Find pybind11
find_package(pybind11 REQUIRED)

# Create pybind11 module
pybind11_add_module(ryzen_llm_bindings
    bitnet_bindings.cpp
)

# Link with required libraries
target_link_libraries(ryzen_llm_bindings PRIVATE
    ryzen_llm_core
    ryzen_llm_bitnet
    # ryzen_llm_optimization  # Temporarily disabled for AVX-512 compatibility testing
)

# Set output directory for Python module
set_target_properties(ryzen_llm_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../python/ryzen_llm
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../python/ryzen_llm
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../python/ryzen_llm
    PREFIX ""
    SUFFIX ".pyd"
)

# Static link standard libraries to avoid DLL dependencies
if(MINGW)
    target_link_options(ryzen_llm_bindings PRIVATE -static-libgcc -static-libstdc++)
endif()