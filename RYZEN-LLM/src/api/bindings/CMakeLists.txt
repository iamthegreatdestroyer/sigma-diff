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
    ryzen_llm_optimization  # Re-enabled for BitNetEngine speculative decoding
)

# Set output directory for Python module
set_target_properties(ryzen_llm_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../python/ryzanstein_llm
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../python/ryzanstein_llm
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../python/ryzanstein_llm
    PREFIX ""
)

# .pyd suffix is Windows-only; on Linux pybind11 sets the correct .so suffix
if(WIN32)
    set_target_properties(ryzen_llm_bindings PROPERTIES SUFFIX ".pyd")
endif()

# Static link standard libraries to avoid DLL dependencies
if(MINGW)
    target_link_options(ryzen_llm_bindings PRIVATE -static-libgcc -static-libstdc++)
endif()