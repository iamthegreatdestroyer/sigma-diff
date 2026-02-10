cmake_minimum_required(VERSION 3.16)
project(StandaloneTest)

find_package(pybind11 REQUIRED)

pybind11_add_module(standalone_test standalone_test.cpp)

set_target_properties(standalone_test PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)