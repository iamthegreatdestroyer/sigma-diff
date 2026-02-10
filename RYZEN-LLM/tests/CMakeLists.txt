# Unit Tests Configuration
# Using GoogleTest framework for comprehensive testing

enable_testing()

# Include unit tests subdirectory (speculative decoding, etc.)
add_subdirectory(unit)

# RWKV Unit Test (Simple version without GoogleTest)
add_executable(test_rwkv unit/test_rwkv_debug.cpp)
target_link_libraries(test_rwkv PRIVATE ryzen_llm_rwkv)
target_include_directories(test_rwkv PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/core/rwkv
    ${CMAKE_SOURCE_DIR}/tests
)
add_test(NAME RWKV COMMAND test_rwkv)

# Channel Mixing Unit Test
add_executable(test_channel_mixing unit/test_channel_mixing_simple.cpp)
target_link_libraries(test_channel_mixing PRIVATE ryzen_llm_rwkv)
target_include_directories(test_channel_mixing PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/core/rwkv
    ${CMAKE_SOURCE_DIR}/tests
)
add_test(NAME ChannelMixing COMMAND test_channel_mixing)

# Print test summary
message(STATUS "")
message(STATUS "========== Tests Configuration ==========")
message(STATUS "Tests enabled (assert-based, no external dependencies)")
message(STATUS "Available tests:")
message(STATUS "  - test_rwkv (RWKV Time Mixing & WKV Operator)")
message(STATUS "  - test_channel_mixing (RWKV Channel Mixing)")
message(STATUS "")
message(STATUS "Run tests with: ctest -V")
message(STATUS "        or  : ./Release/test_rwkv.exe")
message(STATUS "        or  : ./Release/test_channel_mixing.exe")
message(STATUS "=========================================")
message(STATUS "")
