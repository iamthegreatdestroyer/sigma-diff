# BitNet Tests CMakeLists
# Comprehensive test suite for BitNet inference pipeline

# Test: BitNet Inference End-to-End
add_executable(test_bitnet_inference
    test_bitnet_inference.cpp
)

target_link_libraries(test_bitnet_inference
    ryzen_llm_bitnet
    ryzen_llm_tmac
)

add_test(NAME BitNetInferenceTest COMMAND test_bitnet_inference)

message(STATUS "BitNet tests configured:")
message(STATUS "  - test_bitnet_inference")
