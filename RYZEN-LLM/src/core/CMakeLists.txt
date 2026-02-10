# Core inference engines
# [REF:PHASE1] - BitNet, Mamba, RWKV implementations

# Add subdirectories for each model architecture
add_subdirectory(bitnet)
add_subdirectory(tmac)
add_subdirectory(mamba)
add_subdirectory(rwkv)

# Aggregate core library (links all model implementations)
add_library(ryzen_llm_core INTERFACE)

target_link_libraries(ryzen_llm_core INTERFACE
    ryzen_llm_bitnet
    ryzen_llm_tmac
    ryzen_llm_mamba
    ryzen_llm_rwkv
)

target_include_directories(ryzen_llm_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
