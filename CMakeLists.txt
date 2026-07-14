cmake_minimum_required(VERSION 3.16)
project(qwen-kernel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc REQUIRED)

set(SHADERS gemv_f16 gemv_q8_0 gemv_q6_k gemv_iq4_xs gemv_iq3_xxs
    moe_logits moe_route_select moe_route_select_hier moe_select moe_select_256 moe_group_pairs
    moe_gateup_iq3 moe_gateup_iq3_rowtile moe_gateup_iq3_grouped moe_gateup_q8 moe_gateup_q8_64
    moe_down_iq4 moe_down_iq4_grouped moe_down_q6k moe_down_q6k_grouped moe_down_q8
    rmsnorm dn_ab dn_conv dn_qknorm dn_step dn_step_reg dn_step_gate dn_gate add_rmsnorm add_rmsnorm_route vec_add
    fa_prep fa_attn argmax1 argmax2 embed_q6k add_rms3 add_rms3_grouped moe_down_q8b moe_down_q8b_32 dn_convn
    fa_prep_srv fa_attn_srv fa_attn_srv_split fa_attn_srv_split_gqa fa_attn_srv_reduce
    gemm_q8_0 fa_prep_batch fa_attn_batch
    dn_ab_batch dn_conv_batch dn_step_batch dn_gate_batch
    moe_gateup_iq4 gemm_iq4_xs embed_q8_0)
set(SPV_FILES "")
foreach(s ${SHADERS})
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/shaders/${s}.spv
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${GLSLC} -O --target-env=vulkan1.2
                ${CMAKE_SOURCE_DIR}/shaders/${s}.comp
                -o ${CMAKE_BINARY_DIR}/shaders/${s}.spv
        DEPENDS ${CMAKE_SOURCE_DIR}/shaders/${s}.comp
                ${CMAKE_SOURCE_DIR}/shaders/iq_tables.glsl
        COMMENT "glslc ${s}.comp")
    list(APPEND SPV_FILES ${CMAKE_BINARY_DIR}/shaders/${s}.spv)
endforeach()
add_custom_target(shaders ALL DEPENDS ${SPV_FILES})

add_executable(qk src/main.cpp)
target_link_libraries(qk PRIVATE Vulkan::Vulkan)
add_dependencies(qk shaders)

# Shared library exposing the qk.h C ABI (server links this via dlopen).
# Same TU as the CLI, compiled with QK_LIBRARY to drop main(); OUTPUT_NAME qk -> libqk.so.
add_library(qklib SHARED src/main.cpp)
set_target_properties(qklib PROPERTIES OUTPUT_NAME qk)
target_compile_definitions(qklib PRIVATE QK_LIBRARY)
target_link_libraries(qklib PRIVATE Vulkan::Vulkan)
add_dependencies(qklib shaders)
