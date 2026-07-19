cmake_minimum_required(VERSION 3.16)
project(qwen-kernel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(APPLE)
    # Metal port: ObjC++ host, MSL compiled at runtime from shaders/metal/
    # (no offline metal toolchain required). Vulkan/glslc not needed here.
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

    add_custom_target(metal_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/shaders/metal ${CMAKE_BINARY_DIR}/shaders/metal
        COMMENT "copy shaders/metal")

    add_executable(qk src/main_metal.mm)
    target_compile_options(qk PRIVATE -fobjc-arc)
    target_link_libraries(qk PRIVATE "-framework Metal" "-framework Foundation")
    add_dependencies(qk metal_shaders)

    # libqk.dylib: the qk.h C ABI for the Rust server (same TU, no main()).
    add_library(qklib SHARED src/main_metal.mm)
    set_target_properties(qklib PROPERTIES OUTPUT_NAME qk)
    target_compile_definitions(qklib PRIVATE QK_LIBRARY)
    target_compile_options(qklib PRIVATE -fobjc-arc)
    target_link_libraries(qklib PRIVATE "-framework Metal" "-framework Foundation")
    add_dependencies(qklib metal_shaders)
    return()
endif()

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc REQUIRED)

set(SHADERS gemv_f16 gemv_q4_0 gemv_q8_0 gemv_q6_k gemv_iq4_xs gemv_iq3_xxs
    gemma4_embed gemma4_embed_batch gemma4_rms gemma4_elementwise
    gemma4_router gemma4_router_batch gemma4_select gemma4_select_batch
    gemma4_quant_q8 gemma4_gemv_q4_q8 gemma4_gemv_q4_f32
    gemma4_moe_gateup gemma4_moe_gateup_f32 gemma4_moe_gateup_batch
    gemma4_moe_down gemma4_moe_down_batch gemma4_moe_geglu_batch gemma4_moe_reduce_batch
    gemma4_attn_prep gemma4_attn gemma4_attn_prep_f16 gemma4_attn_f16
    gemma4_attn_prep_batch_f16 gemma4_attn_batch_f16 gemma4_mask_f16
    gemma4_cache_scatter_f16 gemma4_argmax
    moe_logits moe_route_select moe_route_select_hier moe_select moe_select_256 moe_group_pairs
    moe_gateup_iq3 moe_gateup_iq3_rowtile moe_gateup_iq3_grouped moe_gateup_q8 moe_gateup_q8_64
    moe_down_iq4 moe_down_iq4_grouped moe_down_q6k moe_down_q6k_grouped moe_down_q8
    rmsnorm dn_ab dn_conv dn_qknorm dn_step dn_step_reg dn_step_gate dn_gate add_rmsnorm add_rmsnorm_route vec_add
    fa_prep fa_attn argmax1 argmax2 embed_q6k add_rms3 add_rms3_grouped moe_down_q8b moe_down_q8b_32 dn_convn
    fa_prep_srv fa_attn_srv fa_attn_srv_split fa_attn_srv_split_gqa fa_attn_srv_reduce
    gemm_q8_0 gemm_q4_0 gemm_q4_0_bn32 fa_prep_batch fa_attn_batch
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
                ${CMAKE_SOURCE_DIR}/shaders/q4_0.glsl
                ${CMAKE_SOURCE_DIR}/shaders/q8_1.glsl
                ${CMAKE_SOURCE_DIR}/shaders/gemm_q4_0_impl.glsl
        COMMENT "glslc ${s}.comp")
    list(APPEND SPV_FILES ${CMAKE_BINARY_DIR}/shaders/${s}.spv)
endforeach()
add_custom_target(shaders ALL DEPENDS ${SPV_FILES})

# Stage-5 parity uses the frozen llama.cpp 571d0d5 cooperative-matrix FA
# module.  Keeping this exact binary avoids compiler-version drift in the
# numerical oracle while the surrounding cache/layout code remains qk-owned.
set(LLAMA_FA_SPV /mnt/data/llama.cpp-master/build/ggml/src/ggml-vulkan/vulkan-shaders.spv/flash_attn_f32_f16_f16acc_cm1.spv)
set(QK_LLAMA_FA_SPV ${CMAKE_BINARY_DIR}/shaders/llama_flash_attn_f16acc_cm1.spv)
set(LLAMA_MM_SPV /mnt/data/llama.cpp-master/build/ggml/src/ggml-vulkan/vulkan-shaders.spv/matmul_q4_0_f32_f16acc_cm1.spv)
set(QK_LLAMA_MM_SPV ${CMAKE_BINARY_DIR}/shaders/llama_matmul_q4_f32_f16acc_cm1.spv)
set(LLAMA_MM_ID_SPV /mnt/data/llama.cpp-master/build/ggml/src/ggml-vulkan/vulkan-shaders.spv/matmul_id_subgroup_q4_0_f32_f16acc_cm1.spv)
set(QK_LLAMA_MM_ID_SPV ${CMAKE_BINARY_DIR}/shaders/llama_matmul_id_q4_f32_f16acc_cm1.spv)
set(LLAMA_COUNT_EXPERTS_SPV /mnt/data/llama.cpp-master/build/ggml/src/ggml-vulkan/vulkan-shaders.spv/count_experts.spv)
set(QK_LLAMA_COUNT_EXPERTS_SPV ${CMAKE_BINARY_DIR}/shaders/llama_count_experts.spv)
set(LLAMA_MV_SPV /mnt/data/llama.cpp-master/build/ggml/src/ggml-vulkan/vulkan-shaders.spv/mul_mat_vec_q4_0_f32_f32_subgroup.spv)
set(QK_LLAMA_MV_SPV ${CMAKE_BINARY_DIR}/shaders/llama_mul_mat_vec_q4_f32.spv)
set(LLAMA_MV_ID_SPV /mnt/data/llama.cpp-master/build/ggml/src/ggml-vulkan/vulkan-shaders.spv/mul_mat_vec_id_q4_0_f32_f32_subgroup.spv)
set(QK_LLAMA_MV_ID_SPV ${CMAKE_BINARY_DIR}/shaders/llama_mul_mat_vec_id_q4_f32.spv)
set(LLAMA_FA_SPLIT_SPV /mnt/data/llama.cpp-master/build/ggml/src/ggml-vulkan/vulkan-shaders.spv/fa_split_k_reduce.spv)
set(QK_LLAMA_FA_SPLIT_SPV ${CMAKE_BINARY_DIR}/shaders/llama_fa_split_k_reduce.spv)
set(LLAMA_RMS_ROPE_F32_SPV /mnt/data/llama.cpp-master/build/ggml/src/ggml-vulkan/vulkan-shaders.spv/rms_norm_mul_rope_f32_f32.spv)
set(QK_LLAMA_RMS_ROPE_F32_SPV ${CMAKE_BINARY_DIR}/shaders/llama_rms_norm_mul_rope_f32_f32.spv)
set(LLAMA_RMS_ROPE_F16_SPV /mnt/data/llama.cpp-master/build/ggml/src/ggml-vulkan/vulkan-shaders.spv/rms_norm_mul_rope_f32_f16.spv)
set(QK_LLAMA_RMS_ROPE_F16_SPV ${CMAKE_BINARY_DIR}/shaders/llama_rms_norm_mul_rope_f32_f16.spv)
set(LLAMA_RMS_SPV /mnt/data/llama.cpp-master/build/ggml/src/ggml-vulkan/vulkan-shaders.spv/rms_norm_f32.spv)
set(QK_LLAMA_RMS_SPV ${CMAKE_BINARY_DIR}/shaders/llama_rms_norm_f32.spv)
add_custom_command(
    OUTPUT ${QK_LLAMA_FA_SPV} ${QK_LLAMA_MM_SPV} ${QK_LLAMA_MM_ID_SPV} ${QK_LLAMA_COUNT_EXPERTS_SPV} ${QK_LLAMA_MV_SPV} ${QK_LLAMA_MV_ID_SPV} ${QK_LLAMA_FA_SPLIT_SPV} ${QK_LLAMA_RMS_ROPE_F32_SPV} ${QK_LLAMA_RMS_ROPE_F16_SPV} ${QK_LLAMA_RMS_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LLAMA_FA_SPV} ${QK_LLAMA_FA_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LLAMA_MM_SPV} ${QK_LLAMA_MM_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LLAMA_MM_ID_SPV} ${QK_LLAMA_MM_ID_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LLAMA_COUNT_EXPERTS_SPV} ${QK_LLAMA_COUNT_EXPERTS_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LLAMA_MV_SPV} ${QK_LLAMA_MV_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LLAMA_MV_ID_SPV} ${QK_LLAMA_MV_ID_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LLAMA_FA_SPLIT_SPV} ${QK_LLAMA_FA_SPLIT_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LLAMA_RMS_ROPE_F32_SPV} ${QK_LLAMA_RMS_ROPE_F32_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LLAMA_RMS_ROPE_F16_SPV} ${QK_LLAMA_RMS_ROPE_F16_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LLAMA_RMS_SPV} ${QK_LLAMA_RMS_SPV}
    DEPENDS ${LLAMA_FA_SPV} ${LLAMA_MM_SPV} ${LLAMA_MM_ID_SPV} ${LLAMA_COUNT_EXPERTS_SPV} ${LLAMA_MV_SPV} ${LLAMA_MV_ID_SPV} ${LLAMA_FA_SPLIT_SPV} ${LLAMA_RMS_ROPE_F32_SPV} ${LLAMA_RMS_ROPE_F16_SPV} ${LLAMA_RMS_SPV}
    COMMENT "copy frozen llama.cpp cooperative Gemma-4 SPIR-V")
add_custom_target(gemma4_llama_fa ALL DEPENDS ${QK_LLAMA_FA_SPV} ${QK_LLAMA_MM_SPV}
    ${QK_LLAMA_MM_ID_SPV} ${QK_LLAMA_COUNT_EXPERTS_SPV} ${QK_LLAMA_MV_SPV}
    ${QK_LLAMA_MV_ID_SPV} ${QK_LLAMA_FA_SPLIT_SPV}
    ${QK_LLAMA_RMS_ROPE_F32_SPV} ${QK_LLAMA_RMS_ROPE_F16_SPV}
    ${QK_LLAMA_RMS_SPV})

add_executable(qk src/main.cpp)
target_link_libraries(qk PRIVATE Vulkan::Vulkan)
add_dependencies(qk shaders)
add_dependencies(qk gemma4_llama_fa)

# Shared library exposing the qk.h C ABI (server links this via dlopen).
# Same TU as the CLI, compiled with QK_LIBRARY to drop main(); OUTPUT_NAME qk -> libqk.so.
add_library(qklib SHARED src/main.cpp)
set_target_properties(qklib PROPERTIES OUTPUT_NAME qk)
target_compile_definitions(qklib PRIVATE QK_LIBRARY)
target_link_libraries(qklib PRIVATE Vulkan::Vulkan)
add_dependencies(qklib shaders)
add_dependencies(qklib gemma4_llama_fa)
