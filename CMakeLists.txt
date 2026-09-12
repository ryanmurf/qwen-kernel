cmake_minimum_required(VERSION 3.16)
project(qwen-kernel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(qk-model-audit src/model_audit.cpp)
add_executable(qk-ple-smoke src/ple_smoke.cpp)
include(CTest)
if(BUILD_TESTING)
    add_executable(test-q5-quants tests/test_q5_quants.cpp)
    add_test(NAME q5-quants COMMAND test-q5-quants)
    add_executable(test-qwen4-ple tests/test_qwen4_ple.cpp)
    add_test(NAME qwen4-ple COMMAND test-qwen4-ple)
    add_executable(test-gguf-metadata tests/test_gguf_metadata.cpp)
    add_test(NAME gguf-metadata COMMAND test-gguf-metadata)
endif()
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

# Explicitly opt-in correctness oracle; never linked into qk or the server.
set(QK_LLAMA_REFERENCE_ROOT "" CACHE PATH "Separately built Flash Next llama.cpp reference checkout")
set(QK_LLAMA_REFERENCE_BUILD "build-vulkan" CACHE STRING "Reference build directory relative to checkout")
if(BUILD_TESTING AND QK_LLAMA_REFERENCE_ROOT)
    find_package(OpenMP REQUIRED)
    set(REFERENCE_BUILD "${QK_LLAMA_REFERENCE_ROOT}/${QK_LLAMA_REFERENCE_BUILD}")
    add_executable(flash-reference tests/flash_reference.cpp)
    target_include_directories(flash-reference PRIVATE
        "${QK_LLAMA_REFERENCE_ROOT}/include" "${QK_LLAMA_REFERENCE_ROOT}/ggml/include")
    target_link_libraries(flash-reference PRIVATE "-Wl,--start-group"
        "${REFERENCE_BUILD}/src/libllama.a"
        "${REFERENCE_BUILD}/ggml/src/libggml.a"
        "${REFERENCE_BUILD}/ggml/src/libggml-base.a"
        "${REFERENCE_BUILD}/ggml/src/libggml-cpu.a"
        "${REFERENCE_BUILD}/ggml/src/ggml-vulkan/libggml-vulkan.a"
        "-Wl,--end-group" Vulkan::Vulkan OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS} pthread)
endif()

set(SHADERS qwen4_ple_conv qwen4_silu qwen4_gdn_params moe_gateup_q5k moe_shared_q5k moe_down_q8_routed qwen4_hc gemv_f16 gemv_q8_0 gemv_q5_1 gemv_q5_k gemv_q6_k gemv_iq4_xs gemv_iq3_xxs moe_logits moe_logits_gemm
    moe_route_select moe_route_select_hier moe_select moe_select_256 moe_group_pairs moe_gateup_iq3
    moe_gateup_iq3_rowtile moe_gateup_iq3_grouped moe_gateup_iq3_grouped_rowtile moe_gateup_q8
    moe_gateup_q8_64 moe_down_iq4 moe_down_iq4_grouped moe_down_iq4_grouped_rowtile moe_down_q6k
    moe_down_q6k_grouped moe_down_q6k_grouped_rowtile moe_down_q8 moe_gateup_iq3_grouped_coopmat
    moe_down_iq4_grouped_coopmat moe_down_q6k_grouped_coopmat moe_gateup_q8_coopmat
    moe_down_q8_coopmat rmsnorm dn_ab dn_conv dn_qknorm dn_step dn_step_reg dn_step_gate dn_gate
    add_rmsnorm add_rmsnorm_route vec_add fa_prep fa_attn argmax1 argmax2 embed_q6k add_rms3
    add_rms3_grouped moe_down_q8b moe_down_q8b_32 dn_convn fa_prep_srv fa_attn_srv fa_attn_srv_split
    fa_attn_srv_split_gqa fa_attn_srv_reduce gemm_q8_0 gemm_q8_0_coopmat gemm_q8_0_coopmat2
    gemm_q8_0_coopmat2_bm64 gemm_q8_0_coopmat_split fa_prep_batch fa_attn_batch
    fa_attn_batch_coopmat fa_attn_batch_coopmat2 dn_ab_batch dn_conv_batch dn_step_batch
    dn_step_batch_split dn_step_batch_cols dn_gate_batch gemv_q4_0 gemma4_embed gemma4_embed_batch
    gemma4_rms gemma4_elementwise gemma4_post_attention_fused gemma4_router gemma4_router_batch
    gemma4_select gemma4_select_batch gemma4_quant_q8 gemma4_gemv_q4_q8 gemma4_gemv_q4_f32
    gemma4_moe_gateup gemma4_moe_gateup_f32 gemma4_moe_gateup_batch gemma4_moe_gateup_grouped
    gemma4_moe_group gemma4_moe_down gemma4_moe_down_batch gemma4_moe_down_grouped
    gemma4_moe_reduce_grouped gemma4_moe_geglu_batch gemma4_moe_reduce_batch gemma4_attn_prep
    gemma4_attn gemma4_attn_prep_f16 gemma4_attn_f16 gemma4_attn_prep_batch_f16
    gemma4_attn_batch_f16 gemma4_attn_split_f16 gemma4_attn_flash_coopmat_f16
    gemma4_attn_batch_flash_coopmat_f16 gemma4_attn_split_reduce gemma4_mask_f16
    gemma4_cache_scatter_f16 gemma4_argmax gemm_q4_0 gemm_q4_0_bn32 gemm_q4_0_coopmat
    gemma4_moe_gateup_grouped_coopmat gemma4_moe_down_grouped_coopmat

    moe_gateup_iq4 gemm_iq4_xs embed_q8_0 moe_down_q5_1 moe_down_shared_q5_1
    qwen4_gemm_q5k qwen4_gemm_q6k qwen4_gemm_q8_0 qwen4_gemm_q5_1 qwen4_gdn_conv_batch qwen4_gdn_step_batch
    qwen4_ple_conv_batch qwen4_moe_gateup_grouped qwen4_moe_down_grouped_q8 qwen4_moe_down_grouped_q51
    qwen4_moe_combine qwen4_silu_mul qwen4_argmax qwen4_moe_gateup_tiled qwen4_moe_down_tiled_q8 qwen4_moe_down_tiled_q51 gemv_q5_k_v1 qwen4_gdn_step qwen4_gemm_coop_q5k qwen4_gemm_coop_q6k qwen4_gemm_coop_q8_0 qwen4_gemm_coop_q5_1 qwen4_moe_gateup_coop qwen4_moe_down_coop_q8 qwen4_moe_down_coop_q51 qwen4_gemm_skinny_q5k qwen4_gemm_skinny_q6k qwen4_gemm_skinny_q8_0 qwen4_gemm_skinny_q5_1 qwen4_gemm_reduce moe_gateup_q5k_v2 moe_shared_q5k_v2 qwen4_gdn_step_p)
set(SPV_FILES "")
foreach(s ${SHADERS})
    set(EXTRA_SHADER_DEPS)
    if(s STREQUAL "moe_gateup_q5k" OR s STREQUAL "moe_shared_q5k")
        list(APPEND EXTRA_SHADER_DEPS ${CMAKE_SOURCE_DIR}/shaders/moe_q5k_impl.glsl)
    endif()
    if(s STREQUAL "moe_gateup_q5k_v2" OR s STREQUAL "moe_shared_q5k_v2")
        list(APPEND EXTRA_SHADER_DEPS ${CMAKE_SOURCE_DIR}/shaders/moe_q5k_impl_v2.glsl)
    endif()
    if(s STREQUAL "moe_down_q5_1" OR s STREQUAL "moe_down_shared_q5_1")
        list(APPEND EXTRA_SHADER_DEPS ${CMAKE_SOURCE_DIR}/shaders/moe_down_q5_1_impl.glsl)
    endif()
    if(s MATCHES "^qwen4_gemm_coop_")
        list(APPEND EXTRA_SHADER_DEPS ${CMAKE_SOURCE_DIR}/shaders/qwen4_gemm_coop_impl.glsl ${CMAKE_SOURCE_DIR}/shaders/qwen4_gemm_dequant.glsl)
    elseif(s MATCHES "^qwen4_gemm_skinny_")
        list(APPEND EXTRA_SHADER_DEPS ${CMAKE_SOURCE_DIR}/shaders/qwen4_gemm_skinny_impl.glsl ${CMAKE_SOURCE_DIR}/shaders/qwen4_gemm_dequant.glsl)
    elseif(s MATCHES "^qwen4_gemm_q")
        list(APPEND EXTRA_SHADER_DEPS ${CMAKE_SOURCE_DIR}/shaders/qwen4_gemm_impl.glsl ${CMAKE_SOURCE_DIR}/shaders/qwen4_gemm_dequant.glsl)
    endif()
    if(s MATCHES "^qwen4_moe_down_grouped_")
        list(APPEND EXTRA_SHADER_DEPS ${CMAKE_SOURCE_DIR}/shaders/qwen4_moe_down_grouped_impl.glsl)
    endif()
    if(s MATCHES "^qwen4_moe_down_tiled_")
        list(APPEND EXTRA_SHADER_DEPS ${CMAKE_SOURCE_DIR}/shaders/qwen4_moe_down_tiled_impl.glsl)
    endif()
    if(s MATCHES "^qwen4_moe_down_coop_")
        list(APPEND EXTRA_SHADER_DEPS ${CMAKE_SOURCE_DIR}/shaders/qwen4_moe_down_coop_impl.glsl)
    endif()
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
                ${EXTRA_SHADER_DEPS}
        COMMENT "glslc ${s}.comp")
    list(APPEND SPV_FILES ${CMAKE_BINARY_DIR}/shaders/${s}.spv)
endforeach()

# Sliding decode needs F32 cooperative accumulation for parity. Compile the
# same source separately so the global F16 path keeps its smaller LDS tile and
# original throughput.
set(GEMMA4_COOPMAT_F32_SPV
    ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_f32.spv)
add_custom_command(
    OUTPUT ${GEMMA4_COOPMAT_F32_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${GLSLC} -O --target-env=vulkan1.2 -DG4_COOPMAT_F32=1
            ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_flash_coopmat_f16.comp
            -o ${GEMMA4_COOPMAT_F32_SPV}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_flash_coopmat_f16.comp
    COMMENT "glslc gemma4_attn_flash_coopmat_f32")
list(APPEND SPV_FILES ${GEMMA4_COOPMAT_F32_SPV})

# Decode prep writes a padded f16 query tile and cooperative attention loads it
# directly, removing the per-workgroup LDS conversion tile. Separate modules
# leave the non-cooperative fallback unchanged.
set(GEMMA4_QUERY_HALF_SPV
    ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_prep_batch_query_half.spv
    ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_query_half_f16.spv
    ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_query_half_f32.spv)
add_custom_command(
    OUTPUT ${GEMMA4_QUERY_HALF_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${GLSLC} -O --target-env=vulkan1.2 -DG4_QUERY_HALF=1
            ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_prep_batch_f16.comp
            -o ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_prep_batch_query_half.spv
    COMMAND ${GLSLC} -O --target-env=vulkan1.2 -DG4_QUERY_HALF=1
            ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_flash_coopmat_f16.comp
            -o ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_query_half_f16.spv
    COMMAND ${GLSLC} -O --target-env=vulkan1.2 -DG4_QUERY_HALF=1
            -DG4_COOPMAT_F32=1
            ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_flash_coopmat_f16.comp
            -o ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_query_half_f32.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_prep_batch_f16.comp
            ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_flash_coopmat_f16.comp
    COMMENT "glslc Gemma 4 half-query decode variants")
list(APPEND SPV_FILES ${GEMMA4_QUERY_HALF_SPV})

# Profiling-only cumulative cooperative variants. Phase 1 runs QK, phase 2
# runs QK+softmax, and the normal module runs the full QK+softmax+PV path.
set(GEMMA4_COOPMAT_PROFILE_SPV
    ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_f16_p1.spv
    ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_f16_p2.spv
    ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_f32_p1.spv
    ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_f32_p2.spv)
add_custom_command(
    OUTPUT ${GEMMA4_COOPMAT_PROFILE_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${GLSLC} -O --target-env=vulkan1.2 -DG4_PROFILE_PHASE=1
            ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_flash_coopmat_f16.comp
            -o ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_f16_p1.spv
    COMMAND ${GLSLC} -O --target-env=vulkan1.2 -DG4_PROFILE_PHASE=2
            ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_flash_coopmat_f16.comp
            -o ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_f16_p2.spv
    COMMAND ${GLSLC} -O --target-env=vulkan1.2 -DG4_COOPMAT_F32=1
            -DG4_PROFILE_PHASE=1
            ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_flash_coopmat_f16.comp
            -o ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_f32_p1.spv
    COMMAND ${GLSLC} -O --target-env=vulkan1.2 -DG4_COOPMAT_F32=1
            -DG4_PROFILE_PHASE=2
            ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_flash_coopmat_f16.comp
            -o ${CMAKE_BINARY_DIR}/shaders/gemma4_attn_flash_coopmat_f32_p2.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/gemma4_attn_flash_coopmat_f16.comp
    COMMENT "glslc Gemma 4 cooperative attention profiling variants")
list(APPEND SPV_FILES ${GEMMA4_COOPMAT_PROFILE_SPV})

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
