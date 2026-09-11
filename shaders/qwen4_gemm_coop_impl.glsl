// Cooperative-matrix twin of qwen4_gemm_impl.glsl: the same 128x64 tile,
// bindings, push constants and dequantization, but weights and activations
// are staged as float16_t and multiplied by 16x16x16 KHR cooperative matrices
// with F32 accumulation (schedule of gemm_q8_0_coopmat2.comp). This is a
// reduced-precision tier: inputs round to f16 before the products. Host
// requires M % 128 == 0, K % 64 == 0 (K % 256 == 0 for K-quants), a 64-wide
// subgroup, and Y rows up to the next multiple of 64 above N to be writable.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require

layout(local_size_x = 256) in;

layout(std430, binding = 1) readonly buffer BufX { float x[]; };
layout(std430, binding = 2) writeonly buffer BufY { float y[]; };

layout(push_constant) uniform PC {
    uint M; uint K; uint N; uint xStride; uint xOff; uint yStride; uint yOff;
} pc;

const uint TILE = 16u, BM = 128u, BN = 64u, BK = 64u, SK = BK + 8u;
shared float16_t Wsh[BM * SK];
shared float16_t Xsh[BN * SK];

#define WSH_WRITE(index, value) Wsh[index] = float16_t(value)
#include "qwen4_gemm_dequant.glsl"

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint subgroup = gl_SubgroupID;
    uint rowBase = gl_WorkGroupID.x * BM;
    uint tokBase = gl_WorkGroupID.z * BN;
    uint kb = pc.K / 32u;
    uint kSteps = pc.K / BK;

    coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>
        c0 = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0),
        c1 = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0),
        c2 = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0),
        c3 = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0),
        d0 = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0),
        d1 = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0),
        d2 = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0),
        d3 = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0);

    uint wrLocal = tid & (BM - 1u);
    uint wbLocal = tid >> 7;
    uint wr = rowBase + wrLocal;
    uint la = tid >> 2u;
    uint k16 = (tid & 3u) * 16u;
    uint xt = tokBase + la;
    bool xLive = xt < pc.N;
    uint xBase = pc.xOff + xt * pc.xStride + k16;

    for (uint ks = 0u; ks < kSteps; ++ks) {
        uint c = ks * 2u + wbLocal;
        if (wr < pc.M && c < kb) dequant32(wr, c, wrLocal * SK + wbLocal * 32u);
        else { [[unroll]] for (uint i = 0u; i < 32u; ++i) Wsh[wrLocal * SK + wbLocal * 32u + i] = float16_t(0.0); }
        [[unroll]] for (uint j = 0u; j < 16u; ++j)
            Xsh[la * SK + k16 + j] = float16_t(xLive ? x[xBase + ks * BK + j] : 0.0);
        barrier();
        [[unroll]] for (uint kt = 0u; kt < BK; kt += TILE) {
            coopmat<float16_t, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseB> b0, b1, b2, b3;
            coopMatLoad(b0, Xsh, kt, SK, gl_CooperativeMatrixLayoutColumnMajor);
            coopMatLoad(b1, Xsh, 16u * SK + kt, SK, gl_CooperativeMatrixLayoutColumnMajor);
            coopMatLoad(b2, Xsh, 32u * SK + kt, SK, gl_CooperativeMatrixLayoutColumnMajor);
            coopMatLoad(b3, Xsh, 48u * SK + kt, SK, gl_CooperativeMatrixLayoutColumnMajor);
            coopmat<float16_t, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseA> a;
            coopMatLoad(a, Wsh, subgroup * TILE * SK + kt, SK, gl_CooperativeMatrixLayoutRowMajor);
            c0 = coopMatMulAdd(a, b0, c0);
            c1 = coopMatMulAdd(a, b1, c1);
            c2 = coopMatMulAdd(a, b2, c2);
            c3 = coopMatMulAdd(a, b3, c3);
            coopMatLoad(a, Wsh, (subgroup + 4u) * TILE * SK + kt, SK, gl_CooperativeMatrixLayoutRowMajor);
            d0 = coopMatMulAdd(a, b0, d0);
            d1 = coopMatMulAdd(a, b1, d1);
            d2 = coopMatMulAdd(a, b2, d2);
            d3 = coopMatMulAdd(a, b3, d3);
        }
        barrier();
    }

    uint row = rowBase + subgroup * TILE;
    uint base = pc.yOff + tokBase * pc.yStride + row;
    coopMatStore(c0, y, base, pc.yStride, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatStore(c1, y, base + TILE * pc.yStride, pc.yStride, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatStore(c2, y, base + 2u * TILE * pc.yStride, pc.yStride, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatStore(c3, y, base + 3u * TILE * pc.yStride, pc.yStride, gl_CooperativeMatrixLayoutColumnMajor);
    base += 4u * TILE;
    coopMatStore(d0, y, base, pc.yStride, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatStore(d1, y, base + TILE * pc.yStride, pc.yStride, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatStore(d2, y, base + 2u * TILE * pc.yStride, pc.yStride, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatStore(d3, y, base + 3u * TILE * pc.yStride, pc.yStride, gl_CooperativeMatrixLayoutColumnMajor);
}
