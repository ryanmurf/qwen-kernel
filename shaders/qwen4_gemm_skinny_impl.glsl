// Split-K batched projection for skinny outputs (M <= 64: the 4-row HC
// inject and the 48-row GDN alpha/beta). Grid: x = 64-token tiles, y = 64-wide
// K splits. Each workgroup dequantizes its M x 64 weight slab and stages the
// 64 x 64 activation tile once, then writes F32 partial sums to
// partial[(split*N + token)*M + row]; qwen4_gemm_reduce folds the splits in
// order, so the result is deterministic. QFMT selects the weight format.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 256) in;
layout(std430, binding = 1) readonly buffer BufX { float x[]; };
layout(std430, binding = 2) writeonly buffer BufP { float partial[]; };
layout(push_constant) uniform PC { uint M; uint K; uint N; uint xStride; uint xOff; } pc;

const uint BM = 64u, BN = 64u, BK = 64u, SK = BK + 1u;
shared float Wsh[BM * SK];
shared float Xsh[BN * SK];
#define WSH_WRITE(index, value) Wsh[index] = (value)
#include "qwen4_gemm_dequant.glsl"

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint tokBase = gl_WorkGroupID.x * BN;
    uint split = gl_WorkGroupID.y;
    uint k0 = split * BK;
    // Weights: (row, 32-block) pairs; M*2 of them.
    if (tid < pc.M * 2u) dequant32(tid >> 1u, k0 / 32u + (tid & 1u), (tid >> 1u) * SK + (tid & 1u) * 32u);
    for (uint idx = tid; idx < BN * BK; idx += 256u) {
        uint tt = idx / BK, kk = idx - tt * BK;
        uint xt = tokBase + tt;
        Xsh[tt * SK + kk] = xt < pc.N ? x[pc.xOff + xt * pc.xStride + k0 + kk] : 0.0;
    }
    barrier();
    // Outputs: M x 64 tokens; thread handles token (tid & 63) and rows tid>>6, +4, +8, ...
    uint tt = tid & 63u;
    for (uint r = tid >> 6u; r < pc.M; r += 4u) {
        float acc = 0.0;
        [[unroll]] for (uint kk = 0u; kk < BK; ++kk) acc += Wsh[r * SK + kk] * Xsh[tt * SK + kk];
        uint tok = tokBase + tt;
        if (tok < pc.N) partial[(split * pc.N + tok) * pc.M + r] = acc;
    }
}
