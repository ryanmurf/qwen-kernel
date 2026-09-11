// Batched native Flash GEMM: Y[N][M] = X[N][K] . W[M][K]^T with W in raw ggml
// blocks (QFMT selects the format in the including wrapper). F32 accumulation
// with the 128x64 tile, 8x4 micro-tile and padded LDS strides of gemm_q8_0.comp.
// Activation rows are read at x[xOff + n*xStride + k] and written at
// y[yOff + n*yStride + m], so slices of wider buffers can be used directly.
// Host requires K % 64 == 0 (K % 256 == 0 for the K-quant formats).
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 256) in;

layout(std430, binding = 1) readonly buffer BufX { float x[]; };
layout(std430, binding = 2) writeonly buffer BufY { float y[]; };

layout(push_constant) uniform PC {
    uint M; uint K; uint N; uint xStride; uint xOff; uint yStride; uint yOff;
} pc;

const uint BM = 128u, BN = 64u, RM = 8u, RN = 4u;
const uint BK = 64u, SK = BK + 1u;
shared float Wsh[BM * SK];
shared float Xsh[BN * SK];

#define WSH_WRITE(index, value) Wsh[index] = (value)
#include "qwen4_gemm_dequant.glsl"

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint tr = tid >> 4;
    uint tc = tid & 15u;
    uint rowBase = gl_WorkGroupID.x * BM;
    uint tokBase = gl_WorkGroupID.z * BN;
    uint kb = pc.K >> 5;

    float acc[RM][RN];
    [[unroll]] for (uint i = 0u; i < RM; ++i)
        [[unroll]] for (uint j = 0u; j < RN; ++j) acc[i][j] = 0.0;

    for (uint b0 = 0u; b0 < kb; b0 += 2u) {
        uint wrLocal = tid & (BM - 1u);
        uint wbLocal = tid >> 7;
        uint wr = rowBase + wrLocal;
        uint c = b0 + wbLocal;
        uint woff = wrLocal * SK + wbLocal * 32u;
        if (wr < pc.M && c < kb) {
            dequant32(wr, c, woff);
        } else {
            [[unroll]] for (uint i = 0u; i < 32u; ++i) Wsh[woff + i] = 0.0;
        }
        for (uint idx = tid; idx < BN * BK; idx += 256u) {
            uint tt = idx / BK;
            uint kk = idx - tt * BK;
            uint xt = tokBase + tt;
            uint xk = (b0 << 5) + kk;
            Xsh[tt * SK + kk] = (xt < pc.N && xk < pc.K) ? x[pc.xOff + xt * pc.xStride + xk] : 0.0;
        }
        barrier();

        [[unroll]] for (uint bb = 0u; bb < 2u; ++bb) {
            [[unroll]] for (uint mi = 0u; mi < RM; ++mi) {
                uint wo = (tr * RM + mi) * SK + bb * 32u;
                uint xo0 = (tc * RN + 0u) * SK + bb * 32u;
                uint xo1 = (tc * RN + 1u) * SK + bb * 32u;
                uint xo2 = (tc * RN + 2u) * SK + bb * 32u;
                uint xo3 = (tc * RN + 3u) * SK + bb * 32u;
                float s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                [[unroll]] for (uint i = 0u; i < 32u; ++i) {
                    float wv = Wsh[wo + i];
                    s0 += wv * Xsh[xo0 + i];
                    s1 += wv * Xsh[xo1 + i];
                    s2 += wv * Xsh[xo2 + i];
                    s3 += wv * Xsh[xo3 + i];
                }
                acc[mi][0] += s0;
                acc[mi][1] += s1;
                acc[mi][2] += s2;
                acc[mi][3] += s3;
            }
        }
        barrier();
    }

    [[unroll]] for (uint mi = 0u; mi < RM; ++mi) {
        uint row = rowBase + tr * RM + mi;
        [[unroll]] for (uint mj = 0u; mj < RN; ++mj) {
            uint tok = tokBase + tc * RN + mj;
            if (row < pc.M && tok < pc.N) y[pc.yOff + tok * pc.yStride + row] = acc[mi][mj];
        }
    }
}
