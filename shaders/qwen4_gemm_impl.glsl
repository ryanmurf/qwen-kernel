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

#if QFMT == 0
struct block_q5_K { float16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
layout(std430, binding = 0) readonly buffer BufW { block_q5_K wb[]; };
#elif QFMT == 1
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; float16_t d; };
layout(std430, binding = 0) readonly buffer BufW { block_q6_K wb[]; };
#elif QFMT == 2
struct block_q8_0 { float16_t d; int8_t qs[32]; };
layout(std430, binding = 0) readonly buffer BufW { block_q8_0 wb[]; };
#else
struct block_q5_1 { float16_t d, m; uint8_t qh[4]; uint8_t qs[16]; };
layout(std430, binding = 0) readonly buffer BufW { block_q5_1 wb[]; };
#endif
layout(std430, binding = 1) readonly buffer BufX { float x[]; };
layout(std430, binding = 2) writeonly buffer BufY { float y[]; };

layout(push_constant) uniform PC {
    uint M; uint K; uint N; uint xStride; uint xOff; uint yStride; uint yOff;
} pc;

const uint BM = 128u, BN = 64u, RM = 8u, RN = 4u;
const uint BK = 64u, SK = BK + 1u;
shared float Wsh[BM * SK];
shared float Xsh[BN * SK];

// Dequantize 32-wide chunk c (0..K/32) of weight row r into Wsh[dst..dst+32).
void dequant32(uint r, uint c, uint dst) {
#if QFMT == 0
    uint b = r * (pc.K / 256u) + c / 8u, g = c % 8u;
    uint sc = g < 4u ? uint(wb[b].scales[g]) & 63u
        : (uint(wb[b].scales[g + 4u]) & 15u) | ((uint(wb[b].scales[g - 4u]) >> 6u) << 4u);
    uint mn = g < 4u ? uint(wb[b].scales[g + 4u]) & 63u
        : (uint(wb[b].scales[g + 4u]) >> 4u) | ((uint(wb[b].scales[g]) >> 6u) << 4u);
    float d = float(wb[b].d) * float(sc), m = float(wb[b].dmin) * float(mn);
    [[unroll]] for (uint i = 0u; i < 32u; ++i) {
        uint lo = (uint(wb[b].qs[(g / 2u) * 32u + i]) >> ((g % 2u) * 4u)) & 15u;
        uint hi = (uint(wb[b].qh[i]) >> g) & 1u;
        Wsh[dst + i] = d * float(lo + 16u * hi) - m;
    }
#elif QFMT == 1
    uint b = r * (pc.K / 256u) + c / 8u;
    float d = float(wb[b].d);
    [[unroll]] for (uint half_ = 0u; half_ < 2u; ++half_) {
        uint gg = (c % 8u) * 2u + half_;  // 16-element scale group in the superblock
        uint h = gg >> 3u, rr = (gg & 7u) >> 1u, is = gg & 1u;
        float sc = float(int(wb[b].scales[h * 8u + rr * 2u + is]));
        uint qlBase = h * 64u + (rr & 1u) * 32u + is * 16u;
        uint qhBase = h * 32u + is * 16u;
        uint shift = rr * 2u;
        bool hi = rr >= 2u;
        [[unroll]] for (uint i = 0u; i < 16u; ++i) {
            uint qlv = uint(wb[b].ql[qlBase + i]);
            uint qhv = uint(wb[b].qh[qhBase + i]);
            uint lo = hi ? (qlv >> 4u) : (qlv & 15u);
            int q = int(lo | (((qhv >> shift) & 3u) << 4u)) - 32;
            Wsh[dst + half_ * 16u + i] = d * sc * float(q);
        }
    }
#elif QFMT == 2
    uint b = r * (pc.K / 32u) + c;
    float d = float(wb[b].d);
    [[unroll]] for (uint i = 0u; i < 32u; ++i) Wsh[dst + i] = d * float(int(wb[b].qs[i]));
#else
    uint b = r * (pc.K / 32u) + c;
    float d = float(wb[b].d), m = float(wb[b].m);
    [[unroll]] for (uint i = 0u; i < 32u; ++i) {
        uint lo = (uint(wb[b].qs[i % 16u]) >> ((i / 16u) * 4u)) & 15u;
        uint hi = (uint(wb[b].qh[i / 8u]) >> (i % 8u)) & 1u;
        Wsh[dst + i] = d * float(lo + 16u * hi) + m;
    }
#endif
}

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
