// Weight-stationary native Q4_0 GEMM. The wrappers select BN=64 for normal
// prefill and BN=32 for small batches. BM=128, BK=64, local_size=256.
layout(local_size_x = 256) in;
layout(constant_id = 0) const uint Q4_BLOCKS_PER_ROW = 88u;

#include "q4_0.glsl"

layout(std430, binding = 0) readonly  buffer BufW { block_q4_0 wb[]; };
layout(std430, binding = 1) readonly  buffer BufX { float x[]; };
layout(std430, binding = 2) writeonly buffer BufY { float y[]; };
layout(push_constant) uniform PC { uint M; uint K; uint N; } pc;

const uint BM = 128u;
const uint BN = Q4_GEMM_BN;
const uint RM = 8u;
const uint RN = Q4_GEMM_BN / 16u;
const uint BK_BLOCKS = 2u;
const uint BK = 64u;
const uint SK = BK + 1u;
shared float Wsh[BM * SK];
shared float Xsh[BN * SK];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint tr = tid >> 4;
    uint tc = tid & 15u;
    uint rowBase = gl_WorkGroupID.x * BM;
    uint tokBase = gl_WorkGroupID.z * BN;
    uint blocksPerRow = Q4_BLOCKS_PER_ROW;

    float acc[RM][RN];
    [[unroll]] for (uint i = 0u; i < RM; ++i)
        [[unroll]] for (uint j = 0u; j < RN; ++j) acc[i][j] = 0.0;

    for (uint b0 = 0u; b0 < blocksPerRow; b0 += BK_BLOCKS) {
        uint wrLocal = tid & (BM - 1u);
        uint blockLocal = tid >> 7;
        uint wr = rowBase + wrLocal;
        uint block = b0 + blockLocal;
        uint woff = wrLocal * SK + blockLocal * 32u;
        if (wr < pc.M && block < blocksPerRow) {
            block_q4_0 q = wb[wr * blocksPerRow + block];
            [[unroll]] for (uint j = 0u; j < 16u; ++j) {
                float lo, hi;
                q4_0_dequant_pair(q, j, lo, hi);
                Wsh[woff + j] = lo;
                Wsh[woff + 16u + j] = hi;
            }
        } else {
            [[unroll]] for (uint j = 0u; j < 32u; ++j) Wsh[woff + j] = 0.0;
        }

        for (uint idx = tid; idx < BN * BK; idx += 256u) {
            uint tokenLocal = idx / BK;
            uint kk = idx - tokenLocal * BK;
            uint token = tokBase + tokenLocal;
            uint k = (b0 << 5) + kk;
            Xsh[tokenLocal * SK + kk] =
                (token < pc.N && k < pc.K) ? x[token * pc.K + k] : 0.0;
        }
        barrier();

        [[unroll]] for (uint bb = 0u; bb < BK_BLOCKS; ++bb) {
            [[unroll]] for (uint mi = 0u; mi < RM; ++mi) {
                uint wo = (tr * RM + mi) * SK + bb * 32u;
                [[unroll]] for (uint ni = 0u; ni < RN; ++ni) {
                    uint xo = (tc * RN + ni) * SK + bb * 32u;
                    float sum = 0.0;
                    [[unroll]] for (uint k = 0u; k < 32u; ++k)
                        sum += Wsh[wo + k] * Xsh[xo + k];
                    acc[mi][ni] += sum;
                }
            }
        }
        barrier();
    }

    [[unroll]] for (uint mi = 0u; mi < RM; ++mi) {
        uint row = rowBase + tr * RM + mi;
        [[unroll]] for (uint ni = 0u; ni < RN; ++ni) {
            uint token = tokBase + tc * RN + ni;
            if (row < pc.M && token < pc.N)
                y[token * pc.M + row] = acc[mi][ni];
        }
    }
}
