// Expert-major tiled routed down-projection. One workgroup owns a 128-row
// tile of one expert's n_embd outputs and walks the expert's pairs in groups
// of 16 tokens; each 64-wide K step dequantizes one 32-block per thread into
// LDS, then a 4-row x 2-token micro tile accumulates. Results are the gated
// partials y[(token*n_used + slot)*n_embd + o]. Host requires n_embd % 128
// == 0 and n_ff % 64 == 0. DOWN_Q51 selects Q5_1 blocks instead of Q8_0.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 256) in;

#if DOWN_Q51
struct block_q5_1 { float16_t d, m; uint8_t qh[4]; uint8_t qs[16]; };
layout(std430, binding = 0) readonly buffer W { block_q5_1 dw[]; };
#else
struct block_q8_0 { float16_t d; int8_t qs[32]; };
layout(std430, binding = 0) readonly buffer W { block_q8_0 dw[]; };
#endif
layout(std430, binding = 1) readonly buffer Hs { float hidden[]; };
struct SelT { uint ids[16]; float w[16]; float wShared; float pad[7]; };
layout(std430, binding = 2) readonly buffer S { SelT sel[]; };
layout(std430, binding = 3) readonly buffer Offsets { uint offsets[]; };
layout(std430, binding = 4) readonly buffer Pairs { uint pairs[]; };
layout(std430, binding = 5) writeonly buffer Y { float y[]; };
layout(push_constant) uniform PC { uint n_embd; uint n_ff; uint n_expert; uint n_used; } pc;

const uint BM = 128u, TN = 16u, BK = 64u, SK = BK + 1u;
shared float Wsh[BM * SK];
shared float Xsh[TN * SK];
shared uint tokTable[TN];
shared uint slotTable[TN];

void main() {
    uint tile = gl_WorkGroupID.x;
    uint eid = gl_WorkGroupID.y;
    uint tid = gl_LocalInvocationID.x;
    if (tile * BM >= pc.n_embd || eid >= pc.n_expert) return;
    uint begin = offsets[eid], end = offsets[eid + 1u];
    if (begin == end) return;

    uint blocks = pc.n_ff / 32u;
    uint rowLocal = tid & 127u;
    uint blockSel = tid >> 7;
    uint rowBlockBase = (eid * pc.n_embd + tile * BM + rowLocal) * blocks;
    uint tr = tid >> 3;
    uint tc = tid & 7u;

    for (uint p0 = begin; p0 < end; p0 += TN) {
        uint np = min(TN, end - p0);
        if (tid < TN) {
            uint pair = tid < np ? pairs[p0 + tid] : 0u;
            tokTable[tid] = pair >> 4u;
            slotTable[tid] = pair & 15u;
        }
        float acc[4][2];
        [[unroll]] for (uint i = 0u; i < 4u; ++i) { acc[i][0] = 0.0; acc[i][1] = 0.0; }
        barrier();
        for (uint k0 = 0u; k0 < pc.n_ff; k0 += BK) {
            uint b = rowBlockBase + k0 / 32u + blockSel;
            uint dst = rowLocal * SK + blockSel * 32u;
#if DOWN_Q51
            float d = float(dw[b].d), m = float(dw[b].m);
            [[unroll]] for (uint i = 0u; i < 32u; ++i) {
                uint lo = (uint(dw[b].qs[i % 16u]) >> ((i / 16u) * 4u)) & 15u;
                uint hi = (uint(dw[b].qh[i / 8u]) >> (i % 8u)) & 1u;
                Wsh[dst + i] = d * float(lo + 16u * hi) + m;
            }
#else
            float d = float(dw[b].d);
            [[unroll]] for (uint i = 0u; i < 32u; ++i) Wsh[dst + i] = d * float(int(dw[b].qs[i]));
#endif
            for (uint idx = tid; idx < TN * BK; idx += 256u) {
                uint tt = idx / BK, kk = idx - tt * BK;
                Xsh[tt * SK + kk] = tt < np
                    ? hidden[(tokTable[tt] * (pc.n_used + 1u) + slotTable[tt]) * pc.n_ff + k0 + kk] : 0.0;
            }
            barrier();
            [[unroll]] for (uint kk = 0u; kk < BK; ++kk) {
                float x0 = Xsh[(tc * 2u) * SK + kk], x1 = Xsh[(tc * 2u + 1u) * SK + kk];
                [[unroll]] for (uint i = 0u; i < 4u; ++i) {
                    float wv = Wsh[(tr * 4u + i) * SK + kk];
                    acc[i][0] += wv * x0; acc[i][1] += wv * x1;
                }
            }
            barrier();
        }
        [[unroll]] for (uint j = 0u; j < 2u; ++j) {
            uint tt = tc * 2u + j;
            if (tt < np) {
                uint tok = tokTable[tt], slot = slotTable[tt];
                float gate = sel[tok].w[slot];
                uint yo = (tok * pc.n_used + slot) * pc.n_embd + tile * BM + tr * 4u;
                [[unroll]] for (uint i = 0u; i < 4u; ++i) y[yo + i] = gate * acc[i][j];
            }
        }
        barrier();
    }
}
