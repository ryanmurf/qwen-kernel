#include <metal_stdlib>
using namespace metal;

// Weighted down-projection, routed AND shared experts in ONE dispatch with
// a single write to y (replaces the down_iq4 -> barrier -> down_q8
// read-modify-write pair):
//   y[o] = sum_s w[s]*(dE[ids[s]][o] · h_s)  +  wShared*(dS[o] · h_shared)
// One SIMDGROUP per output o; lane task ids cover the routed sub-blocks
// first, then the shared-expert q8 blocks. Two entry points share the file:
// moe_down_all_iq4 (37 layers) and moe_down_all_q6k (blk 34/38/39).
// NSG simdgroups per threadgroup; grid z batches queries.

#include "iq_tables.metal"
#include "moe_common.metal"

constant uint NSG [[function_constant(0)]];

kernel void moe_down_all_iq4(device const block_iq4_xs* dwE [[buffer(0)]],
                             device const block_q8_0*   dwS [[buffer(1)]],
                             device const float*        h   [[buffer(2)]],
                             device const SelT*         sel [[buffer(3)]],
                             device float*              y   [[buffer(4)]],
                             constant MoePC&            pc  [[buffer(5)]],
                             uint3 tgpig [[threadgroup_position_in_grid]],
                             uint  sgid  [[simdgroup_index_in_threadgroup]],
                             uint  slid  [[thread_index_in_simdgroup]])
{
    const uint o  = tgpig.x * NSG + sgid;
    const uint rq = tgpig.z;
    if (o >= pc.n_embd) return;

    const uint ho  = rq * (pc.n_used + 1u) * pc.n_ff;
    const uint kb  = pc.n_ff / 256u;    // iq4 super-blocks per row
    const uint nu  = kb * 8u;           // 32-elem sub-blocks per row
    const uint nt  = pc.n_used * nu;    // routed tasks
    const uint kb8 = pc.n_ff / 32u;     // shared-expert q8 blocks per row

    float acc = 0.0f;
    for (uint g = slid; g < nt + kb8; g += 32u) {
        if (g < nt) {                    // routed, IQ4_XS
            const uint s  = g / nu;
            const uint u  = g % nu;
            const uint b  = u >> 3u;
            const uint ib = u & 7u;
            const uint eid = min(sel[rq].ids[s], pc.n_expert - 1u);
            const ulong bi = ((ulong)eid * pc.n_embd + o) * kb + b;

            device const block_iq4_xs& blk = dwE[bi];
            const uint slb = uint(blk.scales_l[ib >> 1u]);
            const uint sh  = uint(blk.scales_h);
            const int  ls  = int(((slb >> (4u * (ib & 1u))) & 0xFu) |
                                 (((sh >> (2u * ib)) & 3u) << 4u)) - 32;
            const float dl = float(blk.d) * float(ls);

            device const packed_uchar4* qp = (device const packed_uchar4*)&blk.qs[ib * 16u];
            device const float4* hp =
                (device const float4*)(h + ho + s * pc.n_ff + b * 256u + ib * 32u);

            float s1 = 0.0f, s2 = 0.0f;
            for (uint j = 0u; j < 4u; ++j) {
                const uchar4 q = uchar4(qp[j]);
                const float4 hlo = hp[j];
                const float4 hhi = hp[4u + j];
                s1 += float(kvalues_iq4nl[q.x & 0xFu]) * hlo.x
                    + float(kvalues_iq4nl[q.y & 0xFu]) * hlo.y
                    + float(kvalues_iq4nl[q.z & 0xFu]) * hlo.z
                    + float(kvalues_iq4nl[q.w & 0xFu]) * hlo.w;
                s2 += float(kvalues_iq4nl[q.x >> 4u]) * hhi.x
                    + float(kvalues_iq4nl[q.y >> 4u]) * hhi.y
                    + float(kvalues_iq4nl[q.z >> 4u]) * hhi.z
                    + float(kvalues_iq4nl[q.w >> 4u]) * hhi.w;
            }
            acc += sel[rq].w[s] * dl * (s1 + s2);
        } else {                         // shared expert, Q8_0
            const uint b = g - nt;
            device const float4* hp =
                (device const float4*)(h + ho + pc.n_used * pc.n_ff + b * 32u);
            acc += sel[rq].wShared * q8_block_dot(dwS[(ulong)o * kb8 + b], hp);
        }
    }

    const float v = simd_sum(acc);
    if (slid == 0u) y[rq * pc.n_embd + o] = v;
}

kernel void moe_down_all_q6k(device const block_q6_K* dwE [[buffer(0)]],
                             device const block_q8_0* dwS [[buffer(1)]],
                             device const float*      h   [[buffer(2)]],
                             device const SelT*       sel [[buffer(3)]],
                             device float*            y   [[buffer(4)]],
                             constant MoePC&          pc  [[buffer(5)]],
                             uint3 tgpig [[threadgroup_position_in_grid]],
                             uint  sgid  [[simdgroup_index_in_threadgroup]],
                             uint  slid  [[thread_index_in_simdgroup]])
{
    const uint o  = tgpig.x * NSG + sgid;
    const uint rq = tgpig.z;
    if (o >= pc.n_embd) return;

    const uint ho  = rq * (pc.n_used + 1u) * pc.n_ff;
    const uint kb  = pc.n_ff / 256u;    // q6k super-blocks per row
    const uint ng  = kb * 16u;          // 16-elem groups per row
    const uint nt  = pc.n_used * ng;    // routed tasks
    const uint kb8 = pc.n_ff / 32u;     // shared-expert q8 blocks per row

    float acc = 0.0f;
    for (uint g = slid; g < nt + kb8; g += 32u) {
        if (g < nt) {                    // routed, Q6_K
            const uint s  = g / ng;
            const uint u  = g % ng;
            const uint b  = u >> 4u;
            const uint gg = u & 15u;
            const uint hh = gg >> 3u;
            const uint r  = (gg & 7u) >> 1u;
            const uint is = gg & 1u;
            const uint eid = min(sel[rq].ids[s], pc.n_expert - 1u);
            const ulong bi = ((ulong)eid * pc.n_embd + o) * kb + b;

            device const block_q6_K& blk = dwE[bi];
            const float d  = float(blk.d);
            const float sc = float(int(blk.scales[hh * 8u + r * 2u + is]));

            const uint qlBase = hh * 64u + (r & 1u) * 32u + is * 16u;
            const uint qhBase = hh * 32u + is * 16u;
            const uint shift  = r * 2u;
            const bool hi     = r >= 2u;

            device const packed_uchar4* qlp = (device const packed_uchar4*)&blk.ql[qlBase];
            device const packed_uchar4* qhp = (device const packed_uchar4*)&blk.qh[qhBase];
            device const float4* hp =
                (device const float4*)(h + ho + s * pc.n_ff + b * 256u + gg * 16u);

            float sacc = 0.0f;
            for (uint i = 0u; i < 4u; ++i) {
                const uchar4 qlv = uchar4(qlp[i]);
                const uchar4 qhv = uchar4(qhp[i]);
                const uint4 lo = hi ? (uint4(qlv) >> 4u) : (uint4(qlv) & 0xFu);
                const int4  q  = int4(lo | (((uint4(qhv) >> shift) & 3u) << 4u)) - 32;
                sacc += dot(float4(q), hp[i]);
            }
            acc += sel[rq].w[s] * d * sc * sacc;
        } else {                         // shared expert, Q8_0
            const uint b = g - nt;
            device const float4* hp =
                (device const float4*)(h + ho + pc.n_used * pc.n_ff + b * 32u);
            acc += sel[rq].wShared * q8_block_dot(dwS[(ulong)o * kb8 + b], hp);
        }
    }

    const float v = simd_sum(acc);
    if (slid == 0u) y[rq * pc.n_embd + o] = v;
}
