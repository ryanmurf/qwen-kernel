#include <metal_stdlib>
using namespace metal;

// Gate+up for routed AND shared experts in ONE dispatch (the shared
// expert rides in slot n_used; a dedicated dispatch would cost a barrier):
// NOTE an inline-reselect variant (moe_pick_all per simdgroup, no select
// stage) measured 39% SLOWER — the +30 register footprint collapses
// occupancy on this DRAM-bound kernel. Selection stays a separate stage.
//   slot s < n_used:  h[s*n_ff + r]      = silu(gE[ids[s]][r]·x) * (uE[ids[s]][r]·x)   (IQ3_XXS)
//   slot s == n_used: h[n_used*n_ff + r] = silu(gS[r]·x) * (uS[r]·x)                    (Q8_0)
// One SIMDGROUP per (slot, row) output; a simdgroup sees exactly one slot,
// so the type divergence is uniform. NSG simdgroups per threadgroup; grid z
// batches queries.

#include "iq_tables.metal"
#include "moe_common.metal"

constant uint NSG [[function_constant(0)]];

// dot of one 32-element IQ4_XS sub-block (superblock blk, group ib) with x.
// Elems [0,16) are the low nibbles of qs[ib*16..+16), [16,32) the high.
// Requires iq_tables.metal (kvalues_iq4nl) included first.
static inline float iq4_group32_dot(device const block_iq4_xs& blk,
                                    uint ib, device const float4* xp) {
    const uint slb = uint(blk.scales_l[ib >> 1u]);
    const uint sh  = uint(blk.scales_h);
    const int  ls  = int(((slb >> (4u * (ib & 1u))) & 0xFu) |
                         (((sh >> (2u * ib)) & 3u) << 4u)) - 32;
    device const packed_uchar4* qp = (device const packed_uchar4*)&blk.qs[ib * 16u];
    float s = 0.0f;
    for (uint j = 0u; j < 4u; ++j) {
        const uchar4 q = uchar4(qp[j]);
        const uint4 lo = uint4(q) & 0xFu;
        const uint4 hi = uint4(q) >> 4u;
        s += dot(float4(float(kvalues_iq4nl[lo.x]), float(kvalues_iq4nl[lo.y]),
                        float(kvalues_iq4nl[lo.z]), float(kvalues_iq4nl[lo.w])), xp[j]);
        s += dot(float4(float(kvalues_iq4nl[hi.x]), float(kvalues_iq4nl[hi.y]),
                        float(kvalues_iq4nl[hi.z]), float(kvalues_iq4nl[hi.w])), xp[4u + j]);
    }
    return float(blk.d) * float(ls) * s;
}

// dot of one 32-element iq3 group (super-block blk, group ib32) with x
static inline float iq3_group32_dot(device const block_iq3_xxs& blk,
                                    uint ib32, device const float4* xp) {
    const uint ao = 64u + 4u * ib32;
    const uint aux = uint(blk.qs[ao]) | (uint(blk.qs[ao + 1u]) << 8u) |
                     (uint(blk.qs[ao + 2u]) << 16u) | (uint(blk.qs[ao + 3u]) << 24u);
    const float db = float(blk.d) * (0.5f + float(aux >> 28u)) * 0.5f;
    device const packed_uchar4* gp = (device const packed_uchar4*)&blk.qs[ib32 * 8u];
    const uchar4 qa = uchar4(gp[0]);
    const uchar4 qb = uchar4(gp[1]);
    float s = 0.0f;
    for (uint l = 0u; l < 4u; ++l) {
        const uint signs = iq_signbyte((aux >> (7u * l)) & 127u);
        const uint g1 = iq3xxs_grid[l < 2u ? (l == 0u ? qa.x : qa.z)
                                           : (l == 2u ? qb.x : qb.z)];
        const uint g2 = iq3xxs_grid[l < 2u ? (l == 0u ? qa.y : qa.w)
                                           : (l == 2u ? qb.y : qb.w)];
        const float4 m1 = float4(uint4(g1, g1 >> 8u, g1 >> 16u, g1 >> 24u) & 255u);
        const float4 m2 = float4(uint4(g2, g2 >> 8u, g2 >> 16u, g2 >> 24u) & 255u);
        const float4 s1 = select(float4(1.0f), float4(-1.0f),
                                 bool4(signs & 1u, signs & 2u, signs & 4u, signs & 8u));
        const float4 s2 = select(float4(1.0f), float4(-1.0f),
                                 bool4(signs & 16u, signs & 32u, signs & 64u, signs & 128u));
        s += dot(m1 * s1, xp[2u * l]) + dot(m2 * s2, xp[2u * l + 1u]);
    }
    return db * s;
}

kernel void moe_gateup_all(device const block_iq3_xxs* gwE [[buffer(0)]],
                           device const block_iq3_xxs* uwE [[buffer(1)]],
                           device const block_q8_0*    gwS [[buffer(2)]],
                           device const block_q8_0*    uwS [[buffer(3)]],
                           device const float*         x   [[buffer(4)]],
                           device const SelT*          sel [[buffer(5)]],
                           device float*               h   [[buffer(6)]],
                           constant MoePC&             pc  [[buffer(7)]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           uint  sgid  [[simdgroup_index_in_threadgroup]],
                           uint  slid  [[thread_index_in_simdgroup]])
{
    const uint out = tgpig.x * NSG + sgid;   // (slot, row) output index
    const uint s   = out / pc.n_ff;
    const uint r   = out % pc.n_ff;
    const uint rq  = tgpig.z;
    if (s > pc.n_used) return;

    const uint xo2 = rq * pc.n_embd;
    const uint ho  = rq * (pc.n_used + 1u) * pc.n_ff;

    float accG = 0.0f, accU = 0.0f;
    if (s < pc.n_used) {                       // routed expert, IQ3_XXS
        const uint kb   = pc.n_embd / 256u;
        const uint nu   = kb * 8u;             // 32-element groups per row
        const uint eid  = min(sel[rq].ids[s], pc.n_expert - 1u);
        const ulong base = ((ulong)eid * pc.n_ff + r) * kb;
        for (uint u = slid; u < nu; u += 32u) {
            const uint b    = u >> 3u;
            const uint ib32 = u & 7u;
            device const float4* xp =
                (device const float4*)(x + xo2 + b * 256u + ib32 * 32u);
            accG += iq3_group32_dot(gwE[base + b], ib32, xp);
            accU += iq3_group32_dot(uwE[base + b], ib32, xp);
        }
    } else {                                   // shared expert, Q8_0
        const uint kb    = pc.n_embd / 32u;
        const ulong base = (ulong)r * kb;
        for (uint b = slid; b < kb; b += 32u) {
            device const float4* xp = (device const float4*)(x + xo2 + b * 32u);
            accG += q8_block_dot(gwS[base + b], xp);
            accU += q8_block_dot(uwS[base + b], xp);
        }
    }

    const float g = simd_sum(accG);
    const float u = simd_sum(accU);
    if (slid == 0u)
        h[ho + s * pc.n_ff + r] = (g / (1.0f + exp(-g))) * u;   // silu(g) * u
}

// IQ4_XS twin (80B routed experts): identical bindings/geometry, only
// the in-kernel dequant differs.
kernel void moe_gateup_all_iq4(device const block_iq4_xs*  gwE [[buffer(0)]],
                           device const block_iq4_xs*  uwE [[buffer(1)]],
                           device const block_q8_0*    gwS [[buffer(2)]],
                           device const block_q8_0*    uwS [[buffer(3)]],
                           device const float*         x   [[buffer(4)]],
                           device const SelT*          sel [[buffer(5)]],
                           device float*               h   [[buffer(6)]],
                           constant MoePC&             pc  [[buffer(7)]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           uint  sgid  [[simdgroup_index_in_threadgroup]],
                           uint  slid  [[thread_index_in_simdgroup]])
{
    const uint out = tgpig.x * NSG + sgid;   // (slot, row) output index
    const uint s   = out / pc.n_ff;
    const uint r   = out % pc.n_ff;
    const uint rq  = tgpig.z;
    if (s > pc.n_used) return;

    const uint xo2 = rq * pc.n_embd;
    const uint ho  = rq * (pc.n_used + 1u) * pc.n_ff;

    float accG = 0.0f, accU = 0.0f;
    if (s < pc.n_used) {                       // routed expert, IQ4_XS
        const uint kb   = pc.n_embd / 256u;
        const uint nu   = kb * 8u;             // 32-element groups per row
        const uint eid  = min(sel[rq].ids[s], pc.n_expert - 1u);
        const ulong base = ((ulong)eid * pc.n_ff + r) * kb;
        for (uint u = slid; u < nu; u += 32u) {
            const uint b    = u >> 3u;
            const uint ib32 = u & 7u;
            device const float4* xp =
                (device const float4*)(x + xo2 + b * 256u + ib32 * 32u);
            accG += iq4_group32_dot(gwE[base + b], ib32, xp);
            accU += iq4_group32_dot(uwE[base + b], ib32, xp);
        }
    } else {                                   // shared expert, Q8_0
        const uint kb    = pc.n_embd / 32u;
        const ulong base = (ulong)r * kb;
        for (uint b = slid; b < kb; b += 32u) {
            device const float4* xp = (device const float4*)(x + xo2 + b * 32u);
            accG += q8_block_dot(gwS[base + b], xp);
            accU += q8_block_dot(uwS[base + b], xp);
        }
    }

    const float g = simd_sum(accG);
    const float u = simd_sum(accU);
    if (slid == 0u)
        h[ho + s * pc.n_ff + r] = (g / (1.0f + exp(-g))) * u;   // silu(g) * u
}
