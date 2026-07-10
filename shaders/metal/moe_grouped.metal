#include <metal_stdlib>
using namespace metal;

// Grouped (prefill) MoE gate+up: instead of every token independently
// reading its 8 experts' rows (n x 6.4 MB/layer of DRAM at n=128), tokens
// are first sorted by expert; then one simdgroup owns one (expert, ffn-row)
// pair and loops the expert's tokens — the row's weights hit DRAM once per
// layer and repeat from cache. Per-row unit order is identical to
// moe_gateup_all, so h is bit-identical to the ungrouped path.

#include "iq_tables.metal"
#include "moe_common.metal"

constant uint NSG [[function_constant(0)]];

// One-threadgroup counting sort of the n*(nUsed) routed assignments over
// 256 experts, plus the shared expert as virtual expert 256 holding every
// token. Outputs: start[258] prefix offsets, aTok/aSlot assignment lists.
struct GroupPC { uint n_embd; uint n_ff; uint n_expert; uint n_used; uint n; };

kernel void moe_group(device const SelT* sel   [[buffer(0)]],
                      device uint*       start [[buffer(1)]],
                      device uint*       aTok  [[buffer(2)]],
                      device uint*       aSlot [[buffer(3)]],
                      constant GroupPC&  pc    [[buffer(4)]],
                      uint3 tid3 [[thread_position_in_threadgroup]])
{
    const uint n = pc.n;
    const uint tid = tid3.x;   // 256 threads
    threadgroup atomic_uint cnt[257];
    threadgroup atomic_uint cur[257];
    for (uint e = tid; e < 257u; e += 256u) {
        atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
        atomic_store_explicit(&cur[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n * pc.n_used; i += 256u) {
        const uint e = min(sel[i / pc.n_used].ids[i % pc.n_used], pc.n_expert - 1u);
        atomic_fetch_add_explicit(&cnt[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint acc = 0;
        for (uint e = 0; e < 257u; ++e) {
            start[e] = acc;
            acc += (e == 256u) ? n : atomic_load_explicit(&cnt[e], memory_order_relaxed);
        }
        start[257] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n * pc.n_used; i += 256u) {
        const uint tok = i / pc.n_used, s = i % pc.n_used;
        const uint e = min(sel[tok].ids[s], pc.n_expert - 1u);
        const uint pos = start[e] + atomic_fetch_add_explicit(&cur[e], 1u, memory_order_relaxed);
        aTok[pos] = tok;
        aSlot[pos] = s;
    }
    for (uint i = tid; i < n; i += 256u) {          // shared expert: all tokens
        aTok[start[256] + i] = i;
        aSlot[start[256] + i] = pc.n_used;
    }
}

// dot of one 32-element iq3 group (same math/order as moe_gateup_all)
static inline float iq3_g32(device const block_iq3_xxs& blk, uint ib32,
                            device const float4* xp) {
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

kernel void moe_gu_grouped(device const block_iq3_xxs* gwE   [[buffer(0)]],
                           device const block_iq3_xxs* uwE   [[buffer(1)]],
                           device const block_q8_0*    gwS   [[buffer(2)]],
                           device const block_q8_0*    uwS   [[buffer(3)]],
                           device const float*         x     [[buffer(4)]],
                           device const uint*          start [[buffer(5)]],
                           device const uint*          aTok  [[buffer(6)]],
                           device const uint*          aSlot [[buffer(7)]],
                           device float*               h     [[buffer(8)]],
                           constant MoePC&             pc    [[buffer(9)]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           uint  sgid  [[simdgroup_index_in_threadgroup]],
                           uint  slid  [[thread_index_in_simdgroup]])
{
    const uint gid = tgpig.x * NSG + sgid;   // (expert, row) pair
    const uint e = gid / pc.n_ff;
    const uint r = gid % pc.n_ff;
    if (e > pc.n_expert) return;             // 0..255 routed, 256 shared
    const uint s0 = start[e], c = start[e + 1] - s0;
    if (c == 0u) return;

    const uint kb = pc.n_embd / 256u;
    const uint hs = (pc.n_used + 1u) * pc.n_ff;

    if (e < pc.n_expert) {                   // routed, IQ3_XXS
        const ulong base = ((ulong)e * pc.n_ff + r) * kb;
        for (uint t = 0u; t < c; ++t) {
            const uint tok = aTok[s0 + t];
            device const float4* xr = (device const float4*)(x + tok * pc.n_embd);
            float accG = 0.0f, accU = 0.0f;
            for (uint u = slid; u < kb * 8u; u += 32u) {
                const uint b = u >> 3u, ib32 = u & 7u;
                device const float4* xp = xr + (b * 256u + ib32 * 32u) / 4u;
                accG += iq3_g32(gwE[base + b], ib32, xp);
                accU += iq3_g32(uwE[base + b], ib32, xp);
            }
            const float g = simd_sum(accG);
            const float uu = simd_sum(accU);
            if (slid == 0u)
                h[tok * hs + aSlot[s0 + t] * pc.n_ff + r] = (g / (1.0f + exp(-g))) * uu;
        }
    } else {                                 // shared expert, Q8_0
        const ulong base = (ulong)r * (pc.n_embd / 32u);
        for (uint t = 0u; t < c; ++t) {
            const uint tok = aTok[s0 + t];
            device const float4* xr = (device const float4*)(x + tok * pc.n_embd);
            float accG = 0.0f, accU = 0.0f;
            for (uint b = slid; b < pc.n_embd / 32u; b += 32u) {
                device const float4* xp = xr + b * 8u;
                accG += q8_block_dot(gwS[base + b], xp);
                accU += q8_block_dot(uwS[base + b], xp);
            }
            const float g = simd_sum(accG);
            const float uu = simd_sum(accU);
            if (slid == 0u)
                h[tok * hs + pc.n_used * pc.n_ff + r] = (g / (1.0f + exp(-g))) * uu;
        }
    }
}
