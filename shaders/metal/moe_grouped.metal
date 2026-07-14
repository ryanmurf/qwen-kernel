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
// n_expert experts, plus the shared expert as virtual expert n_expert holding
// every token. Outputs: start[n_expert+2] prefix offsets, aTok/aSlot lists.
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
    threadgroup atomic_uint cnt[513];        // n_expert (<=512) + shared
    threadgroup atomic_uint cur[513];
    const uint ne1 = pc.n_expert + 1u;
    for (uint e = tid; e < ne1; e += 256u) {
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
        for (uint e = 0; e < ne1; ++e) {
            start[e] = acc;
            acc += (e == pc.n_expert) ? n : atomic_load_explicit(&cnt[e], memory_order_relaxed);
        }
        start[ne1] = acc;
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
        aTok[start[pc.n_expert] + i] = i;
        aSlot[start[pc.n_expert] + i] = pc.n_used;
    }
}

// Prefill companion for packed v5: in addition to the stable expert-sorted
// assignment arrays, emit only the non-empty (expert, 32-assignment tile)
// pairs. Consumers launch a fixed, host-computable upper bound and return on
// sentinels. For routed counts c[e], sum ceil(c[e]/32) is bounded by
// n_expert + ceil(n*n_used/32); the shared expert adds ceil(n/32).
kernel void moe_group_work(device const SelT* sel   [[buffer(0)]],
                           device uint*       start [[buffer(1)]],
                           device uint*       aTok  [[buffer(2)]],
                           device uint*       aSlot [[buffer(3)]],
                           device uint*       work  [[buffer(4)]],
                           constant GroupPC&  pc    [[buffer(5)]],
                           uint3 tid3 [[thread_position_in_threadgroup]])
{
    const uint n = pc.n;
    const uint tid = tid3.x;
    threadgroup atomic_uint cnt[513];
    threadgroup atomic_uint cur[513];
    const uint ne1 = pc.n_expert + 1u;
    for (uint e = tid; e < ne1; e += 256u) {
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
        uint acc = 0u, nw = 0u;
        for (uint e = 0u; e < pc.n_expert; ++e) {
            start[e] = acc;
            const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
            for (uint t0 = 0u; t0 < c; t0 += 32u) {
                work[2u * nw] = e;
                work[2u * nw + 1u] = t0;
                ++nw;
            }
            acc += c;
        }
        start[pc.n_expert] = acc;
        for (uint t0 = 0u; t0 < n; t0 += 32u) {
            work[2u * nw] = pc.n_expert;
            work[2u * nw + 1u] = t0;
            ++nw;
        }
        acc += n;
        start[pc.n_expert + 1u] = acc;
        const uint cap = pc.n_expert + (n * pc.n_used + 31u) / 32u +
                         (n + 31u) / 32u;
        while (nw < cap) {
            work[2u * nw] = pc.n_expert + 1u;
            work[2u * nw + 1u] = 0u;
            ++nw;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n * pc.n_used; i += 256u) {
        const uint tok = i / pc.n_used, s = i % pc.n_used;
        const uint e = min(sel[tok].ids[s], pc.n_expert - 1u);
        const uint pos = start[e] +
            atomic_fetch_add_explicit(&cur[e], 1u, memory_order_relaxed);
        aTok[pos] = tok;
        aSlot[pos] = s;
    }
    for (uint i = tid; i < n; i += 256u) {
        aTok[start[pc.n_expert] + i] = i;
        aSlot[start[pc.n_expert] + i] = pc.n_used;
    }
}

// Decode companion: also emits a sorted compact live-expert list. The list is
// padded with sentinel n_expert+1 through its fixed n*n_used+1 capacity, so
// compact consumers can use a host-known upper-bound grid without an indirect
// dispatch or CPU synchronization.
kernel void moe_group_live(device const SelT* sel   [[buffer(0)]],
                           device uint*       start [[buffer(1)]],
                           device uint*       aTok  [[buffer(2)]],
                           device uint*       aSlot [[buffer(3)]],
                           device uint*       live  [[buffer(4)]],
                           constant GroupPC&  pc    [[buffer(5)]],
                           uint3 tid3 [[thread_position_in_threadgroup]])
{
    const uint n = pc.n;
    const uint tid = tid3.x;
    threadgroup atomic_uint cnt[513];
    threadgroup atomic_uint cur[513];
    const uint ne1 = pc.n_expert + 1u;
    for (uint e = tid; e < ne1; e += 256u) {
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
        uint acc = 0u, nl = 0u;
        for (uint e = 0u; e < pc.n_expert; ++e) {
            start[e] = acc;
            const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
            if (c) live[nl++] = e;
            acc += c;
        }
        start[pc.n_expert] = acc;
        live[nl++] = pc.n_expert;              // shared expert is always live
        acc += n;
        start[pc.n_expert + 1u] = acc;
        const uint cap = n * pc.n_used + 1u;
        while (nl < cap) live[nl++] = pc.n_expert + 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n * pc.n_used; i += 256u) {
        const uint tok = i / pc.n_used, s = i % pc.n_used;
        const uint e = min(sel[tok].ids[s], pc.n_expert - 1u);
        const uint pos = start[e] +
            atomic_fetch_add_explicit(&cur[e], 1u, memory_order_relaxed);
        aTok[pos] = tok;
        aSlot[pos] = s;
    }
    for (uint i = tid; i < n; i += 256u) {
        aTok[start[pc.n_expert] + i] = i;
        aSlot[start[pc.n_expert] + i] = pc.n_used;
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

// v1 (read-once): re-decodes the row per token — DRAM-locality only, and
// MoE is ALU-bound on the iq3 decode, so this measured SLOWER (PORT.md).
// Kept as the bit-exact grouped reference (QK_MOE_GROUPED=1).
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

// v2 (decode-ONCE, QK_MOE_GROUPED=2): one threadgroup per (expert, 32-row
// tile). Each K-chunk of the tile's gate AND up rows is dequantized to
// threadgroup memory once per token-pass (ceil(c/8) passes, c = tokens
// routed to the expert; avg 4 at n=128 vs 8 re-decodes ungrouped) and
// multiplied against up to 8 gathered token columns via simdgroup MMA —
// this cuts the iq3 decode ALU, which is what MoE is actually bound on,
// ~4x. f32 fragments, NOT half: an f16 variant scored 33/36 on prefillcmp
// (the shipping h-GEMM already spends the argmax margin at rel 0.073);
// f32 staging is summation-order-only noise (~1e-6 class) and MMA rate
// doesn't matter here — decode dominates.
constant uint GBM = 32u;   // weight rows per tile (per matrix)
constant uint GBK = 64u;   // K elems staged per chunk (2 iq3 groups / 2 q8 blocks)
constant uint GSK = 65u;   // padded float stride

kernel void moe_gu_grouped2(device const block_iq3_xxs* gwE   [[buffer(0)]],
                            device const block_iq3_xxs* uwE   [[buffer(1)]],
                            device const block_q8_0*    gwS   [[buffer(2)]],
                            device const block_q8_0*    uwS   [[buffer(3)]],
                            device const float*         x     [[buffer(4)]],
                            device const uint*          start [[buffer(5)]],
                            device const uint*          aTok  [[buffer(6)]],
                            device const uint*          aSlot [[buffer(7)]],
                            device float*               h     [[buffer(8)]],
                            constant MoePC&             pc    [[buffer(9)]],
                            uint3 tid3  [[thread_position_in_threadgroup]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint  sgid  [[simdgroup_index_in_threadgroup]],
                            uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;                 // 0..127, 4 simdgroups
    const uint nrt = pc.n_ff / GBM;          // row tiles per expert
    const uint e  = tgpig.x / nrt;           // 0..255 routed, 256 shared
    const uint rt = tgpig.x % nrt;
    const uint s0 = start[e], c = start[e + 1u] - s0;
    if (c == 0u) return;

    const uint K = pc.n_embd;
    const uint hs = (pc.n_used + 1u) * pc.n_ff;
    const uint row0 = rt * GBM;

    threadgroup float WgSh[GBM * GSK];
    threadgroup float WuSh[GBM * GSK];
    threadgroup float Xsh[8u * GSK];
    threadgroup float outb[4u * 144u];       // per-simd G(72)|U(72) bounce

    // staging assignment: one (matrix, row, 32-group) per thread per chunk
    const uint smat = tid >> 6;              // 0 = gate, 1 = up
    const uint srow = (tid >> 1) & 31u;
    const uint sgrp = tid & 1u;
    threadgroup float* dst = (smat ? WuSh : WgSh) + srow * GSK + sgrp * 32u;

    for (uint t0 = 0u; t0 < c; t0 += 8u) {
        simdgroup_float8x8 accG(0.0f), accU(0.0f);

        for (uint k0 = 0u; k0 < K; k0 += GBK) {
            const uint g32 = (k0 >> 5) + sgrp;           // global 32-group
            const uint row = row0 + srow;
            if (e < pc.n_expert) {                       // routed: IQ3_XXS
                device const block_iq3_xxs* mat = smat ? uwE : gwE;
                device const block_iq3_xxs& blk =
                    mat[((ulong)e * pc.n_ff + row) * (K >> 8) + (g32 >> 3)];
                const uint ib32 = g32 & 7u;
                const uint ao = 64u + 4u * ib32;
                const uint aux = uint(blk.qs[ao]) | (uint(blk.qs[ao + 1u]) << 8u) |
                                 (uint(blk.qs[ao + 2u]) << 16u) | (uint(blk.qs[ao + 3u]) << 24u);
                const float db = float(blk.d) * (0.5f + float(aux >> 28u)) * 0.5f;
                for (uint l = 0u; l < 4u; ++l) {
                    const uint signs = iq_signbyte((aux >> (7u * l)) & 127u);
                    const uint g1 = iq3xxs_grid[blk.qs[ib32 * 8u + 2u * l]];
                    const uint g2 = iq3xxs_grid[blk.qs[ib32 * 8u + 2u * l + 1u]];
                    for (uint j = 0u; j < 4u; ++j) {
                        dst[l * 8u + j] = db * float((g1 >> (8u * j)) & 255u) *
                                          (((signs >> j) & 1u) ? -1.0f : 1.0f);
                        dst[l * 8u + 4u + j] = db * float((g2 >> (8u * j)) & 255u) *
                                               (((signs >> (4u + j)) & 1u) ? -1.0f : 1.0f);
                    }
                }
            } else {                                     // shared: Q8_0
                device const block_q8_0* mat = smat ? uwS : gwS;
                device const block_q8_0& blk = mat[(ulong)row * (K >> 5) + g32];
                const float d = float(blk.d);
                device const packed_char4* qp = (device const packed_char4*)blk.qs;
                for (uint i = 0u; i < 8u; ++i) {
                    const char4 q = char4(qp[i]);
                    dst[4u * i + 0u] = d * float(q.x);
                    dst[4u * i + 1u] = d * float(q.y);
                    dst[4u * i + 2u] = d * float(q.z);
                    dst[4u * i + 3u] = d * float(q.w);
                }
            }
            for (uint idx = tid; idx < 8u * GBK; idx += 128u) {
                const uint tt = idx >> 6, kk = idx & 63u;
                const uint ti = t0 + tt;
                Xsh[tt * GSK + kk] = ti < c
                    ? x[(ulong)aTok[s0 + ti] * K + k0 + kk] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kf = 0u; kf < GBK / 8u; ++kf) {
                simdgroup_float8x8 bfr;
                simdgroup_load(bfr, &Xsh[kf * 8u], GSK, ulong2(0, 0), true);
                simdgroup_float8x8 ag, au;
                simdgroup_load(ag, &WgSh[(sgid * 8u) * GSK + kf * 8u], GSK);
                simdgroup_load(au, &WuSh[(sgid * 8u) * GSK + kf * 8u], GSK);
                simdgroup_multiply_accumulate(accG, ag, bfr, accG);
                simdgroup_multiply_accumulate(accU, au, bfr, accU);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float* bufG = &outb[sgid * 144u];
        threadgroup float* bufU = bufG + 72u;
        simdgroup_store(accG, bufG, 9u);
        simdgroup_store(accU, bufU, 9u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
        for (uint jj = 0u; jj < 2u; ++jj) {
            const uint ti = t0 + fj0 + jj;
            if (ti < c) {
                const uint i = s0 + ti;
                const float g = bufG[fi * 9u + fj0 + jj];
                const float u = bufU[fi * 9u + fj0 + jj];
                h[(ulong)aTok[i] * hs + aSlot[i] * pc.n_ff + row0 + sgid * 8u + fi] =
                    (g / (1.0f + exp(-g))) * u;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// v3 (decode-once, WIDE, QK_MOE_GROUPED=3): the perf shape — f16 staging
// buys a 64-row x 32-token-column tile in 25.7 KB, so one pass covers c<=32
// tokens with half the barriers and 4x fewer passes than v2. Precision =
// gemm_q8_0_h / llama.cpp mul_mm_id class (half operands, f32 accumulate):
// semantic correctness is proven by v2 (36/36 with scalar projections);
// v3 shares v2's indexing and is gated by moegcmp isolation + the
// natural-prompt serving parity suite.
constant uint G3M = 64u;   // weight rows per tile (per matrix)
constant uint G3N = 32u;   // token columns per pass
constant uint G3K = 64u;   // K elems staged per chunk
constant uint G3S = 68u;   // padded half stride (8B-aligned rows for half4 stores)

// v4 (decode-once, wide, f32, QK_MOE_GROUPED=4): v3's 32-token-column shape
// with v2's f32 exactness — 32x32 tile in 25 KB. Captures most of the
// grouping win while keeping the MoE contribution at summation-order noise,
// so the serve-test/llama parity gates stay green (total noise = h-GEMM
// class, unchanged from the shipping default).
constant uint G4M = 32u;
constant uint G4N = 32u;
constant uint G4K = 64u;
constant uint G4S = 68u;

template <bool LIVE>
static inline void moe_gu_grouped4_body(device const block_iq3_xxs* gwE,
                                        device const block_iq3_xxs* uwE,
                                        device const block_q8_0* gwS,
                                        device const block_q8_0* uwS,
                                        device const float* x,
                                        device const uint* start,
                                        device const uint* aTok,
                                        device const uint* aSlot,
                                        device float* h,
                                        device const uint* live,
                                        constant MoePC& pc,
                                        threadgroup float* WgSh,
                                        threadgroup float* WuSh,
                                        threadgroup float* Xsh,
                                        threadgroup float* outb,
                                        uint3 tid3, uint3 tgpig,
                                        uint sgid, uint slid)
{
    const uint tid = tid3.x;                 // 0..127, 4 simdgroups
    const uint nrt = pc.n_ff / G4M;          // row tiles per expert
    const uint ei = tgpig.x / nrt;
    const uint e  = LIVE ? live[ei] : ei;
    const uint rt = tgpig.x % nrt;
    if (e > pc.n_expert) return;
    const uint s0 = start[e], c = start[e + 1u] - s0;
    if (c == 0u) return;

    const uint K = pc.n_embd;
    const uint hs = (pc.n_used + 1u) * pc.n_ff;
    const uint row0 = rt * G4M;

    const uint smat = tid >> 6;              // 0 = gate, 1 = up
    const uint srow = (tid >> 1) & 31u;
    const uint sgrp = tid & 1u;
    threadgroup float* dst = (smat ? WuSh : WgSh) + srow * G4S + sgrp * 32u;

    for (uint t0 = 0u; t0 < c; t0 += G4N) {
        simdgroup_float8x8 accG[4], accU[4];
        for (uint i = 0u; i < 4u; ++i) {
            accG[i] = simdgroup_float8x8(0.0f);
            accU[i] = simdgroup_float8x8(0.0f);
        }

        for (uint k0 = 0u; k0 < K; k0 += G4K) {
            const uint g32 = (k0 >> 5) + sgrp;           // global 32-group
            const uint row = row0 + srow;
            if (e < pc.n_expert) {                       // routed: IQ3_XXS
                device const block_iq3_xxs* mat = smat ? uwE : gwE;
                device const block_iq3_xxs& blk =
                    mat[((ulong)e * pc.n_ff + row) * (K >> 8) + (g32 >> 3)];
                const uint ib32 = g32 & 7u;
                const uint ao = 64u + 4u * ib32;
                const uint aux = uint(blk.qs[ao]) | (uint(blk.qs[ao + 1u]) << 8u) |
                                 (uint(blk.qs[ao + 2u]) << 16u) | (uint(blk.qs[ao + 3u]) << 24u);
                const float db = float(blk.d) * (0.5f + float(aux >> 28u)) * 0.5f;
                threadgroup float4* dst4 = (threadgroup float4*)dst;
                for (uint l = 0u; l < 4u; ++l) {
                    const uint signs = iq_signbyte((aux >> (7u * l)) & 127u);
                    const uint g1 = iq3xxs_grid[blk.qs[ib32 * 8u + 2u * l]];
                    const uint g2 = iq3xxs_grid[blk.qs[ib32 * 8u + 2u * l + 1u]];
                    const float4 m1 = float4(uint4(g1, g1 >> 8u, g1 >> 16u, g1 >> 24u) & 255u);
                    const float4 m2 = float4(uint4(g2, g2 >> 8u, g2 >> 16u, g2 >> 24u) & 255u);
                    const float4 s1 = select(float4(1.0f), float4(-1.0f),
                                             bool4(signs & 1u, signs & 2u, signs & 4u, signs & 8u));
                    const float4 s2 = select(float4(1.0f), float4(-1.0f),
                                             bool4(signs & 16u, signs & 32u, signs & 64u, signs & 128u));
                    dst4[2u * l]      = db * m1 * s1;
                    dst4[2u * l + 1u] = db * m2 * s2;
                }
            } else {                                     // shared: Q8_0
                device const block_q8_0* mat = smat ? uwS : gwS;
                device const block_q8_0& blk = mat[(ulong)row * (K >> 5) + g32];
                const float d = float(blk.d);
                device const packed_char4* qp = (device const packed_char4*)blk.qs;
                threadgroup float4* dst4 = (threadgroup float4*)dst;
                for (uint i = 0u; i < 8u; ++i)
                    dst4[i] = d * float4(char4(qp[i]));
            }
            for (uint idx = tid * 4u; idx < G4N * G4K; idx += 128u * 4u) {
                const uint tt = idx >> 6, kk = idx & 63u;
                const uint ti = t0 + tt;
                ((threadgroup float4*)&Xsh[tt * G4S + kk])[0] = ti < c
                    ? *(device const packed_float4*)(x + (ulong)aTok[s0 + ti] * K + k0 + kk)
                    : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kf = 0u; kf < G4K / 8u; ++kf) {
                simdgroup_float8x8 ag, au;
                simdgroup_load(ag, &WgSh[(sgid * 8u) * G4S + kf * 8u], G4S);
                simdgroup_load(au, &WuSh[(sgid * 8u) * G4S + kf * 8u], G4S);
                for (uint nc = 0u; nc < 4u; ++nc) {
                    simdgroup_float8x8 bfr;
                    simdgroup_load(bfr, &Xsh[(nc * 8u) * G4S + kf * 8u], G4S, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(accG[nc], ag, bfr, accG[nc]);
                    simdgroup_multiply_accumulate(accU[nc], au, bfr, accU[nc]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float* bufG = &outb[sgid * 144u];
        threadgroup float* bufU = bufG + 72u;
        const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
        for (uint nc = 0u; nc < 4u; ++nc) {
            simdgroup_store(accG[nc], bufG, 9u);
            simdgroup_store(accU[nc], bufU, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint ti = t0 + nc * 8u + fj0 + jj;
                if (ti < c) {
                    const uint i = s0 + ti;
                    const float g = bufG[fi * 9u + fj0 + jj];
                    const float u = bufU[fi * 9u + fj0 + jj];
                    h[(ulong)aTok[i] * hs + aSlot[i] * pc.n_ff + row0 + sgid * 8u + fi] =
                        (g / (1.0f + exp(-g))) * u;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void moe_gu_grouped4(device const block_iq3_xxs* gwE [[buffer(0)]],
                            device const block_iq3_xxs* uwE [[buffer(1)]],
                            device const block_q8_0* gwS [[buffer(2)]],
                            device const block_q8_0* uwS [[buffer(3)]],
                            device const float* x [[buffer(4)]],
                            device const uint* start [[buffer(5)]],
                            device const uint* aTok [[buffer(6)]],
                            device const uint* aSlot [[buffer(7)]],
                            device float* h [[buffer(8)]],
                            constant MoePC& pc [[buffer(9)]],
                            uint3 tid3 [[thread_position_in_threadgroup]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint sgid [[simdgroup_index_in_threadgroup]],
                            uint slid [[thread_index_in_simdgroup]])
{
    threadgroup float WgSh[G4M * G4S];
    threadgroup float WuSh[G4M * G4S];
    threadgroup float Xsh[G4N * G4S];
    threadgroup float outb[4u * 144u];
    moe_gu_grouped4_body<false>(gwE, uwE, gwS, uwS, x, start, aTok, aSlot, h,
        start, pc, WgSh, WuSh, Xsh, outb, tid3, tgpig, sgid, slid);
}

kernel void moe_gu_grouped4_live(device const block_iq3_xxs* gwE [[buffer(0)]],
                                 device const block_iq3_xxs* uwE [[buffer(1)]],
                                 device const block_q8_0* gwS [[buffer(2)]],
                                 device const block_q8_0* uwS [[buffer(3)]],
                                 device const float* x [[buffer(4)]],
                                 device const uint* start [[buffer(5)]],
                                 device const uint* aTok [[buffer(6)]],
                                 device const uint* aSlot [[buffer(7)]],
                                 device float* h [[buffer(8)]],
                                 device const uint* live [[buffer(9)]],
                                 constant MoePC& pc [[buffer(10)]],
                                 uint3 tid3 [[thread_position_in_threadgroup]],
                                 uint3 tgpig [[threadgroup_position_in_grid]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint slid [[thread_index_in_simdgroup]])
{
    threadgroup float WgSh[G4M * G4S];
    threadgroup float WuSh[G4M * G4S];
    threadgroup float Xsh[G4N * G4S];
    threadgroup float outb[4u * 144u];
    moe_gu_grouped4_body<true>(gwE, uwE, gwS, uwS, x, start, aTok, aSlot, h,
        live, pc, WgSh, WuSh, Xsh, outb, tid3, tgpig, sgid, slid);
}

kernel void moe_gu_grouped3(device const block_iq3_xxs* gwE   [[buffer(0)]],
                            device const block_iq3_xxs* uwE   [[buffer(1)]],
                            device const block_q8_0*    gwS   [[buffer(2)]],
                            device const block_q8_0*    uwS   [[buffer(3)]],
                            device const float*         x     [[buffer(4)]],
                            device const uint*          start [[buffer(5)]],
                            device const uint*          aTok  [[buffer(6)]],
                            device const uint*          aSlot [[buffer(7)]],
                            device float*               h     [[buffer(8)]],
                            constant MoePC&             pc    [[buffer(9)]],
                            uint3 tid3  [[thread_position_in_threadgroup]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint  sgid  [[simdgroup_index_in_threadgroup]],
                            uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;                 // 0..255, 8 simdgroups
    const uint nrt = pc.n_ff / G3M;          // row tiles per expert
    const uint e  = tgpig.x / nrt;           // 0..255 routed, 256 shared
    const uint rt = tgpig.x % nrt;
    const uint s0 = start[e], c = start[e + 1u] - s0;
    if (c == 0u) return;

    const uint K = pc.n_embd;
    const uint hs = (pc.n_used + 1u) * pc.n_ff;
    const uint row0 = rt * G3M;

    threadgroup half  WgSh[G3M * G3S];
    threadgroup half  WuSh[G3M * G3S];
    threadgroup half  Xsh[G3N * G3S];
    threadgroup float outb[8u * 144u];       // per-simd G(72)|U(72) bounce

    // staging assignment: one (matrix, row, 32-group) per thread per chunk
    const uint smat = tid >> 7;              // 0 = gate, 1 = up
    const uint srow = (tid >> 1) & 63u;
    const uint sgrp = tid & 1u;
    threadgroup half* dst = (smat ? WuSh : WgSh) + srow * G3S + sgrp * 32u;

    for (uint t0 = 0u; t0 < c; t0 += G3N) {
        simdgroup_float8x8 accG[4], accU[4];
        for (uint i = 0u; i < 4u; ++i) {
            accG[i] = simdgroup_float8x8(0.0f);
            accU[i] = simdgroup_float8x8(0.0f);
        }

        for (uint k0 = 0u; k0 < K; k0 += G3K) {
            const uint g32 = (k0 >> 5) + sgrp;           // global 32-group
            const uint row = row0 + srow;
            if (e < pc.n_expert) {                       // routed: IQ3_XXS
                device const block_iq3_xxs* mat = smat ? uwE : gwE;
                device const block_iq3_xxs& blk =
                    mat[((ulong)e * pc.n_ff + row) * (K >> 8) + (g32 >> 3)];
                const uint ib32 = g32 & 7u;
                const uint ao = 64u + 4u * ib32;
                const uint aux = uint(blk.qs[ao]) | (uint(blk.qs[ao + 1u]) << 8u) |
                                 (uint(blk.qs[ao + 2u]) << 16u) | (uint(blk.qs[ao + 3u]) << 24u);
                const float db = float(blk.d) * (0.5f + float(aux >> 28u)) * 0.5f;
                threadgroup half4* dst4 = (threadgroup half4*)dst;
                for (uint l = 0u; l < 4u; ++l) {
                    const uint signs = iq_signbyte((aux >> (7u * l)) & 127u);
                    const uint g1 = iq3xxs_grid[blk.qs[ib32 * 8u + 2u * l]];
                    const uint g2 = iq3xxs_grid[blk.qs[ib32 * 8u + 2u * l + 1u]];
                    const float4 m1 = float4(uint4(g1, g1 >> 8u, g1 >> 16u, g1 >> 24u) & 255u);
                    const float4 m2 = float4(uint4(g2, g2 >> 8u, g2 >> 16u, g2 >> 24u) & 255u);
                    const float4 s1 = select(float4(1.0f), float4(-1.0f),
                                             bool4(signs & 1u, signs & 2u, signs & 4u, signs & 8u));
                    const float4 s2 = select(float4(1.0f), float4(-1.0f),
                                             bool4(signs & 16u, signs & 32u, signs & 64u, signs & 128u));
                    dst4[2u * l]      = half4(db * m1 * s1);
                    dst4[2u * l + 1u] = half4(db * m2 * s2);
                }
            } else {                                     // shared: Q8_0
                device const block_q8_0* mat = smat ? uwS : gwS;
                device const block_q8_0& blk = mat[(ulong)row * (K >> 5) + g32];
                const half d = blk.d;
                device const packed_char4* qp = (device const packed_char4*)blk.qs;
                threadgroup half4* dst4 = (threadgroup half4*)dst;
                for (uint i = 0u; i < 8u; ++i)
                    dst4[i] = d * half4(char4(qp[i]));
            }
            for (uint idx = tid * 4u; idx < G3N * G3K; idx += 256u * 4u) {
                const uint tt = idx >> 6, kk = idx & 63u;
                const uint ti = t0 + tt;
                ((threadgroup half4*)&Xsh[tt * G3S + kk])[0] = ti < c
                    ? half4(*(device const packed_float4*)(x + (ulong)aTok[s0 + ti] * K + k0 + kk))
                    : half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kf = 0u; kf < G3K / 8u; ++kf) {
                simdgroup_half8x8 ag, au;
                simdgroup_load(ag, &WgSh[(sgid * 8u) * G3S + kf * 8u], G3S);
                simdgroup_load(au, &WuSh[(sgid * 8u) * G3S + kf * 8u], G3S);
                for (uint nc = 0u; nc < 4u; ++nc) {
                    simdgroup_half8x8 bfr;
                    simdgroup_load(bfr, &Xsh[(nc * 8u) * G3S + kf * 8u], G3S, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(accG[nc], ag, bfr, accG[nc]);
                    simdgroup_multiply_accumulate(accU[nc], au, bfr, accU[nc]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float* bufG = &outb[sgid * 144u];
        threadgroup float* bufU = bufG + 72u;
        const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
        for (uint nc = 0u; nc < 4u; ++nc) {
            simdgroup_store(accG[nc], bufG, 9u);
            simdgroup_store(accU[nc], bufU, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint ti = t0 + nc * 8u + fj0 + jj;
                if (ti < c) {
                    const uint i = s0 + ti;
                    const float g = bufG[fi * 9u + fj0 + jj];
                    const float u = bufU[fi * 9u + fj0 + jj];
                    h[(ulong)aTok[i] * hs + aSlot[i] * pc.n_ff + row0 + sgid * 8u + fi] =
                        (g / (1.0f + exp(-g))) * u;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// v5 (QK_MOE_GROUPED=5): packed 8x8-block shared tiles (llama mul_mm-class).
// Both MMA operands are staged as contiguous 64-half blocks so every
// simdgroup_load is a stride-8 contiguous read -- no strided or transposed
// tile loads (that stride was worth ~1.45x: llama's mul_mm hits 12.0 TFLOPS
// on this box vs 8.2 for the strided skeleton at identical shape).
//   W tiles: block (rb, kb) at [(rb*8 + kb)*64], elem (r&7)*8 + (k&7)
//   X tile:  block (kb, tb) at [(kb*4 + tb)*64], elem (k&7)*8 + (t&7)
// 32 rows/matrix x 32 tokens, K-chunk 64; f16 staging, f32 accumulators
// (llama-identical numeric class); 14.6 KB threadgroup. Token tiles ride
// grid z (early-return past c) so hot experts fan out instead of looping.
// MAP=0: dense expert grid + z token tile; MAP=1: compact live-expert map;
// MAP=2: compact (expert,t0) work-pair map with no grid-z overlaunch.
template <uint MAP>
static inline void moe_gu_grouped5_body(device const block_iq3_xxs* gwE,
                                        device const block_iq3_xxs* uwE,
                                        device const block_q8_0* gwS,
                                        device const block_q8_0* uwS,
                                        device const float* x,
                                        device const uint* start,
                                        device const uint* aTok,
                                        device const uint* aSlot,
                                        device float* h,
                                        device const uint* map,
                                        constant MoePC& pc,
                                        threadgroup half* WgSh,
                                        threadgroup half* WuSh,
                                        threadgroup half* Xsh,
                                        threadgroup float* outb,
                                        uint3 tid3, uint3 tgpig,
                                        uint sgid, uint slid)
{
    const uint tid = tid3.x;                 // 0..127, 4 simdgroups
    const uint nrt = pc.n_ff / G4M;          // 32-row tiles per expert
    const uint ei = tgpig.x / nrt;
    const uint e  = MAP == 2u ? map[2u * ei] :
                    MAP == 1u ? map[ei] : ei; // 0..255 routed, 256 shared
    const uint rt = tgpig.x % nrt;
    if (e > pc.n_expert) return;              // fixed-grid compact sentinel
    const uint s0 = start[e], c = start[e + 1u] - s0;
    if (c == 0u) return;

    const uint K = pc.n_embd;
    const uint hs = (pc.n_used + 1u) * pc.n_ff;
    const uint row0 = rt * G4M;

    // staging assignment: one (matrix, row, 32-group) per thread per chunk
    const uint smat = tid >> 6;              // 0 = gate, 1 = up
    const uint srow = (tid >> 1) & 31u;
    const uint sgrp = tid & 1u;
    threadgroup half* wsh = (smat ? WuSh : WgSh)
                          + ((srow >> 3) * 8u) * 64u + (srow & 7u) * 8u;

    const uint t0 = MAP == 2u ? map[2u * ei + 1u] : tgpig.z * G4N;
    if (t0 >= c) return;
    {
        simdgroup_float8x8 accG[4], accU[4];
        for (uint i = 0u; i < 4u; ++i) {
            accG[i] = simdgroup_float8x8(0.0f);
            accU[i] = simdgroup_float8x8(0.0f);
        }

        for (uint k0 = 0u; k0 < K; k0 += 64u) {
            const uint g32 = (k0 >> 5) + sgrp;           // global 32-group
            const uint row = row0 + srow;
            if (e < pc.n_expert) {                       // routed: IQ3_XXS
                device const block_iq3_xxs* mat = smat ? uwE : gwE;
                device const block_iq3_xxs& blk =
                    mat[((ulong)e * pc.n_ff + row) * (K >> 8) + (g32 >> 3)];
                const uint ib32 = g32 & 7u;
                const uint ao = 64u + 4u * ib32;
                const uint aux = uint(blk.qs[ao]) | (uint(blk.qs[ao + 1u]) << 8u) |
                                 (uint(blk.qs[ao + 2u]) << 16u) | (uint(blk.qs[ao + 3u]) << 24u);
                const float db = float(blk.d) * (0.5f + float(aux >> 28u)) * 0.5f;
                for (uint l = 0u; l < 4u; ++l) {
                    const uint signs = iq_signbyte((aux >> (7u * l)) & 127u);
                    const uint g1 = iq3xxs_grid[blk.qs[ib32 * 8u + 2u * l]];
                    const uint g2 = iq3xxs_grid[blk.qs[ib32 * 8u + 2u * l + 1u]];
                    const float4 m1 = float4(uint4(g1, g1 >> 8u, g1 >> 16u, g1 >> 24u) & 255u);
                    const float4 m2 = float4(uint4(g2, g2 >> 8u, g2 >> 16u, g2 >> 24u) & 255u);
                    const float4 s1 = select(float4(1.0f), float4(-1.0f),
                                             bool4(signs & 1u, signs & 2u, signs & 4u, signs & 8u));
                    const float4 s2 = select(float4(1.0f), float4(-1.0f),
                                             bool4(signs & 16u, signs & 32u, signs & 64u, signs & 128u));
                    threadgroup half4* dst4 =
                        (threadgroup half4*)(wsh + (sgrp * 4u + l) * 64u);
                    dst4[0] = half4(db * m1 * s1);
                    dst4[1] = half4(db * m2 * s2);
                }
            } else {                                     // shared: Q8_0
                device const block_q8_0* mat = smat ? uwS : gwS;
                device const block_q8_0& blk = mat[(ulong)row * (K >> 5) + g32];
                const half d = blk.d;
                device const packed_char4* qp = (device const packed_char4*)blk.qs;
                for (uint l = 0u; l < 4u; ++l) {
                    threadgroup half4* dst4 =
                        (threadgroup half4*)(wsh + (sgrp * 4u + l) * 64u);
                    dst4[0] = d * half4(char4(qp[2u * l]));
                    dst4[1] = d * half4(char4(qp[2u * l + 1u]));
                }
            }
            // X: 32 tokens x 64 K, thread -> (token, k-block), two passes
            for (uint p = 0u; p < 2u; ++p) {
                const uint slot = tid + p * 128u;
                const uint tt = slot >> 3, kb = slot & 7u;
                const uint ti = t0 + tt;
                threadgroup half* xd =
                    &Xsh[(kb * 4u + (tt >> 3)) * 64u + (tt & 7u)];
                if (ti < c) {
                    device const float* xp = x + (ulong)aTok[s0 + ti] * K + k0 + kb * 8u;
                    const float4 a = *(device const packed_float4*)xp;
                    const float4 b = *(device const packed_float4*)(xp + 4u);
                    xd[0u]  = half(a.x); xd[8u]  = half(a.y);
                    xd[16u] = half(a.z); xd[24u] = half(a.w);
                    xd[32u] = half(b.x); xd[40u] = half(b.y);
                    xd[48u] = half(b.z); xd[56u] = half(b.w);
                } else {
                    for (uint j = 0u; j < 8u; ++j) xd[j * 8u] = 0.0h;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kf = 0u; kf < 8u; ++kf) {
                simdgroup_half8x8 ag, au;
                simdgroup_load(ag, &WgSh[(sgid * 8u + kf) * 64u], 8u);
                simdgroup_load(au, &WuSh[(sgid * 8u + kf) * 64u], 8u);
                simdgroup_barrier(mem_flags::mem_none);
                for (uint nc = 0u; nc < 4u; ++nc) {
                    simdgroup_half8x8 bfr;
                    simdgroup_load(bfr, &Xsh[(kf * 4u + nc) * 64u], 8u);
                    simdgroup_multiply_accumulate(accG[nc], ag, bfr, accG[nc]);
                    simdgroup_multiply_accumulate(accU[nc], au, bfr, accU[nc]);
                }
                simdgroup_barrier(mem_flags::mem_none);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float* bufG = &outb[sgid * 144u];
        threadgroup float* bufU = bufG + 72u;
        const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
        for (uint nc = 0u; nc < 4u; ++nc) {
            simdgroup_store(accG[nc], bufG, 9u);
            simdgroup_store(accU[nc], bufU, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint ti = t0 + nc * 8u + fj0 + jj;
                if (ti < c) {
                    const uint i = s0 + ti;
                    const float g = bufG[fi * 9u + fj0 + jj];
                    const float u = bufU[fi * 9u + fj0 + jj];
                    h[(ulong)aTok[i] * hs + aSlot[i] * pc.n_ff + row0 + sgid * 8u + fi] =
                        (g / (1.0f + exp(-g))) * u;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void moe_gu_grouped5(device const block_iq3_xxs* gwE   [[buffer(0)]],
                            device const block_iq3_xxs* uwE   [[buffer(1)]],
                            device const block_q8_0*    gwS   [[buffer(2)]],
                            device const block_q8_0*    uwS   [[buffer(3)]],
                            device const float*         x     [[buffer(4)]],
                            device const uint*          start [[buffer(5)]],
                            device const uint*          aTok  [[buffer(6)]],
                            device const uint*          aSlot [[buffer(7)]],
                            device float*               h     [[buffer(8)]],
                            constant MoePC&             pc    [[buffer(9)]],
                            uint3 tid3  [[thread_position_in_threadgroup]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint  sgid  [[simdgroup_index_in_threadgroup]],
                            uint  slid  [[thread_index_in_simdgroup]])
{
    threadgroup half WgSh[G4M * 64u];
    threadgroup half WuSh[G4M * 64u];
    threadgroup half Xsh[64u * G4N];
    threadgroup float outb[4u * 144u];
    moe_gu_grouped5_body<0>(gwE, uwE, gwS, uwS, x, start, aTok, aSlot, h,
                                 start, pc, WgSh, WuSh, Xsh, outb,
                                 tid3, tgpig, sgid, slid);
}

kernel void moe_gu_grouped5_live(device const block_iq3_xxs* gwE [[buffer(0)]],
                                 device const block_iq3_xxs* uwE [[buffer(1)]],
                                 device const block_q8_0* gwS [[buffer(2)]],
                                 device const block_q8_0* uwS [[buffer(3)]],
                                 device const float* x [[buffer(4)]],
                                 device const uint* start [[buffer(5)]],
                                 device const uint* aTok [[buffer(6)]],
                                 device const uint* aSlot [[buffer(7)]],
                                 device float* h [[buffer(8)]],
                                 device const uint* live [[buffer(9)]],
                                 constant MoePC& pc [[buffer(10)]],
                                 uint3 tid3 [[thread_position_in_threadgroup]],
                                 uint3 tgpig [[threadgroup_position_in_grid]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint slid [[thread_index_in_simdgroup]])
{
    threadgroup half WgSh[G4M * 64u];
    threadgroup half WuSh[G4M * 64u];
    threadgroup half Xsh[64u * G4N];
    threadgroup float outb[4u * 144u];
    moe_gu_grouped5_body<1>(gwE, uwE, gwS, uwS, x, start, aTok, aSlot, h,
                                live, pc, WgSh, WuSh, Xsh, outb,
                                tid3, tgpig, sgid, slid);
}

kernel void moe_gu_grouped5_work(device const block_iq3_xxs* gwE [[buffer(0)]],
                                 device const block_iq3_xxs* uwE [[buffer(1)]],
                                 device const block_q8_0* gwS [[buffer(2)]],
                                 device const block_q8_0* uwS [[buffer(3)]],
                                 device const float* x [[buffer(4)]],
                                 device const uint* start [[buffer(5)]],
                                 device const uint* aTok [[buffer(6)]],
                                 device const uint* aSlot [[buffer(7)]],
                                 device float* h [[buffer(8)]],
                                 device const uint* work [[buffer(9)]],
                                 constant MoePC& pc [[buffer(10)]],
                                 uint3 tid3 [[thread_position_in_threadgroup]],
                                 uint3 tgpig [[threadgroup_position_in_grid]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint slid [[thread_index_in_simdgroup]])
{
    threadgroup half WgSh[G4M * 64u];
    threadgroup half WuSh[G4M * 64u];
    threadgroup half Xsh[64u * G4N];
    threadgroup float outb[4u * 144u];
    moe_gu_grouped5_body<2>(gwE, uwE, gwS, uwS, x, start, aTok, aSlot, h,
                                work, pc, WgSh, WuSh, Xsh, outb,
                                tid3, tgpig, sgid, slid);
}

// ---- IQ4_XS routed gate/up twins (80B) ----

kernel void moe_gu_grouped4_iq4(device const block_iq4_xs*  gwE   [[buffer(0)]],
                            device const block_iq4_xs*  uwE   [[buffer(1)]],
                            device const block_q8_0*    gwS   [[buffer(2)]],
                            device const block_q8_0*    uwS   [[buffer(3)]],
                            device const float*         x     [[buffer(4)]],
                            device const uint*          start [[buffer(5)]],
                            device const uint*          aTok  [[buffer(6)]],
                            device const uint*          aSlot [[buffer(7)]],
                            device float*               h     [[buffer(8)]],
                            constant MoePC&             pc    [[buffer(9)]],
                            uint3 tid3  [[thread_position_in_threadgroup]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint  sgid  [[simdgroup_index_in_threadgroup]],
                            uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;                 // 0..127, 4 simdgroups
    const uint nrt = pc.n_ff / G4M;          // row tiles per expert
    const uint e  = tgpig.x / nrt;           // 0..255 routed, 256 shared
    const uint rt = tgpig.x % nrt;
    const uint s0 = start[e], c = start[e + 1u] - s0;
    if (c == 0u) return;

    const uint K = pc.n_embd;
    const uint hs = (pc.n_used + 1u) * pc.n_ff;
    const uint row0 = rt * G4M;

    threadgroup float WgSh[G4M * G4S];
    threadgroup float WuSh[G4M * G4S];
    threadgroup float Xsh[G4N * G4S];
    threadgroup float outb[4u * 144u];       // per-simd G(72)|U(72) bounce

    const uint smat = tid >> 6;              // 0 = gate, 1 = up
    const uint srow = (tid >> 1) & 31u;
    const uint sgrp = tid & 1u;
    threadgroup float* dst = (smat ? WuSh : WgSh) + srow * G4S + sgrp * 32u;

    for (uint t0 = 0u; t0 < c; t0 += G4N) {
        simdgroup_float8x8 accG[4], accU[4];
        for (uint i = 0u; i < 4u; ++i) {
            accG[i] = simdgroup_float8x8(0.0f);
            accU[i] = simdgroup_float8x8(0.0f);
        }

        for (uint k0 = 0u; k0 < K; k0 += G4K) {
            const uint g32 = (k0 >> 5) + sgrp;           // global 32-group
            const uint row = row0 + srow;
            if (e < pc.n_expert) {                       // routed: IQ4_XS
                device const block_iq4_xs* mat = smat ? uwE : gwE;
                device const block_iq4_xs& blk =
                    mat[((ulong)e * pc.n_ff + row) * (K >> 8) + (g32 >> 3)];
                const uint ib = g32 & 7u;
                const uint slb = uint(blk.scales_l[ib >> 1u]);
                const uint shh = uint(blk.scales_h);
                const int  ls  = int(((slb >> (4u * (ib & 1u))) & 0xFu) |
                                     (((shh >> (2u * ib)) & 3u) << 4u)) - 32;
                const float dl = float(blk.d) * float(ls);
                device const packed_uchar4* qp =
                    (device const packed_uchar4*)&blk.qs[ib * 16u];
                for (uint l = 0u; l < 4u; ++l) {
                    const uchar4 qa = uchar4(qp[(l & 1u) * 2u]);
                    const uchar4 qb = uchar4(qp[(l & 1u) * 2u + 1u]);
                    const uint4 na = (l < 2u) ? (uint4(qa) & 0xFu) : (uint4(qa) >> 4u);
                    const uint4 nb = (l < 2u) ? (uint4(qb) & 0xFu) : (uint4(qb) >> 4u);
                    threadgroup float4* dst4 = ((threadgroup float4*)dst) + 2u * l;
                    dst4[0] = float4(dl * float4(float(kvalues_iq4nl[na.x]), float(kvalues_iq4nl[na.y]),
                                                 float(kvalues_iq4nl[na.z]), float(kvalues_iq4nl[na.w])));
                    dst4[1] = float4(dl * float4(float(kvalues_iq4nl[nb.x]), float(kvalues_iq4nl[nb.y]),
                                                 float(kvalues_iq4nl[nb.z]), float(kvalues_iq4nl[nb.w])));
                }
            } else {                                     // shared: Q8_0
                device const block_q8_0* mat = smat ? uwS : gwS;
                device const block_q8_0& blk = mat[(ulong)row * (K >> 5) + g32];
                const float d = float(blk.d);
                device const packed_char4* qp = (device const packed_char4*)blk.qs;
                threadgroup float4* dst4 = (threadgroup float4*)dst;
                for (uint i = 0u; i < 8u; ++i)
                    dst4[i] = d * float4(char4(qp[i]));
            }
            for (uint idx = tid * 4u; idx < G4N * G4K; idx += 128u * 4u) {
                const uint tt = idx >> 6, kk = idx & 63u;
                const uint ti = t0 + tt;
                ((threadgroup float4*)&Xsh[tt * G4S + kk])[0] = ti < c
                    ? *(device const packed_float4*)(x + (ulong)aTok[s0 + ti] * K + k0 + kk)
                    : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kf = 0u; kf < G4K / 8u; ++kf) {
                simdgroup_float8x8 ag, au;
                simdgroup_load(ag, &WgSh[(sgid * 8u) * G4S + kf * 8u], G4S);
                simdgroup_load(au, &WuSh[(sgid * 8u) * G4S + kf * 8u], G4S);
                for (uint nc = 0u; nc < 4u; ++nc) {
                    simdgroup_float8x8 bfr;
                    simdgroup_load(bfr, &Xsh[(nc * 8u) * G4S + kf * 8u], G4S, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(accG[nc], ag, bfr, accG[nc]);
                    simdgroup_multiply_accumulate(accU[nc], au, bfr, accU[nc]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float* bufG = &outb[sgid * 144u];
        threadgroup float* bufU = bufG + 72u;
        const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
        for (uint nc = 0u; nc < 4u; ++nc) {
            simdgroup_store(accG[nc], bufG, 9u);
            simdgroup_store(accU[nc], bufU, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint ti = t0 + nc * 8u + fj0 + jj;
                if (ti < c) {
                    const uint i = s0 + ti;
                    const float g = bufG[fi * 9u + fj0 + jj];
                    const float u = bufU[fi * 9u + fj0 + jj];
                    h[(ulong)aTok[i] * hs + aSlot[i] * pc.n_ff + row0 + sgid * 8u + fi] =
                        (g / (1.0f + exp(-g))) * u;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}


// IQ4_XS twin of moe_gu_grouped5 (80B routed experts).
// Both MMA operands are staged as contiguous 64-half blocks so every
// simdgroup_load is a stride-8 contiguous read -- no strided or transposed
// tile loads (that stride was worth ~1.45x: llama's mul_mm hits 12.0 TFLOPS
// on this box vs 8.2 for the strided skeleton at identical shape).
//   W tiles: block (rb, kb) at [(rb*8 + kb)*64], elem (r&7)*8 + (k&7)
//   X tile:  block (kb, tb) at [(kb*4 + tb)*64], elem (k&7)*8 + (t&7)
// 32 rows/matrix x 32 tokens, K-chunk 64; f16 staging, f32 accumulators
// (llama-identical numeric class); 14.6 KB threadgroup. Token tiles ride
// grid z (early-return past c) so hot experts fan out instead of looping.
kernel void moe_gu_grouped5_iq4(device const block_iq4_xs*  gwE   [[buffer(0)]],
                            device const block_iq4_xs*  uwE   [[buffer(1)]],
                            device const block_q8_0*    gwS   [[buffer(2)]],
                            device const block_q8_0*    uwS   [[buffer(3)]],
                            device const float*         x     [[buffer(4)]],
                            device const uint*          start [[buffer(5)]],
                            device const uint*          aTok  [[buffer(6)]],
                            device const uint*          aSlot [[buffer(7)]],
                            device float*               h     [[buffer(8)]],
                            constant MoePC&             pc    [[buffer(9)]],
                            uint3 tid3  [[thread_position_in_threadgroup]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint  sgid  [[simdgroup_index_in_threadgroup]],
                            uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;                 // 0..127, 4 simdgroups
    const uint nrt = pc.n_ff / G4M;          // 32-row tiles per expert
    const uint e  = tgpig.x / nrt;           // 0..255 routed, 256 shared
    const uint rt = tgpig.x % nrt;
    const uint s0 = start[e], c = start[e + 1u] - s0;
    if (c == 0u) return;

    const uint K = pc.n_embd;
    const uint hs = (pc.n_used + 1u) * pc.n_ff;
    const uint row0 = rt * G4M;

    threadgroup half  WgSh[G4M * 64u];
    threadgroup half  WuSh[G4M * 64u];
    threadgroup half  Xsh[64u * G4N];
    threadgroup float outb[4u * 144u];       // per-simd G(72)|U(72) bounce

    // staging assignment: one (matrix, row, 32-group) per thread per chunk
    const uint smat = tid >> 6;              // 0 = gate, 1 = up
    const uint srow = (tid >> 1) & 31u;
    const uint sgrp = tid & 1u;
    threadgroup half* wsh = (smat ? WuSh : WgSh)
                          + ((srow >> 3) * 8u) * 64u + (srow & 7u) * 8u;

    const uint t0 = tgpig.z * G4N;               // token tile (grid z)
    if (t0 >= c) return;
    {
        simdgroup_float8x8 accG[4], accU[4];
        for (uint i = 0u; i < 4u; ++i) {
            accG[i] = simdgroup_float8x8(0.0f);
            accU[i] = simdgroup_float8x8(0.0f);
        }

        for (uint k0 = 0u; k0 < K; k0 += 64u) {
            const uint g32 = (k0 >> 5) + sgrp;           // global 32-group
            const uint row = row0 + srow;
            if (e < pc.n_expert) {                       // routed: IQ4_XS
                device const block_iq4_xs* mat = smat ? uwE : gwE;
                device const block_iq4_xs& blk =
                    mat[((ulong)e * pc.n_ff + row) * (K >> 8) + (g32 >> 3)];
                const uint ib = g32 & 7u;
                const uint slb = uint(blk.scales_l[ib >> 1u]);
                const uint shh = uint(blk.scales_h);
                const int  ls  = int(((slb >> (4u * (ib & 1u))) & 0xFu) |
                                     (((shh >> (2u * ib)) & 3u) << 4u)) - 32;
                const float dl = float(blk.d) * float(ls);
                device const packed_uchar4* qp =
                    (device const packed_uchar4*)&blk.qs[ib * 16u];
                for (uint l = 0u; l < 4u; ++l) {
                    const uchar4 qa = uchar4(qp[(l & 1u) * 2u]);
                    const uchar4 qb = uchar4(qp[(l & 1u) * 2u + 1u]);
                    const uint4 na = (l < 2u) ? (uint4(qa) & 0xFu) : (uint4(qa) >> 4u);
                    const uint4 nb = (l < 2u) ? (uint4(qb) & 0xFu) : (uint4(qb) >> 4u);
                    threadgroup half4* dst4 = (threadgroup half4*)(wsh + (sgrp * 4u + l) * 64u);
                    dst4[0] = half4(dl * float4(float(kvalues_iq4nl[na.x]), float(kvalues_iq4nl[na.y]),
                                                 float(kvalues_iq4nl[na.z]), float(kvalues_iq4nl[na.w])));
                    dst4[1] = half4(dl * float4(float(kvalues_iq4nl[nb.x]), float(kvalues_iq4nl[nb.y]),
                                                 float(kvalues_iq4nl[nb.z]), float(kvalues_iq4nl[nb.w])));
                }
            } else {                                     // shared: Q8_0
                device const block_q8_0* mat = smat ? uwS : gwS;
                device const block_q8_0& blk = mat[(ulong)row * (K >> 5) + g32];
                const half d = blk.d;
                device const packed_char4* qp = (device const packed_char4*)blk.qs;
                for (uint l = 0u; l < 4u; ++l) {
                    threadgroup half4* dst4 =
                        (threadgroup half4*)(wsh + (sgrp * 4u + l) * 64u);
                    dst4[0] = d * half4(char4(qp[2u * l]));
                    dst4[1] = d * half4(char4(qp[2u * l + 1u]));
                }
            }
            // X: 32 tokens x 64 K, thread -> (token, k-block), two passes
            for (uint p = 0u; p < 2u; ++p) {
                const uint slot = tid + p * 128u;
                const uint tt = slot >> 3, kb = slot & 7u;
                const uint ti = t0 + tt;
                threadgroup half* xd =
                    &Xsh[(kb * 4u + (tt >> 3)) * 64u + (tt & 7u)];
                if (ti < c) {
                    device const float* xp = x + (ulong)aTok[s0 + ti] * K + k0 + kb * 8u;
                    const float4 a = *(device const packed_float4*)xp;
                    const float4 b = *(device const packed_float4*)(xp + 4u);
                    xd[0u]  = half(a.x); xd[8u]  = half(a.y);
                    xd[16u] = half(a.z); xd[24u] = half(a.w);
                    xd[32u] = half(b.x); xd[40u] = half(b.y);
                    xd[48u] = half(b.z); xd[56u] = half(b.w);
                } else {
                    for (uint j = 0u; j < 8u; ++j) xd[j * 8u] = 0.0h;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kf = 0u; kf < 8u; ++kf) {
                simdgroup_half8x8 ag, au;
                simdgroup_load(ag, &WgSh[(sgid * 8u + kf) * 64u], 8u);
                simdgroup_load(au, &WuSh[(sgid * 8u + kf) * 64u], 8u);
                simdgroup_barrier(mem_flags::mem_none);
                for (uint nc = 0u; nc < 4u; ++nc) {
                    simdgroup_half8x8 bfr;
                    simdgroup_load(bfr, &Xsh[(kf * 4u + nc) * 64u], 8u);
                    simdgroup_multiply_accumulate(accG[nc], ag, bfr, accG[nc]);
                    simdgroup_multiply_accumulate(accU[nc], au, bfr, accU[nc]);
                }
                simdgroup_barrier(mem_flags::mem_none);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float* bufG = &outb[sgid * 144u];
        threadgroup float* bufU = bufG + 72u;
        const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
        for (uint nc = 0u; nc < 4u; ++nc) {
            simdgroup_store(accG[nc], bufG, 9u);
            simdgroup_store(accU[nc], bufU, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint ti = t0 + nc * 8u + fj0 + jj;
                if (ti < c) {
                    const uint i = s0 + ti;
                    const float g = bufG[fi * 9u + fj0 + jj];
                    const float u = bufU[fi * 9u + fj0 + jj];
                    h[(ulong)aTok[i] * hs + aSlot[i] * pc.n_ff + row0 + sgid * 8u + fi] =
                        (g / (1.0f + exp(-g))) * u;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
