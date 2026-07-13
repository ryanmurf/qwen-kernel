#include <metal_stdlib>
using namespace metal;

// Decode-once grouped DOWN projection (prefill, QK_MOE_GROUPED path).
// Same structure as moe_gu_grouped4: one threadgroup per (expert, 32-row
// tile of n_embd), expert rows dequantized to threadgroup f32 once per
// K-chunk per 32-token pass, gathered token h-slices multiplied via f32
// simdgroup MMA. Down outputs need cross-expert accumulation per token, so
// results are written UNWEIGHTED to dY[tok][slot][n_embd]; moe_down_reduce
// then folds the 9 slots with the routing weights into y. f32 fragments =
// summation-order noise only (exact class, same as v4 gate+up).
// Two routed-format entry points (iq4_xs for 37 layers, q6_k for 34/38/39);
// the shared expert (virtual expert 256) is Q8_0 in both.

#include "iq_tables.metal"
#include "moe_common.metal"

constant uint DGM = 32u;   // output rows per tile
constant uint DGN = 32u;   // token columns per pass
constant uint DGK = 64u;   // K elems staged per chunk (2 32-groups)
constant uint DGS = 68u;   // padded float stride (16B-aligned rows for float4)

// decode one 32-elem group of an IQ4_XS row into dst[0..32)
static inline void iq4_stage32(device const block_iq4_xs* mat, ulong rowBlocks,
                               uint row, uint g32, threadgroup float* dst) {
    device const block_iq4_xs& blk = mat[(ulong)row * rowBlocks + (g32 >> 3u)];
    const uint ib = g32 & 7u;
    const uint slb = uint(blk.scales_l[ib >> 1u]);
    const uint sh  = uint(blk.scales_h);
    const int  ls  = int(((slb >> (4u * (ib & 1u))) & 0xFu) |
                         (((sh >> (2u * ib)) & 3u) << 4u)) - 32;
    const float dl = float(blk.d) * float(ls);
    device const packed_uchar4* qp = (device const packed_uchar4*)&blk.qs[ib * 16u];
    threadgroup float4* dst4 = (threadgroup float4*)dst;
    for (uint j = 0u; j < 4u; ++j) {
        const uchar4 q = uchar4(qp[j]);
        const uint4 lo = uint4(q) & 0xFu, hn = uint4(q) >> 4u;
        dst4[j] = dl * float4(float(kvalues_iq4nl[lo.x]), float(kvalues_iq4nl[lo.y]),
                              float(kvalues_iq4nl[lo.z]), float(kvalues_iq4nl[lo.w]));
        dst4[4u + j] = dl * float4(float(kvalues_iq4nl[hn.x]), float(kvalues_iq4nl[hn.y]),
                                   float(kvalues_iq4nl[hn.z]), float(kvalues_iq4nl[hn.w]));
    }
}

// decode one 32-elem group (two 16-elem scale groups) of a Q6_K row
static inline void q6k_stage32(device const block_q6_K* mat, ulong rowBlocks,
                               uint row, uint g32, threadgroup float* dst) {
    device const block_q6_K& blk = mat[(ulong)row * rowBlocks + (g32 >> 3u)];
    const float d = float(blk.d);
    for (uint half16 = 0u; half16 < 2u; ++half16) {
        const uint gg = (g32 & 7u) * 2u + half16;   // 16-elem group in block
        const uint hh = gg >> 3u;
        const uint r  = (gg & 7u) >> 1u;
        const uint is = gg & 1u;
        const float sc = float(int(blk.scales[hh * 8u + r * 2u + is]));
        const uint qlBase = hh * 64u + (r & 1u) * 32u + is * 16u;
        const uint qhBase = hh * 32u + is * 16u;
        const uint shift  = r * 2u;
        const bool hi     = r >= 2u;
        device const packed_uchar4* qlp = (device const packed_uchar4*)&blk.ql[qlBase];
        device const packed_uchar4* qhp = (device const packed_uchar4*)&blk.qh[qhBase];
        threadgroup float4* o4 = (threadgroup float4*)(dst + half16 * 16u);
        for (uint i = 0u; i < 4u; ++i) {
            const uchar4 qlv = uchar4(qlp[i]);
            const uchar4 qhv = uchar4(qhp[i]);
            const uint4 lo = hi ? (uint4(qlv) >> 4u) : (uint4(qlv) & 0xFu);
            const int4  q  = int4(lo | (((uint4(qhv) >> shift) & 3u) << 4u)) - 32;
            o4[i] = (d * sc) * float4(q);
        }
    }
}

static inline void q8_stage32(device const block_q8_0* mat, ulong rowBlocks,
                              uint row, uint g32, threadgroup float* dst) {
    device const block_q8_0& blk = mat[(ulong)row * rowBlocks + g32];
    const float d = float(blk.d);
    device const packed_char4* qp = (device const packed_char4*)blk.qs;
    threadgroup float4* dst4 = (threadgroup float4*)dst;
    for (uint i = 0u; i < 8u; ++i)
        dst4[i] = d * float4(char4(qp[i]));
}

// shared body: ROUTED selects the format via the isQ6 flag
template <typename BLK, bool ISQ6>
static inline void down_grouped_body(device const BLK*        dwE,
                                     device const block_q8_0* dwS,
                                     device const float*      h,
                                     device const uint*       start,
                                     device const uint*       aTok,
                                     device const uint*       aSlot,
                                     device float*            dY,
                                     constant MoePC&          pc,
                                     threadgroup float*       Wsh,
                                     threadgroup float*       Xsh,
                                     threadgroup float*       outb,
                                     uint tid, uint tgx, uint sgid, uint slid)
{
    const uint nrt = pc.n_embd / DGM;        // row tiles per expert
    const uint e  = tgx / nrt;               // 0..255 routed, 256 shared
    const uint rt = tgx % nrt;
    const uint s0 = start[e], c = start[e + 1u] - s0;
    if (c == 0u) return;

    const uint K  = pc.n_ff;                 // 512
    const uint hs = (pc.n_used + 1u) * pc.n_ff;
    const uint row0 = rt * DGM;

    const uint srow = tid >> 1;              // 0..31 (threads 0..63 stage W)
    const uint sgrp = tid & 1u;

    for (uint t0 = 0u; t0 < c; t0 += DGN) {
        simdgroup_float8x8 acc[4];
        for (uint i = 0u; i < 4u; ++i) acc[i] = simdgroup_float8x8(0.0f);

        for (uint k0 = 0u; k0 < K; k0 += DGK) {
            if (tid < 64u) {
                const uint g32 = (k0 >> 5) + sgrp;
                const uint row = row0 + srow;
                threadgroup float* dst = Wsh + srow * DGS + sgrp * 32u;
                if (e < pc.n_expert) {
                    device const BLK* mat = dwE + (ulong)e * pc.n_embd * (K >> 8);
                    if (ISQ6) q6k_stage32((device const block_q6_K*)mat, K >> 8, row, g32, dst);
                    else      iq4_stage32((device const block_iq4_xs*)mat, K >> 8, row, g32, dst);
                } else {
                    q8_stage32(dwS, K >> 5, row, g32, dst);
                }
            }
            for (uint idx = tid * 4u; idx < DGN * DGK; idx += 128u * 4u) {
                const uint tt = idx >> 6, kk = idx & 63u;
                const uint ti = t0 + tt;
                ((threadgroup float4*)&Xsh[tt * DGS + kk])[0] = ti < c
                    ? *(device const packed_float4*)(h + (ulong)aTok[s0 + ti] * hs +
                                                     aSlot[s0 + ti] * pc.n_ff + k0 + kk)
                    : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kf = 0u; kf < DGK / 8u; ++kf) {
                simdgroup_float8x8 a;
                simdgroup_load(a, &Wsh[(sgid * 8u) * DGS + kf * 8u], DGS);
                for (uint nc = 0u; nc < 4u; ++nc) {
                    simdgroup_float8x8 bfr;
                    simdgroup_load(bfr, &Xsh[(nc * 8u) * DGS + kf * 8u], DGS, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc[nc], a, bfr, acc[nc]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float* buf = &outb[sgid * 72u];
        const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
        for (uint nc = 0u; nc < 4u; ++nc) {
            simdgroup_store(acc[nc], buf, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint ti = t0 + nc * 8u + fj0 + jj;
                if (ti < c) {
                    const uint i = s0 + ti;
                    dY[((ulong)aTok[i] * (pc.n_used + 1u) + aSlot[i]) * pc.n_embd +
                       row0 + sgid * 8u + fi] = buf[fi * 9u + fj0 + jj];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void moe_down_grouped_iq4(device const block_iq4_xs* dwE   [[buffer(0)]],
                                 device const block_q8_0*   dwS   [[buffer(1)]],
                                 device const float*        h     [[buffer(2)]],
                                 device const uint*         start [[buffer(3)]],
                                 device const uint*         aTok  [[buffer(4)]],
                                 device const uint*         aSlot [[buffer(5)]],
                                 device float*              dY    [[buffer(6)]],
                                 constant MoePC&            pc    [[buffer(7)]],
                                 uint3 tid3  [[thread_position_in_threadgroup]],
                                 uint3 tgpig [[threadgroup_position_in_grid]],
                                 uint  sgid  [[simdgroup_index_in_threadgroup]],
                                 uint  slid  [[thread_index_in_simdgroup]])
{
    threadgroup float Wsh[DGM * DGS];
    threadgroup float Xsh[DGN * DGS];
    threadgroup float outb[4u * 72u];
    down_grouped_body<block_iq4_xs, false>(dwE, dwS, h, start, aTok, aSlot, dY, pc,
                                           Wsh, Xsh, outb, tid3.x, tgpig.x, sgid, slid);
}

kernel void moe_down_grouped_q6k(device const block_q6_K* dwE   [[buffer(0)]],
                                 device const block_q8_0* dwS   [[buffer(1)]],
                                 device const float*      h     [[buffer(2)]],
                                 device const uint*       start [[buffer(3)]],
                                 device const uint*       aTok  [[buffer(4)]],
                                 device const uint*       aSlot [[buffer(5)]],
                                 device float*            dY    [[buffer(6)]],
                                 constant MoePC&          pc    [[buffer(7)]],
                                 uint3 tid3  [[thread_position_in_threadgroup]],
                                 uint3 tgpig [[threadgroup_position_in_grid]],
                                 uint  sgid  [[simdgroup_index_in_threadgroup]],
                                 uint  slid  [[thread_index_in_simdgroup]])
{
    threadgroup float Wsh[DGM * DGS];
    threadgroup float Xsh[DGN * DGS];
    threadgroup float outb[4u * 72u];
    down_grouped_body<block_q6_K, true>(dwE, dwS, h, start, aTok, aSlot, dY, pc,
                                        Wsh, Xsh, outb, tid3.x, tgpig.x, sgid, slid);
}

kernel void moe_down_grouped_live_iq4(device const block_iq4_xs* dwE [[buffer(0)]],
                                      device const block_q8_0* dwS [[buffer(1)]],
                                      device const float* h [[buffer(2)]],
                                      device const uint* start [[buffer(3)]],
                                      device const uint* aTok [[buffer(4)]],
                                      device const uint* aSlot [[buffer(5)]],
                                      device float* dY [[buffer(6)]],
                                      device const uint* live [[buffer(7)]],
                                      constant MoePC& pc [[buffer(8)]],
                                      uint3 tid3 [[thread_position_in_threadgroup]],
                                      uint3 tgpig [[threadgroup_position_in_grid]],
                                      uint sgid [[simdgroup_index_in_threadgroup]],
                                      uint slid [[thread_index_in_simdgroup]])
{
    const uint nrt = pc.n_embd / DGM;
    const uint e = live[tgpig.x / nrt];
    if (e > pc.n_expert) return;
    const uint mapped = e * nrt + tgpig.x % nrt;
    threadgroup float Wsh[DGM * DGS];
    threadgroup float Xsh[DGN * DGS];
    threadgroup float outb[4u * 72u];
    down_grouped_body<block_iq4_xs, false>(dwE, dwS, h, start, aTok, aSlot,
        dY, pc, Wsh, Xsh, outb, tid3.x, mapped, sgid, slid);
}

kernel void moe_down_grouped_live_q6k(device const block_q6_K* dwE [[buffer(0)]],
                                      device const block_q8_0* dwS [[buffer(1)]],
                                      device const float* h [[buffer(2)]],
                                      device const uint* start [[buffer(3)]],
                                      device const uint* aTok [[buffer(4)]],
                                      device const uint* aSlot [[buffer(5)]],
                                      device float* dY [[buffer(6)]],
                                      device const uint* live [[buffer(7)]],
                                      constant MoePC& pc [[buffer(8)]],
                                      uint3 tid3 [[thread_position_in_threadgroup]],
                                      uint3 tgpig [[threadgroup_position_in_grid]],
                                      uint sgid [[simdgroup_index_in_threadgroup]],
                                      uint slid [[thread_index_in_simdgroup]])
{
    const uint nrt = pc.n_embd / DGM;
    const uint e = live[tgpig.x / nrt];
    if (e > pc.n_expert) return;
    const uint mapped = e * nrt + tgpig.x % nrt;
    threadgroup float Wsh[DGM * DGS];
    threadgroup float Xsh[DGN * DGS];
    threadgroup float outb[4u * 72u];
    down_grouped_body<block_q6_K, true>(dwE, dwS, h, start, aTok, aSlot,
        dY, pc, Wsh, Xsh, outb, tid3.x, mapped, sgid, slid);
}

// y[tok][d] = sum_s w[tok][s] * dY[tok][s][d] + wShared * dY[tok][8][d]
kernel void moe_down_reduce(device const float* dY  [[buffer(0)]],
                            device const SelT*  sel [[buffer(1)]],
                            device float*       y   [[buffer(2)]],
                            constant MoePC&     pc  [[buffer(3)]],
                            uint3 tid3  [[thread_position_in_threadgroup]],
                            uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint d = tgpig.x * 256u + tid3.x;
    const uint rq = tgpig.z;
    if (d >= pc.n_embd) return;
    device const float* base = dY + (ulong)rq * (pc.n_used + 1u) * pc.n_embd + d;
    float acc = 0.0f;
    for (uint s = 0u; s < pc.n_used; ++s)
        acc += sel[rq].w[s] * base[s * pc.n_embd];
    acc += sel[rq].wShared * base[pc.n_used * pc.n_embd];
    y[(ulong)rq * pc.n_embd + d] = acc;
}

// ---- f16 tier (QK_MOE_GROUPED=3 record config): 64x32 tiles, 256 threads,
// half staging (gemm_q8_0_h / mul_mm_id precision class). Each thread
// stages 16 elements per chunk: IQ4_XS splits into nibble planes, Q6_K into
// its native 16-elem scale groups, Q8_0 into block halves.
constant uint DHM = 64u;
constant uint DHN = 32u;
constant uint DHK = 64u;
constant uint DHS = 68u;

static inline void iq4_stage16h(device const block_iq4_xs* mat, ulong rowBlocks,
                                uint row, uint g32, uint h16, threadgroup half* dst) {
    device const block_iq4_xs& blk = mat[(ulong)row * rowBlocks + (g32 >> 3u)];
    const uint ib = g32 & 7u;
    const uint slb = uint(blk.scales_l[ib >> 1u]);
    const uint sh  = uint(blk.scales_h);
    const int  ls  = int(((slb >> (4u * (ib & 1u))) & 0xFu) |
                         (((sh >> (2u * ib)) & 3u) << 4u)) - 32;
    const float dl = float(blk.d) * float(ls);
    device const packed_uchar4* qp = (device const packed_uchar4*)&blk.qs[ib * 16u];
    threadgroup half4* dst4 = (threadgroup half4*)dst;
    for (uint j = 0u; j < 4u; ++j) {
        const uchar4 q = uchar4(qp[j]);
        const uint4 nib = h16 ? (uint4(q) >> 4u) : (uint4(q) & 0xFu);
        dst4[j] = half4(dl * float4(float(kvalues_iq4nl[nib.x]), float(kvalues_iq4nl[nib.y]),
                                    float(kvalues_iq4nl[nib.z]), float(kvalues_iq4nl[nib.w])));
    }
}

static inline void q6k_stage16h(device const block_q6_K* mat, ulong rowBlocks,
                                uint row, uint g32, uint h16, threadgroup half* dst) {
    device const block_q6_K& blk = mat[(ulong)row * rowBlocks + (g32 >> 3u)];
    const float d = float(blk.d);
    const uint gg = (g32 & 7u) * 2u + h16;
    const uint hh = gg >> 3u;
    const uint r  = (gg & 7u) >> 1u;
    const uint is = gg & 1u;
    const float sc = float(int(blk.scales[hh * 8u + r * 2u + is]));
    const uint qlBase = hh * 64u + (r & 1u) * 32u + is * 16u;
    const uint qhBase = hh * 32u + is * 16u;
    const uint shift  = r * 2u;
    const bool hi     = r >= 2u;
    device const packed_uchar4* qlp = (device const packed_uchar4*)&blk.ql[qlBase];
    device const packed_uchar4* qhp = (device const packed_uchar4*)&blk.qh[qhBase];
    const float ds = d * sc;
    threadgroup half4* dst4 = (threadgroup half4*)dst;
    for (uint i = 0u; i < 4u; ++i) {
        const uchar4 qlv = uchar4(qlp[i]);
        const uchar4 qhv = uchar4(qhp[i]);
        const uint4 lo = hi ? (uint4(qlv) >> 4u) : (uint4(qlv) & 0xFu);
        const int4  q  = int4(lo | (((uint4(qhv) >> shift) & 3u) << 4u)) - 32;
        dst4[i] = half4(ds * float4(q));
    }
}

static inline void q8_stage16h(device const block_q8_0* mat, ulong rowBlocks,
                               uint row, uint g32, uint h16, threadgroup half* dst) {
    device const block_q8_0& blk = mat[(ulong)row * rowBlocks + g32];
    const half d = blk.d;
    device const packed_char4* qp = (device const packed_char4*)&blk.qs[h16 * 16u];
    threadgroup half4* dst4 = (threadgroup half4*)dst;
    for (uint i = 0u; i < 4u; ++i)
        dst4[i] = d * half4(char4(qp[i]));
}

template <typename BLK, bool ISQ6>
static inline void down_grouped_body_h(device const BLK*        dwE,
                                       device const block_q8_0* dwS,
                                       device const float*      h,
                                       device const uint*       start,
                                       device const uint*       aTok,
                                       device const uint*       aSlot,
                                       device float*            dY,
                                       constant MoePC&          pc,
                                       threadgroup half*        Wsh,
                                       threadgroup half*        Xsh,
                                       threadgroup float*       outb,
                                       uint tid, uint tgx, uint sgid, uint slid)
{
    const uint nrt = pc.n_embd / DHM;
    const uint e  = tgx / nrt;
    const uint rt = tgx % nrt;
    const uint s0 = start[e], c = start[e + 1u] - s0;
    if (c == 0u) return;

    const uint K  = pc.n_ff;
    const uint hs = (pc.n_used + 1u) * pc.n_ff;
    const uint row0 = rt * DHM;

    const uint srow = tid >> 2;              // 0..63
    const uint sgrp = (tid >> 1) & 1u;       // 32-group within chunk
    const uint sh16 = tid & 1u;              // 16-elem half within group
    threadgroup half* dst = Wsh + srow * DHS + sgrp * 32u + sh16 * 16u;

    for (uint t0 = 0u; t0 < c; t0 += DHN) {
        simdgroup_float8x8 acc[4];
        for (uint i = 0u; i < 4u; ++i) acc[i] = simdgroup_float8x8(0.0f);

        for (uint k0 = 0u; k0 < K; k0 += DHK) {
            const uint g32 = (k0 >> 5) + sgrp;
            const uint row = row0 + srow;
            if (e < pc.n_expert) {
                device const BLK* mat = dwE + (ulong)e * pc.n_embd * (K >> 8);
                if (ISQ6) q6k_stage16h((device const block_q6_K*)mat, K >> 8, row, g32, sh16, dst);
                else      iq4_stage16h((device const block_iq4_xs*)mat, K >> 8, row, g32, sh16, dst);
            } else {
                q8_stage16h(dwS, K >> 5, row, g32, sh16, dst);
            }
            for (uint idx = tid * 4u; idx < DHN * DHK; idx += 256u * 4u) {
                const uint tt = idx >> 6, kk = idx & 63u;
                const uint ti = t0 + tt;
                ((threadgroup half4*)&Xsh[tt * DHS + kk])[0] = ti < c
                    ? half4(*(device const packed_float4*)(h + (ulong)aTok[s0 + ti] * hs +
                                                           aSlot[s0 + ti] * pc.n_ff + k0 + kk))
                    : half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kf = 0u; kf < DHK / 8u; ++kf) {
                simdgroup_half8x8 a;
                simdgroup_load(a, &Wsh[(sgid * 8u) * DHS + kf * 8u], DHS);
                for (uint nc = 0u; nc < 4u; ++nc) {
                    simdgroup_half8x8 bfr;
                    simdgroup_load(bfr, &Xsh[(nc * 8u) * DHS + kf * 8u], DHS, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc[nc], a, bfr, acc[nc]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float* buf = &outb[sgid * 72u];
        const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
        for (uint nc = 0u; nc < 4u; ++nc) {
            simdgroup_store(acc[nc], buf, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint ti = t0 + nc * 8u + fj0 + jj;
                if (ti < c) {
                    const uint i = s0 + ti;
                    dY[((ulong)aTok[i] * (pc.n_used + 1u) + aSlot[i]) * pc.n_embd +
                       row0 + sgid * 8u + fi] = buf[fi * 9u + fj0 + jj];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void moe_down_grouped_h_iq4(device const block_iq4_xs* dwE   [[buffer(0)]],
                                   device const block_q8_0*   dwS   [[buffer(1)]],
                                   device const float*        h     [[buffer(2)]],
                                   device const uint*         start [[buffer(3)]],
                                   device const uint*         aTok  [[buffer(4)]],
                                   device const uint*         aSlot [[buffer(5)]],
                                   device float*              dY    [[buffer(6)]],
                                   constant MoePC&            pc    [[buffer(7)]],
                                   uint3 tid3  [[thread_position_in_threadgroup]],
                                   uint3 tgpig [[threadgroup_position_in_grid]],
                                   uint  sgid  [[simdgroup_index_in_threadgroup]],
                                   uint  slid  [[thread_index_in_simdgroup]])
{
    threadgroup half  Wsh[DHM * DHS];
    threadgroup half  Xsh[DHN * DHS];
    threadgroup float outb[8u * 72u];
    down_grouped_body_h<block_iq4_xs, false>(dwE, dwS, h, start, aTok, aSlot, dY, pc,
                                             Wsh, Xsh, outb, tid3.x, tgpig.x, sgid, slid);
}

kernel void moe_down_grouped_h_q6k(device const block_q6_K* dwE   [[buffer(0)]],
                                   device const block_q8_0* dwS   [[buffer(1)]],
                                   device const float*      h     [[buffer(2)]],
                                   device const uint*       start [[buffer(3)]],
                                   device const uint*       aTok  [[buffer(4)]],
                                   device const uint*       aSlot [[buffer(5)]],
                                   device float*            dY    [[buffer(6)]],
                                   constant MoePC&          pc    [[buffer(7)]],
                                   uint3 tid3  [[thread_position_in_threadgroup]],
                                   uint3 tgpig [[threadgroup_position_in_grid]],
                                   uint  sgid  [[simdgroup_index_in_threadgroup]],
                                   uint  slid  [[thread_index_in_simdgroup]])
{
    threadgroup half  Wsh[DHM * DHS];
    threadgroup half  Xsh[DHN * DHS];
    threadgroup float outb[8u * 72u];
    down_grouped_body_h<block_q6_K, true>(dwE, dwS, h, start, aTok, aSlot, dY, pc,
                                          Wsh, Xsh, outb, tid3.x, tgpig.x, sgid, slid);
}


// ---------------------------------------------------------------------------
// packed-block twins (_p): same 8x8-block staging as moe_gu_grouped5 /
// gemm_q8_0_hp -- every simdgroup_load contiguous stride-8. 64 rows x 32
// tokens, K-chunk 32, 128 threads / 4 simds, 7.3 KB threadgroup, token tiles
// on grid z. The 16-elem decoders write two 8-elem k-blocks (bstep apart).

static inline void iq4_stage16p(device const block_iq4_xs* mat, ulong rowBlocks,
                                uint row, uint g32, uint h16,
                                threadgroup half* d0, threadgroup half* d1) {
    device const block_iq4_xs& blk = mat[(ulong)row * rowBlocks + (g32 >> 3u)];
    const uint ib = g32 & 7u;
    const uint slb = uint(blk.scales_l[ib >> 1u]);
    const uint sh  = uint(blk.scales_h);
    const int  ls  = int(((slb >> (4u * (ib & 1u))) & 0xFu) |
                         (((sh >> (2u * ib)) & 3u) << 4u)) - 32;
    const float dl = float(blk.d) * float(ls);
    device const packed_uchar4* qp = (device const packed_uchar4*)&blk.qs[ib * 16u];
    for (uint j = 0u; j < 4u; ++j) {
        const uchar4 q = uchar4(qp[j]);
        const uint4 nib = h16 ? (uint4(q) >> 4u) : (uint4(q) & 0xFu);
        threadgroup half4* dst4 = (threadgroup half4*)(j < 2u ? d0 : d1) + (j & 1u);
        dst4[0] = half4(dl * float4(float(kvalues_iq4nl[nib.x]), float(kvalues_iq4nl[nib.y]),
                                    float(kvalues_iq4nl[nib.z]), float(kvalues_iq4nl[nib.w])));
    }
}

static inline void q6k_stage16p(device const block_q6_K* mat, ulong rowBlocks,
                                uint row, uint g32, uint h16,
                                threadgroup half* d0, threadgroup half* d1) {
    device const block_q6_K& blk = mat[(ulong)row * rowBlocks + (g32 >> 3u)];
    const float d = float(blk.d);
    const uint gg = (g32 & 7u) * 2u + h16;
    const uint hh = gg >> 3u;
    const uint r  = (gg & 7u) >> 1u;
    const uint is = gg & 1u;
    const float sc = float(int(blk.scales[hh * 8u + r * 2u + is]));
    const uint qlBase = hh * 64u + (r & 1u) * 32u + is * 16u;
    const uint qhBase = hh * 32u + is * 16u;
    const uint shift  = r * 2u;
    const bool hi     = r >= 2u;
    device const packed_uchar4* qlp = (device const packed_uchar4*)&blk.ql[qlBase];
    device const packed_uchar4* qhp = (device const packed_uchar4*)&blk.qh[qhBase];
    const float ds = d * sc;
    for (uint i = 0u; i < 4u; ++i) {
        const uchar4 qlv = uchar4(qlp[i]);
        const uchar4 qhv = uchar4(qhp[i]);
        const uint4 lo = hi ? (uint4(qlv) >> 4u) : (uint4(qlv) & 0xFu);
        const int4  q  = int4(lo | (((uint4(qhv) >> shift) & 3u) << 4u)) - 32;
        threadgroup half4* dst4 = (threadgroup half4*)(i < 2u ? d0 : d1) + (i & 1u);
        dst4[0] = half4(ds * float4(q));
    }
}

static inline void q8_stage16p(device const block_q8_0* mat, ulong rowBlocks,
                               uint row, uint g32, uint h16,
                               threadgroup half* d0, threadgroup half* d1) {
    device const block_q8_0& blk = mat[(ulong)row * rowBlocks + g32];
    const half d = blk.d;
    device const packed_char4* qp = (device const packed_char4*)&blk.qs[h16 * 16u];
    for (uint i = 0u; i < 4u; ++i) {
        threadgroup half4* dst4 = (threadgroup half4*)(i < 2u ? d0 : d1) + (i & 1u);
        dst4[0] = d * half4(char4(qp[i]));
    }
}

template <typename BLK, bool ISQ6>
static inline void down_grouped_body_p(device const BLK*        dwE,
                                       device const block_q8_0* dwS,
                                       device const float*      h,
                                       device const uint*       start,
                                       device const uint*       aTok,
                                       device const uint*       aSlot,
                                       device float*            dY,
                                       constant MoePC&          pc,
                                       threadgroup half*        Wsh,
                                       threadgroup half*        Xsh,
                                       threadgroup float*       outb,
                                       uint tid, uint tgx, uint tgz,
                                       uint sgid, uint slid)
{
    const uint nrt = pc.n_embd / DHM;
    const uint e  = tgx / nrt;
    const uint rt = tgx % nrt;
    const uint s0 = start[e], c = start[e + 1u] - s0;
    if (c == 0u) return;
    const uint t0 = tgz * DHN;
    if (t0 >= c) return;

    const uint K  = pc.n_ff;
    const uint hs = (pc.n_used + 1u) * pc.n_ff;
    const uint row0 = rt * DHM;

    const uint srow = tid >> 1;              // 0..63
    const uint sh16 = tid & 1u;              // 16-elem half of the 32-chunk
    // W: block (rb, kb) at [(rb*4 + kb)*64], elem (r&7)*8 + (k&7)
    threadgroup half* w0 = Wsh + ((srow >> 3) * 4u + sh16 * 2u) * 64u + (srow & 7u) * 8u;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; ++i)
        for (uint j = 0u; j < 4u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint k0 = 0u; k0 < K; k0 += 32u) {
        const uint g32 = k0 >> 5;
        const uint row = row0 + srow;
        if (e < pc.n_expert) {
            device const BLK* mat = dwE + (ulong)e * pc.n_embd * (K >> 8);
            if (ISQ6) q6k_stage16p((device const block_q6_K*)mat, K >> 8, row, g32, sh16, w0, w0 + 64u);
            else      iq4_stage16p((device const block_iq4_xs*)mat, K >> 8, row, g32, sh16, w0, w0 + 64u);
        } else {
            q8_stage16p(dwS, K >> 5, row, g32, sh16, w0, w0 + 64u);
        }
        {   // X: 32 tokens x 32 K, thread -> (token, k-block)
            const uint tt = tid >> 2, kb = tid & 3u;
            const uint ti = t0 + tt;
            threadgroup half* xd = &Xsh[(kb * 4u + (tt >> 3)) * 64u + (tt & 7u)];
            if (ti < c) {
                device const float* xp = h + (ulong)aTok[s0 + ti] * hs +
                                         aSlot[s0 + ti] * pc.n_ff + k0 + kb * 8u;
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

        for (uint kf = 0u; kf < 4u; ++kf) {
            simdgroup_half8x8 a[2];
            simdgroup_load(a[0], &Wsh[((sgid * 2u + 0u) * 4u + kf) * 64u], 8u);
            simdgroup_load(a[1], &Wsh[((sgid * 2u + 1u) * 4u + kf) * 64u], 8u);
            simdgroup_barrier(mem_flags::mem_none);
            for (uint nc = 0u; nc < 4u; ++nc) {
                simdgroup_half8x8 bfr;
                simdgroup_load(bfr, &Xsh[(kf * 4u + nc) * 64u], 8u);
                simdgroup_multiply_accumulate(acc[0][nc], a[0], bfr, acc[0][nc]);
                simdgroup_multiply_accumulate(acc[1][nc], a[1], bfr, acc[1][nc]);
            }
            simdgroup_barrier(mem_flags::mem_none);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* buf = &outb[sgid * 72u];
    const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
    for (uint mr = 0u; mr < 2u; ++mr) {
        for (uint nc = 0u; nc < 4u; ++nc) {
            simdgroup_store(acc[mr][nc], buf, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint ti = t0 + nc * 8u + fj0 + jj;
                if (ti < c) {
                    const uint i = s0 + ti;
                    dY[((ulong)aTok[i] * (pc.n_used + 1u) + aSlot[i]) * pc.n_embd +
                       row0 + (sgid * 2u + mr) * 8u + fi] = buf[fi * 9u + fj0 + jj];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void moe_down_grouped_p_iq4(device const block_iq4_xs* dwE   [[buffer(0)]],
                                   device const block_q8_0*   dwS   [[buffer(1)]],
                                   device const float*        h     [[buffer(2)]],
                                   device const uint*         start [[buffer(3)]],
                                   device const uint*         aTok  [[buffer(4)]],
                                   device const uint*         aSlot [[buffer(5)]],
                                   device float*              dY    [[buffer(6)]],
                                   constant MoePC&            pc    [[buffer(7)]],
                                   uint3 tid3  [[thread_position_in_threadgroup]],
                                   uint3 tgpig [[threadgroup_position_in_grid]],
                                   uint  sgid  [[simdgroup_index_in_threadgroup]],
                                   uint  slid  [[thread_index_in_simdgroup]])
{
    threadgroup half  Wsh[DHM * 32u];
    threadgroup half  Xsh[32u * DHN];
    threadgroup float outb[4u * 72u];
    down_grouped_body_p<block_iq4_xs, false>(dwE, dwS, h, start, aTok, aSlot, dY, pc,
                                             Wsh, Xsh, outb, tid3.x, tgpig.x, tgpig.z, sgid, slid);
}

kernel void moe_down_grouped_p_q6k(device const block_q6_K* dwE   [[buffer(0)]],
                                   device const block_q8_0* dwS   [[buffer(1)]],
                                   device const float*      h     [[buffer(2)]],
                                   device const uint*       start [[buffer(3)]],
                                   device const uint*       aTok  [[buffer(4)]],
                                   device const uint*       aSlot [[buffer(5)]],
                                   device float*            dY    [[buffer(6)]],
                                   constant MoePC&          pc    [[buffer(7)]],
                                   uint3 tid3  [[thread_position_in_threadgroup]],
                                   uint3 tgpig [[threadgroup_position_in_grid]],
                                   uint  sgid  [[simdgroup_index_in_threadgroup]],
                                   uint  slid  [[thread_index_in_simdgroup]])
{
    threadgroup half  Wsh[DHM * 32u];
    threadgroup half  Xsh[32u * DHN];
    threadgroup float outb[4u * 72u];
    down_grouped_body_p<block_q6_K, true>(dwE, dwS, h, start, aTok, aSlot, dY, pc,
                                          Wsh, Xsh, outb, tid3.x, tgpig.x, tgpig.z, sgid, slid);
}

kernel void moe_down_grouped_p_live_iq4(device const block_iq4_xs* dwE [[buffer(0)]],
                                        device const block_q8_0* dwS [[buffer(1)]],
                                        device const float* h [[buffer(2)]],
                                        device const uint* start [[buffer(3)]],
                                        device const uint* aTok [[buffer(4)]],
                                        device const uint* aSlot [[buffer(5)]],
                                        device float* dY [[buffer(6)]],
                                        device const uint* live [[buffer(7)]],
                                        constant MoePC& pc [[buffer(8)]],
                                        uint3 tid3 [[thread_position_in_threadgroup]],
                                        uint3 tgpig [[threadgroup_position_in_grid]],
                                        uint sgid [[simdgroup_index_in_threadgroup]],
                                        uint slid [[thread_index_in_simdgroup]])
{
    const uint nrt = pc.n_embd / DHM;
    const uint ei = tgpig.x / nrt;
    const uint e = live[ei];
    if (e > pc.n_expert) return;
    const uint mapped = e * nrt + tgpig.x % nrt;
    threadgroup half Wsh[DHM * 32u];
    threadgroup half Xsh[32u * DHN];
    threadgroup float outb[4u * 72u];
    down_grouped_body_p<block_iq4_xs, false>(dwE, dwS, h, start, aTok, aSlot,
        dY, pc, Wsh, Xsh, outb, tid3.x, mapped, tgpig.z, sgid, slid);
}

kernel void moe_down_grouped_p_live_q6k(device const block_q6_K* dwE [[buffer(0)]],
                                        device const block_q8_0* dwS [[buffer(1)]],
                                        device const float* h [[buffer(2)]],
                                        device const uint* start [[buffer(3)]],
                                        device const uint* aTok [[buffer(4)]],
                                        device const uint* aSlot [[buffer(5)]],
                                        device float* dY [[buffer(6)]],
                                        device const uint* live [[buffer(7)]],
                                        constant MoePC& pc [[buffer(8)]],
                                        uint3 tid3 [[thread_position_in_threadgroup]],
                                        uint3 tgpig [[threadgroup_position_in_grid]],
                                        uint sgid [[simdgroup_index_in_threadgroup]],
                                        uint slid [[thread_index_in_simdgroup]])
{
    const uint nrt = pc.n_embd / DHM;
    const uint ei = tgpig.x / nrt;
    const uint e = live[ei];
    if (e > pc.n_expert) return;
    const uint mapped = e * nrt + tgpig.x % nrt;
    threadgroup half Wsh[DHM * 32u];
    threadgroup half Xsh[32u * DHN];
    threadgroup float outb[4u * 72u];
    down_grouped_body_p<block_q6_K, true>(dwE, dwS, h, start, aTok, aSlot,
        dY, pc, Wsh, Xsh, outb, tid3.x, mapped, tgpig.z, sgid, slid);
}
