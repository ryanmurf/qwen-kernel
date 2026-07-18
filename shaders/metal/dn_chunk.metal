#include <metal_stdlib>
using namespace metal;

// Chunked gated delta rule (prefill replacement for dn_step_batch).
//
// Sequential recurrence per v-head h (dn_step_batch):
//   a_t = exp(g_t),  d_t = beta_t (v_t - a_t S_{t-1} k_t)
//   S_t = a_t S_{t-1} + d_t k_t^T,   o_t = S_t (qScale q_t)
// with S[j][i] (j = v-dim row, i = k-dim col), k/q L2-normalized by dn_conv.
//
// Chunk of C=64 tokens, L_t = sum_{s<=t} g_s (inclusive, within chunk).
// Unrolling from the chunk-start state S0 gives a unit-lower-triangular
// system for the delta rows D (row t = d_t^T):
//   (I + M) D = beta V - W S0^T,   M[t][s] = beta_t e^{L_t-L_s} (k_t.k_s), s<t
//   w_t = beta_t e^{L_t} k_t
// Since T = (I+M)^{-1} distributes, u~ = T (beta V) and w~ = T W are
// STATE-INDEPENDENT -> computed in parallel over all (head, chunk) pairs.
// The chunk-serial remainder is three small matmuls per chunk:
//   D    = u~ - w~ S0^T
//   o_t  = e^{L_t} S0 (qScale q_t) + sum_{s<=t} Att[t][s] d_s,
//          Att[t][s] = e^{L_t-L_s} qScale (q_t.k_s), s<=t (pre-masked, from solve)
//   S    <- e^{Llast} S0 + sum_s e^{Llast-L_s} d_s k_s^T
// All decay exponents are differences L_a - L_b with a >= b and g <= 0, so
// every factor is in (0,1]: no overflow. Verified vs dn_step_batch by the
// dncmp harness (random inputs + nonzero init state, o and final S).
//
// Three kernels:
//   dn_chunk_kq    grid (hK, 1, nCh): raw MMA dot tiles KK = K K^T and
//                  QK = Q K^T per k-head (lower-triangle 8x8 tiles only;
//                  qScale is folded into the solve's Att build).
//   dn_chunk_solve grid (hV, 1, nCh): builds M, forward-substitutes u~/w~
//                  (thread j owns column j: zero barriers in the solve).
//                  Emits everything the chunk-serial kernel consumes, fully
//                  baked: {u~, -w~, q~, K~} (rows >= Cc zeroed, w~ NEGATED so
//                  D = u~ + (-w~) S0^T is a pure multiply-accumulate chain),
//                  Att packed as 8x8 tiles, and e^{Llast} per (h, chunk).
//   dn_chunk_step  grid (hV, 1, 2): chunk-serial simdgroup-MMA over the
//                  three matmuls; tgpig.z picks a 64-wide column panel of
//                  the state (panels touch disjoint S rows). A scalar
//                  version of this kernel ran ~10x above its ALU cost: the
//                  per-lane same-address broadcast loops serialize on AGX.

struct DnCPC { uint dS; uint hK; uint hV; uint Tn; uint kDiv; };

constant uint C = 64u;  // chunk size

// dn_chunk_step is compiled with dS as a function constant (getPipe tpr=dS):
// compile-time loop bounds let the compiler fully unroll the state-row sweeps
// and keep srow[] in registers instead of scratch.
constant uint DSC [[function_constant(0)]];

kernel void dn_chunk_kq(device const float* conv [[buffer(0)]],
                        device float*       kq   [[buffer(1)]],
                        constant DnCPC&     pc   [[buffer(2)]],
                        uint3 tid3  [[thread_position_in_threadgroup]],
                        uint3 tgpig [[threadgroup_position_in_grid]],
                        uint  sgid  [[simdgroup_index_in_threadgroup]])
{
    const uint kh = tgpig.x;
    const uint c  = tgpig.z;
    const uint j  = tid3.x;
    const uint dS = pc.dS;
    const uint t0 = c * C;
    const uint Cc = min(C, pc.Tn - t0);
    const uint chQkv = (2u * pc.hK + pc.hV) * dS;
    const uint nKT = dS / 8u;

    // K chunk staged as packed 8x8 tiles (row-tile rt, col-tile ct) so every
    // simdgroup_load is contiguous stride-8; rows >= Cc zeroed (blind MMA).
    threadgroup float Ktg[64 * 128];
    for (uint t = 0u; t < C; ++t)
        Ktg[((t >> 3u) * nKT + (j >> 3u)) * 64u + (t & 7u) * 8u + (j & 7u)] =
            t < Cc ? conv[(ulong)(t0 + t) * chQkv + (ulong)pc.hK * dS + (ulong)kh * dS + j]
                   : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* kk = kq + ((ulong)(c * pc.hK + kh) * 2u) * (C * C);
    device float* qk = kk + C * C;

    // lower-triangle 8x8 output tiles (tt >= st): 36 of 64, split across simds
    for (uint q = sgid; q < 36u; q += 4u) {
        // unrank q -> (tt, st) with tt >= st
        uint tt = 0u, acc = 0u;
        while (acc + tt + 1u <= q) { acc += tt + 1u; ++tt; }
        const uint st = q - acc;

        simdgroup_float8x8 ck = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 cq = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kt = 0u; kt < nKT; ++kt) {
            simdgroup_float8x8 ak, aq, bk;
            simdgroup_load(ak, Ktg + (tt * nKT + kt) * 64u, 8u);
            simdgroup_load(bk, Ktg + (st * nKT + kt) * 64u, 8u,
                           ulong2(0, 0), true);   // K^T tile
            simdgroup_multiply_accumulate(ck, ak, bk, ck);
            simdgroup_load(aq, conv + (ulong)(t0 + tt * 8u) * chQkv +
                                   (ulong)kh * dS + kt * 8u, chQkv);
            simdgroup_multiply_accumulate(cq, aq, bk, cq);
        }
        simdgroup_store(ck, kk + (tt * 8u) * C + st * 8u, C);
        simdgroup_store(cq, qk + (tt * 8u) * C + st * 8u, C);
    }
}

kernel void dn_chunk_solve(device const float* conv [[buffer(0)]],
                           device const float* gb   [[buffer(1)]],
                           device const float* kq   [[buffer(2)]],
                           device float*       uw   [[buffer(3)]],
                           device float*       att  [[buffer(4)]],
                           device float*       el   [[buffer(5)]],
                           constant DnCPC&     pc   [[buffer(6)]],
                           uint3 tid3  [[thread_position_in_threadgroup]],
                           uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint h  = tgpig.x;
    const uint c  = tgpig.z;
    const uint j  = tid3.x;
    const uint dS = pc.dS;
    const uint t0 = c * C;
    const uint Cc = min(C, pc.Tn - t0);
    // kDiv semantics as in dn_step.metal (0 = modulo, else h / kDiv)
    const uint kh = pc.kDiv != 0u ? h / pc.kDiv : h % pc.hK;
    const uint chQkv = (2u * pc.hK + pc.hV) * dS;

    // M is strictly lower triangular. Compact storage cuts threadgroup
    // memory from 16 KiB to 7.9 KiB, increasing solve occupancy without
    // changing the forward-substitution arithmetic.
    threadgroup float M[64 * 63 / 2];
    threadgroup float Ltg[64];
    threadgroup float Btg[64];

    // One simdgroup parallelizes 64 decay/beta loads. Every lane executes the
    // same left-to-right shuffle/add chain, retaining the scalar accumulation
    // order without a second threadgroup barrier.
    if (j < 32u) {
        const uint t1 = j + 32u;
        const float g0 = j < Cc ? gb[(ulong)(t0 + j) * 2u * pc.hV + h] : 0.0f;
        const float g1 = t1 < Cc ? gb[(ulong)(t0 + t1) * 2u * pc.hV + h] : 0.0f;
        if (j < Cc)
            Btg[j] = gb[(ulong)(t0 + j) * 2u * pc.hV + pc.hV + h];
        if (t1 < Cc)
            Btg[t1] = gb[(ulong)(t0 + t1) * 2u * pc.hV + pc.hV + h];
        float acc = 0.0f;
#pragma unroll
        for (uint t = 0u; t < 32u; ++t) {
            acc += simd_shuffle(g0, t);
            if (j == 0u && t < Cc) Ltg[t] = acc;
        }
#pragma unroll
        for (uint t = 0u; t < 32u; ++t) {
            acc += simd_shuffle(g1, t);
            if (j == 0u && t + 32u < Cc) Ltg[t + 32u] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float* kk = kq + ((ulong)(c * pc.hK + kh) * 2u) * (C * C);
    device const float* qk = kk + C * C;
    device float* attO = att + (ulong)(c * pc.hV + h) * (C * C);

    for (uint idx = j; idx < C * C; idx += dS) {
        const uint t = idx >> 6u, s = idx & 63u;
        float a = 0.0f;
        if (t < Cc && s <= t) {
            const float e = exp(Ltg[t] - Ltg[s]);
            if (s < t)
                M[t * (t - 1u) / 2u + s] = Btg[t] * e * kk[idx];
            a = e * rsqrt(float(dS)) * qk[idx];   // qScale folded here (QK is raw)
        }
        // Att stored as packed 8x8 tiles for simdgroup_load in the step
        attO[((t >> 3u) * 8u + (s >> 3u)) * 64u + (t & 7u) * 8u + (s & 7u)] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Forward substitution, thread j owns column j of both solves. u/w live
    // as float4[16] with compile-time lane access — dynamically indexed
    // scalar arrays spill to scratch memory (measured 2x slowdown).
    const ulong vOff = (ulong)2u * pc.hK * dS + (ulong)h * dS;
    const ulong kOff = (ulong)pc.hK * dS + (ulong)kh * dS;
    float4 u4[16], w4[16];
    device float* uwO = uw + ((ulong)(c * pc.hV + h) * 4u) * (C * (ulong)dS);
    const uint nG = (Cc + 3u) / 4u;
    for (uint g = 0u; g < nG; ++g) {
        float4 uv = 0.0f, wv = 0.0f;
        // partial dots against completed groups (rows 4g..4g+3 x cols < 4g)
        for (uint s4 = 0u; s4 < g; ++s4) {
            const float4 us = u4[s4], ws = w4[s4];
#pragma unroll
            for (uint l = 0u; l < 4u; ++l) {
                const uint t = g * 4u + l;
                threadgroup const float* mr = M + t * (t - 1u) / 2u + s4 * 4u;
                const float4 mv = float4(mr[0], mr[1], mr[2], mr[3]);
                uv[l] -= dot(mv, us);
                wv[l] -= dot(mv, ws);
            }
        }
        // in-group triangular fixup, lanes serialized at compile time
#pragma unroll
        for (uint l = 0u; l < 4u; ++l) {
            const uint t = g * 4u + l;
            if (t >= Cc) break;
            float ut = Btg[t] * conv[(ulong)(t0 + t) * chQkv + vOff + j] + uv[l];
            float wt = Btg[t] * exp(Ltg[t]) * conv[(ulong)(t0 + t) * chQkv + kOff + j] + wv[l];
            for (uint m = 0u; m < l; ++m) {
                const float mm = M[t * (t - 1u) / 2u + g * 4u + m];
                ut -= mm * uv[m];   // uv[m] now holds the SOLVED u[t'] (see below)
                wt -= mm * wv[m];
            }
            uv[l] = ut;
            wv[l] = wt;
        }
        u4[g] = uv;
        w4[g] = wv;
        // slot 0 = u~, slot 1 = -w~ (negated: step's D pass is then a pure
        // multiply-accumulate chain seeded with u~)
#pragma unroll
        for (uint l = 0u; l < 4u; ++l) {
            const uint t = g * 4u + l;
            if (t < Cc) {
                const uint pk = ((t >> 3u) * (dS / 8u) + (j >> 3u)) * 64u +
                                (t & 7u) * 8u + (j & 7u);
                uwO[pk] = uv[l];
                uwO[C * dS + pk] = -wv[l];
            }
        }
    }
    // slots 2/3: q~ = e^{L_t} qScale q_t and K~ = e^{Llast-L_s} k_s, fully
    // baked so the step kernel reads nothing but scratch. Rows >= Cc of all
    // four slots are zeroed: the step runs blind 8x8 tiles over the full
    // chunk and garbage rows would poison the MMA (0 * inf = NaN).
    // all four slots are PACKED 8x8 tiles: (t/8 * dS/8 + j/8)*64 + ...
    const float qScale = rsqrt(float(dS));
    const float Llast = Ltg[Cc - 1u];
    const ulong qOff = (ulong)kh * dS;
    for (uint t = 0u; t < Cc; ++t) {
        const uint pk = ((t >> 3u) * (dS / 8u) + (j >> 3u)) * 64u +
                        (t & 7u) * 8u + (j & 7u);
        uwO[2u * C * dS + pk] =
            exp(Ltg[t]) * qScale * conv[(ulong)(t0 + t) * chQkv + qOff + j];
        uwO[3u * C * dS + pk] =
            exp(Llast - Ltg[t]) * conv[(ulong)(t0 + t) * chQkv + kOff + j];
    }
    for (uint t = Cc; t < C; ++t) {
        const uint pk = ((t >> 3u) * (dS / 8u) + (j >> 3u)) * 64u +
                        (t & 7u) * 8u + (j & 7u);
        uwO[pk] = 0.0f;
        uwO[C * dS + pk] = 0.0f;
        uwO[2u * C * dS + pk] = 0.0f;
        uwO[3u * C * dS + pk] = 0.0f;
    }
    if (j == 0u) el[c * pc.hV + h] = exp(Llast);
}

kernel void dn_chunk_step(device const float* uw  [[buffer(0)]],
                          device const float* att [[buffer(1)]],
                          device const float* el  [[buffer(2)]],
                          device float*       o   [[buffer(3)]],
                          device float*       sS  [[buffer(4)]],
                          constant DnCPC&     pc  [[buffer(5)]],
                          uint3 tid3  [[thread_position_in_threadgroup]],
                          uint3 tgpig [[threadgroup_position_in_grid]],
                          uint  sgid  [[simdgroup_index_in_threadgroup]])
{
    const uint h   = tgpig.x;
    const uint pan = tgpig.z;          // column panel of the state (0 or 1)
    const uint dS  = DSC;              // compile-time (function constant)
    const uint PW  = dS / 2u;          // panel width (64 for dS=128)
    const uint jp0 = pan * PW;
    const uint nCh = (pc.Tn + C - 1u) / C;
    const uint hVdS = pc.hV * dS;
    const uint nJT = PW / 8u;          // 8 j-tiles per panel
    const uint nKT = dS / 8u;          // 16 k-tiles across the state dim

    // D panel as packed 8x8 tiles: tile (st, jt) at (st*nJT + jt)*64
    threadgroup float Dtg[64 * 64];
    threadgroup float diagTg[64];

    device float* S = sS + (ulong)h * dS * dS;

    for (uint c = 0u; c < nCh; ++c) {
        const ulong mb = ((ulong)(c * pc.hV + h) * 4u) * (C * (ulong)dS);
        device const float* um = uw + mb;                 // u~   [64][dS]
        device const float* wm = um + C * dS;             // -w~  [64][dS]
        device const float* qm = wm + C * dS;             // q~   [64][dS]
        device const float* km = qm + C * dS;             // K~   [64][dS]
        device const float* ac = att + (ulong)(c * pc.hV + h) * (C * C);
        const float elast = el[c * pc.hV + h];

        if (tid3.x < 64u)
            diagTg[tid3.x] = (tid3.x / 8u == (tid3.x & 7u)) ? elast : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // D = u~ + (-w~) S0^T   -> packed tiles in Dtg
        for (uint q = sgid * 16u; q < sgid * 16u + 16u; ++q) {
            const uint st = q >> 3u, jt = q & 7u;
            if (jt >= nJT) continue;
            simdgroup_float8x8 cf;
            simdgroup_load(cf, um + (st * nKT + jp0 / 8u + jt) * 64u, 8u);
            for (uint kt = 0u; kt < nKT; ++kt) {
                simdgroup_float8x8 af, bf;
                simdgroup_load(af, wm + (st * nKT + kt) * 64u, 8u);
                simdgroup_load(bf, S + (ulong)(jp0 + jt * 8u) * dS + kt * 8u, dS,
                               ulong2(0, 0), true);   // S^T tile
                simdgroup_multiply_accumulate(cf, af, bf, cf);
            }
            simdgroup_store(cf, Dtg + (st * 8u + jt) * 64u, 8u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = q~ S0^T + Att D   (Att pre-masked lower+diag, packed tiles)
        for (uint q = sgid * 16u; q < sgid * 16u + 16u; ++q) {
            const uint tt = q >> 3u, jt = q & 7u;
            if (jt >= nJT) continue;
            simdgroup_float8x8 cf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kt = 0u; kt < nKT; ++kt) {
                simdgroup_float8x8 af, bf;
                simdgroup_load(af, qm + (tt * nKT + kt) * 64u, 8u);
                simdgroup_load(bf, S + (ulong)(jp0 + jt * 8u) * dS + kt * 8u, dS,
                               ulong2(0, 0), true);
                simdgroup_multiply_accumulate(cf, af, bf, cf);
            }
            for (uint st = 0u; st < 8u; ++st) {
                simdgroup_float8x8 af, bf;
                simdgroup_load(af, ac + (tt * 8u + st) * 64u, 8u);
                simdgroup_load(bf, Dtg + (st * 8u + jt) * 64u, 8u);
                simdgroup_multiply_accumulate(cf, af, bf, cf);
            }
            simdgroup_store(cf, o + (ulong)(c * C + tt * 8u) * hVdS +
                                    (ulong)h * dS + jp0 + jt * 8u, hVdS);
        }
        // all simds must finish reading S rows before phase 3 rewrites them
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // S[rows jp] = elast S + D^T K~   (diag-tile MMA folds the elast term)
        simdgroup_float8x8 df;
        simdgroup_load(df, diagTg, 8u);
        for (uint q = sgid; q < nJT * nKT; q += 4u) {
            const uint jt = q / nKT, it = q % nKT;
            simdgroup_float8x8 cf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint st = 0u; st < 8u; ++st) {
                simdgroup_float8x8 af, bf;
                simdgroup_load(af, Dtg + (st * 8u + jt) * 64u, 8u,
                               ulong2(0, 0), true);   // D^T tile
                simdgroup_load(bf, km + (st * nKT + it) * 64u, 8u);
                simdgroup_multiply_accumulate(cf, af, bf, cf);
            }
            simdgroup_float8x8 sf;
            device float* sp = S + (ulong)(jp0 + jt * 8u) * dS + it * 8u;
            simdgroup_load(sf, sp, dS);
            simdgroup_multiply_accumulate(cf, df, sf, cf);
            simdgroup_store(cf, sp, dS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}

// Resident-state retile. Each 8-row panel is loaded once, advances every
// 64-token chunk in about 6 KiB of threadgroup memory, and is written once.
static inline void dn_chunk_step_res_body(device const float* uw,
                                          device const float* att,
                                          device const float* el,
                                          device float* o,
                                          device float* sS,
                                          constant DnCPC& pc,
                                          threadgroup float* Stg,
                                          threadgroup float* Dtg,
                                          threadgroup float* diagTg,
                                          uint3 tid3, uint3 tgpig, uint sgid)
{
    constexpr uint PW = 8u;
    constexpr uint NS = 8u;
    const uint h   = tgpig.x;
    const uint pan = tgpig.z;
    const uint dS  = DSC;
    const uint jp0 = pan * PW;
    const uint nCh = (pc.Tn + C - 1u) / C;
    const uint hVdS = pc.hV * dS;
    constexpr uint nJT = PW / 8u;
    const uint nKT = dS / 8u;

    device float* S = sS + (ulong)h * dS * dS;

    for (uint idx = tid3.x; idx < PW * dS; idx += NS * 32u) {
        const uint j = idx / dS, k = idx % dS;
        Stg[idx] = S[(ulong)(jp0 + j) * dS + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0u; c < nCh; ++c) {
        const ulong mb = ((ulong)(c * pc.hV + h) * 4u) * (C * (ulong)dS);
        device const float* um = uw + mb;
        device const float* wm = um + C * dS;
        device const float* qm = wm + C * dS;
        device const float* km = qm + C * dS;
        device const float* ac = att + (ulong)(c * pc.hV + h) * (C * C);
        const float elast = el[c * pc.hV + h];

        if (tid3.x < 64u)
            diagTg[tid3.x] = (tid3.x / 8u == (tid3.x & 7u)) ? elast : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // D = u~ + (-w~) S0^T
        for (uint q = sgid; q < 8u * nJT; q += NS) {
            const uint st = q / nJT, jt = q % nJT;
            simdgroup_float8x8 cf;
            simdgroup_load(cf, um + (st * nKT + jp0 / 8u + jt) * 64u, 8u);
            for (uint kt = 0u; kt < nKT; ++kt) {
                simdgroup_float8x8 af, bf;
                simdgroup_load(af, wm + (st * nKT + kt) * 64u, 8u);
                simdgroup_load(bf, Stg + (jt * 8u) * dS + kt * 8u, dS,
                               ulong2(0, 0), true);
                simdgroup_multiply_accumulate(cf, af, bf, cf);
            }
            simdgroup_store(cf, Dtg + (st * nJT + jt) * 64u, 8u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = q~ S0^T + Att D
        for (uint q = sgid; q < 8u * nJT; q += NS) {
            const uint tt = q / nJT, jt = q % nJT;
            simdgroup_float8x8 cf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kt = 0u; kt < nKT; ++kt) {
                simdgroup_float8x8 af, bf;
                simdgroup_load(af, qm + (tt * nKT + kt) * 64u, 8u);
                simdgroup_load(bf, Stg + (jt * 8u) * dS + kt * 8u, dS,
                               ulong2(0, 0), true);
                simdgroup_multiply_accumulate(cf, af, bf, cf);
            }
            for (uint st = 0u; st < 8u; ++st) {
                simdgroup_float8x8 af, bf;
                simdgroup_load(af, ac + (tt * 8u + st) * 64u, 8u);
                simdgroup_load(bf, Dtg + (st * nJT + jt) * 64u, 8u);
                simdgroup_multiply_accumulate(cf, af, bf, cf);
            }
            simdgroup_store(cf, o + (ulong)(c * C + tt * 8u) * hVdS +
                                    (ulong)h * dS + jp0 + jt * 8u, hVdS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = elast S + D^T K~, retained in Stg for the next chunk.
        simdgroup_float8x8 df;
        simdgroup_load(df, diagTg, 8u);
        for (uint q = sgid; q < nJT * nKT; q += NS) {
            const uint jt = q / nKT, it = q % nKT;
            simdgroup_float8x8 cf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint st = 0u; st < 8u; ++st) {
                simdgroup_float8x8 af, bf;
                simdgroup_load(af, Dtg + (st * nJT + jt) * 64u, 8u,
                               ulong2(0, 0), true);
                simdgroup_load(bf, km + (st * nKT + it) * 64u, 8u);
                simdgroup_multiply_accumulate(cf, af, bf, cf);
            }
            simdgroup_float8x8 sf;
            threadgroup float* sp = Stg + (jt * 8u) * dS + it * 8u;
            simdgroup_load(sf, sp, dS);
            simdgroup_multiply_accumulate(cf, df, sf, cf);
            simdgroup_store(cf, sp, dS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint idx = tid3.x; idx < PW * dS; idx += NS * 32u) {
        const uint j = idx / dS, k = idx % dS;
        S[(ulong)(jp0 + j) * dS + k] = Stg[idx];
    }
}

kernel void dn_chunk_step_res8_s8(device const float* uw  [[buffer(0)]],
                                  device const float* att [[buffer(1)]],
                                  device const float* el  [[buffer(2)]],
                                  device float*       o   [[buffer(3)]],
                                  device float*       sS  [[buffer(4)]],
                                  constant DnCPC&     pc  [[buffer(5)]],
                                  uint3 tid3  [[thread_position_in_threadgroup]],
                                  uint3 tgpig [[threadgroup_position_in_grid]],
                                  uint  sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Stg[8u * 128u];
    threadgroup float Dtg[C * 8u];
    threadgroup float diagTg[64];
    dn_chunk_step_res_body(uw, att, el, o, sS, pc, Stg, Dtg, diagTg,
                           tid3, tgpig, sgid);
}
