#include <metal_stdlib>
using namespace metal;
#include "fa_kv.metal"

// Two-head GQA prefill flash attention for Qwen's 16Q:2KV geometry. One
// simdgroup retains each K/V fragment and applies it to two adjacent Q heads.
// QT=8 cuts the O/P tile to 20 KiB; S16 restores occupancy within that single
// resident TG. The default twin parallelizes the 64-key softmax by assigning
// one simdgroup per (head,row); the exact twin preserves scalar sum order.

struct FaBPC {
    uint tmax; uint dh; uint nRot; uint hQ;
    uint hKV; float eps; float freqBase; uint base;
    uint Tn; uint qbase;
};

template <bool SIMD_SOFTMAX, bool HALF_KV, bool Q8_KV>
static inline void fa_gqa2_body(device const float* qhat,
                                device const uchar* kc,
                                device const uchar* vc,
                                device const float* qfull,
                                device float* att,
                                constant FaBPC& pc,
                                threadgroup float* Otg,
                                threadgroup float* Stg,
                                threadgroup float* mrow,
                                threadgroup float* lrow,
                                threadgroup float* corrTg,
                                threadgroup float* KVf,
                                uint3 tid3, uint sgid, uint slid,
                                uint3 tgpig) {
    constexpr uint HG = 2u, QT = 8u, KB = 64u, NS = 16u, DH = 256u;
    const uint h0 = tgpig.x * HG;
    const uint qstart = tgpig.z * QT;
    if (h0 + HG > pc.hQ || qstart >= pc.Tn) return;
    const uint tid = tid3.x;
    const uint kv = h0 / (pc.hQ / pc.hKV);
    const float qs = rsqrt(float(pc.dh)) * 1.4426950408889634f;

    for (uint i = tid; i < HG * QT * DH; i += NS * 32u) Otg[i] = 0.0f;
    if (tid < HG * QT) { mrow[tid] = -3.4e38f; lrow[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lastq = min(qstart + QT, pc.Tn) - 1u;
    const uint nk = pc.base + lastq + 1u;

    for (uint p0 = 0u; p0 < nk; p0 += KB) {
        const bool fullyUnmasked = p0 + KB <= pc.base + qstart + 1u;

        // One key tile per active simdgroup; the same K fragment feeds both Q
        // heads and produces two independent 8x8 score tiles.
        if (sgid < KB / 8u && p0 + sgid * 8u < nk) {
            simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);
            for (uint dt = 0u; dt < DH / 8u; ++dt) {
                simdgroup_float8x8 q0, q1, kf;
                simdgroup_load(q0, qhat + (ulong)qstart * pc.hQ * DH +
                                       (ulong)h0 * DH + dt * 8u, pc.hQ * DH);
                simdgroup_load(q1, qhat + (ulong)qstart * pc.hQ * DH +
                                       (ulong)(h0 + 1u) * DH + dt * 8u, pc.hQ * DH);
                const ulong ko =
                    (ulong)(kv * pc.tmax + p0 + sgid * 8u) * DH + dt * 8u;
                if (Q8_KV) {
                    threadgroup float* tile = KVf + sgid * 64u;
                    for (uint e = slid; e < 64u; e += 32u) {
                        const ulong row = (ulong)kv * pc.tmax + p0 +
                                          sgid * 8u + e / 8u;
                        const ulong rb = row * KV_Q8_ROW;
                        const float d = *((device const float*)(kc + rb));
                        const char qv = *((device const char*)(
                            kc + rb + KV_Q8_DATA + dt * 8u + e % 8u));
                        tile[e] = d * float(qv);
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_load(kf, tile, 8u, ulong2(0, 0), true);
                } else if (HALF_KV) {
                    threadgroup float* tile = KVf + sgid * 64u;
                    for (uint e = slid; e < 64u; e += 32u)
                        tile[e] = float(((device const half*)kc)[
                            ko + (e / 8u) * DH + e % 8u]);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_load(kf, tile, 8u, ulong2(0, 0), true);
                } else {
                    simdgroup_load(kf, (device const float*)kc + ko,
                                   DH, ulong2(0, 0), true);
                }
                simdgroup_multiply_accumulate(acc0, q0, kf, acc0);
                simdgroup_multiply_accumulate(acc1, q1, kf, acc1);
            }
            simdgroup_store(acc0, Stg + sgid * 8u, KB);
            simdgroup_store(acc1, Stg + QT * KB + sgid * 8u, KB);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (SIMD_SOFTMAX) {
            // S16 maps exactly to 2 heads x 8 query rows. Each lane owns two
            // keys; max is exact, while sum uses the simd tree (the sole f32
            // order change versus the exact twin).
            const uint hr = sgid / QT, r = sgid % QT;
            const uint qidx = qstart + r;
            threadgroup float* sr = Stg + (hr * QT + r) * KB;
            if (qidx < pc.Tn) {
                const uint qpos = pc.base + qidx;
                const uint kcount = min(KB, nk - p0);
                const uint k0 = slid, k1 = slid + 32u;
                const bool v0 = k0 < kcount && (fullyUnmasked || p0 + k0 <= qpos);
                const bool v1 = k1 < kcount && (fullyUnmasked || p0 + k1 <= qpos);
                const float s0 = v0 ? sr[k0] * qs : -3.4e38f;
                const float s1 = v1 ? sr[k1] * qs : -3.4e38f;
                const float bmax = simd_max(max(s0, s1));
                const float oldm = mrow[sgid], newm = max(oldm, bmax);
                const float corr = fast::exp2(oldm - newm);
                const float pv0 = v0 ? fast::exp2(s0 - newm) : 0.0f;
                const float pv1 = v1 ? fast::exp2(s1 - newm) : 0.0f;
                sr[k0] = pv0;
                sr[k1] = pv1;
                const float rsum = simd_sum(pv0 + pv1);
                if (slid == 0u) {
                    mrow[sgid] = newm;
                    lrow[sgid] = lrow[sgid] * corr + rsum;
                    corrTg[sgid] = corr;
                }
            } else {
                sr[slid] = 0.0f;
                sr[slid + 32u] = 0.0f;
                if (slid == 0u) corrTg[sgid] = 1.0f;
            }
        } else if (tid < HG * QT) {
            const uint hr = tid / QT, r = tid % QT;
            const uint qidx = qstart + r;
            if (qidx < pc.Tn) {
                const uint qpos = pc.base + qidx;
                const uint kcount = min(KB, nk - p0);
                threadgroup float* sr = Stg + (hr * QT + r) * KB;
                float bmax = -3.4e38f;
                if (fullyUnmasked) {
                    for (uint k = 0u; k < KB; ++k) {
                        const float s = sr[k] * qs;
                        sr[k] = s; bmax = max(bmax, s);
                    }
                } else {
                    for (uint k = 0u; k < kcount; ++k) {
                        const float s = p0 + k <= qpos ? sr[k] * qs : -3.4e38f;
                        sr[k] = s; bmax = max(bmax, s);
                    }
                }
                const float oldm = mrow[tid], newm = max(oldm, bmax);
                const float corr = fast::exp2(oldm - newm);
                float rsum = 0.0f;
                for (uint k = 0u; k < kcount; ++k) {
                    const float p = fast::exp2(sr[k] - newm);
                    sr[k] = p; rsum += p;
                }
                for (uint k = kcount; k < KB; ++k) sr[k] = 0.0f;
                mrow[tid] = newm;
                lrow[tid] = lrow[tid] * corr + rsum;
                corrTg[tid] = corr;
            } else {
                threadgroup float* sr = Stg + tid * KB;
                for (uint k = 0u; k < KB; ++k) sr[k] = 0.0f;
                corrTg[tid] = 1.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < HG * QT * DH; i += NS * 32u)
            Otg[i] *= corrTg[i / DH];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each simdgroup owns two dimension tiles and reuses P and V across
        // the two Q heads.
        for (uint tix = sgid; tix < DH / 8u; tix += NS) {
            const uint dct = tix;
            simdgroup_float8x8 acc0, acc1;
            simdgroup_load(acc0, Otg + dct * 8u, DH);
            simdgroup_load(acc1, Otg + QT * DH + dct * 8u, DH);
            for (uint kt = 0u; kt < KB / 8u; ++kt) {
                if (p0 + kt * 8u >= nk) continue;
                simdgroup_float8x8 p0f, p1f, vf;
                simdgroup_load(p0f, Stg + kt * 8u, KB);
                simdgroup_load(p1f, Stg + QT * KB + kt * 8u, KB);
                const ulong vo =
                    (ulong)(kv * pc.tmax + p0 + kt * 8u) * DH + dct * 8u;
                if (Q8_KV) {
                    threadgroup float* tile = KVf + sgid * 64u;
                    for (uint e = slid; e < 64u; e += 32u) {
                        const ulong row = (ulong)kv * pc.tmax + p0 +
                                          kt * 8u + e / 8u;
                        const ulong rb = row * KV_Q8_ROW;
                        const float d = *((device const float*)(vc + rb));
                        const char qv = *((device const char*)(
                            vc + rb + KV_Q8_DATA + dct * 8u + e % 8u));
                        tile[e] = d * float(qv);
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_load(vf, tile, 8u);
                } else if (HALF_KV) {
                    threadgroup float* tile = KVf + sgid * 64u;
                    for (uint e = slid; e < 64u; e += 32u)
                        tile[e] = float(((device const half*)vc)[
                            vo + (e / 8u) * DH + e % 8u]);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_load(vf, tile, 8u);
                } else {
                    simdgroup_load(vf, (device const float*)vc + vo, DH);
                }
                simdgroup_multiply_accumulate(acc0, p0f, vf, acc0);
                simdgroup_multiply_accumulate(acc1, p1f, vf, acc1);
            }
            simdgroup_store(acc0, Otg + dct * 8u, DH);
            simdgroup_store(acc1, Otg + QT * DH + dct * 8u, DH);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < HG * QT * DH; i += NS * 32u) {
        const uint hr = i / (QT * DH);
        const uint rem = i % (QT * DH), r = rem / DH, t = rem % DH;
        const uint qidx = qstart + r, h = h0 + hr;
        if (qidx >= pc.Tn) continue;
        const float o = Otg[i] / lrow[hr * QT + r];
        const float g = qfull[(ulong)qidx * pc.hQ * 2u * DH +
                              (ulong)h * 2u * DH + DH + t];
        att[(ulong)qidx * pc.hQ * DH + (ulong)h * DH + t] =
            o * (1.0f / (1.0f + exp(-g)));
    }
}

#define FA_GQA2_KERNEL(NAME, SIMD_SOFTMAX, HALF_KV, Q8_KV, KV_SCRATCH)           \
kernel void NAME(device const float* qhat [[buffer(0)]],                         \
                 device const uchar* kc [[buffer(1)]],                           \
                 device const uchar* vc [[buffer(2)]],                           \
                 device const float* qfull [[buffer(3)]],                        \
                 device float* att [[buffer(4)]],                                \
                 constant FaBPC& pc [[buffer(5)]],                               \
                 uint3 tid3 [[thread_position_in_threadgroup]],                  \
                 uint sgid [[simdgroup_index_in_threadgroup]],                   \
                 uint slid [[thread_index_in_simdgroup]],                        \
                 uint3 tgpig [[threadgroup_position_in_grid]]) {                 \
    threadgroup float Otg[2u * 8u * 256u];                                      \
    threadgroup float Stg[2u * 8u * 64u];                                       \
    threadgroup float mrow[16u], lrow[16u], corrTg[16u];                        \
    threadgroup float KVf[KV_SCRATCH];                                          \
    fa_gqa2_body<SIMD_SOFTMAX, HALF_KV, Q8_KV>(qhat, kc, vc, qfull, att, pc,    \
                                Otg, Stg, mrow, lrow, corrTg, KVf,               \
                                tid3, sgid, slid, tgpig);                        \
}

FA_GQA2_KERNEL(fa_attn_batch_gqa2, true, false, false, 1)
FA_GQA2_KERNEL(fa_attn_batch_gqa2_exact, false, false, false, 1)
FA_GQA2_KERNEL(fa_attn_batch_gqa2_f16, true, true, false, 16u * 64u)
FA_GQA2_KERNEL(fa_attn_batch_gqa2_q8, true, false, true, 16u * 64u)
#undef FA_GQA2_KERNEL
