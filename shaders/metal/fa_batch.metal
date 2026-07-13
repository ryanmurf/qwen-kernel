#include <metal_stdlib>
using namespace metal;
#include "fa_kv.metal"

// Batched-prefill attention pair: each z workgroup is a token offset within
// one prompt chunk for a single slot; K/V cache writes use pos = base + n,
// attention runs causally through pos with online softmax. qbase lets the
// host tile the query axis. Ports of fa_prep_batch.comp / fa_attn_batch.comp.

struct FaBPC {
    uint tmax; uint dh; uint nRot; uint hQ;
    uint hKV; float eps; float freqBase; uint base;
    uint Tn; uint qbase;
};

kernel void fa_prep_batch(device const float* qfull  [[buffer(0)]],
                          device const float* kin    [[buffer(1)]],
                          device const float* vin    [[buffer(2)]],
                          device const float* qn     [[buffer(3)]],
                          device const float* kn     [[buffer(4)]],
                          device float*       qhat   [[buffer(5)]],
                          device uchar*       kc     [[buffer(6)]],
                          device uchar*       vc     [[buffer(7)]],
                          device const float* ropeCS [[buffer(8)]],
                          constant FaBPC&     pc     [[buffer(9)]],
                          uint3 tid3  [[thread_position_in_threadgroup]],
                          uint3 tgpig [[threadgroup_position_in_grid]],
                          uint  sgid  [[simdgroup_index_in_threadgroup]],
                          uint  slid  [[thread_index_in_simdgroup]])
{
    const uint w  = tgpig.x;
    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint n  = tgpig.z;
    const uint pos = pc.base + n;
    const uint qfo = n * pc.hQ * 2u * dh;
    const uint kio = n * pc.hKV * dh;
    const uint qho = n * pc.hQ * dh;

    if (w >= pc.hQ + pc.hKV) {           // v: plain copy into cache
        const uint h = w - pc.hQ - pc.hKV;
        if (t < dh) kv_store(vc, (h * pc.tmax + pos) * dh + t,
                             vin[kio + h * dh + t]);
        return;
    }

    const bool isQ = w < pc.hQ;
    const uint h = isQ ? w : w - pc.hQ;
    float v = 0.0f;
    if (t < dh) v = isQ ? qfull[qfo + h * 2u * dh + t] : kin[kio + h * dh + t];

    threadgroup float sv[256];
    threadgroup float red[8];
    threadgroup float scale;
    const float sg = simd_sum(v * v);
    if (slid == 0u) red[sgid] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
        float tot = 0.0f;
        for (uint i = 0u; i < 8u; ++i) tot += red[i];
        scale = 1.0f / sqrt(tot / float(dh) + pc.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < dh) sv[t] = v * scale * (isQ ? qn[t] : kn[t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_ = pc.nRot / 2u;
    if (t < dh) {
        float outV;
        if (t < pc.nRot) {
            const uint j = t < half_ ? t : t - half_;
            const uint ci = 2u * (pos * half_ + j);
            const float c = ropeCS[ci], s = ropeCS[ci + 1u];
            const float x0 = sv[j], x1 = sv[j + half_];
            outV = t < half_ ? x0 * c - x1 * s : x0 * s + x1 * c;
        } else {
            outV = sv[t];
        }
        if (isQ) qhat[qho + h * dh + t] = outV;
        else     kv_store(kc, (h * pc.tmax + pos) * dh + t, outV);
    }
}

kernel void fa_attn_batch(device const float* qhat  [[buffer(0)]],
                          device const uchar* kc    [[buffer(1)]],
                          device const uchar* vc    [[buffer(2)]],
                          device const float* qfull [[buffer(3)]],
                          device float*       att   [[buffer(4)]],
                          constant FaBPC&     pc    [[buffer(5)]],
                          uint3 tid3  [[thread_position_in_threadgroup]],
                          uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint h  = tgpig.x;
    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint n  = pc.qbase + tgpig.z;
    const uint pos = pc.base + n;
    const uint nk = pos + 1u;
    const uint kv = h / (pc.hQ / pc.hKV);
    const uint qfo = n * pc.hQ * 2u * dh;
    const uint qho = n * pc.hQ * dh;

    threadgroup float q[256];
    threadgroup float sc[256];
    threadgroup float red[256];
    threadgroup float m, l;

    if (t < dh) q[t] = qhat[qho + h * dh + t];
    if (t == 0u) { m = -3.4e38f; l = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qs = rsqrt(float(dh));
    float acc = 0.0f;
    for (uint p0 = 0u; p0 < nk; p0 += 256u) {
        const uint ts = min(256u, nk - p0);

        if (t < ts) {
            float score = 0.0f;
            const ulong kb = (ulong)(kv * pc.tmax + (p0 + t)) * dh;
            for (uint j = 0u; j < dh; ++j) score += q[j] * kv_load(kc, kb + j);
            sc[t] = score * qs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        red[t] = (t < ts) ? sc[t] : -3.4e38f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (t < s) red[t] = max(red[t], red[t + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float oldm = m;
        const float newm = max(oldm, red[0]);
        const float corr = exp(oldm - newm);

        acc *= corr;
        if (t < ts) sc[t] = exp(sc[t] - newm);
        if (t == 0u) l *= corr;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        red[t] = (t < ts) ? sc[t] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (t < s) red[t] += red[t + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (t < dh) {
            for (uint j = 0u; j < ts; ++j)
                acc += sc[j] * kv_load(
                    vc, (ulong)(kv * pc.tmax + (p0 + j)) * dh + t);
        }
        if (t == 0u) { l += red[0]; m = newm; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t < dh) {
        const float o = acc / l;
        const float g = qfull[qfo + h * 2u * dh + dh + t];
        att[qho + h * dh + t] = o * (1.0f / (1.0f + exp(-g)));
    }
}

// -------- MMA flash attention (batched prefill) --------------------------
// Replaces fa_attn_batch's scalar per-thread dots with simdgroup 8x8 MMA and
// query tiling. One threadgroup = one head x a tile of QTM=16 queries; NSGM=8
// simdgroups (256 threads). Online softmax streamed over KBM=64-key blocks.
//   S = Q K^T   (MMA, K^T via transpose-load)  -> masked/scaled -> P
//   O += P V    (MMA)                           with online rescale of O
// O accumulator + softmax state live in threadgroup memory (dh=256 is too fat
// for a register-resident accumulator); tiles are packed dense row-major so
// simdgroup_load strides are contiguous. Same buffers/signature as
// fa_attn_batch (qhat, kc, vc, qfull, att); grid (hQ, 1, ceil(Tn/QTM)).
constant uint QTM = 16u;   // queries per tile (2 MMA row tiles)
constant uint KBM = 64u;   // key block (8 MMA col tiles)
constant uint NSGM = 8u;   // simdgroups per threadgroup (256 threads)

kernel void fa_attn_batch_mma(device const float* qhat  [[buffer(0)]],
                              device const uchar* kc    [[buffer(1)]],
                              device const uchar* vc    [[buffer(2)]],
                              device const float* qfull [[buffer(3)]],
                              device float*       att   [[buffer(4)]],
                              constant FaBPC&     pc    [[buffer(5)]],
                              uint3 tid3  [[thread_position_in_threadgroup]],
                              uint  sgid  [[simdgroup_index_in_threadgroup]],
                              uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint h   = tgpig.x;              // query head
    const uint qt  = tgpig.z;              // query tile
    const uint dh  = pc.dh;                // 256
    const uint hQ  = pc.hQ;
    const uint tmax = pc.tmax;
    const uint kv  = h / (hQ / pc.hKV);    // GQA: KV head
    const uint tid = tid3.x;               // 0..255
    const uint nDT = dh / 8u;              // dim tiles (32)
    const uint qstart = qt * QTM;
    if (qstart >= pc.Tn) return;
    // Keep the online-softmax state in log2 space so both exponentials can use
    // the native fast exp2 path.
    const float qs = rsqrt(float(dh)) * 1.4426950408889634f;

    threadgroup float Otg[QTM * 256];      // [QTM][dh] accumulator
    threadgroup float Stg[QTM * KBM];      // [QTM][KBM] scores then probs
    threadgroup float mrow[QTM], lrow[QTM], corrTg[QTM];

    for (uint i = tid; i < QTM * dh; i += 256u) Otg[i] = 0.0f;
    if (tid < QTM) { mrow[tid] = -3.4e38f; lrow[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // causal key bound: last query in tile is min(qstart+QTM,Tn)-1
    const uint lastq = min(qstart + QTM, pc.Tn) - 1u;
    const uint nk = pc.base + lastq + 1u;  // attend keys 0..base+lastq

    for (uint p0 = 0u; p0 < nk; p0 += KBM) {
        // Every key in blocks ending at/before the first query is causal for
        // all rows. Keep the per-element comparison only for diagonal blocks.
        const bool fullyUnmasked = p0 + KBM <= pc.base + qstart + 1u;

        // ---- Phase A: raw S = Q K^T into Stg (per 8x8 tile) ----
        for (uint tix = sgid; tix < (QTM / 8u) * (KBM / 8u); tix += NSGM) {
            const uint rt = tix / (KBM / 8u);      // 0..1
            const uint ct = tix % (KBM / 8u);      // 0..7
            if (p0 + ct * 8u >= nk) continue;      // whole key-tile masked
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint dt = 0u; dt < nDT; ++dt) {
                simdgroup_float8x8 qf, kf;
                simdgroup_load(qf, qhat + (ulong)(qstart + rt * 8u) * hQ * dh +
                                        (ulong)h * dh + dt * 8u, hQ * dh);
                const ulong ko = (ulong)(kv * tmax + p0 + ct * 8u) * dh + dt * 8u;
                simdgroup_load(kf, (device const float*)kc + ko,
                               dh, ulong2(0, 0), true);                  // K^T
                simdgroup_multiply_accumulate(acc, qf, kf, acc);
            }
            simdgroup_store(acc, Stg + (rt * 8u) * KBM + ct * 8u, KBM);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Phase B1: per-row mask/scale/softmax update, exp -> P ----
        if (tid < QTM) {
            const uint r = tid;
            const uint qidx = qstart + r;
            if (qidx < pc.Tn) {
                const uint qpos = pc.base + qidx;
                const uint kcount = min(KBM, nk - p0);
                float bmax = -3.4e38f;
                if (fullyUnmasked) {
                    for (uint k = 0u; k < KBM; ++k) {
                        const float s = Stg[r * KBM + k] * qs;
                        Stg[r * KBM + k] = s;
                        bmax = max(bmax, s);
                    }
                } else {
                    for (uint k = 0u; k < kcount; ++k) {
                        const float s = p0 + k <= qpos
                                          ? Stg[r * KBM + k] * qs : -3.4e38f;
                        Stg[r * KBM + k] = s;
                        bmax = max(bmax, s);
                    }
                }
                const float oldm = mrow[r];
                const float newm = max(oldm, bmax);
                const float corr = fast::exp2(oldm - newm);
                float rsum = 0.0f;
                for (uint k = 0u; k < kcount; ++k) {
                    const float p = fast::exp2(Stg[r * KBM + k] - newm);
                    Stg[r * KBM + k] = p;
                    rsum += p;
                }
                for (uint k = kcount; k < KBM; ++k) Stg[r * KBM + k] = 0.0f;
                mrow[r] = newm;
                lrow[r] = lrow[r] * corr + rsum;
                corrTg[r] = corr;
            } else {
                for (uint k = 0u; k < KBM; ++k) Stg[r * KBM + k] = 0.0f;
                corrTg[r] = 1.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Phase B2: rescale O by corr (online softmax) ----
        for (uint i = tid; i < QTM * dh; i += 256u) Otg[i] *= corrTg[i / dh];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Phase C: O += P V (per 8x8 O-tile) ----
        for (uint tix = sgid; tix < (QTM / 8u) * nDT; tix += NSGM) {
            const uint rt  = tix / nDT;            // 0..1
            const uint dct = tix % nDT;            // 0..31
            simdgroup_float8x8 acc;
            simdgroup_load(acc, Otg + (rt * 8u) * dh + dct * 8u, dh);
            for (uint kt = 0u; kt < KBM / 8u; ++kt) {
                if (p0 + kt * 8u >= nk) continue;  // P=0 there, skip V read
                simdgroup_float8x8 pf, vf;
                simdgroup_load(pf, Stg + (rt * 8u) * KBM + kt * 8u, KBM);
                const ulong vo = (ulong)(kv * tmax + p0 + kt * 8u) * dh + dct * 8u;
                simdgroup_load(vf, (device const float*)vc + vo, dh);
                simdgroup_multiply_accumulate(acc, pf, vf, acc);
            }
            simdgroup_store(acc, Otg + (rt * 8u) * dh + dct * 8u, dh);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- output: att = (O / l) * sigmoid(gate) ----
    for (uint i = tid; i < QTM * dh; i += 256u) {
        const uint r = i / dh, t = i % dh;
        const uint qidx = qstart + r;
        if (qidx >= pc.Tn) continue;
        const float o = Otg[i] / lrow[r];
        const float g = qfull[(ulong)qidx * hQ * 2u * dh + (ulong)h * 2u * dh + dh + t];
        att[(ulong)qidx * hQ * dh + (ulong)h * dh + t] = o * (1.0f / (1.0f + exp(-g)));
    }
}
