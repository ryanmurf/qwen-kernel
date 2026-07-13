#include <metal_stdlib>
using namespace metal;
#include "fa_kv.metal"

// Full-attention pre-step for one token (port of shaders/fa_prep.comp):
//   TG w in [0,hQ):        q head w: RMS(q)·q_norm, rope, -> qhat
//   TG w in [hQ,hQ+hKV):   k head:   RMS(k)·k_norm, rope, -> Kcache[pos]
//   TG w in [hQ+hKV,+hKV): v head:   copy           -> Vcache[pos]
// qfull layout: head h at [h*2*dh, +dh) = q, [+dh, +2dh) = gate (untouched).
// Rope: NeoX pairs (j, j+nRot/2) on first nRot dims from the precomputed
// cos/sin table.

struct FaPC {
    uint pos; uint tmax; uint dh; uint nRot;
    uint hQ; uint hKV; float eps; float freqBase;
};

kernel void fa_prep(device const float* qfull  [[buffer(0)]],
                    device const float* kin    [[buffer(1)]],
                    device const float* vin    [[buffer(2)]],
                    device const float* qn     [[buffer(3)]],
                    device const float* kn     [[buffer(4)]],
                    device float*       qhat   [[buffer(5)]],
                    device uchar*       kc     [[buffer(6)]],
                    device uchar*       vc     [[buffer(7)]],
                    device const float* ropeCS [[buffer(8)]],
                    constant FaPC&      pc     [[buffer(9)]],
                    uint3 tid3  [[thread_position_in_threadgroup]],
                    uint3 tgpig [[threadgroup_position_in_grid]],
                    uint  sgid  [[simdgroup_index_in_threadgroup]],
                    uint  slid  [[thread_index_in_simdgroup]])
{
    const uint w  = tgpig.x;
    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint rq = tgpig.z;
    const uint qfo = rq * pc.hQ * 2u * dh;
    const uint kio = rq * pc.hKV * dh;
    const uint kvo = rq * pc.hKV * pc.tmax * dh;
    const uint qho = rq * pc.hQ * dh;

    threadgroup float sv[256];
    threadgroup float red[8];
    threadgroup float scale;

    if (w >= pc.hQ + pc.hKV) {           // v: plain copy into cache
        const uint h = w - pc.hQ - pc.hKV;
        const ulong io = kvo + (ulong)(h * pc.tmax + pc.pos) * dh + t;
        const float outV = t < dh ? vin[kio + h * dh + t] : 0.0f;
        if (KV_BYTES == 1u) {
            const float mx = simd_max(abs(outV));
            if (slid == 0u) red[sgid] = mx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t == 0u) {
                float amax = red[0];
                for (uint i = 1u; i < 8u; ++i) amax = max(amax, red[i]);
                scale = amax > 0.0f ? amax / 127.0f : 0.0f;
                *((device float*)(vc + (io >> 8u) * KV_Q8_ROW)) = scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t < dh) {
            if (KV_BYTES == 1u) {
                const int qv = scale > 0.0f
                    ? clamp(int(rint(outV / scale)), -127, 127) : 0;
                const ulong rb = (io >> 8u) * KV_Q8_ROW;
                *((device char*)(vc + rb + KV_Q8_DATA + (io & 255u))) = char(qv);
            } else {
                kv_store(vc, io, outV);
            }
        }
        return;
    }

    const bool isQ = w < pc.hQ;
    const uint h = isQ ? w : w - pc.hQ;
    float v = 0.0f;
    if (t < dh) v = isQ ? qfull[qfo + h * 2u * dh + t] : kin[kio + h * dh + t];

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

    // rope on pairs (j, j+nRot/2), j < nRot/2
    const uint half_ = pc.nRot / 2u;
    float outV = 0.0f;
    if (t < dh) {
        if (t < pc.nRot) {
            const uint j = t < half_ ? t : t - half_;
            const uint ci = 2u * (pc.pos * half_ + j);
            const float c = ropeCS[ci], s = ropeCS[ci + 1u];
            const float x0 = sv[j], x1 = sv[j + half_];
            outV = t < half_ ? x0 * c - x1 * s : x0 * s + x1 * c;
        } else {
            outV = sv[t];
        }
        if (isQ) qhat[qho + h * dh + t] = outV;
    }
    if (!isQ && KV_BYTES == 1u) {
        const float mx = simd_max(abs(outV));
        if (slid == 0u) red[sgid] = mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0u) {
            float amax = red[0];
            for (uint i = 1u; i < 8u; ++i) amax = max(amax, red[i]);
            scale = amax > 0.0f ? amax / 127.0f : 0.0f;
            const ulong row = kvo + (ulong)(h * pc.tmax + pc.pos) * dh;
            *((device float*)(kc + (row >> 8u) * KV_Q8_ROW)) = scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!isQ && t < dh) {
        const ulong io = kvo + (ulong)(h * pc.tmax + pc.pos) * dh + t;
        if (KV_BYTES == 1u) {
            const int qv = scale > 0.0f
                ? clamp(int(rint(outV / scale)), -127, 127) : 0;
            const ulong rb = (io >> 8u) * KV_Q8_ROW;
            *((device char*)(kc + rb + KV_Q8_DATA + (io & 255u))) = char(qv);
        } else {
            kv_store(kc, io, outV);
        }
    }
}
