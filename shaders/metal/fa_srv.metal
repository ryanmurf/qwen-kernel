#include <metal_stdlib>
using namespace metal;

// Serving variants of the attention pair: per-slot positions come from a
// slotPos buffer (slots at different sequence positions batch on grid z),
// and fa_attn_srv uses ONLINE (flash-style) softmax tiled over 256-key
// chunks — no threadgroup cap on context length.
// Ports of shaders/fa_prep_srv.comp / fa_attn_srv.comp.

struct FaPC {
    uint pos; uint tmax; uint dh; uint nRot;
    uint hQ; uint hKV; float eps; float freqBase;
};

kernel void fa_prep_srv(device const float* qfull   [[buffer(0)]],
                        device const float* kin     [[buffer(1)]],
                        device const float* vin     [[buffer(2)]],
                        device const float* qn      [[buffer(3)]],
                        device const float* kn      [[buffer(4)]],
                        device float*       qhat    [[buffer(5)]],
                        device float*       kc      [[buffer(6)]],
                        device float*       vc      [[buffer(7)]],
                        device const float* ropeCS  [[buffer(8)]],
                        device const uint*  slotPos [[buffer(9)]],
                        constant FaPC&      pc      [[buffer(10)]],
                        uint3 tid3  [[thread_position_in_threadgroup]],
                        uint3 tgpig [[threadgroup_position_in_grid]],
                        uint  sgid  [[simdgroup_index_in_threadgroup]],
                        uint  slid  [[thread_index_in_simdgroup]])
{
    const uint w  = tgpig.x;
    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint rq = tgpig.z;
    const uint pos = slotPos[rq];
    const uint qfo = rq * pc.hQ * 2u * dh;
    const uint kio = rq * pc.hKV * dh;
    const uint kvo = rq * pc.hKV * pc.tmax * dh;
    const uint qho = rq * pc.hQ * dh;

    if (w >= pc.hQ + pc.hKV) {           // v: plain copy into cache
        const uint h = w - pc.hQ - pc.hKV;
        if (t < dh) vc[kvo + (h * pc.tmax + pos) * dh + t] = vin[kio + h * dh + t];
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
        else     kc[kvo + (h * pc.tmax + pos) * dh + t] = outV;
    }
}

kernel void fa_attn_srv(device const float* qhat    [[buffer(0)]],
                        device const float* kc      [[buffer(1)]],
                        device const float* vc      [[buffer(2)]],
                        device const float* qfull   [[buffer(3)]],
                        device float*       att     [[buffer(4)]],
                        device const uint*  slotPos [[buffer(5)]],
                        constant FaPC&      pc      [[buffer(6)]],
                        uint3 tid3  [[thread_position_in_threadgroup]],
                        uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint h  = tgpig.x;
    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint rq = tgpig.z;
    const uint n  = slotPos[rq] + 1u;
    const uint kv = h / (pc.hQ / pc.hKV);
    const uint qfo = rq * pc.hQ * 2u * dh;
    const uint kvo = rq * pc.hKV * pc.tmax * dh;
    const uint qho = rq * pc.hQ * dh;

    threadgroup float q[256];
    threadgroup float sc[256];
    threadgroup float red[256];
    threadgroup float m, l;

    if (t < dh) q[t] = qhat[qho + h * dh + t];
    if (t == 0u) { m = -3.4e38f; l = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qs = rsqrt(float(dh));
    float acc = 0.0f;
    for (uint p0 = 0u; p0 < n; p0 += 256u) {
        const uint ts = min(256u, n - p0);

        if (t < ts) {
            float score = 0.0f;
            device const float* kb = kc + kvo + (kv * pc.tmax + (p0 + t)) * dh;
            for (uint j = 0u; j < dh; ++j) score += q[j] * kb[j];
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
                acc += sc[j] * vc[kvo + (kv * pc.tmax + (p0 + j)) * dh + t];
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
