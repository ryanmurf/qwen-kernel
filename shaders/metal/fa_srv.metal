#include <metal_stdlib>
using namespace metal;
#include "fa_kv.metal"

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
                        device uchar*       kc      [[buffer(6)]],
                        device uchar*       vc      [[buffer(7)]],
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

    threadgroup float sv[256];
    threadgroup float red[8];
    threadgroup float scale;

    if (w >= pc.hQ + pc.hKV) {           // v: plain copy into cache
        const uint h = w - pc.hQ - pc.hKV;
        const ulong io = kvo + (ulong)(h * pc.tmax + pos) * dh + t;
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

    const uint half_ = pc.nRot / 2u;
    float outV = 0.0f;
    if (t < dh) {
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
    }
    if (!isQ && KV_BYTES == 1u) {
        const float mx = simd_max(abs(outV));
        if (slid == 0u) red[sgid] = mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0u) {
            float amax = red[0];
            for (uint i = 1u; i < 8u; ++i) amax = max(amax, red[i]);
            scale = amax > 0.0f ? amax / 127.0f : 0.0f;
            const ulong row = kvo + (ulong)(h * pc.tmax + pos) * dh;
            *((device float*)(kc + (row >> 8u) * KV_Q8_ROW)) = scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!isQ && t < dh) {
        const ulong io = kvo + (ulong)(h * pc.tmax + pos) * dh + t;
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

kernel void fa_attn_srv(device const float* qhat    [[buffer(0)]],
                        device const uchar* kc      [[buffer(1)]],
                        device const uchar* vc      [[buffer(2)]],
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
            const ulong kb = kvo + (ulong)(kv * pc.tmax + (p0 + t)) * dh;
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
                    vc, kvo + (ulong)(kv * pc.tmax + (p0 + j)) * dh + t);
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

// Split-K (flash-decoding) variant of fa_attn_srv. Each split threadgroup
// processes one 256-key chunk and leaves an unnormalized (acc, m, l) partial
// for fa_attn_srv_reduce to merge.
constant uint CK = 256u;

struct FaSplitPC {
    uint pos; uint tmax; uint dh; uint nRot;
    uint hQ; uint hKV; float eps; float freqBase;
    uint maxChunks;
};

kernel void fa_attn_srv_split(device const float* qhat    [[buffer(0)]],
                              device const uchar* kc      [[buffer(1)]],
                              device const uchar* vc      [[buffer(2)]],
                              device float*       partial [[buffer(3)]],
                              device const uint*  slotPos [[buffer(4)]],
                              constant FaSplitPC& pc      [[buffer(5)]],
                              uint3 tid3  [[thread_position_in_threadgroup]],
                              uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint h  = tgpig.x;
    const uint c  = tgpig.y;
    const uint rq = tgpig.z;
    const uint p0 = c * CK;
    const uint n  = slotPos[rq] + 1u;
    if (p0 >= n) return;

    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint ts = min(CK, n - p0);
    const uint kv = h / (pc.hQ / pc.hKV);
    const uint kvo = rq * pc.hKV * pc.tmax * dh;
    const uint qho = rq * pc.hQ * dh;

    threadgroup float q[256];
    threadgroup float sc[256];
    threadgroup float red[256];

    if (t < dh) q[t] = qhat[qho + h * dh + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qs = rsqrt(float(dh));
    if (t < ts) {
        float score = 0.0f;
        const ulong kb = kvo + (ulong)(kv * pc.tmax + (p0 + t)) * dh;
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

    const float mLocal = red[0];
    if (t < ts) sc[t] = exp(sc[t] - mLocal);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    red[t] = (t < ts) ? sc[t] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) red[t] += red[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint partialStride = dh + 2u;
    const uint partialBase = ((rq * pc.hQ + h) * pc.maxChunks + c) * partialStride;
    if (t < dh) {
        float accLocal = 0.0f;
        for (uint j = 0u; j < ts; ++j) {
            accLocal += sc[j] * kv_load(
                vc, kvo + (ulong)(kv * pc.tmax + (p0 + j)) * dh + t);
        }
        partial[partialBase + t] = accLocal;
    }
    if (t == 0u) {
        partial[partialBase + dh] = mLocal;
        partial[partialBase + dh + 1u] = red[0];
    }
}

// Q8-row probe specialized for split decode.  One lane owns one key row, so
// it loads each K/V scale exactly once.  The V scales are shared through
// threadgroup memory instead of being reread by every output-dimension lane.
kernel void fa_attn_srv_split_q8(device const float* qhat    [[buffer(0)]],
                                 device const uchar* kc      [[buffer(1)]],
                                 device const uchar* vc      [[buffer(2)]],
                                 device float*       partial [[buffer(3)]],
                                 device const uint*  slotPos [[buffer(4)]],
                                 constant FaSplitPC& pc      [[buffer(5)]],
                                 uint3 tid3  [[thread_position_in_threadgroup]],
                                 uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint h  = tgpig.x;
    const uint c  = tgpig.y;
    const uint rq = tgpig.z;
    const uint p0 = c * CK;
    const uint n  = slotPos[rq] + 1u;
    if (p0 >= n) return;

    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint ts = min(CK, n - p0);
    const uint kv = h / (pc.hQ / pc.hKV);
    const uint qho = rq * pc.hQ * dh;
    const ulong row0 = ((ulong)rq * pc.hKV + kv) * pc.tmax + p0;

    threadgroup float q[256];
    threadgroup float sc[256];
    threadgroup float red[256];
    threadgroup float vd[256];

    if (t < dh) q[t] = qhat[qho + h * dh + t];
    float kd = 0.0f;
    if (t < ts) {
        const ulong rb = (row0 + t) * KV_Q8_ROW;
        kd = *((device const float*)(kc + rb));
        vd[t] = *((device const float*)(vc + rb));
    } else {
        vd[t] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qs = rsqrt(float(dh));
    if (t < ts) {
        const ulong rb = (row0 + t) * KV_Q8_ROW;
        device const char* kq = (device const char*)(kc + rb + KV_Q8_DATA);
        float score = 0.0f;
        for (uint j = 0u; j < dh; j += 4u) {
            const char4 x = *((device const char4*)(kq + j));
            score += q[j] * (kd * float(x.x));
            score += q[j + 1u] * (kd * float(x.y));
            score += q[j + 2u] * (kd * float(x.z));
            score += q[j + 3u] * (kd * float(x.w));
        }
        sc[t] = score * qs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    red[t] = (t < ts) ? sc[t] : -3.4e38f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) red[t] = max(red[t], red[t + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float mLocal = red[0];
    if (t < ts) sc[t] = exp(sc[t] - mLocal);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    red[t] = (t < ts) ? sc[t] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) red[t] += red[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint partialStride = dh + 2u;
    const uint partialBase = ((rq * pc.hQ + h) * pc.maxChunks + c) * partialStride;
    if (t < dh) {
        float accLocal = 0.0f;
        device const char* vp =
            (device const char*)(vc + row0 * KV_Q8_ROW + KV_Q8_DATA + t);
        for (uint j = 0u; j < ts; ++j) {
            accLocal += sc[j] * (vd[j] * float(*vp));
            vp += KV_Q8_ROW;
        }
        partial[partialBase + t] = accLocal;
    }
    if (t == 0u) {
        partial[partialBase + dh] = mLocal;
        partial[partialBase + dh + 1u] = red[0];
    }
}

// Two adjacent Q heads share one KV head and therefore one Q8 row stream.
// Retain independent score/softmax/partial arithmetic while loading each K/V
// byte once for both heads.
kernel void fa_attn_srv_split_q8_gqa2(device const float* qhat    [[buffer(0)]],
                                      device const uchar* kc      [[buffer(1)]],
                                      device const uchar* vc      [[buffer(2)]],
                                      device float*       partial [[buffer(3)]],
                                      device const uint*  slotPos [[buffer(4)]],
                                      constant FaSplitPC& pc      [[buffer(5)]],
                                      uint3 tid3  [[thread_position_in_threadgroup]],
                                      uint3 tgpig [[threadgroup_position_in_grid]])
{
    constexpr uint HG = 2u;
    const uint h0 = tgpig.x * HG;
    const uint c  = tgpig.y;
    const uint rq = tgpig.z;
    const uint p0 = c * CK;
    const uint n  = slotPos[rq] + 1u;
    if (p0 >= n) return;
    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint ts = min(CK, n - p0);
    const uint kv = h0 / (pc.hQ / pc.hKV);
    const uint qho = rq * pc.hQ * dh;
    const ulong row0 = ((ulong)rq * pc.hKV + kv) * pc.tmax + p0;

    threadgroup float q[HG * 256u];
    threadgroup float sc[HG * 256u];
    threadgroup float red[256];
    threadgroup float vd[256];
    threadgroup float ml[HG], ll[HG];
    if (t < dh) {
        q[t] = qhat[qho + h0 * dh + t];
        q[dh + t] = qhat[qho + (h0 + 1u) * dh + t];
    }
    float kd = 0.0f;
    if (t < ts) {
        const ulong rb = (row0 + t) * KV_Q8_ROW;
        kd = *((device const float*)(kc + rb));
        vd[t] = *((device const float*)(vc + rb));
    } else {
        vd[t] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qs = rsqrt(float(dh));
    if (t < ts) {
        const ulong rb = (row0 + t) * KV_Q8_ROW;
        device const char* kq = (device const char*)(kc + rb + KV_Q8_DATA);
        float s0 = 0.0f, s1 = 0.0f;
        for (uint j = 0u; j < dh; j += 4u) {
            const char4 x = *((device const char4*)(kq + j));
            const float x0 = kd * float(x.x), x1 = kd * float(x.y);
            const float x2 = kd * float(x.z), x3 = kd * float(x.w);
            s0 += q[j] * x0;       s1 += q[dh + j] * x0;
            s0 += q[j + 1u] * x1;  s1 += q[dh + j + 1u] * x1;
            s0 += q[j + 2u] * x2;  s1 += q[dh + j + 2u] * x2;
            s0 += q[j + 3u] * x3;  s1 += q[dh + j + 3u] * x3;
        }
        sc[t] = s0 * qs;
        sc[dh + t] = s1 * qs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint hr = 0u; hr < HG; ++hr) {
        const uint sb = hr * dh;
        red[t] = (t < ts) ? sc[sb + t] : -3.4e38f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (t < s) red[t] = max(red[t], red[t + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0u) ml[hr] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < ts) sc[sb + t] = exp(sc[sb + t] - ml[hr]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[t] = (t < ts) ? sc[sb + t] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (t < s) red[t] += red[t + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0u) ll[hr] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint partialStride = dh + 2u;
    if (t < dh) {
        float a0 = 0.0f, a1 = 0.0f;
        device const char* vp =
            (device const char*)(vc + row0 * KV_Q8_ROW + KV_Q8_DATA + t);
        for (uint j = 0u; j < ts; ++j) {
            const float vv = vd[j] * float(*vp);
            a0 += sc[j] * vv;
            a1 += sc[dh + j] * vv;
            vp += KV_Q8_ROW;
        }
        const uint pb0 = ((rq * pc.hQ + h0) * pc.maxChunks + c) * partialStride;
        const uint pb1 = pb0 + pc.maxChunks * partialStride;
        partial[pb0 + t] = a0;
        partial[pb1 + t] = a1;
    }
    if (t == 0u) {
        const uint pb0 = ((rq * pc.hQ + h0) * pc.maxChunks + c) * partialStride;
        const uint pb1 = pb0 + pc.maxChunks * partialStride;
        partial[pb0 + dh] = ml[0]; partial[pb0 + dh + 1u] = ll[0];
        partial[pb1 + dh] = ml[1]; partial[pb1 + dh + 1u] = ll[1];
    }
}

kernel void fa_attn_srv_split_q8_gqa4(device const float* qhat    [[buffer(0)]],
                                      device const uchar* kc      [[buffer(1)]],
                                      device const uchar* vc      [[buffer(2)]],
                                      device float*       partial [[buffer(3)]],
                                      device const uint*  slotPos [[buffer(4)]],
                                      constant FaSplitPC& pc      [[buffer(5)]],
                                      uint3 tid3  [[thread_position_in_threadgroup]],
                                      uint3 tgpig [[threadgroup_position_in_grid]])
{
    constexpr uint HG = 4u;
    const uint h0 = tgpig.x * HG;
    const uint c = tgpig.y, rq = tgpig.z, p0 = c * CK;
    const uint n = slotPos[rq] + 1u;
    if (p0 >= n) return;
    const uint t = tid3.x, dh = pc.dh, ts = min(CK, n - p0);
    const uint kv = h0 / (pc.hQ / pc.hKV), qho = rq * pc.hQ * dh;
    const ulong row0 = ((ulong)rq * pc.hKV + kv) * pc.tmax + p0;

    threadgroup float q[HG * 256u];
    threadgroup float sc[HG * 256u];
    threadgroup float red[256], vd[256], ml[HG], ll[HG];
    if (t < dh)
        for (uint hr = 0u; hr < HG; ++hr)
            q[hr * dh + t] = qhat[qho + (h0 + hr) * dh + t];
    float kd = 0.0f;
    if (t < ts) {
        const ulong rb = (row0 + t) * KV_Q8_ROW;
        kd = *((device const float*)(kc + rb));
        vd[t] = *((device const float*)(vc + rb));
    } else vd[t] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qs = rsqrt(float(dh));
    if (t < ts) {
        const ulong rb = (row0 + t) * KV_Q8_ROW;
        device const char* kq = (device const char*)(kc + rb + KV_Q8_DATA);
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (uint j = 0u; j < dh; j += 4u) {
            const char4 x = *((device const char4*)(kq + j));
            const float x0 = kd * float(x.x), x1 = kd * float(x.y);
            const float x2 = kd * float(x.z), x3 = kd * float(x.w);
#define Q8_GQA4_ACC(J, X)                                                       \
            s0 += q[J] * X; s1 += q[dh + J] * X;                              \
            s2 += q[2u * dh + J] * X; s3 += q[3u * dh + J] * X
            Q8_GQA4_ACC(j, x0); Q8_GQA4_ACC(j + 1u, x1);
            Q8_GQA4_ACC(j + 2u, x2); Q8_GQA4_ACC(j + 3u, x3);
#undef Q8_GQA4_ACC
        }
        sc[t] = s0 * qs; sc[dh + t] = s1 * qs;
        sc[2u * dh + t] = s2 * qs; sc[3u * dh + t] = s3 * qs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint hr = 0u; hr < HG; ++hr) {
        const uint sb = hr * dh;
        red[t] = t < ts ? sc[sb + t] : -3.4e38f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (t < s) red[t] = max(red[t], red[t + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0u) ml[hr] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < ts) sc[sb + t] = exp(sc[sb + t] - ml[hr]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[t] = t < ts ? sc[sb + t] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (t < s) red[t] += red[t + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0u) ll[hr] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint ps = dh + 2u;
    if (t < dh) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        device const char* vp =
            (device const char*)(vc + row0 * KV_Q8_ROW + KV_Q8_DATA + t);
        for (uint j = 0u; j < ts; ++j) {
            const float vv = vd[j] * float(*vp);
            a0 += sc[j] * vv; a1 += sc[dh + j] * vv;
            a2 += sc[2u * dh + j] * vv; a3 += sc[3u * dh + j] * vv;
            vp += KV_Q8_ROW;
        }
        const uint pb = ((rq * pc.hQ + h0) * pc.maxChunks + c) * ps;
        const uint hs = pc.maxChunks * ps;
        partial[pb + t] = a0; partial[pb + hs + t] = a1;
        partial[pb + 2u * hs + t] = a2; partial[pb + 3u * hs + t] = a3;
    }
    if (t == 0u) {
        const uint pb = ((rq * pc.hQ + h0) * pc.maxChunks + c) * ps;
        const uint hs = pc.maxChunks * ps;
        for (uint hr = 0u; hr < HG; ++hr) {
            partial[pb + hr * hs + dh] = ml[hr];
            partial[pb + hr * hs + dh + 1u] = ll[hr];
        }
    }
}

kernel void fa_attn_srv_split_q8_gqa8(device const float* qhat    [[buffer(0)]],
                                      device const uchar* kc      [[buffer(1)]],
                                      device const uchar* vc      [[buffer(2)]],
                                      device float*       partial [[buffer(3)]],
                                      device const uint*  slotPos [[buffer(4)]],
                                      constant FaSplitPC& pc      [[buffer(5)]],
                                      uint3 tid3  [[thread_position_in_threadgroup]],
                                      uint3 tgpig [[threadgroup_position_in_grid]])
{
    constexpr uint HG = 8u;
    const uint h0 = tgpig.x * HG;
    const uint c = tgpig.y, rq = tgpig.z, p0 = c * CK;
    const uint n = slotPos[rq] + 1u;
    if (p0 >= n) return;
    const uint t = tid3.x, dh = pc.dh, ts = min(CK, n - p0);
    const uint kv = h0 / (pc.hQ / pc.hKV), qho = rq * pc.hQ * dh;
    const ulong row0 = ((ulong)rq * pc.hKV + kv) * pc.tmax + p0;

    threadgroup float q[HG * 256u];
    threadgroup float sc[HG * 256u];
    threadgroup float red[256], vd[256], ml[HG], ll[HG];
    if (t < dh)
        for (uint hr = 0u; hr < HG; ++hr)
            q[hr * dh + t] = qhat[qho + (h0 + hr) * dh + t];
    float kd = 0.0f;
    if (t < ts) {
        const ulong rb = (row0 + t) * KV_Q8_ROW;
        kd = *((device const float*)(kc + rb));
        vd[t] = *((device const float*)(vc + rb));
    } else vd[t] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qs = rsqrt(float(dh));
    if (t < ts) {
        const ulong rb = (row0 + t) * KV_Q8_ROW;
        device const char* kq = (device const char*)(kc + rb + KV_Q8_DATA);
        float score[HG];
        for (uint hr = 0u; hr < HG; ++hr) score[hr] = 0.0f;
        for (uint j = 0u; j < dh; j += 4u) {
            const char4 x = *((device const char4*)(kq + j));
            const float xv[4] = {kd * float(x.x), kd * float(x.y),
                                 kd * float(x.z), kd * float(x.w)};
            for (uint k = 0u; k < 4u; ++k)
                for (uint hr = 0u; hr < HG; ++hr)
                    score[hr] += q[hr * dh + j + k] * xv[k];
        }
        for (uint hr = 0u; hr < HG; ++hr) sc[hr * dh + t] = score[hr] * qs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint hr = 0u; hr < HG; ++hr) {
        const uint sb = hr * dh;
        red[t] = t < ts ? sc[sb + t] : -3.4e38f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (t < s) red[t] = max(red[t], red[t + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0u) ml[hr] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < ts) sc[sb + t] = exp(sc[sb + t] - ml[hr]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[t] = t < ts ? sc[sb + t] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (t < s) red[t] += red[t + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0u) ll[hr] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint ps = dh + 2u, hs = pc.maxChunks * ps;
    if (t < dh) {
        float acc[HG];
        for (uint hr = 0u; hr < HG; ++hr) acc[hr] = 0.0f;
        device const char* vp =
            (device const char*)(vc + row0 * KV_Q8_ROW + KV_Q8_DATA + t);
        for (uint j = 0u; j < ts; ++j) {
            const float vv = vd[j] * float(*vp);
            for (uint hr = 0u; hr < HG; ++hr) acc[hr] += sc[hr * dh + j] * vv;
            vp += KV_Q8_ROW;
        }
        const uint pb = ((rq * pc.hQ + h0) * pc.maxChunks + c) * ps;
        for (uint hr = 0u; hr < HG; ++hr) partial[pb + hr * hs + t] = acc[hr];
    }
    if (t == 0u) {
        const uint pb = ((rq * pc.hQ + h0) * pc.maxChunks + c) * ps;
        for (uint hr = 0u; hr < HG; ++hr) {
            partial[pb + hr * hs + dh] = ml[hr];
            partial[pb + hr * hs + dh + 1u] = ll[hr];
        }
    }
}

// Bit-exact compact grid for highly skewed batches. Unlike the rectangular
// split grid, every work item names a live (slot, 256-key chunk) pair.
kernel void fa_attn_srv_split_compact(device const float* qhat    [[buffer(0)]],
                                      device const uchar* kc      [[buffer(1)]],
                                      device const uchar* vc      [[buffer(2)]],
                                      device float*       partial [[buffer(3)]],
                                      device const uint*  slotPos [[buffer(4)]],
                                      device const uint*  work    [[buffer(5)]],
                                      constant FaSplitPC& pc      [[buffer(6)]],
                                      uint3 tid3  [[thread_position_in_threadgroup]],
                                      uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint item = work[tgpig.y];
    const uint h  = tgpig.x;
    const uint c  = item & 0xffffu;
    const uint rq = item >> 16u;
    const uint p0 = c * CK;
    const uint n  = slotPos[rq] + 1u;

    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint ts = min(CK, n - p0);
    const uint kv = h / (pc.hQ / pc.hKV);
    const uint kvo = rq * pc.hKV * pc.tmax * dh;
    const uint qho = rq * pc.hQ * dh;

    threadgroup float q[256];
    threadgroup float sc[256];
    threadgroup float red[256];

    if (t < dh) q[t] = qhat[qho + h * dh + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qs = rsqrt(float(dh));
    if (t < ts) {
        float score = 0.0f;
        const ulong kb = kvo + (ulong)(kv * pc.tmax + (p0 + t)) * dh;
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

    const float mLocal = red[0];
    if (t < ts) sc[t] = exp(sc[t] - mLocal);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    red[t] = (t < ts) ? sc[t] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) red[t] += red[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint partialStride = dh + 2u;
    const uint partialBase = ((rq * pc.hQ + h) * pc.maxChunks + c) * partialStride;
    if (t < dh) {
        float accLocal = 0.0f;
        for (uint j = 0u; j < ts; ++j) {
            accLocal += sc[j] * kv_load(
                vc, kvo + (ulong)(kv * pc.tmax + (p0 + j)) * dh + t);
        }
        partial[partialBase + t] = accLocal;
    }
    if (t == 0u) {
        partial[partialBase + dh] = mLocal;
        partial[partialBase + dh + 1u] = red[0];
    }
}

// Bit-exact producer coarsening. One TG processes a balanced run of original
// 256-key chunks, but emits one independent partial per chunk. The shipping
// reducer therefore sees byte-identical inputs in the same order; only q
// loading and producer scheduling are amortized.
kernel void fa_attn_srv_split_multi(device const float* qhat    [[buffer(0)]],
                                    device const uchar* kc      [[buffer(1)]],
                                    device const uchar* vc      [[buffer(2)]],
                                    device float*       partial [[buffer(3)]],
                                    device const uint*  slotPos [[buffer(4)]],
                                    device const uint2* work    [[buffer(5)]],
                                    constant FaSplitPC& pc      [[buffer(6)]],
                                    uint3 tid3  [[thread_position_in_threadgroup]],
                                    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint2 item = work[tgpig.y];
    const uint h  = tgpig.x;
    const uint rq = item.x >> 16u;
    const uint first = item.y >> 16u;
    const uint count = item.y & 0xffffu;
    const uint n  = slotPos[rq] + 1u;

    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint kv = h / (pc.hQ / pc.hKV);
    const uint kvo = rq * pc.hKV * pc.tmax * dh;
    const uint qho = rq * pc.hQ * dh;

    threadgroup float q[256];
    threadgroup float sc[256];
    threadgroup float red[256];

    if (t < dh) q[t] = qhat[qho + h * dh + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qs = rsqrt(float(dh));
    const uint partialStride = dh + 2u;
    for (uint u = 0u; u < count; ++u) {
        const uint c = first + u;
        const uint p0 = c * CK;
        const uint ts = min(CK, n - p0);
        if (t < ts) {
            float score = 0.0f;
            const ulong kb = kvo + (ulong)(kv * pc.tmax + (p0 + t)) * dh;
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

        const float mLocal = red[0];
        if (t < ts) sc[t] = exp(sc[t] - mLocal);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        red[t] = (t < ts) ? sc[t] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (t < s) red[t] += red[t + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const uint partialBase =
            ((rq * pc.hQ + h) * pc.maxChunks + c) * partialStride;
        if (t < dh) {
            float accLocal = 0.0f;
            for (uint j = 0u; j < ts; ++j)
                accLocal += sc[j] * kv_load(
                    vc, kvo + (ulong)(kv * pc.tmax + (p0 + j)) * dh + t);
            partial[partialBase + t] = accLocal;
        }
        if (t == 0u) {
            partial[partialBase + dh] = mLocal;
            partial[partialBase + dh + 1u] = red[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void fa_attn_srv_reduce(device const float* partial [[buffer(0)]],
                               device const float* qfull   [[buffer(1)]],
                               device float*       att     [[buffer(2)]],
                               device const uint*  slotPos [[buffer(3)]],
                               constant FaSplitPC& pc      [[buffer(4)]],
                               uint3 tid3  [[thread_position_in_threadgroup]],
                               uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint h  = tgpig.x;
    const uint t  = tid3.x;
    const uint rq = tgpig.z;
    const uint dh = pc.dh;
    const uint n  = slotPos[rq] + 1u;
    const uint nc = (n + CK - 1u) / CK;
    const uint partialStride = dh + 2u;
    const uint partialBase = (rq * pc.hQ + h) * pc.maxChunks * partialStride;

    threadgroup float red[256];
    threadgroup float globalM;
    threadgroup float globalL;

    float laneM = -3.4e38f;
    for (uint c = t; c < nc; c += 256u) {
        laneM = max(laneM, partial[partialBase + c * partialStride + dh]);
    }
    red[t] = laneM;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) red[t] = max(red[t], red[t + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) globalM = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float laneL = 0.0f;
    for (uint c = t; c < nc; c += 256u) {
        const uint cb = partialBase + c * partialStride;
        laneL += partial[cb + dh + 1u] * exp(partial[cb + dh] - globalM);
    }
    red[t] = laneL;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) red[t] += red[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) globalL = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t < dh) {
        float numerator = 0.0f;
        for (uint c = 0u; c < nc; ++c) {
            const uint cb = partialBase + c * partialStride;
            numerator += partial[cb + t] * exp(partial[cb + dh] - globalM);
        }
        const uint qfo = rq * pc.hQ * 2u * dh;
        const uint qho = rq * pc.hQ * dh;
        const float g = qfull[qfo + h * 2u * dh + dh + t];
        att[qho + h * dh + t] = (numerator / globalL) *
                                (1.0f / (1.0f + exp(-g)));
    }
}
