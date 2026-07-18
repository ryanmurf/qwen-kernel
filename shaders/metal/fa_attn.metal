#include <metal_stdlib>
using namespace metal;
#include "fa_kv.metal"

// Decode attention over the KV cache for one token, one THREADGROUP per
// q-head (port of shaders/fa_attn.comp):
//   kv-head = h / (hQ/hKV)
//   scores_t = (qhat_h · K[kv,t]) / sqrt(dh),  t <= pos   -> softmax
//   out_j    = sum_t w_t V[kv,t,j]
//   att[h*dh+j] = out_j * sigmoid(qfull[h*2dh + dh + j])   (per-elem gate)
// Positions capped by the threadgroup score array: pos+1 <= 1024.

struct FaPC {
    uint pos; uint tmax; uint dh; uint nRot;
    uint hQ; uint hKV; float eps; float freqBase;
};

kernel void fa_attn(device const float* qhat  [[buffer(0)]],
                    device const uchar* kc    [[buffer(1)]],
                    device const uchar* vc    [[buffer(2)]],
                    device const float* qfull [[buffer(3)]],
                    device float*       att   [[buffer(4)]],
                    constant FaPC&      pc    [[buffer(5)]],
                    uint3 tid3  [[thread_position_in_threadgroup]],
                    uint3 tgpig [[threadgroup_position_in_grid]],
                    uint  sgid  [[simdgroup_index_in_threadgroup]],
                    uint  slid  [[thread_index_in_simdgroup]])
{
    const uint h  = tgpig.x;
    const uint t  = tid3.x;
    const uint dh = pc.dh;
    const uint n  = pc.pos + 1u;
    const uint kv = h / (pc.hQ / pc.hKV);
    const uint rq = tgpig.z;
    const uint qfo = rq * pc.hQ * 2u * dh;
    const uint kvo = rq * pc.hKV * pc.tmax * dh;
    const uint qho = rq * pc.hQ * dh;

    threadgroup float q[256];
    threadgroup float sc[1024];
    threadgroup float red[8];
    threadgroup float bcast;

    if (t < dh) q[t] = qhat[qho + h * dh + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qs = rsqrt(float(dh));
    for (uint p = t; p < n; p += 256u) {
        float s = 0.0f;
        const ulong kb = kvo + (ulong)(kv * pc.tmax + p) * dh;
        for (uint j = 0u; j < dh; ++j) s += q[j] * kv_load(kc, kb + j);
        sc[p] = s * qs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // softmax over sc[0..n)
    float mx = -3.4e38f;
    for (uint p = t; p < n; p += 256u) mx = max(mx, sc[p]);
    mx = simd_max(mx);
    if (slid == 0u) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
        float m = red[0];
        for (uint i = 1u; i < 8u; ++i) m = max(m, red[i]);
        bcast = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = bcast;

    float sum = 0.0f;
    for (uint p = t; p < n; p += 256u) {
        const float e = exp(sc[p] - m);
        sc[p] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (slid == 0u) red[sgid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
        float s = 0.0f;
        for (uint i = 0u; i < 8u; ++i) s += red[i];
        bcast = 1.0f / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = bcast;

    if (t < dh) {
        float o = 0.0f;
        for (uint p = 0u; p < n; ++p)
            o += sc[p] * kv_load(vc, kvo + (ulong)(kv * pc.tmax + p) * dh + t);
        o *= inv;
        const float g = qfull[qfo + h * 2u * dh + dh + t];
        att[qho + h * dh + t] = o * (1.0f / (1.0f + exp(-g)));
    }
}
