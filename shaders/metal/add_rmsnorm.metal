#include <metal_stdlib>
using namespace metal;

// Residual add + RMS norm in one pass (one 256-threadgroup; grid z batches):
//   y[i]  = a[i] + b[i]
//   xn[i] = y[i] * w[i] / sqrt(mean(y^2) + eps)
// Port of shaders/add_rmsnorm.comp.

struct RmsPC { uint n; float eps; };

kernel void add_rmsnorm(device const float* a  [[buffer(0)]],
                        device const float* b  [[buffer(1)]],
                        device const float* w  [[buffer(2)]],
                        device float*       y  [[buffer(3)]],
                        device float*       xn [[buffer(4)]],
                        constant RmsPC&     pc [[buffer(5)]],
                        uint3 tid3  [[thread_position_in_threadgroup]],
                        uint3 tgpig [[threadgroup_position_in_grid]],
                        uint  sgid  [[simdgroup_index_in_threadgroup]],
                        uint  slid  [[thread_index_in_simdgroup]])
{
    const uint t  = tid3.x;
    const uint ro = tgpig.z * pc.n;

    float ss = 0.0f;
    for (uint i = t; i < pc.n; i += 256u) {
        const float v = a[ro + i] + b[ro + i];
        y[ro + i] = v;
        ss += v * v;
    }

    threadgroup float red[8];
    threadgroup float scale;
    const float sg = simd_sum(ss);
    if (slid == 0u) red[sgid] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
        float tot = 0.0f;
        for (uint i = 0u; i < 8u; ++i) tot += red[i];
        scale = 1.0f / sqrt(tot / float(pc.n) + pc.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < pc.n; i += 256u)
        xn[ro + i] = (a[ro + i] + b[ro + i]) * scale * w[i];
}

// Large-slot router path: reproduce simdgroup e=0 of moe_logits_addn exactly,
// persist its scale, and let a following router dispatch reuse the result
// instead of recomputing RMS once per expert row.
kernel void add_rmsnorm_sg(device const float* a      [[buffer(0)]],
                           device const float* b      [[buffer(1)]],
                           device const float* w      [[buffer(2)]],
                           device float*       y      [[buffer(3)]],
                           device float*       xn     [[buffer(4)]],
                           device float*       scales [[buffer(5)]],
                           constant RmsPC&     pc     [[buffer(6)]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           uint  slid  [[thread_index_in_simdgroup]])
{
    const uint ro = tgpig.z * pc.n;
    const uint n4 = pc.n / 4u;
    device const float4* a4 = (device const float4*)(a + ro);
    device const float4* b4 = (device const float4*)(b + ro);
    device const float4* w4 = (device const float4*)w;
    device float4* y4 = (device float4*)(y + ro);
    device float4* xn4 = (device float4*)(xn + ro);
    float ss = 0.0f;
    for (uint k = slid; k < n4; k += 32u) {
        const float4 v = a4[k] + b4[k];
        ss += dot(v, v);
    }
    const float scale = rsqrt(simd_sum(ss) / float(pc.n) + pc.eps);
    for (uint k = slid; k < n4; k += 32u) {
        const float4 v = a4[k] + b4[k];
        y4[k] = v;
        xn4[k] = v * scale * w4[k];
    }
    if (slid == 0u) scales[tgpig.z] = scale;
}
