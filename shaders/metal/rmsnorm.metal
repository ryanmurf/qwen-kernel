#include <metal_stdlib>
using namespace metal;

// out[i] = x[i] * w[i] / sqrt(mean(x^2) + eps). One threadgroup (256);
// grid z batches queries. Port of shaders/rmsnorm.comp.

struct RmsPC { uint n; float eps; };

kernel void rmsnorm(device const float* x [[buffer(0)]],
                    device const float* w [[buffer(1)]],
                    device float*       o [[buffer(2)]],
                    constant RmsPC&     pc [[buffer(3)]],
                    uint3 tid3  [[thread_position_in_threadgroup]],
                    uint3 tgpig [[threadgroup_position_in_grid]],
                    uint  sgid  [[simdgroup_index_in_threadgroup]],
                    uint  slid  [[thread_index_in_simdgroup]])
{
    const uint t  = tid3.x;
    const uint ro = tgpig.z * pc.n;

    float ss = 0.0f;
    for (uint i = t; i < pc.n; i += 256u) ss += x[ro + i] * x[ro + i];

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
    for (uint i = t; i < pc.n; i += 256u) o[ro + i] = x[ro + i] * scale * w[i];
}
