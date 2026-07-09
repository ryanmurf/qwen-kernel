#include <metal_stdlib>
using namespace metal;

// L2-normalize q and k heads of the conv output in place (ggml_l2_norm):
//   head *= 1 / max(sqrt(sum(head^2)), eps)
// One SIMDGROUP per 128-dim head (4 elems per lane), NSG simdgroups per
// threadgroup; grid.x covers 2*hK heads. Port of shaders/dn_qknorm.comp.

constant uint NSG [[function_constant(0)]];

struct QkPC { uint dState; float eps; };

kernel void dn_qknorm(device float*   c  [[buffer(0)]],
                      constant QkPC&  pc [[buffer(1)]],
                      uint3 tgpig [[threadgroup_position_in_grid]],
                      uint  sgid  [[simdgroup_index_in_threadgroup]],
                      uint  slid  [[thread_index_in_simdgroup]])
{
    const uint w = tgpig.x * NSG + sgid;
    device float4* h4 = (device float4*)(c + (ulong)w * pc.dState);
    const uint n4 = pc.dState / 4u;

    float ss = 0.0f;
    for (uint i = slid; i < n4; i += 32u) {
        const float4 v = h4[i];
        ss += dot(v, v);
    }
    const float tot = simd_sum(ss);
    const float scale = 1.0f / max(sqrt(tot), pc.eps);
    for (uint i = slid; i < n4; i += 32u) h4[i] = h4[i] * scale;
}
