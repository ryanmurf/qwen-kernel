#include <metal_stdlib>
using namespace metal;

// Gated RMS norm of the deltanet output:
//   att[h*dS+j] = rmsnorm(o_head_h)[j] * ssm_norm[j] * silu(z[h*dS+j])
// One SIMDGROUP per v-head (4 elems per lane at dS=128), NSG simdgroups
// per threadgroup; grid z batches queries. Port of shaders/dn_gate.comp.

constant uint NSG [[function_constant(0)]];

struct GatePC { uint dState; uint hV; float eps; };

kernel void dn_gate(device const float* o   [[buffer(0)]],
                    device const float* w   [[buffer(1)]],
                    device const float* z   [[buffer(2)]],
                    device float*       att [[buffer(3)]],
                    constant GatePC&    pc  [[buffer(4)]],
                    uint3 tgpig [[threadgroup_position_in_grid]],
                    uint  sgid  [[simdgroup_index_in_threadgroup]],
                    uint  slid  [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x * NSG + sgid;
    if (h >= pc.hV) return;
    const uint base = tgpig.z * pc.hV * pc.dState + h * pc.dState;

    float ss = 0.0f;
    for (uint j = slid; j < pc.dState; j += 32u) {
        const float v = o[base + j];
        ss += v * v;
    }
    const float tot = simd_sum(ss);
    const float scale = 1.0f / sqrt(tot / float(pc.dState) + pc.eps);
    for (uint j = slid; j < pc.dState; j += 32u) {
        const float zv = z[base + j];
        att[base + j] = o[base + j] * scale * w[j] * (zv / (1.0f + exp(-zv)));
    }
}
