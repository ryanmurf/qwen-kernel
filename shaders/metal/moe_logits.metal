#include <metal_stdlib>
using namespace metal;

// Router logits: logits[e] = gate_inp[e] · x, F32 weights — PLUS the
// shared-expert gate dot (gis · x) as virtual row n_expert, so the
// single-simdgroup select kernel never does a latency-bound dot itself.
// Buffer stride is therefore n_expert+1 per query.
// v2: one SIMDGROUP per row, NSG simdgroups per threadgroup, simd_sum only.

#include "moe_common.metal"

constant uint NSG [[function_constant(0)]];

kernel void moe_logits(device const float* gi     [[buffer(0)]],
                       device const float* gis    [[buffer(1)]],
                       device const float* x      [[buffer(2)]],
                       device float*       logits [[buffer(3)]],
                       constant MoePC&     pc     [[buffer(4)]],
                       uint3 tgpig [[threadgroup_position_in_grid]],
                       uint  sgid  [[simdgroup_index_in_threadgroup]],
                       uint  slid  [[thread_index_in_simdgroup]])
{
    const uint e  = tgpig.x * NSG + sgid;
    const uint rq = tgpig.z;
    if (e > pc.n_expert) return;

    device const float* row = e < pc.n_expert ? gi + (ulong)e * pc.n_embd : gis;
    device const float4* g4 = (device const float4*)row;
    device const float4* x4 = (device const float4*)(x + rq * pc.n_embd);
    const uint n4 = pc.n_embd / 4u;

    float acc = 0.0f;
    for (uint k = slid; k < n4; k += 32u) acc += dot(g4[k], x4[k]);
    const float s = simd_sum(acc);
    if (slid == 0u) logits[rq * (pc.n_expert + 1u) + e] = s;
}
