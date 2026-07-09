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

// Fused residual-add + post_attention_norm + router logits: used in the
// decode block chains, where a dedicated add_rmsnorm stage costs a device
// barrier per layer. Every simdgroup recomputes the scalar RMS from the
// SLC-hot xin/attnOut vectors (cheap, register-light); the e==0 simdgroup
// also persists y = xin+attnOut and xn2 = norm for the tail residual and
// the gateup/down consumers.
struct MoeNPC { uint n_embd; uint n_ff; uint n_expert; uint n_used; float eps; };

kernel void moe_logits_addn(device const float* gi      [[buffer(0)]],
                            device const float* gis     [[buffer(1)]],
                            device const float* xin     [[buffer(2)]],
                            device const float* attnOut [[buffer(3)]],
                            device const float* pn      [[buffer(4)]],
                            device float*       logits  [[buffer(5)]],
                            device float*       y       [[buffer(6)]],
                            device float*       xn2     [[buffer(7)]],
                            constant MoeNPC&    pc      [[buffer(8)]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint  sgid  [[simdgroup_index_in_threadgroup]],
                            uint  slid  [[thread_index_in_simdgroup]])
{
    const uint e  = tgpig.x * NSG + sgid;
    const uint rq = tgpig.z;
    if (e > pc.n_expert) return;
    const uint ro = rq * pc.n_embd;
    const uint n4 = pc.n_embd / 4u;

    device const float4* xi4 = (device const float4*)(xin + ro);
    device const float4* ao4 = (device const float4*)(attnOut + ro);
    device const float4* pn4 = (device const float4*)pn;

    float ss = 0.0f;
    for (uint k = slid; k < n4; k += 32u) {
        const float4 v = xi4[k] + ao4[k];
        ss += dot(v, v);
    }
    const float scale = rsqrt(simd_sum(ss) / float(pc.n_embd) + pc.eps);

    if (e == 0u) {
        device float4* y4 = (device float4*)(y + ro);
        device float4* xn4 = (device float4*)(xn2 + ro);
        for (uint k = slid; k < n4; k += 32u) {
            const float4 v = xi4[k] + ao4[k];
            y4[k] = v;
            xn4[k] = v * scale * pn4[k];
        }
    }

    device const float4* g4 = (device const float4*)
        (e < pc.n_expert ? gi + (ulong)e * pc.n_embd : gis);
    float acc = 0.0f;
    for (uint k = slid; k < n4; k += 32u)
        acc += dot(g4[k], (xi4[k] + ao4[k]) * pn4[k]);
    const float lg = simd_sum(acc) * scale;
    if (slid == 0u) logits[rq * (pc.n_expert + 1u) + e] = lg;
}
