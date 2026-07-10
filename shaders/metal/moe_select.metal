#include <metal_stdlib>
using namespace metal;

// Top-n_used expert selection + weights, ONE SIMDGROUP per query.
// v2: the 256-thread barrier-tree version cost ~20 µs/layer in pure
// latency; this one is all simd ops. Each lane owns n_expert/32 logits in
// registers (lane-strided: logit index i*32+lane). Ties resolve to the
// LOWEST expert index, exactly like the tree version and the CPU
// partial_sort reference. NaN/Inf logits are floored so padding can never
// win (see moe_select.comp for the war story).
// Weights = softmax over the selected logits; wShared = sigmoid of the
// gis-dot that moe_logits computed as virtual row n_expert.

#include "moe_common.metal"

kernel void moe_select(device const float* logits [[buffer(0)]],
                       device SelT*        sel    [[buffer(1)]],
                       constant MoePC&     pc     [[buffer(2)]],
                       uint3 tgpig [[threadgroup_position_in_grid]],
                       uint  slid  [[thread_index_in_simdgroup]])
{
    const uint rq = tgpig.z;
    const uint stride = pc.n_expert + 1u;   // moe_logits appends the gis dot

    if (slid == 0u) {
        const float sg = logits[rq * stride + pc.n_expert];
        sel[rq].wShared = 1.0f / (1.0f + exp(-sg));
    }

    // lane-resident logits, sanitized
    const uint perLane = (pc.n_expert + 31u) / 32u;  // 16 for 512 experts
    float v[16];
    for (uint i = 0u; i < perLane && i < 16u; ++i) {
        const uint g = i * 32u + slid;
        float lv = g < pc.n_expert ? logits[rq * stride + g] : -3.4e38f;
        v[i] = (isnan(lv) || isinf(lv)) ? -3.4e38f : lv;
    }

    float wsel[16];
    uint  idsel[16];
    for (uint round = 0u; round < pc.n_used; ++round) {
        float lm = -3.4e38f;
        uint  li = 0u;
        for (uint i = 0u; i < perLane && i < 16u; ++i)
            if (v[i] > lm) { lm = v[i]; li = i; }          // first hit = lowest index
        const float gm = simd_max(lm);
        const uint  g  = li * 32u + slid;                   // global expert index
        const uint  win = simd_min(lm == gm ? g : 0xFFFFu); // lowest index among ties
        if (slid == (win & 31u)) v[win >> 5u] = -3.4e38f;   // winner lane clears it
        wsel[round]  = gm;
        idsel[round] = min(win, pc.n_expert - 1u);          // clamp (all-nonfinite case)
    }

    if (slid == 0u) {
        const float m = wsel[0];                            // first pick has max logit
        float sum = 0.0f;
        for (uint i = 0u; i < pc.n_used; ++i) {
            wsel[i] = exp(wsel[i] - m);
            sum += wsel[i];
        }
        for (uint i = 0u; i < pc.n_used; ++i) {
            sel[rq].ids[i] = idsel[i];
            sel[rq].w[i]   = wsel[i] / sum;
        }
    }
}
