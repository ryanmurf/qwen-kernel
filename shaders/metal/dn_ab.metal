#include <metal_stdlib>
using namespace metal;

// Deltanet per-head decay + write strength for one token:
//   gb[h]      = ssm_a[h] * softplus((ssm_alpha[h]·x) + dt_bias[h])   (log-decay)
//   gb[hv + h] = sigmoid(ssm_beta[h]·x)                               (beta)
// One SIMDGROUP per output (w < hv -> alpha head w; else beta head w-hv);
// NSG simdgroups per threadgroup; grid z batches queries.
// Port of shaders/dn_ab.comp.

constant uint NSG [[function_constant(0)]];

struct AbPC { uint n; uint hv; };

kernel void dn_ab(device const float* x      [[buffer(0)]],
                  device const float* alphaW [[buffer(1)]],
                  device const float* betaW  [[buffer(2)]],
                  device const float* dtBias [[buffer(3)]],
                  device const float* aVec   [[buffer(4)]],
                  device float*       gb     [[buffer(5)]],
                  constant AbPC&      pc     [[buffer(6)]],
                  uint3 tgpig [[threadgroup_position_in_grid]],
                  uint  sgid  [[simdgroup_index_in_threadgroup]],
                  uint  slid  [[thread_index_in_simdgroup]])
{
    const uint w  = tgpig.x * NSG + sgid;
    const uint rq = tgpig.z;
    if (w >= 2u * pc.hv) return;
    const uint xo = rq * pc.n;
    const uint go = rq * 2u * pc.hv;
    const bool isBeta = w >= pc.hv;
    const uint h = isBeta ? w - pc.hv : w;

    device const float4* wp = (device const float4*)((isBeta ? betaW : alphaW) + (ulong)h * pc.n);
    device const float4* xp = (device const float4*)(x + xo);
    float acc = 0.0f;
    for (uint k = slid; k < pc.n / 4u; k += 32u) acc += dot(wp[k], xp[k]);
    const float d = simd_sum(acc);

    if (slid == 0u) {
        if (isBeta) {
            gb[go + pc.hv + h] = 1.0f / (1.0f + exp(-d));
        } else {
            const float v = d + dtBias[h];
            const float sp = v > 20.0f ? v : log(1.0f + exp(v));
            gb[go + h] = aVec[h] * sp;
        }
    }
}
