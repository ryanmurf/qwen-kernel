#include <metal_stdlib>
using namespace metal;

// Gated delta rule, one token, one THREADGROUP per v-head h (thread j =
// state row j):
//   S ← exp(g_h)·S;  sk_j = Σ_i S[j,i]·k_i;  d_j = β_h(v_j − sk_j)
//   S[j,i] ← S[j,i] + k_i·d_j;  o_j = Σ_i S[j,i]·q_i · 1/√dState
// GQA: v-head h uses q/k head (h % hK). conv layout: q [0, hK*dS), k
// [hK*dS, 2*hK*dS), v [2*hK*dS, ...). Port of shaders/dn_step.comp, with
// the q/k L2 normalization (dn_qknorm.comp) FUSED into the staging pass:
// the staged head is fully visible to simdgroup 0, so the norm costs one
// simd_sum instead of a separate dispatch + device barrier. Redundant
// per-v-head recompute of the shared k-head norm is free vs the barrier.

struct StepPC { uint dState; uint hK; uint hV; float eps; };

kernel void dn_step(device const float* conv [[buffer(0)]],
                    device const float* gb   [[buffer(1)]],
                    device float4*      s4   [[buffer(2)]],
                    device float*       o    [[buffer(3)]],
                    constant StepPC&    pc   [[buffer(4)]],
                    uint3 tid3  [[thread_position_in_threadgroup]],
                    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint h  = tgpig.x;
    const uint j  = tid3.x;
    const uint rq = tgpig.z;
    const uint dS = pc.dState;
    const uint nv = dS / 4u;
    const float eps = pc.eps;

    const uint kh    = h % pc.hK;
    const uint co    = rq * (2u * pc.hK + pc.hV) * dS;
    const uint qBase = co + kh * dS;
    const uint kBase = co + pc.hK * dS + kh * dS;
    const uint vBase = co + 2u * pc.hK * dS + h * dS;

    threadgroup float4 qs4[64];
    threadgroup float4 ks4[64];
    const float qScale = rsqrt(float(dS));
    if (j < nv) {   // nv <= 32: staging threads are exactly simdgroup 0
        const float4 q4 = ((device const float4*)(conv + qBase))[j];
        const float4 k4 = ((device const float4*)(conv + kBase))[j];
        const float ssq = simd_sum(dot(q4, q4));
        const float ssk = simd_sum(dot(k4, k4));
        qs4[j] = q4 * (qScale / max(sqrt(ssq), eps));
        ks4[j] = k4 * (1.0f / max(sqrt(ssk), eps));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (j >= dS) return;

    const float decay = exp(gb[rq * 2u * pc.hV + h]);
    const float beta  = gb[rq * 2u * pc.hV + pc.hV + h];
    const float vj    = conv[vBase + j];
    const ulong row   = (ulong)rq * pc.hV * dS * nv + ((ulong)h * dS + j) * nv;

    float sk = 0.0f;
    for (uint i = 0u; i < nv; ++i) sk += dot(s4[row + i], ks4[i]);
    sk *= decay;

    const float dj = beta * (vj - sk);

    float oj = 0.0f;
    for (uint i = 0u; i < nv; ++i) {
        const float4 sn = s4[row + i] * decay + ks4[i] * dj;
        s4[row + i] = sn;
        oj += dot(sn, qs4[i]);
    }
    o[rq * pc.hV * dS + h * dS + j] = oj;
}
