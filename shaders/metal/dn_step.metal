#include <metal_stdlib>
using namespace metal;

// Gated delta rule, one token, one THREADGROUP (= dState threads) per
// q/k head — v3 fuses the whole recurrent path that used to be four
// dispatches (dn_conv -> dn_qknorm -> dn_step -> dn_gate) into one:
//
//   conv+silu:  each thread convolves its own q/k/v channels (kernel 4,
//               causal) and shifts the per-channel state. The q/k channels
//               of a k-head are shared by hV/hK v-heads, so one threadgroup
//               updates them once and then advances all mapped v-heads.
//   l2 norm:    q and k heads normalized in threadgroup memory.
//   delta rule: S ← exp(g_h)·S; d_j = β_h(v_j − S·k); S += k·d_j;
//               o_j = (S·q)/√dS.
//   gated norm: att = rmsnorm(o_head)·ssm_norm·silu(z).
//
// GQA mapping is modulo (qwen35moe) or consecutive (qwen3next). qkv layout:
// q [0, hK*dS), k [hK*dS, 2*hK*dS), v [2*hK*dS, ...) — RAW projection
// values (pre-conv).

struct StepPC { uint dState; uint hK; uint hV; float eps; uint kDiv; };

kernel void dn_step(device const float* qkv    [[buffer(0)]],
                    device float*       convSt [[buffer(1)]],
                    device const float* ker    [[buffer(2)]],
                    device const float* gb     [[buffer(3)]],
                    device float4*      s4     [[buffer(4)]],
                    device const float* z      [[buffer(5)]],
                    device const float* snW    [[buffer(6)]],
                    device float*       att    [[buffer(7)]],
                    constant StepPC&    pc     [[buffer(8)]],
                    uint3 tid3  [[thread_position_in_threadgroup]],
                    uint3 tgpig [[threadgroup_position_in_grid]],
                    uint  sgid  [[simdgroup_index_in_threadgroup]],
                    uint  slid  [[thread_index_in_simdgroup]])
{
    const uint kh = tgpig.x;
    const uint j  = tid3.x;          // 0..dS-1 (threadgroup = dS threads)
    const uint rq = tgpig.z;
    const uint dS = pc.dState;
    const uint nv = dS / 4u;
    const float eps = pc.eps;

    const uint chQkv = (2u * pc.hK + pc.hV) * dS;
    const uint qo    = rq * chQkv;

    // conv + silu + state shift for this thread's channels
    auto conv1 = [&](uint c) -> float {
        device float* st = convSt + (ulong)(rq * chQkv + c) * 3u;
        device const float* kc = ker + (ulong)c * 4u;
        const float s0 = st[0], s1 = st[1], s2 = st[2], xin = qkv[qo + c];
        const float v = s0 * kc[0] + s1 * kc[1] + s2 * kc[2] + xin * kc[3];
        st[0] = s1;
        st[1] = s2;
        st[2] = xin;
        return v / (1.0f + exp(-v));   // silu
    };
    const float qraw = conv1(kh * dS + j);
    const float kraw = conv1((pc.hK + kh) * dS + j);

    // L2 norms of the q and k heads (dS lanes = dS/32 simdgroups)
    threadgroup float qs[256];
    threadgroup float ks[256];
    threadgroup float pq[8], pk[8];
    threadgroup float bcq, bck;
    const float sq = simd_sum(qraw * qraw);
    const float sk = simd_sum(kraw * kraw);
    if (slid == 0u) { pq[sgid] = sq; pk[sgid] = sk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0u) {
        float tq = 0.0f, tk = 0.0f;
        for (uint i = 0u; i < dS / 32u; ++i) { tq += pq[i]; tk += pk[i]; }
        bcq = rsqrt(float(dS)) / max(sqrt(tq), eps);   // includes q scaling
        bck = 1.0f / max(sqrt(tk), eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qs[j] = qraw * bcq;
    ks[j] = kraw * bck;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Delta rule for every v-head mapped to this q/k head. Keeping the mapped
    // heads in one threadgroup is required: separate v-head threadgroups would
    // race while shifting the shared q/k convolution state above.
    threadgroup const float4* qs4 = (threadgroup const float4*)qs;
    threadgroup const float4* ks4 = (threadgroup const float4*)ks;
    const uint nPerK = pc.kDiv != 0u ? pc.kDiv : (pc.hV + pc.hK - 1u) / pc.hK;
    for (uint g = 0u; g < nPerK; ++g) {
        // kDiv == 0: h % hK == kh (qwen35moe modulo tiling).
        // kDiv != 0: h / kDiv == kh (qwen3next consecutive pairing).
        const uint h = pc.kDiv != 0u ? kh * pc.kDiv + g : kh + g * pc.hK;
        if (h >= pc.hV) continue;
        const float vj = conv1(2u * pc.hK * dS + h * dS + j);
        const float decay = exp(gb[rq * 2u * pc.hV + h]);
        const float beta  = gb[rq * 2u * pc.hV + pc.hV + h];
        const ulong row = (ulong)rq * pc.hV * dS * nv + ((ulong)h * dS + j) * nv;

        float sk2 = 0.0f;
        for (uint i = 0u; i < nv; ++i) sk2 += dot(s4[row + i], ks4[i]);
        sk2 *= decay;

        const float dj = beta * (vj - sk2);

        float oj = 0.0f;
        for (uint i = 0u; i < nv; ++i) {
            const float4 sn = s4[row + i] * decay + ks4[i] * dj;
            s4[row + i] = sn;
            oj += dot(sn, qs4[i]);
        }

        // gated RMS norm of the head output (dn_gate fused)
        const float so = simd_sum(oj * oj);
        if (slid == 0u) pq[sgid] = so;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j == 0u) {
            float t = 0.0f;
            for (uint i = 0u; i < dS / 32u; ++i) t += pq[i];
            bcq = 1.0f / sqrt(t / float(dS) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint ao = rq * pc.hV * dS + h * dS + j;
        const float zv = z[ao];
        att[ao] = oj * bcq * snW[j] * (zv / (1.0f + exp(-zv)));
    }
}
