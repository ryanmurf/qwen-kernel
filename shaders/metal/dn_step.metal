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

template <uint NCACHE>
static inline void dn_step_body(device const float* qkv,
                                device float*       convSt,
                                device const float* ker,
                                device const float* gb,
                                device float4*      s4,
                                device const float* z,
                                device const float* snW,
                                device float*       att,
                                constant StepPC&    pc,
                                threadgroup float*  qs,
                                threadgroup float*  ks,
                                threadgroup float*  pq,
                                threadgroup float*  pk,
                                threadgroup float&  bcq,
                                threadgroup float&  bck,
                                uint3 tid3, uint3 tgpig, uint sgid, uint slid)
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
        float oj = 0.0f;
        if (NCACHE != 0u) {
            // Cache a leading row panel; the retained 8-float4 specialization
            // is the measured balance between state rereads and register pressure.
            float4 srow[NCACHE ? NCACHE : 1u];
            for (uint i = 0u; i < NCACHE; ++i) {
                srow[i] = s4[row + i];
                sk2 += dot(srow[i], ks4[i]);
            }
            for (uint i = NCACHE; i < nv; ++i)
                sk2 += dot(s4[row + i], ks4[i]);
            sk2 *= decay;
            const float dj = beta * (vj - sk2);
            for (uint i = 0u; i < NCACHE; ++i) {
                const float4 sn = srow[i] * decay + ks4[i] * dj;
                s4[row + i] = sn;
                oj += dot(sn, qs4[i]);
            }
            for (uint i = NCACHE; i < nv; ++i) {
                const float4 sn = s4[row + i] * decay + ks4[i] * dj;
                s4[row + i] = sn;
                oj += dot(sn, qs4[i]);
            }
        } else {
            for (uint i = 0u; i < nv; ++i) sk2 += dot(s4[row + i], ks4[i]);
            sk2 *= decay;
            const float dj = beta * (vj - sk2);
            for (uint i = 0u; i < nv; ++i) {
                const float4 sn = s4[row + i] * decay + ks4[i] * dj;
                s4[row + i] = sn;
                oj += dot(sn, qs4[i]);
            }
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
    threadgroup float qs[256], ks[256], pq[8], pk[8], bcq, bck;
    dn_step_body<0u>(qkv, convSt, ker, gb, s4, z, snW, att,
                     pc, qs, ks, pq, pk, bcq, bck, tid3, tgpig, sgid, slid);
}

#define DN_STEP_CACHE_KERNEL(NAME, NC)                                                \
kernel void NAME(device const float* qkv    [[buffer(0)]],                            \
                 device float*       convSt [[buffer(1)]],                            \
                 device const float* ker    [[buffer(2)]],                            \
                 device const float* gb     [[buffer(3)]],                            \
                 device float4*      s4     [[buffer(4)]],                            \
                 device const float* z      [[buffer(5)]],                            \
                 device const float* snW    [[buffer(6)]],                            \
                 device float*       att    [[buffer(7)]],                            \
                 constant StepPC&    pc     [[buffer(8)]],                            \
                 uint3 tid3  [[thread_position_in_threadgroup]],                      \
                 uint3 tgpig [[threadgroup_position_in_grid]],                        \
                 uint  sgid  [[simdgroup_index_in_threadgroup]],                      \
                 uint  slid  [[thread_index_in_simdgroup]])                           \
{                                                                                     \
    threadgroup float qs[256], ks[256], pq[8], pk[8], bcq, bck;                       \
    dn_step_body<NC>(qkv, convSt, ker, gb, s4, z, snW, att,                           \
                     pc, qs, ks, pq, pk, bcq, bck, tid3, tgpig, sgid, slid);           \
}

DN_STEP_CACHE_KERNEL(dn_step_res8, 8u)

// Two-dispatch retile: preserve the race-free one-TG-per-k-head convolution
// update, then expose one independent TG per v-head for the matrix recurrence.
// qkv is dead after this stage, so it doubles as normalized conv output.
kernel void dn_step_prep(device float*       qkv    [[buffer(0)]],
                         device float*       convSt [[buffer(1)]],
                         device const float* ker    [[buffer(2)]],
                         constant StepPC&    pc     [[buffer(3)]],
                         uint3 tid3  [[thread_position_in_threadgroup]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint  sgid  [[simdgroup_index_in_threadgroup]],
                         uint  slid  [[thread_index_in_simdgroup]])
{
    const uint kh = tgpig.x, j = tid3.x, rq = tgpig.z;
    const uint dS = pc.dState;
    const uint chQkv = (2u * pc.hK + pc.hV) * dS;
    const uint qo = rq * chQkv;
    auto conv1 = [&](uint c) -> float {
        device float* st = convSt + (ulong)(rq * chQkv + c) * 3u;
        device const float* kc = ker + (ulong)c * 4u;
        const float s0 = st[0], s1 = st[1], s2 = st[2], xin = qkv[qo + c];
        const float v = s0 * kc[0] + s1 * kc[1] + s2 * kc[2] + xin * kc[3];
        st[0] = s1; st[1] = s2; st[2] = xin;
        return v / (1.0f + exp(-v));
    };
    const float qraw = conv1(kh * dS + j);
    const float kraw = conv1((pc.hK + kh) * dS + j);
    threadgroup float pq[8], pk[8], bcq, bck;
    const float sq = simd_sum(qraw * qraw), sk = simd_sum(kraw * kraw);
    if (slid == 0u) { pq[sgid] = sq; pk[sgid] = sk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0u) {
        float tq = 0.0f, tk = 0.0f;
        for (uint i = 0u; i < dS / 32u; ++i) { tq += pq[i]; tk += pk[i]; }
        bcq = rsqrt(float(dS)) / max(sqrt(tq), pc.eps);
        bck = 1.0f / max(sqrt(tk), pc.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qkv[qo + kh * dS + j] = qraw * bcq;
    qkv[qo + (pc.hK + kh) * dS + j] = kraw * bck;

    const uint nPerK = pc.kDiv != 0u ? pc.kDiv : (pc.hV + pc.hK - 1u) / pc.hK;
    for (uint g = 0u; g < nPerK; ++g) {
        const uint h = pc.kDiv != 0u ? kh * pc.kDiv + g : kh + g * pc.hK;
        if (h < pc.hV)
            qkv[qo + (2u * pc.hK + h) * dS + j] =
                conv1((2u * pc.hK + h) * dS + j);
    }
}

template <uint NCACHE>
static inline void dn_step_state_body(device const float* conv,
                                      device const float* gb,
                                      device float4* s4,
                                      device const float* z,
                                      device const float* snW,
                                      device float* att,
                                      constant StepPC& pc,
                                      threadgroup float* qs,
                                      threadgroup float* ks,
                                      threadgroup float* pq,
                                      threadgroup float& bcq,
                                      uint3 tid3, uint3 tgpig,
                                      uint sgid, uint slid)
{
    const uint h = tgpig.x, j = tid3.x, rq = tgpig.z;
    const uint dS = pc.dState, nv = dS / 4u;
    const uint kh = pc.kDiv != 0u ? h / pc.kDiv : h % pc.hK;
    const uint chQkv = (2u * pc.hK + pc.hV) * dS;
    const uint co = rq * chQkv;
    qs[j] = conv[co + kh * dS + j];
    ks[j] = conv[co + (pc.hK + kh) * dS + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const float4* qs4 = (threadgroup const float4*)qs;
    threadgroup const float4* ks4 = (threadgroup const float4*)ks;
    const float vj = conv[co + (2u * pc.hK + h) * dS + j];
    const float decay = exp(gb[rq * 2u * pc.hV + h]);
    const float beta = gb[rq * 2u * pc.hV + pc.hV + h];
    const ulong row = (ulong)rq * pc.hV * dS * nv + ((ulong)h * dS + j) * nv;
    float sk2 = 0.0f, oj = 0.0f;
    if (NCACHE != 0u) {
        float4 srow[NCACHE ? NCACHE : 1u];
        for (uint i = 0u; i < NCACHE; ++i) {
            srow[i] = s4[row + i];
            sk2 += dot(srow[i], ks4[i]);
        }
        for (uint i = NCACHE; i < nv; ++i) sk2 += dot(s4[row + i], ks4[i]);
        sk2 *= decay;
        const float dj = beta * (vj - sk2);
        for (uint i = 0u; i < NCACHE; ++i) {
            const float4 sn = srow[i] * decay + ks4[i] * dj;
            s4[row + i] = sn; oj += dot(sn, qs4[i]);
        }
        for (uint i = NCACHE; i < nv; ++i) {
            const float4 sn = s4[row + i] * decay + ks4[i] * dj;
            s4[row + i] = sn; oj += dot(sn, qs4[i]);
        }
    } else {
        for (uint i = 0u; i < nv; ++i) sk2 += dot(s4[row + i], ks4[i]);
        sk2 *= decay;
        const float dj = beta * (vj - sk2);
        for (uint i = 0u; i < nv; ++i) {
            const float4 sn = s4[row + i] * decay + ks4[i] * dj;
            s4[row + i] = sn; oj += dot(sn, qs4[i]);
        }
    }
    const float so = simd_sum(oj * oj);
    if (slid == 0u) pq[sgid] = so;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0u) {
        float t = 0.0f;
        for (uint i = 0u; i < dS / 32u; ++i) t += pq[i];
        bcq = 1.0f / sqrt(t / float(dS) + pc.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint ao = rq * pc.hV * dS + h * dS + j;
    const float zv = z[ao];
    att[ao] = oj * bcq * snW[j] * (zv / (1.0f + exp(-zv)));
}

#define DN_STEP_STATE_KERNEL(NAME, NC)                                                \
kernel void NAME(device const float* conv [[buffer(0)]],                              \
                 device const float* gb   [[buffer(1)]],                              \
                 device float4* s4        [[buffer(2)]],                              \
                 device const float* z    [[buffer(3)]],                              \
                 device const float* snW  [[buffer(4)]],                              \
                 device float* att        [[buffer(5)]],                              \
                 constant StepPC& pc      [[buffer(6)]],                              \
                 uint3 tid3 [[thread_position_in_threadgroup]],                       \
                 uint3 tgpig [[threadgroup_position_in_grid]],                        \
                 uint sgid [[simdgroup_index_in_threadgroup]],                        \
                 uint slid [[thread_index_in_simdgroup]])                             \
{                                                                                     \
    threadgroup float qs[128], ks[128], pq[8], bcq;                                   \
    dn_step_state_body<NC>(conv, gb, s4, z, snW, att, pc, qs, ks, pq, bcq,            \
                           tid3, tgpig, sgid, slid);                                   \
}

DN_STEP_STATE_KERNEL(dn_step_state, 0u)
DN_STEP_STATE_KERNEL(dn_step_state_res8, 8u)

static inline float simd_sum_ordered(float v) {
    float s = 0.0f;
#pragma unroll
    for (uint i = 0u; i < 32u; ++i) s += simd_shuffle(v, i);
    return s;
}

// Same one-read row-SIMD schedule, but reproduce the streamed kernel's
// left-to-right float4 accumulation exactly instead of using a reduction tree.
kernel void dn_step_state_rowsimd(device const float* conv [[buffer(0)]],
                                  device const float* gb   [[buffer(1)]],
                                  device float4* s4        [[buffer(2)]],
                                  device const float* z    [[buffer(3)]],
                                  device const float* snW  [[buffer(4)]],
                                  device float* att        [[buffer(5)]],
                                  constant StepPC& pc      [[buffer(6)]],
                                  uint3 tid3 [[thread_position_in_threadgroup]],
                                  uint3 tgpig [[threadgroup_position_in_grid]],
                                  uint sgid [[simdgroup_index_in_threadgroup]],
                                  uint slid [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x, j = tid3.x, rq = tgpig.z;
    const uint dS = pc.dState, nv = dS / 4u;
    const uint kh = pc.kDiv != 0u ? h / pc.kDiv : h % pc.hK;
    const uint chQkv = (2u * pc.hK + pc.hV) * dS;
    const uint co = rq * chQkv;
    threadgroup float qs[128], ks[128], otg[128], pq[8], bcq, decayT, betaT;
    qs[j] = conv[co + kh * dS + j];
    ks[j] = conv[co + (pc.hK + kh) * dS + j];
    if (j == 0u) {
        decayT = exp(gb[rq * 2u * pc.hV + h]);
        betaT = gb[rq * 2u * pc.hV + pc.hV + h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const float4* q4 = (threadgroup const float4*)qs;
    threadgroup const float4* k4 = (threadgroup const float4*)ks;
    device float4* S = s4 + ((ulong)rq * pc.hV + h) * dS * nv;
    for (uint row = sgid; row < dS; row += 4u) {
        const float4 old = S[(ulong)row * nv + slid];
        const float sk = simd_sum_ordered(dot(old, k4[slid])) * decayT;
        const float dj = betaT *
            (conv[co + (2u * pc.hK + h) * dS + row] - sk);
        const float4 sn = old * decayT + k4[slid] * dj;
        S[(ulong)row * nv + slid] = sn;
        const float oj = simd_sum_ordered(dot(sn, q4[slid]));
        if (slid == 0u) otg[row] = oj;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float oj = otg[j];
    const float so = simd_sum(oj * oj);
    if (slid == 0u) pq[sgid] = so;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0u) {
        float t = 0.0f;
        for (uint i = 0u; i < dS / 32u; ++i) t += pq[i];
        bcq = 1.0f / sqrt(t / float(dS) + pc.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint ao = rq * pc.hV * dS + h * dS + j;
    const float zv = z[ao];
    att[ao] = oj * bcq * snW[j] * (zv / (1.0f + exp(-zv)));
}
