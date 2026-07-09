#include <metal_stdlib>
using namespace metal;

// Batched (prefill) DeltaNet chain for ONE slot. Ports of dn_ab_batch,
// dn_conv_batch (carry-seeded causal conv + q/k L2 norm + conv-window
// handoff), dn_step_batch (state rows carried in registers across the Tn
// tokens), dn_gate_batch.

struct AbBPC   { uint n; uint hv; uint Tn; };
struct ConvBPC { uint channels; uint dState; uint qkCh; float eps; uint Tn; };
struct StepBPC { uint dState; uint hK; uint hV; uint Tn; };
struct GateBPC { uint dState; uint hV; float eps; uint Tn; };

constant uint NSG [[function_constant(0)]];

kernel void dn_ab_batch(device const float* x      [[buffer(0)]],
                        device const float* alphaW [[buffer(1)]],
                        device const float* betaW  [[buffer(2)]],
                        device const float* dtBias [[buffer(3)]],
                        device const float* aVec   [[buffer(4)]],
                        device float*       gb     [[buffer(5)]],
                        constant AbBPC&     pc     [[buffer(6)]],
                        uint3 tgpig [[threadgroup_position_in_grid]],
                        uint  sgid  [[simdgroup_index_in_threadgroup]],
                        uint  slid  [[thread_index_in_simdgroup]])
{
    const uint w = tgpig.x * NSG + sgid;
    const uint n = tgpig.z;
    if (w >= 2u * pc.hv) return;
    const uint xo = n * pc.n;
    const uint go = n * 2u * pc.hv;
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

// carry layout [channels][3] (k=0 oldest .. 2 newest) — same as the slot
// conv window this kernel seeds via stOut at the last token.
static inline float conv_input_at(device const float* qkv, device const float* carry,
                                  uint channels, uint n, uint ch, int back) {
    const int m = int(n) + back;
    if (m >= 0) return qkv[uint(m) * channels + ch];
    return carry[ch * 3u + uint(3 + m)];
}

kernel void dn_conv_batch(device const float* carry [[buffer(0)]],
                          device const float* qkv   [[buffer(1)]],
                          device const float* ker   [[buffer(2)]],
                          device float*       o     [[buffer(3)]],
                          device float*       stOut [[buffer(4)]],
                          constant ConvBPC&   pc    [[buffer(5)]],
                          uint3 tid3  [[thread_position_in_threadgroup]],
                          uint3 tgpig [[threadgroup_position_in_grid]],
                          uint  sgid  [[simdgroup_index_in_threadgroup]],
                          uint  slid  [[thread_index_in_simdgroup]])
{
    const uint w = tgpig.x;
    const uint t = tid3.x;
    const uint chBase = w * pc.dState;
    const uint n = tgpig.z;
    const uint qo = n * pc.channels;
    const uint ch = chBase + t;

    float v = 0.0f;
    if (ch < pc.channels && t < pc.dState) {
        const float c = conv_input_at(qkv, carry, pc.channels, n, ch, -3) * ker[ch * 4u] +
                        conv_input_at(qkv, carry, pc.channels, n, ch, -2) * ker[ch * 4u + 1u] +
                        conv_input_at(qkv, carry, pc.channels, n, ch, -1) * ker[ch * 4u + 2u] +
                        qkv[qo + ch] * ker[ch * 4u + 3u];
        v = c / (1.0f + exp(-c));
    }

    threadgroup float red[8];
    threadgroup float scale;
    if (chBase < pc.qkCh) {  // q/k head: L2 normalize (uniform branch per tg)
        const float sg = simd_sum(v * v);
        if (slid == 0u) red[sgid] = sg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0u) {
            float tot = 0.0f;
            for (uint i = 0u; i < pc.dState / 32u; ++i) tot += red[i];
            scale = 1.0f / max(sqrt(tot), pc.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v *= scale;
    }
    if (ch < pc.channels && t < pc.dState) o[qo + ch] = v;

    // last token seeds the slot's 3-tap conv window with the raw inputs
    if (n + 1u == pc.Tn && ch < pc.channels && t < pc.dState) {
        stOut[ch * 3u + 0u] = conv_input_at(qkv, carry, pc.channels, n, ch, -2);
        stOut[ch * 3u + 1u] = conv_input_at(qkv, carry, pc.channels, n, ch, -1);
        stOut[ch * 3u + 2u] = qkv[qo + ch];
    }
}

kernel void dn_step_batch(device const float* conv [[buffer(0)]],
                          device const float* gb   [[buffer(1)]],
                          device float*       o    [[buffer(2)]],
                          device float4*      s4   [[buffer(3)]],
                          constant StepBPC&   pc   [[buffer(4)]],
                          uint3 tid3  [[thread_position_in_threadgroup]],
                          uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint h = tgpig.x;
    const uint j = tid3.x;
    const uint dS = pc.dState;
    const uint nv = dS / 4u;

    const uint kh = h % pc.hK;
    const float qScale = rsqrt(float(dS));

    threadgroup float4 qs4[64];
    threadgroup float4 ks4[64];

    // seed each thread's state row from the slot buffer; persist after loop
    const ulong srowBase = ((ulong)h * dS + j) * nv;
    float4 srow[32];   // nv <= 32 (dState <= 128)
    for (uint i = 0u; i < nv; ++i) srow[i] = (j < dS) ? s4[srowBase + i] : float4(0.0f);

    for (uint t = 0u; t < pc.Tn; ++t) {
        const uint co = t * (2u * pc.hK + pc.hV) * dS;
        const uint qBase = co + kh * dS;
        const uint kBase = co + pc.hK * dS + kh * dS;
        const uint vBase = co + 2u * pc.hK * dS + h * dS;

        if (j < nv) {
            qs4[j] = ((device const float4*)(conv + qBase))[j] * qScale;
            ks4[j] = ((device const float4*)(conv + kBase))[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (j < dS) {
            const float decay = exp(gb[t * 2u * pc.hV + h]);
            const float beta  = gb[t * 2u * pc.hV + pc.hV + h];
            const float vj    = conv[vBase + j];

            float sk = 0.0f;
            for (uint i = 0u; i < nv; ++i) sk += dot(srow[i], ks4[i]);
            sk *= decay;

            const float dj = beta * (vj - sk);

            float oj = 0.0f;
            for (uint i = 0u; i < nv; ++i) {
                const float4 sn = srow[i] * decay + ks4[i] * dj;
                srow[i] = sn;
                oj += dot(sn, qs4[i]);
            }
            o[t * pc.hV * dS + h * dS + j] = oj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (j < dS)
        for (uint i = 0u; i < nv; ++i) s4[srowBase + i] = srow[i];
}

kernel void dn_gate_batch(device const float* o   [[buffer(0)]],
                          device const float* w   [[buffer(1)]],
                          device const float* z   [[buffer(2)]],
                          device float*       att [[buffer(3)]],
                          constant GateBPC&   pc  [[buffer(4)]],
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
