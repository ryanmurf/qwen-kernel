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

// Second half of the large-slot split-router path. add_rmsnorm_sg has
// persisted y and the same scale computed by moe_logits_addn's e=0
// simdgroup. Keep scale outside the dot exactly as the fused kernel does.
kernel void moe_logits_scaled(device const float* gi      [[buffer(0)]],
                              device const float* gis     [[buffer(1)]],
                              device const float* y       [[buffer(2)]],
                              device const float* pn      [[buffer(3)]],
                              device const float* scales  [[buffer(4)]],
                              device float*       logits  [[buffer(5)]],
                              constant MoePC&     pc      [[buffer(6)]],
                              uint3 tgpig [[threadgroup_position_in_grid]],
                              uint  sgid  [[simdgroup_index_in_threadgroup]],
                              uint  slid  [[thread_index_in_simdgroup]])
{
    const uint e = tgpig.x * NSG + sgid;
    const uint rq = tgpig.z;
    if (e > pc.n_expert) return;
    const uint n4 = pc.n_embd / 4u;
    device const float4* y4 = (device const float4*)(y + rq * pc.n_embd);
    device const float4* pn4 = (device const float4*)pn;
    device const float4* g4 = (device const float4*)
        (e < pc.n_expert ? gi + (ulong)e * pc.n_embd : gis);
    float acc = 0.0f;
    for (uint k = slid; k < n4; k += 32u)
        acc += dot(g4[k], y4[k] * pn4[k]);
    const float lg = simd_sum(acc) * scales[rq];
    if (slid == 0u) logits[rq * (pc.n_expert + 1u) + e] = lg;
}

// Batched router logits for grouped-prefill chunks: one f32 GEMM over the
// 257 router rows (gis = virtual row 256) instead of a GEMV per token.
// Same 32x32 f32-fragment skeleton as moe_gu_grouped4 minus the dequant;
// order-only noise class. Residual+norm run separately via add_rmsnorm
// (identical outputs to moe_logits_addn's y/xn2). GemmPC: M=257, K, N=n;
// logits stride = M.
struct LgPC { uint M; uint K; uint N; };
constant uint LGS = 68u;

kernel void moe_logits_gemm(device const float* gi     [[buffer(0)]],
                            device const float* gis    [[buffer(1)]],
                            device const float* x      [[buffer(2)]],
                            device float*       logits [[buffer(3)]],
                            constant LgPC&      pc     [[buffer(4)]],
                            uint3 tid3  [[thread_position_in_threadgroup]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint  sgid  [[simdgroup_index_in_threadgroup]],
                            uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;                 // 0..127, 4 simdgroups
    const uint row0 = tgpig.x * 32u;
    const uint tok0 = tgpig.z * 32u;

    threadgroup float Wsh[32u * LGS];
    threadgroup float Xsh[32u * LGS];
    threadgroup float outb[4u * 72u];

    simdgroup_float8x8 acc[4];
    for (uint i = 0u; i < 4u; ++i) acc[i] = simdgroup_float8x8(0.0f);

    for (uint k0 = 0u; k0 < pc.K; k0 += 64u) {
        for (uint idx = tid * 4u; idx < 32u * 64u; idx += 128u * 4u) {
            const uint rr = idx >> 6, kk = idx & 63u;
            const uint r = row0 + rr;
            float4 v = float4(0.0f);
            if (r + 1u < pc.M)
                v = *(device const packed_float4*)(gi + (ulong)r * pc.K + k0 + kk);
            else if (r + 1u == pc.M)
                v = *(device const packed_float4*)(gis + k0 + kk);
            ((threadgroup float4*)&Wsh[rr * LGS + kk])[0] = v;
        }
        for (uint idx = tid * 4u; idx < 32u * 64u; idx += 128u * 4u) {
            const uint tt = idx >> 6, kk = idx & 63u;
            ((threadgroup float4*)&Xsh[tt * LGS + kk])[0] = (tok0 + tt < pc.N)
                ? *(device const packed_float4*)(x + (ulong)(tok0 + tt) * pc.K + k0 + kk)
                : float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kf = 0u; kf < 8u; ++kf) {
            simdgroup_float8x8 a;
            simdgroup_load(a, &Wsh[(sgid * 8u) * LGS + kf * 8u], LGS);
            for (uint nc = 0u; nc < 4u; ++nc) {
                simdgroup_float8x8 bfr;
                simdgroup_load(bfr, &Xsh[(nc * 8u) * LGS + kf * 8u], LGS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[nc], a, bfr, acc[nc]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* buf = &outb[sgid * 72u];
    const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
    for (uint nc = 0u; nc < 4u; ++nc) {
        simdgroup_store(acc[nc], buf, 9u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint row = row0 + sgid * 8u + fi;
        for (uint jj = 0u; jj < 2u; ++jj) {
            const uint tok = tok0 + nc * 8u + fj0 + jj;
            if (row < pc.M && tok < pc.N)
                logits[(ulong)tok * pc.M + row] = buf[fi * 9u + fj0 + jj];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
