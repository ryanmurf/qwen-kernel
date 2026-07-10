#include <metal_stdlib>
using namespace metal;

// Embedding lookup: dequantize token_embd row ids[idx] (Q8_0) into x, then
// apply the first layer's attn_norm (same structure as embed_q6k).
// One threadgroup; grid z batches queries.

struct block_q8_0 {
    half d;
    char qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 block size");

struct EmbPC { uint kdim; uint idx; uint perReq; float eps; };

// Dequantizes the row AND applies the first layer's attn_norm in the same
// threadgroup (saves a 1-tg rmsnorm stage per token).
kernel void embed_q8_0(device const block_q8_0* wb  [[buffer(0)]],
                       device const uint*       ids [[buffer(1)]],
                       device float*            x   [[buffer(2)]],
                       device const float*      nw  [[buffer(3)]],
                       device float*            xn  [[buffer(4)]],
                       constant EmbPC&          pc  [[buffer(5)]],
                       uint3 tid3  [[thread_position_in_threadgroup]],
                       uint3 tgpig [[threadgroup_position_in_grid]],
                       uint  sgid  [[simdgroup_index_in_threadgroup]],
                       uint  slid  [[thread_index_in_simdgroup]])
{
    const uint t  = tid3.x;
    const uint kb = pc.kdim / 32u;
    const uint rq = tgpig.z;
    const uint ro = rq * pc.kdim;
    const ulong base = (ulong)ids[pc.idx + pc.perReq * rq] * kb;
    for (uint b = t; b < kb; b += 256u) {
        device const block_q8_0& blk = wb[base + b];
        const float d = float(blk.d);
        device const packed_char4* qp = (device const packed_char4*)blk.qs;
        device float4* xo = (device float4*)(x + ro + b * 32u);
        for (uint j = 0u; j < 8u; ++j)
            xo[j] = d * float4(char4(qp[j]));
    }

    // reduce and scale
    threadgroup float red[8];
    threadgroup float bc;
    threadgroup_barrier(mem_flags::mem_device);
    float ss = 0.0f;
    for (uint i = t; i < pc.kdim; i += 256u) ss += x[ro + i] * x[ro + i];
    const float sg = simd_sum(ss);
    if (slid == 0u) red[sgid] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
        float tot = 0.0f;
        for (uint i = 0u; i < 8u; ++i) tot += red[i];
        bc = 1.0f / sqrt(tot / float(pc.kdim) + pc.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < pc.kdim; i += 256u) xn[ro + i] = x[ro + i] * bc * nw[i];
}
