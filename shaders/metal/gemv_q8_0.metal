#include <metal_stdlib>
using namespace metal;

// y[M] = W[M,K] · x[K], W in raw ggml Q8_0 blocks (34 B per 32 weights).
// Port of shaders/gemv_q8_0.comp. TPR (function constant 0): threads per
// row; 256/TPR rows per threadgroup. Grid z batches queries (slots): the
// engine points x at rq*K and y at rq*M, mirroring the Vulkan dispatch.

constant uint TPR [[function_constant(0)]];

struct block_q8_0 {
    half d;
    char qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 block size");

kernel void gemv_q8_0(device const block_q8_0* wb  [[buffer(0)]],
                      device const float*      x   [[buffer(1)]],
                      device float*            y   [[buffer(2)]],
                      constant uint2&          mk  [[buffer(3)]],
                      uint3 tid3  [[thread_position_in_threadgroup]],
                      uint3 tgpig [[threadgroup_position_in_grid]],
                      uint  sgid  [[simdgroup_index_in_threadgroup]],
                      uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid  = tid3.x;
    const uint M    = mk.x;
    const uint K    = mk.y;
    const uint rq   = tgpig.z;
    const uint xo2  = rq * K;
    const uint lane = tid % TPR;
    const uint row  = tgpig.x * (256u / TPR) + tid / TPR;

    float acc = 0.0f;
    if (row < M) {
        const uint kb   = K / 32u;
        const ulong base = (ulong)row * kb;
        for (uint b = lane; b < kb; b += TPR) {
            device const block_q8_0& blk = wb[base + b];
            const float d = float(blk.d);
            device const packed_char4* qp = (device const packed_char4*)blk.qs;
            device const float4* xp = (device const float4*)(x + xo2 + b * 32u);
            float s = 0.0f;
            for (uint j = 0u; j < 8u; ++j)
                s += dot(float4(char4(qp[j])), xp[j]);
            acc += d * s;
        }
    }

    threadgroup float partial[8];
    if (TPR <= 32u) {
        for (uint s = TPR / 2u; s > 0u; s >>= 1u)
            acc += simd_shuffle_down(acc, s);
    } else {
        acc = simd_sum(acc);
        if (slid == 0u) partial[sgid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            const uint sgPerRow = TPR / 32u;
            const uint first    = (tid / TPR) * sgPerRow;
            float s = 0.0f;
            for (uint g = 0u; g < sgPerRow; g++) s += partial[first + g];
            acc = s;
        }
    }
    if (lane == 0u && row < M) y[(ulong)rq * M + row] = acc;
}
