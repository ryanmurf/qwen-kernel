#include <metal_stdlib>
using namespace metal;

// Fixed-K Q8_0 GEMV specializations for the engine's dense projections.
// Compile-time trip counts let Metal fully unroll the short block loops.
constant uint TPR [[function_constant(0)]];

struct block_q8_0 {
    half d;
    char qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 block size");

template <uint KFIX>
static inline void gemv_q8_fixed_body(device const block_q8_0* wb,
                                      device const float* x,
                                      device float* y,
                                      constant uint2& mk,
                                      threadgroup float* partial,
                                      uint tid, uint tgx, uint tgz,
                                      uint sgid, uint slid) {
    const uint M = mk.x;
    const uint lane = tid % TPR;
    const uint row = tgx * (256u / TPR) + tid / TPR;
    const uint kb = KFIX / 32u;
    const ulong base = (ulong)row * kb;

    float acc = 0.0f;
#pragma unroll
    for (uint b = lane; b < kb; b += TPR) {
        device const block_q8_0& blk = wb[base + b];
        const float d = float(blk.d);
        device const packed_char4* qp = (device const packed_char4*)blk.qs;
        device const float4* xp =
            (device const float4*)(x + (ulong)tgz * KFIX + b * 32u);
        float s = 0.0f;
#pragma unroll
        for (uint j = 0u; j < 8u; ++j)
            s += dot(float4(char4(qp[j])), xp[j]);
        acc += d * s;
    }

    if (TPR <= 32u) {
        for (uint s = TPR / 2u; s > 0u; s >>= 1u)
            acc += simd_shuffle_down(acc, s);
    } else {
        acc = simd_sum(acc);
        if (slid == 0u) partial[sgid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            const uint sgPerRow = TPR / 32u;
            const uint first = (tid / TPR) * sgPerRow;
            float s = 0.0f;
            for (uint g = 0u; g < sgPerRow; ++g) s += partial[first + g];
            acc = s;
        }
    }
    if (lane == 0u) y[(ulong)tgz * M + row] = acc;
}

kernel void gemv_q8_0_k2048(device const block_q8_0* wb [[buffer(0)]],
                            device const float* x [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant uint2& mk [[buffer(3)]],
                            uint3 tid3 [[thread_position_in_threadgroup]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint sgid [[simdgroup_index_in_threadgroup]],
                            uint slid [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    gemv_q8_fixed_body<2048u>(wb, x, y, mk, partial, tid3.x, tgpig.x,
                              tgpig.z, sgid, slid);
}

kernel void gemv_q8_0_k4096(device const block_q8_0* wb [[buffer(0)]],
                            device const float* x [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant uint2& mk [[buffer(3)]],
                            uint3 tid3 [[thread_position_in_threadgroup]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint sgid [[simdgroup_index_in_threadgroup]],
                            uint slid [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    gemv_q8_fixed_body<4096u>(wb, x, y, mk, partial, tid3.x, tgpig.x,
                              tgpig.z, sgid, slid);
}
