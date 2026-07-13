#include <metal_stdlib>
using namespace metal;

// Slot-batched Q8_0 GEMV. A thread decodes each weight block once and dots it
// with NQ independent activation rows. The grid-z axis selects a group of NQ
// slots, so output stays in the engine's slot-major [slot][M] layout.

constant uint TPR [[function_constant(0)]];

struct block_q8_0 {
    half d;
    char qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 block size");

template <ushort NQ, uint KFIX = 0u>
static inline void gemv_q8_batch_body(device const block_q8_0* wb,
                                      device const float* x,
                                      device float* y,
                                      constant uint2& mk,
                                      threadgroup float* partial,
                                      uint tid, uint tgx, uint tgz,
                                      uint sgid, uint slid) {
    const uint M = mk.x;
    const uint K = KFIX ? KFIX : mk.y;
    const uint lane = tid % TPR;
    const uint row = tgx * (256u / TPR) + tid / TPR;
    const uint rq0 = tgz * NQ;

    float acc[NQ];
    for (ushort rq = 0; rq < NQ; ++rq) acc[rq] = 0.0f;
    if (row < M) {
        const uint kb = K / 32u;
        const ulong base = (ulong)row * kb;
#pragma unroll
        for (uint b = lane; b < kb; b += TPR) {
            device const block_q8_0& blk = wb[base + b];
            device const packed_char4* qp =
                (device const packed_char4*)blk.qs;
            float sb[NQ];
            for (ushort rq = 0; rq < NQ; ++rq) sb[rq] = 0.0f;
            for (uint j = 0u; j < 8u; ++j) {
                const float4 qv = float4(char4(qp[j]));
                for (ushort rq = 0; rq < NQ; ++rq) {
                    device const float4* xp = (device const float4*)(
                        x + (ulong)(rq0 + rq) * K + b * 32u);
                    sb[rq] += dot(qv, xp[j]);
                }
            }
            const float d = float(blk.d);
            for (ushort rq = 0; rq < NQ; ++rq) acc[rq] += d * sb[rq];
        }
    }

    if (TPR <= 32u) {
        for (ushort rq = 0; rq < NQ; ++rq) {
            float v = acc[rq];
            for (uint s = TPR / 2u; s > 0u; s >>= 1u)
                v += simd_shuffle_down(v, s);
            if (lane == 0u && row < M)
                y[(ulong)(rq0 + rq) * M + row] = v;
        }
    } else {
        for (ushort rq = 0; rq < NQ; ++rq) {
            const float v = simd_sum(acc[rq]);
            if (slid == 0u) partial[(uint)rq * 8u + sgid] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u && row < M) {
            const uint sgPerRow = TPR / 32u;
            const uint first = (tid / TPR) * sgPerRow;
            for (ushort rq = 0; rq < NQ; ++rq) {
                float v = 0.0f;
                for (uint g = 0u; g < sgPerRow; ++g)
                    v += partial[(uint)rq * 8u + first + g];
                y[(ulong)(rq0 + rq) * M + row] = v;
            }
        }
    }
}

kernel void gemv_q8_0_b2(device const block_q8_0* wb [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant uint2& mk [[buffer(3)]],
                         uint3 tid3 [[thread_position_in_threadgroup]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint sgid [[simdgroup_index_in_threadgroup]],
                         uint slid [[thread_index_in_simdgroup]]) {
    threadgroup float partial[2 * 8];
    gemv_q8_batch_body<2>(wb, x, y, mk, partial, tid3.x, tgpig.x,
                          tgpig.z, sgid, slid);
}

kernel void gemv_q8_0_b4(device const block_q8_0* wb [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant uint2& mk [[buffer(3)]],
                         uint3 tid3 [[thread_position_in_threadgroup]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint sgid [[simdgroup_index_in_threadgroup]],
                         uint slid [[thread_index_in_simdgroup]]) {
    threadgroup float partial[4 * 8];
    gemv_q8_batch_body<4>(wb, x, y, mk, partial, tid3.x, tgpig.x,
                          tgpig.z, sgid, slid);
}

kernel void gemv_q8_0_b8(device const block_q8_0* wb [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant uint2& mk [[buffer(3)]],
                         uint3 tid3 [[thread_position_in_threadgroup]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint sgid [[simdgroup_index_in_threadgroup]],
                         uint slid [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8 * 8];
    gemv_q8_batch_body<8>(wb, x, y, mk, partial, tid3.x, tgpig.x,
                          tgpig.z, sgid, slid);
}

#define QK_FIXED_BATCH(NAME, NQ, KFIX)                                      \
kernel void NAME(device const block_q8_0* wb [[buffer(0)]],                 \
                 device const float* x [[buffer(1)]],                       \
                 device float* y [[buffer(2)]],                             \
                 constant uint2& mk [[buffer(3)]],                          \
                 uint3 tid3 [[thread_position_in_threadgroup]],             \
                 uint3 tgpig [[threadgroup_position_in_grid]],              \
                 uint sgid [[simdgroup_index_in_threadgroup]],              \
                 uint slid [[thread_index_in_simdgroup]]) {                 \
    threadgroup float partial[NQ * 8];                                      \
    gemv_q8_batch_body<NQ, KFIX>(wb, x, y, mk, partial, tid3.x, tgpig.x,   \
                                 tgpig.z, sgid, slid);                       \
}

QK_FIXED_BATCH(gemv_q8_0_b4_k2048, 4, 2048u)
QK_FIXED_BATCH(gemv_q8_0_b4_k4096, 4, 4096u)
