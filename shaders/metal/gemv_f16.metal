#include <metal_stdlib>
using namespace metal;

// y[M] = W[M,K] · x[K]
// W: row-major fp16. x, y: f32. Port of shaders/gemv_f16.comp.
//
// TPR (function constant 0) = threads per row, power of two 4..256. Each
// 256-thread threadgroup covers 256/TPR consecutive rows. A thread consumes
// one 8-half unit per stride (two half4 loads = 16 B of W), so the host
// passes unitsPerRow = K/8. Reduction: simd_shuffle_down inside a simdgroup
// when TPR <= 32 (subgroupAdd -> simd ops per the port plan), simd_sum plus
// a threadgroup partial across simdgroups when TPR > 32.

constant uint TPR [[function_constant(0)]];

kernel void gemv_f16(device const half*   w   [[buffer(0)]],
                     device const float*  x   [[buffer(1)]],
                     device float*        y   [[buffer(2)]],
                     constant uint2&      mk  [[buffer(3)]],
                     uint tid  [[thread_position_in_threadgroup]],
                     uint wg   [[threadgroup_position_in_grid]],
                     uint sgid [[simdgroup_index_in_threadgroup]],
                     uint slid [[thread_index_in_simdgroup]])
{
    const uint M    = mk.x;
    const uint K    = mk.y;
    const uint lane = tid % TPR;
    const uint row  = wg * (256u / TPR) + tid / TPR;

    float acc = 0.0f;
    if (row < M) {
        const uint kw = K / 8u;  // 8-half units per row
        device const half4*  wr = (device const half4*)(w + (ulong)row * K);
        device const float4* x4 = (device const float4*)x;
        for (uint i = lane; i < kw; i += TPR) {
            acc += dot(float4(wr[2u * i    ]), x4[2u * i    ]);
            acc += dot(float4(wr[2u * i + 1u]), x4[2u * i + 1u]);
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
    if (lane == 0u && row < M) y[row] = acc;
}
