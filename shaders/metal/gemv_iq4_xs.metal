#include <metal_stdlib>
using namespace metal;

// y[M] = W[M,K] · x[K], W in raw ggml IQ4_XS super-blocks (136 B per 256).
// Port of shaders/gemv_iq4_xs.comp. 4.25 bpw: 8 sub-blocks of 32 with 6-bit
// scales and a shared nonlinear 16-value codebook (kvalues_iq4nl). One
// 32-element sub-block per work unit.

#include "iq_tables.metal"

constant uint TPR [[function_constant(0)]];

struct block_iq4_xs {
    half   d;
    ushort scales_h;
    uchar  scales_l[4];
    uchar  qs[128];
};
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs block size");

kernel void gemv_iq4_xs(device const block_iq4_xs* wb  [[buffer(0)]],
                        device const float*        x   [[buffer(1)]],
                        device float*              y   [[buffer(2)]],
                        constant uint2&            mk  [[buffer(3)]],
                        uint3 tid3  [[thread_position_in_threadgroup]],
                        uint3 tgpig [[threadgroup_position_in_grid]],
                        uint  sgid  [[simdgroup_index_in_threadgroup]],
                        uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid  = tid3.x;
    const uint M    = mk.x;
    const uint K    = mk.y;
    const uint lane = tid % TPR;
    const uint row  = tgpig.x * (256u / TPR) + tid / TPR;

    float acc = 0.0f;
    if (row < M) {
        const uint kb    = K / 256u;  // super-blocks per row
        const uint nu    = kb * 8u;   // 32-element sub-blocks per row
        const ulong base = (ulong)row * kb;
        for (uint u = lane; u < nu; u += TPR) {
            const uint b  = u >> 3u;
            const uint ib = u & 7u;

            device const block_iq4_xs& blk = wb[base + b];
            const float d  = float(blk.d);
            const uint  sl = uint(blk.scales_l[ib >> 1u]);
            const uint  sh = uint(blk.scales_h);
            const int   ls = int(((sl >> (4u * (ib & 1u))) & 0xFu) |
                                 (((sh >> (2u * ib)) & 3u) << 4u)) - 32;
            const float dl = d * float(ls);

            device const packed_uchar4* qp =
                (device const packed_uchar4*)&blk.qs[ib * 16u];
            device const float4* xp =
                (device const float4*)(x + b * 256u + ib * 32u);

            float s1 = 0.0f, s2 = 0.0f;
            for (uint j = 0u; j < 4u; ++j) {
                const uchar4 q = uchar4(qp[j]);
                const float4 xlo = xp[j];
                const float4 xhi = xp[4u + j];
                s1 += float(kvalues_iq4nl[q.x & 0xFu]) * xlo.x
                    + float(kvalues_iq4nl[q.y & 0xFu]) * xlo.y
                    + float(kvalues_iq4nl[q.z & 0xFu]) * xlo.z
                    + float(kvalues_iq4nl[q.w & 0xFu]) * xlo.w;
                s2 += float(kvalues_iq4nl[q.x >> 4u]) * xhi.x
                    + float(kvalues_iq4nl[q.y >> 4u]) * xhi.y
                    + float(kvalues_iq4nl[q.z >> 4u]) * xhi.z
                    + float(kvalues_iq4nl[q.w >> 4u]) * xhi.w;
            }
            acc += dl * (s1 + s2);
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
