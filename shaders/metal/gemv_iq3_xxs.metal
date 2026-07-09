#include <metal_stdlib>
using namespace metal;

// y[M] = W[M,K] · x[K], W in raw ggml IQ3_XXS super-blocks (98 B per 256).
// Port of shaders/gemv_iq3_xxs.comp. 3.0625 bpw: qs[0..63] are grid-codebook
// indices (4 magnitudes per byte via iq3xxs_grid), qs[64..95] hold one aux
// u32 per 32 elements: 4x 7-bit sign patterns + 4-bit scale. One 8-element
// group per work unit.

#include "iq_tables.metal"

constant uint TPR [[function_constant(0)]];

struct block_iq3_xxs {
    half  d;
    uchar qs[96];
};
static_assert(sizeof(block_iq3_xxs) == 98, "iq3_xxs block size");

kernel void gemv_iq3_xxs(device const block_iq3_xxs* wb  [[buffer(0)]],
                         device const float*         x   [[buffer(1)]],
                         device float*               y   [[buffer(2)]],
                         constant uint2&             mk  [[buffer(3)]],
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
        const uint nu    = kb * 8u;   // 32-element groups per row
        const ulong base = (ulong)row * kb;
        for (uint u = lane; u < nu; u += TPR) {
            const uint b    = u >> 3u;
            const uint ib32 = u & 7u;  // 32-element group in super-block

            device const block_iq3_xxs& blk = wb[base + b];
            const float d  = float(blk.d);
            const uint  ao = 64u + 4u * ib32;
            const uint aux = uint(blk.qs[ao]) |
                             (uint(blk.qs[ao + 1u]) << 8u) |
                             (uint(blk.qs[ao + 2u]) << 16u) |
                             (uint(blk.qs[ao + 3u]) << 24u);
            const float db = d * (0.5f + float(aux >> 28u)) * 0.5f;

            device const packed_uchar4* gp =
                (device const packed_uchar4*)&blk.qs[ib32 * 8u];
            const uchar4 qa = uchar4(gp[0]);
            const uchar4 qb = uchar4(gp[1]);
            device const float4* xp =
                (device const float4*)(x + b * 256u + ib32 * 32u);

            float s = 0.0f;
            for (uint l = 0u; l < 4u; ++l) {
                const uint signs = iq_signbyte((aux >> (7u * l)) & 127u);
                const uint g1 = iq3xxs_grid[l < 2u ? (l == 0u ? qa.x : qa.z)
                                                   : (l == 2u ? qb.x : qb.z)];
                const uint g2 = iq3xxs_grid[l < 2u ? (l == 0u ? qa.y : qa.w)
                                                   : (l == 2u ? qb.y : qb.w)];
                const float4 m1 = float4(uint4(g1, g1 >> 8u, g1 >> 16u, g1 >> 24u) & 255u);
                const float4 m2 = float4(uint4(g2, g2 >> 8u, g2 >> 16u, g2 >> 24u) & 255u);
                const float4 s1 = select(float4(1.0f), float4(-1.0f),
                                         bool4(signs & 1u, signs & 2u, signs & 4u, signs & 8u));
                const float4 s2 = select(float4(1.0f), float4(-1.0f),
                                         bool4(signs & 16u, signs & 32u, signs & 64u, signs & 128u));
                s += dot(m1 * s1, xp[2u * l]) + dot(m2 * s2, xp[2u * l + 1u]);
            }
            acc += db * s;
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
