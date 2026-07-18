#include <metal_stdlib>
using namespace metal;

// y[M] = W[M,K] · x[K], W in raw ggml IQ3_XXS super-blocks (98 B per 256).
//
// v2, after the M2 kernel face-off: adopts the llama.cpp Metal work shape
// (kernel_mul_mv_iq3_xxs_f32, MIT) — one simdgroup per FOUR consecutive
// rows, grid codebook (256 u32) and sign bytes (128) staged in threadgroup
// memory, each lane owning one 32-element group with x staged in registers
// and reused across the four rows. Geometry: NSG simdgroups per
// threadgroup (function constant 0; llama.cpp ships 2), NR0=4 rows per
// simdgroup; tgMem = 1152 B for grid + signs.

#include "iq_tables.metal"

constant uint NSG [[function_constant(0)]];  // simdgroups per threadgroup

struct block_iq3_xxs {
    half  d;
    uchar qs[96];
};
static_assert(sizeof(block_iq3_xxs) == 98, "iq3_xxs block size");

kernel void gemv_iq3_xxs(device const block_iq3_xxs* wb  [[buffer(0)]],
                         device const float*         x   [[buffer(1)]],
                         device float*               y   [[buffer(2)]],
                         constant uint2&             mk  [[buffer(3)]],
                         threadgroup uchar*          shm [[threadgroup(0)]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint  sgid  [[simdgroup_index_in_threadgroup]],
                         uint  slid  [[thread_index_in_simdgroup]])
{
    const uint M   = mk.x;
    const uint K   = mk.y;
    const uint nb  = K / 256u;

    threadgroup uint*  svalues = (threadgroup uint*)shm;
    threadgroup uchar* ssigns  = (threadgroup uchar*)(svalues + 256);
    {
        const uint tid = sgid * 32u + slid;
        const uint nth = NSG * 32u;
        for (uint i = tid; i < 256u; i += nth) svalues[i] = iq3xxs_grid[i];
        for (uint i = tid; i < 128u; i += nth) ssigns[i] = (uchar)iq_signbyte(i);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint first_row = (tgpig.x * NSG + sgid) * 4u;
    if (first_row >= M) return;
    const uint nrow = min(4u, M - first_row);

    const uint nb32 = nb * 8u;  // 32-element groups per row

    float4 yl[8];
    float  sumf[4] = {0.f, 0.f, 0.f, 0.f};

    for (uint ib32 = slid; ib32 < nb32; ib32 += 32u) {
        device const float4* y4 = (device const float4*)(x + 32u * ib32);
        for (short i = 0; i < 8; ++i) yl[i] = y4[i];

        const uint ibl = ib32 / 8u;  // super-block
        const uint ib  = ib32 % 8u;  // 32-group within it

        for (uint row = 0; row < nrow; ++row) {
            device const block_iq3_xxs& xb = wb[(ulong)(first_row + row) * nb + ibl];
            device const uchar*  q3  = xb.qs + 8u * ib;
            device const ushort* gas = (device const ushort*)(xb.qs + 64u) + 2u * ib;

            const uint aux32 = (uint)gas[0] | ((uint)gas[1] << 16);
            const float d = (float)xb.d * (0.5f + (float)(aux32 >> 28u));

            float sum = 0.f;
            for (short l = 0; l < 4; ++l) {
                const uint signs = ssigns[(aux32 >> (7u * l)) & 127u];
                const uint g1 = svalues[q3[2*l + 0]];
                const uint g2 = svalues[q3[2*l + 1]];
                const float4 m1 = float4(uint4(g1, g1 >> 8u, g1 >> 16u, g1 >> 24u) & 255u);
                const float4 m2 = float4(uint4(g2, g2 >> 8u, g2 >> 16u, g2 >> 24u) & 255u);
                const float4 s1 = select(float4(1.0f), float4(-1.0f),
                                         bool4(signs & 1u, signs & 2u, signs & 4u, signs & 8u));
                const float4 s2 = select(float4(1.0f), float4(-1.0f),
                                         bool4(signs & 16u, signs & 32u, signs & 64u, signs & 128u));
                sum += dot(m1 * s1, yl[2*l]) + dot(m2 * s2, yl[2*l + 1]);
            }
            sumf[row] += d * sum;
        }
    }

    for (uint row = 0; row < nrow; ++row) {
        const float s = simd_sum(sumf[row]);
        if (slid == 0) y[first_row + row] = s * 0.5f;
    }
}
