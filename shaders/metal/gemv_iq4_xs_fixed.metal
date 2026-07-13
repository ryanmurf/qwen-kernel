#include <metal_stdlib>
using namespace metal;

#include "iq_tables.metal"

constant uint NSG [[function_constant(0)]];

// The 80B dense projection families use K={2048,4096} and M divisible by the
// NSG=2 four-row threadgroup tile.  Specializing those invariants removes the
// row-tail branches and fully unrolls the two- or four-iteration block loop.

struct block_iq4_xs {
    half d;
    ushort scales_h;
    uchar scales_l[4];
    uchar qs[128];
};
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs block size");

template <uint KFIX>
static inline void gemv_iq4_fixed_body(device const block_iq4_xs* wb,
                                       device const float* x,
                                       device float* y,
                                       constant uint2& mk,
                                       threadgroup float* shf,
                                       uint tgx, uint tgz,
                                       uint sgid, uint slid) {
    const uint M = mk.x;
    const uint nb = KFIX / 256u;
    const uint firstRow = (tgx * NSG + sgid) * 2u;

    if (sgid == 0u) shf[slid] = float(kvalues_iq4nl[slid % 16u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short ix = slid / 16;
    const short it = slid % 16;
    const short ib = it / 2;
    const short il = it % 2;
    device const float* yb = x + (ulong)tgz * KFIX + ix * 256u
                                 + ib * 32u + il * 8u;

    float4 yl[4];
    float sumf[2] = {0.0f, 0.0f};
    uint32_t aux32[2];
    thread const uchar* q8 = (thread const uchar*)aux32;

#pragma unroll
    for (uint ibl = ix; ibl < nb; ibl += 2u) {
        device const float4* y4 = (device const float4*)yb;
        yl[0] = y4[0]; yl[1] = y4[4];
        yl[2] = y4[1]; yl[3] = y4[5];

#pragma unroll
        for (uint row = 0u; row < 2u; ++row) {
            device const block_iq4_xs& xb = wb[(ulong)(firstRow + row) * nb + ibl];
            device const uint32_t* q4 =
                (device const uint32_t*)(xb.qs + 16u * ib + 8u * il);
            float4 acc1 = 0.0f, acc2 = 0.0f;

            aux32[0] = q4[0] & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4u) & 0x0f0f0f0f;
            acc1 += yl[0] * float4(shf[q8[0]], shf[q8[1]], shf[q8[2]], shf[q8[3]]);
            acc2 += yl[1] * float4(shf[q8[4]], shf[q8[5]], shf[q8[6]], shf[q8[7]]);
            aux32[0] = q4[1] & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4u) & 0x0f0f0f0f;
            acc1 += yl[2] * float4(shf[q8[0]], shf[q8[1]], shf[q8[2]], shf[q8[3]]);
            acc2 += yl[3] * float4(shf[q8[4]], shf[q8[5]], shf[q8[6]], shf[q8[7]]);
            acc1 += acc2;

            const int ls = int(((xb.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xFu) |
                               (((xb.scales_h >> (2 * ib)) & 3u) << 4u)) - 32;
            sumf[row] += float(xb.d) * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }
        yb += 512;
    }

#pragma unroll
    for (uint row = 0u; row < 2u; ++row) {
        const float s = simd_sum(sumf[row]);
        if (slid == 0u) y[(ulong)tgz * M + firstRow + row] = s;
    }
}

kernel void gemv_iq4_xs_k2048(device const block_iq4_xs* wb [[buffer(0)]],
                              device const float* x [[buffer(1)]],
                              device float* y [[buffer(2)]],
                              constant uint2& mk [[buffer(3)]],
                              uint3 tgpig [[threadgroup_position_in_grid]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint slid [[thread_index_in_simdgroup]]) {
    threadgroup float shf[32];
    gemv_iq4_fixed_body<2048u>(wb, x, y, mk, shf, tgpig.x, tgpig.z, sgid, slid);
}

kernel void gemv_iq4_xs_k4096(device const block_iq4_xs* wb [[buffer(0)]],
                              device const float* x [[buffer(1)]],
                              device float* y [[buffer(2)]],
                              constant uint2& mk [[buffer(3)]],
                              uint3 tgpig [[threadgroup_position_in_grid]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint slid [[thread_index_in_simdgroup]]) {
    threadgroup float shf[32];
    gemv_iq4_fixed_body<4096u>(wb, x, y, mk, shf, tgpig.x, tgpig.z, sgid, slid);
}
