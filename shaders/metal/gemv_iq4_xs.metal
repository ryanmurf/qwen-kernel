#include <metal_stdlib>
using namespace metal;

// y[M] = W[M,K] · x[K], W in raw ggml IQ4_XS super-blocks (136 B per 256).
//
// v2, after the M2 kernel face-off: adopts the llama.cpp Metal work shape
// (kernel_mul_mv_iq4_xs_f32, MIT) — one simdgroup per PAIR of consecutive
// rows, codebook staged in threadgroup memory (32 floats), qs read as
// uint32 with 0x0f0f0f0f nibble masks and byte-aliased register lookups,
// x staged in registers and reused across both rows. Was 320 GB/s (TPR
// scheme), llama.cpp 411. Geometry: NSG simdgroups per threadgroup
// (function constant 0; llama.cpp ships 2), NR0=2 rows per simdgroup;
// tgMem = 128 B for the staged codebook.

#include "iq_tables.metal"

constant uint NSG [[function_constant(0)]];  // simdgroups per threadgroup

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
                        uint3 tgpig [[threadgroup_position_in_grid]],
                        uint  sgid  [[simdgroup_index_in_threadgroup]],
                        uint  slid  [[thread_index_in_simdgroup]])
{
    const uint M   = mk.x;
    const uint K   = mk.y;
    const uint nb  = K / 256u;
    const uint first_row = (tgpig.x * NSG + sgid) * 2u;
    const uint rq  = tgpig.z;            // grid z batches queries
    x += (ulong)rq * K;
    y += (ulong)rq * M;

    threadgroup float shf[32];

    // stage the nonlinear codebook once per threadgroup
    if (sgid == 0) shf[slid] = (float)kvalues_iq4nl[slid % 16u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (first_row >= M) return;
    const uint nrow = min(2u, M - first_row);

    const short ix = slid / 16;  // block parity
    const short it = slid % 16;
    const short ib = it / 2;     // sub-block 0..7
    const short il = it % 2;     // 8-element half of the 32

    device const float* yb = x + ix * 256u + ib * 32u + il * 8u;

    float4 yl[4];
    float sumf[2] = {0.f, 0.f};

    uint32_t aux32[2];
    thread const uchar* q8 = (thread const uchar*)aux32;

    for (uint ibl = ix; ibl < nb; ibl += 2u) {
        device const float4* y4 = (device const float4*)yb;
        yl[0] = y4[0];   // elems [0..4)  of the low nibbles
        yl[1] = y4[4];   // elems [0..4)  of the high nibbles
        yl[2] = y4[1];   // elems [4..8)  low
        yl[3] = y4[5];   // elems [4..8)  high

        for (uint row = 0; row < nrow; ++row) {
            device const block_iq4_xs& xb = wb[(ulong)(first_row + row) * nb + ibl];
            device const uint32_t* q4 =
                (device const uint32_t*)(xb.qs + 16u * ib + 8u * il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]      ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4u) & 0x0f0f0f0f;
            acc1 += yl[0] * float4(shf[q8[0]], shf[q8[1]], shf[q8[2]], shf[q8[3]]);
            acc2 += yl[1] * float4(shf[q8[4]], shf[q8[5]], shf[q8[6]], shf[q8[7]]);

            aux32[0] = (q4[1]      ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4u) & 0x0f0f0f0f;
            acc1 += yl[2] * float4(shf[q8[0]], shf[q8[1]], shf[q8[2]], shf[q8[3]]);
            acc2 += yl[3] * float4(shf[q8[4]], shf[q8[5]], shf[q8[6]], shf[q8[7]]);

            acc1 += acc2;

            const int ls = (int)(((xb.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                                 (((xb.scales_h >> (2 * ib)) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 512;  // two super-blocks of x
    }

    for (uint row = 0; row < nrow; ++row) {
        const float s = simd_sum(sumf[row]);
        if (slid == 0) y[first_row + row] = s;
    }
}
