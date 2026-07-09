#include <metal_stdlib>
using namespace metal;

// y[M] = W[M,K] · x[K], W in raw ggml Q6_K super-blocks (210 B per 256).
//
// v2, after the M2 kernel face-off: adopts the llama.cpp Metal work shape
// (kernel_mul_mv_q6_K_f32, MIT) — one simdgroup per PAIR of consecutive
// rows, each lane owning 4 consecutive byte positions in all four block
// quadrants with constant masks (no variable shifts), x staged in registers
// and reused across both rows, two blocks in flight per simdgroup, plain
// simd_sum, no threadgroup memory. Was 354 GB/s (TPR scheme), llama.cpp
// 476; this shape closes the gap. Geometry: NSG simdgroups per threadgroup
// (function constant 0; llama.cpp ships 2), NR0=2 rows per simdgroup.
//
// Grid z batches queries (slots) like the Vulkan engine dispatch.

constant uint NSG [[function_constant(0)]];  // simdgroups per threadgroup

struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char  scales[16];
    half  d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K block size");

kernel void gemv_q6_k(device const block_q6_K* wb  [[buffer(0)]],
                      device const float*      x   [[buffer(1)]],
                      device float*            y   [[buffer(2)]],
                      constant uint2&          mk  [[buffer(3)]],
                      uint3 tgpig [[threadgroup_position_in_grid]],
                      uint  sgid  [[simdgroup_index_in_threadgroup]],
                      uint  slid  [[thread_index_in_simdgroup]])
{
    const uint M   = mk.x;
    const uint K   = mk.y;
    const uint rq  = tgpig.z;
    const uint nb  = K / 256u;
    const uint first_row = (tgpig.x * NSG + sgid) * 2u;
    if (first_row >= M) return;
    const uint nrow = min(2u, M - first_row);

    const short tid = slid / 2;      // 0..15
    const short ix  = slid % 2;      // block parity
    const short ip  = tid / 8;       // 0 or 1: which 128-half
    const short il  = tid % 8;       // 4-byte group within the half
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const uint y_offset   = 128u * ip + l0;
    const uint q_offset_l =  64u * ip + l0;
    const uint q_offset_h =  32u * ip + l0;

    device const float* yy = x + rq * K;

    float sumf[2] = {0.f, 0.f};
    float yl[16];

    for (uint i = ix; i < nb; i += 2u) {
        device const float* yb = yy + i * 256u + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yb[l +  0];
            yl[4*l + 1] = yb[l + 32];
            yl[4*l + 2] = yb[l + 64];
            yl[4*l + 3] = yb[l + 96];
        }

        for (uint row = 0; row < nrow; ++row) {
            device const block_q6_K& blk = wb[(ulong)(first_row + row) * nb + i];
            device const uchar* q1 = blk.ql + q_offset_l;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh = blk.qh + q_offset_h;
            device const char*  sc = blk.scales + is;

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * (float)((char)((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                sums[1] += yl[4*l + 1] * (float)((char)((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                sums[2] += yl[4*l + 2] * (float)((char)((q1[l] >>  4) | ((qh[l] & 0x30) << 0)) - 32);
                sums[3] += yl[4*l + 3] * (float)((char)((q2[l] >>  4) | ((qh[l] & 0xC0) >> 2)) - 32);
            }
            sumf[row] += (float)blk.d * (sums[0] * sc[0] + sums[1] * sc[2] +
                                         sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    for (uint row = 0; row < nrow; ++row) {
        const float s = simd_sum(sumf[row]);
        if (slid == 0) y[(ulong)rq * M + first_row + row] = s;
    }
}
