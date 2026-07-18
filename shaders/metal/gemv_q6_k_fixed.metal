#include <metal_stdlib>
using namespace metal;

// Fixed-K output-head specialization.  Both supported models use K=2048 and
// a vocabulary divisible by the NSG=2 four-row threadgroup tile.
constant uint NSG [[function_constant(0)]];

struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K block size");

kernel void gemv_q6_k_k2048(device const block_q6_K* wb [[buffer(0)]],
                            device const float* x [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant uint2& mk [[buffer(3)]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            uint sgid [[simdgroup_index_in_threadgroup]],
                            uint slid [[thread_index_in_simdgroup]]) {
    const uint M = mk.x;
    const uint firstRow = (tgpig.x * NSG + sgid) * 2u;
    const short tid = slid / 2;
    const short ix = slid % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const uint yOffset = 128u * ip + l0;
    const uint qOffsetL = 64u * ip + l0;
    const uint qOffsetH = 32u * ip + l0;
    device const float* yy = x + (ulong)tgpig.z * 2048u;

    float sumf[2] = {0.0f, 0.0f};
    float yl[16];

#pragma unroll
    for (uint i = ix; i < 8u; i += 2u) {
        device const float* yb = yy + i * 256u + yOffset;
#pragma unroll
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yb[l +  0];
            yl[4*l + 1] = yb[l + 32];
            yl[4*l + 2] = yb[l + 64];
            yl[4*l + 3] = yb[l + 96];
        }

#pragma unroll
        for (uint row = 0u; row < 2u; ++row) {
            device const block_q6_K& blk = wb[(ulong)(firstRow + row) * 8u + i];
            device const uchar* q1 = blk.ql + qOffsetL;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh = blk.qh + qOffsetH;
            device const char* sc = blk.scales + is;

            float4 sums = 0.0f;
#pragma unroll
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] *
                    float((char)((q1[l] & 0xFu) | ((qh[l] & 0x03u) << 4u)) - 32);
                sums[1] += yl[4*l + 1] *
                    float((char)((q2[l] & 0xFu) | ((qh[l] & 0x0Cu) << 2u)) - 32);
                sums[2] += yl[4*l + 2] *
                    float((char)((q1[l] >> 4u) | (qh[l] & 0x30u)) - 32);
                sums[3] += yl[4*l + 3] *
                    float((char)((q2[l] >> 4u) | ((qh[l] & 0xC0u) >> 2u)) - 32);
            }
            sumf[row] += float(blk.d) *
                (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

#pragma unroll
    for (uint row = 0u; row < 2u; ++row) {
        const float s = simd_sum(sumf[row]);
        if (slid == 0u) y[(ulong)tgpig.z * M + firstRow + row] = s;
    }
}
