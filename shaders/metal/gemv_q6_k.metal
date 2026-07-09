#include <metal_stdlib>
using namespace metal;

// y[M] = W[M,K] · x[K], W in raw ggml Q6_K super-blocks (210 B per 256).
// Port of shaders/gemv_q6_k.comp: one 16-element scale group per work unit.
//
// Per ggml dequantize_row_q6_K, for output o in 0..255: h=o/128, r=(o%128)/32,
// l=o%32, is=l/16 ->
//   q = lo4(ql[h*64 + (r&1)*32 + l], hi=r>=2) | ((qh[h*32+l] >> 2r) & 3) << 4, -32
//   y[o] = d * scales[h*8 + r*2 + is] * q

constant uint TPR [[function_constant(0)]];

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
                      uint3 tid3  [[thread_position_in_threadgroup]],
                      uint3 tgpig [[threadgroup_position_in_grid]],
                      uint  sgid  [[simdgroup_index_in_threadgroup]],
                      uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid  = tid3.x;
    const uint M    = mk.x;
    const uint K    = mk.y;
    const uint rq   = tgpig.z;
    const uint xo2  = rq * K;
    const uint lane = tid % TPR;
    const uint row  = tgpig.x * (256u / TPR) + tid / TPR;

    float acc = 0.0f;
    if (row < M) {
        const uint kb    = K / 256u;  // super-blocks per row
        const uint ng    = kb * 16u;  // 16-element groups per row
        const ulong base = (ulong)row * kb;
        for (uint g = lane; g < ng; g += TPR) {
            const uint b  = g >> 4u;
            const uint gg = g & 15u;
            const uint h  = gg >> 3u;
            const uint r  = (gg & 7u) >> 1u;
            const uint is = gg & 1u;

            device const block_q6_K& blk = wb[base + b];
            const float d  = float(blk.d);
            const float sc = float(int(blk.scales[h * 8u + r * 2u + is]));

            const uint qlBase = h * 64u + (r & 1u) * 32u + is * 16u;
            const uint qhBase = h * 32u + is * 16u;
            const uint shift  = r * 2u;
            const bool hi     = r >= 2u;

            device const packed_uchar4* qlp = (device const packed_uchar4*)&blk.ql[qlBase];
            device const packed_uchar4* qhp = (device const packed_uchar4*)&blk.qh[qhBase];
            device const float4* xp =
                (device const float4*)(x + xo2 + b * 256u + gg * 16u);

            float s = 0.0f;
            for (uint i = 0u; i < 4u; ++i) {
                const uchar4 qlv = uchar4(qlp[i]);
                const uchar4 qhv = uchar4(qhp[i]);
                const uint4 lo = hi ? (uint4(qlv) >> 4u) : (uint4(qlv) & 0xFu);
                const int4  q  = int4(lo | (((uint4(qhv) >> shift) & 3u) << 4u)) - 32;
                s += dot(float4(q), xp[i]);
            }
            acc += d * sc * s;
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
    if (lane == 0u && row < M) y[(ulong)rq * M + row] = acc;
}
