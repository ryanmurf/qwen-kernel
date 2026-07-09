#include <metal_stdlib>
using namespace metal;

// Embedding lookup: dequantize token_embd row ids[idx] (Q6_K) into x.
// One threadgroup; thread per 16-element scale group; grid z batches.
// Port of shaders/embed_q6k.comp.

struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char  scales[16];
    half  d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K block size");

struct EmbPC { uint kdim; uint idx; uint perReq; };

kernel void embed_q6k(device const block_q6_K* wb  [[buffer(0)]],
                      device const uint*       ids [[buffer(1)]],
                      device float*            x   [[buffer(2)]],
                      constant EmbPC&          pc  [[buffer(3)]],
                      uint3 tid3  [[thread_position_in_threadgroup]],
                      uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint t  = tid3.x;
    const uint kb = pc.kdim / 256u;
    const uint ng = kb * 16u;
    const uint rq = tgpig.z;
    const uint ro = rq * pc.kdim;
    const ulong base = (ulong)ids[pc.idx + pc.perReq * rq] * kb;
    for (uint g = t; g < ng; g += 256u) {
        const uint b = g >> 4u, gg = g & 15u;
        const uint h = gg >> 3u, r = (gg & 7u) >> 1u, is = gg & 1u;
        device const block_q6_K& blk = wb[base + b];
        const float d  = float(blk.d);
        const float sc = float(int(blk.scales[h * 8u + r * 2u + is]));
        const uint qlBase = h * 64u + (r & 1u) * 32u + is * 16u;
        const uint qhBase = h * 32u + is * 16u;
        const uint shift = r * 2u;
        const bool hi = r >= 2u;
        const uint xo = b * 256u + gg * 16u;
        for (uint i = 0u; i < 16u; ++i) {
            const uint qlv = uint(blk.ql[qlBase + i]);
            const uint qhv = uint(blk.qh[qhBase + i]);
            const uint lo = hi ? (qlv >> 4u) : (qlv & 0xFu);
            const int  q = int(lo | (((qhv >> shift) & 3u) << 4u)) - 32;
            x[ro + xo + i] = d * sc * float(q);
        }
    }
}
