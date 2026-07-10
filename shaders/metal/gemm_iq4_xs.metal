#include <metal_stdlib>
using namespace metal;

#include "iq_tables.metal"

// y[N,M] = W[M,K] x X[N,K]^T with W in raw ggml IQ4_XS super-blocks.
// Packed-block skeleton of gemm_q8_0_hp (64x32 tile, K-chunk 32, 128
// threads, every simdgroup_load contiguous stride-8); the only change is
// the W staging: each thread dequants 16 elems = half an IQ4_XS 32-elem
// sub-block (low nibbles -> elems 0..15, high -> 16..31).
// f16 staging, f32 accumulators (same class as the q8 twin).

struct GemmPC { uint M; uint K; uint N; };

struct block_iq4_xs {
    half   d;
    ushort scales_h;
    uchar  scales_l[4];
    uchar  qs[128];
};
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs block size");

kernel void gemm_iq4_xs_hp(device const block_iq4_xs* wb [[buffer(0)]],
                           device const float*        x  [[buffer(1)]],
                           device float*              y  [[buffer(2)]],
                           constant GemmPC&           pc [[buffer(3)]],
                           uint3 tid3  [[thread_position_in_threadgroup]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           uint  sgid  [[simdgroup_index_in_threadgroup]],
                           uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;              // 0..127 (4 simdgroups)
    const uint rowBase = tgpig.x * 64u;
    const uint tokBase = tgpig.z * 32u;
    const uint kb = pc.K >> 5;            // 32-elem sub-blocks per row

    threadgroup half  Wsh[64u * 32u];
    threadgroup half  Xsh[32u * 32u];
    threadgroup float outb[4u * 72u];

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; ++i)
        for (uint j = 0u; j < 4u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    const uint srow = tid >> 1, shalf = tid & 1u;
    threadgroup half* wsh = Wsh + ((srow >> 3) * 4u) * 64u + (srow & 7u) * 8u;

    for (uint b0 = 0u; b0 < kb; ++b0) {
        {
            const uint wr = rowBase + srow;
            if (wr < pc.M) {
                device const block_iq4_xs& blk = wb[(ulong)wr * (kb >> 3) + (b0 >> 3)];
                const uint ib = b0 & 7u;
                const uint slb = uint(blk.scales_l[ib >> 1u]);
                const uint sh  = uint(blk.scales_h);
                const int  ls  = int(((slb >> (4u * (ib & 1u))) & 0xFu) |
                                     (((sh >> (2u * ib)) & 3u) << 4u)) - 32;
                const float dl = float(blk.d) * float(ls);
                device const packed_uchar4* qp =
                    (device const packed_uchar4*)&blk.qs[ib * 16u];
                for (uint l = 0u; l < 2u; ++l) {
                    const uchar4 q0 = uchar4(qp[2u * l]);
                    const uchar4 q1 = uchar4(qp[2u * l + 1u]);
                    const uint4 n0 = shalf ? (uint4(q0) >> 4u) : (uint4(q0) & 0xFu);
                    const uint4 n1 = shalf ? (uint4(q1) >> 4u) : (uint4(q1) & 0xFu);
                    threadgroup half4* dst4 =
                        (threadgroup half4*)(wsh + (shalf * 2u + l) * 64u);
                    dst4[0] = half4(dl * float4(float(kvalues_iq4nl[n0.x]), float(kvalues_iq4nl[n0.y]),
                                                float(kvalues_iq4nl[n0.z]), float(kvalues_iq4nl[n0.w])));
                    dst4[1] = half4(dl * float4(float(kvalues_iq4nl[n1.x]), float(kvalues_iq4nl[n1.y]),
                                                float(kvalues_iq4nl[n1.z]), float(kvalues_iq4nl[n1.w])));
                }
            } else {
                for (uint l = 0u; l < 2u; ++l) {
                    threadgroup half4* dst4 =
                        (threadgroup half4*)(wsh + (shalf * 2u + l) * 64u);
                    dst4[0] = half4(0.0h);
                    dst4[1] = half4(0.0h);
                }
            }
        }
        {   // X: 32 tokens x 32 K, thread -> (token, k-block)
            const uint tt = tid >> 2, xkb = tid & 3u;
            const uint xt = tokBase + tt;
            const uint xk = (b0 << 5) + xkb * 8u;
            threadgroup half* xd = &Xsh[(xkb * 4u + (tt >> 3)) * 64u + (tt & 7u)];
            if (xt < pc.N && xk < pc.K) {
                device const float* xp = x + (ulong)xt * pc.K + xk;
                const float4 a = *(device const packed_float4*)xp;
                const float4 b = *(device const packed_float4*)(xp + 4u);
                xd[0u]  = half(a.x); xd[8u]  = half(a.y);
                xd[16u] = half(a.z); xd[24u] = half(a.w);
                xd[32u] = half(b.x); xd[40u] = half(b.y);
                xd[48u] = half(b.z); xd[56u] = half(b.w);
            } else {
                for (uint j = 0u; j < 8u; ++j) xd[j * 8u] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kf = 0u; kf < 4u; ++kf) {
            simdgroup_half8x8 a[2];
            simdgroup_load(a[0], &Wsh[((sgid * 2u + 0u) * 4u + kf) * 64u], 8u);
            simdgroup_load(a[1], &Wsh[((sgid * 2u + 1u) * 4u + kf) * 64u], 8u);
            simdgroup_barrier(mem_flags::mem_none);
            for (uint nc = 0u; nc < 4u; ++nc) {
                simdgroup_half8x8 bfr;
                simdgroup_load(bfr, &Xsh[(kf * 4u + nc) * 64u], 8u);
                simdgroup_multiply_accumulate(acc[0][nc], a[0], bfr, acc[0][nc]);
                simdgroup_multiply_accumulate(acc[1][nc], a[1], bfr, acc[1][nc]);
            }
            simdgroup_barrier(mem_flags::mem_none);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* buf = &outb[sgid * 72u];
    const uint fi = slid >> 2, fj0 = (slid & 3u) * 2u;
    for (uint mr = 0u; mr < 2u; ++mr) {
        for (uint nc = 0u; nc < 4u; ++nc) {
            simdgroup_store(acc[mr][nc], buf, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint row = rowBase + (sgid * 2u + mr) * 8u + fi;
            const uint tok0 = tokBase + nc * 8u;
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint tok = tok0 + fj0 + jj;
                if (row < pc.M && tok < pc.N)
                    y[(ulong)tok * pc.M + row] = buf[fi * 9u + fj0 + jj];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
