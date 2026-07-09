#include <metal_stdlib>
using namespace metal;

// Batched (prefill) GEMM: Y[N,M] = X[N,K] · W[M,K]^T, W in raw ggml Q8_0.
// Port of shaders/gemm_q8_0.comp, restructured for Apple's 32 KB
// threadgroup-memory limit: BK = 32 (one Q8_0 block deep per stage) instead
// of RDNA3's 64, so Wsh(128×33) + Xsh(64×33) = 25.3 KB. 16×16 threads, each
// accumulating an 8×4 micro-tile. Grid: x = ceil(M/128), z = ceil(N/64).

struct GemmPC { uint M; uint K; uint N; };

struct block_q8_0 {
    half d;
    char qs[32];
};

constant uint BM = 128u, BN = 64u, RM = 8u, RN = 4u, SK = 33u;

kernel void gemm_q8_0(device const block_q8_0* wb [[buffer(0)]],
                      device const float*      x  [[buffer(1)]],
                      device float*            y  [[buffer(2)]],
                      constant GemmPC&         pc [[buffer(3)]],
                      uint3 tid3  [[thread_position_in_threadgroup]],
                      uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint tid = tid3.x;
    const uint tr = tid >> 4;          // 0..15
    const uint tc = tid & 15u;         // 0..15
    const uint rowBase = tgpig.x * BM;
    const uint tokBase = tgpig.z * BN;
    const uint kb = pc.K >> 5;

    threadgroup float Wsh[BM * SK];
    threadgroup float Xsh[BN * SK];

    float acc[RM][RN];
    for (uint i = 0u; i < RM; ++i)
        for (uint j = 0u; j < RN; ++j) acc[i][j] = 0.0f;

    for (uint b = 0u; b < kb; ++b) {
        // threads 0..127 dequantize one W row-block each; all threads then
        // cooperatively stage the 64×32 activation tile
        if (tid < BM) {
            const uint wr = rowBase + tid;
            const uint woff = tid * SK;
            if (wr < pc.M) {
                device const block_q8_0& blk = wb[(ulong)wr * kb + b];
                const float d = float(blk.d);
                device const packed_char4* qp = (device const packed_char4*)blk.qs;
                for (uint i = 0u; i < 8u; ++i) {
                    const char4 q = char4(qp[i]);
                    Wsh[woff + 4u*i + 0u] = d * float(q.x);
                    Wsh[woff + 4u*i + 1u] = d * float(q.y);
                    Wsh[woff + 4u*i + 2u] = d * float(q.z);
                    Wsh[woff + 4u*i + 3u] = d * float(q.w);
                }
            } else {
                for (uint i = 0u; i < 32u; ++i) Wsh[woff + i] = 0.0f;
            }
        }
        for (uint idx = tid; idx < BN * 32u; idx += 256u) {
            const uint tt = idx >> 5;
            const uint kk = idx & 31u;
            const uint xt = tokBase + tt;
            const uint xk = (b << 5) + kk;
            Xsh[tt * SK + kk] = (xt < pc.N && xk < pc.K) ? x[(ulong)xt * pc.K + xk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint mi = 0u; mi < RM; ++mi) {
            const uint wo = (tr * RM + mi) * SK;
            const uint xo0 = (tc * RN + 0u) * SK;
            const uint xo1 = (tc * RN + 1u) * SK;
            const uint xo2 = (tc * RN + 2u) * SK;
            const uint xo3 = (tc * RN + 3u) * SK;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (uint i = 0u; i < 32u; ++i) {
                const float wv = Wsh[wo + i];
                s0 += wv * Xsh[xo0 + i];
                s1 += wv * Xsh[xo1 + i];
                s2 += wv * Xsh[xo2 + i];
                s3 += wv * Xsh[xo3 + i];
            }
            acc[mi][0] += s0;
            acc[mi][1] += s1;
            acc[mi][2] += s2;
            acc[mi][3] += s3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint mi = 0u; mi < RM; ++mi) {
        const uint row = rowBase + tr * RM + mi;
        for (uint mj = 0u; mj < RN; ++mj) {
            const uint tok = tokBase + tc * RN + mj;
            if (row < pc.M && tok < pc.N) y[(ulong)tok * pc.M + row] = acc[mi][mj];
        }
    }
}
