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

// simdgroup_matrix variant: same tiling contract (grid x = ceil(M/128),
// z = ceil(N/64)), BM=128 x BN=64 per 4-simdgroup threadgroup, BK=32.
// Each simdgroup owns a 32-row band (4 fragment-rows x 8 fragment-cols of
// 8x8 f32 fragments, f32 accumulate — bitwise-precision class as the
// scalar kernel, so the prefill argmax-exactness bar is unchanged).
kernel void gemm_q8_0_sg(device const block_q8_0* wb [[buffer(0)]],
                         device const float*      x  [[buffer(1)]],
                         device float*            y  [[buffer(2)]],
                         constant GemmPC&         pc [[buffer(3)]],
                         uint3 tid3  [[thread_position_in_threadgroup]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint  sgid  [[simdgroup_index_in_threadgroup]],
                         uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;              // 0..127 (4 simdgroups)
    const uint rowBase = tgpig.x * BM;    // BM = 128
    const uint tokBase = tgpig.z * BN;    // BN = 64
    const uint kb = pc.K >> 5;

    threadgroup float Wsh[BM * SK];       // SK = 33
    threadgroup float Xsh[BN * SK];

    simdgroup_float8x8 acc[4][8];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 8; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint b = 0u; b < kb; ++b) {
        // stage: 128 threads dequant one W row-block each (two passes),
        // then cooperatively load the 64x32 X tile
        for (uint r = tid; r < BM; r += 128u) {
            const uint wr = rowBase + r;
            const uint woff = r * SK;
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
        for (uint idx = tid; idx < BN * 32u; idx += 128u) {
            const uint tt = idx >> 5;
            const uint kk = idx & 31u;
            const uint xt = tokBase + tt;
            const uint xk = (b << 5) + kk;
            Xsh[tt * SK + kk] = (xt < pc.N && xk < pc.K) ? x[(ulong)xt * pc.K + xk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // simdgroup sgid owns rows [sgid*32, +32): 4 fragment-rows x 8 cols
        for (uint kf = 0u; kf < 4u; ++kf) {
            simdgroup_float8x8 a[4];
            for (uint mr = 0u; mr < 4u; ++mr)
                simdgroup_load(a[mr], &Wsh[(sgid * 32u + mr * 8u) * SK + kf * 8u], SK);
            for (uint nc = 0u; nc < 8u; ++nc) {
                simdgroup_float8x8 bfr;   // [K][N] fragment = transpose of Xsh [N][K]
                simdgroup_load(bfr, &Xsh[(nc * 8u) * SK + kf * 8u], SK, ulong2(0, 0), true);
                for (uint mr = 0u; mr < 4u; ++mr)
                    simdgroup_multiply_accumulate(acc[mr][nc], a[mr], bfr, acc[mr][nc]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // store: Y is [N][M] row-major -> store transposed fragments via a
    // threadgroup bounce (simdgroup_store writes [M][N] order)
    // Simpler: write each fragment to Y with transpose through Wsh reuse.
    for (uint mr = 0u; mr < 4u; ++mr) {
        for (uint nc = 0u; nc < 8u; ++nc) {
            threadgroup float* buf = &Wsh[sgid * (8u * 9u)];
            simdgroup_store(acc[mr][nc], buf, 9u);   // 8x8 fragment, stride 9
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint row0 = rowBase + sgid * 32u + mr * 8u;
            const uint tok0 = tokBase + nc * 8u;
            // 32 lanes: lane -> (i = slid/4 in 0..7 rows of frag, j = slid%4*2 two cols)
            const uint fi = slid >> 2;
            const uint fj = (slid & 3u) * 2u;
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint row = row0 + fi;
                const uint tok = tok0 + fj + jj;
                if (row < pc.M && tok < pc.N)
                    y[(ulong)tok * pc.M + row] = buf[fi * 9u + fj + jj];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// f16-fragment variant: W dequantized to half, X staged as half, f32
// accumulate — llama.cpp Metal's prefill precision class (accepted via
// prefillcmp argmax-exactness + prefilldecode HANDOFF EXACT). 256 threads
// = 8 simdgroups; each owns a 16-row band (2 fragment-rows x 8 cols).
// C fragments store DIRECTLY to Y[N][M] via transposed simdgroup_store —
// no threadgroup bounce. Safe: M is always a multiple of 8 here, and Y
// rows up to ceil(N/64)*64 exist (bb buffers are maxB-sized, 64 | maxB).
constant uint SKH = 68u;   // 8B-aligned rows for half4 staging stores

kernel void gemm_q8_0_h(device const block_q8_0* wb [[buffer(0)]],
                        device const float*      x  [[buffer(1)]],
                        device float*            y  [[buffer(2)]],
                        constant GemmPC&         pc [[buffer(3)]],
                        uint3 tid3  [[thread_position_in_threadgroup]],
                        uint3 tgpig [[threadgroup_position_in_grid]],
                        uint  sgid  [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid3.x;              // 0..255 (8 simdgroups)
    const uint rowBase = tgpig.x * BM;
    const uint tokBase = tgpig.z * BN;
    const uint kb = pc.K >> 5;

    threadgroup half Wsh[BM * SKH];
    threadgroup half Xsh[BN * SKH];

    simdgroup_float8x8 acc[2][8];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 8; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint b0 = 0u; b0 < kb; b0 += 2u) {
        {   // 256 threads stage 128 rows x 2 blocks: one block each
            const uint rr = tid & (BM - 1u);
            const uint bi = tid >> 7;
            const uint wr = rowBase + rr;
            const uint woff = rr * SKH + bi * 32u;
            threadgroup half4* w4 = (threadgroup half4*)&Wsh[woff];
            if (wr < pc.M && b0 + bi < kb) {
                device const block_q8_0& blk = wb[(ulong)wr * kb + b0 + bi];
                const half d = blk.d;
                device const packed_char4* qp = (device const packed_char4*)blk.qs;
                for (uint i = 0u; i < 8u; ++i)
                    w4[i] = d * half4(char4(qp[i]));
            } else {
                for (uint i = 0u; i < 8u; ++i) w4[i] = half4(0.0h);
            }
        }
        for (uint idx = tid * 4u; idx < BN * 64u; idx += 256u * 4u) {
            const uint tt = idx >> 6;
            const uint kk = idx & 63u;
            const uint xt = tokBase + tt;
            const uint xk = (b0 << 5) + kk;
            ((threadgroup half4*)&Xsh[tt * SKH + kk])[0] = (xt < pc.N && xk < pc.K)
                ? half4(*(device const packed_float4*)(x + (ulong)xt * pc.K + xk))
                : half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kf = 0u; kf < 8u; ++kf) {
            simdgroup_half8x8 a[2];
            for (uint mr = 0u; mr < 2u; ++mr)
                simdgroup_load(a[mr], &Wsh[(sgid * 16u + mr * 8u) * SKH + kf * 8u], SKH);
            for (uint nc = 0u; nc < 8u; ++nc) {
                simdgroup_half8x8 bfr;
                simdgroup_load(bfr, &Xsh[(nc * 8u) * SKH + kf * 8u], SKH, ulong2(0, 0), true);
                for (uint mr = 0u; mr < 2u; ++mr)
                    simdgroup_multiply_accumulate(acc[mr][nc], a[mr], bfr, acc[mr][nc]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // store via per-simd threadgroup bounce. (A direct transposed device
    // simdgroup_store was blamed for a 21/36 prefillcmp failure that turned
    // out to be a host thread-count mismatch — the direct store is untested
    // in isolation and worth revisiting; the bounce is proven.)
    threadgroup float outb[8 * 8 * 9];
    for (uint mr = 0u; mr < 2u; ++mr) {
        for (uint nc = 0u; nc < 8u; ++nc) {
            threadgroup float* buf = &outb[sgid * (8u * 9u)];
            simdgroup_store(acc[mr][nc], buf, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint row0 = rowBase + sgid * 16u + mr * 8u;
            const uint tok0 = tokBase + nc * 8u;
            const uint fi = tid3.y * 0u + ((tid & 31u) >> 2);
            const uint fj = ((tid & 31u) & 3u) * 2u;
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint row = row0 + fi;
                const uint tok = tok0 + fj + jj;
                if (row < pc.M && tok < pc.N)
                    y[(ulong)tok * pc.M + row] = buf[fi * 9u + fj + jj];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// bisect variant: half fragments with the EXACT BK=32 staging of the
// passing f32 twin — isolates two-block staging vs half-fragment math.
constant uint SKH2 = 34u;

kernel void gemm_q8_0_h32(device const block_q8_0* wb [[buffer(0)]],
                          device const float*      x  [[buffer(1)]],
                          device float*            y  [[buffer(2)]],
                          constant GemmPC&         pc [[buffer(3)]],
                          uint3 tid3  [[thread_position_in_threadgroup]],
                          uint3 tgpig [[threadgroup_position_in_grid]],
                          uint  sgid  [[simdgroup_index_in_threadgroup]],
                          uint  slid  [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;
    const uint rowBase = tgpig.x * BM;
    const uint tokBase = tgpig.z * BN;
    const uint kb = pc.K >> 5;

    threadgroup half Wsh[BM * SKH2];
    threadgroup half Xsh[BN * SKH2];
    threadgroup float outb[4 * 8 * 9];

    simdgroup_float8x8 acc[4][8];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 8; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint b = 0u; b < kb; ++b) {
        for (uint r = tid; r < BM; r += 128u) {
            const uint wr = rowBase + r;
            const uint woff = r * SKH2;
            if (wr < pc.M) {
                device const block_q8_0& blk = wb[(ulong)wr * kb + b];
                const half d = blk.d;
                device const packed_char4* qp = (device const packed_char4*)blk.qs;
                for (uint i = 0u; i < 8u; ++i) {
                    const char4 q = char4(qp[i]);
                    Wsh[woff + 4u*i + 0u] = d * half(q.x);
                    Wsh[woff + 4u*i + 1u] = d * half(q.y);
                    Wsh[woff + 4u*i + 2u] = d * half(q.z);
                    Wsh[woff + 4u*i + 3u] = d * half(q.w);
                }
            } else {
                for (uint i = 0u; i < 32u; ++i) Wsh[woff + i] = 0.0h;
            }
        }
        for (uint idx = tid; idx < BN * 32u; idx += 128u) {
            const uint tt = idx >> 5;
            const uint kk = idx & 31u;
            const uint xt = tokBase + tt;
            const uint xk = (b << 5) + kk;
            Xsh[tt * SKH2 + kk] = (xt < pc.N && xk < pc.K) ? half(x[(ulong)xt * pc.K + xk]) : 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kf = 0u; kf < 4u; ++kf) {
            simdgroup_half8x8 a[4];
            for (uint mr = 0u; mr < 4u; ++mr)
                simdgroup_load(a[mr], &Wsh[(sgid * 32u + mr * 8u) * SKH2 + kf * 8u], SKH2);
            for (uint nc = 0u; nc < 8u; ++nc) {
                simdgroup_half8x8 bfr;
                simdgroup_load(bfr, &Xsh[(nc * 8u) * SKH2 + kf * 8u], SKH2, ulong2(0, 0), true);
                for (uint mr = 0u; mr < 4u; ++mr)
                    simdgroup_multiply_accumulate(acc[mr][nc], a[mr], bfr, acc[mr][nc]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint mr = 0u; mr < 4u; ++mr) {
        for (uint nc = 0u; nc < 8u; ++nc) {
            threadgroup float* buf = &outb[sgid * (8u * 9u)];
            simdgroup_store(acc[mr][nc], buf, 9u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint row0 = rowBase + sgid * 32u + mr * 8u;
            const uint tok0 = tokBase + nc * 8u;
            const uint fi = slid >> 2;
            const uint fj = (slid & 3u) * 2u;
            for (uint jj = 0u; jj < 2u; ++jj) {
                const uint row = row0 + fi;
                const uint tok = tok0 + fj + jj;
                if (row < pc.M && tok < pc.N)
                    y[(ulong)tok * pc.M + row] = buf[fi * 9u + fj + jj];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Wide-token variant for large-N prefill: BM=64 x BN=128 (vs h's 128x64).
// Same f16 staging/precision class; W-decode per output element HALVES
// because each staged 64-row tile is multiplied against 128 token columns.
// 8 simdgroups: each owns an 8-row band x all 16 column fragments.
// Grid: x = ceil(M/64), z = ceil(N/128). QK_GEMM=h2.
constant uint BM2 = 64u, BN2 = 128u;

kernel void gemm_q8_0_h2(device const block_q8_0* wb [[buffer(0)]],
                         device const float*      x  [[buffer(1)]],
                         device float*            y  [[buffer(2)]],
                         constant GemmPC&         pc [[buffer(3)]],
                         uint3 tid3  [[thread_position_in_threadgroup]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint  sgid  [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid3.x;              // 0..255 (8 simdgroups)
    const uint rowBase = tgpig.x * BM2;
    const uint tokBase = tgpig.z * BN2;
    const uint kb = pc.K >> 5;

    threadgroup half Wsh[BM2 * SKH];
    threadgroup half Xsh[BN2 * SKH];
    threadgroup float outb[8u * 72u];

    simdgroup_float8x8 acc[16];
    for (uint j = 0u; j < 16u; ++j) acc[j] = simdgroup_float8x8(0.0f);

    // staging assignment: (row, block, half-block) per thread
    const uint rr = tid >> 2;             // 0..63
    const uint sbi = (tid >> 1) & 1u;     // block within BK=64 chunk
    const uint shf = tid & 1u;            // 16-elem half of the block

    for (uint b0 = 0u; b0 < kb; b0 += 2u) {
        {
            const uint wr = rowBase + rr;
            threadgroup half4* w4 =
                (threadgroup half4*)&Wsh[rr * SKH + sbi * 32u + shf * 16u];
            if (wr < pc.M && b0 + sbi < kb) {
                device const block_q8_0& blk = wb[(ulong)wr * kb + b0 + sbi];
                const half d = blk.d;
                device const packed_char4* qp =
                    (device const packed_char4*)&blk.qs[shf * 16u];
                for (uint i = 0u; i < 4u; ++i)
                    w4[i] = d * half4(char4(qp[i]));
            } else {
                for (uint i = 0u; i < 4u; ++i) w4[i] = half4(0.0h);
            }
        }
        for (uint idx = tid * 4u; idx < BN2 * 64u; idx += 256u * 4u) {
            const uint tt = idx >> 6;
            const uint kk = idx & 63u;
            const uint xt = tokBase + tt;
            const uint xk = (b0 << 5) + kk;
            ((threadgroup half4*)&Xsh[tt * SKH + kk])[0] = (xt < pc.N && xk < pc.K)
                ? half4(*(device const packed_float4*)(x + (ulong)xt * pc.K + xk))
                : half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kf = 0u; kf < 8u; ++kf) {
            simdgroup_half8x8 a;
            simdgroup_load(a, &Wsh[(sgid * 8u) * SKH + kf * 8u], SKH);
            for (uint nc = 0u; nc < 16u; ++nc) {
                simdgroup_half8x8 bfr;
                simdgroup_load(bfr, &Xsh[(nc * 8u) * SKH + kf * 8u], SKH, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[nc], a, bfr, acc[nc]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* buf = &outb[sgid * 72u];
    const uint fi = (tid & 31u) >> 2;
    const uint fj0 = (tid & 3u) * 2u;
    for (uint nc = 0u; nc < 16u; ++nc) {
        simdgroup_store(acc[nc], buf, 9u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint row = rowBase + sgid * 8u + fi;
        const uint tok0 = tokBase + nc * 8u;
        for (uint jj = 0u; jj < 2u; ++jj) {
            const uint tok = tok0 + fj0 + jj;
            if (row < pc.M && tok < pc.N)
                y[(ulong)tok * pc.M + row] = buf[fi * 9u + fj0 + jj];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
