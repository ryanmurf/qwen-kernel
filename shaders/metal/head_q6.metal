#include <metal_stdlib>
using namespace metal;

// Weight-stationary Q6_K output head for small verifier batches.  This lives
// separately so the exact shipping GEMV remains an independent fallback.

struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char  scales[16];
    half  d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K block size");

struct HeadPC { uint M; uint K; uint N; uint span; };

static inline void q6_stage32(device const block_q6_K* mat, uint rowBlocks,
                              uint row, uint g32, threadgroup float* dst) {
    device const block_q6_K& blk = mat[(ulong)row * rowBlocks + (g32 >> 3u)];
    const float d = float(blk.d);
    for (uint half16 = 0u; half16 < 2u; ++half16) {
        const uint gg = (g32 & 7u) * 2u + half16;
        const uint hh = gg >> 3u;
        const uint r  = (gg & 7u) >> 1u;
        const uint is = gg & 1u;
        const float sc = float(int(blk.scales[hh * 8u + r * 2u + is]));
        const uint qlBase = hh * 64u + (r & 1u) * 32u + is * 16u;
        const uint qhBase = hh * 32u + is * 16u;
        const uint shift = r * 2u;
        const bool hi = r >= 2u;
        device const packed_uchar4* qlp =
            (device const packed_uchar4*)&blk.ql[qlBase];
        device const packed_uchar4* qhp =
            (device const packed_uchar4*)&blk.qh[qhBase];
        threadgroup float4* o4 = (threadgroup float4*)(dst + half16 * 16u);
        for (uint i = 0u; i < 4u; ++i) {
            const uchar4 qlv = uchar4(qlp[i]);
            const uchar4 qhv = uchar4(qhp[i]);
            const uint4 lo = hi ? (uint4(qlv) >> 4u) : (uint4(qlv) & 0xFu);
            const int4 q = int4(lo | (((uint4(qhv) >> shift) & 3u) << 4u)) - 32;
            o4[i] = (d * sc) * float4(q);
        }
    }
}

constant uint HBM = 32u;
constant uint HBN = 8u;
constant uint HBK = 64u;
constant uint HSK = 68u;

kernel void head_q6_gemm_b8_f32(device const block_q6_K* w [[buffer(0)]],
                                 device const float* x      [[buffer(1)]],
                                 device float* y            [[buffer(2)]],
                                 constant HeadPC& pc         [[buffer(3)]],
                                 uint3 tid3 [[thread_position_in_threadgroup]],
                                 uint3 tgpig [[threadgroup_position_in_grid]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint slid [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;
    const uint row0 = tgpig.x * HBM;
    const uint tok0 = tgpig.z * HBN;
    const uint rowBlocks = pc.K >> 8u;
    threadgroup float Wsh[HBM * HSK];
    threadgroup float Xsh[HBN * HSK];
    threadgroup float outb[4u * 72u];
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0u; k0 < pc.K; k0 += HBK) {
        if (tid < 64u) {
            const uint rr = tid >> 1u;
            const uint g = tid & 1u;
            const uint row = row0 + rr;
            threadgroup float* dst = Wsh + rr * HSK + g * 32u;
            if (row < pc.M)
                q6_stage32(w, rowBlocks, row, (k0 >> 5u) + g, dst);
            else
                for (uint i = 0u; i < 32u; ++i) dst[i] = 0.0f;
        }
        for (uint idx = tid; idx < HBN * HBK; idx += 128u) {
            const uint tq = idx >> 6u, kk = idx & 63u;
            const uint tok = tok0 + tq;
            Xsh[tq * HSK + kk] = tok < pc.N
                ? x[(ulong)tok * pc.K + k0 + kk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kf = 0u; kf < HBK / 8u; ++kf) {
            simdgroup_float8x8 af, bf;
            simdgroup_load(af, Wsh + (sgid * 8u) * HSK + kf * 8u, HSK);
            simdgroup_load(bf, Xsh + kf * 8u, HSK, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, af, bf, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* buf = outb + sgid * 72u;
    simdgroup_store(acc, buf, 9u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint fi = slid >> 2u;
    const uint fj = (slid & 3u) * 2u;
    for (uint jj = 0u; jj < 2u; ++jj) {
        const uint row = row0 + sgid * 8u + fi;
        const uint tok = tok0 + fj + jj;
        if (row < pc.M && tok < pc.N)
            y[(ulong)tok * pc.M + row] = buf[fi * 9u + fj + jj];
    }
}

struct HeadTopPC { uint M; uint K; uint N; uint tiles; uint materialize; };

// Same arithmetic as head_q6_gemm_b8_f32, but reduce each 32-row tile to one
// stable candidate per token.  Optionally retain the final token's complete
// logit row for sampling/top-k without storing every verifier row.
kernel void head_q6_gemm_b8_top1_f32(device const block_q6_K* w [[buffer(0)]],
                                      device const float* x      [[buffer(1)]],
                                      device float* vals         [[buffer(2)]],
                                      device uint* idxs          [[buffer(3)]],
                                      device float* lastLogits   [[buffer(4)]],
                                      constant HeadTopPC& pc      [[buffer(5)]],
                                      uint3 tid3 [[thread_position_in_threadgroup]],
                                      uint3 tgpig [[threadgroup_position_in_grid]],
                                      uint sgid [[simdgroup_index_in_threadgroup]],
                                      uint slid [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;
    const uint row0 = tgpig.x * HBM;
    const uint tok0 = tgpig.z * HBN;
    const uint rowBlocks = pc.K >> 8u;
    threadgroup float Wsh[HBM * HSK];
    threadgroup float Xsh[HBN * HSK];
    threadgroup float outb[4u * 72u];
    threadgroup float sgVals[4u * HBN];
    threadgroup uint sgIdxs[4u * HBN];
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0u; k0 < pc.K; k0 += HBK) {
        if (tid < 64u) {
            const uint rr = tid >> 1u, g = tid & 1u, row = row0 + rr;
            threadgroup float* dst = Wsh + rr * HSK + g * 32u;
            if (row < pc.M) q6_stage32(w, rowBlocks, row, (k0 >> 5u) + g, dst);
            else for (uint i = 0u; i < 32u; ++i) dst[i] = 0.0f;
        }
        for (uint p = tid; p < HBN * HBK; p += 128u) {
            const uint tq = p >> 6u, kk = p & 63u, tok = tok0 + tq;
            Xsh[tq * HSK + kk] = tok < pc.N
                ? x[(ulong)tok * pc.K + k0 + kk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kf = 0u; kf < HBK / 8u; ++kf) {
            simdgroup_float8x8 af, bf;
            simdgroup_load(af, Wsh + (sgid * 8u) * HSK + kf * 8u, HSK);
            simdgroup_load(bf, Xsh + kf * 8u, HSK, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, af, bf, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* buf = outb + sgid * 72u;
    simdgroup_store(acc, buf, 9u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint fi = slid >> 2u, fj = (slid & 3u) * 2u;
    if (pc.materialize) {
        for (uint jj = 0u; jj < 2u; ++jj) {
            const uint row = row0 + sgid * 8u + fi, tok = tok0 + fj + jj;
            if (row < pc.M && tok + 1u == pc.N)
                lastLogits[row] = buf[fi * 9u + fj + jj];
        }
    }
    if (slid < HBN) {
        float best = -3.4e38f;
        uint besti = 0xFFFFFFFFu;
        for (uint r = 0u; r < 8u; ++r) {
            const uint id = row0 + sgid * 8u + r;
            if (id >= pc.M) continue;
            const float v = buf[r * 9u + slid];
            if (v > best || (v == best && id < besti)) { best = v; besti = id; }
        }
        sgVals[sgid * HBN + slid] = best;
        sgIdxs[sgid * HBN + slid] = besti;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HBN) {
        const uint tok = tok0 + tid;
        if (tok < pc.N) {
            float best = -3.4e38f;
            uint besti = 0xFFFFFFFFu;
            for (uint s = 0u; s < 4u; ++s) {
                const float v = sgVals[s * HBN + tid];
                const uint id = sgIdxs[s * HBN + tid];
                if (v > best || (v == best && id < besti)) { best = v; besti = id; }
            }
            const ulong o = (ulong)tok * pc.tiles + tgpig.x;
            vals[o] = best;
            idxs[o] = besti;
        }
    }
}

// Wider row tile: identical per-logit arithmetic, eight simdgroups, and half
// as many candidate pairs.  Kept as an isolated geometry for the M4 occupancy
// crossover; the 22-KiB TG footprint permits only one resident TG/core.
kernel void head_q6_gemm_b8_top1_f32_m64(device const block_q6_K* w [[buffer(0)]],
                                          device const float* x      [[buffer(1)]],
                                          device float* vals         [[buffer(2)]],
                                          device uint* idxs          [[buffer(3)]],
                                          device float* lastLogits   [[buffer(4)]],
                                          constant HeadTopPC& pc      [[buffer(5)]],
                                          uint3 tid3 [[thread_position_in_threadgroup]],
                                          uint3 tgpig [[threadgroup_position_in_grid]],
                                          uint sgid [[simdgroup_index_in_threadgroup]],
                                          uint slid [[thread_index_in_simdgroup]])
{
    const uint tid = tid3.x;
    const uint row0 = tgpig.x * 64u;
    const uint tok0 = tgpig.z * HBN;
    const uint rowBlocks = pc.K >> 8u;
    threadgroup float Wsh[64u * HSK];
    threadgroup float Xsh[HBN * HSK];
    threadgroup float outb[8u * 72u];
    threadgroup float sgVals[8u * HBN];
    threadgroup uint sgIdxs[8u * HBN];
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0u; k0 < pc.K; k0 += HBK) {
        if (tid < 128u) {
            const uint rr = tid >> 1u, g = tid & 1u, row = row0 + rr;
            threadgroup float* dst = Wsh + rr * HSK + g * 32u;
            if (row < pc.M) q6_stage32(w, rowBlocks, row, (k0 >> 5u) + g, dst);
            else for (uint i = 0u; i < 32u; ++i) dst[i] = 0.0f;
        }
        for (uint p = tid; p < HBN * HBK; p += 256u) {
            const uint tq = p >> 6u, kk = p & 63u, tok = tok0 + tq;
            Xsh[tq * HSK + kk] = tok < pc.N
                ? x[(ulong)tok * pc.K + k0 + kk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kf = 0u; kf < HBK / 8u; ++kf) {
            simdgroup_float8x8 af, bf;
            simdgroup_load(af, Wsh + (sgid * 8u) * HSK + kf * 8u, HSK);
            simdgroup_load(bf, Xsh + kf * 8u, HSK, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, af, bf, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* buf = outb + sgid * 72u;
    simdgroup_store(acc, buf, 9u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint fi = slid >> 2u, fj = (slid & 3u) * 2u;
    if (pc.materialize) {
        for (uint jj = 0u; jj < 2u; ++jj) {
            const uint row = row0 + sgid * 8u + fi, tok = tok0 + fj + jj;
            if (row < pc.M && tok + 1u == pc.N)
                lastLogits[row] = buf[fi * 9u + fj + jj];
        }
    }
    if (slid < HBN) {
        float best = -3.4e38f;
        uint besti = 0xFFFFFFFFu;
        for (uint r = 0u; r < 8u; ++r) {
            const uint id = row0 + sgid * 8u + r;
            if (id >= pc.M) continue;
            const float v = buf[r * 9u + slid];
            if (v > best || (v == best && id < besti)) { best = v; besti = id; }
        }
        sgVals[sgid * HBN + slid] = best;
        sgIdxs[sgid * HBN + slid] = besti;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HBN) {
        const uint tok = tok0 + tid;
        if (tok < pc.N) {
            float best = -3.4e38f;
            uint besti = 0xFFFFFFFFu;
            for (uint s = 0u; s < 8u; ++s) {
                const float v = sgVals[s * HBN + tid];
                const uint id = sgIdxs[s * HBN + tid];
                if (v > best || (v == best && id < besti)) { best = v; besti = id; }
            }
            const ulong o = (ulong)tok * pc.tiles + tgpig.x;
            vals[o] = best;
            idxs[o] = besti;
        }
    }
}

struct HeadReducePC { uint tiles; uint N; };
kernel void head_top1_reduce_batch(device const float* vals [[buffer(0)]],
                                   device const uint* idxs  [[buffer(1)]],
                                   device uint* tok         [[buffer(2)]],
                                   constant HeadReducePC& pc [[buffer(3)]],
                                   uint3 tid3 [[thread_position_in_threadgroup]],
                                   uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint tid = tid3.x, n = tgpig.z;
    if (n >= pc.N) return;
    float best = -3.4e38f;
    uint besti = 0xFFFFFFFFu;
    for (uint i = tid; i < pc.tiles; i += 256u) {
        const ulong o = (ulong)n * pc.tiles + i;
        const float v = vals[o];
        const uint id = idxs[o];
        if (v > best || (v == best && id < besti)) { best = v; besti = id; }
    }
    threadgroup float rv[256];
    threadgroup uint ri[256];
    rv[tid] = best;
    ri[tid] = besti;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s && (rv[tid + s] > rv[tid] ||
                        (rv[tid + s] == rv[tid] && ri[tid + s] < ri[tid]))) {
            rv[tid] = rv[tid + s];
            ri[tid] = ri[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) tok[n] = ri[0];
}
