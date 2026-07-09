// Shared declarations for the fused-MoE kernel chain (Metal ports of
// shaders/moe_*.comp). Pulled in via the loadMetalSource include inliner.
// Requires iq_tables.metal to be included first where IQ formats are used.

struct SelT {
    uint  ids[8];
    float w[8];
    float wShared;
    float pad[15];
};

struct MoePC {
    uint n_embd;
    uint n_ff;
    uint n_expert;
    uint n_used;
};

struct block_q8_0 {
    half d;
    char qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 block size");

struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char  scales[16];
    half  d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K block size");

struct block_iq4_xs {
    half   d;
    ushort scales_h;
    uchar  scales_l[4];
    uchar  qs[128];
};
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs block size");

struct block_iq3_xxs {
    half  d;
    uchar qs[96];
};
static_assert(sizeof(block_iq3_xxs) == 98, "iq3_xxs block size");

// dot of one full q8_0 block (32 elems) with v[0..8) float4s
static inline float q8_block_dot(device const block_q8_0& blk,
                                 device const float4* vp) {
    device const packed_char4* qp = (device const packed_char4*)blk.qs;
    float s = 0.0f;
    for (uint j = 0u; j < 8u; ++j)
        s += dot(float4(char4(qp[j])), vp[j]);
    return float(blk.d) * s;
}

// Inline expert selection, run redundantly by every consumer SIMDGROUP.
// The logits (n_expert+1 floats, the +1 being the shared-gate dot from
// moe_logits) are SLC-hot after the logits barrier, and the whole top-8
// costs ~2 µs of lane-parallel ALU — cheaper than a dedicated select
// dispatch + device barrier per layer. Same semantics as moe_select.comp:
// NaN/Inf floored, ties to the LOWEST expert index, weights = softmax over
// the selected logits. Returns this slot's expert id and weight (or the
// shared-expert sigmoid gate for slot == n_used) to every lane.
// ids[0..nUsed) and ws[0..nUsed) get the routed picks; wShared the
// shared-expert sigmoid gate. Every lane receives the full result.
static inline void moe_pick_all(device const float* logits, uint nExp,
                                uint nUsed, uint slid,
                                thread uint* ids, thread float* ws,
                                thread float& wShared) {
    wShared = 1.0f / (1.0f + exp(-logits[nExp]));
    const uint perLane = (nExp + 31u) / 32u;  // 8 for 256 experts
    float v[8];
    for (uint i = 0u; i < perLane && i < 8u; ++i) {
        const uint g = i * 32u + slid;
        float lv = g < nExp ? logits[g] : -3.4e38f;
        v[i] = (isnan(lv) || isinf(lv)) ? -3.4e38f : lv;
    }
    float m0 = 0.0f, sum = 0.0f;
    for (uint round = 0u; round < nUsed; ++round) {
        float lm = -3.4e38f;
        uint  li = 0u;
        for (uint i = 0u; i < perLane && i < 8u; ++i)
            if (v[i] > lm) { lm = v[i]; li = i; }
        const float gm = simd_max(lm);
        const uint  g  = li * 32u + slid;
        const uint  win = simd_min(lm == gm ? g : 0xFFFFu);
        if (slid == (win & 31u)) v[win >> 5u] = -3.4e38f;
        if (round == 0u) m0 = gm;
        const float e = exp(gm - m0);
        sum += e;
        ids[round] = min(win, nExp - 1u);
        ws[round] = e;
    }
    const float inv = 1.0f / sum;
    for (uint round = 0u; round < nUsed; ++round) ws[round] *= inv;
}
