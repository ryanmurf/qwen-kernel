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
