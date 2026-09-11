// Shared block declarations and 32-element dequantization for the batched
// native Flash GEMM tiers. The includer defines QFMT (0 Q5_K, 1 Q6_K, 2 Q8_0,
// 3 Q5_1) and WSH_WRITE(index, value) to store into its LDS tile.
#if QFMT == 0
struct block_q5_K { float16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
layout(std430, binding = 0) readonly buffer BufW { block_q5_K wb[]; };
#elif QFMT == 1
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; float16_t d; };
layout(std430, binding = 0) readonly buffer BufW { block_q6_K wb[]; };
#elif QFMT == 2
struct block_q8_0 { float16_t d; int8_t qs[32]; };
layout(std430, binding = 0) readonly buffer BufW { block_q8_0 wb[]; };
#else
struct block_q5_1 { float16_t d, m; uint8_t qh[4]; uint8_t qs[16]; };
layout(std430, binding = 0) readonly buffer BufW { block_q5_1 wb[]; };
#endif

// Dequantize 32-wide chunk c (0..K/32) of weight row r into WSH_WRITE(dst+i, v).
void dequant32(uint r, uint c, uint dst) {
#if QFMT == 0
    uint b = r * (pc.K / 256u) + c / 8u, g = c % 8u;
    uint sc = g < 4u ? uint(wb[b].scales[g]) & 63u
        : (uint(wb[b].scales[g + 4u]) & 15u) | ((uint(wb[b].scales[g - 4u]) >> 6u) << 4u);
    uint mn = g < 4u ? uint(wb[b].scales[g + 4u]) & 63u
        : (uint(wb[b].scales[g + 4u]) >> 4u) | ((uint(wb[b].scales[g]) >> 6u) << 4u);
    float d = float(wb[b].d) * float(sc), m = float(wb[b].dmin) * float(mn);
    [[unroll]] for (uint i = 0u; i < 32u; ++i) {
        uint lo = (uint(wb[b].qs[(g / 2u) * 32u + i]) >> ((g % 2u) * 4u)) & 15u;
        uint hi = (uint(wb[b].qh[i]) >> g) & 1u;
        WSH_WRITE(dst + i, d * float(lo + 16u * hi) - m);
    }
#elif QFMT == 1
    uint b = r * (pc.K / 256u) + c / 8u;
    float d = float(wb[b].d);
    [[unroll]] for (uint half_ = 0u; half_ < 2u; ++half_) {
        uint gg = (c % 8u) * 2u + half_;
        uint h = gg >> 3u, rr = (gg & 7u) >> 1u, is = gg & 1u;
        float sc = float(int(wb[b].scales[h * 8u + rr * 2u + is]));
        uint qlBase = h * 64u + (rr & 1u) * 32u + is * 16u;
        uint qhBase = h * 32u + is * 16u;
        uint shift = rr * 2u;
        bool hi = rr >= 2u;
        [[unroll]] for (uint i = 0u; i < 16u; ++i) {
            uint qlv = uint(wb[b].ql[qlBase + i]);
            uint qhv = uint(wb[b].qh[qhBase + i]);
            uint lo = hi ? (qlv >> 4u) : (qlv & 15u);
            int q = int(lo | (((qhv >> shift) & 3u) << 4u)) - 32;
            WSH_WRITE(dst + half_ * 16u + i, d * sc * float(q));
        }
    }
#elif QFMT == 2
    uint b = r * (pc.K / 32u) + c;
    float d = float(wb[b].d);
    [[unroll]] for (uint i = 0u; i < 32u; ++i) WSH_WRITE(dst + i, d * float(int(wb[b].qs[i])));
#else
    uint b = r * (pc.K / 32u) + c;
    float d = float(wb[b].d), m = float(wb[b].m);
    [[unroll]] for (uint i = 0u; i < 32u; ++i) {
        uint lo = (uint(wb[b].qs[i % 16u]) >> ((i / 16u) * 4u)) & 15u;
        uint hi = (uint(wb[b].qh[i / 8u]) >> (i % 8u)) & 1u;
        WSH_WRITE(dst + i, d * float(lo + 16u * hi) + m);
    }
#endif
}
