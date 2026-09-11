#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/quants.h"
#include "../src/gguf.h"

int main() {
    assert(ggmlRowBytes(GGML_Q5_1, 160) == 120);
    assert(ggmlRowBytes(GGML_Q5_K, 2560) == 1760);
    assert(ggmlRowBytes(GGML_Q5_1, 159) == 0);
    assert(ggmlRowBytes(GGML_Q5_K, 255) == 0);
    for (unsigned seed = 0; seed < 64; ++seed) {
        block_q5_1 small{};
        small.d = qk_f32_to_f16(0.5f);
        small.m = qk_f32_to_f16(-7.0f);
        float expectedSmall[32], actualSmall[32];
        for (unsigned i = 0; i < 32; ++i) {
            unsigned q = (i * 7 + seed) % 32;
            small.qs[i % 16] |= (q & 15) << (4 * (i / 16));
            small.qh[i / 8] |= (q >> 4) << (i % 8);
            expectedSmall[i] = q * 0.5f - 7.0f;
        }
        dequant_row_q5_1(&small, actualSmall, 32);
        for (unsigned i = 0; i < 32; ++i) assert(actualSmall[i] == expectedSmall[i]);

        block_q5_K large{};
        large.d = qk_f32_to_f16(0.5f);
        large.dmin = qk_f32_to_f16(0.25f);
        unsigned scales[8], mins[8];
        for (unsigned g = 0; g < 8; ++g) {
            scales[g] = (g * 9 + seed) % 64;
            mins[g] = (g * 13 + seed) % 64;
        }
        for (unsigned g = 0; g < 4; ++g) {
            large.scales[g] = scales[g] | ((scales[g + 4] >> 4) << 6);
            large.scales[g + 4] = mins[g] | ((mins[g + 4] >> 4) << 6);
            large.scales[g + 8] = (scales[g + 4] & 15) | ((mins[g + 4] & 15) << 4);
        }
        float expectedLarge[256], actualLarge[256];
        for (unsigned g = 0; g < 8; ++g) {
            for (unsigned i = 0; i < 32; ++i) {
                unsigned q = (g * 7 + i * 3 + seed) % 32;
                large.qs[(g / 2) * 32 + i] |= (q & 15) << (4 * (g % 2));
                large.qh[i] |= (q >> 4) << g;
                expectedLarge[g * 32 + i] = 0.5f * scales[g] * q - 0.25f * mins[g];
            }
        }
        dequant_row_q5_K(&large, actualLarge, 256);
        for (unsigned i = 0; i < 256; ++i) assert(actualLarge[i] == expectedLarge[i]);
    }
    std::puts("Q5 packing, scales, high bits, offsets and row sizes: PASS");
}
