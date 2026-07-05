// ggml quantization block layouts + CPU dequant, ported faithfully from
// llama.cpp ggml/src/ggml-quants.c (MIT). These must stay bit-exact with
// upstream so kernels validated here read real GGUF tensors correctly.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "quant_tables.h"

static inline float qk_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            x = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

static inline uint16_t qk_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t h   = sign | ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)h;
}

// ---- block layouts (must match ggml-common.h exactly) ----

#define QK8_0 32
struct block_q8_0 {
    uint16_t d;          // fp16 delta
    int8_t   qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 block size");

#define QK_K 256
struct block_q6_K {
    uint8_t  ql[QK_K / 2];      // lower 4 bits
    uint8_t  qh[QK_K / 4];      // upper 2 bits
    int8_t   scales[QK_K / 16]; // 8-bit sub-block scales
    uint16_t d;                 // fp16 super-block scale
};
static_assert(sizeof(block_q6_K) == 210, "q6_K block size");

// ---- CPU dequant (reference) ----

static inline void dequant_row_q8_0(const block_q8_0* x, float* y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = qk_f16_to_f32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) *y++ = d * x[i].qs[j];
    }
}

static inline void dequant_row_q6_K(const block_q6_K* x, float* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = qk_f16_to_f32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

struct block_iq4_xs {
    uint16_t d;                  // fp16 super-block scale
    uint16_t scales_h;           // high 2 bits of the 8 sub-scales
    uint8_t  scales_l[QK_K / 64];// low 4 bits, two per byte
    uint8_t  qs[QK_K / 2];       // codebook indices, two per byte
};
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs block size");

struct block_iq3_xxs {
    uint16_t d;                  // fp16 super-block scale
    uint8_t  qs[3 * QK_K / 8];   // 64 grid indices + 8x aux u32 (signs/scales)
};
static_assert(sizeof(block_iq3_xxs) == 98, "iq3_xxs block size");

static inline void dequant_row_iq4_xs(const block_iq4_xs* x, float* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = qk_f16_to_f32(x[i].d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                           (((x[i].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

static inline void dequant_row_iq3_xxs(const block_iq3_xxs* x, float* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    uint32_t aux32;
    for (int64_t i = 0; i < nb; i++) {
        const float d = qk_f16_to_f32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* scales_and_signs = qs + QK_K / 4;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}
