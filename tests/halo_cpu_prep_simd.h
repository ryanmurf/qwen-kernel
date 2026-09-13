#pragma once
// Experimental CPU preparation kernels, not wired into the serving engine.
// Compile with -ffp-contract=off; preserve the reference multiply/add order.
#include "../src/quants.h"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define QK_PREP_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl"), noinline))
inline bool haloPrepAvx512Available() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl");
}

QK_PREP_AVX512 inline void haloPrepQ51(const block_q5_1* blocks, float* out, int64_t k) {
    assert(k >= 0 && k % 32 == 0);
    const __m512i shifts = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    for (int64_t b = 0; b < k/32; ++b) {
        const auto& w = blocks[b];
        uint32_t high;
        memcpy(&high,w.qh,sizeof(high));
        const __m512 d = _mm512_set1_ps(qk_f16_to_f32(w.d));
        const __m512 m = _mm512_set1_ps(qk_f16_to_f32(w.m));
        const __m512i bytes = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)w.qs));
        for (int half = 0; half < 2; ++half) {
            const __m512i lo = _mm512_and_si512(half ? _mm512_srli_epi32(bytes,4) : bytes,
                                               _mm512_set1_epi32(15));
            const __m512i hi = _mm512_and_si512(_mm512_srlv_epi32(
                _mm512_set1_epi32(int(high >> (16*half))),shifts),_mm512_set1_epi32(1));
            const __m512 q = _mm512_cvtepi32_ps(_mm512_or_si512(lo,_mm512_slli_epi32(hi,4)));
            _mm512_storeu_ps(out+b*32+half*16,_mm512_add_ps(_mm512_mul_ps(d,q),m));
        }
    }
}

QK_PREP_AVX512 inline void haloPrepQ5K(const block_q5_K* blocks, float* out, int64_t k) {
    assert(k >= 0 && k % 256 == 0);
    for (int64_t b = 0; b < k/256; ++b) {
        const auto& w = blocks[b];
        const float d = qk_f16_to_f32(w.d), dm = qk_f16_to_f32(w.dmin);
        for (uint32_t g = 0; g < 8; ++g) {
            const uint32_t sc = g < 4 ? w.scales[g]&63
                : (w.scales[g+4]&15) | ((w.scales[g-4]>>6)<<4);
            const uint32_t mn = g < 4 ? w.scales[g+4]&63
                : (w.scales[g+4]>>4) | ((w.scales[g]>>6)<<4);
            const __m512 scale = _mm512_set1_ps(d*sc), offset = _mm512_set1_ps(dm*mn);
            for (uint32_t half = 0; half < 2; ++half) {
                const __m512i bytes = _mm512_cvtepu8_epi32(
                    _mm_loadu_si128((const __m128i*)(w.qs+(g/2)*32+half*16)));
                const __m512i high = _mm512_cvtepu8_epi32(
                    _mm_loadu_si128((const __m128i*)(w.qh+half*16)));
                const __m512i lo = _mm512_and_si512(g%2 ? _mm512_srli_epi32(bytes,4) : bytes,
                                                   _mm512_set1_epi32(15));
                const __m512i hi = _mm512_and_si512(_mm512_srlv_epi32(high,_mm512_set1_epi32(g)),
                                                   _mm512_set1_epi32(1));
                const __m512 q = _mm512_cvtepi32_ps(_mm512_or_si512(lo,_mm512_slli_epi32(hi,4)));
                _mm512_storeu_ps(out+b*256+g*32+half*16,
                                _mm512_sub_ps(_mm512_mul_ps(scale,q),offset));
            }
        }
    }
}
#undef QK_PREP_AVX512
#else
inline bool haloPrepAvx512Available() { return false; }
inline void haloPrepQ51(const block_q5_1* b,float* o,int64_t k) { dequant_row_q5_1(b,o,k); }
inline void haloPrepQ5K(const block_q5_K* b,float* o,int64_t k) { dequant_row_q5_K(b,o,k); }
#endif
