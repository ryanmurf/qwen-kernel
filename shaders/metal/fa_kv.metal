// Compile-time KV-cache storage width shared by every attention kernel.
// Current hosts inject QK_KV_BYTES=4 (exact f32 default), 2 (f16), or 1
// (row-Q8) before compilation. An older host supplies no define and therefore
// gets exact f32 without needing to understand a new function constant.
#ifndef QK_KV_BYTES
#define QK_KV_BYTES 4u
#endif
constant uint KV_BYTES = QK_KV_BYTES;

// Q8 cache tier (KV_BYTES=1): one 272-byte record per logical 256-float row.
// A shared f32 scale sits at byte 0 and signed values start at byte 16.  The
// padding keeps every row/vector load 16-byte aligned.
constant ulong KV_Q8_ROW = 272u;
constant ulong KV_Q8_DATA = 16u;

static inline float kv_load(device const uchar* p, ulong i) {
    if (KV_BYTES == 1u) {
        const ulong rb = (i >> 8u) * KV_Q8_ROW;
        const float d = *((device const float*)(p + rb));
        const char q = *((device const char*)(p + rb + KV_Q8_DATA + (i & 255u)));
        return d * float(q);
    }
    if (KV_BYTES == 2u) return float(((device const half*)p)[i]);
    return ((device const float*)p)[i];
}

static inline void kv_store(device uchar* p, ulong i, float v) {
    if (KV_BYTES == 1u) {
        const ulong rb = (i >> 8u) * KV_Q8_ROW;
        const float d = *((device const float*)(p + rb));
        const int q = d > 0.0f ? clamp(int(rint(v / d)), -127, 127) : 0;
        *((device char*)(p + rb + KV_Q8_DATA + (i & 255u))) = char(q);
    } else if (KV_BYTES == 2u) ((device half*)p)[i] = half(v);
    else                ((device float*)p)[i] = v;
}
