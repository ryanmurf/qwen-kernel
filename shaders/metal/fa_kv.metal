// Compile-time KV-cache storage width shared by every attention kernel.
// The host specializes function constant 0 to 4 (exact f32 default) or 2
// (opt-in f16 storage with f32 attention accumulation).
constant uint KV_BYTES [[function_constant(0)]];

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
