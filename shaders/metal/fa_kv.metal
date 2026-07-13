// Compile-time KV-cache storage width shared by every attention kernel.
// The host specializes function constant 0 to 4 (exact f32 default) or 2
// (opt-in f16 storage with f32 attention accumulation).
constant uint KV_BYTES [[function_constant(0)]];

static inline float kv_load(device const uchar* p, ulong i) {
    if (KV_BYTES == 2u) return float(((device const half*)p)[i]);
    return ((device const float*)p)[i];
}

static inline void kv_store(device uchar* p, ulong i, float v) {
    if (KV_BYTES == 2u) ((device half*)p)[i] = half(v);
    else                ((device float*)p)[i] = v;
}
