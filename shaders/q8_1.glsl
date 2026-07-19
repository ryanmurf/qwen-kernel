// ggml Q8_1 activation block used by llama.cpp's Vulkan MMVQ path.
struct block_q8_1 {
    float16_t d;
    float16_t s;
    int8_t qs[32];
};

#ifdef Q4_Q8_DOT
float q4_0_q8_1_fragment(block_q4_0 a, block_q8_1 b, uint fragment) {
    uint offset = fragment*4u;
    int integerDot = 0;
    for (uint j = 0u; j < 4u; ++j) {
        uint packed = uint(a.qs[offset + j]);
        integerDot += int(packed & 15u)*int(b.qs[offset + j]);
        integerDot += int(packed >> 4u)*int(b.qs[offset + j + 16u]);
    }
    // llama.cpp retains the separately-rounded fp16 sum*d member and applies
    // one quarter of the Q4 zero-point correction per eight-value fragment.
    return float(a.d)*(float(integerDot)*float(b.d) - 2.0*float(b.s));
}
#endif
