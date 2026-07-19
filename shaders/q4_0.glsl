// Canonical native ggml Q4_0 block decode shared by decode GEMV and prefill
// GEMM. Low nibbles are elements 0..15; high nibbles are 16..31.
struct block_q4_0 {
    float16_t d;
    uint8_t qs[16];
};

void q4_0_dequant_pair(block_q4_0 block, uint j, out float lo, out float hi) {
    uint packed = uint(block.qs[j]);
    float d = float(block.d);
    lo = d * float(int(packed & 15u) - 8);
    hi = d * float(int(packed >> 4u) - 8);
}
