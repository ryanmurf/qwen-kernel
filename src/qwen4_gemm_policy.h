#pragma once
#include "gguf.h"
#include <cstring>
#include <stdexcept>

inline bool qwen4CompactGemmRequested(const char* value) {
    if (!value || !strcmp(value, "baseline")) return false;
    if (!strcmp(value, "compact")) return true;
    throw std::runtime_error("QK_FLASH_GEMM must be baseline or compact");
}

// Opt-in Halo policy: only the model shapes with measured wins in the
// scalar-F32 ABBA operator sweep. Leave skinny/small projections alone.
inline bool qwen4CompactGemmShape(uint32_t type, uint32_t m, uint32_t k, uint32_t rows) {
    if (rows < 64 || rows > 512) return false;
    switch (type) {
        case GGML_Q5_K:
            return (k == 2560 && (m == 6144 || m == 12288)) ||
                   (rows >= 256 && m == 2560 && k == 6144);
        case GGML_Q6_K: return m == 10240 && k == 2560;
        case GGML_Q8_0: return rows >= 256 && m == 2560 && k == 640;
        case GGML_Q5_1: return m == 10240 && k == 320;
        default: return false;
    }
}
