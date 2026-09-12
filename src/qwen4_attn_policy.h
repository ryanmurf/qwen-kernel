#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>

inline bool qwen4Vec4AttentionRequested(const char* value) {
    if (!value || !strcmp(value, "baseline")) return false;
    if (!strcmp(value, "vec4")) return true;
    throw std::runtime_error("QK_FLASH_ATTN_BATCH must be baseline or vec4");
}

// Explicit cooperative-matrix selection retains its existing shader/QB.
inline uint32_t qwen4AttentionQueryBlock(bool vec4, bool cooperative) {
    return vec4 && !cooperative ? 8u : 16u;
}
