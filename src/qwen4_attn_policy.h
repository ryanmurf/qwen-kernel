#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>

enum class Qwen4DecodeAttention { Serial, Split, Ordered, Loads };

inline Qwen4DecodeAttention qwen4DecodeAttentionRequested(const char* value) {
    if (!value || !strcmp(value, "serial")) return Qwen4DecodeAttention::Serial;
    if (!strcmp(value, "split")) return Qwen4DecodeAttention::Split;
    if (!strcmp(value, "ordered")) return Qwen4DecodeAttention::Ordered;
    if (!strcmp(value, "loads")) return Qwen4DecodeAttention::Loads;
    throw std::runtime_error("QK_ATTN_DECODE must be serial, split, ordered or loads");
}

inline bool qwen4DecodeLoadsSupported(uint32_t vendor, uint32_t device, uint32_t capacity) {
    return vendor == 0x1002 && device == 0x1586 && capacity >= 1 && capacity <= 32768;
}

inline bool qwen4Vec4AttentionRequested(const char* value) {
    if (!value || !strcmp(value, "baseline")) return false;
    if (!strcmp(value, "vec4")) return true;
    throw std::runtime_error("QK_FLASH_ATTN_BATCH must be baseline or vec4");
}

// Explicit cooperative-matrix selection retains its existing shader/QB.
inline uint32_t qwen4AttentionQueryBlock(bool vec4, bool cooperative) {
    return vec4 && !cooperative ? 8u : 16u;
}
