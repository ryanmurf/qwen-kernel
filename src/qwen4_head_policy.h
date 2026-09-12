#pragma once
#include <cstdint>

// Keep the final tile identical to the all-position path: same input offset,
// row count, shader choice and reduction order. Never replace it with GEMV.
constexpr uint32_t qwen4HeadTile = 64;
constexpr uint32_t qwen4HeadFirstTile(uint32_t rows, bool lastOnly) {
    return lastOnly && rows ? ((rows - 1) / qwen4HeadTile) * qwen4HeadTile : 0;
}
