#include "../src/qwen4_head_policy.h"
#include <algorithm>
#include <cassert>

int main() {
    assert(qwen4HeadFirstTile(0, true) == 0);
    for (uint32_t rows = 1; rows <= 65536; ++rows) {
        assert(qwen4HeadFirstTile(rows, false) == 0);
        uint32_t lastStart = 0;
        for (uint32_t start = 0; start < rows; start += qwen4HeadTile)
            lastStart = start;
        const auto start = qwen4HeadFirstTile(rows, true);
        assert(start == lastStart);
        assert(rows - start >= 1 && rows - start <= qwen4HeadTile);
        assert((rows - 1) % qwen4HeadTile == rows - start - 1);
    }
}
