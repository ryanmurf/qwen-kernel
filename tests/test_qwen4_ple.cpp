#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/qwen4_ple.h"
#include <cmath>

int main() {
    Qwen4PleConfig p;
    p.ngram = 3; p.headsPerNgram = 2; p.width = 32; p.eos = 100;
    p.multipliers = {3, 7, UINT64_MAX - 12};
    p.offsets = {0, 13, 26, 39}; p.vocabs = {13, 13, 13, 13};
    p.validate(52);
    for (const std::vector<uint32_t>& history : std::vector<std::vector<uint32_t>>{
            {}, {5}, {6, 5}, {6, 100}, {100, 5}, {6, 100, 5}, {7, 8, 9}}) {
        auto rows = p.rows(11, history);
        // Independently construct a complete padded context and EOS cut.
        uint64_t ctx[3] = {11, 100, 100};
        for (size_t i = 0; i < 2 && i < history.size(); ++i) ctx[i+1] = history[history.size()-1-i];
        if (ctx[1] == 100) ctx[2] = 100;
        uint64_t h2 = (ctx[0]*3) ^ (ctx[1]*7);
        uint64_t h3 = h2 ^ (ctx[2]*(UINT64_MAX-12));
        assert(rows == std::vector<uint32_t>({uint32_t(h2%13), uint32_t(h2%13+13),
                                              uint32_t(h3%13+26), uint32_t(h3%13+39)}));
    }
    std::vector<block_q5_1> blocks(52);
    for (size_t r = 0; r < blocks.size(); ++r) {
        auto& b = blocks[r]; b.d = qk_f32_to_f16(0.5f); b.m = qk_f32_to_f16(float(r));
        for (unsigned j = 0; j < 32; ++j) {
            unsigned q = (r + j) % 32;
            b.qs[j%16] |= (q & 15) << (4*(j/16));
            b.qh[j/8] |= (q >> 4) << (j%8);
        }
    }
    GgufTensor table;
    table.type = GGML_Q5_1; table.nDims = 2; table.ne[0] = 32; table.ne[1] = 52;
    table.nbytes = blocks.size() * sizeof(block_q5_1); table.data = (const uint8_t*)blocks.data();
    for (size_t cacheSize : {4, 16, 4096}) {
        Qwen4PleLookup lookup(table, p, cacheSize);
        for (unsigned repeat = 0; repeat < 2; ++repeat) for (uint32_t token = 0; token < 20; ++token) {
            auto rows = p.rows(token, {7,8});
            float got[128]; lookup.gather(rows, got);
            for (size_t head = 0; head < 4; ++head) for (size_t j = 0; j < 32; ++j)
                assert(got[head*32+j] == rows[head] + float((rows[head]+j)%32)*0.5f);
        }
        if (cacheSize == 4096) assert(lookup.hits > 0 && lookup.misses <= 52);
        bool rejected = false;
        float got[32];
        try { lookup.gather({52}, got); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    std::puts("PLE hashes, EOS resets, uint64 wraparound, sparse gather and bounded cache: PASS");
}
