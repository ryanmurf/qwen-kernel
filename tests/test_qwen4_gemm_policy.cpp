#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/qwen4_gemm_policy.h"
#include <cassert>
#include <cstdio>

int main() {
    assert(!qwen4CompactGemmRequested(nullptr));
    assert(!qwen4CompactGemmRequested("baseline"));
    assert(qwen4CompactGemmRequested("compact"));
    for (const char* bad : {"", "0", "1", "ordered", "Compact", " compact"}) {
        bool threw=false;
        try { qwen4CompactGemmRequested(bad); } catch (const std::runtime_error&) { threw=true; }
        assert(threw);
    }
    for (uint32_t n : {0u,1u,63u,64u,65u,127u,128u,255u,256u,511u,512u,513u}) {
        const bool wide=n>=64 && n<=512, large=n>=256 && n<=512;
        assert(qwen4CompactGemmShape(GGML_Q5_K,6144,2560,n)==wide);
        assert(qwen4CompactGemmShape(GGML_Q5_K,12288,2560,n)==wide);
        assert(qwen4CompactGemmShape(GGML_Q5_K,2560,6144,n)==large);
        assert(qwen4CompactGemmShape(GGML_Q6_K,10240,2560,n)==wide);
        assert(qwen4CompactGemmShape(GGML_Q8_0,2560,640,n)==large);
        assert(qwen4CompactGemmShape(GGML_Q5_1,10240,320,n)==wide);
        assert(!qwen4CompactGemmShape(GGML_Q5_K,512,2560,n));
        assert(!qwen4CompactGemmShape(GGML_Q5_K,640,2560,n));
        assert(!qwen4CompactGemmShape(GGML_Q5_K,320,10240,n));
        assert(!qwen4CompactGemmShape(GGML_Q5_K,4,10240,n));
        assert(!qwen4CompactGemmShape(GGML_Q5_K,48,2560,n));
        assert(!qwen4CompactGemmShape(GGML_Q6_K,320,10240,n));
        assert(!qwen4CompactGemmShape(GGML_Q6_K,512,2560,n));
        assert(!qwen4CompactGemmShape(GGML_F32,6144,2560,n));
        assert(!qwen4CompactGemmShape(GGML_Q5_K,6143,2560,n));
        assert(!qwen4CompactGemmShape(GGML_Q5_1,10240,319,n));
    }
    std::puts("compact GEMM opt-in parser, measured shapes, boundaries and fallback: PASS");
}
