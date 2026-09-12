#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/qwen4_attn_policy.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>

int main() {
    assert(!qwen4Vec4AttentionRequested(nullptr));
    assert(!qwen4Vec4AttentionRequested("baseline"));
    assert(qwen4Vec4AttentionRequested("vec4"));
    for (const char* bad : {"", "0", "1", "Vec4", "vec4-q8", " vec4", "ordered"}) {
        bool threw=false;
        try { qwen4Vec4AttentionRequested(bad); } catch (const std::runtime_error&) { threw=true; }
        assert(threw);
    }
    assert(qwen4AttentionQueryBlock(false,false)==16);
    assert(qwen4AttentionQueryBlock(true,false)==8);
    assert(qwen4AttentionQueryBlock(false,true)==16);
    assert(qwen4AttentionQueryBlock(true,true)==16);
    for (uint32_t tile : {1u,7u,8u,9u,15u,16u,17u,64u,128u,512u}) {
        for (bool vec4 : {false,true}) for (bool coop : {false,true}) {
            uint32_t qb=qwen4AttentionQueryBlock(vec4,coop), groups=(tile+qb-1)/qb;
            assert(groups*qb>=tile && (groups-1)*qb<tile);
            assert(16%qb==0); // existing host qbase alignment is compatible
        }
    }
    std::puts("batch attention opt-in parser, query blocks, coop precedence and dispatch bounds: PASS");
}
