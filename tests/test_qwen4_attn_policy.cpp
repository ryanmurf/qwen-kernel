#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/qwen4_attn_policy.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>

int main() {
    assert(qwen4DecodeAttentionRequested(nullptr)==Qwen4DecodeAttention::Serial);
    assert(qwen4DecodeAttentionRequested("serial")==Qwen4DecodeAttention::Serial);
    assert(qwen4DecodeAttentionRequested("split")==Qwen4DecodeAttention::Split);
    assert(qwen4DecodeAttentionRequested("ordered")==Qwen4DecodeAttention::Ordered);
    assert(qwen4DecodeAttentionRequested("loads")==Qwen4DecodeAttention::Loads);
    for (const char* bad : {"", "0", "1", "Loads", "loads ", "vec4", "loads-f16"}) {
        bool threw=false;
        try { qwen4DecodeAttentionRequested(bad); } catch (const std::runtime_error&) { threw=true; }
        assert(threw);
    }
    for (uint32_t capacity : {1u,128u,2048u,8192u,16384u,32768u})
        assert(qwen4DecodeLoadsSupported(0x1002,0x1586,capacity));
    assert(!qwen4DecodeLoadsSupported(0x1002,0x1586,0));
    assert(!qwen4DecodeLoadsSupported(0x1002,0x1586,32769));
    assert(!qwen4DecodeLoadsSupported(0x1002,0x744c,32768));
    assert(!qwen4DecodeLoadsSupported(0x10de,0x1586,32768));
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
