#include <metal_stdlib>
using namespace metal;

// Two-pass argmax over the vocab logits; ties resolve to the LOWER index
// (greedy sampling convention — must match llama.cpp for token parity).
// Ports of shaders/argmax1.comp / argmax2.comp. argmax2 additionally
// records the winner into rb[pos] so a whole generation stays GPU-resident.

struct Am1PC { uint n; uint span; };
struct Am2PC { uint m; uint pos; };

kernel void argmax1(device const float* logits [[buffer(0)]],
                    device float*       vals   [[buffer(1)]],
                    device uint*        idxs   [[buffer(2)]],
                    constant Am1PC&     pc     [[buffer(3)]],
                    uint3 tid3  [[thread_position_in_threadgroup]],
                    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint w = tgpig.x;
    const uint t = tid3.x;
    const uint rq = tgpig.z;
    const uint lo = rq * pc.n;
    const uint start = w * pc.span;
    const uint end = min(start + pc.span, pc.n);

    threadgroup float rv[256];
    threadgroup uint  ri[256];

    float bv = -3.4e38f;
    uint bi = 0u;
    for (uint i = start + t; i < end; i += 256u)
        if (logits[lo + i] > bv) { bv = logits[lo + i]; bi = i; }
    rv[t] = bv;
    ri[t] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s && (rv[t + s] > rv[t] || (rv[t + s] == rv[t] && ri[t + s] < ri[t]))) {
            rv[t] = rv[t + s];
            ri[t] = ri[t + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) { vals[rq * 64u + w] = rv[0]; idxs[rq * 64u + w] = ri[0]; }
}

kernel void argmax2(device const float* vals [[buffer(0)]],
                    device const uint*  idxs [[buffer(1)]],
                    device uint*        tok  [[buffer(2)]],
                    device uint*        rb   [[buffer(3)]],
                    constant Am2PC&     pc   [[buffer(4)]],
                    uint3 tid3  [[thread_position_in_threadgroup]],
                    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint t = tid3.x;
    const uint rq = tgpig.z;
    threadgroup float rv[256];
    threadgroup uint  ri[256];
    rv[t] = t < pc.m ? vals[rq * 64u + t] : -3.4e38f;
    ri[t] = t < pc.m ? idxs[rq * 64u + t] : 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s && (rv[t + s] > rv[t] || (rv[t + s] == rv[t] && ri[t + s] < ri[t]))) {
            rv[t] = rv[t + s];
            ri[t] = ri[t + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) {
        tok[rq] = ri[0];
        rb[pc.pos] = ri[0];
    }
}
