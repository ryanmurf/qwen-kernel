#include <metal_stdlib>
using namespace metal;

// out[i] = a[i] + b[i], one 256-threadgroup. Port of shaders/vec_add.comp.

struct AddPC { uint n; };

kernel void vec_add(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float*       o [[buffer(2)]],
                    constant AddPC&     pc [[buffer(3)]],
                    uint3 tid3 [[thread_position_in_threadgroup]])
{
    for (uint i = tid3.x; i < pc.n; i += 256u) o[i] = a[i] + b[i];
}
