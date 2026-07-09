#include <metal_stdlib>
using namespace metal;

// Depthwise causal conv (kernel 4) over the mixed qkv channels for one
// token, then SiLU; shifts the per-channel conv state (last 3 raw inputs).
// Port of shaders/dn_conv.comp. 256 threads per tg, grid.x covers channels.

struct ConvPC { uint channels; };

kernel void dn_conv(device float*       st  [[buffer(0)]],
                    device const float* qkv [[buffer(1)]],
                    device const float* ker [[buffer(2)]],
                    device float*       o   [[buffer(3)]],
                    constant ConvPC&    pc  [[buffer(4)]],
                    uint3 gid3 [[thread_position_in_grid]])
{
    const uint c = gid3.x;
    if (c >= pc.channels) return;

    const float s0 = st[c * 3u], s1 = st[c * 3u + 1u], s2 = st[c * 3u + 2u];
    const float xin = qkv[c];
    const float v = s0 * ker[c * 4u] + s1 * ker[c * 4u + 1u] +
                    s2 * ker[c * 4u + 2u] + xin * ker[c * 4u + 3u];
    o[c] = v / (1.0f + exp(-v));   // silu
    st[c * 3u]      = s1;
    st[c * 3u + 1u] = s2;
    st[c * 3u + 2u] = xin;
}
