// Expert-major batched routed down-projection. One workgroup owns one
// (expert, output row), dequantizes its n_ff weights once (thread tid holds
// elements j*64+tid) and writes each pair's gated dot product to the routed
// partial buffer y[(token*n_used + slot)*n_embd + o]. Host requires
// n_ff == 64*KPT. DOWN_Q51 selects Q5_1 blocks instead of Q8_0.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 64) in;
const uint KPT = 10u;

#if DOWN_Q51
struct block_q5_1 { float16_t d, m; uint8_t qh[4]; uint8_t qs[16]; };
layout(std430, binding = 0) readonly buffer W { block_q5_1 dw[]; };
#else
struct block_q8_0 { float16_t d; int8_t qs[32]; };
layout(std430, binding = 0) readonly buffer W { block_q8_0 dw[]; };
#endif
layout(std430, binding = 1) readonly buffer Hs { float hidden[]; };
struct SelT { uint ids[16]; float w[16]; float wShared; float pad[7]; };
layout(std430, binding = 2) readonly buffer S { SelT sel[]; };
layout(std430, binding = 3) readonly buffer Offsets { uint offsets[]; };
layout(std430, binding = 4) readonly buffer Pairs { uint pairs[]; };
layout(std430, binding = 5) writeonly buffer Y { float y[]; };
layout(push_constant) uniform PC { uint n_embd; uint n_ff; uint n_expert; uint n_used; } pc;

shared float red[2];

void main() {
    uint o = gl_WorkGroupID.x;
    uint eid = gl_WorkGroupID.y;
    uint tid = gl_LocalInvocationID.x;
    if (o >= pc.n_embd || eid >= pc.n_expert) return;
    uint begin = offsets[eid], end = offsets[eid + 1u];
    if (begin == end) return;

    uint blocks = pc.n_ff / 32u;
    uint base = (eid * pc.n_embd + o) * blocks;
    float qv[KPT];
    [[unroll]] for (uint j = 0u; j < KPT; ++j) {
        uint e = j * 64u + tid;
        uint b = base + e / 32u, i = e % 32u;
#if DOWN_Q51
        uint lo = (uint(dw[b].qs[i % 16u]) >> ((i / 16u) * 4u)) & 15u;
        uint hi = (uint(dw[b].qh[i / 8u]) >> (i % 8u)) & 1u;
        qv[j] = float(dw[b].d) * float(lo + 16u * hi) + float(dw[b].m);
#else
        qv[j] = float(dw[b].d) * float(int(dw[b].qs[i]));
#endif
    }

    for (uint pi = begin; pi < end; ++pi) {
        uint pair = pairs[pi];
        uint rq = pair >> 4u;
        uint s = pair & 15u;
        uint ho = (rq * (pc.n_used + 1u) + s) * pc.n_ff + tid;
        float acc = 0.0;
        [[unroll]] for (uint j = 0u; j < KPT; ++j) acc += qv[j] * hidden[ho + j * 64u];
        float sum = subgroupAdd(acc);
        if (gl_NumSubgroups > 1u) {
            if (gl_SubgroupInvocationID == 0u) red[gl_SubgroupID] = sum;
            barrier();
            if (tid == 0u) sum = red[0] + red[1];
        }
        if (tid == 0u) y[(rq * pc.n_used + s) * pc.n_embd + o] = sel[rq].w[s] * sum;
        if (gl_NumSubgroups > 1u) barrier();
    }
}
