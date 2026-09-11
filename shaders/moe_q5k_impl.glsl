#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x_id = 0) in;
struct block_q5_K { float16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
layout(std430, binding = 0) readonly buffer G { block_q5_K gw[]; };
layout(std430, binding = 1) readonly buffer U { block_q5_K uw[]; };
layout(std430, binding = 2) readonly buffer X { float x[]; };
#if Q5_SHARED
layout(std430, binding = 3) writeonly buffer H { float h[]; };
#else
struct SelT { uint ids[16]; float w[16]; float wShared; float pad[7]; };
layout(std430, binding = 3) readonly buffer S { SelT sel[]; };
layout(std430, binding = 4) writeonly buffer H { float h[]; };
#endif
layout(push_constant) uniform PC { uint n_embd; uint n_ff; uint n_expert; uint n_used; } pc;
shared float pg[256], pu[256];

uvec2 scale_min(block_q5_K b, uint g) {
    return g < 4u ? uvec2(uint(b.scales[g]) & 63u, uint(b.scales[g+4u]) & 63u)
        : uvec2((uint(b.scales[g+4u]) & 15u) | ((uint(b.scales[g-4u]) >> 6u) << 4u),
                (uint(b.scales[g+4u]) >> 4u) | ((uint(b.scales[g]) >> 6u) << 4u));
}
uint quant(block_q5_K b, uint g, uint i) {
    return ((uint(b.qs[(g/2u)*32u+i]) >> ((g%2u)*4u)) & 15u) |
           (((uint(b.qh[i]) >> g) & 1u) << 4u);
}
void main() {
    uint wg = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x, rq = gl_WorkGroupID.z;
#if Q5_SHARED
    uint s = pc.n_used, row = wg, expert = 0u;
    bool valid = row < pc.n_ff;
#else
    uint s = wg / pc.n_ff, row = wg % pc.n_ff;
    bool valid = s < pc.n_used;
    uint expert = valid ? min(sel[rq].ids[s], pc.n_expert - 1u) : 0u;
#endif
    float ag = 0.0, au = 0.0;
    if (valid) {
        uint blocks = pc.n_embd / 256u;
        uint base = (expert * pc.n_ff + row) * blocks;
        for (uint sub = tid; sub < pc.n_embd / 32u; sub += gl_WorkGroupSize.x) {
            uint b = base + sub / 8u, g = sub % 8u;
            uvec2 gs = scale_min(gw[b], g), us = scale_min(uw[b], g);
            float gd = float(gw[b].d) * float(gs.x), gm = float(gw[b].dmin) * float(gs.y);
            float ud = float(uw[b].d) * float(us.x), um = float(uw[b].dmin) * float(us.y);
            for (uint i = 0u; i < 32u; ++i) {
                float v = x[rq * pc.n_embd + sub * 32u + i];
                ag += (gd * float(quant(gw[b], g, i)) - gm) * v;
                au += (ud * float(quant(uw[b], g, i)) - um) * v;
            }
        }
    }
    float sg = subgroupAdd(ag), su = subgroupAdd(au);
    if (gl_SubgroupInvocationID == 0u) { pg[gl_SubgroupID] = sg; pu[gl_SubgroupID] = su; }
    barrier();
    if (tid == 0u && valid) {
        float g = 0.0, u = 0.0;
        for (uint i = 0u; i < gl_NumSubgroups; ++i) { g += pg[i]; u += pu[i]; }
        h[(rq * (pc.n_used + 1u) + s) * pc.n_ff + row] = (g / (1.0 + exp(-g))) * u;
    }
}
