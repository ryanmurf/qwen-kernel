// Word-addressed twin of moe_q5k_impl.glsl (routed gate/up when Q5_SHARED==0,
// shared expert when 1). Each lane owns one 32-element Q5_K sub-block of the
// gate and up rows and reads its scale words, eight qh words and eight qs
// words as 32-bit loads plus eight vec4 activation loads, like gemv_q5_k.
// The per-sub-block sum is d*sum(q*x) - m*sum(x), so results are F32-close
// (not bit-identical) to the byte-addressed kernel, which stays as the v1
// path for A/B (QK_MOE_GU=v1). Same bindings, push constants, workgroup
// specialization and output layout.
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer G { uint gw[]; };
layout(std430, binding = 1) readonly buffer U { uint uw[]; };
layout(std430, binding = 2) readonly buffer X { vec4 x4[]; };
#if Q5_SHARED
layout(std430, binding = 3) writeonly buffer H { float h[]; };
#else
struct SelT { uint ids[16]; float w[16]; float wShared; float pad[7]; };
layout(std430, binding = 3) readonly buffer S { SelT sel[]; };
layout(std430, binding = 4) writeonly buffer H { float h[]; };
#endif
layout(push_constant) uniform PC { uint n_embd; uint n_ff; uint n_expert; uint n_used; } pc;
shared float pg[256], pu[256];

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
        uint rowBase = (expert * pc.n_ff + row) * blocks;
        for (uint sub = tid; sub < pc.n_embd / 32u; sub += gl_WorkGroupSize.x) {
            uint b = rowBase + (sub >> 3u), g = sub & 7u, base = b * 44u;
            uint qsBase = base + 12u + (g >> 1u) * 8u, shift = (g & 1u) * 4u;
            uint xo = (rq * pc.n_embd + sub * 32u) / 4u;
            // gate
            vec2 gdm = unpackHalf2x16(gw[base]);
            uint gs0 = gw[base + 1u], gs1 = gw[base + 2u], gs2 = gw[base + 3u];
            uint gsc, gmn;
            if (g < 4u) { gsc = (gs0 >> (8u * g)) & 63u; gmn = (gs1 >> (8u * g)) & 63u; }
            else { uint j = 8u * (g - 4u); uint a = (gs2 >> j) & 255u;
                   gsc = (a & 15u) | ((((gs0 >> j) & 255u) >> 6u) << 4u);
                   gmn = (a >> 4u) | ((((gs1 >> j) & 255u) >> 6u) << 4u); }
            // up
            vec2 udm = unpackHalf2x16(uw[base]);
            uint us0 = uw[base + 1u], us1 = uw[base + 2u], us2 = uw[base + 3u];
            uint usc, umn;
            if (g < 4u) { usc = (us0 >> (8u * g)) & 63u; umn = (us1 >> (8u * g)) & 63u; }
            else { uint j = 8u * (g - 4u); uint a = (us2 >> j) & 255u;
                   usc = (a & 15u) | ((((us0 >> j) & 255u) >> 6u) << 4u);
                   umn = (a >> 4u) | ((((us1 >> j) & 255u) >> 6u) << 4u); }
            float sg = 0.0, su = 0.0, xs = 0.0;
            [[unroll]] for (uint j = 0u; j < 8u; ++j) {
                uint gqs = gw[qsBase + j] >> shift, gqh = gw[base + 4u + j] >> g;
                uint uqs = uw[qsBase + j] >> shift, uqh = uw[base + 4u + j] >> g;
                vec4 xv = x4[xo + j];
                vec4 gq = vec4(float((gqs & 15u) | ((gqh & 1u) << 4u)),
                               float(((gqs >> 8u) & 15u) | (((gqh >> 8u) & 1u) << 4u)),
                               float(((gqs >> 16u) & 15u) | (((gqh >> 16u) & 1u) << 4u)),
                               float(((gqs >> 24u) & 15u) | (((gqh >> 24u) & 1u) << 4u)));
                vec4 uq = vec4(float((uqs & 15u) | ((uqh & 1u) << 4u)),
                               float(((uqs >> 8u) & 15u) | (((uqh >> 8u) & 1u) << 4u)),
                               float(((uqs >> 16u) & 15u) | (((uqh >> 16u) & 1u) << 4u)),
                               float(((uqs >> 24u) & 15u) | (((uqh >> 24u) & 1u) << 4u)));
                sg += dot(gq, xv); su += dot(uq, xv);
                xs += xv.x + xv.y + xv.z + xv.w;
            }
            ag += gdm.x * float(gsc) * sg - gdm.y * float(gmn) * xs;
            au += udm.x * float(usc) * su - udm.y * float(umn) * xs;
        }
    }
    float rg = subgroupAdd(ag), ru = subgroupAdd(au);
    if (gl_SubgroupInvocationID == 0u) { pg[gl_SubgroupID] = rg; pu[gl_SubgroupID] = ru; }
    barrier();
    if (tid == 0u && valid) {
        float g = 0.0, u = 0.0;
        for (uint i = 0u; i < gl_NumSubgroups; ++i) { g += pg[i]; u += pu[i]; }
        h[(rq * (pc.n_used + 1u) + s) * pc.n_ff + row] = (g / (1.0 + exp(-g))) * u;
    }
}
