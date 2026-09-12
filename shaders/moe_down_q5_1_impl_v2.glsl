// Word-addressed twin of moe_down_q5_1_impl.glsl (routed when SHARED_DOWN==0,
// shared when 1): six dword loads and eight vec4 loads per 24-byte block.
// F32-close, not bit-identical, to the byte-addressed v1 (QK_MOE_DOWN=v1).
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : enable
layout(local_size_x_id = 0) in;
struct SelT { uint ids[16]; float w[16]; float wShared; float pad[7]; };
layout(std430,binding=0) readonly buffer W { uint w32[]; };
layout(std430,binding=1) readonly buffer H { vec4 h4[]; };
layout(std430,binding=2) readonly buffer S { SelT sel[]; };
layout(std430,binding=3) buffer Y { float y[]; };
layout(push_constant) uniform PC { uint n_embd,n_ff,n_expert,n_used; } pc;
shared float partial[256];
void main() {
    uint row=gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x;
    uint tid=gl_LocalInvocationID.x, rq=gl_WorkGroupID.z, blocks=pc.n_ff/32u;
    float acc=0.0;
    if (row<pc.n_embd) for (uint task=tid; task<blocks*(SHARED_DOWN!=0 ? 1u : pc.n_used); task+=gl_WorkGroupSize.x) {
        uint s=SHARED_DOWN!=0 ? pc.n_used : task/blocks, b=task%blocks;
        uint expert=SHARED_DOWN!=0 ? 0u : min(sel[rq].ids[s],pc.n_expert-1u);
        uint base=((expert*pc.n_embd+row)*blocks+b)*6u;
        uint ho=((rq*(pc.n_used+1u)+s)*pc.n_ff+b*32u)/4u;
        vec2 dm=unpackHalf2x16(w32[base]);
        uint qh=w32[base+1u];
        float dotQ=0.0, sumX=0.0;
        [[unroll]] for (uint j=0u; j<4u; ++j) {
            uint qs=w32[base+2u+j];
            vec4 xl=h4[ho+j], xh=h4[ho+4u+j];
            uint e=4u*j;
            vec4 ql=vec4(float((qs&15u)|(((qh>>e)&1u)<<4u)),
                         float(((qs>>8u)&15u)|(((qh>>(e+1u))&1u)<<4u)),
                         float(((qs>>16u)&15u)|(((qh>>(e+2u))&1u)<<4u)),
                         float(((qs>>24u)&15u)|(((qh>>(e+3u))&1u)<<4u)));
            vec4 qhv=vec4(float(((qs>>4u)&15u)|(((qh>>(e+16u))&1u)<<4u)),
                          float(((qs>>12u)&15u)|(((qh>>(e+17u))&1u)<<4u)),
                          float(((qs>>20u)&15u)|(((qh>>(e+18u))&1u)<<4u)),
                          float(((qs>>28u)&15u)|(((qh>>(e+19u))&1u)<<4u)));
            dotQ+=dot(ql,xl)+dot(qhv,xh);
            sumX+=xl.x+xl.y+xl.z+xl.w+xh.x+xh.y+xh.z+xh.w;
        }
        float gate=SHARED_DOWN!=0 ? sel[rq].wShared : sel[rq].w[s];
        acc+=gate*(dm.x*dotQ+dm.y*sumX);
    }
    float wave=subgroupAdd(acc);
    if (gl_SubgroupInvocationID==0u) partial[gl_SubgroupID]=wave;
    barrier();
    if (tid==0u && row<pc.n_embd) {
        float sum=0.0;
        for (uint i=0u; i<gl_NumSubgroups; ++i) sum+=partial[i];
        if (SHARED_DOWN!=0) y[rq*pc.n_embd+row]+=sum;
        else y[rq*pc.n_embd+row]=sum;
    }
}
