#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(local_size_x_id = 0) in;
struct block_q5_1 { float16_t d,m; uint8_t qh[4]; uint8_t qs[16]; };
struct SelT { uint ids[16]; float w[16]; float wShared; float pad[7]; };
layout(std430,binding=0) readonly buffer W { block_q5_1 weights[]; };
layout(std430,binding=1) readonly buffer H { float hidden[]; };
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
        uint bi=(expert*pc.n_embd+row)*blocks+b;
        uint hi=(rq*(pc.n_used+1u)+s)*pc.n_ff+b*32u;
        float dotQ=0.0, sumX=0.0;
        for (uint i=0u; i<32u; ++i) {
            uint lo=(uint(weights[bi].qs[i%16u])>>((i/16u)*4u))&15u;
            uint hiBit=(uint(weights[bi].qh[i/8u])>>(i%8u))&1u;
            float x=hidden[hi+i];
            dotQ+=float(lo+16u*hiBit)*x; sumX+=x;
        }
        float gate=SHARED_DOWN!=0 ? sel[rq].wShared : sel[rq].w[s];
        acc+=gate*(float(weights[bi].d)*dotQ+float(weights[bi].m)*sumX);
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
