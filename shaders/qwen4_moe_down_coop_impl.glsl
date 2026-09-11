// Cooperative-matrix twin of qwen4_moe_down_tiled_impl.glsl (reduced-precision
// tier: f16 inputs, F32 accumulation). A 128-row output tile of one expert
// against 16-token pair groups; each 64-wide subgroup owns two 16-row tiles.
// Gated partials are scattered to y[(token*n_used + slot)*n_embd + o] through
// LDS. Host requires n_embd % 128 == 0, n_ff % 64 == 0 and a 64-wide subgroup.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require

layout(local_size_x = 256) in;

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

const uint TILE = 16u, BM = 128u, TN = 16u, BK = 64u, SK = BK + 8u, SO = TN + 4u;
shared float16_t Wsh[BM * SK];
shared float16_t Xsh[TN * SK];
shared float Osh[BM * SO];
shared uint tokTable[TN];
shared uint slotTable[TN];

void main() {
    uint tile = gl_WorkGroupID.x;
    uint eid = gl_WorkGroupID.y;
    uint tid = gl_LocalInvocationID.x;
    uint wave = gl_SubgroupID;
    if (tile * BM >= pc.n_embd || eid >= pc.n_expert) return;
    uint begin = offsets[eid], end = offsets[eid + 1u];
    if (begin == end) return;

    uint blocks = pc.n_ff / 32u;
    uint rowLocal = tid & 127u;
    uint blockSel = tid >> 7;
    uint rowBlockBase = (eid * pc.n_embd + tile * BM + rowLocal) * blocks;

    for (uint p0 = begin; p0 < end; p0 += TN) {
        uint np = min(TN, end - p0);
        if (tid < TN) {
            uint pair = tid < np ? pairs[p0 + tid] : 0u;
            tokTable[tid] = pair >> 4u;
            slotTable[tid] = pair & 15u;
        }
        coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>
            c0 = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0),
            c1 = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0);
        barrier();
        for (uint k0 = 0u; k0 < pc.n_ff; k0 += BK) {
            uint b = rowBlockBase + k0 / 32u + blockSel;
            uint dst = rowLocal * SK + blockSel * 32u;
#if DOWN_Q51
            float d = float(dw[b].d), m = float(dw[b].m);
            [[unroll]] for (uint i = 0u; i < 32u; ++i) {
                uint lo = (uint(dw[b].qs[i % 16u]) >> ((i / 16u) * 4u)) & 15u;
                uint hi = (uint(dw[b].qh[i / 8u]) >> (i % 8u)) & 1u;
                Wsh[dst + i] = float16_t(d * float(lo + 16u * hi) + m);
            }
#else
            float d = float(dw[b].d);
            [[unroll]] for (uint i = 0u; i < 32u; ++i) Wsh[dst + i] = float16_t(d * float(int(dw[b].qs[i])));
#endif
            for (uint idx = tid; idx < TN * BK; idx += 256u) {
                uint tt = idx / BK, kk = idx - tt * BK;
                Xsh[tt * SK + kk] = float16_t(tt < np
                    ? hidden[(tokTable[tt] * (pc.n_used + 1u) + slotTable[tt]) * pc.n_ff + k0 + kk] : 0.0);
            }
            barrier();
            [[unroll]] for (uint kt = 0u; kt < BK; kt += TILE) {
                coopmat<float16_t, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseB> bx;
                coopMatLoad(bx, Xsh, kt, SK, gl_CooperativeMatrixLayoutColumnMajor);
                coopmat<float16_t, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseA> a;
                coopMatLoad(a, Wsh, (wave * 2u) * TILE * SK + kt, SK, gl_CooperativeMatrixLayoutRowMajor);
                c0 = coopMatMulAdd(a, bx, c0);
                coopMatLoad(a, Wsh, (wave * 2u + 1u) * TILE * SK + kt, SK, gl_CooperativeMatrixLayoutRowMajor);
                c1 = coopMatMulAdd(a, bx, c1);
            }
            barrier();
        }
        coopMatStore(c0, Osh, (wave * 2u) * TILE * SO, SO, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(c1, Osh, (wave * 2u + 1u) * TILE * SO, SO, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        for (uint idx = tid; idx < BM * TN; idx += 256u) {
            uint r = idx / TN, tt = idx - r * TN;
            if (tt < np) {
                uint tok = tokTable[tt], slot = slotTable[tt];
                y[(tok * pc.n_used + slot) * pc.n_embd + tile * BM + r] = sel[tok].w[slot] * Osh[r * SO + tt];
            }
        }
        barrier();
    }
}
