# 80B port brief — Metal side (from tron, 2026-07-10)

Goal: generalize the Metal engine for **Qwen3-Next-80B-A3B-Instruct** and bring it up
as a pipe-worker so we can split-serve it across tron+midnight. Tron's Vulkan side is
done and gated: **pull commit `99cdedf`** ("Engine: generalize for Qwen3-Next-80B…")
from origin/main — it is the exact recipe; this file is the map.

## The model file (already on your disk)
`models/Qwen3-Next-80B-A3B-Instruct-IQ4_XS-qk.gguf` (42.8 GB). It is REPACKED for our
kernel set: every tensor ∈ {F32, IQ4_XS, Q8_0, Q6_K}. Notably: token_embd **Q8_0**,
output (head) Q6_K, all shexp weights **Q8_0**, routed gate/up/down exps **IQ4_XS**,
dense projections **IQ4_XS except the 12 attn_v which are Q8_0**, all DN small tensors
F32 except **ssm_ba which is IQ4_XS and fused** (see below). Do not use any non `-qk`
80B file.

## Shape deltas vs the 35B (everything else is IDENTICAL — dims, heads, rope, conv)
- 48 layers (attention every 4th, same `(i+1)%4` rule; detect by tensor presence as today)
- 512 experts, top-10 (read `<arch>.expert_used_count`; arch = `qwen3next`)
- eos 151645 / bos 151643 (read `tokenizer.ggml.{eos,bos}_token_id`)
- NO MTP tensors (stripped upstream), no new graph ops — shared expert path unchanged

## Work list (mirror of tron diff 99cdedf)
1. GGUF KV reads: block_count, expert_used_count, eos/bos; n_expert/n_ff_exp from
   `ffn_gate_inp`/`ffn_gate_exps` shapes. Kill every hardcoded 40/256/8.
2. moe_select kernel: 512 lanes (n_expert ≤ 512); **SelT widens to
   `{ids[16], w[16], wShared, pad[7]}` = 160 B** — update every consumer
   (gateup iq3/iq4/q8, down iq4/q6k/q8/q8b) and host-side buffer sizes/strides.
3. NEW kernels (all three are clones with a different in-kernel dequant):
   - `moe_gateup_iq4` (structure of gateup_iq3, IQ4_XS sub-block decode of down_iq4)
   - `gemm_iq4_xs` (batched-prefill GEMM; same tiling as gemm_q8_0, W-staging
     dequants one 32-elem IQ4_XS sub-block per thread)
   - `embed_q8_0` (trivial)
4. Dense projections are per-tensor Q8_0|IQ4_XS → per-projection pipe choice
   (qkv/q, z/k, v, wo/ssm_out flags). Decode gemv + prefill gemm both.
5. **ssm_ba de-interleave (the one structural change).** qwen3next fuses beta+alpha:
   `ssm_ba [2048, 64]`, IQ4_XS. Rows are interleaved per k-head group g (16 groups
   of 4 rows): rows `g*4+0, g*4+1` = **beta** for v-heads `2g, 2g+1`; rows
   `g*4+2, g*4+3` = **alpha** for the same v-heads. Dequant to F32 on the host at
   load, de-interleave into the engine's existing split alpha[32][2048] /
   beta[32][2048] layout, upload — dn_ab kernel unchanged.
   (Order verified against llama.cpp master `src/models/qwen3next.cpp` view offsets:
   beta at offset 0, alpha at offset 2 within each group.)
6. MoE buffers sized at runtime: logits n_expert, hidden (n_used+1)*n_ff_exp,
   sel 160 B/token; push constants {n_embd, n_ff_exp, n_expert, n_used} runtime.

## Gates (same bar as always)
- 35B regression stays green (the SelT widening touches the 35B path!)
- your moe blk test vs CPU ref on the 80B blk 0 (tron: max_rel_err 5.5e-05, sel exact)
- stage-load smoke of a few 80B layers
- token-exact refs exist on tron at `/mnt/data/models/refs-80b/ref{1..3}.json`
  (llama.cpp CPU, temperature 0, 100 tok, generated from THIS exact repacked file)

## Serving plan (after your gates)
- Do NOT disturb the 35B worker on 18100 (prod split path).
- Bring the 80B worker up as a SECOND process, port **18200**, layers ~**12:48**
  (tron head owns ~0:12 — final split point TBD after VRAM check; your 36 layers
  ≈ 31-33 GB weights, fits the 51 GiB budget with ctx room; QK_MLOCK=1 as usual).
- I'll wire the tron head + deployment (task #43) once you report the worker up.

Write status/questions to PORT.md as usual. — tron
