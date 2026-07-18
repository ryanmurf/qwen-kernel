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

## NOTE from tron (task #43, Fri Jul 10 18:10:12 MDT 2026): 80B worker :18200 stopped by tron agent
Reason: (1) user reports memory pressure on this mac — the worker wires 32 GB;
(2) first cross-box gate FAILED (coherent-but-wrong-context continuations,
deterministic; 35B split via :18100 is token-exact, so the bug is 80B-specific,
side unknown yet). Bisecting head-vs-worker on tron with llama.cpp CPU hidden
states now. Do NOT restart :18200 until I report back — will ask with exact
config when needed. The :18100 35B worker is untouched and needed (prod path).


## UPDATE from tron (task #43): root cause found — worker exonerated for the wire
## failure, but you have the SAME two engine bugs to fix. Pull 64192d5.

The failed gate was HEAD-side; your worker faithfully continued garbage-in.
Retraction with apologies. But both bugs live in code you ported, so your
Metal twins need the same fixes before the token gate can pass (your layers
[12,48) hold 27 DN layers):

1. **gemv_iq4_xs z-batching** (shaders/gemv_iq4_xs.comp): the Vulkan original
   had NO gl_WorkGroupID.z handling — batched positions all recomputed
   position 0 (x stride K, y stride M). Your pass-2 notes say you added
   z-batching on Metal — VERIFY yours actually offsets both x and y for z>0;
   the random-data harness runs z=1 and will not catch it.

2. **DeltaNet GQA k-pairing is ARCH-DEPENDENT** (dn_step.comp,
   dn_step_batch.comp — the real bug): qwen35moe tiles v->k modulo
   (kh = h % hK); **qwen3next pairs consecutively: k-head g serves v-heads
   2g and 2g+1, i.e. kh = h / (hV/hK) = h / 2.** Your ablock gate ran blk 3
   (attention), so no DN block ever hit this. Proof: simulated both wirings
   from engine taps against llama.cpp's per-op dumps — consecutive scores
   cos 0.999972 on the block output, modulo 0.9921. Fix shape in 64192d5:
   new kDiv push constant (0 = modulo/35B, hV/hK = qwen3next), set from
   general.architecture at open. 35B behavior is bit-preserved (all 35B
   gates re-run green, single-box AND split serve token-exact).

After pulling: patch your dn_step twins the same way, rebuild, re-run your
35B parity (must stay exact, kDiv=0 there), then relaunch the worker with
the same config as before:
    QK_MLOCK=1 qk pipe-worker 18200 12:48 32768 2
Heads-up: the user reported memory pressure on this box earlier today — the
worker re-wires ~32 GB. Relaunch anyway (it is the product), but keep your
build/test footprint lean while it runs.

Tron-side status: head [0,12) matches llama l_out-11 at cos 0.999+ for all
positions (residue = llama's own Q8_K activation-quant noise). The decisive
token gate vs refs-80b runs from here the moment your worker is back on
:18200 — report in this file or just bring the port up; I monitor both.

Debug kit if you want to re-derive any of this on Metal (all in 64192d5,
env-gated): QK_PIPE_DUMP (hidden frames), QK_PIPE_SERIAL (1-token frames),
QK_DUMP_X (post-embd readback), QK_DUMP_TAPS (per-op layer-0 readbacks), plus
the llama.cpp common/debug.cpp raw-dump patch recipe in tron's session notes.

## CLOSED from tron: 80B split GATE PASSED (2026-07-10 evening)

Your rebuilt worker is numerically EXACT — the decisive gate ran through the
real split (in-cluster head + your :18200): ref3 100/100 tokens exact; ref1
and ref2 prefix-exact to CERTIFIED llama near-ties (llama top-2 gaps 0.006
and 0.11 logprob; qk picks llama's #2 — that's llama's Q8_K activation-quant
noise vs our f32-exact activations, not drift). Determinism x2 exact,
/v1/messages clean, ~30 tok/s steady single-stream decode over WiFi. Your
c38e951 chunked-DN change introduced no drift (divergences land only on
llama's coin-flips). Full runbook + gate semantics: docs/split-serving.md
(tron 414bee7).

Operational state: `./switch.sh split80` on tron flips prod traffic to the
80B (one GPU backend at a time; 35B is the default). Keep the :18200 worker
resident — it is the product now, same standing as :18100. Task #43 closed.
Nice work on the fast turnaround.
