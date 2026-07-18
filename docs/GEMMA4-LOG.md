# Gemma 4 26B-A4B port — engineering log

Running log for the Gemma-4 port on branch `gemma4-port`. Newest entries at the
bottom. Every checkpoint is a commit so regressions can be bisected. Design doc:
[`GEMMA4-PLAN.md`](GEMMA4-PLAN.md).

## Fixed facts (verified, do not re-derive)

Target: `/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf`
14,439,363,584 B (13.45 GiB), 658 tensors, arch `gemma4`, text-only
(do not load the vision projector).

| fact | value | how verified |
|---|---|---|
| layer split | **25 sliding / 5 global**, globals at 0-idx 5,11,17,23,29 | counted the 30 `sliding_window_pattern` booleans in the GGUF |
| sliding attn | hd 256, 16 Q heads, 8 KV heads, rope base 1e4 | GGUF KV + `blk.0` tensor shapes |
| global attn | hd 512, 16 Q heads, 2 KV heads, rope base 1e6 | GGUF KV + `blk.5` tensor shapes |
| global V proj | **absent** — K proj reused as V input | no `blk.5.attn_v.weight` in tensor table |
| global K vs V | **distinct in cache** — K gets learned k_norm + rope, V gets plain unweighted rms_norm and no rope | `gemma4.cpp:247-262` |
| attention scale | **1.0**, NOT 1/sqrt(head_dim) | `gemma4.cpp:11` |
| LM head | tied `token_embd`, the sole **Q6_K** tensor, 605,552,640 B | tensor histogram: F32 x392, Q4_0 x265, Q6_K x1 |
| activation | **GELU**, not SiLU | `gemma4.cpp` build_ffn / build_moe_ffn |
| expert layout | `ffn_gate_up_exps` fused in-file (2816,1408,128) -> 2 GEMVs/expert | tensor table |
| dense + MoE | **parallel branches** off attn_out, summed | `gemma4.cpp:300-343` |
| router input | rms_norm(attn_out) * 1/sqrt(2816) * `ffn_gate_inp.scale` | `gemma4.cpp:316-320` |
| softcap | `30*tanh(x/30)`; monotone, cannot change greedy argmax | GGUF `final_logit_softcapping` |
| MTP/NextN | **absent** from this GGUF | grepped tensor names, zero hits |
| active traffic | **2,380,055,608 B/token** | two independent codex derivations agree |

Dead for this model: every `dn_*.comp` (DeltaNet) shader and all recurrent
state/carry/snapshot code.

## Baseline — llama.cpp master `a935fbf`, Vulkan, measured 2026-07-18

Command shape: `llama-bench -m <model> -ngl 99`, card selected with
`GGML_VK_VISIBLE_DEVICES` (2 = XTX, 1 = XT).

| card | pp512 | tg128 |
|---|---:|---:|
| RX 7900 XTX | 3505.04 +/- 103.25 | **139.36 +/- 1.20** |
| RX 7900 XT | 2389.09 +/- 6.50 | **119.34 +/- 0.08** |

XTX depth sweep: tg64 139.74 (d0) -> 124.32 (d4096) -> 117.36 (d16384). The
mild decay is the SWA cap on 25 of 30 layers behaving as designed.

XT/XTX tg ratio is 0.856 vs a bandwidth ratio of 0.833. This looks like clean
bandwidth scaling, but it does **not** prove decode is bandwidth-bound: the XT
also has 84 CUs vs 96 (ratio 0.875), so compute and bandwidth scale together
across these two cards and the test cannot separate the two hypotheses.

At 2.380 GB/token, 139.36 tok/s implies only ~331.7 GB/s effective — about 36%
of the 920 GB/s measured simple-stream rate. That gap is the opportunity.

Note: `GEMMA4-PLAN.md` section 5 predates this measurement and guesses 130 tok/s
(range 100-160). The guess brackets the real number; treat 139.36/3505 as
authoritative and the plan's provisional baseline as superseded.

## Process notes

- **rtk transforms `grep`/`sed`.** A plain `grep -n` returned "25 matches in 3
  files" including hits from unrelated files — it is doing semantic search, not
  literal matching, and reported line numbers past EOF. Use the Read tool (or
  `rtk proxy`) for anything load-bearing.
- **Never assume a background codex run died.** An early run was presumed dead
  from a zero-byte log, but was still executing and later overwrote
  `GEMMA4-PLAN.md`, clobbering a reviewed revision. Check the specific task id.

## Checkpoints

| commit | stage | what landed |
|---|---|---|
| `0be6893` | — | design plan committed (planning only, no engine code) |
