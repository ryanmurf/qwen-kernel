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

## Historical baseline — superseded by current 571d0d5 results below

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

These values remain useful historical context. The current decode comparator is
the 571d0d5 depth sweep recorded in the latest entry below; 3505.04 tok/s
remains the historical prefill bar until a clean current-revision run replaces
it.

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

### 2026-07-18 — Stage 0 partial: artifact ledger and Q4 measurement harness

- Added `tests/gemma4/generate_manifest.py` and the generated `manifest.json`.
  The generator passed the exact local/remote SHA-256
  `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`,
  14,439,363,584-byte size, 52 metadata KVs, 658 tensor ranges, F32/Q4_0/Q6_K
  histogram 392/265/1, 25 sliding plus 5 global layers at 5/11/17/23/29,
  global `attn_v` absence, and exactly 2,380,055,608 active bytes/token. The
  Hugging Face 14.4 GB label is rounded decimal display; its SHA matches local.
- Extended the standalone Vulkan format harness with a measurement-only Q4_0
  block reader, CPU reference dequantizer, and shader. `/usr/bin/cmake` build,
  `spirv-val --target-env vulkan1.2`, and a small llvmpipe CPU-reference check
  passed (`max_rel_err=1.65e-05`). No Gemma graph, attention, MoE, or production
  GEMV path was added.
### 2026-07-18 — Q4_0 bandwidth measured on real hardware (unblocking the above)

Run from a shell that does have `/dev/dri`. `QK_TPR` was added to `runGemv` so
the geometry can be swept; the derived default pinned tpr=256 for this shape.
288 MiB matrix (M=32768, K=16384), 2000 GPU-timestamped iterations, VRAM
resident, correctness PASS against CPU dequant (max_rel_err <= 2.8e-4).

| tpr | XTX GB/s | XT GB/s |
|---:|---:|---:|
| 32 | **801.5** | **558.3** |
| 64 | 753.1 | 510.2 |
| 128 | 790.6 | 517.1 |
| 256 | 766.2 | 492.3 |

tpr=32 wins on both cards, as the plan's lane-occupancy argument predicted.
XTX tpr=32 repeats: 801.9 / 801.7 / 801.4 / 801.4 — reproducible to +/-0.3.

Against the existing format anchors:

| format | XTX GB/s | XT GB/s | Q4_0 relative |
|---|---:|---:|---|
| Q8_0 | 927.8 | 775.0 | — |
| f16 | 917.8 | 778.8 | — |
| Q6_K | 800.4 | — | — |
| **Q4_0** | **801.5** | **558.3** | 87% of f16 on XTX, **72% on XT** |
| IQ4_XS | 729.0 | — | — |

Two findings:

1. **Q4_0 lands at ~801 GB/s on XTX — essentially level with Q6_K, well above
   IQ4_XS (729), but 14% below Q8_0.** The plan's idealized floors quote 900
   GB/s and are therefore ~12% optimistic. The section-5 *budget* table is not:
   its implied effective rates are 761 (attention), 717 (experts), 700 (shared
   FFN) and 776 (head) GB/s, all at or below the measured large-shape 801.5, as
   they should be for smaller real shapes. **The 270 tok/s XTX prediction
   survives this measurement.**
2. **Q4_0 scales worse than bandwidth across cards** (0.70 XT/XTX vs a 0.833
   bandwidth ratio), while f16/Q8_0 scale at ~0.85. The nibble unpack has real
   ALU cost that the 84-CU card feels more. The plan's XT prediction of 222
   tok/s (range 210-235) assumed near-bandwidth scaling and should be revised
   down pending Stage 1 measurement on real shapes.

Caveat: this is one large, perfectly-shaped matrix. It is an upper bound, not
evidence that the 1-3 MiB expert shapes will reach it. Stage 1 must re-measure
on real Gemma shapes before any of this is treated as settled.

### 2026-07-18 — Stage 0 sandbox limitation (fully superseded)

- **Stage 0 gate blocked; stopped as required.** The codex sandbox has no
  `/dev/dri`. qk enumerated only llvmpipe and rejected `QK_DEVICE_PCI=1a:00.0`;
  llama.cpp reported `ggml_vulkan: No devices found.` Host sysfs showed both
  target cards idle (0% busy), but device nodes were unavailable. Consequently
  there is no honest Q4_0 GB/s result, no same-day llama.cpp baseline, and no
  twice-reproduced parity fixture. Blocked machine-readable records and exact
  host rerun commands are in `tests/gemma4/` and
  `bench/results-gemma4-llamacpp.jsonl`; acceptance numbers were not lowered or
  backfilled from earlier runs.

### 2026-07-18 — Stage 0 complete; Stage 1 standalone loader and quant kernels

- Updated the reference to llama.cpp
  `571d0d540df04f25298d0e159e520d9fc62ed121`. The already-measured XTX
  comparator is tg128 137.73 tok/s at d0, 127.77 at d4096, and 120.76 at
  d16384; it was not rerun here. The local `common/debug.cpp`
  `QK_DUMP_DIR`/`QK_DUMP_FILTER` patch remains available for later graph parity.
- Froze six numeric-ID fixtures: ordinary chat, coding, first-continuation
  positions 1023/1024/1025, and an exact 8192-ID context. Each stores the exact
  server/generator commands, sampler settings, reference commit, model SHA-256,
  inputs, continuations, and two matching continuation hashes. All used
  temperature zero, penalties off, `cache_prompt=false`, and `ignore_eos=true`.
- Fixed the Qwen `serve-bench` `produced=0` issue. `qk_slot_start` legitimately
  leaves a short serial prefill tail; the benchmark had treated a zero-output
  prefill-progress call as a stop. A bounded-progress guard now permits it.
  Verification at XTX `gpu_busy=0%`: ids2 emitted its reference first token
  198 and ids4 emitted reference token 13 with `QK_CHUNK=1`.
- Added a Stage-1-only Gemma loader with explicit roles for all 658 tensors,
  exact shape/type/range checks, native 18-byte Q4_0 recognition, global-layer
  V-projection absence checks, and explicit vision-name skipping. The target
  passed with 658 mapped text tensors and zero vision tensors. Full-file encoded
  payload is 13,771,929,600 Q4_0 + 605,552,640 Q6_K + 46,056,568 other bytes;
  this is distinct from the 1,728,460,800 active Q4_0 bytes per token because
  the file stores all 128 experts.
- Promoted Q4_0 GEMV from a measurement-only shader to the standalone native
  decode primitive and added BM128/BK64/local256 Q4_0 GEMMs for BN32 and BN64.
  GEMV/GEMM share `q4_0_dequant_pair`; K=704/2112/2816 block counts are pipeline
  specializations. No Gemma graph, attention, MoE scheduling, or layer assembly
  was added.
- All real Q4 shapes passed CPU-dequant versus GPU correctness. GEMM passed
  every output at both N=32/BN32 and N=64/BN64 (worst max-abs/RMS
  `2.86e-6`). `spirv-val --target-env vulkan1.2` passed for `gemv_q4_0`, both
  Q4 GEMMs, `gemv_q6_k`, and both argmax passes.

Q4_0 GEMV was swept over TPR 8/16/32/64/128/256 using real model weights,
2,000 GPU-timestamped iterations, and descriptor-cycled copies spanning at
least 128 MiB so the 1-13 MiB tensors could not remain hot in Infinity Cache.
Every timing run below started at XTX `gpu_busy=0%`; rates count encoded weight
bytes only.

| shape | best TPR | shader us | GB/s | gpu_busy | workgroups | cold/hot |
|---|---:|---:|---:|---:|---:|---:|
| sliding Q 4096x2816 | 128 | 9.5 | **685.8** | 0% | 2048 | 1.242x |
| global Q 8192x2816 | 128 | 17.2 | **755.7** | 0% | 4096 | 1.179x |
| shared FFN 2112x2816 | 256 | 6.5 | **513.6** | 0% | 2112 | 1.087x |
| shared down 2816x2112 | 16 | 8.0 | **416.8** | 0% | 176 | 1.158x |
| one expert gate/up 1408x2816 | 256 | 5.4 | **416.4** | 0% | 1408 | 1.112x |
| one expert down 2816x704 | 64 | 4.3 | **259.5** | 0% | 704 | 1.145x |

The small-expert result is the important finding: gate/up reaches only 52% of
the 801.5-GB/s large-stream upper bound and down only 32%. A hot repeated-tensor
run misleadingly exceeded 1 TB/s on the 12.4-MiB global-Q tensor, which is why
cache-cold cycling is now part of the harness. TPR32 is not a general winner;
the real-shape optima span TPR16 through TPR256.

The standalone Q4 GEMM is correct but clearly under-occupied at these small
output counts. At XTX `gpu_busy=0%`, N=32/BN32 delivered 2.83/5.76/1.59/2.07/
1.01/1.97 TFLOP/s for sliding Q, global Q, shared gate, shared down, expert
gate/up, and expert down. N=64/BN64 delivered 3.22/6.24/1.76/2.32/1.18/2.22
TFLOP/s. The dispatch has only 11-64 workgroups for most of these cases; this is
a Stage-6 retiling target, not a waived correctness gate.

The exact 262144x2816 Q6_K head also passed CPU dequant and GPU two-pass argmax
(ID 78183 on both, lower-index tie order). A TPR 8/16/32/64/128/256 sweep at
XTX `gpu_busy=0%` selected TPR16: **805.3 GB/s**, 752.0 us for GEMV plus 5.4 us
for the 64-workgroup argmax reduction. Full machine-readable sweeps are in
`tests/gemma4/stage1-results.json`.
