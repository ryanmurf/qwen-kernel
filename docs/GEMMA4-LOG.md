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

### 2026-07-18 — Stage 0 sandbox limitation (superseded for bandwidth, still open for fixtures)

- **Stage 0 gate blocked; stopped as required.** The codex sandbox has no
  `/dev/dri`. qk enumerated only llvmpipe and rejected `QK_DEVICE_PCI=1a:00.0`;
  llama.cpp reported `ggml_vulkan: No devices found.` Host sysfs showed both
  target cards idle (0% busy), but device nodes were unavailable. Consequently
  there is no honest Q4_0 GB/s result, no same-day llama.cpp baseline, and no
  twice-reproduced parity fixture. Blocked machine-readable records and exact
  host rerun commands are in `tests/gemma4/` and
  `bench/results-gemma4-llamacpp.jsonl`; acceptance numbers were not lowered or
  backfilled from earlier runs.
