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

### 2026-07-18 — Stages 2--4 graph primitives, attention, and grouped top-8 MoE

- Implemented the Stage 2 dense primitives in Gemma graph order: tied Q6_K
  embedding/head, `sqrt(2816)` embedding scale, weighted and unweighted RMS
  norm, GELU, residual/output scalars, `30*tanh(x/30)`, and stable greedy
  argmax. The dense CPU/GPU gate passes; representative max-abs/RMS values are
  `4.86e-7` embedding, `3.76e-5` dense post-norm, and `5.33e-6` for a
  256-row tied-head slice. The adversarial softcap near-tie selects lower ID 17.
- Implemented both Stage 3 attention geometries with `f_attention_scale=1`:
  25 layer-local 1024-slot circular SWA caches and five linear global caches.
  Global raw V aliases the K projection input, while learned K norm+RoPE and
  unweighted V RMS write separate cache buffers. Q/K/V, cache slots,
  probabilities, attention outputs, output projections, and attention residual
  hidden states pass at positions 0, 1, 1023, 1024, and 1025. The worst
  boundary probability absolute error is `7.93e-6`; worst attention-output
  max-abs/RMS is `1.15e-4`. Real llama.cpp dumps pass for layer-0 sliding and
  layer-5 global raw projections (worst `2.59e-6`) and layer-0 post-RoPE Q/K/V
  (worst `1.86e-6`).
- Implemented the Stage 4 parallel shared/routed branches, the router's distinct
  unnormalized-attention input path, a 128-lane stable selector with an exact
  64-byte top-8 ABI, fused expert gate/up, per-expert down scaling, the three
  post norms, branch sum, attention residual, and layer-output scalar. Equal
  adversarial logits select IDs `[3,9,12,17,20,50,90,127]`. A real
  llama.cpp position-1 dump passes every branch tensor and the complete sparse
  block; the worst dump max-abs/RMS is `4.35e-6`.
- The dump campaign exposed an important backend numerical fact. On this AMD
  Vulkan path llama.cpp quantizes F32 activations to Q8_1 for Q4 projections at
  K=2816 and K=2112, but not for expert down at K=704. Reproducing that split,
  including the separately rounded Q8_1 sum member, closed initial 2--24%
  branch errors without relaxing a gate. The grouped down dispatch preserves
  per-expert reduction -> expert scale -> route weight -> stable serial-rank
  addition.

The final grouped campaign used 2,000 GPU-timestamped iterations and cycled 16
disjoint top-8 selections spanning all 128 experts, so each route reads a
different 17,842,176-byte gate/up payload and 8,921,088-byte down payload.
Every timing below began at XTX `gpu_busy_percent=0`.

| dispatch | winner | cold us / busy | repeat us / busy | GB/s | isolated baseline | delta |
|---|---:|---:|---:|---:|---:|---:|
| Q8_1 activation quant | -- | 2.52 / 0% | 2.66 / 0% | -- | -- | -- |
| grouped gate/up | TPR64 | 34.92 / 0% | 34.21 / 0% | **521.6** | 416.4 | **+25.3%** |
| grouped down | TPR64 | 26.62 / 0% | 26.79 / 0% | **333.1** | 259.5 | **+28.3%** |
| two weight dispatches | 64+64 | 59.26 / 0% | 59.22 / 0% | **451.9** | 346.6 effective | **+30.4%** |
| integrated quant+pair | 64+64 | 51.20 / 0% | 50.84 / 0% | -- | -- | -- |

The grouped design is a clear win over 480 tiny expert GEMVs, but it does
**not** reach the assumed 700 GB/s. The two weight dispatches are 35.4% below
that assumption and cost 1.7766 ms across 30 layers. The integrated
quant+gate/up+down phase is reproducibly faster (1.5252 ms/30 layers) because
the quant dispatch raises XTX auto-DPM residency; these separately timed phases
must not be subtracted from one another. Replacing Stage 1's 2.32-ms routed
term in the 5.14-ms budget yields about **230 tok/s** from the integrated phase,
or **218 tok/s** using the conservative separate-pair timing, rather than the
~252 tok/s 700-GB/s projection. These are budget replacements, not Stage 5
full-model throughput.

All 12 new SPIR-V modules pass `spirv-val --target-env vulkan1.2`; the build,
all three stage gates, real-dump gates, and `git diff --check` pass. CTest has no
registered tests. Full machine-readable results are in
`tests/gemma4/stage2-4-results.json`. Stage 5 remains the next run: persistent
30-layer serial assembly against the frozen numeric fixtures.

### 2026-07-18 — Stage 5 assembled, token-exact; superseded/disqualified

> **Historical record only.** Stage 5 copied llama.cpp `571d0d5`'s compiled
> SPIR-V into qk and dispatched it across the hot path. The parity result was
> real, but the qk/llama performance comparison was circular and the build was
> not reproducible. Stage 6 removes those binaries. None of the Stage 5
> performance numbers below are accepted as qk-native results.

- Added the persistent text-only Gemma 4 engine and wired all 30 layers. Layers
  5, 11, 17, 23, and 29 use global attention; the other 25 use sliding
  attention. Every sliding layer owns an independent canonical 1024-cell f16
  circular K/V ring, while every global layer owns an independent linear f16
  K/V cache. The decode graph keeps all model weights and KV state resident on
  the GPU. Prefill runs in chunks of up to 512 tokens.
- Numerical bisection against the supplied llama.cpp dump patch had established
  that token parity requires the oracle's exact Q4xF32 decode matvec/matvec-id,
  cooperative Q4xF32 prompt matmul, fused RMS+RoPE, and split-K flash-attention
  reduction order. Those frozen `571d0d5` SPIR-V modules were then part of the
  build. `QK_G4_DUMP_DIR` retains per-layer/per-op evidence capture, but all
  diagnostic copies are absent from normal benchmark command buffers.
- Added `gemma4-stage5-fixtures`, `gemma4-generate`, `gemma4-bench`, and
  integrated timestamp profiling for decode and pp512. The benchmark timer
  excludes cache reset and depth-prefix preparation. Decode measures exactly
  128 fresh one-token evaluations; pp512 measures exactly one 512-token prompt.

The hard parity command was:

```text
QK_DEVICE_PCI=1a:00.0 QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf ./build/qk gemma4-stage5-fixtures
```

It passed all six frozen fixtures token-for-token: ordinary chat 32/32, coding
32/32, the 1023/1024/1025 ring cases 16/16 each, and the 8192-token global case
16/16. Repeating the ordinary and coding fixtures from empty caches raised the
cumulative evidence to **1,024 exact generated tokens**. No near-tie waiver was
used. The XTX was at `gpu_busy_percent=0` before the run.

All head-to-head campaigns used the same model, XTX PCI `1a:00.0`, f16 K and V,
and began at 0% GPU busy. Values are medians with min--max spread over five
repetitions:

| test | qk tok/s | qk busy | llama.cpp tok/s | llama busy | qk / llama | result |
|---|---:|---:|---:|---:|---:|---|
| pp512 | 2508.74 (2500.69--2538.72) | 0% | **3432.78** (3405.56--3596.48) | 0% | 0.731x | qk loses |
| tg128 d0 | **146.10** (145.04--146.50) | 0% | 139.85 (139.19--139.93) | 0% | **1.045x** | qk wins |
| tg128 d4096 | **129.79** (129.53--130.37) | 0% | 127.89 (126.03--127.90) | 0% | **1.015x** | qk wins |
| tg128 d16384 | 104.15 (103.92--104.26) | 0% | **120.33** (119.09--120.66) | 0% | 0.866x | qk loses |

Exact qk commands (one idle-start campaign per line):

```text
QK_DEVICE_PCI=1a:00.0 QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf ./build/qk gemma4-bench pp 512 5
QK_DEVICE_PCI=1a:00.0 QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf ./build/qk gemma4-bench tg 0 5
QK_DEVICE_PCI=1a:00.0 QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf ./build/qk gemma4-bench tg 4096 5
QK_DEVICE_PCI=1a:00.0 QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf ./build/qk gemma4-bench tg 16384 5
```

Exact llama.cpp commands:

```text
GGML_VK_VISIBLE_DEVICES=2 /mnt/data/llama.cpp-master/build/bin/llama-bench -m /mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf -dev Vulkan0 -sm none -mg 0 -ngl 99 -fa on -ctk f16 -ctv f16 -p 512 -n 0 -d 0 -b 8192 -ub 512 -r 5 --delay 5 -o jsonl
GGML_VK_VISIBLE_DEVICES=2 /mnt/data/llama.cpp-master/build/bin/llama-bench -m /mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf -dev Vulkan0 -sm none -mg 0 -ngl 99 -fa on -ctk f16 -ctv f16 -p 0 -n 128 -d 0,4096,16384 -b 8192 -ub 512 -r 5 --delay 5 -o jsonl
```

The ~230 tok/s component budget does not survive full token-exact assembly.
Integrated timestamps account for the measured time rather than a shortened
harness:

| profile | attention | shared expert | routed experts | residual/norm | head | accounted | busy |
|---|---:|---:|---:|---:|---:|---:|---:|
| tg d0 | 2.238 ms (34.5%) | 0.936 ms (14.4%) | 1.993 ms (30.7%) | 0.266 ms (4.1%) | 1.058 ms (16.3%) | 6.492 ms | 0% |
| tg d16384 | 4.958 ms (53.6%) | 0.931 ms (10.1%) | 2.041 ms (22.1%) | 0.262 ms (2.8%) | 1.060 ms (11.5%) | 9.251 ms | 0% |
| pp512 | 36.558 ms (18.1%) | 17.972 ms (8.9%) | **144.880 ms (71.9%)** | 1.117 ms (0.6%) | 1.109 ms (0.6%) | 201.637 ms | 0% |

At d0, the parity-preserving frozen Q4xF32 projection and expert kernels are
materially slower than the Stage-1 Q4xQ8 component assumptions, while the tied
head alone costs about 1.06 ms. At d16384, attention rises by 2.72 ms and
becomes 53.6% of the token, which explains the deep-context loss to llama.cpp.
For pp512 the grouped batched MoE path dominates at 144.88 ms, explaining the
prefill loss. These are measured attributions, not inferred substitutions.

Raw samples, commands, GPU-busy readings, ratios, and profile summaries are in
`bench/results-gemma4-qk.jsonl`. The README benchmark section now records the
mixed outcome; qk is not described as faster than llama.cpp at all depths.

### 2026-07-19 — Stage 6 de-vendored, token-exact on qk kernels; honest loss

Stage 6 removes the Stage 5 shortcut completely. `CMakeLists.txt` no longer
copies, depends on, or names any module from llama.cpp's build tree, and the
engine no longer loads or dispatches those modules. A fresh `build-stage6`
configure and full build succeeded using only `shaders/*.comp`; the forbidden
reference search for `/mnt/data/llama.cpp-master`, `vulkan-shaders.spv`, and
`llama_*.spv` is empty in `CMakeLists.txt`, `src/`, and `shaders/`. The six new
qk-native modules also pass `spirv-val --target-env vulkan1.2`.

The parity repair was numerical, not binary imitation:

- Decode Q4 projections and routed experts use qk's Q4/Q8 kernels. Dot products
  and all reduction accumulators remain F32. The grouped pp down kernel stores
  its completed, scaled partial as f16 to bound workspace, then performs the
  final route-weight reduction in F32; it never accumulates a dot product in
  f16.
- Decode attention now emits qk-owned split-K softmax states and merges them
  with a stable 64-lane max/sum tree plus linear F32 output accumulation. This
  follows the published llama.cpp reduction algorithm, reimplemented against
  qk's own layouts and source shaders.
- The 8192-token prompt exposed a genuine router-sensitivity problem. Before
  the repair, the first final-token divergence was absolute position 8197:
  expected token 818, actual 4661, qk raw-logit gap `0.824741`. The oracle's
  softcapped logits were token 818 `28.826958` and token 4661 `28.625372`, gap
  `0.201586`, so this was not waived as a near-tie. Layerwise bisection found
  the first large hidden-state jump at layer 3 (relative error `8.89%`); the
  routed branch was `30.834%` relative error and the combined branch `7.746%`.
  At the top-8 router boundary qk selected expert 15 over 107 by `0.040723`,
  while llama.cpp selected 107 over 15 by only `0.003892`. Keeping serial
  long-prompt K/V projections in F32 from position 2048 removed that routing
  flip and closed the global-context fixture.
- The first SWA failure had been position 1036, expected token 506 versus 1816;
  llama.cpp's softcapped top-two gap was `0.195`. Applying the same qk-native
  split/reduction order to both global and sliding attention closed every ring
  boundary without a waiver.

The final gate was run from an empty cache with XTX
`gpu_busy_percent=0` at launch and completion:

```text
QK_DEVICE_PCI=1a:00.0 QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf ./build-stage6/qk gemma4-stage6-fixtures
```

All six fixtures passed token-for-token: ordinary chat 32/32, coding 32/32,
the 1023/1024/1025 SWA cases 16/16 each, and the 8192-token global case 16/16.
Repeating ordinary and coding from empty caches produced **1,024 exact generated
tokens** cumulatively. No waiver was used.

The formal fixture gate deliberately selects the serial accuracy path
(`QK_G4_CHUNK=1`, F32 long-prompt K/V from position 2048). The pp512 timing
uses the separate qk-native grouped batch path. As a spot check, that batched
path also reproduced ordinary chat 32/32, but the six-fixture/1,024-token claim
above is specifically for the accuracy path; it is not being silently extended
to every prompt batching choice.

The retained pp optimization is an expert-major grouped MoE path: it builds
expert assignment lists, reuses each expert's weights across up to 32 assigned
tokens, performs F32 gate/up and down GEMMs, and reduces the eight routes in
stable rank order. Routed-expert pp time fell from `437.954 ms` in the initial
de-vendored direct path (launch busy 0%) to `230.786 ms` in the final profile
(launch busy 0%). The complete pp profile fell from `757.893 ms` to
`519.250 ms`. Dense GEMM retile and attention-order experiments that regressed
performance or parity were reverted.

The honest final comparison is a loss in all four cases. Values are medians
with min--max spread over five repetitions. Every campaign started at
`gpu_busy_percent=0`; the accepted llama.cpp campaigns also ended at 0%.

| test | qk tok/s | qk busy | llama.cpp tok/s | llama busy | qk / llama | result |
|---|---:|---:|---:|---:|---:|---|
| pp512 | 984.80 (911.79--984.87) | 0% | **3405.88** (3386.36--3588.79) | 0% | 0.289x | qk loses |
| tg128 d0 | 105.43 (104.88--105.49) | 0% | **139.94** (139.56--140.23) | 0% | 0.753x | qk loses |
| tg128 d4096 | 64.27 (64.19--64.31) | 0% | **125.73** (123.79--126.43) | 0% | 0.511x | qk loses |
| tg128 d16384 | 25.57 (25.55--25.62) | 0% | **120.22** (118.75--120.45) | 0% | 0.213x | qk loses |

One llama.cpp pp campaign was explicitly rejected: it began at 0% busy, but a
separate `qk prefillcmp` process took the XTX to 100% during the run, producing
an invalid 72.425 tok/s average. It is retained as a rejected record in JSONL,
not used in the table. The replacement pp campaign began and ended at 0%.

Exact qk commands:

```text
QK_DEVICE_PCI=1a:00.0 QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf ./build-stage6/qk gemma4-bench pp 512 5
QK_DEVICE_PCI=1a:00.0 QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf ./build-stage6/qk gemma4-bench tg 0 5
QK_DEVICE_PCI=1a:00.0 QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf ./build-stage6/qk gemma4-bench tg 4096 5
QK_DEVICE_PCI=1a:00.0 QK_GGUF=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf ./build-stage6/qk gemma4-bench tg 16384 5
```

The llama.cpp commands are the Stage 5 commands above, rerun against
`571d0d5` with f16 K/V and Vulkan flash attention. Final qk-native profiles
attribute the losses as follows (all launches at 0% busy; d16384 also ended at
0%):

| profile | attention | shared expert | routed experts | residual/norm | head | accounted | busy |
|---|---:|---:|---:|---:|---:|---:|---:|
| tg d0 | 2.878 ms (39.3%) | 1.106 ms (15.1%) | 2.018 ms (27.6%) | 0.257 ms (3.5%) | 1.061 ms (14.5%) | 7.319 ms | 0% |
| tg d16384 | **34.444 ms (88.5%)** | 1.146 ms (2.9%) | 2.020 ms (5.2%) | 0.271 ms (0.7%) | 1.057 ms (2.7%) | 38.939 ms | 0% |
| pp512 | **215.886 ms (41.6%)** | 70.376 ms (13.6%) | **230.786 ms (44.4%)** | 1.114 ms (0.2%) | 1.089 ms (0.2%) | 519.250 ms | 0% |

At deep context, attention is the unambiguous loss: the parity-preserving
explicit F32 split tree costs about 1.05 ms in each capped sliding layer and
1.55 ms in each growing global layer at d16384. Better GQA reuse and a faster
order-stable reduction remain the largest decode opportunity. At pp512,
grouped MoE roughly halved the original native routed term, but attention plus
routed experts still consume 86.0% of the prompt. At d0 the head remains a
fixed 1.06 ms. These are qk's own kernels and an honest result; Stage 6 does
not claim to beat llama.cpp at any measured depth.

### 2026-07-19 — Stage 7 native cooperative-matrix attention

Stage 7 keeps the Stage 6 de-vendoring intact. The only new GPU artifacts are
qk GLSL sources compiled by qk's normal build. A forbidden-path scan remains
empty for `/mnt/data/llama.cpp-master/build`, `vulkan-shaders.spv`, and
`llama_*.spv`; both new modules pass `spirv-val --target-env vulkan1.2`. The
source comments credit llama.cpp `571d0d5`'s `flash_attn_cm1.comp` as an
algorithmic reference. No compiled SPIR-V or binary was copied, linked, loaded,
or executed from that project.

Before changing attention, five independent launches split the existing qk
kernel into profiling-only score, softmax, and output phases. Every launch
began at XTX `gpu_busy_percent=0`. The phase-specialized path is slightly faster
than the unchanged fused pipeline because specialization changes compiler
scheduling, so its phase sum is reported separately from the fused baseline:

| profile before changes | attention | sliding / global | prep | score | softmax | output | finalize |
|---|---:|---:|---:|---:|---:|---:|---:|
| tg d16384, fused | **34.520 ms** (34.452--34.536) | -- | -- | -- | -- | -- | -- |
| tg d16384, phase run | 33.261 ms (33.120--33.392) | 26.452 / 6.822 ms | 1.086 ms | **21.775 ms** | 0.213 ms | **9.615 ms** | 0.584 ms |
| pp512, fused | **218.060 ms** (217.635--218.377) | -- | -- | -- | -- | -- | -- |
| pp512, phase run | 221.273 ms (220.035--222.700) | 172.151 / 49.111 ms | **90.187 ms** | **60.521 ms** | 6.694 ms | 16.401 ms | **47.571 ms** |

At d16384, the direct phase run assigns 18.394/0.127/6.587 ms of
score/softmax/output to the 25 sliding layers and 3.381/0.086/3.030 ms to the
five global layers. The sliding total is larger because there are five times as
many sliding layers, even though each global layer is individually more
expensive and grows with depth. At pp512 the corresponding medians are
44.721/5.579/12.687 ms sliding and 15.798/1.111/3.715 ms global. This is the
requested measured attribution of the Stage 6 34.444-ms attention term, not a
cost-model estimate.

The first retained change exploits GQA directly in qk's scalar fallback. One K
or V cache load now serves all query heads owned by that KV head—eight heads in
a global layer and two in a sliding layer—while preserving each head's original
dimension and position accumulation order. The full six-fixture gate passed
immediately after this change with 1,024 exact generated tokens. In isolation,
d16384 improved from 25.567 to **29.253 tok/s** (29.195--29.259, busy 0%);
pp512 was effectively flat at 989.022 tok/s (916.948--989.437, busy 0%).

The XTX advertises `VK_KHR_cooperative_matrix` with 16x16x16 f16-input shapes
and both f16 and F32 accumulators. The qk-native decode flash kernel assigns one
workgroup to a KV head and split. A cooperative QK tile produces scores for all
GQA rows, an online F32 max/sum state avoids materializing the probability
matrix, and a cooperative PV tile updates the output numerator. The unusually
wide global dh=512 state is held in shared memory rather than registers:
16 KiB query, 0.5 KiB score, 0.5 KiB probability, 16 KiB live PV, and 16 KiB
F32 output state, about 49 KiB plus scalar state. Compile-time cooperative-pipe
specializations use dh=512/GQA=8 for global and dh=256/GQA=2 for sliding, so the
sliding allocation is about 19 KiB rather than inheriting the global budget.

Global decode is parity-safe and is now the default on a compatible device.
At d16384 its five-layer attention median falls from 6.822 to **1.869 ms**, a
72.6% reduction. Sliding cooperative decode is not parity-safe: with
`QK_G4_COOPMAT_SLIDING=1`, `swa_position_1024` diverges at continuation index
13 (absolute position 1037), expected token 1816, actual 506. There is no
waiver. That path remains explicitly opt-in; default sliding decode uses the
parity-safe F32 scalar path with GQA reuse.

Prefill uses a second native flash kernel with one workgroup per prompt token
and KV head. It applies the same eight-head global and two-head sliding K/V
reuse and specializes shared allocation to each shape. Both global and sliding
prefill are default on compatible devices. Ordinary and coding grouped-batch
spot checks reproduce their fixture continuations exactly; the SWA grouped
batch output is unchanged from Stage 6's scalar grouped-batch output. As in
Stage 6, the formal fixture claim below deliberately covers the serial accuracy
path selected by the gate, not every prompt grouping choice.

The final default gate began at 0% XTX busy and passed all six fixtures:
ordinary chat 32/32, coding 32/32, all three 1023/1024/1025 ring cases 16/16,
and global-context-8192 16/16. Repeating ordinary and coding from empty caches
raised the cumulative evidence to **1,024 exact generated tokens**. No waiver
was used. The post-run busy reading was 97%, recorded as an observation rather
than an idle claim.

Final default performance is below. Each campaign contains five repetitions
and began at `gpu_busy_percent=0`; spreads are min--max. llama.cpp values are
the accepted Stage 6 campaigns on the same XTX, model, f16 K/V setting, and
`571d0d5` revision.

| test | Stage 7 qk tok/s | vs Stage 6 | llama.cpp tok/s | qk / llama | result |
|---|---:|---:|---:|---:|---|
| pp512 | 1046.54 (968.33--1047.91), busy 0% | **+6.3%** | 3405.88, busy 0% | 0.307x | qk loses |
| tg128 d0 | 119.02 (118.20--119.21), busy 0% | **+12.9%** | 139.94, busy 0% | 0.850x | qk loses |
| tg128 d4096 | 73.73 (73.62--73.84), busy 0% | **+14.7%** | 125.73, busy 0% | 0.586x | qk loses |
| tg128 d16384 | 31.61 (31.60--31.65), busy 0% | **+23.7%** | 120.22, busy 0% | 0.263x | qk loses |

Five-launch final profiles, each launch beginning at busy 0%, put d16384
attention at **26.793 ms** (26.768--27.018): 24.938 ms sliding and 1.869 ms
global. This is a 22.4% reduction from the reprofiled 34.520-ms fused baseline,
but the retained scalar sliding layers now account for 93.1% of attention.
pp512 attention is **186.133 ms** (184.830--186.249): 148.852 ms sliding and
37.149 ms global, 14.6% below the reprofiled fused baseline. Prompt attention
prep/finalize still cost 89.871/44.268 ms, limiting the end-to-end pp gain.

For completeness, the rejected sliding decode mode was also measured over five
repetitions from busy 0%. It reaches **69.709 tok/s** at d16384
(69.660--69.752), 0.580x llama.cpp and 2.21x the parity-safe default. That is a
real speed result but not an accepted result: it fails the non-negotiable ring
fixture above and therefore remains behind `QK_G4_COOPMAT_SLIDING=1`.

Raw profile launches, benchmark samples, commands, busy readings, ratios,
parity disposition, and validation results are appended to
`bench/results-gemma4-qk.jsonl`. The tree is intentionally dirty and no commit
was created.

### 2026-07-19 — Stage 8 parity-safe sliding cooperative attention

Stage 8 began by tracing the failure that Stage 7 had attributed to the
1024-cell circular ring. For the prediction at absolute position 1037 (the
kernel evaluation is position 1036), temporary instrumentation copied every K
and V element actually addressed by KV head 0 in layer 0, plus the absolute and
physical row indices. Scalar and cooperative traces were byte-identical across
1024 rows and 256 dimensions for both K and V: absolute rows 13--1036 mapped to
physical slots 13--1023 followed by 0--12. Both 2,105,344-byte trace payloads
had SHA-256 `f47eca36f2fbc145692ff99c5c425e5783ea565d4455ef2dd80c62f26fa571cf`.
The Q/K/V inputs at that layer were also byte-identical. The suspected ring
mapping defect was therefore disproved rather than guessed around.

The actual defect was numerical amplification. Stage 7's f16 cooperative QK,
f16 probability, and f16 PV path changed layer-0 attention output by up to
0.031534 (relative L2 0.004238) despite identical cache reads. Small attention
differences then crossed routed-expert boundaries in the early layers. Merely
switching QK to F32 fixed `swa_position_1024` but still failed 1023/1025; F32
softmax/value variants moved which boundary token failed. Those rejected
variants were not shipped.

The retained implementation compiles the same qk-native shader source a second
time with `G4_COOPMAT_F32=1`. Its sliding specialization uses F32 cooperative
QK, stores the active split's scores in shared memory, reproduces the scalar
256-lane F32 max/sum tree, and processes PV tiles chronologically with an F32
cooperative accumulator. The first three sliding layers retain scalar decode
attention because layer-dump bisects showed that their score-rounding error is
the parity boundary; the other **22 of 25 sliding layers** now use cooperative
QK/PV by default. Global decode retains Stage 7's faster f16 specialization.
`QK_G4_NO_COOPMAT` remains the device/path opt-out; the failing
`QK_G4_COOPMAT_SLIDING=1` experiment is no longer required.

The final default gate began at XTX `gpu_busy_percent=0` and passed all six
fixtures with no waiver: ordinary chat 32/32, coding 32/32,
`swa_position_1023`, `_1024`, and `_1025` 16/16 each, and
`global_context_8192` 16/16. Repeats from empty caches raised the cumulative
evidence to **1,024 exact generated tokens**. The immediate post-gate busy
reading was 96%.

Final default performance follows. Every campaign contains five repetitions,
began at `gpu_busy_percent=0`, and records the immediate post-campaign busy
reading; spreads are min--max. llama.cpp values remain the accepted same-XTX
`571d0d5` measurements from Stage 6.

| test | Stage 8 qk tok/s | vs Stage 7 default | llama.cpp tok/s | qk / llama | post busy |
|---|---:|---:|---:|---:|---:|
| pp512 | 1042.54 (961.87--1045.09) | -0.4% | 3405.88 | 0.306x | 71% |
| tg128 d0 | 137.57 (137.13--137.77) | **+15.6%** | 139.94 | **0.983x** | 77% |
| tg128 d4096 | 108.36 (108.32--108.49) | **+47.0%** | 125.73 | **0.862x** | 94% |
| tg128 d16384 | 64.08 (64.04--64.15) | **+102.7% / 2.03x** | 120.22 | **0.533x** | 99% |

The parity-safe result is 8.1% below Stage 7's rejected 69.709 tok/s path, but
it more than doubles the accepted d16384 default without bending the token
gate. Relative to the Stage 7 default, the five-launch d16384 attention median
falls from 26.793 to **10.854 ms** (-59.5%): sliding falls from 24.938 to
**9.032 ms** (-63.8%) and global remains effectively flat at 1.818 ms. The
five-launch spreads are 10.818--10.968, 9.000--9.145, and 1.805--1.823 ms;
every launch began at busy 0%.

The next bottleneck was remeasured rather than inferred. At d16384, attention
is still **10.854 ms / 71.1%** of the 15.267-ms accounted median; routed
experts are 2.015 ms, shared experts 1.076 ms, and the head 1.057 ms. At d0,
attention and routed experts are now nearly tied at **2.416 / 2.258 ms**
(33.3% / 31.1%), followed by shared experts at 1.231 ms and the head at
1.067 ms. Further decode work should first remove the three scalar sliding
layers without changing their score arithmetic, then address routed experts;
the head is no longer the sole d0 target.

Both generated modules pass `spirv-val --target-env vulkan1.2`. The forbidden
binary/path scan is empty. No compiled SPIR-V or binary was copied, linked,
loaded, or executed from another project, the temporary trace instrumentation
was removed, the tree is intentionally dirty, and no commit was created.

### 2026-07-19 — Stage 9: first own-kernel win, deep-context attribution, and prefill attack

Stage 9 gets over the line at d0 without weakening the parity contract.
**This is the project's first genuine Gemma 4 win on its own kernels:** the
final default reaches **144.78 tok/s** at tg128 d0 versus llama.cpp's
**139.94 tok/s**, a parity-safe **1.035x / +3.46%** result on the same XTX.

The winning change fuses the post-attention chain that previously dispatched
weighted RMS norm, residual add, shared-branch RMS norm, and routed-branch RMS
norm separately. One qk-native shader preserves the post-norm and residual
intermediates, performs the second RMS reduction once, and applies the two
independent branch weights. The control flag is `QK_G4_NO_NORM_FUSION`.
Measured in the assembled d0 graph, the fused path was **144.107 tok/s**
(143.548--144.167) versus **137.861 tok/s** (136.864--138.016) unfused,
**+4.53%** end to end. Both five-repetition campaigns began at busy 0%; their
immediate post-campaign readings were 76% and 77% respectively.

The cooperative F32 max/sum tree also now omits leading levels whose upper
half contains only the neutral values `NEG_MAX` or zero. For the short sliding
splits this removes exact no-op reductions and barriers while leaving the
surviving arithmetic tree unchanged. The optimization passed the complete
fixture gate and contributes at every depth.

The tempting early-layer cooperative path was tested and rejected for the
same reason Stage 8 warned about. On the final fused build, F32 cooperative
attention on layers 0--2 reaches **148.566 tok/s** (147.411--148.763, five
reps, busy 0%--69%) versus the accepted default's 144.782 tok/s, but
`swa_position_1023` fails 13--14 generated tokens after the perturbation.
Testing layers 1--2 and layer 2 alone also failed that fixture. The all-three
experiment remains available only behind
`QK_G4_COOPMAT_EARLY`; it is off by default and explicitly not an accepted
result. A tied-head Q6_K blocks-per-row specialization also produced no
assembled gain (**137.764 tok/s**, 136.358--137.968, busy 0%--76%) and was
reverted.

Deep-context attention was split into measured cooperative phases rather than
inferred from the fused timestamp. The profiler builds qk-native cumulative
QK, QK+softmax/reduction, and full QK+softmax/reduction+PV variants and reports
their deltas. Across five d16384 launches, all beginning at busy 0% and ending
at 95%, the medians are:

| layer class | QK | softmax/reduction | PV | cooperative core |
|---|---:|---:|---:|---:|
| 25 sliding layers | 2.271 ms | **3.205 ms** | 1.927 ms | **7.403 ms** |
| 5 global layers | 0.022 ms | **0.885 ms** | 0.515 ms | **1.422 ms** |

The cumulative instrumentation dispatches three variants and intentionally
inflates its own total wall time; the component deltas above are the useful
result. Sliding attention still dominates, but the largest measured piece is
now its F32 softmax/reduction, not QK or PV. The neutral-level elision improves
the final d16384 benchmark by 2.23% over Stage 8, but d16384 remains only
0.545x llama.cpp, so reduction/softmax is the next concrete deep-context
target.

Prefill was re-profiled after the attention and norm changes. The five-launch
accounted median is 486.623 ms: routed experts are **230.279 ms / 47.3%**,
attention **184.548 ms / 37.9%**, shared experts **69.593 ms / 14.3%**, and
residual/head are 1.115/1.089 ms. Every launch began at busy 0%; immediate
post-launch readings were 89--90%. Routed experts remain the largest term, but
they no longer account for the old 86% share.

`docs/amd-opt/REPORT.md` and the source diff of Qwen branch `prefill-opt`
(`cf899c1`) were reviewed before changing the Gemma schedule. Gemma already
counting-sorts top-8 assignments into expert-major order and runs tiled Q4_0
grouped GEMMs, so Qwen's adjacent-row grouped GEMV is not a direct transfer to
Gemma's 128-expert top-8 layout. No binary or compiled shader was copied,
linked, loaded, or executed from that work.

Two Gemma tile changes were measured and rejected:

- BN64 fell to **890.331 tok/s** (818.279--891.652, busy 0%--72%), consistent
  with added LDS/register pressure.
- A corrected BM32 variant first measured 1062.335 tok/s, but its repeat was
  **1054.864 tok/s** versus **1054.397 tok/s** for BM64, only +0.04% and inside
  noise. BM32 exactly matched BM64 and all 64 frozen grouped ordinary/coding
  tokens, but it was still reverted because it did not establish an
  end-to-end win. An earlier 1778.85 tok/s BM32 number was discarded: the
  loader still had a BM64 shift and skipped the second K block, so it was not
  a valid performance result.

The exact final BM64 tree then passed all six fixtures with no waiver:
ordinary chat 32/32, coding 32/32, the three SWA boundary fixtures 16/16 each,
and global-context 16/16. Empty-cache repeats bring the cumulative gate to
**1,024 token-exact generated tokens**. The gate began at XTX busy 0% and the
immediate post-gate reading was 95%.

Final default performance follows. Every campaign has five repetitions,
began at `gpu_busy_percent=0`, and records the immediate post-campaign reading;
spreads are min--max. llama.cpp values are the accepted same-XTX `571d0d5`
measurements from Stage 6.

| test | Stage 9 qk tok/s | vs Stage 8 | llama.cpp tok/s | qk / llama | post busy |
|---|---:|---:|---:|---:|---:|
| pp512 | 1051.85 (973.59--1052.14) | +0.9% | 3405.88 | 0.309x | 66% |
| tg128 d0 | **144.78 (144.01--145.05)** | **+5.2%** | 139.94 | **1.035x, qk wins** | 72% |
| tg128 d4096 | 112.72 (112.60--112.83) | **+4.0%** | 125.73 | 0.896x | 93% |
| tg128 d16384 | 65.51 (65.45--65.66) | **+2.2%** | 120.22 | 0.545x | 98% |

All seven affected/generated modules pass `spirv-val` for Vulkan 1.2. The
changed-source forbidden binary/path scan is empty and `git diff --check`
passes. Raw accepted and rejected measurements, phase samples, busy readings,
commands, parity disposition, and validation results are appended to
`bench/results-gemma4-qk.jsonl`. The tree is intentionally dirty and no commit
was created.

### 2026-07-19 — Stage 10: deeper decode and prefill, without a second unsafe win

Stage 10 improves every requested workload while retaining the Stage 9 d0 win
and the complete parity gate. It does not manufacture a claim that the data do
not support: qk still trails llama.cpp at pp512, d4096, and d16384.

The first target was the apparent deep-context softmax/reduction inversion.
Direct phase measurements at d4096 put the Stage 9-style cooperative path near
637 us QK, 924 us softmax/reduction, and 615 us PV for sliding attention. The
following exact-order or resource-only formulations were tested and rejected
end to end:

- a parallel rank mapping for the scalar F32 tree: 112.792 tok/s
  (112.670--112.869, busy 0%--94%);
- subgroup shuffles with the same reduction tree: 112.832 tok/s
  (112.648--112.898, busy 0%--94%);
- specialization-sized reduction scratch with the unused probability tile
  removed: 112.911 tok/s (112.784--113.029, busy 0%--94%);
- compact cooperative PV scratch: 111.828 tok/s (111.559--111.947, busy
  0%--94%); and
- a split-state shuffle reduction: 112.721 tok/s (112.625--112.754, busy
  0%--93%).

A distributed exponential pass retained the block max/sum ordering but failed
the formal boundary fixture at continuation 14 (expected token 1638, actual
13315), so it was removed. No waiver was used. The one retained score-tree
change replaces LDS exchange with same-tree subgroup shuffles in the actual
F16 split shader. A paired five-repetition d4096 campaign measured 113.111
tok/s (112.959--113.146) versus 112.785 (112.732--112.862) for the restored
control, both busy 0%--94%: a small but repeatable +0.29% assembled win.

More importantly, decode now has a separate source-built half-query path. The
batch-preparation shader writes the padded F16 query only for cooperative
decode layers, and the cooperative shader loads it directly instead of
converting the same query into an LDS tile in every workgroup. Separate SPIR-V
modules leave the original fallback untouched; `QK_G4_NO_COOPMAT_QUERY_HALF`
is the opt-out. This is now the parity-safe default. Final phase medians show
why it helps rather than attributing the gain to softmax:

| depth/path | QK us | softmax/reduction us | PV us |
|---|---:|---:|---:|
| d4096 sliding | 639.56 | 927.00 | 521.12 |
| d4096 global | 19.60 | 336.12 | 145.00 |
| d16384 sliding | 2249.28 | 3193.20 | 1648.92 |
| d16384 global | 22.16 | 886.08 | 389.20 |

Against the Stage 9 d16384 profile, sliding PV drops 14.4% and global PV drops
24.4%, while sliding softmax changes by only -0.4%. Thus the unusual softmax
share is real for this exact-order implementation; the safe reduction rewrites
tried here could not turn it into an end-to-end win.

For prefill, the Qwen counting-sort and adjacent-row work was reviewed before
new shader work. The counting-sort idea already transfers: Gemma already sorts
its top-8 assignments into expert-major order. The adjacent-row Qwen GEMV
shader itself does not transfer directly to Gemma's fused
`ffn_gate_up_exps (2816,1408,128)` and 128-expert layout. Its useful scheduling
principle does transfer, and Gemma already applies it inside the grouped
BM64xBN32 Q4_0 kernels, with each lane producing RM4xRN2 outputs.

Two parity-preserving prefill changes were retained:

- sliding prefill attention now packs eight adjacent query tokens and both GQA
  heads into all 16 cooperative-matrix rows. `QK_G4_PREFILL_QUERY_TOKENS`
  keeps 1/2/4/8 available for measurement; eight is the default. Five-rep tile
  medians were 1095.117 tok/s for four, 1076.419 for two, and 1043.122 for one;
  four tied eight, so the full-row schedule remains selected.
- the grouped gate/up and down kernels reduce the K tile from two Q4_0 blocks
  (64 values) to one block (32 values), preserving Q4 block accumulation order
  while roughly halving their LDS footprint. With only gate/up at K32, pp512
  reached 1184.274 tok/s (1107.447--1188.173, busy 0%--70%); with only down at
  K32 it reached 1111.569 (1036.468--1112.834, busy 0%--71%). Both together
  reached 1202.949 (1127.115--1203.926, busy 0%--70%). BM32 plus K32 regressed
  to 1147.871 (1077.633--1149.416, busy 0%--70%) and was reverted.

The final prefill phase medians are 183.359 ms routed experts, 168.977 ms
attention, 71.019 ms shared experts, 1.122 ms residual/norm, and 1.100 ms head
(425.584 ms accounted). Routed experts fall 20.4% and attention 8.4% from
Stage 9. Their new shares are 43.1%, 39.7%, and 16.7% for routed, attention,
and shared respectively.

The default path passes all six fixtures without a waiver: ordinary chat,
coding, the three SWA boundary fixtures, and global context, for **1,024
token-exact generated tokens**. The gate began at XTX busy 0% and ended at
96%. A separate batched coding check was also exact for 32/32 tokens.

Final default performance follows. Each campaign has five repetitions, starts
at `gpu_busy_percent=0`, and reports min--max spread and the immediate post-run
busy reading. llama.cpp remains revision `571d0d5` on the same XTX.

| test | Stage 10 qk tok/s | vs Stage 9 | llama.cpp tok/s | qk / llama | post busy |
|---|---:|---:|---:|---:|---:|
| pp512 | 1195.750 (1117.464--1197.440) | **+13.7%** | 3405.88 | 0.351x | 70% |
| tg128 d0 | **146.191 (142.437--146.271)** | **+1.0%** | 139.943 | **1.045x, qk wins** | 76% |
| tg128 d4096 | 115.314 (115.107--115.368) | **+2.3%** | 125.734 | 0.917x | 94% |
| tg128 d16384 | 68.035 (68.027--68.095) | **+3.9%** | 120.224 | 0.566x | 99% |

The remaining gaps are 8.3% at d4096, 43.4% at d16384, and 64.9% at pp512.
All retained changes win end to end; isolation-only or parity-failing changes
remain rejected. Fourteen generated modules pass `spirv-val --target-env
vulkan1.2`, the forbidden binary/path scan is empty, `git diff --check` passes,
and no compiled artifact from another project was copied, linked, loaded, or
executed. The source tree is intentionally dirty and no commit was created.

### 2026-07-19 — Stage 11: medium-context decode win

Stage 11 reaches the remaining realistic target without weakening parity. The
final default measures **129.889 tok/s** at tg128 d4096 versus llama.cpp
`571d0d5` at **125.734 tok/s**: **1.033x / +3.30%**, and +12.64% over the
Stage 10 default. Together with d0 at 146.962 versus 139.943, qk now wins
single-token decode at both short and medium context on the XTX.

The stage began with a fresh control rather than relying on the Stage 10
number. Five repetitions measured 115.394 tok/s (115.147--115.512, busy
0%--93%). A five-launch production-path profile measured 8.309 ms median GPU
time and attributed 4.389 ms to attention, 1.827 ms to routed experts, 1.064
ms to the head, 0.902 ms to shared experts, and 0.263 ms to residual/norm
(busy 0%--78%). Attention itself was 3.535 ms sliding and 0.855 ms global;
1.082 ms was preparation, 2.660 ms the fused attention/reduce core, and 0.663
ms finalization.

The apparent sliding-depth growth was not an over-read of the ring. Every
shader already bounds its useful positions to the final 1024 entries. The host
did, however, derive split width from the full padded context. At d4096 that
left only about six useful splits per KV head, so the three parity-sensitive
scalar sliding layers serialized long score/value loops and the cooperative
layers also had less useful parallelism than at d1024.

Two related launch changes are retained:

- At medium context (`kvLength` above 2048 and no more than 8192), every
  sliding layer uses 32-token absolute-aligned splits. Short-context geometry
  remains unchanged, and the upper bound guarantees that all partial states
  fit the existing 256-slot allocation. Only the contiguous split interval
  intersecting the 1024-token ring is dispatched. The reducer keeps the full
  absolute split numbering and the same neutral values for undispatched
  splits, so empty-state elision itself does not change the reduction tree.
- Q/K/V and attention-output projections use the Stage-1-measured TPR128
  geometry (two rows per workgroup) only from position 2048 onward and only
  from layer 3 onward. Short context and layers 0--2 keep TPR64/four rows per
  workgroup. Separate pipelines use the same qk-owned GLSL modules; no external
  binary is involved.

The scope guards are correctness requirements, not benchmark conveniences. A
32-token split at all depths reached 120.779 tok/s (120.364--120.855, busy
0%--93%) but failed `swa_position_1023` at continuation 14 (expected 1638,
actual 13315; gate busy 0%--80%). Restoring the established split geometry
through 2048 tokens passed the full gate (busy 0%--95%). Likewise, applying
TPR128 at short context failed the same boundary/token; restricting it to
positions at or beyond 2048 passed all fixtures, including the 8192-context
case (busy 0%--96%). No waiver was used.

The progression at d4096 was repeatable. Fixed 64-token sliding splits reached
120.005 tok/s (119.904--120.157, busy 0%--93%). Scoped 32-token splits in the
22 cooperative layers reached 120.779 (120.364--120.855, busy 0%--93%). The
scoped TPR changes then reached 122.149 (121.884--122.240, busy 0%--93%).
Extending 32-token splits to the three scalar early layers supplied the large
remaining gain: 129.981 (129.576--130.012, busy 0%--93%). After removing a
marginal argmax experiment and restoring the original softcap path, the final
campaign measured 129.889 (129.613--130.115, busy 0%--93%).

Final production and cumulative phase profiles show the mechanism. Each row is
the median of five launches; the final campaigns began at busy 0% and ended at
77% (production) or 80% (cumulative phases).

| d4096 profile | fresh control | final | change |
|---|---:|---:|---:|
| production attention | 4.389 ms | 3.421 ms | -22.1% |
| production sliding attention | 3.535 ms | 2.558 ms | -27.6% |
| production global attention | 0.855 ms | 0.854 ms | flat |
| production fused attention/reduce core | 2.660 ms | 1.718 ms | -35.4% |
| sliding QK phase | 626.24 us | 152.72 us | -75.6% |
| sliding softmax/reduction phase | 923.60 us | 492.28 us | -46.7% |
| sliding PV phase | 501.28 us | 499.44 us | -0.4% |

Thus Stage 10's PV work remains valuable, but PV is not where Stage 11's gain
comes from. Finer useful splits restore CU occupancy for QK and the ordered
reduction; PV was already essentially flat. Global phases also remain flat,
confirming that each global K/V read already serves all eight query heads.

Several ideas were measured and rejected end to end:

- dispatching only non-empty splits with the old coarse geometry was just
  115.545 tok/s (115.460--115.655, busy 0%--93%), +0.13% with overlapping
  spreads; neutral elision is retained only as an enabling part of the finer
  split schedule;
- same-tree subgroup shuffles in the general Q4xQ8 GEMV regressed to 115.240
  (115.166--115.369, busy 0%--93%), and in routed gate/up regressed to 120.152
  (119.954--120.183, busy 0%--93%); both were reverted;
- F16 cooperative PV accumulation in layers 3+ was flat at 115.347
  (115.197--115.782, busy 0%--93%) and was reverted without spending a parity
  campaign;
- 32-token global splits added only +0.08% in the assembled candidate, and the
  Stage-1 shared gate/down TPR256+16 pair regressed to 120.523
  (120.478--120.752, busy 0%--92%); both were reverted;
- direct-logit argmax reached 121.215 (120.622--121.640, busy 0%--93%) in
  isolation but had overlapping spread and was unnecessary for the final win,
  so the original softcap comparison was restored.

Final default performance follows. Every campaign has five repetitions,
began at `gpu_busy_percent=0`, and reports the immediate post-run reading.

| test | Stage 11 qk tok/s | vs Stage 10 | llama.cpp tok/s | qk / llama | post busy |
|---|---:|---:|---:|---:|---:|
| pp512 | 1192.890 (1109.630--1194.142) | -0.2% | 3405.88 | 0.350x | 68% |
| tg128 d0 | **146.962 (145.525--147.068)** | +0.5% | 139.943 | **1.050x, qk wins** | 74% |
| tg128 d4096 | **129.889 (129.613--130.115)** | **+12.6%** | 125.734 | **1.033x, qk wins** | 93% |
| tg128 d16384 | 68.479 (68.128--68.652) | +0.7% | 120.224 | 0.570x | 98% |

The exact final binary passes ordinary chat, coding, all three SWA boundary
fixtures, and global context for **1,024 token-exact generated tokens** (busy
0%--95%). Fourteen relevant generated modules pass `spirv-val --target-env
vulkan1.2`; the forbidden binary/path scan is empty; `git diff --check`
passes. No compiled SPIR-V or binary from another project was copied, linked,
loaded, or executed. The tree is intentionally dirty, no commit was created,
and the d4096 stopping target is honestly met rather than relaxed.
