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
