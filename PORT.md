# PORT.md — Metal port log (Apple M4 Max)

Port of the qk engine (Vulkan/RDNA3) to Metal on a MacBook Pro M4 Max
(40-core GPU, 64 GB unified, 546 GB/s theoretical). Model:
`Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (15.45 GiB, unsloth UD-Q3_K_M). Charter:
`../GOAL.md`. Branch: `metal-port`.

## M0 — llama.cpp Metal baseline (2026-07-08)

**Gate decision: NOT triggered — proceed with the port.** llama.cpp Metal
decodes at 84 tok/s = 38% of the theoretical-bandwidth ceiling (42% of
measured achievable bandwidth). The 70% gate would have required ≥141–156
tok/s. Headroom for the port is ~2.3× on the weights-only ceiling.

### llama-bench

llama.cpp master `f2d1c2f` (2026-07-08), Metal backend, Release, static,
`GGML_METAL_EMBED_LIBRARY=ON`, defaults (`-fa auto`, mmap on, all layers on
GPU), `-r 5`, plugged in, `caffeinate -i`. Two full runs back-to-back; run 2
is thermally soaked.

| test | run 1 (t/s) | run 2, soaked (t/s) |
|---|---|---|
| pp512 | 1452.08 ± 11.54 | 1452.60 ± 8.02 |
| pp2048 | 1424.97 ± 12.43 | 1325.23 ± 53.15 |
| tg128 | 84.00 ± 0.30 | **84.18 ± 0.10** |

Cross-check: `llama-cli` single-turn greedy (`--temp 0`, chat template,
thinking mode on by default) reports 84.0 t/s generation, 153.9 t/s on a
short prompt; output coherent. Decode is thermally flat (84.00 → 84.18);
prefill droops ~7% at pp2048 when soaked (compute-bound; decode at ~207 GB/s
doesn't stress the package).

### Measured GPU bandwidth

Metal microbench (`bench/metal_membw.swift`): 4 GiB private buffer, coalesced
float4 grid-stride read-sum, 40×32×4 threadgroups × 256, timed over 8
dispatches, 3 trials: **493.1–495.6 GB/s** = 90.5% of the 546 GB/s
theoretical. `recommendedMaxWorkingSetSize` = 51 GiB (matters for the
Qwen3-Next-80B follow-on: ~45 GB Q4 fits, tight).

### Active set per decoded token (exact, from GGUF tensor metadata)

`bench/active_bytes.py` parses the GGUF header and sums tensor bytes by
class (routed experts scaled by top-8/256):

| class | tensors | stored GB | read GB/token |
|---|---|---|---|
| dense — attn/DeltaNet/norms/router/output head | 452 | 1.881 | 1.881 |
| routed experts (top-8 of 256) | 120 | 14.158 | 0.442 |
| shared expert (always on) | 160 | 0.134 | 0.134 |
| token_embd (row lookup) | 1 | 0.417 | ~0 |
| **total weights** | | **16.590** | **2.457** |

Non-weight traffic per token at tg128: DeltaNet state read+write ≈ 0.126 GB
(30 linear layers × 32 heads × 128×128 f32 × 2), KV cache ≈ 0.003 GB
(10 full-attn layers, 2 KV heads × 256, f16, short ctx). Negligible KV at
short context; grows to ~0.04 GB/token at ctx 2048.

### Ceilings and extraction

| basis | ceiling (tok/s) | llama.cpp 84.2 = |
|---|---|---|
| 546 GB/s theoretical, weights only (charter method) | 222 | 37.9% |
| 494 GB/s measured, weights only | 201 | 41.9% |
| 494 GB/s measured, weights + DeltaNet state | 191 | 44.0% |

llama.cpp's effective decode bandwidth: 84.2 × 2.457 GB = **207 GB/s** of
494 measured. (For calibration: the charter's 220–270 band assumed 2.0–2.5
GB/token; the real number is 2.457, i.e. the bottom of the band, 222 tok/s.)

Targets in absolute terms: DoD ≥1.3× llama.cpp = **≥109.5 tok/s** (= 54% of
the measured-bandwidth ceiling). RDNA3-parity extraction (this engine gets
~76% of achievable there) would be **~153 tok/s ≈ 1.8×** llama.cpp. Stretch
150 tok/s = 75% of measured ceiling — same territory as the RDNA3 result.

### Architecture facts pinned down for the port (from GGUF metadata)

- arch `qwen35moe`: 40 blocks, hidden 2048, `full_attention_interval = 4` →
  10 full-attention layers (blk 3,7,…,39), 30 gated-DeltaNet layers.
- Full attn (gated): `attn_q` 2048→8192 (16 heads × 256 **× 2 — Q fused with
  output gate**), `attn_k`/`attn_v` 2048→512 (2 KV heads, GQA 8:1), per-head
  q/k RMS norm, partial RoPE dims [11,11,10,0] of 64, freq base 1e7.
- DeltaNet: fused `attn_qkv` 2048→8192 (q 16×128 grouped, k 16×128, v
  32×128), `attn_gate` 2048→4096, `ssm_out` 4096→2048, conv1d k=4 over 8192
  channels, `ssm_alpha`/`ssm_beta` 2048→32 (per-head decay/gate, 32 heads),
  state 32×128×128 f32, `ssm_norm` RMS over 128.
- MoE (every layer): router `ffn_gate_inp` 2048→256 in F32, 256 experts
  top-8, expert FFN 512, shared expert FFN 512.
- Output head `output.weight` Q6_K 2048×248320 = **417 MB — 17% of every
  token's read budget**; vocab 248,320. Prime fusion/quant target.
- Quant mix: IQ3_XXS 80 t (8.22 GB, expert bulk), IQ4_XS 37 t (5.28 GB),
  Q6_K 5 t (1.49 GB), Q8_0 250 t (1.49 GB, dense projections), F32 361 t
  (0.10 GB, norms/router/ssm small tensors). GEMV coverage needed: Q8_0,
  Q6_K, IQ4_XS, IQ3_XXS (+ F32 helpers) — same set as the Vulkan engine.
- ggml Metal on this device: simdgroup reduction + matrix available, bfloat
  available, "tensor API" (M5+) not available, unified memory + residency
  sets in use.

## M1 — Metal harness + f16 GEMV (2026-07-08)

**Delivered:** Metal host skeleton + the `qk` CLI on macOS + `gemv_f16`
validated against the CPU reference and measured at **94–95% of theoretical
DRAM bandwidth** on big tensors (515 GB/s on the 971 MiB output-head shape;
the streaming-microbench ceiling from M0 was 494, so the GEMV beats the
naive streamer).

New files: `src/main_metal.mm` (ObjC++ TU mirroring `src/main.cpp`
one-to-one: same CLI, same scale-aware validation, same output format),
`shaders/metal/gemv_f16.metal`. CMake grows an `if(APPLE)` branch; the
Vulkan/Linux path is untouched. MSL is JIT-compiled at runtime from
`shaders/metal/` (mirrors the `loadSpv` flow, `QK_SHADER_DIR` wins; no
offline metal toolchain needed — relevant on macOS 26 where `xcrun metal`
is a separate download). Buffers are `storageModeShared`: upload/readback
are plain memcpy on UMA, no staging path exists. Spec constant → function
constant (TPR); push constants → `setBytes`; `subgroupAdd` → `simd_sum` /
`simd_shuffle_down`; the Vulkan bench's barrier-serialized dispatch loop →
one encoder whose Y write-after-write hazard serializes dispatches;
timing via `MTLCommandBuffer` GPUStart/EndTime.

f16 GEMV, correctness PASS on every shape (max_rel_err ≤ 2.5e-4, tol 1e-2):

| shape (M×K) | W | tpr | GB/s | note |
|---|---|---|---|---|
| 16384×8192 | 256 MiB | 256 | 510.5 | suite default |
| 8192×8192 | 128 MiB | 256 | 518–546 | run-to-run spread, partial SLC |
| 248320×2048 | 971 MiB | 64 | **515.5** | output-head shape, cleanest DRAM number |
| 8192×2048 | 32 MiB | 64 | 611 | **fits SLC — inflated, see below** |
| 8192×512 | 8 MiB | 32 | 298 | launch-bound (~6 µs/dispatch floor) |
| 4096×128 | 1 MiB | 16 | 70 | latency-bound; simd_shuffle path exercised |

Findings, in port-note form:

- **The RDNA3 TPR heuristic left 40% of bandwidth on the table** for
  M-huge/K-small shapes: at tpr 256 the head GEMV gives each thread a
  single 16 B load (315 GB/s); tpr 32–128 → 517–523 GB/s. New rule: after
  the Vulkan skinny-row shrink, keep halving TPR while per-thread work
  < 4 units and ≥1024 threadgroups remain. `QK_TPR=<n>` overrides for
  crossover experiments (M6).
- **M4 Max SLC (~48 MB) inflates small-W benches:** the 32 MiB shape
  "streams" at 611–754 GB/s because iterations re-hit cache. Real decode
  streams 2.5 GB/token through the SLC, so per-layer GEMVs in a token step
  run at DRAM speed — bench conclusions must come from big or rotating
  weights (M6 must re-check crossovers against real steady-state). Flip
  side: activations/state are small and can live in SLC essentially free —
  fused blocks (M3+) should lean on that.
- Per-dispatch overhead floor is ~6 µs at small sizes — visible on the
  8 MiB expert-shaped GEMV (298 GB/s). The full-layer fusion strategy
  (pre-recorded command buffers, few submits) matters on Metal exactly as
  it did on Vulkan.
- `qk list`/`qk gguf` are wired; this model has no F16 tensors, so
  real-tensor GEMV validation starts when the quant kernels land (M2).

(Resolved blocker: HTTPS pushes failed for the whole session — stale gh
token + keychain locked to non-interactive shells. SSH auth worked;
remote switched to git@github.com and the branch is pushed.)

## M2 — quant GEMVs: q8_0, q6_k, iq4_xs, iq3_xxs (2026-07-08)

**Delivered:** all four quant GEMV kernels ported to MSL, PASS against the
CPU dequant references on synthetic random blocks AND on real tensors
mmap'd from the GGUF. `qk suite` runs the full five-kernel matrix; `qk gguf
<tensor>` works for every weight type in this model.

Kernels are hand-written MSL from the GLSL (not SPIRV-Cross): same block
structs (static_asserted to ggml sizes), same work decomposition, subgroup
ops → simd ops. `gemv_q8_0`/`gemv_q6_k` keep the grid-z query batching the
engine uses for multi-slot decode. `#include "iq_tables.metal"` is resolved
by a tiny include inliner in `loadMetalSource` (runtime MSL has no include
paths). One MSL gotcha vs GLSL: thread-position attributes must be all
scalar or all vector — mixing `uint tid` with `uint3 tgpig` is a compile
error.

Correctness (tol 1e-2, scale-aware rel err; all PASS):

| kernel | synthetic 8192×8192 | real tensor | max_rel_err (synth / real) |
|---|---|---|---|
| q8_0 | PASS | blk.0.attn_qkv 2048→8192 (17 MB) | 1.9e-4 / 1.5e-4 |
| q6_k | PASS | output.weight 2048→248320 (398 MB) | 6.0e-5 / 2.0e-4 |
| iq4_xs | PASS | blk.0.ffn_down_exps[:,:,0] | 1.3e-4 / 2.3e-5 |
| iq3_xxs | PASS | blk.0.ffn_gate_exps[:,:,0] | 1.6e-4 / 9.8e-6 |

Bandwidth after the face-off rework (v2 kernels, see next section):

| kernel | synth 8192×8192 | real tensor | note |
|---|---|---|---|
| f16 | 520 | 515 (head shape, M1) | 94–95% of 546 |
| q8_0 | 511 (522 @ 32768×8192) | 708 (attn_qkv, SLC) | 96% of peak, done |
| q6_k | 499 | **499 output.weight (398 MB)** | head now 0.84 ms/token |
| iq4_xs | 407 | tiny slice, latency-bound | +34% vs v1 |
| iq3_xxs | 353 | tiny slice, latency-bound | +27% vs v1, beats llama.cpp |

First iq3_xxs step (before the face-off): 32-element work unit instead of
the GLSL's 8 (aux u32 read once instead of 4×, packed_uchar4 grid loads)
— 201 → 288 GB/s. The v2 rework below took it to 353.

**Decode projection at v2 kernel speeds** (per-token active set from M0):
Q8_0 1.49 GB @ 522 = 2.85 ms; head Q6_K 0.417 @ 499 = 0.84 ms; expert IQ3
0.257 @ 353 = 0.73 ms; expert IQ4 0.165 @ 407 = 0.41 ms; Q6_K expert-downs
0.021 = 0.05 ms; F32 0.10 + DeltaNet state 0.126 @ ~520 = 0.43 ms →
**≈ 5.31 ms ⇒ 188 tok/s GEMV-streaming bound**. RDNA3 realized ~75% of its
equivalent bound end-to-end; 75% here ⇒ ~141 tok/s ≈ 1.68× llama.cpp. The
1.3× DoD (109.5 tok/s) has real margin; the 150 stretch needs ~80%
realization — fusion quality (M3/M6) decides it.

### M2b — kernel face-off vs llama.cpp Metal, and the rework it forced

Ryan asked for a llama.cpp-vs-qk comparison on this box. Method: llama.cpp
`test-backend-ops perf -o MUL_MAT` (build `f2d1c2f`, its own Metal kernels,
n=1 decode GEMV, m=4096 k=14336) vs `qk <fmt> 4096 14336` — identical
shapes, identical bytes, both loop-hot. µs/op is the comparator.

Round 1 (my straight GLSL ports): f16 and q8_0 at parity or ahead; q6_K
**26% behind** (136.4 vs 101.3 µs), iq4_xs **22% behind** (97.8 vs 75.9),
iq3_xxs 7% behind. So I read their kernels (MIT) and adopted the work
shape for those three: one simdgroup per NR0 consecutive rows (2/2/4),
NSG=2 simdgroups per 64-thread threadgroup, x staged in registers and
reused across rows, q6_K quadrant dequant via constant masks (no variable
shifts), iq4_xs qs as uint32 + 0x0f0f0f0f nibble masks + byte-aliased
codebook lookups from threadgroup memory, iq3_xxs grid+signs staged in
threadgroup memory; sign application kept vectorized (float4 select) —
that last bit is mine and is why iq3_xxs ends up *ahead* of llama.cpp.

Final standings, same shape, this box (µs/op, lower wins):

| format | llama.cpp | qk v2 | qk vs llama.cpp |
|---|---|---|---|
| f16 | 230.5 | 229.3 | +0.5% |
| q8_0 | 125.0 | 121.5 | **+3%** |
| q6_K | 101.3 | 106.3 | −4.7% |
| iq4_xs | 75.9 | 80.4 | −5.6% |
| iq3_xxs | 78.4 | 68.2 | **+15%** |

NSG sweep confirmed llama.cpp's nsg=2 beats 4 and 8 on all three reworked
kernels (e.g. iq3_xxs: 68.2 / 75.0 µs at nsg 2/4). Weighted by this
model's actual per-token byte mix (Q8_0-heavy dense + IQ3 experts), the qk
set is now net faster than llama.cpp's kernels for this workload; the
remaining ~5% q6_K and iq4_xs gaps are unroll/codegen detail — M6. The small-M
regression from fixed row-pair geometry (expert slices, M=512: 64
threadgroups) doesn't matter for decode — the engine's MoE path batches
8 experts × slots on the z-axis (M3).

Quant type census for the port (which kernel serves what, per token):
Q8_0 = all dense attn/DeltaNet projections + shared experts (1.49 GB);
Q6_K = output head + token_embd (row lookup) + ffn_down_exps in blk
34/38/39 only (unsloth dynamic bump); IQ3_XXS = all routed gate/up experts
(80 tensors, 8.2 GB stored); IQ4_XS = routed down experts in the other 37
layers. F32 = norms, routers, ssm small tensors.

## M3 — fused blocks: MoE step, DeltaNet block, attention block (2026-07-09)

**Delivered:** the three fused decode structures, each validated against the
CPU reference and timed. `qk moe`, `qk block`, `qk ablock` all run on Metal.

| structure | correctness | µs (this GPU) | note |
|---|---|---|---|
| MoE step (`qk moe 0`) | PASS 1.2e-5, router MATCH | **64.2 µs/layer** | 4 dispatches, was 148.5 as a naive port |
| MoE step, q6k downs (`qk moe 34`) | PASS 7.0e-6 | 87.1 | blk 34/38/39 variant |
| DeltaNet block (`qk block 0`) | PASS ≤9.3e-5 ×3 tokens; state drift 6e-8 | **178.4 µs** | 14 dispatches incl. MoE |
| attention block (`qk ablock 3`) | PASS ≤8.3e-5 ×3 tokens | **170.2 µs** (pos=2) | 13 dispatches incl. MoE |

Projected decode from block timings: 30×178.4 + 10×170.2 + head 0.84 ms +
embed/argmax ≈ **8.0 ms/token ≈ 126 tok/s ≈ 1.50× llama.cpp** — above the
1.3× DoD line before M4 glue and M6 tuning. (Stretch 150 needs ~1.3 ms more
off the blocks; see the overhead notes below.)

Structural findings that drove the design (measured by
`bench/metal_barrier_cost.swift`):

- **memoryBarrier between dispatches costs ~3.5 µs** (not the problem);
  **threadgroup launch throughput ~150 tgs/µs is the real tax** — a
  2048-threadgroup dispatch takes ~14 µs even empty. Kernels therefore use
  one SIMDGROUP per output row (not one 256-thread group), NSG=4 simds per
  threadgroup, which cut the MoE chain 148.5 → 82 µs.
- **A lone simdgroup doing a 2048-element dot is latency-bound (~20 µs)**
  — no other warps hide DRAM latency. moe_select's shared-gate dot moved
  into the parallel moe_logits dispatch (virtual row n_expert): 84 → 64 µs.
- Merging the shared expert INTO the routed gateup/down dispatches (4
  dispatches, 3 barriers, single y write) simplifies ordering; hazard
  tracking is off (untracked buffers) and ordering is explicit
  memoryBarrier in a concurrent-dispatch encoder — the Vulkan model.
- dn_qknorm is FUSED into dn_step's staging pass (the head is resident in
  threadgroup memory; norm = one simd_sum) — one dispatch + one device
  barrier saved; k-head norms recompute per v-head, cheaper than the
  barrier.
- Remaining per-block time is ~76 µs genuine serial work (qkv GEMV ~30,
  delta state R/W 8.4 MB ~21, ssm_out GEMV ~16) + ~40 µs stage overhead
  (launch + barrier + single-tg latency kernels: rms/addN/add/select).
  M6 leads: merge rms into the preceding residual add (add_rmsnorm
  pattern engine-wide), atomic last-tg-standing logits→select fusion,
  nr0=2 row pairs in the MoE kernels, f16 h/activations.

Kernel inventory after M3: 19 MSL kernels; every decode-path operation of
the hybrid architecture now has a validated Metal implementation.

## M4 — end-to-end greedy decode: token-exact parity at 1.43× (2026-07-09)

**`qk token` runs the full model — and beats llama.cpp Metal with exact
correctness.**

- **Parity: greedy output is token-for-token identical to llama.cpp Metal**
  (same build, same GGUF, same input ids, temperature 0) on 3 prompts ×
  100 tokens — 100/100 each. References via `llama-server /completion`
  with `"temperature":0` (NOTE: do NOT pass `"samplers":[]` — that removes
  the temperature stage and silently makes the reference non-greedy; an
  hour went there).
- **Decode: 8.26–8.36 ms/token = 119.7–122.1 tok/s**, thermally flat over
  repeated runs. llama.cpp Metal tg128 on this box: 84.2 tok/s →
  **1.43×** (DoD ≥1.3× met; stretch 150 = M6 target).
- Prefill in this harness is serial per-token (~9 ms/token after warmup;
  first command buffer pays ~1.6 s of first-touch page-in for the 15 GB
  resident set). Batched prefill is M5 (`prefillbench` on RDNA3 got 4.2×
  serial).

Structure: the whole generation is TWO command buffers (one prefill, one
decode). GPU-resident sampling: head GEMV (q6_k, nsg=2) → two-pass argmax
(ties to lower index, matching llama.cpp greedy) → winner recorded to a
history buffer and fed straight into `embed_q6k` for the next position —
the host reads ids only after the last token. Layer chaining uses
`add_rmsnorm` as the layer tail (residual sum + NEXT layer's attn_norm;
output_norm after layer 39), so a layer is 13 (attn) / 14 (deltanet)
dispatches. ~15.0 GB resident (UMA, untracked shared buffers).

Per-token budget at 8.3 ms: 30 deltanet blocks ≈ 5.3 ms + 10 attn blocks
≈ 1.7 ms + head+argmax+embed ≈ 1.0 ms + inter-block glue ≈ 0.3. M6 levers,
in expected-value order: q8_0 GEMV shapes dominate the blocks (qkv 30 µs +
out 16 µs per deltanet block — nr0-pair rework like M2b's q6_k could
reclaim ~15-20%); the ~40 µs/block fixed overhead (single-tg latency
kernels + launches); iq3 gateup ALU.

## M6a — fusion experiments: what moved, what didn't (2026-07-09)

Three structural fusions attempted after M4, each validated (block/ablock
PASS, token parity re-verified 100/100 ×3 prompts after every change):

1. **dn_step v3** — conv+silu (idempotent state shift), q/k L2 norm, delta
   rule, and the gated output norm in ONE dispatch (was four). The shared
   q/k channels of a k-head are convolved redundantly by the hV/hK
   v-head threadgroups; the state shift writes identical values (benign
   race). Block: 178.4 → 172.8 µs. KEPT.
2. **Inline expert re-selection** (moe_pick_all per consumer simdgroup, no
   select stage): correct but **39% SLOWER** (MoE 64 → 89 µs) — the
   +30-register footprint (v[8]+ids[8]+ws[8]) collapses occupancy on the
   DRAM-bound expert kernels and latency hiding dies. REVERTED; the
   dedicated 1-simdgroup select stage stays. Lesson: on Apple, a fat
   memory-bound kernel buys nothing from absorbing scalar work that
   needs registers.
3. **moe_logits_addn** (residual add + post_attention_norm folded into the
   router-logits dispatch; every simd recomputes the scalar RMS from
   SLC-hot vectors — register-light, unlike #2) and **embed_q6k + layer-0
   rms fusion**: correct, ~neutral on time (−1 stage/layer and −1/token
   but block time flat within noise). KEPT (fewer stages = simpler M5
   chunk encoding), but the honest reading:

**Stage-count reduction has hit diminishing returns.** Removing ~200
stages per token (5/layer) moved end-to-end <2%. The block runs at an
effective ~350 GB/s against 60.3 MB/layer while the same kernels do
500-522 standalone — the gap is INSIDE the bandwidth-bound stages:
per-stage DRAM ramp from idle after each barrier, and the delta-state
access pattern (each thread walks a private 512 B row; line utilization
recovers only across 8-iteration windows). Decode: 118.6–119.8 tok/s
(vs 119.7–122.1 pre-fusion — flat), parity intact.

M6 leads, updated by evidence: (a) ~~delta-state transpose~~ TESTED —
[i][j] scalar layout is 16% WORSE (203 vs 175 µs/block): 4× instruction
count and dependent scalar chains beat the coalescing win; the float4
row layout already recovers line utilization over its 8-iteration
windows. Reverted. (b) nr0 row-pairing for the in-block q8_0 GEMV
shapes; (c) probe the post-barrier bandwidth ramp with a two-dispatch
microbench — if real, fewer/fatter stages is the endgame; (d) f16
h/activations; (e) speculative decoding (docs/spec-decode-qk-plan.md)
is the step-change lever once serving works.

## M5 — serving on Metal: engine, harnesses, server, CLI round trip (2026-07-09)

**The full serving stack runs on this laptop.** `libqk.dylib` (same TU as
the CLI, `QK_LIBRARY`), the unmodified-in-spirit Rust server (one 12-line
compatibility patch: Claude Code ≥2.2 sends in-messages `system` turns),
and the qk.h engine implemented on Metal with the fused kernel set.

Gates, all green on first full runs:

| gate | result |
|---|---|
| serve-test 1 slot | TOKEN-EXACT vs llama.cpp greedy reference |
| prefillcmp (14 sizes × 3 seeds) | 36/36 argmax MATCH, worst rel 1.3e-6 |
| prefilldecode ×3 seeds | HANDOFF EXACT |
| serve-test 4 slots | all slots identical YES, 104.8 tok/s aggregate |
| serve-test2 staggered | OK |
| cachetest | warm ≡ cold YES |
| /v1/messages | "The capital of France is Paris." |
| Claude CLI tool round trip | **num_turns=2, is_error=false** at ctx 32768 |

The CLI round trip is the real-world stress test: Claude Code's 2026
system prompt is **29.8k tokens**. Turn 1 cold-prefills 29,809 tokens;
turn 2 hits the prefix cache with `reuse=29809 prefill=44` — the
cross-turn O(delta) design carried over from RDNA3 intact, as plain UMA
memcpys (snapshots are host vectors now; no staging buffers, no Vulkan
copy machinery).

Metal-specific engine notes: no pre-recorded command buffers exist on
Metal, so the serial step is encoded per token (~0.4 ms host) with grid
z = highest-active-slot; per-slot state stripes bind via setBuffer:offset;
gemm_q8_0 restructured to BK=32 (Apple's 32 KB threadgroup ceiling vs
RDNA3's 64 KB LDS); srv attention uses online softmax — context is bounded
by KV memory, not threadgroup arrays (32k validated).

Prefill today: 214–241 tok/s batched = 2.5–2.9× serial (Vulkan reached
4.2×; llama.cpp Metal does 1452). Phase B owns this: the scalar f32 GEMM
and the z-per-token MoE are the gaps.

## Phase B (in progress) — prefill: 2.9× serial, GEMM variants benched (2026-07-09)

`caseBGemm` ports the GEMM validation harness. Measured (M=8192, K=2048):

| variant | N=128 | N=512 | correctness |
|---|---|---|---|
| scalar f32, BK=32 (shipping) | 1.92 TFLOPS | 2.34 TFLOPS | PASS 4.5e-4 |
| simdgroup_float8x8, BK=32 | 1.03 TFLOPS | 1.39 TFLOPS | PASS 2e-3 — **slower**: f32 MMA ≈ scalar on Apple + fragment-store tail |
| simdgroup_half8x8, BK=64 | — | — | **FAIL 0.9 rel — bug** (structural, not f16 rounding; bisect vs the passing f32 twin next: BK=64 two-block staging is the delta) |

UPDATE (same day): the half8x8 "bug" was three things at once, now
resolved: (1) the synthetic 1e-2 tolerance is unpassable for ANY f16-input
GEMM under cancellation (llama.cpp's own prefill would fail it) — the real
gate is model-logit argmax-exactness; (2) a rewrite to 8 simdgroups + BK=64
+ direct transposed device simdgroup_store regressed the gates, which was
finally a HOST bug — dispatching the 256-thread kernel with 128 threads
(half the simds never staged/computed); (3) with threads fixed and the
proven bounce store, **gemm_q8_0_h is the engine default**: prefillcmp
36/36 TOKEN-EXACT (worst rel logit 0.073 — llama.cpp's f16 precision
class), prefilldecode HANDOFF EXACT, prefillbench **285–318 tok/s =
3.6–3.85× serial** cold, ~3.0–3.5× thermally soaked. QK_GEMM=scalar forces
the bit-exact f32 path (247 tok/s).

Remaining to B1's 4× and B2's 1452, updated by measurement
(QK_PREFILL_SKIP stage isolation at N=128, chunk 408 ms): **MoE = 218 ms
(53%)**, projections ≈ 48 ms (f16 GEMM did its job), DeltaNet batch ≈ 0,
attention ≈ noise. A read-once grouped gu (token→expert counting sort +
one simdgroup per (expert,row) looping its tokens; bit-identical results,
36/36) measured SLOWER (522 vs 408 ms) — decisive evidence the MoE stage
is **ALU-bound on iq3 decode (~152 GB/s effective), not DRAM-bound**. The
real lever is decode-ONCE: dequantize each expert row to threadgroup
half-precision and multiply all its tokens GEMM-style (the gemm_q8_0_h
structure with an assignment gather). Kernels + QK_MOE_GROUPED env kept
for that next round.

Budget at N=128 (530 ms/chunk): projections ≈180 ms (scalar GEMM), MoE
≈180 ms (ungrouped expert reads), dn_step_batch ≈60–120 ms (32 tgs,
serial over Tn — low occupancy by design, state in registers), attention +
small ops ≈50 ms.

## C1 — no-copy weights: 15.5 GB RSS → 318 MB (2026-07-09)

The engine now wraps the entire GGUF mmap in ONE
`newBufferWithBytesNoCopy` MTLBuffer (page-aligned by mmap; length rounded
up to the page). Every weight binding became a (buffer, offset) pair via a
`WB` wrapper that converts implicitly from a plain buffer, so activation
bindings didn't change textually; per-tensor offsets come straight from
the GGUF parse (32-byte aligned, satisfying setBuffer's float4 needs).
State/activation buffers stay allocated.

Measured (serve-test process): **maximum RSS 318 MB, peak footprint
508 MB** — weights stream from file-backed pages on demand, zero copies at
open, no double residency. Output token-exact; prefilldecode HANDOFF
EXACT and prefillcmp 36/36 re-verified after the binding surgery. This is
now BETTER than llama.cpp's mmap path (which still faults the whole model
into its own dirty pages for GPU residency on load).

## D — cross-box pipeline split on Metal (tron brief, 2026-07-10)

Merged origin/main 0c508e2 (clean; the in-messages system-turn patch had
converged with upstream d39168a). Metal engine implements the frozen ABI:
`QK_LAYERS=a:b` partial load (layer states only in range; logits/argmax
buffers only on the last stage; weights are free either way — no-copy),
`qk_stage_run` chunked over maxB with hidden-in → first-layer rms entry
and hidden-out = raw residual rows via UMA memcpy; the boundary layer
binds a zeroed dummy next-norm (dead xn), exactly the Vulkan trick.
`qk pipe`/`qk pipe-worker` compiled from main.cpp verbatim.

Gates (brief §4), all token-exact vs the unsplit engine AND the llama.cpp
greedy references:

| gate | result |
|---|---|
| a. in-process split 20 (deltanet boundary) | GEN exact |
| a. in-process split 24 (attention-layer boundary) | GEN exact |
| b. TCP worker localhost, split 20 | GEN exact |
| b. reconnect, second prompt | GEN exact |

Timing (localhost, warm): 9.55 ms/tok split vs 8.3 unsplit → split
overhead ≈ 1.2 ms/tok including loopback TCP of the 8 KB hidden row.
In-process two-engine mode is slower (52–76 ms/tok — two Metal contexts
contending in one process); use the worker for real numbers. For tron
(gate c): launch the worker here with slots/ctx covering the head, e.g.
`QK_LAYERS=22:40 QK_GGUF=... build/qk pipe-worker <port> 22:40 2 32768`
— s2 on midnight measured ~5.3 ms/tok at split 20 with 4k ctx; balance S
per the brief using tron's ~5.8 ms/tok full-model decode.

**Gate (c) PASSED (tron-driven, 2026-07-10): midnight served tron's
qk-server over the LAN token-exact at pipe level and byte-identical
through the full `--split-next` HTTP path. 17.2 ms/tok engine-level
(midnight stage ~5 ms + WiFi RTT ~4.3 ms), 50 tok/s streamed. Two GPU
vendors — RDNA3 head, Apple Metal worker — one bit-exact model.**
Follow-ons (wired link, S sweep, slots=4 worker) tracked on the tron
side; the midnight worker stays up on :18100.

One false alarm worth recording: an apparent engine-vs-reference
divergence on prompt 3 was a stale memory of the PRE-fix (accidentally
sampled) reference — the regenerated greedy ref matches the engine
exactly. Trust files, not recollection.

### Method notes (M0)

- llama.cpp built in-tree at `../llama.cpp` (fresh clone of master, same-day);
  `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON`. Note: on this master,
  `llama-cli` requires `LLAMA_BUILD_SERVER=ON` (cli links `llama-server-impl`).
- The 546 GB/s denominator is the spec sheet; 494 GB/s is what a pure
  streaming-read kernel actually sustains (90.5%). Extraction percentages
  against 546 are therefore conservative lower bounds.
- DeltaNet state bytes assume f32 state resident on GPU, read+write per
  token per linear layer; conv-state traffic (~0.1 MB/layer) ignored.

## Execution plan — beat llama.cpp Metal on EVERY axis (2026-07-09)

Scorecard today (this box, this GGUF) and the plan to flip every ✗:

| axis | llama.cpp | qk | plan |
|---|---|---|---|
| decode 1-stream | 84.2 tok/s | **120.5 ✓** (parity ✓) | stretch to 150 (C3) |
| prefill batched | 1452 tok/s | **318** (argmax-exact, 3.85× serial) | B: grouped-MoE GEMM next |
| serving/multi-slot | llama-server | **✓ full stack + CLI round trip** | done (M5) |
| agg. throughput | --parallel N | **146.3 tok/s @ 8 slots** (1.74× their 1-stream) | B3 head-to-head pending |
| long ctx | 256k | **✓ 32k engine / 29.8k proven live** | C2 f16 KV for headroom |
| load time / RSS | mmap no-copy | **✓ 318 MB RSS, zero-copy open** | done (C1) |
| quality | reference | ✓ token-exact everywhere | parity gate on every change |

**Phase A — serving foundation (M5):**
A1 port batch/srv kernels: gemm_q8_0, fa_prep/attn_batch, fa_prep/attn_srv,
   dn_*_batch, moe_down_q8b, add_rms3 (batched-prefill + multi-slot chains).
A2 qk.h C ABI engine in main_metal.mm; build libqk.dylib (APPLE branch).
A3 harnesses green: prefillcmp, prefillbench, prefilldecode, serve-test,
   serve-test2, cachetest — all token-exact.
A4 prefix cache snapshot/restore = plain UMA memcpys.
A5 Rust server dlopens libqk.dylib; /v1/messages up; claude CLI round trip.

**Phase B — prefill/throughput supremacy:**
B1 batched prefill ≥4× own serial (DoD bar) via chunked GEMM chain.
B2 profile → simdgroup_matrix GEMM for q8/iq3/iq4 to ≥1452 tok/s pp512
   (llama.cpp ≈ 31% of f16 peak here; beatable with fused-MoE GEMM).
B3 multi-slot aggregate decode > llama.cpp --parallel at 2/4/8/16 slots.

**Phase C — polish past llama.cpp:**
C1 no-copy weights: ONE MTLBuffer over the whole GGUF mmap
   (newBufferWithBytesNoCopy needs page alignment — the mmap base is;
   per-tensor via setBuffer:offset:, GGUF aligns tensors to 32 B ✓)
   → load ≈ llama.cpp, peak RSS −15 GB.
C2 long context: fa srv kernels (chunked softmax past the 1024 cap),
   f16 KV cache (llama.cpp default — keeps comparisons apples-to-apples),
   ctx 16384 serving config like the RDNA3 deploy.
C3 decode 150 stretch: post-barrier bandwidth-ramp microbench, nr0
   row-pairing on in-block q8 GEMVs, then spec-decode per
   docs/spec-decode-qk-plan.md as the step change.
C4 final scorecard in README + PORT.md: decode, prefill, aggregate, TTFT,
   load, RSS, CLI-path E2E — every row green vs llama.cpp Metal.

Rules of engagement (unchanged): every kernel lands with CPU-reference
validation; every phase ends with the token-parity gate ×3 prompts; every
result (positive or negative) recorded here; commit per green step.
