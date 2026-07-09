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

### Method notes

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
