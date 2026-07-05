# qwen-kernel

Custom Vulkan compute kernels for RDNA3 (Radeon RX 7900 XT), aimed at
megakernel-style fused decode for **Qwen3.6-35B-A3B** — the model served by
llama.cpp (`gemma` namespace, `llama-server:vulkan`) on this machine.
Inspired by [KernelBench Mega](https://kernelbench.com/mega), which is
CUDA-only; this is the RDNA3/Vulkan equivalent.

## Status

- **M1 (done): harness + fp16 GEMV** — `qk` races a GLSL compute GEMV
  (`y = W·x`, the core decode op) against a CPU reference: correctness check,
  then GPU-timestamp bandwidth/FLOPs measurement.
- **M2 (done): quantized GEMV (Q8_0, Q6_K)** — dequant-in-kernel on raw ggml
  blocks (8/16-bit storage), validated bit-exact against a CPU reference
  ported from ggml-quants.c, on both synthetic blocks and real tensors mmap'd
  from the GGUF. Includes a minimal GGUF reader (`src/gguf.h`) and 2D dispatch
  for M > 65535 (the 248k-row LM head).
- **M3 (done): skinny-K + IQ quants** — TPR spec constant packs 256/TPR rows
  into one workgroup for short rows (+60% on `attn_qkv`, +10% on the LM
  head); IQ4_XS (nonlinear codebook) and IQ3_XXS (grid codebook +
  parity-computed signs) kernels validated on real expert tensors. Every
  weight format in the model is now covered. Tables in `src/quant_tables.h` /
  `shaders/iq_tables.glsl` are generated from ggml-common.h.

Measured on RX 7900 XT (RADV), server pod resident:

| kernel | test | µs/iter | GB/s | GFLOP/s |
|---|---|---|---|---|
| f16    | synthetic 16384×8192 (256 MiB) | 345 | 779 | 779 |
| q8_0   | synthetic 16384×16384 (272 MiB) | 369 | 773 | 1455 |
| q6_k   | synthetic 16384×16384 (210 MiB) | 293 | 752 | 1833 |
| iq4_xs | synthetic 16384×16384 (136 MiB) | 236 | 606 | 2279 |
| iq3_xxs| synthetic 16384×16384 (98 MiB)  | 208 | 495 | 2583 |
| q8_0   | real `blk.0.attn_qkv.weight` 8192×2048 | 20 | 896* | 1682 |
| q6_k   | real `output.weight` (LM head) 248320×2048 | 581 | 720 | 1752 |
| iq4_xs | real `blk.0.ffn_down_exps` expert 0, 2048×512 | 4.1 | — | 509 |
| iq3_xxs| real `blk.0.ffn_gate_exps` expert 0, 512×2048 | 4.0 | — | 519 |

\* cache-resident (17 MiB < 80 MB Infinity Cache). Peak VRAM bandwidth is
~800 GB/s: big Q formats sit at 90–97% of it; IQ formats are ALU-bound on
codebook unpacking (highest GFLOP/s, lower byte rate). Per-expert GEMVs are
launch-bound at ~4 µs — 24 of those per layer per token is exactly why M4
fuses them.

Note: despite the filename, this GGUF has **no Q3_K tensors**. Unsloth
Dynamic used Q8_0 (all dense per-token projections), Q6_K (LM head +
embeddings), and IQ3_XXS/IQ4_XS (the 256-expert tensors — the bulk).

- **M4 (done): fused MoE decode step** (`qk moe <layer>`) — the entire
  MoE-FFN for one token in ONE queue submission of six dispatches:
  router logits (F32) ∥ shared-expert gate/up (Q8_0) → top-8 select +
  softmax weights + sigmoid shared gate (selection stays on-GPU; the
  full-softmax denominator cancels under renorm, so selection is by raw
  logit) → routed gate/up (IQ3_XXS, gate+up in one pass) → weighted down
  (IQ4_XS, or Q6_K for layers 34/38/39) → shared down added (Q8_0).
  Validated end-to-end against a CPU reference implementing llama.cpp's
  exact `build_moe_ffn` semantics on real weights: router expert ids +
  weights MATCH, output max_rel_err ~1e-5.

| layer | down type | µs/layer-MoE | active weights |
|---|---|---|---|
| blk.0 / blk.20 | IQ4_XS | 38.5 | 15.6 MiB |
| blk.34 (also 38, 39) | Q6_K | 58.2 | 17.9 MiB |

  All-40-layer MoE-FFN share: **~1.6 ms/token** (~423 GB/s on active
  weights; ~54% of the bandwidth ceiling — the IQ unpack ALU cost and
  per-dispatch fixed overhead are the next optimization targets).

- **M5 (done): fused deltanet decode block** (`qk block <layer>`) — a full
  layer (30 of 40 are gated-deltanet) in ONE submission of 17 dispatches:
  RMS norm → QKV/z projections (Q8_0) ∥ per-head α/β → depthwise conv(4) +
  SiLU with on-GPU rolling conv state → per-head L2 norm of q/k → gated
  delta rule `S ← exp(g)·S + β·k(v−Sk)ᵀ, o = Sq/√d` (state 32×128×128,
  persistent on GPU) → gated RMS (`· ssm_norm · silu(z)`) → ssm_out →
  residual + post-norm → full M4 MoE chain → residual. Validated over
  multiple tokens against a CPU reference ported op-for-op from llama.cpp's
  qwen35moe graph: output max_rel_err ~1e-5 per token, delta state drift
  after 3 tokens ≤ 3e-7 abs.

| metric | value |
|---|---|
| one deltanet block | **95.5 µs** (was 334.5 before parallelizing the α/β projections) |
| 30 deltanet blocks | 2.87 ms/token |
| + LM head (M2/M3) | 0.58 ms |
| est. full model (full-attn blocks pending) | **~4.5 ms/token ≈ 220 tok/s** |
| llama.cpp Vulkan, same GPU/model (measured live) | 15.8 ms/token = 63.2 tok/s |

  Roughly **3× headroom** vs the running server, with caveats: the 10
  full-attention layers are estimated at deltanet cost (unimplemented),
  sampling excluded, single token, short context. GQA head mapping follows
  ggml_repeat modulo-tiling semantics (v-head h → k-head h%16).

- **M6 (done): full-attention blocks + end-to-end generation.**
  - `qk ablock <layer>`: the 10 non-recurrent layers (3,7,...,39) — q(+gate)/k/v
    projections, per-head RMS + partial NeoX rope (IMROPE degenerates to this
    for text-only equal positions; verified against ggml's sector math),
    KV-cache attention with mul_mat-broadcast GQA grouping (q-head h → kv
    h/8), sigmoid output gate, wo, post-norm, MoE. Validated 3 tokens vs CPU
    reference: max_rel_err ~1e-4, **80.6 µs/block**.
  - `qk token <ids> <n>`: the whole model — Q6_K embedding lookup, all 40
    blocks with persistent per-layer states, final norm, Q6_K LM head over
    the 248,320 vocab, greedy sampling. ~15.4 GB resident (quiesce the
    llama-server pod first).

**End-to-end proof** (prompt "The Eiffel Tower is located in the city of",
identical token ids fed to both engines, temperature 0):

| | llama.cpp Vulkan (live server) | qwen-kernel |
|---|---|---|
| first 12 greedy tokens | `11751 11 9338 13 1049 557 5617 303 220 16 23 23` | **identical** |
| text | " Paris, France. It was built in 188…" | same (continues "…1889 for the World's Fair and was originally intended to be") |
| decode (same prompt, back-to-back runs) | 72.5–86.4 tok/s (13.8–11.6 ms/tok) | **172.0 tok/s** (5.82 ms/token) |
| prefill | 37.5 tok/s | **173 tok/s** (5.77 ms/token, sequential decode-path) |

Token-exact greedy parity with llama.cpp across the full stack at
**2.0–2.35× the server's decode speed**. Optimization history (ms/token):
33.3 → 12.8 (argmax was scanning logits in write-combined host memory)
→ 11.4 (pre-recorded per-position command buffers + in-place residual add)
→ 6.06 (fully GPU-resident sampling loop: GPU 2-pass argmax over the 248k
vocab, GPU Q6_K embedding lookup, whole prefill and whole decode each ONE
vkQueueSubmit — the host reads the generated ids at the end)
→ 5.88 (layer-tail fusion: 3-way residual add + NEXT layer's RMS norm
in one dispatch, routed & shared MoE down-projections made concurrent)
→ **5.82** (subgroupAdd reductions replacing shared-memory trees in 12
shaders; conv+SiLU+state-shift+q/k-L2-norm fused into one dispatch).
The optimization arc has converged: 76% of the ~230 tok/s kernel-time
ceiling. What remains is the serial dependency chain itself (~8 barrier
drains/layer) and cold weight streaming — closing it needs multi-token
batched decode (speculative verify GEMMs), not more micro-fusion.

**Negative result, measured:** repacking IQ3_XXS into dword-aligned
25-uint blocks (d as f32 + pre-assembled aux words + grid bytes) and
reading IQ4_XS as raw uints changed nothing — 5.81 vs 5.82 ms/token,
outputs identical. RADV/ACO already coalesces the byte loads; the IQ
kernels are ALU-chain-bound, not fetch-bound. The experiment was reverted
(this repo's history has it at the commit before this note if ever needed).
- Barrier reduction (split barriers / finer scopes), vectorized IQ3_XXS
  loads, subgroup reductions.
- Batched (N-token) verify-pass GEMM for speculative decoding (see
  docs/speculative-decoding.md — routed experts stay weight-stationary
  skinny, dense weights become proper GEMM).

Architecture facts for this model (`qwen35moe`): n_embd 2048, 40 blocks,
hybrid attention (gated-deltanet SSM layers + full-attention layers,
`attn_qkv` 2048→8192, gate 2048→4096), 256 experts top-8 + 1 shared,
expert FFN 512, vocab 248320.

## Build & run

```bash
/usr/bin/cmake -B build        # NB: bare `cmake`/`ninja` on this box are broken pip shims
/usr/bin/cmake --build build -j

./build/qk                          # synthetic suite: f16, q8_0, q6_k, iq4_xs, iq3_xxs
./build/qk f16|q8_0|q6_k|iq4_xs|iq3_xxs [M] [K] [iters]
./build/qk gguf <tensor> [iters]    # real weights, e.g. output.weight
./build/qk moe [layer] [iters]      # fused MoE decode step on real weights
./build/qk block [layer] [tokens] [iters]  # full deltanet block, one submission
./build/qk ablock [layer] [tokens] [iters] # full-attention block (layers 3,7,...)
./build/qk token <ids-file> <nGen> [tmax]  # end-to-end greedy generation (server must be scaled down)
./build/qk list [filter]            # list tensors in the GGUF
```

Env: `QK_DEVICE=<n>` forces the Vulkan device index (default: first discrete
GPU); `QK_SHADER_DIR` overrides the SPIR-V directory; `QK_GGUF` overrides the
model path.

## Notes

- The llama-server pod keeps all 20 GB of VRAM resident. `qk`'s buffers
  (~0.5 GiB at default sizes) evict a slice of it temporarily (amdgpu pages it
  back on the server's next inference). For clean bandwidth numbers, quiesce
  the server first: `kubectl scale deploy gemma-server -n gemma --replicas=0`
  (and back to 1 after).
- Default W is 256 MiB — deliberately larger than Navi 31's 80 MB Infinity
  Cache so the number approximates true VRAM bandwidth (~800 GB/s peak on the
  XT). Sizes that fit in cache will read much higher.
- Reference shaders to study/steal from: llama.cpp's Vulkan backend at
  `~/intellij/ggerganov/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`.
