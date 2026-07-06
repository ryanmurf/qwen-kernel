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
The single-stream optimization arc has converged: 76% of the ~230 tok/s
kernel-time ceiling. What remains is the serial dependency chain itself
(~8 barrier drains/layer) and cold weight streaming.

**Batched multi-request decode** (`qk token <ids> <n> [tmax] [batch]`):
every token-path shader carries a request index on the dispatch z-axis —
weights shared across streams, activations/states/KV/logits striped per
request. N identical greedy streams double as validation (all must emit
the reference sequence byte-identically; they do at N=1/4/8/16).

| batch | ms/step | per-stream tok/s | aggregate tok/s |
|---|---|---|---|
| 1 | 5.82 | 171.8 | 171.8 |
| 2 | 8.48 | 117.9 | 235.7 |
| 4 | 13.55 | 73.8 | 295.2 |
| 8 | 28.86 | 34.7 | 277.2 |
| 16 | 41.72 | 24.0 | **383.5** |

Dense weights and per-step overheads amortize; what doesn't is the routed
expert reads (union of top-8 grows with N) and the deltanet recurrent
state (N × 2 MB × 30 layers × ~3 accesses/step — at N=16 that's ~2.9 GB/step
of state traffic, the new bandwidth wall; note the N=8 cache-crossover
dip). Real serving atop this needs per-stream prompts/positions (pos in a
per-request buffer instead of push constants) and EOS handling — the
kernel machinery is already stream-capable.

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

## Serving layer & CPU-precompute optimizations

A safe Rust serving layer (`server/`, binary `qk-server`) fronts the engine:
llama.cpp-compatible HTTP (`/completion`, `/tokenize`, `/v1/chat/completions`,
…), a GGUF-native byte-level BPE tokenizer that reproduces llama.cpp's output
token-for-token (verified against captured fixtures), and chat templating. All
untrusted-input handling is safe Rust; the only `unsafe` is the one-thread
`dlopen` FFI (`include/qk.h`) to the C++/Vulkan engine (`libqk`). See
`server/README.md` and `docs/server-spec.md`.

The engine side (`libqk.so`) is a persistent per-slot state machine: N slots
each carry their own sequence position and input token (sourced per-slot from a
buffer via `fa_prep_srv`/`fa_attn_srv`), so requests of different prompt lengths
— some prefilling, some decoding — batch into one dispatch. Run it:

```bash
/usr/bin/cmake --build build --target qklib -j     # build libqk.so
cd server && cargo build --release
QK_SHADER_DIR=../build/shaders ./target/release/server \
    --model /path/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf \
    --engine-lib ../build/libqk.so --port 8080 --slots 4 --ctx 1024 --chunk 8
```

Validated end-to-end on real GPU inference (server quiesced): tokenization
matches llama.cpp exactly; greedy `/completion` matches the single-stream
reference; 4 concurrent distinct prompts each reproduce their own reference
byte-for-byte; EOS termination, SSE streaming, and chat templating all correct;
malformed input returns structured 400s without crashing. **135 tok/s single
stream, 216 tok/s aggregate across 4 concurrent streams** (4 slots, ctx 1024) —
single-stream beats the llama.cpp server (72–86 tok/s), aggregate ~2.7×. The
engine pre-records one step command buffer per dispatch depth and submits the
one matching the highest active slot, so a lone request dispatches a single
z-slice (full 1-slot weight bandwidth) instead of paying for all four idle
slots. (`qk serve-test <ids> <n> [slots]` drives the same C ABI in-process for
regression checks.)

Two CPU-precompute optimizations, both verified token-exact against the
llama.cpp reference:

- **RoPE cos/sin table.** The NeoX partial-rope angles depend only on
  `(pos, j)`, so `fa_prep` no longer recomputes `pow()/cos()/sin()` per element
  — they're precomputed on CPU into an SSBO indexed by `[pos][j]`. Decode-clock
  impact is **within noise** (5.78–5.87 vs 5.82 ms/step): the step is
  bandwidth-bound on weight reads, not transcendental-bound (same lesson as the
  IQ3 repack). Kept because it's the correct foundation for per-request
  positions in a multi-slot engine (each slot indexes the table by its own
  position). *Precomputing the quantized weights themselves would be a
  pessimization* — it expands the ~2.5 GB of compressed reads to ~8–10 GB and
  makes the bandwidth-bound loop slower; the quant kernels dequant-on-read for
  exactly this reason.

- **Prefix-state caching (`qk warm`).** The highest-value serving lever. A
  shared prefix (e.g. a fixed system prompt) is prefilled once; its ~60 MB of
  per-slot recurrent state — delta-rule `S` + conv window across 30 recurrent
  layers, K/V caches across 10 attention layers — is snapshotted, then restored
  (cloned) into a new request's slot instead of re-running prefill.

  | shared prefix | cold prefill | warm restore | TTFT speedup | warm≡cold |
  |---|---|---|---|---|
  | 10 tokens | 57.9 ms | 0.3 ms | 207× | YES |
  | 64 tokens | 340.7 ms | 0.3 ms | 1083× | YES |

  Restore is O(state) (constant); prefill is O(prefix), so the win grows with
  prefix length. Snapshot costs ~0.5–1.0 ms, paid once when a prefix is first
  seen. The warm-decoded token stream is byte-identical to the cold stream,
  proving the snapshot/restore preserves state exactly.

- **Batched prefill (`qk prefillcmp` / `prefillbench` / `prefilldecode`).** The
  serial engine prefills a prompt one token at a time — each a full forward,
  including the 248k-row LM head whose logits are then discarded. Batched prefill
  processes the whole prompt chunk in ONE command buffer with the token index on
  the dispatch z-axis: the dense projections become a register-blocked Q8_0 GEMM
  that reads each weight once for all N tokens (vs N re-reads by the per-token
  GEMV); the 30 gated-deltanet layers run the recurrence as one workgroup-per-head
  kernel that loops the chunk internally with the delta-rule state `S` resident in
  registers (the sequential dependency collapsed into one dispatch); the 10
  attention layers use batched flash-attention writing K/V at `pos=base+n`; and
  the MoE / embed / norm / residual ops reuse the per-token kernels dispatched
  z=N. Below a measured ~48-token crossover the projections fall back to per-token
  GEMV (the 128×64 GEMM tile wastes work when N fills under one tile), so the path
  is optimal at every chunk width.

  `qk_slot_start` uses it automatically: a fresh (prefix-cache-miss) prompt of
  17–129 tokens is batch-prefilled for all but its last token (LM head skipped
  entirely), then the existing serial step feeds the final token — producing the
  first generated token exactly as the all-serial path does, reading the
  batched-filled K/V + delta-rule `S` + conv window. Shorter prompts, `>maxB`
  prompts, and prefix-cache hits fall back to per-token serial prefill;
  `QK_NO_BATCH=1` forces serial. Any slot can be targeted (the per-slot state
  bindings are rebound before recording), so multi-slot concurrency is preserved.

  Correctness is proven at three levels, all token-for-token identical to serial:
  the batched forward's last-token logits are **bit-identical** (`max|Δlogit| = 0`,
  GEMV path) across chunk sizes 1–128 × 3 seeds (`prefillcmp`); the state handoff
  reproduces all-serial generation exactly (`prefilldecode`, N = 16–100); and the
  wired server emits identical tokens on 40/100/128-token prompts (GEMV & GEMM
  paths) and the 12-token serial fallback (`serve-test` vs `QK_NO_BATCH=1`). The
  GEMM path (N ≥ 48) differs from serial by ~7e-7 rel (tiled accumulation order) —
  argmax-stable. Independently correctness-reviewed (no bug found).

  | prefill, 128-token prompt | serial | batched | speedup |
  |---|---|---|---|
  | isolated forward (`prefillbench`) | 1196 ms | 286 ms | **4.18×** |
  | full request + 8 gen (`serve-test`, prefill included) | 867 ms | 284 ms | **3.05×** |

  ~447 tok/s prefill throughput at a 128-token chunk (vs serial ~107); the GEMM
  itself sustains ~6.3 TFLOP/s (3.6–4.8× a serial GEMV at N = 64–256). Caveat:
  `prefillBatchLast` is from-empty single-chunk only — continuing an existing slot
  or a `>maxB` prompt would need a `base` push constant, per-layer conv-carry
  seeding, and dropping the internal `resetSlot` (documented in-source).

- **Fork mode (`QK_FORK`) — same-prompt requests share one prefill.** For internal
  best-of-N / duplicate-prompt bursts, an opt-in mode: because batched prefill runs
  eagerly and synchronously in `slot_start`, the first request's prompt is prefilled
  by the time `slot_start` returns, so it snapshots that `[0,K)` prefill into the
  prefix cache (keyed by `prompt[0:K]`). The next same-prompt request hits
  `matchPrefix` and `restoreInto`s it (one state copy) instead of re-running the whole
  prefill. `slot_start` is called per request on the single engine thread, so request
  1 caches and 2..N fork — covering concurrent, not just staggered, duplicates.
  Output is token-for-token identical to independent prefill (validated: 4 same-prompt
  slots all identical to the serial reference); **~2.68× faster for 4 same-prompt slots**
  (874 → 326 ms: one prefill + three forks vs four prefills). Off by default — it adds a
  state snapshot to each fresh prompt, so it only pays off when duplicates actually
  occur (the 30 recurrent layers' delta-rule state still diverges per sample the moment
  generations differ, so this shares the *prefill*, not decode-time VRAM).

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
./build/qk token <ids-file> <nGen> [tmax] [batch]  # end-to-end greedy generation (server must be scaled down)
./build/qk warm <ids-file> <nGen> [tmax]   # prefix-cache cold/warm-start demo (TTFT)
./build/qk serve-test <ids-file> <nGen> [nSlots] [tmax]  # drive the libqk C ABI in-process
./build/qk prefillcmp [N<=128] [ctx]       # batched-prefill logits vs serial (token-exact sweep)
./build/qk prefillbench [ctx]              # batched vs serial prefill timing
./build/qk prefilldecode [N] [M] [ctx]     # batched prefill -> serial decode == all-serial (handoff)
./build/qk list [filter]            # list tensors in the GGUF
```

Env: `QK_DEVICE=<n>` forces the Vulkan device index (default: first discrete
GPU); `QK_SHADER_DIR` overrides the SPIR-V directory; `QK_GGUF` overrides the
model path; `QK_NO_BATCH=1` disables batched prefill in the serving path (forces
per-token serial prefill — used as the correctness reference); `QK_FORK=1` enables
fork mode (same-prompt requests share one prefill via the prefix cache).

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
