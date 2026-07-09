# qwen-kernel

A from-scratch Vulkan inference engine + serving stack for **Qwen3.6-35B-A3B**
on RDNA3 (Radeon RX 7900 XT). One model, fully specialized: hand-written
compute kernels for every weight format in the GGUF, the whole hybrid
gated-DeltaNet + MoE architecture fused into pre-recorded command buffers, and
a safe-Rust server speaking the **Anthropic Messages API** — so Claude Code
runs against it directly. Inspired by
[KernelBench Mega](https://kernelbench.com/mega) (CUDA-only); this is the
RDNA3/Vulkan equivalent, taken all the way to a serving engine.

## Benchmarks

**End-to-end through the Claude Code CLI** (same GGUF, same GPU, one backend
at a time — full methodology and raw data in [`bench/`](bench/README.md),
2026-07-08):

| scenario (median wall) | qk-server | llama.cpp Vulkan (master) | qk |
|---|---|---|---|
| single-shot generate ~350 tok | **5.3 s** | 32.5 s | **6.1×** |
| Bash tool call (2-turn agentic loop) | **1.5 s** | 6.3 s | **4.3×** |
| multi-turn resumed session, per turn | **2.3 s** | 14.3 s | **6.2×** |
| warm decode through the CLI | **65 tok/s** | 12 tok/s | **5.4×** |

**Engine-level**, measured on the 7900 XT (RADV), token-exact greedy parity
with llama.cpp verified across the full stack:

| metric | qk | llama.cpp Vulkan |
|---|---|---|
| single-stream decode | **172 tok/s** (5.82 ms/tok) | 72–86 tok/s |
| aggregate decode, 16 streams | **384 tok/s** | — |
| batched prefill (128-tok chunk) | **~450 tok/s** (4.2× own serial) | 37.5 tok/s |
| warm start from prefix cache | 0.3 ms restore (vs 341 ms/64-tok prefill) | — |

Correctness bar throughout: greedy output is **token-for-token identical** to
llama.cpp on identical input ids, batched paths are validated bit-identical
(or argmax-stable at ~1e-7 rel) against serial references, and the server's
tokenizer reproduces llama.cpp byte-for-byte.

## How it works

- **`src/main.cpp` + `shaders/`** — the engine (`libqk.so` / `qk` CLI).
  GEMV/GEMM kernels for Q8_0, Q6_K, IQ4_XS, IQ3_XXS, F16 at 90–97% of VRAM
  bandwidth on the big formats; the fused MoE step (256 experts, top-8 +
  shared) and gated-DeltaNet recurrence (state resident on GPU); full
  attention with GQA + partial NeoX rope; GPU-resident argmax sampling. A
  whole decode step is pre-recorded command buffers — one queue submit per
  chunk, host reads ids at the end. N slots batch on the dispatch z-axis, so
  concurrent requests of different lengths share every weight read.
- **`server/`** — `qk-server`, a safe-Rust (axum) HTTP layer over the one
  `dlopen`'d engine thread: llama.cpp-compatible endpoints plus native
  Anthropic `POST /v1/messages` (SSE streaming, hermes `<tool_call>` parsing,
  Qwen3 chat template), a GGUF-native BPE tokenizer, per-slot admission,
  context-fit trimming with fast-fail. See `docs/server-spec.md`.
- **Prefix / cross-turn KV reuse** — a prompt's KV + recurrent state is
  snapshotted at the **conversation-history boundary** and restored on the
  next turn, so a growing agentic session prefills only each turn's delta
  (O(N) per session instead of O(N²)). Production hit-rate ~88%; agentic
  turns ~5× faster. `QK_PCACHE_LOG=1` prints per-request reuse stats.
- **Batched prefill** — prompt chunks run as one command buffer with a
  register-blocked Q8_0 GEMM (weights read once per chunk instead of once per
  token) and the DeltaNet recurrence collapsed into one workgroup-per-head
  dispatch. Auto-selected; `QK_NO_BATCH=1` forces the serial reference path.
  Chunk width saturates at 128 (measured; 256 is token-exact but +1%).
- **Pipeline split (`QK_LAYERS=a:b`)** — an engine instance can own just
  transformer layers `[a,b)`: the first stage also owns the embedding, the
  last owns final-norm + head + argmax, and the ~8 KB/token residual row
  crosses stage boundaries through the `qk_stage_run` ABI (in-process or
  TCP — `qk pipe` / `qk pipe-worker`). Greedy output is token-exact vs the
  unsplit engine at every tested boundary; two stages of a 20-layer half
  hold ~8.5 GB each vs ~15.4 GB whole. Foundation for serving models larger
  than one GPU (e.g. tron + midnight once the Metal port lands).
- **`deploy/`** — k8s deployment sharing one GPU with a llama.cpp fallback
  backend (`switch.sh qk|gemma`), image build script, and an
  orphaned-GPU-process reaper. See `deploy/README.md`.

Model facts (`qwen35moe`): n_embd 2048, 40 blocks (30 gated-DeltaNet + 10
full-attention), 256 experts top-8 + 1 shared, vocab 248,320. The UD-Q3_K_M
GGUF actually contains Q8_0 / Q6_K / IQ4_XS / IQ3_XXS tensors — no Q3_K.
The engine is deliberately hard-wired to this architecture; that
specialization is where the speed comes from.

## Build & run

```bash
/usr/bin/cmake -B build      # NB: bare `cmake`/`ninja` on this box are broken pip shims
/usr/bin/cmake --build build -j

# serve (Anthropic + llama.cpp-compatible HTTP)
cd server && cargo build --release
QK_SHADER_DIR=../build/shaders ./target/release/server \
    --model /path/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf \
    --engine-lib ../build/libqk.so --port 8080 --slots 2 --ctx 16384 --chunk 8
```

Engine CLI (`./build/qk …`) — benchmarks and correctness harnesses:

```
qk                       # synthetic kernel suite (f16, q8_0, q6_k, iq4_xs, iq3_xxs)
qk gguf <tensor>         # GEMV on real weights          qk moe|block|ablock [layer]
qk token <ids> <n> [tmax] [batch]   # end-to-end greedy generation
qk warm <ids> <n>        # prefix-cache TTFT demo        qk serve-test <ids> <n> [slots]
qk prefillcmp|prefillbench|prefilldecode   # batched-prefill exactness / timing / handoff
qk verify <ids> <n> [K,..]   # spec-decode verify rounds (oracle draft): exactness + c(K)
qk pipe <ids> <n> [split] [tmax] [host:port]   # pipeline-split parity/timing (2 stages)
qk pipe-worker <port> [a:b] [tmax]             # serve one stage over TCP
qk list [filter]         # tensors in the GGUF
```

Env knobs:

| var | meaning |
|---|---|
| `QK_GGUF`, `QK_DEVICE`, `QK_SHADER_DIR` | model path, Vulkan device index, SPIR-V dir |
| | (`QK_GGUF`/`--model` also accept the first shard of an llama.cpp-style split model, `…-00001-of-NNNNN.gguf`) |
| `QK_FORK=1` | prefix cache: same-prefix requests restore instead of re-prefilling |
| `QK_LAYERS=a:b` | pipeline split: this engine owns layers `[a,b)` only, driven via `qk_stage_run` |
| `QK_SPEC=1` | prompt-lookup speculative decoding (exact output; ~1.5× on echo-heavy generation) |
| `QK_SPEC_K`, `QK_SPEC_L`, `QK_SPEC_LOG=1` | verify width (8), trigger n-gram length (6), per-request `[spec]` stats |
| `QK_PCACHE`, `QK_PCACHE_LOG=1` | prefix-cache LRU depth (default 3), per-request stats |
| `QK_NO_BATCH=1` | force serial prefill (correctness reference) |
| `QK_MAXB` | batch-prefill chunk width (default 128; buffers scale with it) |
| `QK_SUBMIT_LAYERS`, `QK_ATTN_BUDGET` | submit granularity / attention tile budget (amdgpu ~10 s ring-timeout guards) |
| `QK_MAX_TOOL_CHARS` | server: cap on a single tool_result before context-fit trimming |

## Docs

- [`bench/README.md`](bench/README.md) — the CLI-path benchmark vs llama.cpp
  (harness + raw per-call results)
- [`docs/server-spec.md`](docs/server-spec.md) — serving-layer spec
- [`docs/speculative-decoding.md`](docs/speculative-decoding.md) — spec-decode
  landscape assessment (llama.cpp era) and MoE verify economics
- [`docs/spec-decode-qk-plan.md`](docs/spec-decode-qk-plan.md) — design +
  implementation plan for speculation in this engine
- [`deploy/README.md`](deploy/README.md) — single-GPU switchable deployment

## Notes

- Kernel-bench numbers want a quiet GPU: scale the serving pod down first
  (`kubectl scale deploy qk-server -n gemma --replicas=0`, and back to 1
  after). The engine needs ~15.4 GB resident for end-to-end generation.
- Development history — the M1→M6 bring-up (GEMV → quantized kernels → fused
  MoE → DeltaNet blocks → attention → end-to-end), the optimization arc
  33.3 → 5.82 ms/token, and measured negative results (IQ3 repacking, RoPE
  precompute) — lives in this file's git history and the commit log.
- Reference shaders: llama.cpp's Vulkan backend
  (`ggml/src/ggml-vulkan/vulkan-shaders/`).
