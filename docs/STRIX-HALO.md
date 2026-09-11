# Strix Halo native Flash Next port

Status, 2026-09-10: full-model F32 logit/greedy/reset parity and real native
dual-GPU HTTP/Claude tests pass. MTP, batched prefill and prefix snapshots
remain unimplemented. Production was stopped with user approval for testing;
the native trial now serves port 8091, with the old configuration preserved.
This is a runtime-only test cutover, not a persistent deployment. No Halogen
binary has been installed or executed; its checkpoint is a reference download,
not a format that this engine currently accepts.

## Current target

The installed `Qwen3.8-Flash-Next-Uncensored-Q5_K_M` is a three-shard GGUF with
architecture `qwen4exp`, not the older `qwen3next` or `qwen35moe` engine target:

| Property | Installed GGUF |
| --- | --- |
| Layers / embedding width | 48 / 2560 |
| Experts / selected / FF width | 512 / 10 / 640 |
| Hyper-connection streams / low rank | 4 / 320 |
| Residual width between layers | 10240 floats |
| Linear-attention K/V heads / state width | 16 / 48 / 128 |
| Full-attention Q/KV heads / head width | 24 / 2 / 256 |
| PLE | layer 1, 16 heads of 160 elements, n-gram 3, convolution 4 |
| PLE table | Q5_1, 320001536 rows, 35.763 GiB |
| Total recognized encoded tensors | 124.886 GiB |

Although indexer tensors and top-k metadata exist, all 48
`attention.compress_ratios` entries are zero in this checkpoint. The current
reference graph bypasses QSA when the ratio is zero. The initial port can use
dense attention; it must reject nonzero ratios until QSA is implemented.

Important numerical differences: GDN's output gate is sigmoid, not SiLU; its
L2 normalization is `1/sqrt(sum(x*x) + eps)`, not the older max-clamped norm.
HC RMSNorm is per residual stream, not over the complete 10240-wide vector.
PLE hashes use wrapping uint64 products and EOS-cut token history.

## Implemented and checked

- Native CPU/GPU Q5_K and Q5_1 layouts, packed scale/min/high-bit decoding.
- Halo-only GEMV thread choices for the measured 640x2560 and 320x10240 Q5_K
  shapes; explicit `QK_TPR` still overrides. XTX defaults are unchanged.
- Subgroup-shuffle skinny-row reductions avoid workgroup shared-memory barriers.
- Native HC per-stream RMSNorm, sigmoid/mean collapse, and residual injection,
  with token batching. These elementwise operators are not the whole HC module.
- GDN sigmoid-gated RMSNorm and PLE signed-square-root dot gating, checked at
  the actual 48x128 GDN shape as well as HC shapes and zero-input streams.
- Complete native Q5_K/Q8_0 MoE operator chain, including shared experts and
  deterministic top-10 routing. One-layer weight uploads use a bounded 16 MiB
  staging buffer instead of a staging allocation as large as an expert tensor.
- PLE row hashing, disk-backed CPU row gathering, bounded four-way cache
  (2686976 bytes at the default 4096 rows for this model). Actual-table smoke:
  80 misses, 112 hits, finite values and exact cached/uncached equality.
- GGUF float/uint64-array metadata, bounds checks, mapped-file lifetime cleanup,
  quantized payload sizing. Native Flash serving is opt-in and rejects
  unsupported shapes. Dense GEMV on the enormous sparse PLE table is rejected.
- `QK_DEVICE_NAME` selects a unique device-name substring; PCI selection retains
  priority. Empty, missing, or ambiguous names fail instead of guessing a GPU.

Validation: three CPU CTest suites; 35 Q5 configurations per GPU; three HC
operations across six shapes per GPU; actual expert and HC weight tests on Halo
and XTX. This is operator-level correctness, **not token/logit parity**.

## Preliminary kernel measurements

Device: Radeon 8060S / RADV STRIX_HALO, vendor 1002/device 1586. Both this Halo
and NAVI31 expose subgroup size 64 and 16x16x16 KHR cooperative matrices with
F16 inputs and F32 accumulation. Halo shared-memory limit: 65536 bytes.

Randomized geometry sweep: three repetitions per shape/TPR, 300 timed dispatches,
64 MiB rotating weight footprint, correctness checked first. The checkpoint
download was running during the sweep, so these are provisional kernel timings,
not controlled end-to-end measurements or proof of DRAM bandwidth saturation.

| Q5 shape (M x K) | TPR | Before shuffle (us) | After shuffle (us) |
| --- | --- | --- | --- |
| Q5_1 4096 x 160 | 8 | 4.1 | 3.8 |
| Q5_1 10240 x 320 | 16 | 13.7 | 13.3 |
| Q5_K 640 x 2560 | 64 | 8.4 | 8.4 |
| Q5_K 320 x 10240 | 128 | 14.1 | 14.1 |

Q5_K's previous derived TPRs were 128 and 256 for the last two shapes (8.8 and
14.8 us in the baseline sweep). Changing launch geometry and changing reduction
code are separate optimizations; do not conflate them. Several slower geometries
improved more, but they are not the selected defaults.

Actual-weight Halo spot checks, 64 MiB rotation: expert-0 gate 8.7 us, HC attention
down 14.3 us, HC attention up 13.6 us; all within the CPU-reference tolerance.
Raw synthetic sweep data: `bench/results-halo-q5-{baseline,shuffle}.jsonl`.
GPU smoke records: `bench/results-halo-qwen4-smoke.jsonl`.

### Native MoE sweep and rejected host-import path

The actual layer-0 expert chain passes the CPU reference for all 27 randomized
geometry trials (three repetitions of nine gate/down workgroup combinations).
The median with 128/128 threads is 235.9 us, versus 259.9 us with the initial
64/256 choice: about 9.2% lower operator time. This becomes a Halo-only default;
`QK_MOE_Q5_WG` and `QK_MOE_Q8_WG` retain 64/128/256 overrides. Layers 0, 17 and 47
pass with different random inputs on both devices, and all-zero input passes
including deterministic tie routing. The active footprint is reused between
iterations; these results are not full-model throughput or a DRAM-only benchmark.
Raw results: `bench/results-halo-moe-q5.jsonl` and
`bench/results-halo-moe-validation.jsonl`.

The expanded GPU harness checks 70 Q5 cases plus five elementwise operators
over eight shapes on each of two devices (150 operator checks). Records:
`bench/results-halo-qwen4-expanded.jsonl`. The unchanged Rust server also passes
all 38 tests, including stub-backed Anthropic and split-stage tests. These are
protocol regressions, not proof that Flash Next is serving through that server.

`QK_IMPORT_WEIGHTS=1` in `qk gguf` probes direct immutable GGUF host import.
The local driver rejects that allocation with VkResult -13; there is no silent
copy fallback. Mode 2 makes one aligned host copy and shares it with Vulkan;
that is **not** zero-copy file mapping. It passes correctness but is slower and
variable on this node (three-trial medians 23.8/39.0/49.4 us for the expert gate,
HC down and HC up, versus 4.7/7.6/6.6 us for device-local hot-cache tests).
Both import modes remain off by default. They require `QK_COLD_MIB=0`; import
benchmarks do not clone the mapped weights. Raw data: `bench/results-halo-import.jsonl`.
The test allocator requires host-visible/coherent memory and keeps the imported
payload alive until Vulkan releases it, as required by the
[Vulkan host-import contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkImportMemoryHostPointerInfoEXT.html).

## Native prefix validation

`qk qwen4-prefix LAST_LAYER TOKEN REFERENCE.f32 STEPS` now runs the first one
through four layers of the installed checkpoint using only this engine's
Vulkan kernels. It includes complete HC modules, GDN recurrence with the
Qwen4-specific norm/gate semantics, dense full attention with partial text
RoPE, MoE, and PLE's projections, norms, gate and dilated convolution.

PLE uses a nine-frame circular history instead of shifting 10240 channels per
token. The 35.763 GiB table stays disk-backed; only the selected rows are
dequantized and uploaded. Reset clears both recurrent/convolution state and
token history. The harness is limited to 32 tokens and 8 GiB of weights;
the four-layer fixture holds 7.867 GiB of weights with 16 MiB staging. It is
separate from the opt-in `qk_open` adapter described below. These earlier prefix
checks did not change production.

Actual-model checks (all require per-frame relative RMS < 1e-5):

| Native device / layers | Tokens | Relative RMS | Worst-frame relative RMS |
| --- | --- | --- | --- |
| Halo / first 1 | 198..205 | 3.18e-7 | 3.51e-7 |
| XTX / first 1 | 198..205 | 3.32e-7 | 3.81e-7 |
| Halo / first 2 | 198..213 | 3.59e-7 | 5.59e-7 |
| Halo / first 2, includes EOS | 248042..248057 | 4.00e-7 | 6.15e-7 |
| Halo / first 4 | 198..213 | 4.57e-7 | 6.21e-7 |

Every case reproduces its first frame exactly after resetting a used graph.
These are intermediate-activation checks, **not full-model logits or tokens**.
Records: `bench/results-halo-qwen4-prefix.jsonl`.

The optional `flash-reference` test executable links to the separately built
llama.cpp fork at `2dff8596dcb7bdf765d24d44e1155d51f04c82b7`. It is never
linked into this runtime or server. Only the requested prefix's weights are
placed on Halo; remaining weights are CPU-mapped. A graph callback captures
the requested layer and stops that forward pass. Reference jobs on this node
used user-systemd memory limits (6 GiB high / 8 GiB max).

For a like-for-like F32 oracle, disable Vulkan MMVQ and F16, and disable flash
attention (the test executable does the latter). The default CPU reference
quantizes matrix inputs; the default Vulkan path also uses reduced-precision
inputs/attention. Initial first-layer CPU relative RMS was 0.0116, and the
four-layer flash-attention comparison was 0.000144. Using the F32 reference
resolved these differences; the tolerance was tightened, not loosened. These
reference settings are for accuracy comparisons, not production recommendations.

Build and run the optional reference on an idle, adequately sized GPU:

```bash
cmake -S . -B build-halo -DCMAKE_BUILD_TYPE=Release \
  -DQK_LLAMA_REFERENCE_ROOT=/path/to/llama-qwen-next-b10685
cmake --build build-halo --target qk flash-reference -j4

QK_REFERENCE_GPU_PREFIX=3 GGML_VK_DISABLE_MMVQ=1 GGML_VK_DISABLE_F16=1 \
  build-halo/flash-reference MODEL.gguf reference.f32 198 l_last-3 16
python3 tests/gpu_qwen4_prefix.py MODEL.gguf reference.f32 --last-layer 3 --steps 16
```

`QK_REFERENCE_DUMP=1` captures layer-0 intermediates, while
`QK_LAYER_DUMP=/path/to/prefix` captures native intermediates by token/layer.
Do not run large prefix jobs on a nearly full GPU; an idle request slot does
not mean sufficient free VRAM. During the earlier production-backed checks,
XTX had room for the one-layer test, not the four-layer test.

### Remaining expert format and placement

Twenty-four layers use Q5_1 expert down-projections rather than Q8_0. New native
routed/shared Q5_1 down kernels pass independent CPU checks on layers 6, 7, 30
and 40 plus zero-input routing on both GPUs. Two randomized sweeps each cover
27 configurations. In the longer 3000-iteration sweep, the 64/128 gate/down
choice has a 193.8 us median versus 211.9 us for 128/128. Timings vary with
cache/clocks; this selects a provisional Halo Q5_1-path default, not a claim
of full-model speedup. Q8_0-path defaults remain 128/128. Records:
`bench/results-halo-moe-q51-{initial,steady,validation}.jsonl`.

The audit now reports weight-only split sizes. A candidate Halo-first 0:37,
XTX-tail 37:48 layout is 67.394 / 21.322 GiB, excluding mapped embeddings,
KV and scratch. The XTX owns the large vocabulary head, and the boundary
carries all 10240 residual floats (40 KiB per token). This complete layout
has now been loaded and tested. These are weight sizes, not free VRAM;
any future MTP head still needs its own memory plan.

## Experimental serving adapter

`QK_NATIVE_FLASH=1` enables the native graph in `libqk`; without it, `qk_open`
rejects this architecture before GPU initialization. The graph has all 48
layers, both expert-down formats and the final HC/vocabulary head implemented.
The complete split passes the full-model and real HTTP checks below. It remains
opt-in because performance features and broader long-context testing are unfinished.

The adapter requires one sequence, `QK_LAYERS=a:b`, and the Rust local/split
driver. Stage boundaries carry 10240 floats, not the older 2048. Invalid tokens,
slots, positions, context bounds and nonfinite residual input are rejected.
Snapshot capacity is zero; MTP and batched prefill are not implemented. Ordinary
prefill currently loops over tokens. It is not the old engine's fast GEMM path.

Native per-token input copies, compute and readback share one queue submission;
the PLE stage previously needed four. KV reset changes the logical length and
does not clear unreachable cache rows; recurrence and convolution state still
clear. The four-layer oracle/reset test passes unchanged after both changes.
The non-reset command sequence is now reusable too: PLE reads its position
from a buffer, as attention already did. Descriptor sets and dispatches need
not be rebuilt every token. `QK_FLASH_REPLAY=0` retains the original path for
A/B tests. Debug taps disable replay. Replay and re-recording produced identical
full logits for all 16 tested positions.

Actual C-ABI stage checks, each over 16 frames:

| Stage | Relative RMS vs F32 oracle | Chunk/reset result |
| --- | --- | --- |
| Halo 0:2 | 3.59e-7 | 16 vs 5+1+7+3 frames bit-exact |
| Halo 2:4, oracle residual input | 3.06e-7 | 16 vs 5+1+7+3 frames bit-exact |

Run `tests/gpu_qwen4_stage.py MODEL.gguf reference-prefix.f32`; for the second
stage use `reference-prefix4-f32.f32 --layers 2:4 --input reference-prefix.f32`.
These exercise the actual `libqk` ABI, not the stub, but they do not include
the vocabulary head or TCP transport. Records: `bench/results-halo-qwen4-stage.jsonl`.

Serving allocations require Vulkan memory-budget reporting and reject host-memory
spill. The 0:37 / 32768-context Halo stage estimates 68.855 GiB including KV,
recurrence and scratch allowance. Its attempted open correctly refused the
occupied device before uploading weights. After the production model was stopped,
the complete native model loaded successfully. A dedicated test window is required;
an idle production request slot alone does not free its GPU allocations.

Flash uses its actual GGUF Jinja template, including string-method compatibility,
JSON filters, open `<think>` cue and `xhigh` effort by default. Override effort
with `QK_REASONING_EFFORT=low|medium|high|xhigh`. Template errors do not silently
fall back to an older model format. The Anthropic adapter renders structured
tool history through this template and parses both Flash XML function/parameter
calls and legacy JSON calls. It preserves string arguments, checks declared
primitive types, rejects malformed/duplicate XML parameters and does not interpret
XML entities. Open reasoning is discarded with a bounded parser buffer, including
fragmented closing tags. No generated tools are executed by this parser.

The Rust suite has 44 tests; the actual Flash tokenizer/template/history checks
were enabled with `QK_FLASH_GGUF` and passed. Other API/split engine integration
tests remain stub-backed. The older real-vocabulary fixture is optional and was
not enabled in this run. Native workers bind loopback by default; `QK_PIPE_HOST`
can explicitly select another IPv4 interface, which needs firewall protection.

### Full-model and real API validation

The pinned F32 oracle was extended to place layers 0:37 on Halo and 37:48 plus
the head on XTX, leaving token/PLE embeddings CPU-mapped. It captured all layer
boundaries and 248320 logits per position for input IDs 198..213. Across the
16 full native forwards, worst logit relative RMS was 1.253e-6 and maximum
absolute error was 2.003e-5. Every greedy choice matched; top-20 extraction
matched the native full logits, and resetting reproduced the first logit row
bit-for-bit. Repeated at 8192 context capacity after adding command replay:
same results, plus replay/re-record bit equality. This is a short sequence
within that capacity, not an 8192-token parity test.

Records: `bench/results-halo-qwen4-full.jsonl` and
`bench/results-halo-qwen4-replay.jsonl`. The accuracy loop's per-position time
includes Python boundary checking and is **not** decode throughput. A separate
hot-cache A/B loop omits reference math and logit readback: three 16-token runs
per mode gave medians 26.16 tok/s re-recorded and 27.51 tok/s replayed. The small
sample and overlapping timing ranges make this a preliminary ~5% gain, not a
robust throughput guarantee or a long-context benchmark.

The actual Rust server and TCP worker also passed teacher-forced token checks,
normal Claude replies, XML tool calls, tool-result round trips, Anthropic SSE,
reset after disconnect, two queued requests without output mixing, and invalid
token rejection. Tests use a synthetic `echo` tool and never execute a tool.
The initial 35-token prompt / 96-token streamed counting run measured 1.62 s
to first token and 22.69 streamed tokens/s thereafter, before command replay.
Use client-side timings from the harness: the legacy server's `predicted_ms`
field includes prefill and its `prompt_ms` field is a placeholder, so those
reported fields are not an independent decode benchmark.

To reproduce in a dedicated maintenance window:

```bash
QK_REFERENCE_FULL_SPLIT=37 QK_REFERENCE_LAYERS=1 \
  GGML_VK_DISABLE_MMVQ=1 GGML_VK_DISABLE_F16=1 \
  build-halo/flash-reference MODEL.gguf reference-full.f32 198 result_output 16
python3 tests/gpu_qwen4_full.py MODEL.gguf reference-full.f32 --ctx 8192 --compare-replay
cd server
cargo build --release --locked --target-dir ../build-halo/rust -j4
cd ..
# Separate terminals; neither command stops production automatically:
bash deploy/run-native-flash-trial.sh worker MODEL.gguf 8192
bash deploy/run-native-flash-trial.sh server MODEL.gguf 8192
python3 tests/native_flash_http.py
```

Trial ports are loopback-only 8194 (HTTP) and 8195 (worker). The trial sets
`QK_PREFILL_CHUNK=16` to bound serial-prefill cancellation stalls; override it
up to 128 when measuring throughput. This is chunking, not batched GEMM prefill.

The same native split also loaded at 32768 context capacity and passed the
real API suite. This proves allocation and short-request handling at that
capacity, not long-context quality or full-window throughput. Repeated short
decode measurements ranged from 23.27 to 26.62 tok/s. The separate 512-token
prompt / one-output-token test took 15.45 s (~33.14 prompt tokens/s), including
request overhead and one decode. Records: `bench/results-halo-native-http32.jsonl`
and `bench/results-halo-native-bench32.jsonl`.

The disconnect test found and fixed a server bug: `flush_slot` previously
noticed a closed channel only while sending pending output, but prefill has
no output. It now checks channel closure even when its pending queue is empty.
The 768-token abandoned prefill followed by a one-token request went from
23.59 s to 0.812 s at chunk size 16. The older HTTP record labels the eventual
reset as PASS; it did not yet enforce a cancellation-latency bound. The updated
harness requires <5 s. Two unit tests cover empty-pending disconnect and
backpressure retention. Record: `bench/results-halo-native-cancel-fixed.jsonl`.
This latency is measured directly against port 8194; a buffering frontend can
delay forwarding the disconnect and must be measured separately.

### Fast reference backend: quality regression, not a clean speed baseline

The existing llama.cpp server was restarted in isolation with its original
F16 KV / flash-attention / MTP settings at 32768 context capacity, on the same
GGUF and two GPUs. Its 35-token counting prompt produced malformed output
(missing numbers, repeated commas). A fresh process with no draft model and
`--spec-type none` failed the same check, emitting repeated reasoning markers
and ending after ten tokens. Therefore the fault is not isolated to MTP; the
root cause remains unresolved. These are observed failures of this local fast
configuration, not a claim that every configuration of the fork is defective.
The conservative F32, non-flash-attention oracle above passed independently.

The 512-token / one-output-token request took 1.058 s with MTP and 0.979 s
without it, versus 15.45 s in the native serial-prefill implementation. That
large prefill gap is the next major optimization target. The MTP run streamed
29.83 tokens/s, but those tokens failed the quality check; this is **not** a
quality-equivalent speed comparison. The no-MTP run was too short for a useful
decode comparison. Raw output and client timings are preserved in
`bench/results-halo-reference-mtp.jsonl` and
`bench/results-halo-reference-nospec.jsonl`. Benchmark-only mode records failures
without asserting quality; the full API suite requires a coherent counting prefix.

### Max maintenance trial

The native 32768-capacity split uses transient user-systemd units
`qwen-native-flash-worker32` (XTX, loopback 8195) and
`qwen-native-flash-server32` (Halo HTTP, loopback 8194). A separate
`qwen-native-flash-router` fronts the existing port 8091 using this node's
already-installed `deploy/prefill-router.py` from the original worktree, with
both upstream URLs set to port 8194 and handoff thresholds disabled. That
node-local router is not part of this branch. It allows loopback and
192.168.0.0/24, rejects browser-origin requests, and retains body/concurrency
bounds. This is a trusted-LAN endpoint, not an authenticated public service.
The unchanged Claude proxy on loopback 8092 forwards to 8091.
It retains its existing behavior of stripping top-level system prompts; that
behavior was not changed by this port.

The complete API suite, except the direct-engine cancellation timing, passed
through port 8091 after cutover: teacher-forced tokens, coherent counting SSE,
Claude text and XML tools, tool-result history, Claude SSE, queued isolation,
and invalid-token rejection. Record:
`bench/results-halo-native-gateway32.jsonl`. Browser-origin health requests
were rejected with 403 and the unexposed `/slots` path with 404. The existing
gateway only notices a disconnected client when forwarding output, so its
prefill cancellation delay is not covered by the 0.812-second engine result.
Streaming `READY` and a structured `echo` tool call also passed through the
unchanged Claude proxy on port 8092; see
`bench/results-halo-native-claude-proxy.jsonl`.

Worker/head memory limits are 20/24 GiB with 512 MiB swap limits per service;
GPU device-local allocations have a separate budget checked by the engine.
The gateway is capped at 1 GiB. All three trial units use `NoNewPrivileges`
and disable core dumps. The old `qwen-kernel-next` and
`qwen-kernel-next-router` units remain stopped, with their files unchanged.
Do not start the old router to front the native engine: it requires the old
GPU worker and would load a competing model.

These trial units are **runtime-only**. They do not establish reboot persistence;
the old enabled startup configuration is still present and has the quality
regression described above. Stop all three native trial units before loading
another complete model. Permanent cutover and any rollback require an explicit
backend choice and another health/quality check, not merely a healthy HTTP port.

## Reproduce

```bash
cmake -S . -B build-halo -DCMAKE_BUILD_TYPE=Release
cmake --build build-halo -j4
ctest --test-dir build-halo --output-on-failure

QK_DEVICE_NAME=STRIX_HALO build-halo/qk counters
python3 tests/gpu_qwen4_smoke.py --device STRIX_HALO
python3 tests/gpu_qwen4_smoke.py --device NAVI31
python3 bench/halo_q5_sweep.py
python3 bench/halo_moe_sweep.py /path/to/model-00001-of-00003.gguf

build-halo/qk-model-audit /path/to/model-00001-of-00003.gguf
build-halo/qk-ple-smoke /path/to/model-00001-of-00003.gguf
QK_DEVICE_NAME=STRIX_HALO QK_GGUF=/path/to/model-00001-of-00003.gguf \
  build-halo/qk gguf blk.0.hc_attn_down.weight 300
```

GPU scripts check the existing router's idle status before each job. On a
dedicated test machine without that router, use `--status-url ''` only after
ensuring there are no serving requests. A status check is not an exclusive GPU
reservation; final benchmarks need a dedicated window with the server drained.

## Remaining implementation and performance gates

1. Expand full-model parity coverage to longer contexts, more prompts and
   autoregressive sequences, beyond the initial 16-position F32 comparison.
2. Continue multi-turn, reset, EOS, cancellation and chunk-boundary regression
   checks while implementing each performance feature.
3. Establish a memory plan from measured device budgets. Do not upload the PLE
   table, blindly reserve 262144 contexts, or confuse system RAM with Vulkan's
   advertised device-local budget. The direct-host-import experiment above did
   not produce a usable fast path; retain device-local weights and bounded staging.
4. Preserve the native Anthropic API and pipeline split. Transfer all four HC
   streams at stage boundaries; do not reuse the old model's hidden-vector size.
5. Benchmark and fuse decode operations; add cooperative-matrix batched prefill,
   grouped expert execution, then the actual MTP sidecar and rollback state.
   Prompt lookup is not equivalent to the model's learned MTP head.
6. Only switch the serving endpoint after correctness, capacity, tool-call,
   cancellation and repeated-request checks pass, with the old backend available.

## Reference checkpoint download

`deploy/fetch-halogen-flash.sh MODEL_DIRECTORY` downloads a pinned Hugging Face
snapshot of `peonist-ai/halogen-qwen3.8-flash-next`, revision
`214a45c7106f515faf3fb72db0cf9a1bf67bfd77`. It selects the base, quality overlay,
vision weights and tokenizer assets; it does not download the optional speed
overlay or run any remote code. The three weight files and tokenizer JSON are
checked against `deploy/halogen-flash.sha256`.

On the development node, the transfer completed and a separate bounded-memory
SHA-256 verification passed for all four manifest entries at 19:35 MDT on
2026-09-10. Total on-disk download is approximately 119 GiB. The initial transfer
unit exited before verification, so the successful separate check is the
verification evidence—not the transfer unit's exit status.
