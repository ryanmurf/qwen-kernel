# Strix Halo native Flash Next port

Status, 2026-09-11 (15:05 MDT): the native exact-tier split is serving port
8091 again after the reboot (commit 1b1e0a0 plus this update). Full-model F32
logit/greedy/reset parity, batched-prefill parity and the real dual-GPU
HTTP/Claude tests pass. Measured through the HTTP path with the lookup tables
warm and the XTX at DPM level `high`: decode 34.3-35.1 tok/s, first token
0.55 s, 512 distinct-token prompt 2.55 s (201 tok/s), 2048 tokens 10.3 s.
With the tables cold (`QK_PLE_PREFETCH=0`, the script default) the same build
measured 27.7-32.8 tok/s and 6.75 s for 512 tokens. The cooperative-matrix
tier stays opt-in and only partially validated. MTP and prefix snapshots
remain unimplemented. The trial units are runtime-only (not enabled at boot);
the old boot stack was stopped by the operator after the reboot and its unit
files are unchanged. No Halogen binary has been installed or executed.

Recovery on 2026-09-11 followed the incident notes below: safety fixes first
(fail-closed prefetch guard, validated floor, bounded tier harness; commit
263f58a), then the 4-layer prefix oracle and batch checks, then the XTX
worker and the Halo server loaded one at a time with prefetch off and
VRAM/GTT/MemAvailable checked before and after each load (Halo GTT 69.1 GiB,
no driver errors), then the API suites on 8194/8091/8092. Table warming was
then enabled by restarting only the Halo server (worker still loaded, no
other GPU work) under a 52G/58G unit: 36.170 GiB touched in 87 s, MemFree
42 -> 4.9 GiB, MemAvailable steady near 46 GiB, cgroup 43 GiB, no driver
errors. The floor guard uses MemAvailable, which counts the warmed cache as
reclaimable, so it only catches gross exhaustion; the operating rule that
prevents the incident is that no other GPU load may start while a stage
holds the resident tables, and every load is preceded by a drain check
(GTT/VRAM back near idle, tables paged out by the closing stage).

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

## Batched prefill and decode campaign (2026-09-11)

All numbers below are client-side timings from `tests/native_flash_http.py`
against the 32768-context native split (Halo layers 0:37 on 8194, XTX 37:48
plus head via the loopback worker), measured in one session on the same
hardware and checkpoint. The counting prompt has 35 tokens and streams 96;
"diverse" prefill prompts use 512 or 2048 distinct token ids, which defeats the
PLE row cache the way real text does. Raw records:
`bench/results-halo-native-batched-*.jsonl` and
`bench/results-halo-native-decodeopt-*.jsonl`.

| Build | Decode tok/s | First token | 512-token prefill | 2048-token prefill |
| --- | --- | --- | --- | --- |
| Serial baseline (4314808) | 26.4 | 1.08 s | 15.95 s (32 tok/s) | not measured |
| Batched prefill (3d87ba4) | 23.7-26.8 | 0.58-0.87 s | 2.53 s (202 tok/s) | 10.08 s (203 tok/s) |
| Decode campaign (this commit) | 35.5 (36.0 via 8091) | 0.54 s | 2.45 s uniform, 2.48 s diverse (206-209 tok/s) | 9.95 s diverse (206 tok/s) |

The complete API suite passed on 8194 (cancellation 3.29 s at the 512-token
chunk), on 8091 (`--skip-cancel`) and through the 8092 proxy (streaming READY
1.2 s, echo tool call 5.0 s versus 12.6 s before) after each of the two commits.

### Batched prefill

Multi-token `qk_stage_run` calls now run `Qwen4Graph::forwardBatch`; single
positions keep the replayable serial path and continue exactly from a batch.
The batched graph reuses every activation buffer with `QK_FLASH_BATCH` rows
(default 512, rounded to 64-row tiles, halved automatically while the device
budget is short; 0 disables batching). Kernels, all F32 accumulation:

- `qwen4_gemm_{q5k,q6k,q8_0,q5_1}`: 128x64 tiled GEMM with per-thread 32-block
  dequantization into LDS (Y[N][M] = X[N][K] W^T), strided/offset activation
  rows so slices of wider buffers are usable; skinny outputs (M < 128) keep the
  z-batched GEMV. Measured about 3.7 TFLOPS scalar F32 on Halo.
- `qwen4_gdn_conv_batch` / `qwen4_gdn_step_batch`: causal conv seeded from the
  serial conv window and the register-carried delta rule with the Flash
  sigmoid gate and shared-gamma RMS, one workgroup per head across the chunk.
- `qwen4_ple_conv_batch`: dilated conv over the nine-frame circular history,
  one thread per channel walking the chunk, so a batch leaves the history
  exactly as serial decoding would.
- `fa_prep_batch` / `fa_attn_batch` (existing kernels) for full attention,
  query-tiled under `QK_ATTN_BUDGET` (default 2M query*key pairs).
- MoE: `moe_logits_gemm` (32-row multiples) or `moe_logits`, `moe_select_256`,
  `moe_group_pairs`, then `qwen4_moe_gateup_tiled` (Q5_K) and
  `qwen4_moe_down_tiled_{q8,q51}`: one workgroup per (expert, 128-row tile)
  walks that expert's pairs in 16-token groups with LDS-dequantized weights,
  so expert weights are read once per 16 tokens. Shared experts run as dense
  GEMMs plus `qwen4_silu_mul`; `qwen4_moe_combine` folds routed partials in
  slot order. `QK_MOE_GROUPED=pairs` keeps the per-pair reduction kernels
  (44 ms per layer at 512 tokens versus 18.7 ms tiled; rejected default).
- Head: 64-row output tiles with `qwen4_argmax` per position; the final
  position's full logits stay available to `qk_stage_logits`/`qk_stage_topk`.

Validation: `qk qwen4-batch 3 198 N` compares serial and batched prefix
outputs for whole, mixed (5+1+7+3) and batch-then-serial chunking at 16 to
512 tokens (worst frame relative RMS 5e-7 to 2.5e-6, all PASS);
`tests/gpu_qwen4_batch.py` repeats this on the full dual-GPU split against the
F32 oracle (every greedy id matches over 16 positions, boundary rows within
2.5e-6, final logits within 1.3e-6, reset bit-exact). Records:
`bench/results-halo-native-batch-parity.jsonl` and the decodeopt parity file.
Prefill GPU time at 512 tokens is about 57 ms per Halo layer (2.1 s per chunk),
dominated by expert gate/up (18.7 ms), dense GEMMs (about 24 ms) and expert
down (8.5 ms); the Rust head now accepts `QK_PREFILL_CHUNK` up to 512 (default
frame 128 unchanged) and the trial script sets 512.

### Decode findings and changes

`QK_FLASH_PROFILE=1` (per-shader GPU time, `2` for every dispatch) and
`QK_FLASH_TIMING=1` (host phases of the serial forward) were added. On Halo,
the large projections already stream at 205-225 GB/s (the measured memory
roofline), so the GPU spends about 24 ms per token in 37 layers; the two
largest costs were outside the kernels:

- PLE row gathering took 8.5 ms per token: the 35.763 GiB table is
  disk-mapped and its rows are hash-random, so most lookups were NVMe page
  faults under the 16 GiB service memory limit. The first stage now releases
  the uploaded weights' page cache (`MADV_PAGEOUT`, 67.4 GiB) and reads ahead
  and touches the PLE table and the token embedding (36.17 GiB, about 60-110 s
  in the background after load; `QK_PLE_PREFETCH=0` disables it). A diverse
  64-token prefill fell from 1.06 s to 0.69 s and the per-token gather to well
  under 1 ms once resident.
- The XTX head stage spent 3.8 ms per token copying logits out of uncached
  write-combined staging memory; the staging buffer is now host-cached
  (0.13 ms), which also speeds batched hidden-row readback.
- `gemv_q5_k` reads blocks as 32-bit words (2-12% faster per shape; the old
  kernel remains as `gemv_q5_k_v1.comp`, `QK_Q5K_SHADER` selects it in `qk
  q5_k`). Q6_K threads per row now derive from 32-wide units, with 64 lanes on
  NAVI31 for 2560-wide rows (head GEMV 714 -> 596 us in isolation).
- `qwen4_gdn_step`: 256-thread delta step (two lanes per state row);
  `QK_GDN_STEP=v1` keeps `dn_step_gate`. Independent projections (GDN qkv/gate/
  alpha/beta, attention q/k/v, HC up/inject, PLE key/value, routed/shared
  gate-up) no longer drain the GPU between them.
- Rejected after measurement: 16-bit-word Q6_K and routed Q8_0 kernels (596 ->
  725 us and 241 -> 254 us; the byte-addressed originals stay), and forcing the
  Halo iGPU performance level (no change).
- XTX DPM: `power_dpm_force_performance_level=high` on 0000:68:00.0 cut its
  stage from 6.4 to 4.2 ms per token (33.2 -> 34.8 tok/s in the
  `tests/gpu_qwen4_timing.py` loop; four alternating rounds). It is a stock
  DPM level, set at runtime for this trial, reversible with `auto`;
  `deploy/91-amdgpu-navi31-perf-high.rules` is an optional, uninstalled rule.

Serial full-model parity and replay bit-equality were re-run after these
changes (`bench/results-halo-native-decodeopt-parity.jsonl`); the hot
16-token loop measured 34-35 tok/s (26-27.5 before).

### Memory plan and runtime state

Transient units (recreate with `systemd-run` after a reboot; nothing enabled):
`qwen-native-flash-worker32` (XTX, MemoryHigh 12G / MemoryMax 20G),
`qwen-native-flash-server32` (Halo, MemoryHigh 52G / MemoryMax 58G so the
36.17 GiB of lookup tables stay resident; MemoryCurrent settles near 47 GiB),
`qwen-native-flash-router` (8091, unchanged), each with `NoNewPrivileges`,
512 MiB swap limits and no core dumps. With everything running the node shows
121 GiB total, about 73 GiB used (the Halo weights live in system memory),
45 GiB cache and 48 GiB available. The XTX performance level is `high` until
reboot or `echo auto`. The old enabled boot configuration is unchanged.

### Remaining limits

- No cooperative-matrix path: all batched GEMMs are scalar F32 (about 3.7
  TFLOPS), which caps prefill near 210 tok/s; an F16 coopmat tier would need
  its own labeled quality evidence.
- Decode is bandwidth-bound on Halo at about 24 ms of GPU time per token
  against a 15.5 ms roofline; the rest is many small dispatches, the F32
  router logits and expert gate/up at 186 GB/s.
- Batched attention cost grows with the key count (query tiles bounded by
  `QK_ATTN_BUDGET`); long-context prefill throughput was not measured.
- PLE residency needs about 36 GiB of page cache on the first stage's node.
- MTP, prefix snapshots and multi-sequence serving remain unimplemented.

## Cooperative-matrix tier (opt-in, partially validated)

`QK_FLASH_COOPMAT=1` switches the batched path to f16-input, F32-accumulate
16x16x16 KHR cooperative-matrix kernels: `qwen4_gemm_coop_{q5k,q6k,q8_0,q5_1}`
(dense projections, 128x64 tiles, the same dequantization as the scalar GEMM
through `qwen4_gemm_dequant.glsl`), `qwen4_moe_gateup_coop` and
`qwen4_moe_down_coop_{q8,q51}` (per-expert 128-row tiles over 16-token pair
groups, silu(gate)*up on the accumulator elements, scattered through LDS) and
the existing `fa_attn_batch_coopmat` for full attention.
`QK_FLASH_COOPMAT_MOE=0` keeps the scalar expert tiles under the tier. The HC
low-rank rows now use a 384-float stride so the 320-row down projection fills
complete tiles in either tier. The exact tier (scalar F32) stays the default
and the parity harnesses force it.

Measured on the 4-layer prefix graph at 512 tokens (Halo): GPU time 218 ms
exact -> 118 ms coopmat; warm batch wall 0.230 s -> 0.138 s. Per-kernel:
dense Q5_K GEMM 35 -> 10 ms, Q6_K 14.8 -> 2.7 ms, expert gate/up 72.6 ->
49 ms, expert down 33.7 -> 18.5 ms, attention 5.8 -> 2.7 ms. The remaining
exact-tier cost in that profile is the skinny 4/48-row projections (14 ms per
4 layers through the z-batched GEMV); `QK_FLASH_SKINNY=splitk` selects a
split-K kernel (`qwen4_gemm_skinny_*` + `qwen4_gemm_reduce`) that compiles and
is wired but has not been run on the GPU yet.

Quality evidence so far (reduced precision, not F32 parity): the synthetic
`qk qwen4-gemm` check gives 2.7e-4 relative RMS for the coopmat GEMM (f16
input rounding); `qk qwen4-batch` in the tier reports a median frame relative
RMS of 1.3e-3 against serial with about 6% of frames above 1e-2, consistent
with near-tie expert-routing flips; `tests/gpu_qwen4_tier.py` on a 256-token
model-generated sequence (dense coopmat only, before the expert kernels)
agreed with serial on 255/256 greedy ids (`bench/results-halo-native-coopmat-tier.jsonl`),
batched 1.15 s versus 1.47 s exact and 7.17 s serial for the two stages. The
extended KL/next-token log-probability run of that harness (which now also
covers the expert and attention coopmat kernels) was interrupted by the
incident below and has not produced a result. The tier therefore remains
opt-in and unmeasured end-to-end over HTTP.

## Incident 2026-09-11 (00:41-01:20 MDT): Halo driver out of memory

While the serving units were stopped for the coopmat window, repeated GPU
test loads (the 8 GiB prefix graph, then the full 67 GiB Halo stage for the
tier harness) ran with about 53 GiB of page cache still resident from the
PLE readahead of earlier runs and with the driver's retained TTM pages. The
kernel log shows `amdgpu 0000:c1:00.0` failing page-table updates with -12
at 00:41:40 and 00:45:21, then "Not enough memory for command submission" at
00:46:06. gnome-shell's submission was rejected with -12 at 00:46:00 and the
user's GNOME session (running since 2026-09-05) ended at 00:46:05; GDM's
greeter is on seat0 now, so the desktop must be logged into again. The tier
harness (`python3 tests/gpu_qwen4_tier.py`, PID 497595) has been in
uninterruptible sleep in `drm_suballoc_new` inside a command submission since
00:46, and a systemd close helper is stuck in the same allocator while freeing
another DRM file. All GPU rings show last emitted == last signaled and a
debugfs `amdgpu_gpu_recover` at 01:19:44 succeeded ("device wedged, but
recovered through reset") without unblocking either task; the leaked IB
sub-allocations from the failed submissions apparently survive the reset.
The Halo GTT still reports 61.7 GiB in use by the stuck process, so a full
native stage (68.9 GiB estimate) cannot load until the node reboots. No
reboot was performed: it affects other sessions on the machine and is the
operator's decision.

Recovery steps after the reboot (the old enabled units will start first;
stop them, then recreate the trial units as in "Memory plan and runtime
state", with the server unit at MemoryHigh 52G / MemoryMax 58G, set the XTX
performance level to `high` if wanted, and run `tests/native_flash_http.py`
against 8194 and 8091). Two guards were added for the page-cache plan and
compiled but not yet exercised: the readahead stops while `MemAvailable` is
below `QK_PLE_PREFETCH_FLOOR_GIB` (default 16), and the tables are paged out
when the stage closes. GPU experiments must not run beside a stage that holds
the resident tables; a dedicated window means stopping the units and waiting
for their memory to return.
