# Strix Halo native Flash Next port

September 12 status: **Halo-only serving is restored on 8091**, with
`QK_FLASH_PREFILL_LAST=1` explicitly enabled in the current transient server.
The source-level default remains off. The [full-model campaign](../bench/RESULTS-halo-last-head-full.md)
passes 498 exact-logit cross-policy/reset comparisons and all matched HTTP,
Claude/tool and seeded-output checks. Median first-token latency is 2.8–5.0%
lower across 128–16384 prompt tokens. At 16K, median decode is 2.7% lower
with overlapping ranges; this is not a universal throughput claim. The
same-final-tile optimization preserves the all-ID ABI, F32 precision and
state processing. Direct 8194, gateway 8091 and proxy 8092 restoration checks
pass, with zero model cgroup swap or pressure during the observation window.
The approved bounded unused-TTM cleanup did not lower the 24 GiB launch
guard. Separate [attention LDS probes](../bench/RESULTS-halo-attention-lds.md)
passed operator correctness but did not justify a new serving path. Their
experimental shaders remain unintegrated. The [earlier two-layer result](../bench/RESULTS-halo-last-head.md)
is not a whole-model speedup claim. Nothing was enabled at boot.

Validated milestone `0453754`, 2026-09-11: **the native engine served port
8091 from the Strix Halo iGPU alone** (all 48 layers plus the
head on 0000:c1:00.0; the XTX holds no weights and does no compute; Vulkan
device enumeration still opens tiny bookkeeping handles on it, as root's
fdinfo check showed, and the single-mode restore helper does not need or
probe it). Commit bd8b87f added the two-heap placement
that makes the 88.7 GiB of GPU weights fit across the 70.7 GiB device-local
heap and the host-visible heap (21.9 GiB in the host heap at 32768 context),
with fail-closed per-heap and physical budgets. The later milestones add
request-scoped PLE row prefetch, decode fusions, a hardened restore helper
and the measurements below. Full-model F32-oracle parity, batched
parity with a stronger reset check and the real HTTP/Claude suites all pass on
the single device. See "Halo-only serving" for numbers and limits. The
cooperative-matrix tier stays opt-in. MTP and prefix snapshots remain
unimplemented. The native units are transient (not started at boot); the
legacy two-GPU boot stack can no longer start (its three unit names are
masked with backups, see "Legacy stack containment"). No Halogen binary has
been installed or executed.

The earlier two-GPU split (Halo 0:37 + XTX 37:48) remains available as an
explicit `MODE=split` of the restore helper and its measurements stay below
for reference; they are not Halo-only results.

## Performance research leads

- [pwilkin/llama.cpp](https://github.com/pwilkin/llama.cpp) — recorded at the
  user's request on 2026-09-12 as possible inspiration for further Halo
  optimization. This is a research lead, not a validated performance result
  or a selected replacement backend. Remote branch tips inspected that day:
  [`strix-halo` at f5daaa3](https://github.com/pwilkin/llama.cpp/commit/f5daaa3cfa6358e5dd398911ec741813745a5440)
  and [`strix-halo-for-halobox` at 27fd0cf](https://github.com/pwilkin/llama.cpp/commit/27fd0cf2cd068447f4e0b530f258be6349661420).
  The latter describes AMD prefill work under `ggml-cuda`: BF16 WMMA GEMM,
  PLE/GDN convolution, gated normalization, expert reduction and bounded
  routing scratch. These are leads for a Vulkan implementation, not drop-in
  shaders; BF16 arithmetic belongs to a separately validated precision tier.
  Its commit notes also flag tensor-pointer lifetime hazards in optional
  marking features, and shadow-weight eligibility/cleanup fixes worth
  reviewing before reuse. Inspect the actual implementation and test any
  candidate with our correctness gates and matched full-model benchmarks.
  No code from this fork was imported or run for this entry; the source
  review did not change the completed last-output prefill A/B.

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

## Cooperative-matrix tier (opt-in, measured)

`QK_FLASH_COOPMAT=1` switches the batched path to f16-input, F32-accumulate
16x16x16 KHR cooperative-matrix kernels: `qwen4_gemm_coop_{q5k,q6k,q8_0,q5_1}`
(dense projections, 128x64 tiles, the same dequantization as the scalar GEMM
through `qwen4_gemm_dequant.glsl`), `qwen4_moe_gateup_coop` and
`qwen4_moe_down_coop_{q8,q51}` (per-expert 128-row tiles over 16-token pair
groups, silu(gate)*up on the accumulator elements, scattered through LDS) and
the existing `fa_attn_batch_coopmat` for full attention.
`QK_FLASH_COOPMAT_MOE=0` keeps the scalar expert tiles under the tier. The HC
low-rank rows use a 384-float stride so the 320-row down projection fills
complete tiles in either tier. The F32 tier stays the default; the trial
script pins `QK_FLASH_COOPMAT=0` unless the operator overrides it, and the
oracle harness (`tests/gpu_qwen4_batch.py`) forces the F32 tier.

Quality and speed, measured 2026-09-11 with `tests/gpu_qwen4_tier.py` on the
full 0:37/37:48 split (dedicated window, prefetch off, 256 positions of a
model-generated sequence, two identical runs; records in
`bench/results-halo-native-coopmat-tier.jsonl`):

| Metric (256 positions) | F32 batched vs serial | coopmat batched vs serial |
| --- | --- | --- |
| Greedy id agreement | 256/256 | 254/256 (positions 28, 55) |
| Per-position KL, mean / median / p99 / max (nats) | 7.9e-6 / - / - / 5.6e-4 | 6.4e-4 / 1.7e-5 / 8.9e-3 / 2.9e-2 |
| Next-token mean log-prob delta | - | +3.9e-4 |
| Both stages, 256 tokens | 1.66 s (serial 7.22 s) | 0.94 s |

The KL rows compare the head re-run one position at a time on each tier's
batched hidden rows against the serial rows. The F32 batched tier is not
bit-identical to serial: it agrees to F32 rounding (KL below 1e-4) for the
first 182 positions of this sequence and then deviates slightly (max KL
5.6e-4, greedy unchanged), which is consistent with one near-tie expert
routing flip under a different F32 summation order (batched attention and
the router GEMM reduce in a different order than the serial kernels). The
16-position F32-oracle parity (`tests/gpu_qwen4_batch.py`) is unaffected.
The coopmat tier trades a mean 6.4e-4 nats of divergence and 2/256 greedy
flips for 1.76x faster prefill on this split; it remains opt-in and has not
been exercised through the HTTP suite.

Prefix-graph timings (4 layers, 512 tokens, GPU time from `QK_FLASH_PROFILE`):
Halo 253-256 ms F32 tier in the 15:00 session (218 ms in the 00:30 session,
same code; the difference tracks the day's device state and is why A/B
numbers are only compared within one session), coopmat 118 ms (00:30
session); XTX 58 ms coopmat. In the prefix batch check the coopmat tier shows
a median frame relative RMS of 1.6e-3 against serial with 6-9% of frames
above 1e-2 (expert-routing flips inside four layers), so `qk qwen4-batch`
reports the tier as FAIL against its reduced-precision budget; the
token-level harness above is the evidence that counts for serving.

Rejected after measurement (Halo, 512 tokens, F32 tier, same session):
a split-K kernel for the 4/48-row projections (`QK_FLASH_SKINNY=splitk`,
256.4 ms total versus 256.0 ms with the z-batched GEMV) and the masked
128-row GEMM tile for them (`QK_FLASH_SKINNY=gemm`, 252.9 ms); neither
changed the total in that experiment. The current `projectBatch` path uses
masked scalar GEMM for skinny projections at T >= 64, unless
`QK_FLASH_SKINNY=gemv` is set; do not infer the current default from this
historical comparison. Note for readers of that older
profile: dispatches recorded with `fence=false` carried no timestamp, so their
GPU time is attributed to the next fenced dispatch (the "gemv_q5_k" row at
512 tokens mostly contains the unfenced GDN qkv/gate/alpha GEMMs). Since
`0453754`, profiling fences every dispatch; the older result is not an
exclusive per-shader profile.

### Page-cache hygiene (added 2026-09-11)

`systemctl stop` sends SIGTERM; the Rust server previously exited without
closing the engine, so the warmed tables stayed in the page cache after the
stage was gone (observed: 42 GiB cached, 5 GiB free with all units stopped).
The server now shuts the engine down on SIGTERM as well as SIGINT, which
runs the stage's page-out on close. `deploy/release-model-cache.py` drops the
model shards' page cache with a targeted `POSIX_FADV_DONTNEED` for the case
where a process died without it, and `deploy/restore-native-flash.sh` runs
the guarded start sequence. Its current default is Halo-only, with no full
table warming; only explicit `MODE=split` starts an XTX worker before the
Halo server. See "Restore helper" for the current behavior and limits.

## Incident 2026-09-11 (00:41-01:20 MDT): Halo driver out of memory

Historical incident, resolved by the user-authorized reboot at 07:12:46 on
2026-09-11. The account and recovery plan below describe the pre-reboot
state, not a current stuck process or a recommendation to reset a GPU.
For current operation, use the Halo-only restore procedure below; the old
two-GPU unit names are now masked.

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

## Halo-only serving (2026-09-11)

Configuration: `deploy/restore-native-flash.sh MODEL 32768 0 single` (the
default mode) starts one server unit (`run-native-flash-trial.sh single`,
`--local-driver`, HTTP loopback 8194, 24G/32G unit limits) and the trusted-LAN
router on 8091. Placement printed by the serving engine at context 32768:
device-local 69.15 GiB (weights 66.84 + KV/state 1.86 + batch rows 0.45; heap
headroom 69.4 after a 1 GiB margin), host heap 21.88 GiB in 36 routed-expert
tensors (type 2, heap 0, HOST_VISIBLE|HOST_COHERENT; headroom 34.2), MemTotal
121.2 GiB, reserve 12 GiB. (The parity harnesses load at context 8192, where
the KV allowance is smaller: 67.92 GiB of weights stay device-local and 20.80
GiB in 34 tensors spill.) Load takes 73-105 s; Halo GTT reads 91.4 GiB
afterwards; the unit's own cgroup was 0.67 GiB with a 1.32 GiB peak on
the served instance (root's reading), not a bound: each uploaded tensor's
file pages are dropped as it goes and the on-demand PLE rows are charged to
it, so it grows with the working set up to the unit's 24G/32G limits. The driver's retained page pool (67.5 GiB before the
first load, read from `ttm_page_pool` by root) is reused by the allocations,
which is why MemFree only moves a few GiB per load.

GPU reads from the host heap cost 2-7% more than device-local on this part
(operator probe with the actual type/heap/flags printed, no fallback:
Q5_K 2560x6144 51.5 -> 54.9 us, Q6_K 10240x2560 92.7 -> 94.5 us at 64 MiB
rotation), so only expert tensors (10 of 512 read per token) are placed there.

Correctness on the single device (Q5_K_M, ctx 8192, F32 tier, prefetch 0).
The two harness records below were produced by the bd8b87f build, before the
PLE row-prefetch and batched row-history precompute changes of 7d34606; the
current build's own prefix, full-model, chunk and reset results are listed in
"Current-build verification" further down:

- `tests/gpu_qwen4_full.py --single`: 16 positions against the F32 oracle,
  worst logit relative RMS 1.33e-6, every greedy id, reset exact, replay
  bit-exact (`bench/results-halo-only-full-parity.jsonl`).
- `tests/gpu_qwen4_batch.py --single`: serial, whole-batch, mixed 5+1+7+3
  and batch-then-serial chunking all within 1.3e-6 of the oracle; the reset
  check now requires the clean first row captured before any batched work to
  equal, bit for bit, the first row after a different prompt fed whole and in
  mixed chunks (`bench/results-halo-only-batch-parity.jsonl`).
- `tests/native_flash_http.py` on 8194 (all eight checks, cancellation 4.6 s),
  on 8091 and the 8092 proxy stream/tool checks.

Measured through HTTP, Halo only, ctx 32768, F32 tier, no table warming (the
36 GiB table cannot fit beside the model). Cache condition per row: every
configuration ran in a fresh server process after the helper requested
release of the shards' page cache, and each ran the 512-distinct prompt
first. Here "first touch" identifies that first workload invocation after
the cache-release procedure, not a census proving every PLE page was cold.
The 2048 prompt shares its first 512 ids with the 512 prompt, so its
first quarter was already resident; the second repetition of the default
configuration re-used the working set (page-cache warm, beyond the 4096-row
in-process cache). Records: `bench/results-halo-only-ab-bench32.jsonl`.

| Halo-only configuration | Cache condition | Decode tok/s | 512 distinct tokens | 2048 distinct tokens | 512 uniform |
| --- | --- | --- | --- | --- | --- |
| Serial prefill (`QK_FLASH_BATCH=0`), row prefetch on | first touch | 32.7-32.9 | 16.6 s (31 tok/s) | 77.5 s (26 tok/s) | 16.1 s |
| Batched 512, row prefetch off (`QK_PLE_ROW_PREFETCH=0`) | first touch | 27.3-32.9 | 7.27 s (70 tok/s) | 24.9 s (82 tok/s) | 3.06 s |
| Batched 512, row prefetch on (default), rep 1 | first touch | 32.85-32.92 | 3.07 s (167 tok/s) | 13.3 s (154 tok/s) | 2.94 s (174 tok/s) |
| Batched 512, row prefetch on (default), rep 2 | repeated working set | 32.87-32.92 | 3.03 s (169 tok/s) | 12.9 s (159 tok/s) | 2.96 s (173 tok/s) |

Batched prefill is 5.4x the serial path on this device for the measured
512-distinct prompt. With the same startup and first-workload procedure,
the PLE row prefetch (`MADV_RANDOM` on the disk-mapped table plus
one asynchronous `MADV_WILLNEED` per uncached row of a request before the
serial gather) cuts 512 distinct tokens from 7.27 to 3.07 s and 2048 from
24.9 to 13.3 s, without table warming or unbounded memory; the repeated
working set (rep 2) is only 1-3% faster, so first-touch prefill is now close
to page-cache-warm prefill. The remaining difference between distinct and
uniform prompts (3.07 versus 2.94 s) has not been attributed by phase
measurement: uniform prompts also route to the same experts every token, so
expert grouping and the routed-expert working set differ, not only PLE
faults. Decode on the short counting prompt is unchanged by the prefill
configuration; at 32.9 tok/s the
single device is within 15% of the split stack's 35-38 tok/s, where the XTX
ran the last 11 layers and the head as a successive pipeline stage (not in
parallel for single-token decode) with warm tables.

For comparison with the earlier split figures, first-token latency on the
35-token counting prompt is 0.59 s and the 512-token prefill of the split
stack was 2.55-2.67 s with warm tables (not comparable: different device set
and warm tables).

Rejected or discarded in this campaign: the 20:19 measurement of the default
configuration (a legacy-unit start test overlapped it for 4 s; see below), and
the earlier idea of warming the whole table on the single device (libqk now
rejects `QK_PLE_PREFETCH=1` when the tables cannot fit beside the plan).

### Legacy stack containment (2026-09-11)

`claude-qwen-proxy.service` carried `Wants=qwen-kernel-prefill-router.service`,
and that router `Requires` the legacy Halo decode worker and `Wants` the XTX
prefill worker: a proxy restart or a reboot would have loaded the old two-GPU
llama.cpp stack beside the 91 GiB native model. During this work a start test
of the router, run before its mask was actually in place (the real unit file
had made `mask` refuse), pulled the legacy loaders for about ten seconds
(20:20:09-20:20:19 MDT) before they were stopped; the kernel log shows no
amdgpu error, but measurements overlapping that interval were discarded and
legacy units must never be tested by starting them again. Containment,
reversible and backed up in `~/.config/systemd/user/backup-halo-only-2026-09-11/`
(originals, enablement list, rollback commands): the three legacy unit names
(`qwen-kernel-prefill-router`, `qwen-kernel-prefill-xtx`,
`qwen-kernel-decode-halo`) are masked with their files moved to the backup;
`qwen-kernel-next-router` is linked but no longer enabled. A proxy drop-in
(`claude-qwen-proxy.service.d/10-halo-only.conf`) was added to reset the
`Wants=`/`After=` lists, but `systemctl show` still reports the original
`Wants=qwen-kernel-prefill-router.service` on the loaded unit, so the drop-in
is documentation only: the masks are what make that dependency inert (a
masked unit cannot be started by `Wants=`, and the proxy still starts
normally). Reboot behavior now: the proxy starts alone, no model starts; the
operator runs the restore helper.

### Restore helper

`deploy/restore-native-flash.sh MODEL [CONTEXT=32768] [WARM=0] [MODE=single]`
validates its arguments first, refuses while a native unit is active, a qk or
server process runs, a GPU process is stuck, or the Halo (and in split mode
the XTX) still holds memory; releases the model shards' page cache; starts
the server (and, in split mode only, the XTX worker first) and fails
explicitly, stopping only the units it started, if a readiness deadline
passes; requires an HTTP 200 `{"status":"ok"}` from 8194 and 8091. Its
failure paths were exercised without GPU loads (bad arguments, missing
model, active units). `QK_FLASH_BATCH`, `QK_PLE_ROW_PREFETCH`,
`QK_FLASH_COOPMAT`, `QK_FLASH_FUSE`, `QK_MOE_GU` and `QK_GDN_STEP` are
forwarded into the unit when set in this milestone. The live-process guard
checks executable identity rather than matching binary names anywhere in
a shell command, so benchmark parent shells do not trigger false positives.

### Remaining limits (Halo-only)

- At `0453754`, warm decode is about 33.4 tok/s on the 35-token counting
  prompt. The profile below identifies quantized projections and routed
  experts as the main measured costs; an approximate bandwidth roofline is
  not proof that all remaining time is dispatch overhead.
- Distinct-token prefill still depends on the PLE page-cache working set and
  expert grouping. Their separate contributions have not been measured by
  an end-to-end phase breakdown. A larger bounded row cache is a hypothesis
  for repeated conversations, not a verified gain.
- The plan leaves about 1.3 GiB of device-local headroom at 32768 context and
  512 batch rows; longer contexts need the KV allowance re-checked.
- Coopmat tier, MTP, prefix snapshots and multi-sequence serving are unchanged.

## Current-build verification and decode fusions (`0453754`, 2026-09-11)

Build: `0453754` (decode fusions, word-addressed expert gate/up, profiling
fences and helper fixes). Standalone numerical tests and profiling ran on
the Strix Halo alone with the serving units stopped. The production A/B and
HTTP suites then ran against restored single-device servers; the candidate
was left serving on 8091. Later tuning may temporarily stop that endpoint.

Numerical checks on this build (`bench/results-halo-only-fused-parity.jsonl`,
`bench/results-halo-only-fused-prefix.txt` for the prefix lines):

- 4-layer prefix oracle: relative RMS 4.77e-7 (fused) and 4.80e-7 (control),
  both PASS; prefix batch check at 128 tokens PASS for whole, mixed and
  batch-then-serial chunking.
- Full model, 16 positions against the F32 oracle: PASS (all greedy ids, reset
  exact, replay bit-exact); batched parity with the strong reset check: PASS.
- Teacher-forced kernel-configuration comparison (`tests/gpu_qwen4_teacher.py`,
  256 deterministic positions, the same build with control
  `QK_FLASH_FUSE=0 QK_MOE_GU=v1` versus the fused defaults): 256/256 greedy
  agreement, KL mean 3.7e-12 nats, max
  2.4e-11, largest logit relative RMS 2.2e-6
  (`bench/results-halo-only-teacher-control-vs-fused.json`).

Changes measured: the HC low-rank down projection now applies the silu
epilogue in the GEMV (specialization constants on `gemv_q5_k`/`gemv_q6_k`,
one separate SiLU dispatch removed per HC module: two per layer plus the
output head, 97 total), the GDN step computes its per-head
decay/beta in place (`qwen4_gdn_step_p`, one dispatch per GDN layer removed),
and the routed and shared expert gate/up kernels read their Q5_K blocks as
32-bit words with vec4 activations (`moe_gateup_q5k_v2`, `moe_shared_q5k_v2`).
Rollback knobs: `QK_FLASH_FUSE=0`, `QK_MOE_GU=v1`, `QK_GDN_STEP=v1`; the
restore helper forwards them.

Prefix-level serial GPU time per token (4 layers, every dispatch fenced,
alternating runs): fused+v2 2.474/2.478 ms, control 2.522/2.523 ms, fusions
alone 2.511 ms, v2 alone 2.501 ms.

Production A/B (HTTP, profiling off, fresh server per configuration, controls
then candidate, two repetitions of three prefill workloads each;
`bench/results-halo-only-fusion-ab-bench32.jsonl`). Each workload is preceded
by the same 35-token counting prompt with a reset context and 96 generated
tokens. Warm decode is the median of the five streams after the first stream
on each server; it is not decode at a 512- or 2048-token context. All twelve
streams produced identical, coherent counting prefixes.

| Configuration | First stream tok/s | Warm decode median tok/s | 512 distinct, reps 1 / 2 | 2048 distinct, reps 1 / 2 | 512 uniform, reps 1 / 2 |
| --- | --- | --- | --- | --- | --- |
| Controls (`QK_FLASH_FUSE=0 QK_MOE_GU=v1`) | 32.28 | 32.92 | 3.09 / 3.10 s | 13.19 / 12.90 s | 2.99 / 2.90 s |
| Fused defaults | 32.95 | 33.35 | 3.03 / 3.03 s | 13.23 / 12.98 s | 2.99 / 2.98 s |

Warm decode improved 1.30% in this A/B; the 256-position comparison above
also bounds the observed numerical change. The 512-token prefill times are
slightly lower, but the batch kernels were unchanged and the other prefill
cases show no consistent gain; do not generalize that small difference.
This is a short-context result from one control/candidate ordering, not a
statistical guarantee for every workload or a long-context decode benchmark.

The fused build's instrumented decode profile (24 tokens after 64 distinct
prompt tokens, replay off, every dispatch fenced;
`bench/results-halo-only-decode-profile-fused.txt`) is 31.1 ms GPU per token.
Its timestamps attribute each fenced interval to one dispatch, including
associated synchronization. Those fences change scheduling, so this is a
bottleneck profile, not production throughput or a measured fixed profiling
overhead. The rows are:
dense Q5_K GEMVs 9.4 ms (375 dispatches), routed expert gate/up 5.4 ms,
Q6_K GEMVs 4.5 ms (49 dispatches, byte loads), routed Q8_0 down 2.3 ms, Q5_1
down 1.6 ms, router logits 1.4 ms, HC-up Q5_1 GEMVs 1.4 ms, GDN step 1.2 ms,
HC elementwise 1.2 ms (294 dispatches), shared experts 0.8 ms, expert select
0.6 ms, attention 0.6 ms. Next measurable candidates, each to be gated by an
actual-weight microbench and the full-model checks before becoming a
default: word-addressed Q6_K GEMV (210-byte blocks, 2-byte alignment), routed
Q8_0 and Q5_1 down kernels (34- and 24-byte blocks), and folding the HC
combine into the following norm.

API verification on the served fused build: all eight checks on 8194
(`bench/results-halo-only-fused-http32.jsonl`), the gateway suite on 8091 and
the 8092 proxy stream/tool checks. Separately: when the upstream 8091 is
down, the loopback proxy on 8092 returns a malformed HTTP status line
(`BadStatusLine` in the client) instead of a clean 502; that is an existing
error-path protocol issue of the proxy, not a model result, and it is not
counted as passed API verification.

### Word-addressed Q5_1 and Q6_K variants: rejected as defaults (2026-09-11)

Following the exclusive decode profile, word-addressed twins of the Q6_K
GEMV (`gemv_q6_k_v2`, 210-byte blocks fetched as containing dwords), the
Q5_1 GEMV (`gemv_q5_1_v2`, six dwords per 24-byte block) and the routed and
shared Q5_1 expert down kernels (`moe_down_q5_1_v2`,
`moe_down_shared_q5_1_v2`) were written and gated. Measurements on the Halo
(`bench/results-halo-only-q6q51-experiment.txt`), alternating byte-addressed
v1 against v2 with the actual memory type printed:

| Kernel / shape | v1 | v2 |
| --- | --- | --- |
| Q5_1 GEMV 10240x320 (HC up) | 12.5-12.6 us | 12.3-12.4 us |
| Q5_1 GEMV 320x10240 | 12.4-12.6 us | 12.4-12.5 us |
| Q6_K GEMV 10240x2560 | 92.4-95.1 us | 139.6-139.9 us |
| Q6_K GEMV 2560x6144 | 55.6-56.4 us | 76.0-76.2 us |
| Q6_K GEMV 320x10240 | 13.3-13.4 us | 18.1-18.5 us |
| Layer-6 expert chain with Q5_1 down (actual weights) | 186.1-187.3 us | 189.8-189.9 us |
| 4-layer prefix serial GPU time per token, all three v2 | 2.478-2.480 ms | 2.636-2.650 ms |

The Q6_K variant is 40-50% slower (the 2-byte block misalignment turns most
loads into two-word fetches plus shifts) and the Q5_1 variants are within
1% either way, so all three stay off: the byte-addressed kernels remain the
defaults and `QK_Q51_GEMV=v2`, `QK_MOE_DOWN=v2`, `QK_Q6K_GEMV=v2` opt into
the variants for reference. Review found that the first `gemv_q6_k_v2`
fetched the block's f16 scale with a four-byte helper that could read one
word past the buffer on the last block of a tensor; it now extracts the
2-byte-aligned half from its single containing word, weight buffers are
rounded up to whole dwords, and the fixed variant passes the operator check
at odd block counts (7x256, 9x512, 65x2560). The variants are numerically
F32-close to the originals (all-v2 build: full-model oracle parity PASS,
teacher-forced comparison against the served kernels 256/256 greedy, KL max
5.4e-11).

Two process notes from this experiment: the chain's "controls" restore
actually served the all-v2 build because the helper did not yet forward the
three new knobs (it does now, and it prints the unit's actual environment
after start, which the operator should check against the intended
configuration), and the chain was interrupted before its production A/B, so
no production rows exist for these variants; they were rejected on the
isolated and prefix-level measurements above.

### Served baseline after the experiment (2026-09-11)

The stack serving 8091 now runs the code defaults (fused decode path,
word-addressed Q5_K expert gate/up, byte-addressed Q5_1/Q6_K/Q8 kernels, PLE
row prefetch), rebuilt and restored in a dedicated window: 4-layer prefix
oracle 4.77e-7 PASS, prefix batch check PASS, the served process's real
environment read from `/proc/PID/environ` (`QK_PLE_PREFETCH=0
QK_FLASH_COOPMAT=0`, no v2 knobs), all eight API checks on 8194, the gateway
suite on 8091 and the 8092 proxy checks. Two repetitions (Halo only, ctx
32768, no table warming; `bench/results-halo-only-baseline-bench32.jsonl`):
decode 33.3 tok/s, 512 distinct tokens 3.05-3.08 s, 2048 distinct tokens
13.1-13.2 s, 512 uniform 2.93 s. GPU experiments are paused here pending the
multi-prefill backend comparison (`bench/prefill_matrix.py`, root-owned).
For a Halo-only llama.cpp comparison the local fork's single-device recipe
(`deploy/run-qwen-next-trial.sh halo` in the original worktree: `--device
<STRIX_HALO> --split-mode none --fit-target 16384 -ngl 99 -fa on -b 2048 -ub
1024 --load-mode mmap`, MTP draft, Jinja, xhigh) would need the fork's
`GGML_VK_ALLOW_SYSMEM_FALLBACK=1` to exceed the 70.7 GiB device-local heap,
a memory-limited unit, the shard-cache release before launch, no concurrent
GPU load, and its output quality checked first (the fast configuration
produced malformed counting output on 2026-09-10).
