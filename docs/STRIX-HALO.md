# Strix Halo native Flash Next port

Status, 2026-09-10: first four native layers validated, not end-to-end serving.
No production service or port has been changed by this branch. No Halogen
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
  quantized payload sizing. The unsupported graph fails before initializing a
  GPU. Dense GEMV on the enormous sparse PLE table is rejected.
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
not exposed through `qk_open` or the serving endpoint.

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
not mean sufficient free VRAM. The active production XTX currently has room
for the one-layer test, not the four-layer test.

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
carries all 10240 residual floats (40 KiB per token). This has not been loaded
or benchmarked as a complete native model. Context, scratch, device budgets
and any future MTP head still need to fit; these numbers are not free VRAM.

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

1. Extend the validated native prefix to the complete `qwen4exp` graph,
   including the remaining expert formats and output head, then the serving ABI.
2. Compare intermediate activations and greedy token IDs against the pinned
   working reference on exactly the same GGUF and input IDs. Include multi-turn,
   reset, EOS, and chunk-boundary cases before exposing requests.
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
