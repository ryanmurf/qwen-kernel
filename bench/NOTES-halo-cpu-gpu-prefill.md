# CPU/GPU prefill and memory-layout investigation

Follow-up: the [CPU SIMD and expert-layout experiments](RESULTS-halo-expert-prep.md)
now have measured operator results. No hybrid serving backend is implemented.
The notes below preserve the initial read-only proposal, before those tests.

September 13, 2026, Max. **Read-only investigation and proposed experiments;
no hybrid backend implemented or benchmarked.** The ongoing decode-load
campaign is frozen and must finish without competing CPU/GPU benchmarks.

## Verified hardware and current execution

Local `lscpu` reports Ryzen AI Max+ 395, 16 physical cores / 32 threads,
one NUMA node, 64 MiB L3 across two instances, and AVX-512 support including
F, DQ, BW, VL, VNNI and BF16. AMD's [395 specifications](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html)
confirm 16 Zen 5 cores, AVX-512 and a 256-bit LPDDR5x interface. The CPU
and integrated GPU share physical memory and package resources; their
separate compute peaks are not automatically additive for this workload.
No clocks, power limits, memory allocation or firmware settings were changed.

`src/qwen4_graph.h::forwardBatch` currently prepares token embeddings and
PLE rows on the CPU, uploads them, then submits GPU layers. PLE row requests
for the current batch are issued before its serial gather. Preparation of
the *next* batch is not pipelined across this synchronous function boundary.
`src/quants.h` contains portable scalar-loop Q5_1/Q5_K dequantization, not an
explicit AVX-512 implementation. This is a source observation, not a claim
that the compiler emitted no SIMD instructions.

The approximately 21.9 GiB of expert tensors in the host-visible Vulkan
heap are still **GPU-computed**. Host-memory placement is not CPU offload.
The disk-backed 35.8 GiB PLE table is accessed by bounded row lookup; warming
or expanding the entire table is not part of this proposal.

## What the measurements do and do not support

The [completed profile](RESULTS-halo-combined-profile.md) attributes about
51% of the instrumented 16K prefill GPU interval to attention, 22% to dense
projections/head and 24% to experts/routing. That profile used baseline
prefill attention with per-dispatch fences; current serving uses vec4
prefill attention. It is not a current uninstrumented CPU-stall measurement.
CPU preparation therefore must be timed separately before projecting a gain.

The previous persistent transposed-K experiment was exact but slower and
remains rejected. The [decode-load candidate](RESULTS-halo-decode-loads.md)
instead improves access scheduling while preserving the F32 row-major
cache and accumulation order. Memory layout and access scheduling are
related but different interventions; neither should be assumed faster.

## Bounded follow-up experiment, after the decode comparison

1. Measure CPU embedding decode, PLE index/prefetch/gather, upload, GPU
   execution and waits separately; retain end-to-end first-token latency.
2. Test runtime-dispatched AVX2/AVX-512 preparation against the portable
   reference, including unaligned rows, tails and exact arithmetic checks.
   Preserve a portable fallback; do not globally enable unsafe math/FMA
   changes merely to obtain SIMD.
3. Prepare at most one next prompt chunk while the GPU handles the current
   one, initially with one worker, then bounded 2/4/8-core variants. Size and
   account for double buffers explicitly. Prompt tokens permit early PLE
   lookup; future decode tokens and model-selected experts are not known.
4. Preserve request-local history, cache ownership, resets, cancellation and
   buffer lifetimes. The existing PLE cache is mutable and not thread-safe;
   parallel calls to `gather` cannot simply share it without a design change.
5. Only then consider splitting independent expert work across processors.
   Moving consecutive whole layers to CPU/GPU is normally serial offload,
   not overlap. Vectorized weight decoding that creates large expanded
   copies may lose to extra memory traffic even when its arithmetic is fast.
6. Compare fixed prompts at 2K/8K/16K/31K, fresh KV, repeated counterbalanced
   launches, same precision/model and matched output lengths. Track total
   latency, CPU/GPU utilization, temperature and memory/swap pressure.
   Include cache-group affinity experiments without assuming one NUMA node
   means every core shares one L3. Keep only a correctness-checked net win.

These are hypotheses, not a forecast of speedup or a change to the frozen
decode campaign's promotion criteria. GPU attention and GEMM improvements
remain important even if CPU preparation becomes entirely hidden.
