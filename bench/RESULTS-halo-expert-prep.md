# Expert-weight layout and AVX-512 preparation on Max

September 13, 2026. **Experimental operators, not a deployed model speedup.**
No serving library, shader, model weight, precision, firmware or kernel was
changed. Both expert trials restored and verified the previous serving
stack on 8194/8091/8092; the final restoration passed at 21:40 UTC.

## CPU preparation: a small, measured opportunity

Implemented standalone AVX-512 Q5_1/Q5_K dequantization with runtime CPU
feature checks. The portable reference and SIMD implementation are compiled
with `-ffp-contract=off`; arithmetic order is retained. The real checkpoint's
token embeddings and selected PLE rows use the existing 2048-token fixture
prefix. No GPU work, CPU layer offload, thread pool or pipeline overlap is
implemented. These functions are not wired into the engine.

The successful CPU controller pins one physical core (CPU 2), caps memory at
1 GiB with swap disabled, and verifies that the native server PID and its GPU
engine counters did not change. It completed in 2.92 s, with a 260,251,648-byte
cgroup peak and zero swap; serving binaries were unchanged.

Warm preparation medians from four cells per mode in ABBAABBA order, each
cell averaging 30 repetitions within one process:

| Tokens | Portable, ms | AVX-512, ms | Slice speedup | Milliseconds saved |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.343 | 0.067 | 5.10x | 0.276 |
| 512 | 1.571 | 0.496 | 3.17x | 1.075 |
| 2048 | 8.158 | 4.268 | 1.91x | 3.890 |

This includes embedding dequantization, copying four residual streams and
dequantizing preselected compressed PLE rows. It excludes index gathering,
file faults, cache lookup, GPU upload and model execution. In particular,
it re-decodes every selected PLE row, whereas live PLE cache hits can avoid
that arithmetic. These are not API prefill times or proportional model gains.

The separately observed 512-token PLE prefetch/gather cost was about 33.48 ms
in the first run and 7.25 ms in the repeat. File cache was **not flushed**;
the lookup cache is cumulative across 128/512/2048 sizes. That difference
is evidence of cache sensitivity, not an AVX-512 or pipeline speedup. It
makes bounded next-chunk lookup overlap worth measuring separately.

Validation: 307,200 Q5_1 and 2,457,600 Q5_K synthetic values match bit-for-bit,
including finite half subnormals, signed zeros, unaligned rows/output and
zero-length input. Actual-model preparation compares every output byte.
A separate AddressSanitizer/UndefinedBehaviorSanitizer test passes protected
page-end rows and output buffers. NaN/Inf model payload semantics have not
been established by these finite-data tests.

The first CPU supervisor mistakenly combined systemd `--wait` with
`RemainAfterExit`, so its clean child waited after completion and a subsequent
manual stop removed live peak counters before the audit. Its overall receipt
is FAIL and is retained. The corrected repeat polls the exited state, records
the limits/peaks before stopping, and passes. Failed receipts are not replaced.

## Expert gate/up: layout versus load width

The existing GPU prefill kernel already groups token/expert pairs and reuses
dequantized weights across 16 tokens. Its GGUF Q5_K blocks remain row-major;
neighboring lanes read different rows at a 1760-byte row stride for this shape.

The first sweep compares original byte-addressed rows, word-addressed rows
and a word-transposed layout within 128-row tiles. The new layout places the
same 32-bit word from neighboring rows together. It preserves all encoded
weight bits and the 176-byte block size; it does not expand weights to F32.
CPU packing is outside GPU timestamps, with an exact reverse-map check.

Real weights: layers 0 and 47, 32 sampled experts per layer (stride 16;
offsets 0 and 15 respectively), gate and up, dimensions 640 x 2560.
Fixtures are synthetic activations/grouped pairs, **not actual model routing
histograms**. Each token has one exercised expert slot; inactive slots remain
poisoned. Both sweeps include 32 experts with 1/8/16/17 pairs each, one hot
expert with 64 pairs, and a final decreasing-size case of 16 experts with
three pairs and scaled activations. Native DEVICE_LOCAL buffers are used;
the mixed host-heap placement of the full model is not reproduced.

First sweep: 72 cells, baseline/words/SoA/SoA/words/baseline per case.
Word addressing alone is slower. SoA cuts kernel time roughly 7–17% in
most cases (one 16-pair case improves only 3.5%), but is about 6% slower
for the single hot expert. All outputs match exactly, unused output/guard
regions stay poisoned, and sampled FP64 references pass.

## Splitting a hot expert's token work

The second sweep retains the original and SoA controls, and gives each
16-pair tile its own workgroup in a third SoA variant. Weight/dequantization,
accumulation order, activation, output scatter and quantization are unchanged.
This addresses the hot-expert case's limited number of concurrent workgroups.
It does not split a dot-product reduction or change its floating-point order.

Second-sweep GPU milliseconds, medians of two cells per mode; each cell has
four timed dispatches after warm-up. Slash-separated values show the two
tested layers separately, not a pooled confidence interval:

| Active experts x pairs | Layer | Original | SoA | SoA + pair split |
| --- | ---: | ---: | ---: | ---: |
| 32 x 1 | 0 / 47 | 0.961 / 0.965 | 0.795 / 0.794 | 0.799 / 0.796 |
| 32 x 8 | 0 / 47 | 1.015 / 1.014 | 0.854 / 0.858 | 0.856 / 0.853 |
| 32 x 16 | 0 / 47 | 1.090 / 1.089 | 1.049 / 1.038 | 1.032 / 1.045 |
| 32 x 17 | 0 / 47 | 1.876 / 1.861 | 1.734 / 1.745 | 1.824 / 1.821 |
| 1 x 64 | 0 / 47 | 0.696 / 0.700 | 0.740 / 0.738 | 0.189 / 0.194 |
| 16 x 3 | 0 / 47 | 0.464 / 0.467 | 0.391 / 0.389 | 0.390 / 0.390 |

All 72 second-sweep cells pass the same numerical/guard checks. The hot case
is 3.61–3.67x faster than original. Splitting is not uniformly better than
unsplit SoA: at 17 pairs per expert, it gives back some of the layout gain.
Both controls are retained so that tradeoff is visible.

Transfers and packing are outside timestamps; weights are repeatedly reused.
These are gate/up operator timings, not all experts/routing, full prefill,
decode, or a model-wide 3.6x gain. Two cells in one GPU launch per sweep
are not an independent-launch statistical estimate.

## Before integration

- Capture actual per-layer expert occupancy before selecting a scheduling
  threshold. The hot-expert result motivates a compact GPU tile work queue;
  launching a full expert x maximum-token-tile grid would waste empty groups.
- Test layout variants against **decode too**. A prefill-friendly transpose
  may make single-token weight access worse. Duplicating the full gate/up
  weights is not acceptable on this memory-constrained model.
- Compare a single compatible persistent layout with bounded, per-layer
  temporary packing, counting packing time, scratch bytes and extra traffic.
  Do not promote based only on the timed inner kernel.
- Profile and overlap request-local CPU lookup/preparation with GPU chunks;
  retain bounded buffering, cancellation/reset semantics and exact outputs.
  The existing mutable PLE cache is not safe for concurrent gathers.
- Run a full-model numerical/API/latency/resource gate before serving changes.
  The unrelated decode-load campaign's 31K device loss remains unresolved;
  it was not excused or deployed as part of this experiment.

Private evidence: `/home/ryan/qk-cpu-prep-ZJhtVI` (first supervisor failure),
`/home/ryan/qk-cpu-prep-repeat-YOjQvm` (successful CPU repeat),
`/home/ryan/qk-expert-layout-tAFAFc` and
`/home/ryan/qk-expert-split-jU4yTC`.
Public raw logs and receipts: [results-halo-expert-prep](results-halo-expert-prep/).

The archived controllers are Max-specific provenance, not portable launch
scripts: they name local model paths and stop/restore this node's serving
units. Review and adapt them before reuse. The experiment sources remain
under `tests/halo_cpu_prep*`, `tests/halo_expert_layout*` and
`tests/halo_expert_split*`, outside the production build. Build commands are
in the C++ source headers. The split shaders compile with `-DSOA=1`; the
first sweep additionally uses `-DSOA=0` for its word-addressed control.

Validation after archiving: 31 Python regression tests pass, including four
new evidence/coverage/resource/failure checks. CPU protected-page tests pass
with AddressSanitizer and UndefinedBehaviorSanitizer. `git diff --check`
passes. The default serving build does not include the experimental files.
