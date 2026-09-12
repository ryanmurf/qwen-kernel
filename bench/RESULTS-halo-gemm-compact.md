# Halo scalar-F32 prefill GEMM experiment

Status, 2026-09-12 12:08 UTC: **operator and same-build full-model checks
passed; matched API checks are running.** `QK_FLASH_GEMM=compact` is opt-in. Unset it, or
set `baseline`, for the unchanged default. Decode attention remains serial;
the ordered-attention API experiment did not establish a speed win.

The compact kernel keeps the 128-output-row / 64-token tile and each
32-element F32 dot/accumulation sequence. It reduces the shared-memory K
tile from 64 to 32 elements (49920 to 25344 LDS bytes), and reuses each
activation load across eight output rows. Baseline shaders remain separate.
The four integrated SPVs are byte-identical to the tested private variants;
all 177 pre-existing SPVs are unchanged. No lower-precision tier is enabled.

## Numerical evidence and the initial tolerance failure

The initial five-shape / four-format / four-implementation sweep produced
80 cells. All 60 candidate outputs were byte-identical to their original
GPU baselines, including poisoned output padding; all output values were
finite. However, the original Q6_K shader and all three variants failed the
same ad-hoc FP64 tolerance on the same large shape. That run remains **FAIL**
in [the original raw record](results-halo-gemm-first.json), not relabeled.

A focused replay located the tolerance miss at row 11026 / token 138 of
M=12288, K=2560, N=512:

| Calculation | Value |
| --- | ---: |
| Original and all candidate GPU shaders | 0.43466544151306152 |
| CPU F32, same blocked fused multiply-add order | 0.43466544151306152 |
| CPU FP64 dot | 0.43469850975088775 |
| Absolute difference from FP64 | 0.0000330682378262 |
| Original tolerance at this output | 0.0000286939701950 |

The sum of absolute products was 5129.95, versus a final value near 0.435.
The CPU calculation with the same F32 fused/block accumulation reproduces
the GPU value exactly. The original fixed FP64 residual bound therefore
rejects this correctly reproduced F32 calculation; this is not a numerical
change introduced by the compact candidate. Diagnostic source, logs, and
hash-bound controller metadata remain in `/home/ryan/qk-gemm-tune-qxmgg1/`.

The next test uses an explicit **CPU F32 blocked-FMA agreement plus exact
GPU-baseline gate**, while retaining FP64 errors and original-tolerance miss
counts as separate diagnostics. It covers the actual quantization formats
and projection shapes read from this model's layer 0 and layer 3 metadata:
13 shape/format combinations at 64, 128, 256 and 512 input rows, in ABBA
order (original, compact, compact, original), eight timed dispatches per
cell. All **208 cells passed**, with zero CPU F32 mismatches. Every candidate
output and every repeated original output was byte-identical to its first
original output. The CPU reference checks every small-shape output and 257
deterministically sampled outputs of larger shapes; full GPU-baseline
comparison covers every output, not just those samples.

The retained FP64 diagnostic recorded 112 tolerance misses across the
repeated sampled comparisons (84 Q5_K, 28 Q6_K), with zero F32-emulation
mismatches. No full-model logit or quality threshold has been relaxed.

## Operator performance and shape selection

Native-style type 0 / heap 1 DEVICE_LOCAL allocations were used on the
integrated 8060S, PCI 0000:c1:00.0. Staging uploads/readbacks are outside GPU
timestamps; model weights were not loaded, and no external-GPU compute was
used. These are synthetic operator measurements, **not model throughput**.

Medians of the two timed cells per implementation at 512 input rows:

| Format | Output rows M | Inner width K | Original (ms) | Compact (ms) | Operator speed ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q5_K | 12288 | 2560 | 7.668 | 5.627 | 1.36× |
| Q5_K | 6144 | 2560 | 3.732 | 2.525 | 1.48× |
| Q5_K | 2560 | 6144 | 3.508 | 2.398 | 1.46× |
| Q6_K | 10240 | 2560 | 6.997 | 6.409 | 1.09× |
| Q8_0 | 2560 | 640 | 0.385 | 0.314 | 1.23× |
| Q5_1 | 10240 | 320 | 0.846 | 0.670 | 1.26× |

Small/skinny projections often had no gain; Q8_0 M=2560/K=640 at 64 rows
was slower. The opt-in policy therefore selects only the six shapes above.
It requires at least 256 input rows for Q5_K M=2560/K=6144 and Q8_0
M=2560/K=640; the other four shapes use it at 64–512 rows. All other shapes
retain baseline kernels. The separately requested cooperative-matrix tier
takes precedence; it is disabled throughout this F32 experiment.

Raw model-shape data and build/source/shader hashes:
[results-halo-gemm-model-shapes.json](results-halo-gemm-model-shapes.json).
The integrated policy's parser, shape selection, boundary cases, and
fallbacks are covered by the `qwen4-gemm-policy` CTest.

Reproduce the operator check in an exclusive, drained Halo window:

```bash
c++ -O2 -std=c++17 -pthread tests/halo_gemm_model_shapes.cpp -lvulkan -o /path/to/halo-gemm-model-shapes
QK_SHADER_DIR=/path/to/build-halo/shaders /path/to/halo-gemm-model-shapes 8
```

The public harness uses the integrated shader filenames with the same
binary contents. Original runs and diagnostic binaries are preserved in
the private experiment directory. Do not run this alongside a serving model.

## Integrated full-model gate

The same-build compact and baseline replays completed successfully, each
with the exact shared 16,384-token prefix plus 128 teacher positions at
context 32,768. Both reset/repeated the entire prefix and tail exactly.
Attention remained serial, with scalar F32 prefill, F32 KV and no MTP.
The only launch/configuration difference was `QK_FLASH_GEMM`.

All 129 saved full-vocabulary logit rows were **byte-identical**, not merely
within tolerance. All 128 ABI-level greedy IDs agreed with the saved row
argmaxes; maximum relative RMS and KL were zero. The pre-existing numerical
helper and its `1e-5` RMS / zero-flip gate were unchanged. Both complete
128,133,120-byte dumps have SHA-256
`3042e28a258ee2bca38ee1d9ee2e39cc68bb584d92ce42076dc7378f41852a01`.

The audited library SHA-256 is
`2edd589b4f36595ea8619606750dbd35560b2d4950d9f6f87774bcf2d8f745b2`;
all 181 shader hashes match between manifests and the frozen build. Logs
confirm baseline dispatch in the control and an actual compact Q5_1
M=10240/K=320/512-row dispatch in the candidate. Neither run hit the memory
watchdog. The dump also matches the prior serial-attention build, but that
cross-build observation is separate from this matched-build gate.

Compact ran 06:53:03–07:01:31 UTC; baseline 07:04:22–07:13:47 UTC on
2026-09-12. Their replay body times were 416.444 and 459.113 seconds,
respectively. Those include two prefills, two tails, and readbacks, and
are **not** an API throughput benchmark or an established speedup.

The [raw gate](results-halo-gemm-compact-long-gate.json) binds controller,
manifest and log hashes. Original logs/dumps and the strict pair auditor
are preserved in `/home/ryan/qk-compact-checks-RvIhNE/`.

Next gate: the HTTP/Claude-tool suite and request-level prefill A/B at
128, 512, 2048, 8192 and 16384 tokens with 128-token generation. Both modes
use the same build, serial attention, precision and fixture. The read-only
`compare_native_gemm.py` auditor requires complete matched workloads,
identical outputs, the bound bit-exact gate and explicit actual GEMM
dispatch evidence. Until repeated API measurements establish a model-level
benefit, keep `baseline` as the serving default.
