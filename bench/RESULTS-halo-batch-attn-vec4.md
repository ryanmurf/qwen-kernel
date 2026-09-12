# Halo vector-LDS F32 batched attention

Status, 2026-09-12: **96 operator checks and the same-build full-model
replay comparison passed bit-exactly. No API speed claim or default
promotion.** Review follow-up adds direct prefill-output and partial-prefix
checks; see [follow-up results](RESULTS-halo-review-followup.md).
`QK_FLASH_ATTN_BATCH=vec4` selects the new path on Strix Halo only. Unset it,
or use `baseline`, for the unchanged default. Cooperative-matrix selection
takes precedence. GEMM and single-position decode choices are independent;
the full-model test holds those at baseline and serial, respectively.

## Change and correctness boundary

The candidate stores Q/K tiles in padded `vec4` shared-memory arrays and
loads four components together. It still accumulates their products in the
original dimension order. TK remains 16, preserving the existing online
softmax and V-accumulation ordering. Reducing queries per workgroup from 16
to 8 also reduces shared storage and per-thread query accumulators. The host
retains its original query-tile budget/alignment and dispatches twice as
many 8-query workgroups. F32 inputs, KV and arithmetic remain unchanged.

The model-free experiment compared baseline, vec4/QB16 and vec4/QB8 in
symmetric A-B-C-C-B-A order at each of 16 cases, with four timed dispatches
per cell. Cases cover partial query blocks, nonzero query offsets, the
actual 128-query tiles at 16K and 64-query tiles at 32K, scaled-query softmax
stress, and a decreasing final length to expose stale state. Unused KV and
query rows and all output padding are poisoned with NaNs.

All **96 cells passed**. Every candidate output and repeated baseline
output matched every bit of the first baseline, including padding. There
were zero nonfinite live outputs, padding changes, or FP64-reference
tolerance misses. The independent FP64 reference checks the first/last
live queries at heads 0/11/12/23; baseline-bit comparison checks the entire
output buffer, not just the reference samples. The fixed FP64 tolerance is
`2e-5 * (1 + abs(reference))`. No full-model logit threshold was changed.

This test validates the isolated operator on synthetic inputs. It does
not replace full-model, HTTP/tool-use, or request-level performance gates.

## Operator measurements

Integrated Radeon 8060S only, PCI 0000:c1:00.0. Buffers use native-style
pure DEVICE_LOCAL memory, type 0/heap 1; staging transfers are outside GPU
timestamps. No model weights were loaded. DRM auditing showed no external
GPU engine work. Both return code and the controller's coverage gate were
zero/success, with no memory abort. Run 12:30:48–12:32:50 UTC.

Medians of the two cells per implementation; each cell averages four
timed dispatches. Times cover one attention dispatch, not a whole model.

| Cached prefix base | Chunk rows | Query offset | Query tile | Baseline (ms) | Vec4/QB16 (ms) | Vec4/QB8 (ms) | Baseline / QB8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 512 | 0 | 512 | 5.940 | 3.603 | 2.678 | 2.22× |
| 15872 | 512 | 0 | 128 | 127.368 | 73.710 | 65.317 | 1.95× |
| 15872 | 512 | 384 | 128 | 124.190 | 76.323 | 66.418 | 1.87× |
| 32256 | 512 | 0 | 64 | 147.668 | 90.598 | 78.196 | 1.89× |
| 32256 | 512 | 448 | 64 | 148.357 | 93.290 | 80.461 | 1.84× |

QB8 was faster in all 16 tested cases, so only it is integrated. These
operator speed ratios must not be presented as API/model speed ratios.
The preceding compact-GEMM and ordered-decode experiments demonstrate why
operator timing alone is insufficient for default promotion.

The combined ratio contains two changes. At base 15872/query offset 0,
baseline to vec4/QB16 is 1.73x, and QB16 to QB8 adds 1.13x. At base 32256/
offset 0 those factors are 1.63x and 1.16x. QB8 doubles the query workgroup
count and approximately doubles logical KV reads at long prefixes. Physical
DRAM traffic was not measured; caches and causal lengths matter. Re-measure
at the deployed context rather than extrapolating the isolated ratio.

## Reproduction and build identity

[Raw cells, controller and build/source hashes](results-halo-batch-attn-vec4.json).
Original private artifacts remain in `/home/ryan/qk-batch-attn-tune-fTSti4/`.
The tested binary SHA-256 is
`7c1106ba44a550471ee064b46330173b0d5e13cc9132dd2d35982b32b6ce4a1f`.
The integrated QB8 SPV is byte-identical to the private tested variant:
`478984813b900773ca8b60625fdcb48d2b8a8ac658ee63bbfe44624320830a90`.
All 181 previous SPVs remain byte-identical. The 182-shader integrated build
has library SHA-256
`0b3efa4732d1dad9b2ebdbc7fc2688a6501a32672a146953654739ad3cd3bdd3`.
Five CTests, including the new opt-in/query-block policy test, pass; the
unchanged full-model numerical helper's 31 CPU tests also pass.

The original public harness changed only include paths and the integrated
QB8 shader filename. Review follow-up also re-poisons output between warm-up
and timing; the updated harness passed all 96 cases in a separate run.
In an **exclusive, drained Halo
window**, prepare a fresh private output directory with these commands
(replace `/path/to/operator-dir` with that directory):

```bash
c++ -O2 -std=c++17 -pthread tests/halo_batch_attn_operator.cpp -lvulkan -o /path/to/operator-dir/operator
glslc -O --target-env=vulkan1.2 shaders/fa_attn_batch.comp -o /path/to/operator-dir/fa_attn_batch.spv
glslc -O --target-env=vulkan1.2 shaders/fa_attn_batch_vec4.comp -o /path/to/operator-dir/fa_attn_batch_vec4.spv
glslc -O --target-env=vulkan1.2 -DTEST_QB=16 tests/fa_attn_batch_vec4_experiment.comp -o /path/to/operator-dir/fa_attn_batch_vec4_q16.spv
QK_SHADER_DIR=/path/to/operator-dir /path/to/operator-dir/operator 4
```

The private run also validated all three SPVs with `spirv-val`, enforced
1 GiB/2 GiB cgroup memory high/max with zero allowed swap, and used a live
host-memory/DRM watchdog. Never run this harness alongside a serving model.

## Completed same-build full-model gate

The integrated vec4 replay completed at 12:44:05 UTC after starting at
12:36:35. It used the existing exact 16K prompt plus 128 teacher positions,
context 32768 and all 48 layers. Reset and a complete repeated prefix/tail
were exact; the controller exited zero without a memory abort or library
change. Logs confirm actual QB8 dispatch. Serial decode and baseline GEMM
isolate this candidate.

The complete 128,133,120-byte logit dump is byte-identical to both the
previous baseline and the new same-build baseline. Both hashes are
`3042e28a258ee2bca38ee1d9ee2e39cc68bb584d92ce42076dc7378f41852a01`.
The strict [same-build audit](results-halo-batch-attn-vec4-long-gate.json)
passed: all 129 saved full-vocabulary rows are identical, all 128 ABI greedy
IDs match, and both runs passed complete reset/repeat checks. These rows are
the clean single-position row and decode tail; they do not directly check
prefill-boundary logits, which is addressed by the new follow-up harness.
The replay body took 356.646 seconds, including two prefills, two tails
and readbacks. It is not an API throughput measurement.

The same-build baseline control ran 12:45:08–12:54:21 UTC, exiting zero with
no memory abort or library change. Its replay body took 444.375 seconds;
neither combined replay-body time is an API throughput measurement. The HTTP
suite and repeated request-level A/B remain required. Original manifests,
logs, controller records and auditor are in `/home/ryan/qk-vec4-checks-fdyHmg/`.
Defaults remain unchanged throughout validation.
