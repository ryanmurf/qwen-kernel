# Halo review follow-up

2026-09-12. The independent Opus 5/max-effort review found no proven kernel
correctness bug, but identified missing boundary coverage and insufficient
API performance evidence. Defaults remain baseline GEMM/batch attention and
serial decode. This follow-up does not change any runtime kernel or shader.

## Completed operator checks

- **GEMM: 312/312 cells passed**, including added N=65 and N=309 for every
  one of the 13 model shapes, in addition to 64/128/256/512. Four cells per
  case, A-B-B-A, each averaging four timed dispatches. All candidate outputs
  and padding are baseline-bit-exact and sampled CPU blocked-F32-FMA checks
  pass. Legacy FP64 residuals remain recorded, not used as a replacement
  performance/correctness claim. Run 14:52:37–14:53:47 UTC.
- **Attention: 96/96 cells passed** after re-poisoning the output between
  warm-up and timing. All output/padding bits match; no nonfinite live
  outputs, padding changes, or sampled FP64 tolerance misses. The original
  cases and A-B-C-C-B-A order are unchanged. Run 14:56:59–14:59:01 UTC.

JSON cell lines are extracted verbatim from the original logs:
[GEMM](results-halo-review-gemm.jsonl),
[attention](results-halo-review-attention.jsonl).
These are isolated operator checks, not model/API speed measurements.

Both runs used Halo PCI0000:c1:00.0 only, pure DEVICE_LOCAL buffers with
transfers outside GPU timestamps, and 1G/2G cgroup high/max with no allowed
swap. Controllers exited zero without memory aborts; DRM observations show
no external GPU engine work. Original logs, controllers and the full build
manifest are in `/home/ryan/qk-review-followup-8dzGtw/`.

## Build reproduction

The new `QK_HALO_OPERATOR_TESTS=ON` CMake option exposes compile-only targets
`halo_gemm_model_shapes` and `halo_batch_attn_operator`, including the QB16
control shader. They are excluded from normal builds and never run in CTest.

```bash
cmake -S . -B /path/to/fresh-build -DQK_HALO_OPERATOR_TESTS=ON -DBUILD_TESTING=OFF
cmake --build /path/to/fresh-build --target halo_gemm_model_shapes halo_batch_attn_operator -j4
```

Execution still requires an exclusive, drained Halo window with memory and
DRM watchdogs; building these targets does not grant a safe execution window.
The fresh build contains 180 current server shaders plus the QB16 test
control. All 180 server shaders match the old frozen build byte-for-byte;
all 181 new files pass `spirv-val --target-env vulkan1.2`. The two historical
SPVs without build rules are absent from the fresh build, not deleted from
historical evidence. The serving library remains unchanged at SHA-256
`0b3efa4732d1dad9b2ebdbc7fc2688a6501a32672a146953654739ad3cd3bdd3`.
Operator build-manifest SHA-256:
`bc4972908b41a632892920777c0a6f07aab3e092c471760b0e3552f969a8c97c`.

## Full-model prefill-output gate (passed)

`tests/gpu_qwen4_prefill_gate.py` captures full logits and checks ABI argmax
at the last position of every prefill chunk, plus a clean row and every
teacher-tail position. It repeats all saved rows after reset and requires
finite, byte-identical results. It is an explicit same-build baseline/vec4
or baseline/compact comparison; the older decode gate and its artifacts are
unchanged. Seventeen CPU tests cover corruption, identity, missing rows,
partial chunks, reset/repeat, nonfinite values and inherited unsafe knobs.

Current workload: the existing 16,512 IDs, a 16,331-token prefix ending in
a **459-token partial chunk**, and 181 teacher-tail positions. Each run
saves 214 rows (1 clean + 32 prefill boundaries + 181 tail). The candidate
ran 14:59:49–15:07:22 UTC and passed all self-checks: all 214 rows repeated
bit-exactly, all were finite and ABI argmax matched at every saved row.
Its replay body took 376.024 seconds, not an API throughput measurement.
The same-build baseline ran 15:09:43–15:19:16 UTC and passed the same checks;
its body took 466.386 seconds. Both controllers completed with no error or
identity change. Baseline had approximately 10 MiB of observed cgroup swap;
this is not reported as a zero-swap full-model pair. The candidate's sampled
cgroup swap was zero. No memory guard aborted either run.

The [strict paired gate](results-halo-review-prefill-vec4-gate.json) **passed**:
all 214 saved full-vocabulary rows are byte-identical, with every recorded
ABI greedy ID checked against its logit row. Both 212,561,920-byte dumps have
SHA-256 `3201fd0ca69a34158ae6578a33e9e45244a0dd4ecd9dd90c2132fa0d207efe79`.
An additional [controller audit](results-halo-review-prefill-controllers.json)
confirms matched commands except policy/unit/output, one frozen build, and
the candidate's actual QB8 dispatch. Boundary logits do not cover every
intermediate prefill token; this is kernel equivalence, not a general quality
eval or a request-throughput measurement. Compact GEMM has not run this new
prefill-output gate; its prior full-model gate remains separate.

The public producer/auditor replaces private-only gate logic for this new
experiment. In separately guarded exclusive windows, run its `dump` command
once with `--policy baseline` and once with `--policy vec4`, each with the
same `--library`, `--shader-dir`, `--ids`, `--tail 181`, `--ctx 32768`, and
distinct `--out` files. `--confirm-exclusive` is an acknowledgement, not a
resource guard. Preserve the launch commands, actual backend logs, resource
watchdog results and manifests as well as the dumps. After both complete:

```bash
python3 -E tests/gpu_qwen4_prefill_gate.py compare /path/to/baseline.f32 /path/to/vec4.f32 --policy vec4 --output /path/to/new-gate.json
```

This command is read-only on the dumps and refuses to overwrite its output.

## Repeated API comparison (passed; mixed performance)

`bench/compare_native_batch_attention.py` requires both the prior bit-exact
16K+128 gate and this new partial-prefill-output gate, as well as matched
F32 settings, serial decode, baseline GEMM, complete HTTP checks, launch
and actual dispatch evidence, source/build hashes, token counts and outputs.
Its eight new CPU tests pass; older comparator tests remain unchanged.
The baseline ran 15:21:23–15:54:11 UTC, followed by vec4 at
15:54:37–16:22:35 UTC. Each completed three repetitions per size
(128/512/2048/8192/16384), 128 generated tokens, the existing shared fixture
and all eight HTTP checks. Both complete 30-cell matrices passed the
[strict paired audit](results-halo-batch-attn-api-r3-comparison.json):
all prompt/output token and text hashes match, every stream has the exact
requested length and a coherent counting prefix, no prompt-prefix reuse,
and actual environment/dispatch evidence matches except the selected
attention policy and allowed run identity. This is not a broad quality eval.

Medians, three samples per mode. TTFT is client-observed time to the first
generated token; decode rate excludes that first-token delay. Negative TTFT
change is better; positive decode-rate change is better.

| Prompt tokens | Baseline TTFT (s) | Vec4 TTFT (s) | TTFT change | Baseline decode (tok/s) | Vec4 decode (tok/s) | Decode change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.213 | 1.213 | +0.01% | 32.675 | 32.664 | -0.04% |
| 512 | 3.293 | 3.206 | -2.65% | 30.536 | 30.416 | -0.39% |
| 2048 | 13.847 | 13.285 | -4.06% | 23.302 | 23.118 | -0.79% |
| 8192 | 72.180 | 62.642 | -13.21% | 11.389 | 11.589 | +1.75% |
| 16384 | 198.991 | 159.245 | -19.97% | 7.190 | 6.868 | -4.47% |

At 16K, baseline TTFT ranged 197.297–199.647 seconds and vec4
157.053–159.506 seconds. The separate one-output-token prefill probes had
medians 200.324 versus 158.194 seconds (21.03% lower). These are request
measurements, not the roughly 1.9x isolated attention-operator speedup.
The 16K decode ranges were 7.122–7.270 versus 6.568–7.074 tok/s: every
candidate sample was below every baseline sample in this ordering. At 8K,
candidate decode ranged 10.569–11.616 tok/s, including the slower first
sample; no sample was discarded. Full ranges and raw cells are retained.

**No default promotion.** Long-context prefill improved in this repeated
pair, but long-context decode regressed, and only baseline-then-vec4 order
was measured. A reverse-order check is still required. The selected knob
changes prefill, not the serial decode dispatch; the cause of the decode
difference is unproven. Three repetitions do not establish statistical
significance, general model quality, or performance on other workloads.
This is also not a new llama.cpp/F16 or MTP comparison; those precision and
backend differences remain separate from this same-build F32 A/B.

The [controller audit](results-halo-batch-attn-api-r3-controller-audit.json)
checked successful HTTP/benchmark/server/stop statuses, matching launch and
source identities, unchanged library/server/shaders, admitted before/after
memory snapshots, and recorded server DRM work only on Halo. Both runs had
zero **sampled cgroup** swap; the host still had pre-existing swap usage.
Halo GTT drained to 155,525,120 bytes between runs and again after vec4.
Temperature samples, including loading, HTTP and benchmarking, ranged
34–74 C for baseline and 38–76 C for vec4. Median sampled GPU clocks were
2892 and 2891 MHz. Coarse telemetry is not per-request attribution and does
not prove thermal throttling or explain the decode difference. No clocks,
power limits, kernel, precision, model, or runtime build changed.

Verbatim raw matrices and terminal controller records:

- [Baseline matrix](results-halo-batch-attn-api-baseline-r3-0b3efa47.jsonl)
  and [controller](results-halo-batch-attn-api-baseline-r3-controller.json).
- [Vec4 matrix](results-halo-batch-attn-api-vec4-r3-0b3efa47.jsonl)
  and [controller](results-halo-batch-attn-api-vec4-r3-controller.json).

Original logs, metadata, the guarded API controller and full-model producer
evidence remain in `/home/ryan/qk-review-followup-8dzGtw/`. The public API
comparison is reproducible without those private paths:

```bash
python3 -E bench/compare_native_batch_attention.py bench/results-halo-batch-attn-api-baseline-r3-0b3efa47.jsonl bench/results-halo-batch-attn-api-vec4-r3-0b3efa47.jsonl --long-gate bench/results-halo-batch-attn-vec4-long-gate.json --prefill-gate bench/results-halo-review-prefill-vec4-gate.json
```

## Serving restored

After both benchmark servers stopped and Halo memory drained, the guarded
restore helper brought baseline serving back on loopback 8194 and the
trusted-LAN gateway on **8091**. Actual process environment was checked:
Halo only, baseline GEMM and batch attention, serial decode, F32, context
32768, prefill chunk 512, no table warming/cooperative matrices/profiling.
No XTX worker was launched and nothing was enabled at boot.

All [eight direct-backend HTTP checks](results-halo-review-restored-http8194.jsonl)
passed. The [gateway checks](results-halo-review-restored-http8091.jsonl)
passed all seven applicable checks plus coherent streaming; the direct-engine
disconnect-cancellation check was intentionally excluded with `--skip-cancel`
because this gateway buffers requests. These are restoration smoke checks,
not additional cells in the API performance comparison. Test commands:

```bash
python3 -E tests/native_flash_http.py --url http://127.0.0.1:8194
python3 -E tests/native_flash_http.py --url http://127.0.0.1:8091 --skip-cancel
```

## Next measurable target (not implemented)

Read-only inspection found that `Qwen4Graph::headBatch` computes the full
vocabulary projection and greedy ID for every prefill position, in 64-row
tiles. The local server's `Phase::Prefilling` retains only `ids.last()`.
This may leave avoidable head work, but its share of actual prefill time has
not been measured. No speedup is claimed and no runtime code was changed.

The existing `qk_stage_run` ABI explicitly returns an ID for every position;
silently dropping those results would break callers and numerical tests.
A future experiment should first measure head cost in a separate profiling
window, then consider an explicit last-output serving entry point with the
old all-ID path preserved. It must retain final full logits/top-k, state
continuation, partial chunks, reset/repeat, cancellation, and API/tool-use
behavior. Changing accumulation order or using the serial GEMV in place of
the final batched tile would require a new numerical comparison. Profiled
timings must remain separate from uninstrumented API measurements.
