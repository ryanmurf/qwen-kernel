# Native Flash: opt-in last-output prefill

Update, September 12: the [complete full-model campaign](RESULTS-halo-last-head-full.md)
passed all 498 exact-logit comparisons and matched HTTP/Claude checks.
Median whole-model first-token latency fell 2.8–5.0%; see the report for
the 16K decode tradeoff and full ranges. Serving is restored with the flag
explicitly enabled; the source-level default remains off. The earlier
admission block below is historical; resource cleanup was approved.

The original September 12, 2026 experiment on Max, documented below, used
the **last two layers plus output head**, not the entire model. At that
point, the unchanged 24 GiB admission guard rejected approximately 20.6 GiB
available RAM and unused-TTM cleanup was awaiting approval. That historical
block is resolved; the full-model validation and restored serving above
supersede it.

## Change

Local prefill consumes only the final prediction, but the native graph was
projecting all prefill rows over the 248,320-entry vocabulary. The new
optional `qk_stage_run_last` ABI keeps the **same final 64-row vocabulary
tile** and skips its predecessors. Final partial tiles retain their original
row count, input offset, shader and reduction order. HC preparation, all
transformer layers, KV/recurrent updates and final full logits remain intact.

At batch 512 this removes seven of eight vocabulary projections/argmaxes.
At 64 or fewer rows it removes none. Multi-chunk calls retain one head tile
per internal chunk; this is not yet a no-head intermediate-prefill design.

The original `qk_stage_run` still returns every position's ID. Decode,
remote split frames, precision, attention policy, model files, native MTP
and native snapshot support are unchanged. Rust local prefill opts in only
with `QK_FLASH_PREFILL_LAST=1`; the restore helper forwards that flag when
explicitly set. Libraries without the optional symbol fall back to the old
ABI. Only `-7` (unsupported, no mutation) permits retry; execution failures
never replay a potentially state-mutating forward. Default is off.

## Measured partial-stage result

Same newly built library, Halo `0000:c1:00.0`, layers `[46,48)` (one GDN and
one full-attention layer) plus the real Q6_K vocabulary head. Synthetic
deterministic F32 residual inputs, context 2048, batch capacity 512, F32
baseline GEMM/attention, serial decode, no cooperative-matrix or PLE warming.
This is not text generation, coding quality, full-model prefill throughput,
or a comparison with Nathanw/llama.cpp. No XTX compute.

Each shape is warmed on both paths. Six observations per arm follow three
ABBA cycles in one process; profiling is off. Wall time covers the complete
stage ABI call, including execution/synchronization/download, but excludes
Python full-logit comparison and top-k validation.

| Input rows | All-ID median | Last-output median | Partial-stage speedup |
| ---: | ---: | ---: | ---: |
| 128 | 104.09 ms | 82.45 ms | 1.26x |
| 512 | 322.64 ms | 154.85 ms | 2.08x |
| 1024 (two chunks) | 653.99 ms | 318.39 ms | 2.05x |

At 512 rows the ranges were 311.15–334.37 ms and 148.48–160.32 ms.
An earlier exploratory run also passed and measured 306.16 vs 153.24 ms;
the table uses the final rebuilt library and frozen harness. No failed GPU
gate is omitted. Do not extrapolate the roughly 2x partial-stage speedup to
all 48 layers: most of those layers' work is unaffected.

Separate diagnostic timestamps show eight Q6_K vocabulary GEMM/argmax
pairs becoming one: 199.489 ms vs 21.635 ms in that profiled pass. These are
instrumented, single-pass attribution numbers, not the unprofiled medians.

## Correctness and compatibility

- Exact final logits across all 248,320 entries at 13 initial lengths:
  1, 2, 63, 64, 65, 127, 128, 129, 309, 511, 512, 513, 1024.
- After each initial run, continuation sizes 1, 2 and 63 also match exactly:
  52 checked logit rows, including serial/batch transitions and resets.
- Top-20 IDs and values agree with independently sorted full logits;
  output canaries remain intact. Zero residuals and invalid
  slot/count/base/context/null/nonfinite-input checks pass.
- A separate run of the untouched old library matches all 52 final-logit
  hashes and all 52 complete all-ID buffer hashes from the new old-ABI path.
  All 180 fresh shader files match their preserved counterparts. The old
  directory has two additional historical override shaders, neither selected
  here: `gemv_q6_k_v1.spv` and `moe_down_q8_routed_v1.spv`.
- C++ Release build and 6 CTest suites pass. Rust debug tests (44 tests) and
  release build pass. New CPU tests cover old/new ABI availability, `-7`
  fallback, no retry on `-6`, multi-frame local prefill, sampled continuation
  and reset. The offline auditor also rejects ten tampered evidence cases.

The partial model occupies 5.016 GiB of Vulkan allocations, including
4.410 GiB of weights. Each test used a dedicated transient unit with
MemoryHigh=8G, MemoryMax=10G, MemorySwapMax=256M, NoNewPrivileges and no core
dumps. Final gate cgroup peak was 918,581,248 bytes; swap peak was zero and
all memory-event counters were zero. Cgroup memory is **not** the complete
GPU allocation accounting. All test processes exited successfully. No
driver/kernel/clock/power changes, global cache flush, GPU reset, TTM shrink
or model replacement occurred during this partial-stage experiment. The
full-model admission guard was not lowered.

## Evidence and reproduction

- [GPU harness](native_last_head.py), [offline audit](audit_native_last_head.py).
- [Final gate and timings](results-halo-last-head.jsonl),
  [old-library reference](results-halo-last-head-reference.jsonl).
- [Profile metadata](results-halo-last-head-profile.jsonl),
  [raw GPU profile](results-halo-last-head-profile.log),
  [audit result](results-halo-last-head-audit.json).
- Private builds, launch-unit logs and exploratory run:
  `/home/ryan/qk-last-head-IIfPmw`.

New native library SHA256:
`068369b87806608ed7d00efb2b72d3bd3e0f19abfd0ed67d7f452b8e952d8931`.
New release server SHA256:
`10bf29d115627fdf35a9305e0e2e1a9de1dea5a6bd8357627aad0c051fa1925a`.
Those exact tested library/server bytes are now installed in `build-halo`
after the full-model campaign. Verified original backups remain in
`/home/ryan/qk-last-head-full-VjRrMt/rollback`; owner-only permissions were
preserved. The private experiment builds remain available as evidence.

Offline check, no GPU or model load:

```bash
python3 bench/audit_native_last_head.py \
  bench/results-halo-last-head.jsonl \
  bench/results-halo-last-head-reference.jsonl \
  bench/results-halo-last-head-profile.jsonl \
  bench/results-halo-last-head-profile.log
python3 -m unittest discover -s bench -p test_native_last_head_audit.py
```

For a new GPU experiment, build out-of-tree and run `native_last_head.py`
inside its documented bounded cgroup with exactly the QK environment in
the raw metadata (adjust only shader/library/model paths for that build).
Require drained GPUs, no other inference process and 12 GiB available for
this approximately 5 GiB partial-stage load; the harness enforces these
conditions. Run `--reference-only` against the preserved library separately,
and `--profile-only` with `QK_FLASH_PROFILE=2` separately. Never overlap them.

The full-model and HTTP gates above are complete. Last-output prefill is
enabled for the current prefill-focused testing configuration, not made a
universal default. Longer-output throughput and reverse-order repeats remain
useful follow-ups; the source default and explicit flag retain rollback.
