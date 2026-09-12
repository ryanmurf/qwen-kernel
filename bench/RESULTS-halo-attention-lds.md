# Halo attention LDS follow-ups: no promotion

September 12, 2026, Max. Both isolated experiments passed all **128 operator
cells each**, with exact baseline bits, no nonfinite live output, unchanged
poisoned padding and no FP64-reference tolerance misses. Neither showed a
consistent new performance win, so **none of these shaders was integrated
or enabled for serving**. These are negative/indeterminate operator results,
not full-model benchmarks.

The probes ran only after both [last-output HTTP arms](RESULTS-halo-last-head-full.md)
had completed and stopped. Each used Halo `0000:c1:00.0` alone, native-style
DEVICE_LOCAL placement, GPU timestamps excluding staging transfers, and
four timed dispatches per cell after a warm-up. Four implementations ran in
A-B-C-D-D-C-B-A order at each of 16 cases. The tables show medians of the two
cells per implementation (each cell averages four dispatches). This is one
counterbalanced sweep, not a broad statistical characterization.

## Decode: same-order reductions and K staging

The shuffle variant finishes the original max/sum binary tree inside one
subgroup, preserving its pairwise arithmetic order. K staging cooperatively
loads 32 dimensions at a time through a padded 256-by-33 F32 shared array,
then retains the original per-key dot-product order. The fourth variant
combines them. No precision, softmax tile, V-accumulation order or gating
formula changes. The test covers short/boundary/long lengths, two independent
slots, scaled queries, decreasing length and NaN-poisoned unused KV/output.
FP64 samples cover heads 0/11/12/23 in every active slot; bit comparison
covers the entire output and guard buffer.

| Live keys | Original (ms) | Shuffle (ms) | K staging (ms) | Both (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 16384 | 5.861 | 6.032 | 11.549 | 12.189 |
| 32768 | 12.103 | 13.934 | 23.470 | 26.387 |

K staging was substantially slower here. Shuffle also failed to show a
consistent benefit; isolated favorable cells do not justify promotion.
[All cells, build/source hashes and resource summary](results-halo-attention-decode-lds.json).
Original private sources, binary and logs:
`/home/ryan/qk-attn-decode-lds-iVmGa0`.

## Batched prefill: score-array padding

The only candidate change was `sc[QB][TK]` to `sc[QB][TK+1]`, tested
separately on scalar QB16 and the existing vec4 QB8 implementation. TK
remained 16; query/key tiles, arithmetic and dispatch geometry were unchanged.
The existing batch-attention harness was adapted to the four-mode ordering;
it retains partial tiles, nonzero query offsets, scaled queries, decreasing
length, poisoned unused rows and full-buffer exact comparison.

Each row below is one query tile within a 512-position chunk:

| Cached prefix / query offset | Scalar (ms) | Scalar padded (ms) | Vec4 (ms) | Vec4 padded (ms) |
| --- | ---: | ---: | ---: | ---: |
| 15872 / 0 (128 queries) | 117.619 | 122.695 | 59.625 | 65.051 |
| 15872 / 384 (128 queries) | 123.348 | 121.864 | 62.737 | 60.797 |
| 32256 / 0 (64 queries) | 143.514 | 146.311 | 76.695 | 73.362 |
| 32256 / 448 (64 queries) | 147.465 | 146.639 | 72.668 | 75.556 |

Padding produced mixed results, including regressions. The already-known
vec4-versus-scalar benefit is not a new padding gain and does not resolve
the existing vec4 API/reverse-order questions.
[All cells, build/source hashes and resource summary](results-halo-attention-scorepad.json).
Private sources, build, logs and the guarded `run.py` controller:
`/home/ryan/qk-attn-scorepad-XYEa8w`.

## Safety and identity

Both original-control SPVs are byte-identical to the corresponding shaders
in the frozen full-model build. All four SPVs per experiment passed
`spirv-val`; controller source, included project headers, binaries and
shader hashes stayed unchanged. Both units used 1 GiB memory-high / 2 GiB
hard limits, zero allowed swap, NoNewPrivileges and disabled cores. Both
controllers, GPU programs and resource observers exited successfully and
stopped their own units. No serving model ran alongside either probe.

Decode/batch cgroup peaks were 290250752 / 197001216 bytes; swap and all
observed memory-event counters were zero. Sampled minimum available RAM was
26.736 / 26.840 GiB, maximum GPU temperature 42 / 45 C, with no watchdog
alarms or observed non-Halo engine work. The decode probe was short: only
six approximately three-second resource samples. These are sampled server
fdinfo observations, not a whole-host exclusivity proof, and cgroup memory
does not account for every GPU allocation.

The exact source variants and compiler commands remain in the private
directories above. Do not rerun their completed controllers in place or
overwrite evidence; reproduce in a fresh directory and exclusive GPU window.
No full-model gate was run for these rejected candidates.
