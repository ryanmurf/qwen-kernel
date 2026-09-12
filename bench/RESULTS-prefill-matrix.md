# Max: same-token prefill matrix

Native baseline completed on 2026-09-11 (03:24:48 to approximately 03:55 UTC
on September 12). This is the full Qwen3.8-Flash-Next-Uncensored-Q5_K_M model
on the Strix Halo only, 32768 context capacity, one sequence, F32 native
math/KV, 512-token prefill chunks, no MTP, no cooperative-matrix tier, no
full PLE warming; row prefetch and the verified decode fusions are enabled.
Kernel revision: `0649101`. No native binary or shader changed during the run.

Median of three requests per cell, 128 requested/count-verified output tokens:

| Prompt tokens | Prefill + one token (s) | First token (s) | Decode (tokens/s) | Decode min..max |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.209 | 1.205 | 32.76 | 32.50..32.82 |
| 512 | 3.144 | 3.145 | 30.56 | 30.54..30.60 |
| 2048 | 13.530 | 13.591 | 23.32 | 23.13..23.43 |
| 8192 | 69.956 | 70.076 | 12.62 | 12.33..12.74 |
| 16384 | 194.483 | 193.133 | 8.05 | 7.33..8.62 |

All 15 decode requests produced the same valid counting prefix, all 128
tokens, with identical token/text hashes. Every one-token probe returned
token 16 (`1`). This sanity check is not a general quality evaluation.
The 16K decode spread is material; do not present its median as an exact rate.

These are client-observed wall times. The one-token probe includes request
handling and first-token work; it is not pure GPU prefill. Native
`prompt_ms` is zero/unavailable, and its reported predicted time includes
prefill, so the table uses measured streaming timestamps and token counts.
Each decode request follows a separate same-prompt one-token probe.
`cache_prompt=false` throughout; native request logs report `reuse=0`.
No OS-wide cache flushing. First at size is not cold: sizes share prefixes,
the model was already running, and file-cache working sets warm over time.
The LAN router logged no requests from the start of the run until root
paused it at about 03:29:51 UTC; it remained paused for the rest of the run.
Host memory stayed near 26 GiB available, server cgroup swap was zero,
and the XTX stayed idle (enumeration handles only, no model weights).

Raw measurements: [results-halo-prefill-matrix-native-0649101.jsonl](results-halo-prefill-matrix-native-0649101.jsonl).
Harness: `f2463a1`; CPU regression tests expanded in `73fdaba`.
Fixture SHA256: `a37dfebb993cab5cf754a0d63a31875df93d106e61f6351273ad91d427942fc7`.
Run ID: `a9032062-9409-4b27-af9a-a80883ae0505`.
The deterministic default `prepare` workload reproduces this fixture with
the same tokenizer. Full local fixture and metadata are retained under
`/home/ryan/qk-prefill-matrix-eyUgzZ`.

Served native artifact SHA256:

- `libqk.so`: `0782c0ff527a78c02ca52d1872a755835ec02a73b6229fcfd5f485d76cb2d3a8`
- Rust server: `aa497bdb0e6b8a70c8d9abe34f6956c91c04cc1ecd5bace53d9e493ce273ef9c`

## Halo-only llama.cpp comparison (completed)

The installed fork `2dff8596dcb7bdf765d24d44e1155d51f04c82b7`, with its
local grammar fix, completed the same matrix on September 12 at about
04:22:52 UTC (September 11 local time). Run started at 04:15:26 UTC. This
configuration uses **F16 KV and flash attention**, batch 2048 / microbatch 1024,
one sequence, context 32768, full Halo offload, and **no MTP or projector**.
Native above uses F32 math/KV and prefill chunk 512. These are different
precision tiers, not an accuracy-equivalent kernel-only A/B.

Medians of three repetitions, identical prompt IDs and 128 output tokens:

| Prompt tokens | Prefill + one token (s) | First token (s) | Decode (tokens/s) | Decode min..max |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.552 | 0.548 | 28.55 | 28.48..28.56 |
| 512 | 1.209 | 1.189 | 28.41 | 28.24..28.47 |
| 2048 | 4.428 | 4.443 | 28.06 | 28.04..28.06 |
| 8192 | 18.271 | 18.290 | 27.49 | 27.46..27.50 |
| 16384 | 38.361 | 38.365 | 26.78 | 26.77..26.81 |

All 15 decode outputs pass the counting check and have the same token/text
hashes as native. All 30 measured requests report `timings.cache_n=0`.
llama.cpp's `tokens_cached` field describes retained sequence length after
a request; it is not a cache-hit count. No output or failed run was excluded.
The earlier malformed fast llama runs are historical: this fresh,
single-Halo configuration passes this workload. Counting is still not a
general accuracy, tool-use, long-conversation or MTP validation.

Direct comparison, again with the precision difference above:

| Prompt tokens | Native first token (s) | llama.cpp first token (s) | Native decode (t/s) | llama.cpp decode (t/s) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.205 | 0.548 | 32.76 | 28.55 |
| 512 | 3.145 | 1.189 | 30.56 | 28.41 |
| 2048 | 13.591 | 4.443 | 23.32 | 28.06 |
| 8192 | 70.076 | 18.290 | 12.62 | 27.49 |
| 16384 | 193.133 | 38.365 | 8.05 | 26.78 |

Native retains a short-prompt decode advantage, but loses it between 512 and
2048 prompt tokens. At 16K llama.cpp delivers about 3.33x the decode rate,
with first-token latency about one-fifth of native's. This motivates attention
and prefill optimization; it does not identify one kernel as the sole cause.

The launcher `1f717fb` validates `GGML_VK_VISIBLE_DEVICES=1` as only Halo,
then selects the resulting `Vulkan0`; this also keeps model host staging on
Halo. DRM client 375 at PCI c1:00.0 owns the model and increasing compute
counters. The external XTX has only Mesa enumeration handles (12 KiB VRAM,
2 MiB GTT, no engine counters), unchanged through the run; no external GPU
model computation was used.

Before this launch, with every model stopped and Halo GTT 155525120 bytes,
the unused TTM page pool retained about 96 GiB. The unchanged 24 GiB
MemAvailable launch guard refused to start. A bounded invocation of only
the kernel's unused `page_pool_shrink` released about 7.23 GiB until 28 GiB
was available; no `tt_shrink`, pool-cap change, driver reset, reboot or
OS-wide cache flush. Named shard file cache was advised out before loading.
A 128-token smoke preceded the matrix; this is a warmed working-set
comparison, not a cold-load benchmark. Native was also already warm.
The unused-pool operation follows
[Linux v7.0 TTM pool code](https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/ttm/ttm_pool.c).

The reference unit limits were 8 GiB memory-high, 12 GiB memory-max,
256 MiB swap-max. They do not account for all GPU GTT. No OOM or memory-max
events were recorded; the unit had about62 MiB swap at completion
(98 MiB peak), so this is not reported as zero-swap. Sampled host headroom
during the matrix was roughly23.5–24GiB, sampled Halo temperature reached
80°C. No clocks or power limits were changed. The unit was stopped after
completion and Halo GTT returned to 155525120 bytes.

Raw measurements:
[results-halo-prefill-matrix-llama-fast-2dff8596.jsonl](results-halo-prefill-matrix-llama-fast-2dff8596.jsonl).
Run ID: `680ca308-8bae-4d97-aa91-36165bdb6cef`.
Config SHA256: `2498ae4de01ad756f27be4ed691c0688ef8e10a9b83e6c9aced5564e00b52a97`.
Server SHA256 (verified before and after):
`81a765e3a4df9cc9911854bb25224400e1244be2a07fd3c3eb30204d95d46762`.
Full observed settings, model-shard identities and cache procedure are in the
raw run header. Harness source SHA256:
`fe08d077535d959f841f2d35d009d791a6dc1455c718d5c852534b2bb4249a66`.

## Remaining work

The user's “recipe” backend is still unidentified. Conservative llama F32
and MTP presets have not run this matrix; do not label the fast/no-MTP rows
as either of those. Split-K native attention remains an opt-in candidate
pending its exclusive build, full-model/long-context gates and performance
A/B; no candidate speedup has been established. Prefill is unchanged by
that candidate and remains a separate optimization opportunity.
