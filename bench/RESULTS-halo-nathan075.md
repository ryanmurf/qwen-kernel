# Max: Nathanw v0.7.5, Q5_K_M and cache-aware sessions

Both benchmark matrices are complete and audited. No serving-default
promotion. Measurements are from September 12, 2026 on Max's Strix Halo only.

## What is being compared

The [Nathanw v0.7.5 portable Vulkan release](https://github.com/Nathanw1014/strix-halo-llamacpp/releases/tag/v0.7.5)
is pinned to payload source `dff60048744f99cb0af68d02be2314456b3269dc`
(`b10677-dff60048` at runtime). Its release manifest identifies Mesa
`d18d598e2` and libdrm 2.4.133. The bundled RADV library was verified in the
running process's memory maps; nothing replaced the system graphics driver.
Portable archive SHA256:
`8dce9b405e8643b88adf81f4000abce796e825f3888147085cb8a5c5e3f94a3d`.

Both configurations use the existing three-shard
`Qwen3.8-Flash-Next-Uncensored-Q5_K_M`, not a new quantization. The existing
shared Q8_0 MTP draft is the only additional model for the MTP arm. No image
projector is loaded. The [article motivating this test](https://sleepingrobots.com/dreams/halogen-vs-nathanw-qwen38-flash-next-strix-halo/)
used a different, lighter AtomicChat quantization and combined MTP with
`ngram-mod`; this experiment does not reproduce that entire recipe or its
representative agent workloads.

Shared settings:

```text
Halo c1:00.0 only; GGML_VK_VISIBLE_DEVICES=1; --device Vulkan0
--split-mode none -ngl 99 --n-cpu-moe 0 --fit off
--load-mode mmap --no-host --no-repack --tensor-read-lazy on
-ot per_layer_token_embd.weight=CPU
-fa on -c 65536 -ctk q8_0 -ctv q8_0 -b 8192 -ub 2048 -t 4 -tb 4
--parallel 1 --cache-reuse 1 --cache-ram 0 --ctx-checkpoints 8
--jinja --reasoning-effort xhigh --metrics
```

The portable launcher's MMID optimizations retain their shipped defaults,
including `GGML_VK_FA_WAVE32=0`. `GGML_VK_ALLOW_SYSMEM_FALLBACK=1` is explicit.
The actual filtered environment, mapped libraries, full command and file
identities are in each raw run's header. The external XTX is not used.

The plain arm explicitly uses `--spec-type none`. The MTP arm adds:

```text
-md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-n-min 0
--spec-draft-p-min 0.75 --spec-draft-device Vulkan0 --spec-draft-ngl 99
-ctkd f16 -ctvd f16
```

No adaptive-draft flag is supported by this pinned build's help, and none
was supplied. No synthetic speculation or log-probability collection is used.

## Procedure and interpretation

The [session harness](README-llama-session-matrix.md) runs seven prompt sizes,
three repetitions, and three conditions per repetition: fresh KV, exact
repeat, then an appended user turn including the repeat's generated token
IDs. Each request generates exactly 128 tokens. Sampling is greedy, slot 0.
There are 63 measured cells per complete configuration. A raw stream smoke
and separate HTTP integration checks precede each matrix.

Fresh means no reused KV, **not cold OS/file cache**. Only named model-shard
cache was advised out before each model load; no global cache flush.
Actual `timings.cache_n`, not retained `tokens_cached`, establishes reuse.
The expanded fixture matches every overlapping prompt ID from our earlier
prefill matrix; its SHA256 is
`f078febf194df6fef97f62da147c8a8653d52b274ead53ea74b87de6b395c658`.

Counting is highly predictable and favors speculative decoding. Identical
counting tokens do not establish general quality, long-context retrieval,
coding speed, grammar robustness or agent performance. Exact repeats are
best-case cached requests. Both configurations run in fixed order, not a
counterbalanced multi-launch experiment.

## Completed no-MTP results

Medians of three, client-observed seconds/tokens per second:

| Base prompt | Fresh first token (s) | Repeat first token (s) | Follow-up first token (s) | Fresh decode (t/s) | Fresh decode min..max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.678 | 0.091 | 0.456 | 29.35 | 29.33..29.36 |
| 512 | 1.283 | 0.091 | 0.455 | 29.31 | 29.31..29.34 |
| 2048 | 3.944 | 0.092 | 0.460 | 29.16 | 29.15..29.16 |
| 8192 | 16.039 | 0.099 | 0.468 | 28.69 | 28.66..28.71 |
| 16384 | 33.563 | 0.101 | 0.484 | 28.25 | 28.25..28.29 |
| 32768 | 73.951 | 0.103 | 0.516 | 27.46 | 27.23..27.47 |
| 56320 | 148.608 | 0.109 | 0.576 | 26.29 | 26.04..26.33 |

All 63 outputs have the same valid counting token hash. All 21 fresh cells
have zero cached tokens. All 42 reuse cells have genuine hits: repeats
reuse `base_prompt - 4` tokens; follow-ups reuse `base_prompt + 127` and
evaluate 38 new tokens. No failed cell is omitted.

The [earlier installed-fork result](RESULTS-prefill-matrix.md) was 38.365 s
to first token and 26.78 t/s at 16K; this run is 33.563 s and 28.25 t/s.
That is historical context, not a controlled fork-only speedup: the older
run used F16 KV, batch 2048 / microbatch 1024, 32K context, a different
driver/build, and a one-token probe before each decode request. The earlier
native runs also use a different F32 precision tier. Do not attribute the
whole difference to a single kernel or compare cached latency to fresh
prefill latency.

Raw baseline: [results-halo-nathan075-plain.jsonl](results-halo-nathan075-plain.jsonl).
Private full evidence: `/home/ryan/qk-nathan075-b6cta3`.

## Completed MTP comparison

All 63 MTP cells passed. The offline cross-audit reconstructed every prompt
from the fixture and preceding output: **63/63 paired prompt hashes and
63/63 paired output hashes match** the plain arm. Shared launch settings and
frozen build identity also match. Both arms passed the same HTTP checks.

| Base prompt | Fresh first token (s) | Repeat first token (s) | Follow-up first token (s) | Fresh decode (t/s) | Fresh decode min..max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.706 | 0.090 | 0.480 | 66.89 | 66.88..67.10 |
| 512 | 1.401 | 0.091 | 0.483 | 66.60 | 66.38..66.63 |
| 2048 | 4.502 | 0.093 | 0.494 | 65.37 | 62.09..66.11 |
| 8192 | 18.410 | 0.101 | 0.513 | 59.97 | 59.66..61.20 |
| 16384 | 38.845 | 0.118 | 0.546 | 57.46 | 56.36..57.75 |
| 32768 | 86.688 | 0.142 | 0.641 | 52.96 | 52.95..53.03 |
| 56320 | 169.551 | 0.169 | 0.715 | 48.84 | 48.58..49.00 |

The matrix generated and accepted 6804 draft tokens: 100% acceptance on
this counting task. This is a favorable speculation case, not a forecast
for coding or tool-heavy conversations. No n-gram speculation was enabled.
The server's `/props` default field says `none` even in the MTP arm; actual
completion settings report `none,draft-mtp`, and per-request draft counts
and server metrics establish that MTP ran. Do not infer its state from that
default field alone.

MTP approximately doubles decode throughput at 16K (28.25 to 57.46 t/s)
and improves it 1.86x at 55K (26.29 to 48.84 t/s). But it increases fresh
first-token latency. For the **measured 128-token outputs**, the extra
prefill cost outweighs the decode saving on fresh long prompts:

| Base prompt | Fresh total, plain (s) | Fresh total, MTP (s) | Follow-up total, plain (s) | Follow-up total, MTP (s) |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 8.302 | 6.538 | 4.827 | 2.459 |
| 8192 | 20.471 | 20.539 | 4.895 | 2.659 |
| 16384 | 38.054 | 41.047 | 4.980 | 2.760 |
| 32768 | 78.576 | 89.086 | 5.136 | 3.073 |
| 56320 | 153.440 | 172.154 | 5.383 | 3.351 |

These are medians of measured full stream wall times, not totals assembled
from median component timings. On this task, prefer no MTP for first-token
latency / short answers to fresh long prompts; MTP benefits cached turns.
Representative coding/tool workloads and a separately labelled MTP-plus-
`ngram-mod` trial remain necessary before selecting a production default.

MTP matrix headroom was 15.94–20.20 GiB; sampled peak GPU temperature was
84 °C. Cgroup peak swap was 100462592 bytes (95.8 MiB). Both arms triggered
memory-high reclaim, but recorded zero memory-max/OOM events. Both stopped
cleanly with unchanged frozen identities.

Raw MTP: [results-halo-nathan075-mtp.jsonl](results-halo-nathan075-mtp.jsonl).
Cross-audit, complete summaries, memory evidence, API checks and source
artifact hashes: [results-halo-nathan075-audit.json](results-halo-nathan075-audit.json).

## Safety and compatibility

Production native serving was stopped router-first for an exclusive test.
Each portable instance is a transient user service on loopback port 8193,
with memory-high 12 GiB, memory-max 16 GiB, swap-max 512 MiB,
`NoNewPrivileges=yes` and core dumps disabled. Cgroup limits do not account
for all GPU GTT. The loopback benchmark configuration is not a hardened
network deployment; the bundled server warns about its default permissive
CORS/no-key settings. Do not expose it directly to the LAN.

The controller enforces 24 GiB available host memory before launch, an
8 GiB live floor, and a 12 GiB ceiling on other anonymous/shared/swap usage.
It requires drained Halo GTT before launch and checks actual device counters
while running. It stops only its own unit on completion or failure. No
kernel, GPU reset, power/clock setting, firmware, model weights or permanent
serving configuration changed.

The completed plain arm passed teacher-forced HTTP, coherent streaming,
Claude text/tool invocation/tool-result/streaming, prefill cancellation,
concurrent isolation and invalid-token checks. Sampled matrix headroom was
24.24–25.85 GiB; sampled peak GPU temperature was 84 °C. Cgroup peak swap
was 100446208 bytes (95.8 MiB), not zero. Memory-high reclaim occurred;
recorded memory-max and OOM counters remained zero. Its server stopped
cleanly and all frozen file identities matched before and after the run.

The first serving-restoration attempt stopped at the unchanged admission
guard: only about 21 GiB was available after targeted model-shard cache
release. All model processes were stopped and Halo GTT had drained to
148 MiB, but the driver retained about 95.6 GiB in its unused page pool.
No restart was forced and no unused-pool shrink was performed. Approval
has been requested for the existing bounded unused-pool cleanup before
restoring native serving; the native endpoint is currently offline.
