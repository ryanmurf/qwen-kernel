# Combined prefill and full-model long-context profile

September 13, 2026, Max — **combined prefill enabled; serving restored and
verified on the existing ports**.
Private evidence: `/home/ryan/qk-combined-profile-kGqp5a`.
The complete corrected audit selects last-output + vec4 prefill attention
under the predeclared criteria. The original observer-audit failure and
precautionary fallback are preserved below. The separate cache-layout probe
was slower and is rejected. Direct API 8194, gateway 8091 and Claude proxy 8092
checks plus the five-minute restored-resource audit pass.

## Frozen configuration and scope

Same existing three-shard Qwen3.8-Flash-Next-Uncensored-Q5_K_M, all 48 layers and
head on Halo PCI 0000:c1:00.0. No XTX model work. Context 32768, one slot,
512-token prefill chunks, baseline F32 GEMM, serial decode, F32 KV, no MTP,
snapshots, cooperative matrices or whole-table PLE warming. No runtime binary
or shader change in this first phase.

Library SHA256:
`068369b87806608ed7d00efb2b72d3bd3e0f19abfd0ed67d7f452b8e952d8931`.
Server SHA256:
`10bf29d115627fdf35a9305e0e2e1a9de1dea5a6bd8357627aad0c051fa1925a`.
Private tested build `/home/ryan/qk-last-head-IIfPmw`; these are byte-identical
to the installed library/server. The private 180-shader set also exactly
matches the preceding full-model gate. All selected files are frozen.

## Correctness protocol

`native_flash_combined_gate.py` runs vec4 attention with all-ID / last-output /
last-output / all-ID policies, resetting at each pass. All 15 cases from the
preceding baseline-attention gate are retained: 13 short/boundary cases,
8,501 positions and 16,693 positions including 309 teacher-input positions after
the 8K/16K benchmark prefixes, then 32 serial teacher positions per long case.
Every 512-position prefill boundary and partial boundary is checked.

All 664 full-vocabulary rows must match the frozen **same-library** baseline
hashes, covering 166 distinct baseline rows. Each live row also checks finite
logits, independent top-20/greedy ordering and output canaries. This establishes
cross-configuration equivalence on these inputs, not a new external quality
oracle. Prior evidence is read-only and its library/model/shaders/fixture are
checked before it can serve as the reference.

Completed at 12:10 UTC: **664/664 full-vocabulary rows match exactly**,
all 15 cases and reset/repeat passes. Resource audit: minimum available RAM
25.952 GiB, maximum sampled temperature 75 C, zero model cgroup swap and
zero memory events. [Raw gate](results-halo-combined-gate.jsonl),
[audit](results-halo-combined-gate-audit.json).

## Full-model profiling protocol

`native_flash_long_profile.py` profiles the entire 2K/8K/16K prefill and 16 greedy
decode positions at each depth, with last-output enabled via the ABI and
baseline attention. It records 100 step markers. The native profiler fences
every dispatch and disables replay; per-shader GPU attribution is therefore
**instrumented**, not production throughput. Wall time and GPU intervals are
kept separate. Dense-projection attribution includes the vocabulary head;
shader categories do not magically separate every semantic operator.

Completed at 12:16 UTC: all 100 steps and resource checks pass. Percentages
below describe the fenced GPU intervals, not fractions of API wall time:

| Prompt | Phase | Attention | Dense projections + head | Experts + routing | Other |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2,048 | Prefill | 9.8% | 41.4% | 43.7% | 5.1% |
| 2,048 | Decode | 25.4% | 37.1% | 31.6% | 6.0% |
| 8,192 | Prefill | 30.9% | 31.6% | 33.6% | 3.9% |
| 8,192 | Decode | 56.9% | 21.7% | 18.1% | 3.3% |
| 16,384 | Prefill | 51.1% | 22.3% | 23.8% | 2.8% |
| 16,384 | Decode | 71.1% | 14.5% | 12.3% | 2.1% |

At 16K, the instrumented decode averages 133.0595 ms GPU and 135.8695 ms
wall per token; attention accounts for about 94.55 ms of that GPU interval.
The instrumented whole prefill is 193.111 s GPU / 193.845 s wall. The
remainder is not a separately measured pure CPU cost: copies, synchronization,
readbacks and profiler overhead contribute. Do not substitute these values
for the unprofiled HTTP measurements below.

This identifies two priorities: attention at long context, and projections/
experts for shorter prefill. It does not establish the gain from a proposed
kernel. Resource audit: minimum available RAM 26.016 GiB, maximum 72 C,
zero model cgroup swap and zero memory events.
[Structured steps](results-halo-long-profile.jsonl),
[native attribution log](results-halo-long-profile.log),
[audited categories and per-shader totals](results-halo-long-profile-audit.json).

## Counterbalanced HTTP protocol

Eight separate model launches:

| Launch | Last-output | Batch attention | Name |
| ---: | ---: | --- | --- |
| 0 | 0 | baseline | all |
| 1 | 1 | baseline | last |
| 2 | 0 | vec4 | vec4 |
| 3 | 1 | vec4 | combined |
| 4 | 1 | vec4 | combined |
| 5 | 0 | vec4 | vec4 |
| 6 | 1 | baseline | last |
| 7 | 0 | baseline | all |

Per launch, the unchanged HTTP/Claude/tool/cancellation/isolation suite runs
first, followed by two identical seeded 32-output samples. The exact shared
fixture then runs 128/512/2048/8192 input with 128 output, and 16384 input with 512 output.
Each stream follows a separate one-output probe. Thus there are two independent
launch observations per configuration/cell. Every request has fresh KV; file/PLE
caches are not flushed between requests. Actual environment, executable,
mapped library and runtime policy announcements are checked.

Two additional 256-token-budget code-review streams exercise a Python LRU cache
and the actual Vulkan attention source. Natural early EOS is retained, output
counts/hashes are checked across configurations, and generated code is never
executed. These are unscored workload probes, not proof of general coding quality.

All 80 counting cells, 16 code probes and seeded outputs must pass cross-launch
checks. Reverse launch order mitigates monotonic drift but does not eliminate
all carry-over effects. Two launches per mode are not a universal speed estimate.

## Predeclared deployment decision

Compare combined with the currently deployed **last-output + baseline
attention**, using this campaign only. Promote combined as an explicit transient
serving opt-in only if all correctness/API/resource audits pass and:

- Median 16K first-token latency is at least 10% lower.
- Median 16K total 512-output stream is at least 5% lower.
- Median 16K decode loses no more than 3%.
- No tested median total stream is more than 3% slower.

Otherwise restore the original last-output 1/baseline-attention configuration.
Source defaults remain unchanged. The decision is scoped to these workloads.

## Completed HTTP measurements

All eight launches, 80 counting cells, 16 code probes and seeded repeats pass
the complete corrected audit. Each timing below is the median of **two separate
launches**; parentheses retain the observed range. These are fresh-KV HTTP
measurements, not isolated GPU timings. Do not pool them with earlier campaigns.

### 16K prompt, 512-token answer

| Configuration | First token, s (range) | Decode, tok/s (range) | Whole stream, s (range) |
| --- | ---: | ---: | ---: |
| all | 196.70 (188.37–205.02) | 7.14 (6.84–7.43) | 268.43 (257.11–279.75) |
| last | 198.10 (197.47–198.73) | 7.07 (6.91–7.24) | 270.37 (268.08–272.66) |
| vec4 | 159.43 (159.35–159.50) | 6.94 (6.84–7.03) | 233.11 (232.00–234.23) |
| combined | 153.22 (152.80–153.63) | 6.92 (6.79–7.04) | 227.12 (226.21–228.04) |

The first baseline launch was faster than its reverse-order repeat; its
range is deliberately retained. This campaign does not establish a separate
last-output-only win over all-ID at every size. The selected comparison is
**combined versus the previously deployed last-output + baseline attention**.

| Input / output tokens | First-token change | Decode-rate change | Whole-stream change |
| ---: | ---: | ---: | ---: |
| 128 / 128 | -0.47% | -0.37% | 0.18% |
| 512 / 128 | -2.19% | -0.14% | -0.85% |
| 2048 / 128 | -4.98% | -0.83% | -3.28% |
| 8192 / 128 | -14.09% | 3.07% | -12.50% |
| 16384 / 512 | -22.66% | -2.23% | -16.00% |

Negative latency/total-time change is better; positive decode-rate change
is better. At 16K this saves **44.88 seconds to first token and 43.25 seconds
per 512-token answer** in the two-launch medians. The 2.23% decode reduction
is within the predeclared 3% bound; short-stream overhead peaks at 0.18%.
All four deployment criteria pass. This is a prefill-focused workload result,
not a universal throughput or code-quality guarantee.

[Complete audited summaries](results-halo-combined-http-audit.json),
[raw per-launch records](results-halo-combined-http/),
[decision using the unchanged criteria](results-halo-combined-http/decision-recovered.json).

### Unscored code-review probes

All eight launches return identical output hashes and lengths for each
workload. These prompts review a Python LRU implementation and the actual
Vulkan attention source; generated code is not executed and correctness of
the prose is not graded. The table reports actual input/output token counts
and two-launch medians, not coding capability.

| Configuration | Workload | Input / output | First token, s | Decode, tok/s | Whole stream, s |
| --- | --- | ---: | ---: | ---: | ---: |
| all | python-lru-review | 139 / 256 | 1.44 | 30.36 | 9.84 |
| all | vulkan-attention-review | 1122 / 256 | 7.88 | 26.04 | 17.67 |
| last | python-lru-review | 139 / 256 | 1.40 | 30.72 | 9.70 |
| last | vulkan-attention-review | 1122 / 256 | 7.51 | 26.04 | 17.30 |
| vec4 | python-lru-review | 139 / 256 | 1.44 | 30.49 | 9.81 |
| vec4 | vulkan-attention-review | 1122 / 256 | 7.84 | 23.79 | 18.64 |
| combined | python-lru-review | 139 / 256 | 1.39 | 30.24 | 9.82 |
| combined | vulkan-attention-review | 1122 / 256 | 7.33 | 25.86 | 17.19 |

### Shutdown-observer audit correction

The first complete audit **failed**, and the controller safely restored the
previous last-output-only configuration. The original failure and fallback
decision remain in the archive. Launches 6 and 7 had completed every workload,
but the original observer exited when it sampled the normal `stop-sigterm`
transition instead of waiting for `dead`. Their synchronous stop commands
returned zero. No benchmark measurements were rerun or overwritten.

The corrected auditor accepts that transition **only with supplemental,
contemporaneous systemd evidence**: the same named unit/boot/supervisor,
matching stop-job ID, successful job result, completion after the original
"all work completed" sample and before the controller's end, and bounded final
memory/swap peaks. The original controller log is hash-bound to the completed
sample. Missing, failed, reordered or inconsistent evidence still fails.
The stop jobs completed in 0.403 and 0.306 seconds; both final swap peaks were
zero. Regression tests reject corrupted evidence. Numerical, performance,
temperature and memory thresholds are unchanged.

[Original failure](results-halo-combined-http/chain-original.result.json),
[launch 6 stop evidence](results-halo-combined-http/http-06-last.stop-evidence.json),
[launch 7 stop evidence](results-halo-combined-http/http-07-all.stop-evidence.json),
[bound recovery receipt](results-halo-combined-http/recovery-v2.json).
The first queued follow-up did not touch production after the failed audit.
A separate recovery controller verified the supplemental evidence before
performing any subsequent operator work or configuration change.

### Isolated cache-layout follow-up: rejected

Persistent transposed K (plain, stride +16 and stride +32) passed all 128
operator cells with baseline-exact full outputs, FP64 sampled-head checks,
two-slot/decreasing-length coverage and poisoned padding. Its baseline SPV
matches the frozen native shader. **Every candidate was slower at long context**:

| Live keys | Original, ms | Transposed, ms | Pad16, ms | Pad32, ms |
| ---: | ---: | ---: | ---: | ---: |
| 16384 | 5.848 | 8.865 | 9.228 | 9.661 |
| 32768 | 11.737 | 23.085 | 21.661 | 23.195 |

Each cell averages four GPU-timed dispatches; two cells per mode/case run in
ABCDDCBA order. Uploads and construction/maintenance of the transposed cache
are excluded from those intervals. This cannot support a full-model gain.
The separate native integration draft was therefore **not built, merged,
full-model tested or deployed**. Serial decode remains unchanged.
[Operator raw log](results-halo-ktranspose.log),
[controller/build/source identities](results-halo-ktranspose-controller.json),
[operator audit](results-halo-ktranspose-audit.json).

## Restored deployment

Final verification completed at 14:15 UTC. The live native server uses
`QK_FLASH_PREFILL_LAST=1` and `QK_FLASH_ATTN_BATCH=vec4`, with baseline GEMM,
serial decode, F32 KV/math, context 32768, one slot and 512-token prefill chunks.
The library/server bytes are unchanged from the frozen tested build above.
No MTP, prefix snapshots, cooperative matrices or whole-PLE-table warming
were enabled. The slower transposed-cache draft was not installed.

Direct API 8194 passes all 9 checks including cancellation; gateway 8091 passes its
8 checks (cancellation omitted for the buffering gateway); proxy 8092 passes
Claude text and streaming. The final 100-sample, five-minute observation
shows minimum 26.177 GiB available RAM, maximum 51 C, zero model cgroup swap and
zero memory events. Only the Halo has model compute; the XTX's tiny
enumeration buffers remain 12 KiB VRAM / 2 MiB GTT. Cgroup high/max 24/32 GiB,
swap cap 512 MiB, NoNewPrivileges, disabled cores and owner-only binary
permissions are verified, as are the preserved rollback files. The restore
observer was also corrected to parse explicit DRM memory units.

These are still **transient, explicit opt-ins**, not boot-enabled units or
new source defaults. The existing Claude proxy's behavior was not changed.
[Final deployment receipt](results-halo-combined-restored.json),
[completed follow-up controller](results-halo-combined-http/final-result.json).

The profile still identifies long-context attention and short-prefill dense/
expert projections as the next targets. This campaign narrows the gap; it
does not claim to match the differently configured llama.cpp precision/MTP
tiers or establish broad coding quality.

## Safety and offline audits

The HTTP campaign contains 2037 resource samples: minimum available RAM
25.917 GiB, maximum sampled GPU temperature 77 C, maximum model cgroup memory
1443409920 bytes, zero model cgroup swap and zero observed memory-event
counters. The operator's 32 samples show minimum 26.832 GiB available, maximum
34 C, 295837696 bytes cgroup peak and zero swap/events. These figures do not
claim zero whole-host swap. Original telemetry hashes are in the archive's
[provenance record](results-halo-combined-http/provenance.json).

24 GiB available-RAM admission and 12 GiB other-anonymous/shared/swap budget are
unchanged. Model cgroups use 24/32 GiB high/max, 512 MiB swap cap, NoNewPrivileges,
cores disabled and bounded runtimes. A 3-second observer stops only its named
test unit on available RAM < 8 GiB, temperature >= 93 C, hard memory/OOM events or
non-Halo compute/model allocation. Cgroups omit some GPU allocations; sampled
model-process DRM is not proof of whole-host exclusivity at every instant.

One initial gate launch was refused while prior GPU memory was still draining;
its failed admission receipt is preserved. Only the previously approved bounded
unused-TTM cleanup and named-shard cache advice are allowed before loads. No
global cache flush, swapoff, GPU reset, clocks/power/kernel/firmware changes.

```
python3 bench/audit_native_flash_campaign.py gate /home/ryan/qk-combined-profile-kGqp5a
python3 bench/audit_native_flash_campaign.py profile /home/ryan/qk-combined-profile-kGqp5a
python3 bench/audit_native_flash_campaign.py http /home/ryan/qk-combined-profile-kGqp5a
python3 -m unittest discover -s bench -p 'test_native_flash_campaign.py' -v
```

These audits are read-only and do not load the model. Do not rerun controllers
in the completed output directory; they intentionally refuse existing outputs.
