# Combined prefill and full-model long-context profile

September 13, 2026, Max — **correctness and profiles complete; HTTP campaign
in progress; no new serving-speed claim yet**.
Private evidence: `/home/ryan/qk-combined-profile-kGqp5a`.
Production native serving is paused during the exclusive tests. The controller
restores the previously validated last-output configuration unless the complete
new campaign passes its declared promotion criteria.

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
preceding baseline-attention gate are retained:13 short/boundary cases,
8,501 positions and16,693 positions including 309 teacher-input positions after
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
for the forthcoming unprofiled HTTP measurements.

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

## Safety and offline audits

24 GiB available-RAM admission and 12 GiB other-anonymous/shared/swap budget are
unchanged. Model cgroups use24/32 GiB high/max,512 MiB swap cap, NoNewPrivileges,
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
