# Native Flash last-output prefill: full-model validation

September 12, 2026, Max. **Full-model correctness and HTTP/resource gates pass.**
Median client first-token latency fell 2.8–5.0% across the five tested
prompt sizes. At 16K, median decode was 2.7% lower, with overlapping ranges;
this is not a universal throughput improvement. **Serving is restored on
8091 with the flag explicitly enabled**, while the source-level default
stays off. The [partial-stage experiment](RESULTS-halo-last-head.md)
describes the implementation and its separately measured two-layer gain.

## Same-build full-model correctness

Both policies ran all 48 layers and the head of the installed three-shard
`Qwen3.8-Flash-Next-Uncensored-Q5_K_M` on Halo PCI `0000:c1:00.0` alone.
Context 32768, one slot, prefill chunk 512, baseline F32 GEMM/attention,
serial decode, F32 KV, no cooperative matrices, no MTP and no whole-table
PLE warming. The old all-ID ABI remains unchanged; the candidate preserves
the same final 64-row vocabulary tile and skips earlier head tiles.

For each of 15 cases the order was all/last/last/all, restarting from base
zero for each pass. The short initial lengths were 1, 2, 63, 64, 65, 127,
128, 129, 309, 511, 512, 513 and 1024, each followed by continuations of
1, 2 and 63 positions. The two long cases used the benchmark's 8192/16384
token prompts plus 309 deterministic teacher IDs, then 32 serial teacher
positions. Every 512-position prefill boundary and the partial 309-position
boundary was checked, as was every serial tail position.

All **498 cross-policy/reset comparisons** matched bit-for-bit across the
complete 248320-entry F32 vocabulary. There were 166 distinct baseline
rows and 664 total checked rows. Independently sorted top-20 values/IDs,
ABI greedy IDs, finite outputs and output canaries also passed. This is
same-build kernel equivalence, not a new external-model oracle or a broad
language-quality evaluation. Gate closed successfully at 21:55 UTC.

Evidence: [raw gate](results-halo-last-head-full-gate.jsonl),
[strict audit](results-halo-last-head-full-gate.json),
[sampled resources](results-halo-last-head-full-gate-resources.json),
[harness](native_last_head_full.py).

## Matched HTTP protocol

The two isolated launches use the same new server, native library, shaders,
model shards, fixture and clean environment. Only `QK_FLASH_PREFILL_LAST`
changes: 0 for all-ID, 1 for last-output. Actual process environment,
loaded library and policy announcement are checked. The source-level
default is off. Build/helper/shader hashes are frozen before either run
and verified after each; model shard sizes/mtimes are also bound.

Each arm runs the real HTTP/Claude suite (text, tool call, tool result,
streaming, cancellation, concurrent isolation and invalid-token rejection),
then repeats a seeded sampled 512-token request twice with 32 output tokens
(`temperature=0.8`, `top_p=0.9`, seed 59270). The sampled outputs must repeat
exactly within each arm and match across arms.

The timed matrix uses exact shared token IDs at 128, 512, 2048, 8192 and
16384 positions. Each size has three repetitions: a one-output-token probe
then a 128-token stream. KV state is fresh for every request; runtime logs
confirm `reuse=0`. PLE/OS caches are not flushed between requests. One-time
guarded unused-TTM cleanup and model-shard-only cache advice precede each
load. Compatibility checks precede both matrices; profiling is off.

The probe includes prefill, one generated token and HTTP overhead. Streaming
TTFT is measured separately by the client; decode excludes its first token.
Native `prompt_ms=0` is not treated as a GPU prefill timer. All output lengths,
stream token counts, counting prefixes, prompt hashes and output hashes are
audited, not inferred from a successful HTTP status.

Both arms completed: baseline 21:55:51–22:29:37 UTC; last-output
22:29:59–23:02:18 UTC. All 30 paired prompt cells, all 15 paired 128-token
outputs and seeded sampled outputs matched exactly. Both compatibility
suites passed, including cancellation (4.553 s baseline, 4.460 s last).

Medians of three observations per arm:

| Prompt tokens | All-ID TTFT (s) | Last-output TTFT (s) | Lower TTFT | All-ID decode (t/s) | Last-output decode (t/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.221 | 1.187 | 2.8% | 32.738 | 32.730 |
| 512 | 3.218 | 3.066 | 4.7% | 30.529 | 30.472 |
| 2048 | 13.974 | 13.291 | 4.9% | 23.124 | 23.084 |
| 8192 | 72.876 | 69.210 | 5.0% | 11.255 | 11.428 |
| 16384 | 205.096 | 196.611 | 4.1% | 7.030 | 6.838 |

At 16K, TTFT ranges were 202.703–205.399 s and 196.042–197.230 s;
decode ranges were 6.978–7.519 and 6.775–7.431 t/s. Separate one-token
probe medians were 204.581 and 196.589 s. The decode shader/precision is
unchanged, but that does not justify erasing the measured decode difference
or assigning it a cause. A reverse-order repeat would strengthen any
general throughput claim. At 16K, median total time for the 128-token stream
fell from 222.289 to 215.185 s (3.2% lower). This request workload benefits
overall; much longer output workloads should be measured separately.

Evidence: [paired full audit with all ranges](results-halo-last-head-full-audit.json),
[all-ID HTTP matrix](results-halo-last-head-http-all.jsonl),
[last-output HTTP matrix](results-halo-last-head-http-last.jsonl),
[all-ID resources](results-halo-last-head-http-all-resources.json),
[last-output resources](results-halo-last-head-http-last-resources.json).

Limitations: fixed launch order (all then last), only three repetitions per
size, and a synthetic counting continuation. This is not a general coding
benchmark. Comparisons with historical runs or the separately configured
Nathanw/MTP backend are not controlled last-output speedup measurements.

## Safety, identity and rollback

Both model units use MemoryHigh 24 GiB, MemoryMax 32 GiB, MemorySwapMax
512 MiB, NoNewPrivileges and disabled core dumps. The unchanged launch
admission requires at least 24 GiB available and at most 12 GiB for other
jobs' anonymous/shared memory plus used swap. A roughly three-second
observer stops only its own test unit on low memory, excessive temperature,
hard memory-limit/OOM events, or external-GPU compute/model allocations.
Cgroup memory does not include every GPU allocation on this node.

The baseline's 674 samples showed minimum available RAM 26.022 GiB, maximum
observed Halo temperature 76 C, zero cgroup swap and zero memory-event
counters. The last-output arm's 645 samples showed minimum available RAM
26.061 GiB, maximum temperature 75 C, zero cgroup swap and zero memory-event
counters. The XTX held only enumeration bookkeeping (12 KiB VRAM and 2 MiB
GTT), with no observed compute. Its resource audit also checks the clean
controlled stop and unchanged build identity. The full-model numerical
gate likewise recorded zero cgroup swap/pressure and no watchdog alarm.

Bounded cleanup was explicitly approved. Before the baseline it reported
242253 unused 4 KiB TTM pages freed (about 0.924 GiB); before last-output,
236172 pages (about 0.901 GiB). It did not touch live GPU model allocations.
No global cache drop, swapoff, GPU reset, clock,
power, driver, kernel or firmware change was made for this campaign.
Models were not deleted or replaced. Production routing was paused during
testing and is now restored on 8091; the existing 8092 Claude proxy
configuration is unchanged.

New native library SHA256:
`068369b87806608ed7d00efb2b72d3bd3e0f19abfd0ed67d7f452b8e952d8931`.
New release server SHA256:
`10bf29d115627fdf35a9305e0e2e1a9de1dea5a6bd8357627aad0c051fa1925a`.
Private build: `/home/ryan/qk-last-head-IIfPmw`.
Full campaign evidence: `/home/ryan/qk-last-head-full-VjRrMt`.
Verified original library/server backups are in that directory's `rollback/`.
The exact new pair is now installed in `build-halo`, retaining mode 0700.
The running executable and mapped library were checked against those hashes.

## Restored serving and follow-up decision

The reviewed restore helper started the full model on loopback 8194 and the
existing trusted-LAN gateway on 8091. Current runtime: last-output flag 1,
baseline GEMM/batch attention, serial decode, F32, one slot, context 32768,
prefill chunk 512, no MTP/snapshots/cooperative matrices or full PLE warming.
Actual environment and runtime announcements confirm the selected paths.
The server and router are transient user units; nothing was enabled at boot.
The opt-in is selected for the measured prefill-focused workload, not as a
claim that it is best for every output length. Set the flag to 0 for rollback
on a future controlled restart; the old all-ID ABI remains available.

The [final restoration audit](results-halo-last-head-restored-audit.json)
passes all direct 8194 checks, the applicable gateway 8091 checks (direct
disconnect-cancellation intentionally skipped for the buffering gateway),
and text/streaming smoke through Claude proxy 8092. Their raw records are
[direct](results-halo-last-head-restored-http8194.jsonl),
[gateway](results-halo-last-head-restored-http8091.jsonl), and
[Claude proxy](results-halo-last-head-restored-http8092.jsonl).
These are smoke checks, not extra benchmark cells.

The approximately five-minute restored-server observation recorded minimum
available RAM 26.198 GiB, maximum Halo temperature 55 C, zero model cgroup swap,
zero memory-event counters and no non-Halo engine work. Other processes can
still have old swap pages; this is not a claim of zero host-wide swap use.
The observer completed and exited without stopping the healthy server.
Before restoration the approved cleanup freed 239780 unused 4 KiB pages
(about 0.915 GiB); the 24 GiB launch guard still passed unchanged.

Two isolated [attention LDS experiments](RESULTS-halo-attention-lds.md) also
completed after the HTTP pair: 256 operator cells passed exact bits/FP64 and
padding checks, but the new variants were slower or inconsistent. They were
not integrated or enabled, and no full-model claims are made for them.

## Offline reproduction

No GPU or model load is needed for these evidence checks:

```bash
python3 bench/audit_native_last_head_full.py bench/results-halo-last-head-full-gate.jsonl \
  --all-http bench/results-halo-last-head-http-all.jsonl \
  --last-http bench/results-halo-last-head-http-last.jsonl
python3 -m unittest discover -s bench -p 'test_native_last_head*.py' -v
```

For private complete resource logs, use
`bench/audit_native_last_head_resources.py CONTROLLER.json OBSERVER.jsonl`.
Do not rerun a model experiment beside serving or an operator benchmark;
use the exclusive, bounded launch protocol recorded in the evidence.
