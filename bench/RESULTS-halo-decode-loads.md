# Same-order decode load scheduling on Strix Halo

September 13, 2026, Max. **Numerical gate passed, but the HTTP campaign
failed during the second 31K request. Candidate not deployed; previous
serving restored and verified at 21:21 UTC.**

Campaign: `/home/ryan/qk-decode-full-MFem1G`.
The previous combined-prefill / serial-decode installation is the fallback.

## Campaign failure and restoration

The first serial and first loads launches completed all six HTTP sizes and
API/code/resource checks. The second loads launch completed through 16K,
then lost its GPU context during the 31K request at 21:14:35 UTC. The kernel
log identifies a gfx_0.0.0 ring timeout in the test server and successful
automatic ring recovery; RADV reported guilty-context hard recovery and
`VK_ERROR_DEVICE_LOST`. The HTTP client received an incomplete stream.
There was no completed second 31K result and the final serial launch was
not attempted. The failed launch also lacks a clean observed shutdown.

The candidate changes decode, while the server log has no completed 31K
prefill marker. This suggests the failure occurred during prefill, but
does not establish which shader, submission, driver or hardware condition
caused it. The lack of a numerical mismatch or a memory watchdog alarm
does not excuse the device loss. No promotion criterion was relaxed and
no failed/missing result was filled in or rerun in place.

The original fallback library remained installed. All restored API checks
passed on 8194/8091/8092, followed by 100 resource samples over 297.84 s:
minimum available RAM 26.036 GiB, maximum sampled temperature 59 C, zero
model cgroup swap and zero memory events. The restored audit is PASS;
the overall campaign remains failed, with `restored: true`.

Preliminary first-pair decode rates (not counterbalanced final medians):

| Input tokens | Serial, tok/s | Loads, tok/s |
| ---: | ---: | ---: |
| 128 | 32.83 | 33.19 |
| 512 | 30.90 | 33.10 |
| 2048 | 23.19 | 31.56 |
| 8192 | 11.75 | 26.01 |
| 16384 | 7.25 | 20.34 |
| 31744 | 4.39 | 14.54 |

Second loads observations at 8K/16K were 26.03/20.14 tok/s before the
failure. These promising partial results do not establish serving stability
or justify enabling the candidate. All private receipts remain under the
campaign directory, including the incomplete matrix and error logs.

## Candidate

`QK_ATTN_DECODE=loads` selects `fa_attn_srv_loads.spv` on Strix Halo only,
with supported context 1..32768. Unset or `serial` retains the original path.
This is an inference shader change, not an OS kernel or driver change.

The candidate reads K in vectors of four floats, then accumulates each dot
in its original x/y/z/w order. V loads for 32 consecutive keys are issued
before their original, sequential multiply-add chain. Tail keys use the
original scalar loop. KV remains F32, row-major and the same size. The
256-key softmax tiles, reduction trees, online correction, per-dimension
sum order, output gate, descriptors and workgroup grid are unchanged.
There is no new persistent cache, scratch allocation or split reduction.
Prefill remains last-output + vec4, with baseline F32 GEMM, one slot and
512-token chunks. No MTP, prefix snapshots or cooperative matrices.

Candidate library SHA256:
`56ab8b435724bc2b429b2b2e6ef7e394e1b77096a4faa15f4b8d69b4312b771e`.
Candidate decode SPV SHA256:
`a25706ea5cd438d92e0ba3af96d704e27978cb64c9e5bab9b0d62a2a3064d18d`.
The integrated SPV exactly matches the winning isolated operator. Every
one of the 180 pre-existing shaders matches the frozen previous build.
The Rust server binary is unchanged.

## Isolated operator results

Two completed sweeps, 144 cells each, on Halo PCI 0000:c1:00.0 alone.
Each uses 18 cases, A-B-C-D-D-C-B-A order and four GPU-timed dispatches per
cell after a warm-up. Native DEVICE_LOCAL placement; transfers are outside
the timestamp interval. Full outputs match the original bit-for-bit;
sampled FP64 reference heads also pass. Tests retain NaN padding/guards,
two unequal slots, scaled queries, partial tiles and decreasing lengths.

The first sweep separated K vector loads from V8 scheduling. The second
compared V8/V16/V32 with the same K4 path. These are operator timings,
not full-model decode throughput. Below are medians of two cells from the
second sweep, each cell averaging four dispatches:

| Live keys | Original, ms | K4/V8, ms | K4/V16, ms | K4/V32, ms |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.521 | 0.138 | 0.115 | 0.105 |
| 8,192 | 2.140 | 0.555 | 0.458 | 0.409 |
| 16,384 | 6.046 | 1.476 | 1.182 | 1.034 |
| 32,768 | 11.874 | 2.935 | 2.361 | 2.042 |

V32 is approximately 5.84x / 5.82x faster at 16K / 32K in this sweep.
This does not imply that the entire model becomes 5.8x faster or context
cost becomes constant. Other model operations still take time.

Private evidence:

- `/home/ryan/qk-decode-loads-Pmiv4K`: first completed operator.
- `/home/ryan/qk-decode-tune-RQQjnZ`: **no operator run**; controller recovery.
- `/home/ryan/qk-decode-tune2-PrmxAh`: completed V8/V16/V32 comparison and
  verified restoration on 8194/8091/8092.

The first controller's restore failed because its parent accidentally had
NoNewPrivileges, preventing the already-approved bounded cleanup helper.
The next controller caught an already-unloaded production unit during its
stop step and ran no operator, but successfully restored serving. The final
tuning controller completed both operator and restoration. All original
receipts, including those failures, are retained; none was overwritten or
treated as a successful full campaign. Model child hardening was unchanged.

## Full-model protocol

Completed at 20:29 UTC: **664/664 full-vocabulary rows match exactly**.
All 15 cases, four passes each, and clean closure passed. The resource
audit retained 396 samples, maximum gap 3.023 seconds, minimum available
RAM 26.003 GiB, maximum sampled temperature 78 C, cgroup peak
1,364,393,984 bytes and zero recorded swap. No memory-high/max/OOM event
was reported at closure. [Raw gate](results-halo-decode-loads/gate.jsonl),
[controller and resource audit](results-halo-decode-loads/gate.controller.json).

The exact-logit gate retains all 15 previous cases and all/last/last/all
passes: 664 full-vocabulary rows, including every long-prefix chunk
boundary and serial continuation. It compares against the immutable
pre-change full-model F32 reference. Only the enabling library and newly
added decode shader may differ; every existing shader must match.

After that gate passes, four independent HTTP launches run serial / loads /
loads / serial, using the same newly built library and shader set. Each
launch gets the full HTTP/Claude/tool/cancellation/isolation suite, repeated
seeded sampling, six counting streams and two unscored code-review probes.

Counting sizes are 128, 512, 2048, 8192, 16384 and 31744 input tokens.
Outputs are 128 below 16K and 512 at 16K/31K. Each measured stream gets
fresh KV and **one prefill, without the earlier one-token probe**; do not
pool these timings with the prior campaign. 31K leaves answer space within
the unchanged 32K capacity. The expanded existing fixture preserves every
overlapping prompt ID; no candidate-specific tokenization occurs.

Before any HTTP result exists, promotion requires all correctness/API/
resource checks and:

- At least 25% higher median decode at both 16K and 31K.
- No tested median decode more than 3% lower.
- No tested median total stream or first-token latency more than 3% higher.

Two launches per mode are not broad statistical proof. Counting and
unscored code probes do not establish general model quality. Production
promotion, if accepted, remains an explicit transient opt-in, not a source
default or boot-behavior change.
