# Halo-only decode-attention experiment

Status (2026-09-12 UTC): **split-K with chunk 256 failed the long-context
numerical gate. It is not approved as a serving default.** Unset
`QK_ATTN_DECODE`, or set it to `serial`, for the existing path. No speedup
is claimed. Batched prefill is unchanged by this experiment.

The opt-in `QK_ATTN_DECODE=split` path uses the existing F32 split/reduce
shaders with 24 query heads, 2 KV heads, head width 256, and a fixed dispatch
grid suitable for command-buffer replay. `QK_ATTN_CHUNK` defaults to 256;
16..1024 is accepted. Other chunk values have not passed this full-model
long-context gate.

## Evidence

All model computation used the integrated Radeon 8060S, PCI 0000:c1:00.0,
on Max. The three-shard Qwen3.8-Flash-Next-Uncensored-Q5_K_M model used
context 32768, one sequence, scalar F32 prefill and F32 KV, prefill chunk
512, no MTP, and no whole-PLE-table warming. The candidate library SHA256
was `5cce18bb11e8e5fa5e9328ea818aac2f9bd1b0e1e55057c36fb590c06bc950ac`.
Both modes used the same library, shaders and token IDs. Actual backend
mode announcements were checked against each process's clean launch
environment; labels alone were not accepted as proof of mode selection.

- Four-layer prefix oracle and batched/mixed checks passed.
- Full-model 16-position F32 oracle and four batched/mixed scenarios passed.
- A 256-position teacher-forced serial/split pair was bit-for-bit identical.
  At chunk 256 this exercises only one active split, not the long merge.
- Both long runs passed exact clean-state reset and full-prefix/full-tail
  repeat checks, with successful unit exit status and no memory-watchdog
  abort. All 128 ABI-returned greedy IDs matched their own saved logits.
- The cross-mode long comparison **failed**: 127/128 greedy agreement,
  maximum relative RMS 0.0404351 versus the 1e-5 bound, mean KL 4.72487e-5
  nats, maximum KL 0.00265966 nats. No thresholds were relaxed.

The long input is the exact 16384-token shared benchmark prompt followed
by 128 deterministic teacher IDs `(1000 + 7*i) % 248320`. Each mode prefills
the 16384-token prefix, decodes every teacher position, resets, and repeats
the entire sequence. It does not skip the middle of the tail.

The first 49 tail rows stayed below the RMS bound. The first large change
appeared at tail index 49 (zero-based input position 16433): relative RMS
0.0310883. The only greedy flip was at tail index 70 / position 16454:
serial token 290, split token 470. The serial logit margin between those
choices was 0.0528975; split favored its choice by 0.0604277. An abrupt
change after small rounding differences is consistent with a discrete
expert-routing change, but layer-level evidence is still needed to
establish the cause. Reset success does not establish cross-mode accuracy.

Raw gate record: [results-halo-split-attn-long-gate.json](results-halo-split-attn-long-gate.json).
The full ~123 MiB-per-mode logit dumps, model/shader manifests, launch logs,
and per-position analysis remain in `/home/ryan/qk-root-checks-ZU4evg/`.
The gate record binds their hashes and the observed backend modes.

The long test bodies took 423.34 s serial and 483.42 s split, excluding
model load. These totals include two prefills, two tails and logit
readbacks; they are **not isolated decode speed measurements**. The
performance A/B is withheld pending a passing candidate. The existing
[prefill matrix](RESULTS-prefill-matrix.md) remains the measured baseline.

## Reproduction and next checks

`tests/gpu_qwen4_attn_split.py generate --fixture ... --size 16384
--teacher-tail 128` creates the prompt-plus-tail IDs without loading a
model. Run `dump` once per mode with `--tail 128 --ctx 32768` in separate
exclusive GPU windows, then `compare` with its unchanged default gates.
Do not build or change shaders between dumps. Confirm actual native mode
announcements and compare the ABI greedy IDs to the saved rows as well as
running the helper's numerical comparison.

Next: capture layer boundaries at the first divergence and check the
attention operator independently before trying another split strategy.
Retain the failed data; do not promote this configuration based on the
short tests or a favorable throughput-only result.

## Traced cause and replacement candidate (2026-09-12 UTC)

The focused trace now localizes the divergence. Both modes replayed their
own uninstrumented logits bit-for-bit through tail index 49; enabling layer
taps at indices 48 and 49 did not change either result. Each trace contains
1304 captured files. At index 48, every captured layer boundary differs by
less than 1e-4 relative RMS (worst 7.0e-6). At index 49 / position 16433,
layer 41's FFN input differs by 3.862e-6, while its FFN output jumps to
0.252153 relative RMS. Earlier boundaries remain close.

Replaying the unchanged GPU router/selector on those captured FFN inputs
with the actual layer-41 F32 weights confirms the discrete change:

- Serial top ten: 59, 433, 331, 496, 55, 445, 217, 294, 42, **84**.
- Split top ten: 59, 433, 331, 496, 55, 445, 217, 294, 42, **404**.
- The serial margin between experts 84 and 404 is 2.38418579e-7 (one F32
  ULP at these logits); split favors 404 by 3.09944153e-6. Both actual GPU
  selections agree with the sorted GPU router logits.

Thus small upstream attention differences cross an expert-selection
boundary. This is not evidence of a reset failure or a broken selector.
The input/hash metadata, raw router logits and selection results are in
`/home/ryan/qk-root-checks-ZU4evg/h1-router*`; complete trace inventories and
per-layer comparisons are preserved there too.

### Operator timing, not model throughput

An independent synthetic test checks serial, split-K, GQA-grouped split-K,
and accumulation-preserving score/value kernels against a CPU FP64
reference. It tests 1, 255, 256, 257, 1025, 16433, 32768, then 127 keys;
the decreasing final length and NaN-poisoned unused KV/partials help catch
stale or out-of-bounds live-context reads. All 144 cells passed. Every
accumulation-preserving output was also bit-for-bit equal to serial.

The native-memory run used the same allocator preference as Qwen4Graph:
type 0 / heap 1 / DEVICE_LOCAL, not the earlier mapped type 3. Uploads and
readbacks sit outside the GPU timestamp interval. Numbers below are one
fixed-order sweep, eight timed repetitions per cell, not a multi-run median.

| Keys | Original serial | Ordered Q-group 12 / dimension stripe 128 | GQA4 split, chunk 32 |
| ---: | ---: | ---: | ---: |
| 16433 | 5.898 ms | 5.343 ms | 1.330 ms |
| 32768 | 11.733 ms | 10.603 ms | 2.774 ms |

The ordered variant reduces this operator time by about 9–10% while keeping
the tested serial outputs exact. It stages coalesced K tiles in shared
memory, reuses each tile across the 12 Q heads sharing a KV head, and keeps
the original dot, online-softmax, and per-dimension V accumulation order.
An initial uncoalesced score kernel was slower and is retained only as a
diagnostic control. GQA4/chunk32 is substantially faster in isolation but
does not promise serial-exact outputs; it needs its own full-model quality
evaluation. Neither timing is a claim of full-model speed or broad quality.

Raw native-memory data and build/shader hashes:
[results-halo-attention-operator-local.json](results-halo-attention-operator-local.json).
`tests/halo_attn_operator.cpp` builds as a standalone executable against
this repository's Vulkan helpers; set `QK_SHADER_DIR` and
`QK_OPERATOR_DEVICE_LOCAL=1` and run only in an exclusive Halo window.
`tests/halo_router_replay.cpp` replays only the captured router inputs and
~5 MiB of router weights, without loading the full model onto the GPU.

`QK_ATTN_DECODE=ordered` is now wired as an **opt-in** replacement candidate
with a 3 MiB score buffer at context32768. Serial remains the default.
The full-model 16-position F32 oracle, mixed/whole prefill, batch-to-decode
handoff and exact-reset checks passed with library SHA256
`ef562d291d37e0780d431442722a41f1c8d60e9d30ebc1991c0ad032444f8299`.
Worst final-logit RMS across the four oracle scenarios was 1.284e-6, with
no argmax mismatches. Long replay and API performance checks are still
pending; these short and synthetic checks do not establish those results.
