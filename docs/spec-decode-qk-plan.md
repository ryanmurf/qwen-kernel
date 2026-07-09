# Speculative decoding in the qk engine — design + implementation plan

> **STATUS 2026-07-09: P0–P3 shipped** (`qk verify` / `qk speccmp` harnesses,
> `QK_SPEC` in the deployment). Measured: oracle full-accept 2.1–2.6× (K=8–64,
> token-exact at every position); rollback token-exact under forced partial
> accepts incl. 127 consecutive rejections; echo-heavy generation 1.48×
> end-to-end at avg accept 7.48/8; non-echo workloads untriggered (+1% noise);
> 75 s verify soak peaked 99 °C junction. One design change vs. the plan below:
> scratch persistence needed **no shader flag** — slot state is bound by
> descriptor offset, so verify mode is copy-live→scratch + rebind
> (`copyDnStripes`/`verifyRound` in `src/main.cpp`). v1 speculates only when a
> single slot is active; multi-slot fairness awaits production `[spec]` data.

Task #5 deliverable. Companion to [speculative-decoding.md](speculative-decoding.md), which
assessed spec decode for the *llama.cpp* serving path (2026-04). Since then the rig moved to the
from-scratch qk engine (`src/main.cpp` + `server/`), which changes the calculus in two ways:

1. **The verify primitive already exists and is token-exact.** `prefillBatchLast` runs an N-token
   batched forward for any `base ≥ 0` (conv-carry seeding, causal attention, chunked DeltaNet scan),
   validated by `prefillcmp` (42/42 argmax matches at N ≤ 256, worst rel logit diff 7e-07) and
   `prefilldecode` (batched-prefill → serial-decode handoff exact).
2. **First-party economics beat the community numbers.** The old doc's [web/FLAG] pessimism
   ("net slowdowns even at 100% acceptance") came from llama.cpp on this model. Measured here
   (7900 XT, `prefillbench 4096`, 2026-07-07, task #3 window):

   | path | cost | per-token |
   |---|---|---|
   | serial decode (recorded step CBs, chunk=8) | 76.7 ms / 8 tok | **9.6 ms** |
   | batched forward, N=8 (incl. all-position logits) | 26.7 ms | 3.3 ms |
   | batched forward, N=32 (interp.) | ~79 ms | 2.5 ms |
   | batch cost model (fit N=8..256) | **c(N) ≈ 9.4 + 2.17·N ms** | — |

   The routed-expert union effect (§5 of the old doc — expert weights don't amortize at small N)
   is *included* in these measurements; the fixed-overhead amortization (submits, barriers,
   per-step CB walk of 40 layers) plus the dense/shared-weight amortization dominate it.

**Bottom line: a verify round of K=8 costs 2.8 serial tokens; K=32 costs ~8. Speculation pays
whenever expected accepted tokens per round clears that bar, and loses when it doesn't — so the
design is trigger-gated: speculate only on high-confidence drafts, else stay serial.**

## Why acceptance can be exact-match

Decode is **greedy GPU argmax** (`argmax1/argmax2` inside the recorded decode CBs; no
temperature/sampling in the ABI). So acceptance = "draft token t equals argmax of the verify
logits at position t−1", and the output sequence is **provably identical to serial decode** —
no rejection-sampling machinery, and A/B validation is bit-exact string comparison.

## Draft source: prompt-lookup (n-gram), no draft model

Per the old doc §4: no compatible small model shares the 248,320 vocab (the 27B-mtp GGUF is dense
and useless as a draft), there is no MTP head in our GGUF, and VRAM headroom is ~2 GB at
slots=2/ctx=16384. Prompt-lookup costs zero VRAM and zero training, and the actual workload
(agentic coding: tool-output echoes, code repetition, re-emitted diffs) is the best case for it.

- Draft: find the longest suffix of the generated-so-far sequence (length ≥ `L_min`, default 6)
  that recurs in the slot's token history (`Slot.prompt` + `Slot.genTokens`, both already
  engine-side); propose the K tokens that followed the match.
- Trigger gate: fire only on match length ≥ `L_min` and a unique (or longest-unique) match.
  No trigger → normal serial chunk. This bounds the worst case: rounds that would accept ~0
  never run.
- K: start at 8 (= chunk). Echo spans should use K up to 32 — c(32) ≈ 79 ms for up to 33
  committed tokens ≈ 2.4 ms/tok, ~4× serial. Accepted-but-not-yet-emitted tokens go in a
  per-slot queue and dribble out ≤ chunk per `qk_step_chunk` call, so **the C ABI does not
  change** (a queue-draining call does no GPU work).

## The DeltaNet rollback problem, solved for our shaders

Softmax-attention KV needs no rollback: verify writes K/V at positions `[base, base+K)`; on
rejection at k the garbage beyond `base+k` is never read (causal masking bounds reads to the
active batch) and is deterministically overwritten by the next round. The prefix-cache
(`snapshotSlot` at slot_start) is untouched — disjoint lifecycle.

The gated-DeltaNet layers (30 of 40; S state = hV·dS·dS·4 = 32·128·128·4 = **2 MB/layer/slot**,
~60 MB total + 96 KB/layer conv windows) update in place inside `dn_step_batch`, which loads the
slot stripe, scans Tn tokens in registers, and **persists the final state unconditionally**
(same for the conv window in `dn_conv_batch`). Partial acceptance therefore needs one of the
old doc's design-menu options. Chosen: **scratch-persist + promote-on-full-accept +
commit-pass-on-partial** — a snapshot/replay hybrid tuned to what the shaders already do:

1. Add a push-constant flag to `dn_step_batch`/`dn_conv_batch`: *persist to the scratch stripe*
   (a second per-slot stripe in the existing state buffers) instead of the live one. Verify
   rounds always run in this mode; the live state stays at the last committed position.
2. **Full accept (the common case on gated triggers):** the scratch state *is* the correct
   post-K state — promote it with a GPU-local stripe copy (`copyStripes` machinery exists;
   ~60 MB ≈ well under 1 ms). Round cost: c(K) + copy.
3. **Partial accept (k < K):** live state is still at `base`; the k accepted tokens' KV is
   already correct. Before the slot can serial-decode again, run one **commit pass** —
   `prefillBatchLast` over just the k accepted tokens with persist-to-live (they are committed,
   so unconditional persistence is exact; this is the same base>0 path `qk_slot_start` already
   uses). Cost c(k) ≤ c(K). If the next round triggers immediately, fold the k tokens into its
   verify batch instead (lagged commit) and skip the extra pass.

   Invariant: **a slot never enters the serial step CBs with uncommitted recurrent state.**

## Per-position argmax (the one missing GPU piece)

`prefillBatchLast(wantLogits=true)` computes all N×vocab logits into `bbLogits` but copies only
the last row. Verify needs argmax per position: dispatch the existing `pAm1/pAm2` reduction with
z = N over `bbLogits` rows (row offset via push constant), read back N u32s. No new shader
algorithms — a batched wrapper around what decode already does every token.

## Expected performance (honest)

Per round, K=8: cost 27 ms (+~0 promote / +c(k) commit), commits k+1 ∈ [1, 9] tokens.

| outcome | ms/token | vs serial 9.6 ms |
|---|---|---|
| full accept, K=8 | 3.0 | 3.2× |
| accept 4 of 8 | ~7.5 (incl. commit pass) | 1.3× |
| accept 0 | 27 + commit ≈ 36 | 0.27× ← why gating exists |
| full accept, K=32 | 2.4 | 4.0× |

With `L_min=6` gating, untriggered decode is byte-identical to today (zero overhead — the n-gram
scan is a CPU hash lookup per emitted token). End-to-end on agentic traffic this is workload-
dependent: heavy tool-echo/code-edit turns should see 2–4× decode; novel-prose turns are
untouched. A defensible blended guess is **1.2–1.8× decode throughput**, to be replaced by
`specbench` numbers before any default-on decision.

Note on engine-thread occupancy: a K=32 verify round blocks the engine ~80 ms — the same order
as one serial chunk (77 ms) and far below the prefill chunks (285 ms at 128) the slot-fairness
machinery already tolerates.

## Implementation plan

| phase | work | est. |
|---|---|---|
| P0 | Batched argmax over `bbLogits` rows + N-u32 readback; `qk_verify`-shaped internal helper (prefillBatchLast + argmaxes, no persist flag yet — harness-only) | ½ day |
| P1 | Scratch-stripe persist flag in `dn_step_batch`/`dn_conv_batch`; stripe promote on full accept; commit-pass path; `speccmp` harness proving byte-exactness vs serial across seeds × K × partial-accept forcing | 1–2 days |
| P2 | Prompt-lookup draft (hash map of L_min-grams over slot history, engine-side), trigger gate, spec round inside `qk_step_chunk` decode loop, accepted-token dribble queue; `QK_SPEC=1` gate, off by default | 1–2 days |
| P3 | `[spec]` per-request log line (rounds, triggered %, accept len histogram — mirror `[pcache]`), `specbench` on synthetic echo + a real transcript replay; tune `L_min`/K; junction-temp check under sustained verify load | 1 day |

Ship order matters: P0+P1 are pure engine + harness (no server or behavior change); P2 flips
nothing on by default; only after P3's replay numbers does `QK_SPEC=1` go in the deployment.

## Risks / open questions

- **Acceptance on real traffic is the whole bet.** If replay shows triggered rounds accepting
  < ~3 tokens on average, stop at P1 (the batched-argmax + scratch-persist work is reusable for
  any future draft source, including a trained EAGLE-style head or an MTP-converted GGUF).
- Scratch stripes add ~60 MB VRAM per slot (~120 MB at slots=2) — fits current headroom (~2 GB),
  but re-check against ctx growth plans.
- The n-gram map must be rebuilt on prefix-cache restore (history changes without token-by-token
  appends); build it lazily at first trigger check per turn.
- amdgpu ~10 s gfx-ring timeout: verify batches (≤ 48 tokens ≈ 115 ms) are far below the prefill
  chunks already tiled for this; no new exposure.
- With the qwen validator shelved there is **no production consumer** today; this plan is
  shelf-ready design, not scheduled work.
