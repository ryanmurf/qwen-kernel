# PORT.md — Metal port log (Apple M4 Max)

Port of the qk engine (Vulkan/RDNA3) to Metal on a MacBook Pro M4 Max
(40-core GPU, 64 GB unified, 546 GB/s theoretical). Model:
`Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (15.45 GiB, unsloth UD-Q3_K_M). Charter:
`../GOAL.md`. Branch: `metal-port`.

## Scorecard snapshot vs llama.cpp Metal — 2026-07-10 (for Ryan)

Same box, same GGUF, greedy. All qk numbers lap-thermal (cool-state where
noted); the plugged-in hard-surface record run is still owed.

| axis | llama.cpp | qk | qk/llama |
|---|---|---|---|
| decode 1-stream, engine (tg128-equiv) | 84.2 tok/s | **119.7** (8.36 ms/tok) | **1.42×** |
| decode 1-stream, serving path | 80.1 | **~110** steady (9.1 ms GPU/step) | **1.31×** |
| aggregate 2 / 4 / 8 streams | 114.5 / 135.2 / 149.8 | **149.8 / 182.1 / 195.7** | **1.31 / 1.35 / 1.31×** |
| pp512 (parity-exact config, v4) | 1452 | ~510 | 0.35× |
| pp512 (record config, v3 = llama's f16 class) | 1452 | **744** cool | **0.51×** |
| prefill vs own serial (B1 ≥4×) | — | 6.5–7.5× | gate PASSED |
| weights RSS at open | ~17 GB | **318 MB** (no-copy; mlock opt-in for serving) | — |
| ctx in production | — | 32768 (cross-box worker; 29.8k-tok real prompt served) | — |
| correctness | reference | prefillcmp 36/36; 4 prompts token-exact vs llama refs (incl. 1040-tok through 512-chunks); handoff exact; cross-box byte-identical | — |

**Since M5:** prefill head-tax fix (the z=n output head — B1 passed on the
spot); decode-once grouped MoE (v2 f32-proof, v3 f16 record, v4 f32 default
at chunk ≥192) + grouped down + vectorized staging → pp512 229 → 744;
B3 aggregate beats llama-server --parallel at every N; qkp2 wire + state
ABI same-day; mmap-eviction pathology found and fixed (QK_MLOCK /
QK_COPY_WEIGHTS — no-copy weights degrade 2–6× after memory-pressure
events and don't self-heal); bisect infra (tests/ parity fixtures +
scripts/bisect_gate.sh, ~6 s/step).

**pp512 update, same day:** GEMM'd router logits in the grouped regime
(moe_logits_gemm + add_rmsnorm rebind; gates green, semantics proven at
scalar 36/36 @ 9.7e-5) → **v3 620/614 ms = 825/834 tok/s, gap 1.74×**;
v4 exact-config 740 ms = 692 tok/s (probe 545 GB/s, coolest state yet —
scorecard table above predates this).

**pp512 gap = 1.95× and fully budgeted** (N=512 chunk, ms): grouped
gate+up 230, projections 139 (GEMM-saturated at 10.4 TFLOPs), misc/logits
113, grouped down 87, DeltaNet chain 76, attention 32. Named levers:
mul_mm_id-class register-tiled dequant for gate+up, chunked delta-rule for
DN, GEMM-ified router logits. This is deep-kernel work, in progress.

**Cross-box worker:** local steady state (S=33 shape, ctx 32768): stage
GPU 2.3 ms + 0.15 ms submit, s2+net 2.61 ms/tok — the 7–9 ms fixed
per-frame cost tron measures does NOT reproduce with a raw client past
warmup. With RTT now 1.26 ms it's the dominant cross-box lever; two
candidates are with tron to discriminate (first-touch GPU page mappings
over the first ~32 frames vs op3-per-frame KV-stripe memcpy ≈ 9 ms at
32k ctx — probes in CROSSBOX-BRIEF.md, QK_STAGE_STATS=1 on the worker
reads the split directly).

## 2026-07-10 (later): packed-block round + 80B port — both landed

### pp512: 834 -> 1213 tok/s (llama gap 1.74x -> 1.19x)
Root cause of the mul_mm gap found by benching llama's OWN kernels at our exact
shapes (test-backend-ops perf, scratch cases, since reverted): their per-op
mul_mat_id beat ours 1.45x and their dense q8 GEMM 12.0 vs our 8.2 TFLOPS at
2048x2048 N=512 — generic inner-loop deficit, not MoE-specific. The difference:
llama stages shared tiles as PACKED 64-half 8x8 blocks so every simdgroup_load
is contiguous stride-8; ours were strided (68) with transposed B loads.
Three commits, each gated (parity ids1-4 + long, prefillcmp, moegcmp):
- moe_gu_grouped5 (a776c5b): packed tiles + token tiles on grid z. Kernel
  4.79 -> 3.09 ms at n=512; pp512 1044.
- gemm_q8_0_hp (5da0a53): packed proj GEMM, 525.8 -> 360.2 us = 11.9 TFLOPS
  (llama parity at 356.7); promoted to default (QK_GEMM=h keeps old). pp512 1129.
- moe_down_grouped_p (efef4ff): packed down kernels. pp512 1212/1213
  interleaved (v3 control 903/904). Config: QK_MOE_GROUPED=5 (+ default hp gemm).
Budget before the round (N=512 isolation): gu 225.4 / proj 139.9 / down 82.7 /
dn 65.4 / attn 19.4 / resid 57.8 ms of 590.6. The packed rounds attacked the
first three. Remaining named levers: DN chunked delta rule (65 ms), residual
bucket, attn. llama 1443.7 +/- 7.4 re-measured same day.

### 80B port (PORT-80B-BRIEF.md, mirror of tron 99cdedf): DONE, worker live
- Pass 1 (45f7823): GGUF-driven shapes (48L/512E/top-10 from KVs+tensors),
  SelT widened to ids[16]/w[16] = 160 B across select/group/all consumers,
  runtime MoE buffers/push constants. 35B green (parity, parity-long, moegcmp
  values unchanged, caseMoe 1.2e-5).
- Pass 2 (5ad107b): embed_q8_0, moe_gateup_all_iq4, grouped iq4 twins (v4/v5),
  gemm_iq4_xs_hp, gemv_iq4_xs z-batching, per-projection Q8_0|IQ4_XS pipe
  flags, ssm_ba dequant + de-interleave (beta rows g*4+{0,1}, alpha g*4+{2,3}).
  Gates: 80B caseMoe blk 0 PASS (sel exact, max_rel 1.1e-4 — tron saw 5.5e-5
  Vulkan-side, same class), ablock blk 3 BIT-EXACT (exercises iq4 P1/P2 + q8
  attn_v + iq4 wo), serve smoke coherent + deterministic.
- Two Metal-specific infra pieces the 42.8 GB file forced:
  * weight WINDOWS — file > maxBufferLength (~36 GB): mmap split into
    page-aligned no-copy buffers cut at tensor boundaries. Boundary-page
    seam bug (first-byte window match read past the window -> all-zero
    logits) fixed by whole-tensor containment; QK_WIN_MAX=<GB> forces
    windows for debugging; 35B token-exact through 4 forced windows.
  * stage-scoped QK_MLOCK — a split stage wires only owned tensors
    (whole-file mlock would have wired 58 GB next to the 35B worker).
- Workers: :18100 35B 22:40 UNTOUCHED (rss 15.5 GB). :18200 80B 12:48
  nCtx 32768 2 slots, 2 windows, stage mlock 32.0 GB, rss ~30 GB.
  Total wired ~48 GB / 64.
- For tron: worker is up per the brief ('report the worker up' -> task #43
  head wiring). Token-exact refs vs refs-80b/ref{1..3}.json still owed —
  run them over the wire once the head connects, or ship the ref files and
  I'll gate locally. caseToken ('token' mode) is still 35B-only (q6k embed +
  q8 proj assumptions); serve/stage paths are the gated ones.

## 2026-07-10 (evening): chunked delta rule (pp512 1333, gap 1.08x) + tron's kDiv fix mirrored, worker relaunched

### pp512: 1213 -> 1332/1333 tok/s (chunked gated delta rule, c38e951)
dn_step_batch was the last token-serial kernel: 32 TGs looping all 512
tokens with 2 barriers each. Replaced for batched prefill by the chunked
delta rule (derivation in dn_chunk.metal header; llama.cpp's
delta-net-base.cpp chunking is the same algebra): per 64-token chunk,
(I+M)D = betaV - W S0^T with M strictly lower -> T=(I+M)^-1 distributes, so
u~=T(betaV), w~=TW, Att are state-independent (parallel over head x chunk);
the chunk-serial rest is 3 matmuls/chunk. Numbers (dncmp, Tn=512, 1 layer):
- scalar chunk-serial kernel: 5769 us — WORSE than sequential (2845).
  Finding 1: dynamically indexed thread-local scalar arrays (u[64], dcol[64])
  spill to scratch; float4[16] with compile-time lanes promotes. Fixed solve
  575->180 us, step barely moved.
- Finding 2 (the big one): per-lane same-address loops — every thread
  reading the same staged row (st4[t*nv+i], Att row) — serialize on AGX at
  ~10x ALU cost. Phase-bisect: D-sweep 560 us, o-loop 1095 us, S-update
  374 us against a ~47+70+47 us ALU model. No scalar layout fixes this;
  the answer is the hardware path: simdgroup MMA (packed-block round redux).
- dn_chunk_step v2: the 3 matmuls as simdgroup_float8x8 MMA, state streamed
  from device (transpose-loads for S^T/D^T), solve pre-bakes {u~, -w~ (negated
  so D-pass is pure multiply_accumulate), q~, K~, packed 8x8 Att, e^Llast},
  diag-tile MMA folds the elast*S term, tgpig.z = 64-wide state column panel
  (panels touch disjoint S rows -> 64 TGs). Step 2658 -> 607 us; chain
  kq 376 + solve 303 + step 607 = 1336 us vs sequential 2845 (2.13x).
Gates: dncmp o<=6.5e-4 / state<=4.9e-3 over Tn={1,5,64,65,200,512} (now both
kDiv pairings), parity + parity-long token-exact, prefilldecode HANDOFF
EXACT, prefillcmp 41/42 — the sole miss is the DOCUMENTED N=48 seed=42
near-tie (margin 0.042, chunk-off control 42/42; historical flap case).
pp512 batched-only A/B/A/B after 60s cooldown (QK_MOE_GROUPED=5, lap):
430.4/430.2 ms (chunk off, =1190) -> 384.5/384.1 ms = 1332/1333 tok/s.
Gap vs llama 1443.7: 1.083x. QK_DN_CHUNK=0 keeps the old path.
Remaining DN headroom: kq 376 us (scalar dots; MMA-able), solve 303, and
the step's 607 vs ~160 us flop model. Then residual bucket (57.8 ms),
attn (19.4 ms).

### tron 64192d5 mirrored: DN k-pairing is ARCH-DEPENDENT (+ gemv-z verified)
Tron's cross-box gate failure was head-side, but both engine bugs live in
ported code. (1) gemv_iq4_xs z-batching: our Metal port already offsets
both x (rq*K) and y (rq*M) — verified, no change. (2) THE bug: qwen3next
pairs v-heads consecutively onto k-heads (kh = h/2), not modulo
(kh = h % hK, qwen35moe). Every 80B DN layer ran with wrong k/q rows —
coherent-but-wrong generation; ablock gated blk 3 (attention) so no DN
block was ever reffed, and dncmp/random harnesses use the same mapping on
both sides so they can't see it. Mirror of tron's fix: kDiv push constant
(0 = modulo, else h/kDiv) in dn_step, dn_step_batch, dn_chunk_solve
(dn_chunk_kq/step never map v->k), set from general.architecture at open;
dncmp now gates BOTH pairings vs dn_step_batch. 35B parity + parity-long
re-run token-exact (kDiv=0 is bit-identical). 80B serve smoke now reads as
consistent first-person dialogue (was context-drifty before the fix).
- Worker :18200 relaunched per tron (same config, 12:48 32768 2, mlock)
  after tron's agent stopped it 18:10 (user memory pressure + the gate
  fail). Tron runs the decisive token gate vs refs-80b over the wire.
- Merged origin/main 64192d5 into metal-port (server/deploy/tooling side
  is tron's; engine twins patched here).
- **GATE CLOSED (tron, same evening): 80B split token gate PASSED** through
  the real split (in-cluster head + this worker): ref3 100/100 exact;
  ref1/ref2 prefix-exact to certified llama near-ties (top-2 gaps 0.006 /
  0.11 logprob — llama's Q8_K activation noise, qk picks the #2). ~30 tok/s
  steady over WiFi, determinism x2 exact. NOTE: the gated worker ran the
  NEW build — so this also validates chunked DN (c38e951) at kDiv=2
  end-to-end on the 80B. The "token-exact refs owed" item from the earlier
  80B section is closed. :18200 is prod-standing now (same as :18100);
  runbook docs/split-serving.md (tron 414bee7). Task #43 closed.

## 2026-07-12 -- pp512 BEATS llama.cpp: flash-attention MMA (1468 vs 1452)

The last un-flipped scorecard axis. Prefill was 0.96x llama (1399 vs 1452);
everything else already wins (decode 1.42x, aggregate 1.31x, RSS 318 MB vs
17 GB). Stage isolation at N=512 (v5 record config, thermal-stable full=366.0
+/-0.3 ms): gu 124 / down 79 / dn 78 / proj 70 / attn 31 ms. gu/down/proj are
mature (llama-parity packed kernels); attn was the one stage far above its flop
model (~7x) -- fa_attn_batch was NAIVE (scalar per-thread dh=256 dot, one
threadgroup per (head,query), full K/V re-read per query, no MMA).

**fa_attn_batch_mma** (shaders/metal/fa_batch.metal): tiled online
flash-attention. Grid (hQ, 1, ceil(Tn/16)); QTM=16 queries/tile, KBM=64-key
blocks, 8 simdgroups. S=Q K^T and O=P V via simdgroup_float8x8 MMA (K^T via
transpose-load, same pattern as dn_chunk_kq); O accumulator + online-softmax
state in threadgroup memory (dh=256 too fat for a register-resident O -- this
matches llama.cpp's own dk256 kernel choice); causal mask, GQA 16:2, sigmoid
output gate fused in the epilogue. Key-tiles fully past the causal bound are
skipped (P=0). Same buffers/signature as the scalar kernel; opt-in via
QK_FA_MMA=1 (engine dispatch picks pAttnBM with the tiled grid).

Numbers (N=512, QK_MOE_GROUPED=5, QK_MLOCK, lap-thermal, interleaved A/B):
attn 30.2 -> 13.2 ms (-56%); pp512 365.9/366.1 -> 348.5/348.9 ms =
1399 -> 1468/1469 tok/s. llama.cpp 1452 (llama-bench -r5, same box/GGUF):
qk 1468 = 1.011x -- prefill flipped to a WIN, the +17 ms clearing the ~7 ms
run-to-run noise band. Gates (all green): scalar-proj prefillcmp 36/36 @ rel
1e-6 (f32-exact vs serial -- the MMA attn adds only fp-order noise);
default-hp prefillcmp 35/36 (sole miss = documented N=48 seed=42 near-tie,
dice at the f16 floor -- scalar batched flips the same die); prefilldecode
HANDOFF EXACT; base>0 multi-chunk (ids4 1040 tok, 3 chunks via serve-test)
MMA == scalar batched EXACT. Still opt-in pending a squeeze pass (13 ms leaves
headroom vs the ~86% ALU-util ceiling; llama.cpp levers not yet applied:
fast::exp2 + log2e-folded scale, per-KV-block causal skip-mask,
simdgroup_barrier(mem_none) scheduling, nsg sweep) and prod promotion
(rebuild build/ + worker relaunch + tron gates).

**Thermal-robust confirmation (same-day, warm/soaked state):** the 1468 figure
was cool-state. Re-measured llama.cpp pp512 in the CURRENT soaked state (in-tree
llama-bench -r5) = 1419.8 +/- 9.5 tok/s -- llama ALSO droops under sustained
soak (its 1452 was cool; PORT.md M0's "thermally flat" held only across 2 runs,
not a long soak). Same warm state, back-to-back: qk-MMA 356.6-357.5 ms =
1432-1436 tok/s vs llama 1419.8 (360.6 ms). So qk wins **1.010x warm** and
1.011x cool -- the prefill lead is THERMALLY ROBUST, ~1% both states, not a
cool-state artifact. (Earlier scare of "qk 357 < llama 1452" was apples/oranges:
qk-warm vs llama-cool.)

**codex gpt-5.6-sol squeeze pass (perf-neutral, kept):** handed the kernel to a
Codex agent for a squeeze. Its managed shell had no Metal device (GPU-denied),
so it flew blind and applied only provably-safe transforms: softmax to log2
space with fast::exp2 (log2e folded into qs), and per-KV-block causal skip
(fully-causal blocks bypass the per-element mask). Verified here: scalar-proj
prefillcmp 36/36 @ rel 7.9e-7 (tighter), prefilldecode HANDOFF EXACT, base>0
serve-test EXACT. Interleaved A/B (shader hot-swap via QK_SHADER_DIR) vs the
pre-squeeze kernel: identical within 0.5 ms -- a WASH. The 13 ms attn is
MMA/bandwidth-bound, not exp/mask-bound, so those levers don't move it; the
real remaining lever is tile/nsg geometry + register-vs-tgmem O, which needs
on-GPU measurement (codex couldn't). Kept the change anyway (correct + the
idiomatic exp2 path).

## 2026-07-12 (later) -- split-K decode attention: 1.36x deep-context decode

Decode/serving attention fa_attn_srv is ONE threadgroup per (q-head, slot)
serially walking the whole KV in 256-key blocks -- fine at short ctx (attn
tiny), but at the 80B worker's deep context it's a long serial walk in ~32 TGs
on 40 cores (the pathology tron flagged; tron shipped split-K, byte-exact,
2.4x deep-ctx on their Vulkan side). Short-ctx tg128 decode (120 tok/s) is
near its structural limit -- the real decode lever is long-ctx, which is what
prod serves.

**fa_attn_srv_split + fa_attn_srv_reduce** (flash-decoding, fa_srv.metal):
split runs one TG per 256-key chunk (grid hQ x nChunks x nSlots) emitting an
unnormalized (acc[dh], m, l) partial (over-length chunk-TGs early-return);
reduce merges partials per (head,slot) with the online-softmax rescale + gate.
Mathematically exact. Kernels by Codex gpt-5.6-sol; engine wiring (partials
scratch, dspY chunk-axis dispatch, host active-chunk count) mine. Opt-in
QK_FA_SPLIT=1; default scalar unchanged.

Measured (35B, ctx 8000 = 32 chunks, worker down, QK_NO_EOS forces 96 decode
steps, QK_MLOCK, interleaved): scalar 15516 ms vs split 13027 ms for
prefill+96 decode -> **25.9 ms/step decode savings** (prefill identical).
Decode step 99 -> 73 ms = **1.36x deep-ctx decode**, grows with ctx. Gates:
serve-test TOKEN-EXACT vs scalar at ctx 1040 (5 chunks) AND 8000 (32 chunks).

**1.36x is a STRUCTURAL ceiling, not the inner loop.** CK sweep 64 vs 256
perf-identical (512 TGs already fill 40 cores). A codex-authored coalesced
simdgroup-per-key inner loop (32 lanes -> 32 contiguous dims + simd_sum,
token-exact) measured 13176 ms = ~1% SLOWER than the scalar-parallel inner
loop -- reverted. At ctx 8000 the KV reads run at ~40 GB/s of 493 (nowhere
near bandwidth-bound), so the limit is the cold latency-bound per-step command
buffer, not memory coalescing. Past 1.36x needs a structural change
(command-buffer pipelining / multi-step batching / spec-decode), not kernel
tuning. QK_NO_EOS = bench-only flag (force nGen, ignore EOS).

Status: split-K is in the prod build/ (opt-in, off). tron owns the 80B decode
gate -- recommend enabling QK_FA_SPLIT=1 on the worker (dims match 35B) and
gating, for the long-ctx serving win.

## 2026-07-12 (later) -- perf-lever evaluation: what was tried, measured, removed

Systematic pass over the remaining perf areas. Key result that reframes decode:

**Decode is GPU-COMPUTE-bound at ~56% of bandwidth, NOT submit-latency-bound
and NOT KV-bandwidth-bound.** Measured (QK_STEP_STATS, 35B, 80 decode steps):
gpu 8.91 ms/step, encode 0.16 ms, commit+wait 9.11 ms -- so submit overhead is
only ~0.2 ms (2%); the step is 96% GPU compute. 8.91 ms streams the 2.457 GB
active set at 276 GB/s of the 493 ceiling. The 44% gap lives INSIDE the command
buffer (post-barrier DRAM ramp across ~600 small per-layer dispatches), which
M6a already showed fusion doesn't fix (<2%). Implication: kernel-level KV/GEMV
tuning and command-buffer pipelining CANNOT move single-stream decode; only
**speculative decoding** (amortize the per-step cost over K tokens) changes it.

**f16 KV cache -- TRIED, no speed win, REMOVED.** Codex gpt-5.6-sol drafted
full fp16 KV storage / fp32 compute (all kc/vc writers fa_prep{,_srv,_batch} +
readers fa_attn{,_srv,_srv_split,_batch,_batch_mma} -> device half*, +the KV
alloc strides *4->*2; DeltaNet state stays fp32). Built clean; self-consistent
(scalar-proj prefillcmp 36/36 @ rel 6e-4, no float/half mismatch); short-ctx
decode byte-identical to f32 KV. BUT ctx-8000 decode: f16 13424 ms vs f32
13404 ms = NO win (decode is compute-bound per above, not KV-bandwidth-bound),
and f16 KV diverges from f32 at long ctx (accumulated rounding flips the greedy
path -- the accepted f16-KV class, but it forfeits our f32 token-exactness).
Only upside is ~1.8 GB KV memory (of 32 GB wired) -> a modest MEMORY/context
lever, not a speed lever. **Reverted** (git checkout, tree back to 2a8f8a2);
kept the finding here. Revisit only if very-deep-ctx (49k) decode turns out
bandwidth-bound (untestable locally -- 42 GB model won't mlock in free RAM).

**Aggregate multi-slot -- plateaus at 8 slots.** 16 slots 199.6 tok/s vs 8
slots 195.5 (+2%): confirms the weight-read cap (we re-read weights per slot,
GEMV z=slots). Lever = slot-batched decode GEMV (read a weight row once, dot
all slots) ~1.5x at 8 slots -- but the prod worker is 2 slots, so modest there;
mainly a high-concurrency win. NOT yet implemented. Also found a CORRECTNESS
BUG: at 16 slots serve-test reports "all slots identical: NO" (8 slots = YES) --
same prompt must be deterministic-identical, so >8-slot serving is broken;
needs a bisect (unrelated to the above changes).

Standing map after this pass: single-stream decode near structural limit
(spec-decode is the only game-changer, big project); prefill won (1.01x, DN
solve/step MMA is the remaining ~10-20 ms incremental); aggregate has
slot-batched GEMV (modest for 2-slot prod) gated behind the 16-slot bug.

## 2026-07-12 (Codex remaining-lever pass) -- compact DN solve, 9.6% kernel win

Three token-exact forward-solve layouts were measured back-to-back at Tn=512.
The two larger rewrites were removed:

- Blocked 8x8 f32-MMA forward substitution: a prep kernel emitted N=-M and
  four simdgroups solved four 8-column panels per TG. The first high-occupancy
  geometry measured prep 144.9 + solve 399.3 us versus the scalar solve's
  279.5 us; reusing M in one heavier TG measured 143.6 + 438.1 us. Both were
  correct in dncmp, but AGX f32 MMA + the diagonal barriers lost decisively.
- Split-RHS scalar solve: 256 threads shared M while each thread kept only one
  float4[16] RHS live (half the per-thread solve state). It remained exact but
  measured 292.4 us versus 277-280 us, so it was also removed.

The kept change stores only M's strict lower triangle in threadgroup memory:
2016 floats / 7.9 KiB instead of a zero-padded 4096 floats / 16 KiB. Arithmetic
and output are bit-identical; the smaller allocation admits another resident
solve TG per core. Interleaved isolated A/B/A/B: dense solve 281.0/281.8 us,
compact 254.5/254.0 us (**9.6% faster**); full DN chain 960.5/963.3 ->
937.4/937.4 us. pp512 is a small but repeatable end-to-end movement: five-way
interleaved medians 347.06 -> 346.46 ms (1475 -> 1478 tok/s, ~0.17%).

All acceptance gates green: scalar-projection prefillcmp 36/36 TOKEN-EXACT
(worst rel 7.9e-7); dncmp all PASS for kDiv 0 and 2; block/ablock PASS; ids3
20-token decode exact; serve-test 8-slot identical YES; prefilldecode HANDOFF
EXACT.

## M0 — llama.cpp Metal baseline (2026-07-08)

**Gate decision: NOT triggered — proceed with the port.** llama.cpp Metal
decodes at 84 tok/s = 38% of the theoretical-bandwidth ceiling (42% of
measured achievable bandwidth). The 70% gate would have required ≥141–156
tok/s. Headroom for the port is ~2.3× on the weights-only ceiling.

### llama-bench

llama.cpp master `f2d1c2f` (2026-07-08), Metal backend, Release, static,
`GGML_METAL_EMBED_LIBRARY=ON`, defaults (`-fa auto`, mmap on, all layers on
GPU), `-r 5`, plugged in, `caffeinate -i`. Two full runs back-to-back; run 2
is thermally soaked.

| test | run 1 (t/s) | run 2, soaked (t/s) |
|---|---|---|
| pp512 | 1452.08 ± 11.54 | 1452.60 ± 8.02 |
| pp2048 | 1424.97 ± 12.43 | 1325.23 ± 53.15 |
| tg128 | 84.00 ± 0.30 | **84.18 ± 0.10** |

Cross-check: `llama-cli` single-turn greedy (`--temp 0`, chat template,
thinking mode on by default) reports 84.0 t/s generation, 153.9 t/s on a
short prompt; output coherent. Decode is thermally flat (84.00 → 84.18);
prefill droops ~7% at pp2048 when soaked (compute-bound; decode at ~207 GB/s
doesn't stress the package).

### Measured GPU bandwidth

Metal microbench (`bench/metal_membw.swift`): 4 GiB private buffer, coalesced
float4 grid-stride read-sum, 40×32×4 threadgroups × 256, timed over 8
dispatches, 3 trials: **493.1–495.6 GB/s** = 90.5% of the 546 GB/s
theoretical. `recommendedMaxWorkingSetSize` = 51 GiB (matters for the
Qwen3-Next-80B follow-on: ~45 GB Q4 fits, tight).

### Active set per decoded token (exact, from GGUF tensor metadata)

`bench/active_bytes.py` parses the GGUF header and sums tensor bytes by
class (routed experts scaled by top-8/256):

| class | tensors | stored GB | read GB/token |
|---|---|---|---|
| dense — attn/DeltaNet/norms/router/output head | 452 | 1.881 | 1.881 |
| routed experts (top-8 of 256) | 120 | 14.158 | 0.442 |
| shared expert (always on) | 160 | 0.134 | 0.134 |
| token_embd (row lookup) | 1 | 0.417 | ~0 |
| **total weights** | | **16.590** | **2.457** |

Non-weight traffic per token at tg128: DeltaNet state read+write ≈ 0.126 GB
(30 linear layers × 32 heads × 128×128 f32 × 2), KV cache ≈ 0.003 GB
(10 full-attn layers, 2 KV heads × 256, f16, short ctx). Negligible KV at
short context; grows to ~0.04 GB/token at ctx 2048.

### Ceilings and extraction

| basis | ceiling (tok/s) | llama.cpp 84.2 = |
|---|---|---|
| 546 GB/s theoretical, weights only (charter method) | 222 | 37.9% |
| 494 GB/s measured, weights only | 201 | 41.9% |
| 494 GB/s measured, weights + DeltaNet state | 191 | 44.0% |

llama.cpp's effective decode bandwidth: 84.2 × 2.457 GB = **207 GB/s** of
494 measured. (For calibration: the charter's 220–270 band assumed 2.0–2.5
GB/token; the real number is 2.457, i.e. the bottom of the band, 222 tok/s.)

Targets in absolute terms: DoD ≥1.3× llama.cpp = **≥109.5 tok/s** (= 54% of
the measured-bandwidth ceiling). RDNA3-parity extraction (this engine gets
~76% of achievable there) would be **~153 tok/s ≈ 1.8×** llama.cpp. Stretch
150 tok/s = 75% of measured ceiling — same territory as the RDNA3 result.

### Architecture facts pinned down for the port (from GGUF metadata)

- arch `qwen35moe`: 40 blocks, hidden 2048, `full_attention_interval = 4` →
  10 full-attention layers (blk 3,7,…,39), 30 gated-DeltaNet layers.
- Full attn (gated): `attn_q` 2048→8192 (16 heads × 256 **× 2 — Q fused with
  output gate**), `attn_k`/`attn_v` 2048→512 (2 KV heads, GQA 8:1), per-head
  q/k RMS norm, partial RoPE dims [11,11,10,0] of 64, freq base 1e7.
- DeltaNet: fused `attn_qkv` 2048→8192 (q 16×128 grouped, k 16×128, v
  32×128), `attn_gate` 2048→4096, `ssm_out` 4096→2048, conv1d k=4 over 8192
  channels, `ssm_alpha`/`ssm_beta` 2048→32 (per-head decay/gate, 32 heads),
  state 32×128×128 f32, `ssm_norm` RMS over 128.
- MoE (every layer): router `ffn_gate_inp` 2048→256 in F32, 256 experts
  top-8, expert FFN 512, shared expert FFN 512.
- Output head `output.weight` Q6_K 2048×248320 = **417 MB — 17% of every
  token's read budget**; vocab 248,320. Prime fusion/quant target.
- Quant mix: IQ3_XXS 80 t (8.22 GB, expert bulk), IQ4_XS 37 t (5.28 GB),
  Q6_K 5 t (1.49 GB), Q8_0 250 t (1.49 GB, dense projections), F32 361 t
  (0.10 GB, norms/router/ssm small tensors). GEMV coverage needed: Q8_0,
  Q6_K, IQ4_XS, IQ3_XXS (+ F32 helpers) — same set as the Vulkan engine.
- ggml Metal on this device: simdgroup reduction + matrix available, bfloat
  available, "tensor API" (M5+) not available, unified memory + residency
  sets in use.

## M1 — Metal harness + f16 GEMV (2026-07-08)

**Delivered:** Metal host skeleton + the `qk` CLI on macOS + `gemv_f16`
validated against the CPU reference and measured at **94–95% of theoretical
DRAM bandwidth** on big tensors (515 GB/s on the 971 MiB output-head shape;
the streaming-microbench ceiling from M0 was 494, so the GEMV beats the
naive streamer).

New files: `src/main_metal.mm` (ObjC++ TU mirroring `src/main.cpp`
one-to-one: same CLI, same scale-aware validation, same output format),
`shaders/metal/gemv_f16.metal`. CMake grows an `if(APPLE)` branch; the
Vulkan/Linux path is untouched. MSL is JIT-compiled at runtime from
`shaders/metal/` (mirrors the `loadSpv` flow, `QK_SHADER_DIR` wins; no
offline metal toolchain needed — relevant on macOS 26 where `xcrun metal`
is a separate download). Buffers are `storageModeShared`: upload/readback
are plain memcpy on UMA, no staging path exists. Spec constant → function
constant (TPR); push constants → `setBytes`; `subgroupAdd` → `simd_sum` /
`simd_shuffle_down`; the Vulkan bench's barrier-serialized dispatch loop →
one encoder whose Y write-after-write hazard serializes dispatches;
timing via `MTLCommandBuffer` GPUStart/EndTime.

f16 GEMV, correctness PASS on every shape (max_rel_err ≤ 2.5e-4, tol 1e-2):

| shape (M×K) | W | tpr | GB/s | note |
|---|---|---|---|---|
| 16384×8192 | 256 MiB | 256 | 510.5 | suite default |
| 8192×8192 | 128 MiB | 256 | 518–546 | run-to-run spread, partial SLC |
| 248320×2048 | 971 MiB | 64 | **515.5** | output-head shape, cleanest DRAM number |
| 8192×2048 | 32 MiB | 64 | 611 | **fits SLC — inflated, see below** |
| 8192×512 | 8 MiB | 32 | 298 | launch-bound (~6 µs/dispatch floor) |
| 4096×128 | 1 MiB | 16 | 70 | latency-bound; simd_shuffle path exercised |

Findings, in port-note form:

- **The RDNA3 TPR heuristic left 40% of bandwidth on the table** for
  M-huge/K-small shapes: at tpr 256 the head GEMV gives each thread a
  single 16 B load (315 GB/s); tpr 32–128 → 517–523 GB/s. New rule: after
  the Vulkan skinny-row shrink, keep halving TPR while per-thread work
  < 4 units and ≥1024 threadgroups remain. `QK_TPR=<n>` overrides for
  crossover experiments (M6).
- **M4 Max SLC (~48 MB) inflates small-W benches:** the 32 MiB shape
  "streams" at 611–754 GB/s because iterations re-hit cache. Real decode
  streams 2.5 GB/token through the SLC, so per-layer GEMVs in a token step
  run at DRAM speed — bench conclusions must come from big or rotating
  weights (M6 must re-check crossovers against real steady-state). Flip
  side: activations/state are small and can live in SLC essentially free —
  fused blocks (M3+) should lean on that.
- Per-dispatch overhead floor is ~6 µs at small sizes — visible on the
  8 MiB expert-shaped GEMV (298 GB/s). The full-layer fusion strategy
  (pre-recorded command buffers, few submits) matters on Metal exactly as
  it did on Vulkan.
- `qk list`/`qk gguf` are wired; this model has no F16 tensors, so
  real-tensor GEMV validation starts when the quant kernels land (M2).

(Resolved blocker: HTTPS pushes failed for the whole session — stale gh
token + keychain locked to non-interactive shells. SSH auth worked;
remote switched to git@github.com and the branch is pushed.)

## M2 — quant GEMVs: q8_0, q6_k, iq4_xs, iq3_xxs (2026-07-08)

**Delivered:** all four quant GEMV kernels ported to MSL, PASS against the
CPU dequant references on synthetic random blocks AND on real tensors
mmap'd from the GGUF. `qk suite` runs the full five-kernel matrix; `qk gguf
<tensor>` works for every weight type in this model.

Kernels are hand-written MSL from the GLSL (not SPIRV-Cross): same block
structs (static_asserted to ggml sizes), same work decomposition, subgroup
ops → simd ops. `gemv_q8_0`/`gemv_q6_k` keep the grid-z query batching the
engine uses for multi-slot decode. `#include "iq_tables.metal"` is resolved
by a tiny include inliner in `loadMetalSource` (runtime MSL has no include
paths). One MSL gotcha vs GLSL: thread-position attributes must be all
scalar or all vector — mixing `uint tid` with `uint3 tgpig` is a compile
error.

Correctness (tol 1e-2, scale-aware rel err; all PASS):

| kernel | synthetic 8192×8192 | real tensor | max_rel_err (synth / real) |
|---|---|---|---|
| q8_0 | PASS | blk.0.attn_qkv 2048→8192 (17 MB) | 1.9e-4 / 1.5e-4 |
| q6_k | PASS | output.weight 2048→248320 (398 MB) | 6.0e-5 / 2.0e-4 |
| iq4_xs | PASS | blk.0.ffn_down_exps[:,:,0] | 1.3e-4 / 2.3e-5 |
| iq3_xxs | PASS | blk.0.ffn_gate_exps[:,:,0] | 1.6e-4 / 9.8e-6 |

Bandwidth after the face-off rework (v2 kernels, see next section):

| kernel | synth 8192×8192 | real tensor | note |
|---|---|---|---|
| f16 | 520 | 515 (head shape, M1) | 94–95% of 546 |
| q8_0 | 511 (522 @ 32768×8192) | 708 (attn_qkv, SLC) | 96% of peak, done |
| q6_k | 499 | **499 output.weight (398 MB)** | head now 0.84 ms/token |
| iq4_xs | 407 | tiny slice, latency-bound | +34% vs v1 |
| iq3_xxs | 353 | tiny slice, latency-bound | +27% vs v1, beats llama.cpp |

First iq3_xxs step (before the face-off): 32-element work unit instead of
the GLSL's 8 (aux u32 read once instead of 4×, packed_uchar4 grid loads)
— 201 → 288 GB/s. The v2 rework below took it to 353.

**Decode projection at v2 kernel speeds** (per-token active set from M0):
Q8_0 1.49 GB @ 522 = 2.85 ms; head Q6_K 0.417 @ 499 = 0.84 ms; expert IQ3
0.257 @ 353 = 0.73 ms; expert IQ4 0.165 @ 407 = 0.41 ms; Q6_K expert-downs
0.021 = 0.05 ms; F32 0.10 + DeltaNet state 0.126 @ ~520 = 0.43 ms →
**≈ 5.31 ms ⇒ 188 tok/s GEMV-streaming bound**. RDNA3 realized ~75% of its
equivalent bound end-to-end; 75% here ⇒ ~141 tok/s ≈ 1.68× llama.cpp. The
1.3× DoD (109.5 tok/s) has real margin; the 150 stretch needs ~80%
realization — fusion quality (M3/M6) decides it.

### M2b — kernel face-off vs llama.cpp Metal, and the rework it forced

Ryan asked for a llama.cpp-vs-qk comparison on this box. Method: llama.cpp
`test-backend-ops perf -o MUL_MAT` (build `f2d1c2f`, its own Metal kernels,
n=1 decode GEMV, m=4096 k=14336) vs `qk <fmt> 4096 14336` — identical
shapes, identical bytes, both loop-hot. µs/op is the comparator.

Round 1 (my straight GLSL ports): f16 and q8_0 at parity or ahead; q6_K
**26% behind** (136.4 vs 101.3 µs), iq4_xs **22% behind** (97.8 vs 75.9),
iq3_xxs 7% behind. So I read their kernels (MIT) and adopted the work
shape for those three: one simdgroup per NR0 consecutive rows (2/2/4),
NSG=2 simdgroups per 64-thread threadgroup, x staged in registers and
reused across rows, q6_K quadrant dequant via constant masks (no variable
shifts), iq4_xs qs as uint32 + 0x0f0f0f0f nibble masks + byte-aliased
codebook lookups from threadgroup memory, iq3_xxs grid+signs staged in
threadgroup memory; sign application kept vectorized (float4 select) —
that last bit is mine and is why iq3_xxs ends up *ahead* of llama.cpp.

Final standings, same shape, this box (µs/op, lower wins):

| format | llama.cpp | qk v2 | qk vs llama.cpp |
|---|---|---|---|
| f16 | 230.5 | 229.3 | +0.5% |
| q8_0 | 125.0 | 121.5 | **+3%** |
| q6_K | 101.3 | 106.3 | −4.7% |
| iq4_xs | 75.9 | 80.4 | −5.6% |
| iq3_xxs | 78.4 | 68.2 | **+15%** |

NSG sweep confirmed llama.cpp's nsg=2 beats 4 and 8 on all three reworked
kernels (e.g. iq3_xxs: 68.2 / 75.0 µs at nsg 2/4). Weighted by this
model's actual per-token byte mix (Q8_0-heavy dense + IQ3 experts), the qk
set is now net faster than llama.cpp's kernels for this workload; the
remaining ~5% q6_K and iq4_xs gaps are unroll/codegen detail — M6. The small-M
regression from fixed row-pair geometry (expert slices, M=512: 64
threadgroups) doesn't matter for decode — the engine's MoE path batches
8 experts × slots on the z-axis (M3).

Quant type census for the port (which kernel serves what, per token):
Q8_0 = all dense attn/DeltaNet projections + shared experts (1.49 GB);
Q6_K = output head + token_embd (row lookup) + ffn_down_exps in blk
34/38/39 only (unsloth dynamic bump); IQ3_XXS = all routed gate/up experts
(80 tensors, 8.2 GB stored); IQ4_XS = routed down experts in the other 37
layers. F32 = norms, routers, ssm small tensors.

## M3 — fused blocks: MoE step, DeltaNet block, attention block (2026-07-09)

**Delivered:** the three fused decode structures, each validated against the
CPU reference and timed. `qk moe`, `qk block`, `qk ablock` all run on Metal.

| structure | correctness | µs (this GPU) | note |
|---|---|---|---|
| MoE step (`qk moe 0`) | PASS 1.2e-5, router MATCH | **64.2 µs/layer** | 4 dispatches, was 148.5 as a naive port |
| MoE step, q6k downs (`qk moe 34`) | PASS 7.0e-6 | 87.1 | blk 34/38/39 variant |
| DeltaNet block (`qk block 0`) | PASS ≤9.3e-5 ×3 tokens; state drift 6e-8 | **178.4 µs** | 14 dispatches incl. MoE |
| attention block (`qk ablock 3`) | PASS ≤8.3e-5 ×3 tokens | **170.2 µs** (pos=2) | 13 dispatches incl. MoE |

Projected decode from block timings: 30×178.4 + 10×170.2 + head 0.84 ms +
embed/argmax ≈ **8.0 ms/token ≈ 126 tok/s ≈ 1.50× llama.cpp** — above the
1.3× DoD line before M4 glue and M6 tuning. (Stretch 150 needs ~1.3 ms more
off the blocks; see the overhead notes below.)

Structural findings that drove the design (measured by
`bench/metal_barrier_cost.swift`):

- **memoryBarrier between dispatches costs ~3.5 µs** (not the problem);
  **threadgroup launch throughput ~150 tgs/µs is the real tax** — a
  2048-threadgroup dispatch takes ~14 µs even empty. Kernels therefore use
  one SIMDGROUP per output row (not one 256-thread group), NSG=4 simds per
  threadgroup, which cut the MoE chain 148.5 → 82 µs.
- **A lone simdgroup doing a 2048-element dot is latency-bound (~20 µs)**
  — no other warps hide DRAM latency. moe_select's shared-gate dot moved
  into the parallel moe_logits dispatch (virtual row n_expert): 84 → 64 µs.
- Merging the shared expert INTO the routed gateup/down dispatches (4
  dispatches, 3 barriers, single y write) simplifies ordering; hazard
  tracking is off (untracked buffers) and ordering is explicit
  memoryBarrier in a concurrent-dispatch encoder — the Vulkan model.
- dn_qknorm is FUSED into dn_step's staging pass (the head is resident in
  threadgroup memory; norm = one simd_sum) — one dispatch + one device
  barrier saved; k-head norms recompute per v-head, cheaper than the
  barrier.
- Remaining per-block time is ~76 µs genuine serial work (qkv GEMV ~30,
  delta state R/W 8.4 MB ~21, ssm_out GEMV ~16) + ~40 µs stage overhead
  (launch + barrier + single-tg latency kernels: rms/addN/add/select).
  M6 leads: merge rms into the preceding residual add (add_rmsnorm
  pattern engine-wide), atomic last-tg-standing logits→select fusion,
  nr0=2 row pairs in the MoE kernels, f16 h/activations.

Kernel inventory after M3: 19 MSL kernels; every decode-path operation of
the hybrid architecture now has a validated Metal implementation.

## M4 — end-to-end greedy decode: token-exact parity at 1.43× (2026-07-09)

**`qk token` runs the full model — and beats llama.cpp Metal with exact
correctness.**

- **Parity: greedy output is token-for-token identical to llama.cpp Metal**
  (same build, same GGUF, same input ids, temperature 0) on 3 prompts ×
  100 tokens — 100/100 each. References via `llama-server /completion`
  with `"temperature":0` (NOTE: do NOT pass `"samplers":[]` — that removes
  the temperature stage and silently makes the reference non-greedy; an
  hour went there).
- **Decode: 8.26–8.36 ms/token = 119.7–122.1 tok/s**, thermally flat over
  repeated runs. llama.cpp Metal tg128 on this box: 84.2 tok/s →
  **1.43×** (DoD ≥1.3× met; stretch 150 = M6 target).
- Prefill in this harness is serial per-token (~9 ms/token after warmup;
  first command buffer pays ~1.6 s of first-touch page-in for the 15 GB
  resident set). Batched prefill is M5 (`prefillbench` on RDNA3 got 4.2×
  serial).

Structure: the whole generation is TWO command buffers (one prefill, one
decode). GPU-resident sampling: head GEMV (q6_k, nsg=2) → two-pass argmax
(ties to lower index, matching llama.cpp greedy) → winner recorded to a
history buffer and fed straight into `embed_q6k` for the next position —
the host reads ids only after the last token. Layer chaining uses
`add_rmsnorm` as the layer tail (residual sum + NEXT layer's attn_norm;
output_norm after layer 39), so a layer is 13 (attn) / 14 (deltanet)
dispatches. ~15.0 GB resident (UMA, untracked shared buffers).

Per-token budget at 8.3 ms: 30 deltanet blocks ≈ 5.3 ms + 10 attn blocks
≈ 1.7 ms + head+argmax+embed ≈ 1.0 ms + inter-block glue ≈ 0.3. M6 levers,
in expected-value order: q8_0 GEMV shapes dominate the blocks (qkv 30 µs +
out 16 µs per deltanet block — nr0-pair rework like M2b's q6_k could
reclaim ~15-20%); the ~40 µs/block fixed overhead (single-tg latency
kernels + launches); iq3 gateup ALU.

## M6a — fusion experiments: what moved, what didn't (2026-07-09)

Three structural fusions attempted after M4, each validated (block/ablock
PASS, token parity re-verified 100/100 ×3 prompts after every change):

1. **dn_step v3** — conv+silu (idempotent state shift), q/k L2 norm, delta
   rule, and the gated output norm in ONE dispatch (was four). The shared
   q/k channels of a k-head are convolved redundantly by the hV/hK
   v-head threadgroups; the state shift writes identical values (benign
   race). Block: 178.4 → 172.8 µs. KEPT.
2. **Inline expert re-selection** (moe_pick_all per consumer simdgroup, no
   select stage): correct but **39% SLOWER** (MoE 64 → 89 µs) — the
   +30-register footprint (v[8]+ids[8]+ws[8]) collapses occupancy on the
   DRAM-bound expert kernels and latency hiding dies. REVERTED; the
   dedicated 1-simdgroup select stage stays. Lesson: on Apple, a fat
   memory-bound kernel buys nothing from absorbing scalar work that
   needs registers.
3. **moe_logits_addn** (residual add + post_attention_norm folded into the
   router-logits dispatch; every simd recomputes the scalar RMS from
   SLC-hot vectors — register-light, unlike #2) and **embed_q6k + layer-0
   rms fusion**: correct, ~neutral on time (−1 stage/layer and −1/token
   but block time flat within noise). KEPT (fewer stages = simpler M5
   chunk encoding), but the honest reading:

**Stage-count reduction has hit diminishing returns.** Removing ~200
stages per token (5/layer) moved end-to-end <2%. The block runs at an
effective ~350 GB/s against 60.3 MB/layer while the same kernels do
500-522 standalone — the gap is INSIDE the bandwidth-bound stages:
per-stage DRAM ramp from idle after each barrier, and the delta-state
access pattern (each thread walks a private 512 B row; line utilization
recovers only across 8-iteration windows). Decode: 118.6–119.8 tok/s
(vs 119.7–122.1 pre-fusion — flat), parity intact.

M6 leads, updated by evidence: (a) ~~delta-state transpose~~ TESTED —
[i][j] scalar layout is 16% WORSE (203 vs 175 µs/block): 4× instruction
count and dependent scalar chains beat the coalescing win; the float4
row layout already recovers line utilization over its 8-iteration
windows. Reverted. (b) nr0 row-pairing for the in-block q8_0 GEMV
shapes; (c) probe the post-barrier bandwidth ramp with a two-dispatch
microbench — if real, fewer/fatter stages is the endgame; (d) f16
h/activations; (e) speculative decoding (docs/spec-decode-qk-plan.md)
is the step-change lever once serving works.

## M5 — serving on Metal: engine, harnesses, server, CLI round trip (2026-07-09)

**The full serving stack runs on this laptop.** `libqk.dylib` (same TU as
the CLI, `QK_LIBRARY`), the unmodified-in-spirit Rust server (one 12-line
compatibility patch: Claude Code ≥2.2 sends in-messages `system` turns),
and the qk.h engine implemented on Metal with the fused kernel set.

Gates, all green on first full runs:

| gate | result |
|---|---|
| serve-test 1 slot | TOKEN-EXACT vs llama.cpp greedy reference |
| prefillcmp (14 sizes × 3 seeds) | 36/36 argmax MATCH, worst rel 1.3e-6 |
| prefilldecode ×3 seeds | HANDOFF EXACT |
| serve-test 4 slots | all slots identical YES, 104.8 tok/s aggregate |
| serve-test2 staggered | OK |
| cachetest | warm ≡ cold YES |
| /v1/messages | "The capital of France is Paris." |
| Claude CLI tool round trip | **num_turns=2, is_error=false** at ctx 32768 |

The CLI round trip is the real-world stress test: Claude Code's 2026
system prompt is **29.8k tokens**. Turn 1 cold-prefills 29,809 tokens;
turn 2 hits the prefix cache with `reuse=29809 prefill=44` — the
cross-turn O(delta) design carried over from RDNA3 intact, as plain UMA
memcpys (snapshots are host vectors now; no staging buffers, no Vulkan
copy machinery).

Metal-specific engine notes: no pre-recorded command buffers exist on
Metal, so the serial step is encoded per token (~0.4 ms host) with grid
z = highest-active-slot; per-slot state stripes bind via setBuffer:offset;
gemm_q8_0 restructured to BK=32 (Apple's 32 KB threadgroup ceiling vs
RDNA3's 64 KB LDS); srv attention uses online softmax — context is bounded
by KV memory, not threadgroup arrays (32k validated).

Prefill today: 214–241 tok/s batched = 2.5–2.9× serial (Vulkan reached
4.2×; llama.cpp Metal does 1452). Phase B owns this: the scalar f32 GEMM
and the z-per-token MoE are the gaps.

## Phase B (in progress) — prefill: 2.9× serial, GEMM variants benched (2026-07-09)

`caseBGemm` ports the GEMM validation harness. Measured (M=8192, K=2048):

| variant | N=128 | N=512 | correctness |
|---|---|---|---|
| scalar f32, BK=32 (shipping) | 1.92 TFLOPS | 2.34 TFLOPS | PASS 4.5e-4 |
| simdgroup_float8x8, BK=32 | 1.03 TFLOPS | 1.39 TFLOPS | PASS 2e-3 — **slower**: f32 MMA ≈ scalar on Apple + fragment-store tail |
| simdgroup_half8x8, BK=64 | — | — | **FAIL 0.9 rel — bug** (structural, not f16 rounding; bisect vs the passing f32 twin next: BK=64 two-block staging is the delta) |

UPDATE (same day): the half8x8 "bug" was three things at once, now
resolved: (1) the synthetic 1e-2 tolerance is unpassable for ANY f16-input
GEMM under cancellation (llama.cpp's own prefill would fail it) — the real
gate is model-logit argmax-exactness; (2) a rewrite to 8 simdgroups + BK=64
+ direct transposed device simdgroup_store regressed the gates, which was
finally a HOST bug — dispatching the 256-thread kernel with 128 threads
(half the simds never staged/computed); (3) with threads fixed and the
proven bounce store, **gemm_q8_0_h is the engine default**: prefillcmp
36/36 TOKEN-EXACT (worst rel logit 0.073 — llama.cpp's f16 precision
class), prefilldecode HANDOFF EXACT, prefillbench **285–318 tok/s =
3.6–3.85× serial** cold, ~3.0–3.5× thermally soaked. QK_GEMM=scalar forces
the bit-exact f32 path (247 tok/s).

Remaining to B1's 4× and B2's 1452, updated by measurement
(QK_PREFILL_SKIP stage isolation at N=128, chunk 408 ms): **MoE = 218 ms
(53%)**, projections ≈ 48 ms (f16 GEMM did its job), DeltaNet batch ≈ 0,
attention ≈ noise. A read-once grouped gu (token→expert counting sort +
one simdgroup per (expert,row) looping its tokens; bit-identical results,
36/36) measured SLOWER (522 vs 408 ms) — decisive evidence the MoE stage
is **ALU-bound on iq3 decode (~152 GB/s effective), not DRAM-bound**. The
real lever is decode-ONCE: dequantize each expert row to threadgroup
half-precision and multiply all its tokens GEMM-style (the gemm_q8_0_h
structure with an assignment gather). Kernels + QK_MOE_GROUPED env kept
for that next round.

**Thermal methodology (Ryan flagged lap-throttling):** this box visibly
sags under sustained load (+22% on the serial baseline during the long
GEMM session: 1543→1890 ms), so single-shot cross-run comparisons are
unsafe. House rule from here: perf verdicts run **A/B/A interleaved with
the serial column as thermal control**. The grouped-MoE verdict re-ran
under that protocol — control 1548/1548/1557 ms (±0.5%, stable), grouped
508.6 vs ungrouped 407.7/419.8 ms — confirmed slower on merit, not heat.
Instant GPU throttle probe: `qk f16 8192 8192 100` vs the ~518–546 GB/s
cool reference (516.6 measured = full speed). For record runs: hard
surface, not a lap.

Budget at N=128 (530 ms/chunk): projections ≈180 ms (scalar GEMM), MoE
≈180 ms (ungrouped expert reads), dn_step_batch ≈60–120 ms (32 tgs,
serial over Tn — low occupancy by design, state in registers), attention +
small ops ≈50 ms.

### Phase B round 2 — head tax + decode-once grouped MoE (2026-07-09)

**Head tax (found by accident, worth more than the thing I was looking
for):** `prefillBatchLast` dispatched the output head (248320×2048 Q6_K
GEMV ≈ 417 MB of weight reads) with **z = n — once per prefill position**
whenever `wantLogits` was set, then copied out only the last row. The
serving path never paid it (`wantLogits=false`), but every prefill
benchmark did — and llama.cpp's pp512 computes ONE logit row. Fix:
logits-only callers get the head for the last row only (`argmaxOut`
callers keep z=n — the pipe wire contract returns n greedy ids per
frame). prefillcmp bit-identical after (36/36 @ 0.073). N=128 chunk:
435 → 354 ms. **B1 (≥4× serial) passes at 4.26–4.69× with everything
else unchanged.**

**Decode-once grouped MoE, done right this time.** v1 (read-once) failed
because `iq3_g32` sat inside the token loop — same decode ALU, only DRAM
locality. The fix is hoisting the dequant: stage the expert row-tile in
threadgroup memory once per K-chunk, multiply all its gathered tokens via
simdgroup MMA. Four variants in `moe_grouped.metal`, selected by
QK_MOE_GROUPED (env forces the variant at all n; default = v4 at chunk
n ≥ 192, QK_MOE_GROUP_N tunes):

| variant | shape | precision | moegcmp/layer n=512 (uniform sel) | notes |
|---|---|---|---|---|
| ungrouped | row-per-simd | f32 exact | 13.95 ms | prior default |
| 1 | read-once loop | bitwise = ungrouped | 14.63 ms | control |
| 2 | 32 rows × 8 cols | f32 MMA | 11.92 ms | exactness probe |
| 3 | 64 × 32, f16 stage | h-GEMM / mul_mm_id class | **6.06 ms** | record config |
| 4 | 32 × 32, f32 | f32 MMA | 9.55 ms | **DEFAULT** (n ≥ 192) |

Scaling (uniform routing = grouping's worst case): v3 is 0.91× at n=128,
1.51× at 256, 2.31× at 512, 2.69× at 1024 — decode amortization grows
with tokens-per-expert, exactly the mul_mm_id curve. Grouping below
n≈192 loses; hence the threshold.

**Correctness chain** (`qk moegcmp` = 4-way GPU-vs-GPU h diff on one
layer, random x + uniform distinct routing):
- v1 bitwise: 0/2359296 entries differ at n=512.
- v2/v4 vs ungrouped: max_rel ~4e-3 AT NEAR-ZERO h entries (abs ~4e-7)
  — f32 summation-order class. With scalar projections the whole model
  is **prefillcmp 36/36, worst rel 8e-6 (v2) / 7.4e-4 (v4)**: grouped
  semantics are exact; h-GEMM noise is the only f16 in the pipeline.
- v3 f16: same class as the shipping h-GEMM (max_rel at cancellation
  zeros, abs ~2e-4).
- Under the default h-GEMM, forced-grouped prefillcmp reads 35/36 (v2,
  v4): the flipped cell (N=48 seed=42, argmax margin 0.042) is a
  random-token near-tie where baseline noise 0.073 already dominates —
  ANY reordering rolls that die. The gate for grouped work is therefore
  the compound: moegcmp isolation + scalar-proj 36/36 + serving parity.
- Serving parity, forced v4 at all n: ids1/2/3 TOKEN-EXACT vs llama.cpp
  refs, prefilldecode HANDOFF EXACT.
- **Long-prompt gate (the one that exercises default grouping): 1040
  natural tokens (PORT.md head via llama /tokenize), QK_MAXB=512 → two
  full 512-token grouped chunks + 15 serial; 64 greedy tokens
  TOKEN-EXACT vs llama-server temperature-0 for ungrouped, default-v4
  AND v3-f16.** (v3 does flip 1 token in 244 on the SMALL-prompt suite
  — prompt3 token 43 — recorded; that's why v3 stays opt-in.)

**Engine numbers at N=512 single chunk (QK_MAXB=512), interleaved with
serial control, lap-thermal so ratios are the hard data:**

| config | batch ms | speedup vs own serial | tok/s |
|---|---|---|---|
| ungrouped | 2239–2388 | 4.32–4.69× | 214–229 |
| v4 default | 1567–1726 | **5.68–5.69×** | 297–327 |
| v3 record | 1438–1661 | **6.20–6.51×** | 309–356 |

pp512 standing after gate+up only: ~310–356 tok/s vs llama.cpp 1452.
Contamination note: one bench triple was invalidated by a leftover
llama-server holding 17 GB (kill %1 in a fresh shell is a no-op — check
`ps`, not job tables); numbers above are from clean runs.

### Phase B round 3 — grouped DOWN projection (2026-07-09)

Same decode-once structure for the down mats (`moe_down_grouped.metal`):
one threadgroup per (expert, 32-row tile of n_embd), K = n_ff = 512,
f32 staging + f32 MMA, both routed formats (IQ4_XS ×37 layers, Q6_K
×34/38/39) plus the shared Q8_0 expert. Down accumulates ACROSS experts
per token, so grouped threadgroups write unweighted per-slot results to
a dY[tok][slot][2048] scratch (bbMDy) and `moe_down_reduce` folds the 9
slots with the routing weights — two extra dispatches + one barrier per
layer, noise. Engages together with grouped gate+up (same n ≥ 192
default threshold; QK_MOE_GROUPED forces).

Gates, all green in one pass:
- serve-test ids1/2/3 forced-v4: TOKEN-EXACT; prefilldecode HANDOFF EXACT.
- scalar-proj + fully-grouped prefillcmp: **36/36 @ 1.4e-6** (tighter
  than gu-only — order noise, coin-toss direction).
- default h-GEMM + forced-v4: **36/36 @ 0.075** — the N=48 near-tie that
  flipped in round 2 landed back on the matching side; those cells are
  dice at the noise floor, and the full config currently rolls 36/36.
- Long prompt (1040 tok, two 512 grouped chunks): TOKEN-EXACT vs
  llama.cpp for default-v4 AND v3-f16.

N=512 interleaved (serial control wobbled ±18% — lap thermals; ratios
are the hard data):

| config | batch ms | speedup vs own serial | tok/s |
|---|---|---|---|
| ungrouped | 2050 | 4.28× | 250 |
| v4 + grouped down (DEFAULT) | 1015–1424 | **6.57–6.74×** | 359–504 |
| v3 + grouped down (record) | 1198 | **7.51×** | 427 |

pp512 standing: **~427 tok/s conservative (v3), 504 best-observed (v4,
coolest run) vs llama.cpp 1452 — gap ~2.9–3.4×**, from 6.3× at the
start of Phase B round 2. Next: stage isolation at N=512 to re-rank the
remaining fat (attention O(N²), dn chain, select/logits, projections),
retune QK_MOE_GROUP_N now that down is grouped too, f16 down variant
for the v3 record config, and the record run on a hard surface.

### Phase B round 3.5 — f16 down + vectorized staging (2026-07-09)

- **f16 grouped down** (`moe_down_grouped_h_*`, 64×32 tiles, 16 elems
  per staging thread — IQ4_XS nibble planes and Q6_K 16-elem scale
  groups split naturally). QK_MOE_GROUPED=3 is now f16 end-to-end in
  the MoE.
- **Stage isolation at N=512** (v4 config, 1003 ms chunk): gu 406 ms,
  down+reduce 196, projections 176, dn 104, **attention 16 (the O(N²)
  worry was wrong)**, head/logits/norms ~105. MoE was still 60% — hence
  the two moves above.
- **Vectorized threadgroup staging**: the dequant stages wrote 32 scalar
  tg stores per thread per chunk; half4/float4 stores (row strides
  66/65 → 68 for alignment) are bit-identical (moegcmp max_rel
  unchanged to the digit) and large: v3 gu 6.06 → **4.79 ms/layer**
  (−21%), v4 gu 9.57 → **7.21** (−25%).
- moegcmp per-layer at n=512 uniform (worst case): ungrouped 13.9 | v4
  7.21 (f32-exact) | v3 4.79 (f16).
- Crossovers with down grouped (interleaved, stable controls): v4 still
  ties at N=128 / loses at 96 → threshold 192 stands. **v3 wins already
  at N=96** (214 vs 251 ms) and by 21% at 128 — its crossover is below
  the default serving chunk, relevant if the record config ever becomes
  default.
- Gates re-run over the vectorized kernels: forced-v4 serve-test
  TOKEN-EXACT, prefilldecode HANDOFF EXACT, long-prompt (2×512 grouped
  chunks) TOKEN-EXACT for default AND v3.
- N=512 cool-state: **v3 775.7 ms → 660 tok/s pp512**. Same-sequence
  later runs decay to 1230 ms purely thermally (lap). llama.cpp 1452:
  **gap now ~2.2× cool**.

Cross-box note: wire bumped to qkp2 mid-round (tron) — pipe harness
re-mirrored from main, qk_state_n/save/load implemented over the pcache
snap buffers, gates (a)/(b) token-exact incl. reconnect, live worker on
:18100 relaunched on the new build.

### B3 — multi-slot aggregate: beats llama-server --parallel at every N (2026-07-09)

| concurrent streams | llama-server --parallel (tok/s agg) | qk serve-test (tok/s agg) | qk / llama |
|---|---|---|---|
| 1 | 80.1 | 96.8 | 1.21× |
| 2 | 114.5 | 149.8 | **1.31×** |
| 4 | 135.2 | 182.1 | **1.35×** |
| 8 | 149.8 | 195.7 | **1.31×** |

Protocol: same 21-token prompt, greedy, N simultaneous streams; llama =
N concurrent /completion requests (temperature 0, warmed server, box
solo); qk = `serve-test <ids> 256 <N>` with QK_MLOCK=1, box solo.
Scaling 1→8: qk 2.02×, llama 1.87×. Both sides are weight-read-bound at
8 slots; our decode re-reads weights per slot (GEMV z=slots) — a
slot-batched decode GEMV (read a weight row once, dot 8 slot
activations) is the named lever for the next tier (~1.5× aggregate
ceiling at 8 slots by active-bytes math). Steady-state single-stream
serving decode: 9.1 ms/step GPU ≈ 110 tok/s (engine) vs caseToken's
8.37 — the 0.75 ms is fa_srv slot indirection + argmax; llama tg128 is
84.2.

**Operational pathology found on the way (cost half a day of confusing
numbers): no-copy mmap weights degrade 2–6× after ANY memory-pressure
event** (here: llama-server's 17 GB residency for the head-to-head,
plus an 8-slot ctx-16384 llama config pushing swap to 6.3 GB). Evicted
GGUF pages make every subsequent submit re-wire GPU mappings — 0.04 s
user / 1.87 s sys per 4 s of serving, steps 9 ms → 20–50 ms — and it
does NOT self-heal even at 84% free RAM (page cache warm ≠ GPU wired).
caseToken was immune (it copies weights into device buffers), which is
how the "regression" was isolated to the buffer policy. Fixes, both
landed: `QK_MLOCK=1` wires the mapping (zero-copy preserved — the
serving config; the :18100 worker now runs with it) and
`QK_COPY_WEIGHTS=1` copies into a device-owned buffer (RSS +16.6 GB,
llama.cpp-equivalent policy). Default stays plain no-copy mmap (C1's
318 MB RSS) — same trade llama.cpp makes without --mlock. House rule:
serving/benchmark configs set QK_MLOCK=1; any anomalous slowdown gets
`/usr/bin/time -l` first (high sys = rewiring signature).
`QK_STEP_STATS=1` prints per-step gpu/encode/wait breakdown from
stepChunk for exactly this kind of triage.

## C1 — no-copy weights: 15.5 GB RSS → 318 MB (2026-07-09)

The engine now wraps the entire GGUF mmap in ONE
`newBufferWithBytesNoCopy` MTLBuffer (page-aligned by mmap; length rounded
up to the page). Every weight binding became a (buffer, offset) pair via a
`WB` wrapper that converts implicitly from a plain buffer, so activation
bindings didn't change textually; per-tensor offsets come straight from
the GGUF parse (32-byte aligned, satisfying setBuffer's float4 needs).
State/activation buffers stay allocated.

Measured (serve-test process): **maximum RSS 318 MB, peak footprint
508 MB** — weights stream from file-backed pages on demand, zero copies at
open, no double residency. Output token-exact; prefilldecode HANDOFF
EXACT and prefillcmp 36/36 re-verified after the binding surgery. This is
now BETTER than llama.cpp's mmap path (which still faults the whole model
into its own dirty pages for GPU residency on load).

## D — cross-box pipeline split on Metal (tron brief, 2026-07-10)

Merged origin/main 0c508e2 (clean; the in-messages system-turn patch had
converged with upstream d39168a). Metal engine implements the frozen ABI:
`QK_LAYERS=a:b` partial load (layer states only in range; logits/argmax
buffers only on the last stage; weights are free either way — no-copy),
`qk_stage_run` chunked over maxB with hidden-in → first-layer rms entry
and hidden-out = raw residual rows via UMA memcpy; the boundary layer
binds a zeroed dummy next-norm (dead xn), exactly the Vulkan trick.
`qk pipe`/`qk pipe-worker` compiled from main.cpp verbatim.

Gates (brief §4), all token-exact vs the unsplit engine AND the llama.cpp
greedy references:

| gate | result |
|---|---|
| a. in-process split 20 (deltanet boundary) | GEN exact |
| a. in-process split 24 (attention-layer boundary) | GEN exact |
| b. TCP worker localhost, split 20 | GEN exact |
| b. reconnect, second prompt | GEN exact |

Timing (localhost, warm): 9.55 ms/tok split vs 8.3 unsplit → split
overhead ≈ 1.2 ms/tok including loopback TCP of the 8 KB hidden row.
In-process two-engine mode is slower (52–76 ms/tok — two Metal contexts
contending in one process); use the worker for real numbers. For tron
(gate c): launch the worker here with slots/ctx covering the head, e.g.
`QK_LAYERS=22:40 QK_GGUF=... build/qk pipe-worker <port> 22:40 2 32768`
— s2 on midnight measured ~5.3 ms/tok at split 20 with 4k ctx; balance S
per the brief using tron's ~5.8 ms/tok full-model decode.

**Gate (c) PASSED (tron-driven, 2026-07-10): midnight served tron's
qk-server over the LAN token-exact at pipe level and byte-identical
through the full `--split-next` HTTP path. 17.2 ms/tok engine-level
(midnight stage ~5 ms + WiFi RTT ~4.3 ms), 50 tok/s streamed. Two GPU
vendors — RDNA3 head, Apple Metal worker — one bit-exact model.**
Follow-ons (wired link, S sweep, slots=4 worker) tracked on the tron
side; the midnight worker stays up on :18100.

**Arc complete (tron final, 2026-07-09): clean-sweep S sweep with this
box idle — S=22 optimal; single-stream 17.0 ms/tok, two-stream
aggregate 12.5 ms/tok (80 tok/s), token-exact throughout.** The wire
moved to qkp2 (op3/op4 state ops) mid-arc and this side re-gated on it
same-day. tron's sweep attributes ~7 ms of layer-insensitive per-frame
cost to this worker; local repro of the S=33 shape (raw pipe client,
ctx 32768) finds NO fixed cost at steady state — gpu 2.3 ms + 0.15 ms
submit, s2+net 2.61 ms/tok — but a first-~32-frames warmup regime
(9.4 ms wall vs 3.1 gpu: lazy GPU page-table mappings on the mmap
weights; mlock wires CPU pages only) and an op3-per-frame candidate
(~134 MB KV stripes per attn layer at 32k ctx ≈ 9 ms) that would both
read as a fixed tax. Probes for tron's side are in CROSSBOX-BRIEF.md;
QK_STAGE_STATS=1 on the worker prints the gpu/wall split directly.
Deploy templates (qk-server-split.yaml, midnight-qk-worker.plist)
adopted into the tree from main; the worker keeps nohup+caffeinate for
now. Remaining cross-box items are external: wired ethernet, and the
per-frame question above.

One false alarm worth recording: an apparent engine-vs-reference
divergence on prompt 3 was a stale memory of the PRE-fix (accidentally
sampled) reference — the regenerated greedy ref matches the engine
exactly. Trust files, not recollection.

### Method notes (M0)

- llama.cpp built in-tree at `../llama.cpp` (fresh clone of master, same-day);
  `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON`. Note: on this master,
  `llama-cli` requires `LLAMA_BUILD_SERVER=ON` (cli links `llama-server-impl`).
- The 546 GB/s denominator is the spec sheet; 494 GB/s is what a pure
  streaming-read kernel actually sustains (90.5%). Extraction percentages
  against 546 are therefore conservative lower bounds.
- DeltaNet state bytes assume f32 state resident on GPU, read+write per
  token per linear layer; conv-state traffic (~0.1 MB/layer) ignored.

## Execution plan — beat llama.cpp Metal on EVERY axis (2026-07-09)

Scorecard today (this box, this GGUF) and the plan to flip every ✗:

| axis | llama.cpp | qk | plan |
|---|---|---|---|
| decode 1-stream | 84.2 tok/s | **120.5 ✓** (parity ✓) | stretch to 150 (C3) |
| prefill batched | 1452 tok/s | **318** (argmax-exact, 3.85× serial) | B: grouped-MoE GEMM next |
| serving/multi-slot | llama-server | **✓ full stack + CLI round trip** | done (M5) |
| agg. throughput | --parallel N | **146.3 tok/s @ 8 slots** (1.74× their 1-stream) | B3 head-to-head pending |
| long ctx | 256k | **✓ 32k engine / 29.8k proven live** | C2 f16 KV for headroom |
| load time / RSS | mmap no-copy | **✓ 318 MB RSS, zero-copy open** | done (C1) |
| quality | reference | ✓ token-exact everywhere | parity gate on every change |

**Phase A — serving foundation (M5):**
A1 port batch/srv kernels: gemm_q8_0, fa_prep/attn_batch, fa_prep/attn_srv,
   dn_*_batch, moe_down_q8b, add_rms3 (batched-prefill + multi-slot chains).
A2 qk.h C ABI engine in main_metal.mm; build libqk.dylib (APPLE branch).
A3 harnesses green: prefillcmp, prefillbench, prefilldecode, serve-test,
   serve-test2, cachetest — all token-exact.
A4 prefix cache snapshot/restore = plain UMA memcpys.
A5 Rust server dlopens libqk.dylib; /v1/messages up; claude CLI round trip.

**Phase B — prefill/throughput supremacy:**
B1 batched prefill ≥4× own serial (DoD bar) via chunked GEMM chain.
B2 profile → simdgroup_matrix GEMM for q8/iq3/iq4 to ≥1452 tok/s pp512
   (llama.cpp ≈ 31% of f16 peak here; beatable with fused-MoE GEMM).
B3 multi-slot aggregate decode > llama.cpp --parallel at 2/4/8/16 slots.

**Phase C — polish past llama.cpp:**
C1 no-copy weights: ONE MTLBuffer over the whole GGUF mmap
   (newBufferWithBytesNoCopy needs page alignment — the mmap base is;
   per-tensor via setBuffer:offset:, GGUF aligns tensors to 32 B ✓)
   → load ≈ llama.cpp, peak RSS −15 GB.
C2 long context: fa srv kernels (chunked softmax past the 1024 cap),
   f16 KV cache (llama.cpp default — keeps comparisons apples-to-apples),
   ctx 16384 serving config like the RDNA3 deploy.
C3 decode 150 stretch: post-barrier bandwidth-ramp microbench, nr0
   row-pairing on in-block q8 GEMVs, then spec-decode per
   docs/spec-decode-qk-plan.md as the step change.
C4 final scorecard in README + PORT.md: decode, prefill, aggregate, TTFT,
   load, RSS, CLI-path E2E — every row green vs llama.cpp Metal.

Rules of engagement (unchanged): every kernel lands with CPU-reference
validation; every phase ends with the token-parity gate ×3 prompts; every
result (positive or negative) recorded here; commit per green step.

## 80B standalone arc (2026-07-13 night): residency root cause, grouped-MoE fix, llama.cpp beaten on decode

Context: llama-bench (this box, worker stopped) runs the whole 80B at
68.8-70.6 tok/s decode / ~1017 tok/s pp512 — beating the tron+midnight
split (48.7 tok/s e2e, network exonerated by the ethernet A/B). Our
standalone had emitted garbage (token 15 repeat, 7.6 tok/s), and the box
kernel-panicked at 21:07 mid-investigation.

Findings (each verified tonight, clean box, same repacked GGUF):
1. The panic is watchdogd starvation — the 42.9GB standalone run paging
   against the 32GB-wired worker — not a Metal driver bug. The garbage
   runs left 8 GPU-restart reports (gpuEvent-qk-*-112552.ips, "BIF0 page
   fault", read) hours before the panic.
2. Window COUNT was a red herring: maxBufferLength is 38.88 GiB on this
   M4 Max, so llama.cpp maps the same file as 2 overlapping views
   (ggml_metal_buffer_map). The real difference is GPU residency:
   llama.cpp wraps every buffer in an MTLResidencySet with
   requestResidency + heartbeat; we relied on lazy per-submit wiring
   (QK_MLOCK wires CPU pages only — the M0 warmup note foreshadowed
   this). Under pressure: rewire degradation, then ABORTED command
   buffers → stale logits → argmax constant token.
3. Fix ed53ed1: MTLResidencySet over the weight mapping at qk_open
   (commit + requestResidency + attach to queue; QK_NO_RSET=1 A/B).
   5e1e0b8 scopes it to full-model engines — split stages keep
   stage-scoped mlock + lazy wiring by design.
4. Grouped-MoE batched prefill was CORRUPT on the 80B: moe_group wrote
   the shared-expert aSlot bucket at hardcoded start[256] (n_expert=256
   on the 35B masked it; 512 on the 80B put it mid-table). prefillcmp
   exploded exactly at the moeGroupN=192 crossover (max|dlogit| to 26).
   Fix cd119d2. The split-prod worker never saw it only because the
   grouped gates (ids4/moegcmp) are 35B-shaped — harness hole below.

Scorecard (this box, clean, identical GGUF):

| axis | llama.cpp Metal | qk standalone |
|---|---|---|
| decode short-ctx | 70.62 tok/s (tg128, 14.16 ms) | **92.5 tok/s (10.81 ms wall / 10.6 GPU)** |
| decode @ ctx~2100 | — | ~78 tok/s |
| prefill N=512 | 1017 tok/s | 550 tok/s grouped-fixed (282 ungrouped) |
| greedy parity | reference | 39-tok prompt ->128: 126/128 exact, 2 near-ties CERTIFIED (llama top-2 gaps 0.054/0.022, qk picks llama #2); 2103-tok prompt ->64: **64/64 EXACT** |

Where llama.cpp's 14.16 ms/token goes: 2047 dispatched Metal ops per
decode graph (GGML_METAL_GRAPH_DEBUG histogram: 389 MUL_MAT + 131
MUL_MAT_ID doing the work; ~1500 small ops — 139 CONT/CPY data
movement, 109 GET_ROWS, 64 L2_NORM, 64 REPEAT, 44 each
SUM_ROWS/CLAMP/DIV — the ggml DeltaNet decomposition). Ours: one
command buffer per token of fused chains; measured budget MoE 5.0 ms
(48 x 104 µs, 232 GB/s over 23.1 MiB active/layer), attention 2.3 ms
(12 x 191 µs at pos 0), DN ~2.3 ms (36 layers, by subtraction), head
0.51 ms (503 GB/s) — reconciles with the 10.1 ms step GPU time.

Open items:
- Prefill gap 550 vs 1017 tok/s: grouped-MoE GEMM is the lever (B-phase
  class work). A benign f16-class "mild regime" exists for N>32 in
  prefillcmp (worst rel 0.044, one N=96 near-tie argmax flip;
  insensitive to QK_GEMM=scalar / QK_DN_CHUNK=0 / QK_MOE_GROUPED=0 —
  kernel-variant rounding, llama.cpp's own prefill precision class).
- 80B batch harness hole: caseBlock needs split ssm_alpha/beta (80B
  fuses ssm_ba), moegcmp is IQ3-shaped. Port both so the next grouped
  bug cannot hide behind the 35B gates.
- Serving decision (tron): standalone on this box beats llama.cpp
  decode by 31% and the split by 90%, with the full server stack
  (prefix reuse, snapshots, sampling, tool-retry) intact. Worker
  relaunched on :18200 (fixed binary, same 12:48 config) so prod runs
  the split meanwhile; the single-box switch is yours to call.

Repro anchors: QK_GGUF=<80B-qk.gguf> ./build/qk serve-test /tmp/big_a.ids 128 1 2048
(certify vs llama-server /completion temperature 0, n_probs 3, near-tie
bar 0.15 logprob); prefillcmp 512 2048 with QK_MAXB=512; llama-bench
reproduced 70.62/1017 same-night same-box.

### 80B prefill counterattack (2026-07-13 late): **1029 tok/s — llama.cpp beaten**

Profile first: the repaired v4 default at N=512 was 939.3 ms / 545
tok/s under the timestamp trace. `moe-gateup` was 542.4 ms and
`moe-down` 197.8 ms: **79% of the whole pass was grouped MoE**. The
existing packed-f16 v5 immediately cut those to ~202/107 ms, but the
80B IQ4 path could not use `moe_group_work`; it still launched the dense
513-expert x token-tile grid and returned from empty pairs.

The landed path adds the IQ4_XS twin of the compact `(expert,t0)` work
consumer and makes v5 the default only for 512-expert IQ4 models at the
unchanged N>=192 crossover. The raw all-f16 form traced at 493.5 ms /
1038 tok/s, but it GREW the prefillcmp envelope to rel 0.069 at N=256 —
red, not promoted. The shared Q8_0 expert was the numerical amplifier:
every token traverses it in every layer. Keeping only shared gate+up in
the existing v4 f32 MMA class brought the exact same cell to rel 0.007
and the full sweep back inside the prior envelope. Making shared down
f32 too was rejected at 978 traced tok/s (down 99 -> 116 ms); packed-f16
down stays.

Final clean full-curve record (`f16` probe 515.7 GB/s immediately
before it):

| N | old/default control | new default | result |
|---:|---:|---:|---|
| 128 | same ungrouped path | **281 tok/s** | unchanged crossover regime |
| 256 | 453 tok/s (forced v4) | **823 tok/s** | 1.82x |
| 512 | 545-550 tok/s | **1029 tok/s** (497.49 ms) | **1.87x; beats llama.cpp 1017.5** |

An isolated cool N=512 run reached 495.33 ms / 1034 tok/s. One full
sweep immediately after the long decode gate sagged to 853 purely from
soak (every size was ~22% down); it is excluded, per the established
thermal rule. The final traced budget is ~203 ms gate+up, 99 ms down,
80 ms recurrent projections, 27 ms output projections, 19 ms attention
projections, and ~70 ms everything else. MoE remains 61% of the pass,
but the absolute target is green.

One pre-existing 35B harness hole surfaced while applying tonight's
stated gates: the untouched 21:01 `build-perf/qk` baseline and the new
binary both produced the same N=48 seed=42 flip (44/45, rel 0.10). N=48
is the first tiled Q8 GEMM shape. Selecting its existing tiled f32 twin
only below N=64 restores **45/45** with f32-level N=48 error (~2e-6),
without touching N=128/256 or the 80B IQ4 projection path.

Final gates, default config:

- 35B `serve-test ids3 44`: reference prefix exact; prefillcmp **45/45**.
- 80B short certified stream: **128/128 file-exact**.
- 80B 2103-token grouped prompt: **64/64 file-exact**.
- 80B prefillcmp: accepted **44/45**, only N=96 seed=1234; every N>=192
  MATCH; worst rel **0.042** (was ~0.044).
- Decode guard: 512-step average GPU **10.55 ms** (10.76 ms submit+wait),
  no regression.

Rollback/debug remains explicit: `QK_MOE_GROUPED=4` restores the old
80B f32 grouped kernels; `QK_MOE_WORK=0` restores v5's dense z-grid.
