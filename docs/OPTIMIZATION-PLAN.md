# Metal optimization plan — M4 Max

Date: 2026-07-12  
Branch: `metal-port`  
Target: Qwen3.6-35B-A3B on the 40-core M4 Max GPU; preserve the Qwen3-Next-80B
split-worker path unless a lever is explicitly 35B-only.

## Objective and operating rules

Chase every credible inference optimization until it either lands with a
measured win and the required correctness gates, or reaches a documented hard
wall. Rank is by expected end-to-end value, not implementation ease. Large
programs are split into bisectable sub-levers; each accepted sub-lever gets one
commit and its result is recorded in this file.

- Build and run only `build-perf/qk`. Never build or execute `build/qk` during
  this campaign.
- Before GPU benchmarking, stop tmux session `qk80-worker` with `C-c` and write
  `DOWN` to `/tmp/qk80-worker-status.txt`. Before a long pause or at completion,
  relaunch the exact production command from the mandate with `QK_PCACHE=6`,
  then write `UP`.
- Use the 35B GGUF with `QK_MLOCK=1`. Check `f16 8192 8192 100` before and
  between comparisons; approximately 500 GB/s is a usable cool-state probe.
- Performance decisions use interleaved A/B/A/B (or A/B/A) with the bandwidth
  probe and a control column. Reject movements inside thermal/noise bounds.
- New behavior is opt-in until its gates and rollback path are proven. Do not
  commit a failed experiment; revert its implementation and record the result
  here.
- Every accepted implementation commit ends with the required
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.

There are pre-existing uncommitted prototypes in `gemv_q6_k.metal` and
`gemv_q8_0.metal`. They are treated as user-owned: preserve them, establish
their correctness/performance independently, and do not silently absorb them
into an unrelated commit.

## Baseline and constraints from `PORT.md`

- One-stream decode: about 8.35 ms/token (120 tok/s). The command submission
  component is only about 2%; the command buffer streams the 2.457 GB active
  weight set at about 276 GB/s, 56% of measured bandwidth. Hundreds of dependent
  stages repeatedly ramp DRAM after barriers. Prior stage fusion moved less
  than 2%.
- pp512 record configuration: about 1400 tok/s with
  `QK_MAXB=512 QK_MOE_GROUPED=5 QK_PB_ONLY=512 QK_PB_NOSERIAL=1`; MMA attention
  has measured about 1468 tok/s in a clean/cool run versus llama.cpp about 1452.
- Deep-context decode: `QK_FA_SPLIT=1` is about 1.36x at context 8000. Its
  coalesced inner-loop rewrite and chunk-size sweep did not improve it.
- Aggregate decode plateaus near eight slots because dense and expert weights
  are dispatched independently for each slot. The >8-slot DeltaNet race was
  fixed by commit `8f232ff`; eight-slot determinism is the mandatory gate.
- DeltaNet chunk chain at N=512 is roughly 0.94 ms/layer after the compact
  triangular scratch win. Scalar solve is about 0.25 ms; the step remains far
  above its flop model.
- Metal does not yet contain the Vulkan engine's `QK_SPEC`, `verify`, or
  `speccmp` implementation. Metal does already have batched per-position head +
  argmax in `prefillBatchLast`, so it has a verifier starting point.

## Correctness and benchmark gates

Gate abbreviations used below:

| Gate | Required check |
|---|---|
| G0 | Clean `cmake --build build-perf -j`; no changes to `build/`. |
| G1 | Relevant isolated kernel harness, GPU reference comparison, bounds/tail sizes, and both 35B/80B quant variants when applicable. |
| G2 | `QK_GEMM=scalar ./build-perf/qk prefillcmp 128 2048`: 36/36 token-exact, worst relative error near 1e-6. |
| G3 | `./build-perf/qk dncmp 512 30` all PASS and `./build-perf/qk block 0 3 200` PASS. |
| G4 | `./build-perf/qk ablock 3 3 200` PASS. |
| G5 | `./build-perf/qk token tests/ids3.txt 20` starts with the mandated 20-token sequence. |
| G6 | `./build-perf/qk serve-test tests/ids3.txt 200 8 512`: all slots identical YES. Run 2/4/8-slot A/B throughput for aggregate changes; run 16 slots as a stress gate when allocations permit. |
| G7 | `./build-perf/qk prefilldecode 128 24 2048`: HANDOFF EXACT. |
| G8 | Speculation: oracle full-accept exact; forced rejection at every position; long rejection streak; serial-versus-spec generated stream exact; coherent/self-consistent sampling mode. |
| G9 | Long context: scalar-versus-split generated tokens exact at contexts 1040 and 8000, including heterogeneous slot lengths. |
| G10 | Cache/state: `cachetest`, `serve-test2`, prefix restore/fork, live-length save/restore, memory bounds, and split-stage smoke as relevant. |

Unless a row explicitly says otherwise, a default-path change must pass
G0-G7. A numerically lossy storage/compute experiment is never silently
promoted: its gate is serial-versus-batched/self-consistent argmax plus coherent
long output, and the precision tradeoff remains opt-in and documented.

## Ranked optimization queue

EV bands: **very high** is a plausible step change (20%+ on an important
axis), **high** is 5-20%, **medium** is 1-5%, and **low** is sub-1% or a
memory/startup/edge-workload benefit. Percentages are hypotheses until an A/B
measurement closes the row.

### 1. Metal speculative decoding program — very high EV

**Expected value:** The only credible 1-stream step change: roughly 1.2-2x on
repetitive code/tool-output traffic, with an oracle full-accept ceiling to be
measured on Metal. Novel prose should remain at baseline through a no-trigger
gate. It can also amortize deep-context attention and the 600-stage token walk.

**Approach:** Port the already-shipped Vulkan design in small commits. (a) Add
`verify` and `speccmp` harnesses around Metal `prefillBatchLast`, and measure
K={2,4,8,16,32,64} before integrating policy. Keep target layers, all-position
head, and argmax in the same command buffer. (b) Allocate one recurrent scratch
stripe; copy live DeltaNet/conv state into it, bind the batch verifier to
scratch, promote scratch on full accept, and replay the accepted prefix into
live state on partial accept. KV beyond the accepted position is ignored and
overwritten. (c) Port prompt-lookup/n-gram drafting, queued emission, stats,
adaptive K/L, and the single-active-slot safety rule under `QK_SPEC=1`.
(d) Revisit multi-slot fairness only after single-slot economics are positive.
Use ranks 2-4 below to reduce verifier cost, especially the N-row Q6_K head.

**Correctness gate:** G0, G2-G8, cache restore/fork, EOS inside a queued round,
context-boundary tails, K tails not divisible by the kernel tile, and at least
100-token comparisons in corruption modes {never, every draft, each interior
position, consecutive rejection}. Default-off until exact.

**Risk:** High. Recurrent rollback mistakes can silently corrupt later tokens;
the current N-row head rereads 417 MB per row and may make short verification a
net loss; n-gram acceptance is workload-dependent. A missing compatible draft
model/MTP head is a hard wall for model-based drafting, not for prompt lookup.

### 2. Weight-stationary 2/4/8-slot dense projections and head — very high EV

**Expected value:** About 1.2-1.5x eight-slot aggregate and a large reduction in
small-N speculative verify cost. Two-slot production should also benefit,
though less. Dense weights plus the 417 MB head are most of the reusable bytes.

**Approach:** Specialize Q8_0 and Q6_K kernels for B={2,4,8}: decode each quant
block once, preserve each RHS's existing accumulation order, and dot all active
slot vectors before eviction. Wire exact-count pipelines into qkv/z/out,
q/k/v/o, and head dispatches; retain the existing z-batched fallback for other
counts and sparse layouts. Generalize to IQ4_XS for the 80B projection set only
after the 35B path wins. Add a small-N Q6_K GEMM/weight-stationary path for
spec-verification N that is not exactly 2/4/8.

**Correctness gate:** G0-G7; isolated B=2/4/8 random and real-weight comparisons;
heterogeneous inputs (not only identical slots); output tails; forced fallback;
80B `ablock`/serve smoke for IQ4 wiring. Benchmark aggregate 1/2/4/8 and verifier
K=2/4/8 separately.

**Risk:** High register pressure can erase weight reuse, and array-indexed
accumulators may spill. Exact slot count and active-slot layout complicate host
dispatch. Existing uncommitted B=8 shader prototypes require independent
validation and may be replaced only deliberately.

### 3. Slot-batched MoE, shared expert, and router — very high EV

**Expected value:** Required to turn dense slot batching into a real aggregate
win; target is the remaining path to about 1.5x at eight slots. Shared-expert
weights reuse perfectly. Routed-expert reuse depends on cross-slot routing but
identical/related traffic has large reuse and arbitrary traffic still benefits
from grouping the union once.

**Approach:** After `moe_select`, flatten up to B*(top-k+shared) assignments,
group them by expert, and run weight-stationary gate/up and down over each
expert's assigned slots. Reuse/adapt the prefill assignment arrays, but design a
small-B kernel rather than invoking tiles tuned for N>=192. Always give the
shared Q8_0 expert a B-wide kernel. Batch the F32 router and preserve per-slot
top-k/tie order. Compare direct rank-wise B kernels with compact expert-union
dispatch; choose dynamically by collision count if both regimes matter.

**Correctness gate:** G0-G7 plus randomized heterogeneous SelT routing, shared
and routed outputs against the current per-slot path, all supported routed
formats (IQ3_XXS/IQ4_XS gate-up; IQ4_XS/Q6_K down), and staggered `serve-test2`.

**Risk:** High. Uniform independent routes have few collisions at B=8, so
sorting/dispatch overhead can exceed reuse. Down reduction must retain routing
weights and slot ownership exactly. Register-heavy inline selection already
lost 39%, so selection remains a separate compact stage.

### 4. Grouped-MoE token-tile persistence / indirect dispatch — high EV

**Expected value:** Plausible 5-15% pp512. V5 dispatches z=ceil(N/32) for every
expert/row tile; natural N=512 routing averages about 16 assignments/expert, so
most z tiles return empty, while the shared expert rereads its weight tile up to
16 times.

**Approach:** Build two variants. First, one TG per (expert,row tile) loads the
weight tile once and loops over all 32-token assignment tiles, resetting/storing
accumulators per tile. Second, let `moe_group` write max-count/indirect dispatch
arguments so only live z tiles launch. Apply to packed gate/up and packed down,
with a dedicated dense shared-expert GEMM if that wins. Keep the current V5 as
control and retain enough TG parallelism for pathological routing concentrated
on one expert.

**Correctness gate:** G0-G7, `moegcmp` at N={48,96,128,192,512,1024}, natural
and all-to-one/uniform routing, tail counts 1/31/32/33/511/512, and pp512
stage-isolated plus end-to-end A/B.

**Risk:** Medium-high. Serial token tiles can reduce parallelism for hot
experts; indirect-dispatch visibility/order on a concurrent Metal encoder must
be explicit; shared/routed quant layouts differ.

### 5. DeltaNet chunk-step resident-state retile — high EV

**Expected value:** Plausible 3-8% pp512 and the largest remaining DeltaNet
opportunity. The current step is about 3.7x its flop model and repeatedly uses
device state between 64-token chunks.

**Approach:** Retile state rows and sweep 8/16/32-row panels with 2/4/8/16
simdgroups. Load each panel once, run all chunks while it remains resident, and
write once; remove device barriers/reloads inside the chunk loop. Sweep TG and
packed/transposed MMA layouts. If residency loses occupancy, retain a narrower
streamed variant and only minimize barriers/panel replication.

**Correctness gate:** G0-G3, G5-G7; dncmp Tn={1,5,63,64,65,127,128,200,511,512}
and kDiv={0,2}; state error bounds no worse than current; isolated step/chain
timings and pp512 A/B.

**Risk:** High. The 32 KB TG limit is tight; bank conflicts and reduced
residency can overwhelm saved state traffic. Reordered f32 MMA may enlarge state
drift enough to flip a token.

### 6. DeltaNet triangular solve: Neumann/doubling and SIMD solve — high EV

**Expected value:** Up to 0.15-0.20 ms/layer in the optimistic case, roughly
1-4% pp512. Current compact scalar solve is about 0.25 ms/layer.

**Approach:** For each 64x64 strictly-lower M, evaluate an exact nilpotent
doubling product for `(I+M)^-1` with the correct alternating signs, using packed
8x8 MMA and applying the resulting transform to all RHS panels. Also evaluate a
SIMD forward-substitution wavefront that broadcasts solved rows without dynamic
private arrays. Count prep+inverse+RHS application, not solve alone. Reuse the
compact scalar solve as the hard control.

**Correctness gate:** Same exhaustive G3 matrix as rank 5, G2/G5/G7, compare
both transformed RHS and final recurrent state, and require an end-to-end win.

**Risk:** High. A prior blocked MMA forward solve lost badly because diagonal
barriers and f32 MMA overhead dominated; doubling adds matrix multiplications
and changes rounding. Stop if total chain cost is not lower, even if inverse
formation is fast.

### 7. Fuse grouped-down reduction with residual/RMS tail — high EV

**Expected value:** Plausible 2-6% pp512 by removing `bbMY` traffic, eight
down-reduce TGs per token, a barrier, and a separate one-TG residual/RMS pass.

**Approach:** One TG per token loops over 2048 dimensions, reduces the
top-k+shared expert rows with exact routing weights, adds the residual, performs
the RMS reduction, and writes raw + next-normalized outputs. Use vectorized
loads and compare 128/256/512-thread shapes. For decode, separately test a
last-TG/atomic completion fusion only if it does not put selection state in the
memory-bound expert kernels.

**Correctness gate:** G0-G7, grouped variants 3/4/5, both down quant formats,
N tails and random routing weights. Isolate down+reduce+tail and then pp512.

**Risk:** Medium. More work per TG can become latency-bound; f32 reduction order
may affect near ties; atomic completion can serialize or violate visibility.

### 8. Batched Q6_K head GEMM and fused top-1 reduction — high EV

**Expected value:** Major verifier improvement, about 5-10% decode ceiling from
the head only, and up to about 500 MB scratch reduction at maxB=512. Normal
decode improvement from fusing argmax is likely 1-3%.

**Approach:** Build a packed weight-stationary Q6_K MxN kernel for N>=2 and a
fused GEMV-head first-stage top-1 path for N=1. Do not materialize full logits
when only argmax/top-k is requested: emit per-TG candidates and reduce with
stable lower-id tie breaking. Preserve a full-logit path for sampling/top-k and
debug. Allocate `bbLogits` to actual need rather than maxB*vocab.

**Correctness gate:** G0-G7, every-row argmax against materialized logits for
random/real hidden vectors, deliberate ties, N tails, top-k split-stage checks,
and G8 once speculation exists.

**Risk:** High. Head GEMV already streams near 500 GB/s; larger per-TG work may
lower bandwidth. Any accumulation-order change can flip close logits. Sampling
requires values, not just ids.

#### 8d. Optional half-input/f32-accumulate verifier head — medium EV

**Expected value:** Another 2-5% oracle-verifier improvement after the exact
head work, with larger isolated wins as K grows. No one-token decode benefit.

**Approach:** Behind an explicit precision flag, round staged Q6 dequantized
weights and hidden activations to half while retaining f32 MMA accumulators.
Use a 64-row tile and B={8,16,32} column widths so one weight load serves the
whole verifier panel; repeat B32 rather than growing private accumulators past
the occupancy cliff. Fuse the same stable candidate reduction as the f32 path.

**Correctness gate:** Precision-class self-consistency: every-position batched
argmax against serial on the full prefill sweep and several natural prompts,
coherent top-k/materialized final logits, oracle and forced-rollback streams,
handoff/cache/multi-slot gates, and quantified logit drift.

**Risk:** Medium-high. Half input rounding can flip a close winner on an unseen
prompt even with f32 accumulation. It must remain opt-in and must not alter the
default exact f32 path.

### 9. Prefill flash-attention geometry and GQA reuse sweep — medium-high EV

**Expected value:** Attention is now about 13 ms of pp512; a 20-40% kernel win
is about 0.7-1.5% end-to-end, with larger value at longer prompts.

**Approach:** Parameterize/specialize QTM={8,16,32}, KBM={32,64,128},
NSG={4,8}, query/output ownership, and barrier placement. Test register-resident
O for QTM=8 and narrower O panels. Test processing 2-4 Q heads sharing one KV
head in a TG so K/V tiles are loaded once, within the 32 KB limit. Keep log2
softmax and causal block skipping, already measured neutral but correct.

**Correctness gate:** G0, G2, G4, G5-G7; base>0/multi-chunk exactness; N and
causal tails; isolated attention and pp512/pp2048 A/B.

**Risk:** Medium-high. The current kernel already won 56%; extra fragments can
spill or cut occupancy, and K/V traffic may not dominate.

### 10. Shape-specific dense Q8_0/IQ4_XS GEMM autotuning — medium EV

**Expected value:** 1-4% pp512. The packed Q8 kernel matches llama.cpp at its
main shape, but one global 64x32 geometry serves very different M/K projection
shapes.

**Approach:** Benchmark existing scalar/sg/h/h2/hp variants and new tile shapes
per real projection family (M=512/2048/4096/8192, K=2048/4096, N
48-1024). Select pipelines per shape/crossover at open; specialize function
constants so dead branches disappear. Repeat for IQ4_XS on an isolated 80B
tensor without stopping correctness at 35B.

**Correctness gate:** G0-G7, `bgemm` and real tensor comparisons, N tails,
stage-isolated projection time and pp512. 80B block/serve smoke for IQ4.

**Risk:** Medium. Microbench/SLC winners may lose in the full command buffer;
more pipeline variants increase startup and maintenance.

### 11. Quant GEMV/codegen squeeze for one-stream decode — medium EV

**Expected value:** 1-5% decode if real-command-buffer extraction improves.
Standalone Q8 is already near bandwidth ceiling; Q6_K and IQ4_XS trail
llama.cpp by about 5%, and in-command-buffer ramp/occupancy is the target.

**Approach:** Profile real shapes, then sweep TPR/NR0/NSG and function-constant
specialization; compare row pairing, vector loads, explicit unroll, packed
scale/codebook staging, software prefetch, and smaller/fatter TGs. Target qkv,
ssm_out, and head shapes separately. Evaluate a bundled independent-projection
dispatch only if concurrent dispatch is not already overlapping them.

**Correctness gate:** G0-G7, all quant synthetic/real suites, full token stream,
and A/B of full decode rather than loop-hot microbench alone.

**Risk:** Medium. Standalone bandwidth can be misleading because SLC and the
dependent layer chain dominate; register growth often reduces latency hiding.

### 12. Active-slot compaction and count-specialized scheduling — medium EV

**Expected value:** Large for staggered/sparse occupancy, near zero for the
all-eight-slots benchmark. Avoids doing full model work for holes below maxZ and
enables the exact B=2/4/8 kernels.

**Approach:** Maintain a compact active-slot index map and make state/KV kernels
indirect through it, or compact activation rows while binding persistent state
by mapped slot. Bucket exact active counts to specialized paths. Preserve
logical output ordering and prefix-cache ownership.

**Correctness gate:** G0-G7, `serve-test2` with holes at every index, slot
finish/restart, cache restore/fork, heterogeneous positions, and randomized
active masks.

**Risk:** Medium-high. Every stateful shader needs consistent physical/logical
mapping; an indexing error can cross-contaminate clients.

### 13. Concurrent multi-slot prefill batching — medium-high EV

**Expected value:** Potential 1.3-2x aggregate TTFT when several prompts arrive
together; no effect on isolated pp512 or steady one-stream decode.

**Approach:** Batch ready prompt chunks across slots into one token matrix with
per-row slot/base metadata. Reuse dense GEMM and grouped MoE across the union;
make attention KV writes and DeltaNet carry/state persistence slot-indexed.
Preserve synchronous API semantics initially through an internal queue/barrier,
then expose only if the server can coalesce starts safely.

**Correctness gate:** G0-G7, serial-vs-coalesced logits for different prompts
and lengths, handoff exact per slot, cache/fork, staggered arrival, and aggregate
TTFT/throughput.

**Risk:** High implementation complexity and API scheduling changes. DeltaNet
chunking currently assumes one contiguous recurrent state.

### 14. Decode DeltaNet state-update retile/MMA — medium EV

**Expected value:** 1-4% one-stream decode if the 30 recurrent layers improve;
also helps aggregate. The state update is a repeated 128x128 rank-one/matvec
operation and about 126 MB/token of state traffic.

**Approach:** Isolate phases of `dn_step`, sweep head/panel ownership, use
simdgroup MMA for 8x8 state tiles where f32 is competitive, vectorize state
loads/stores, and test keeping q/k/v/gate fragments resident through update and
output. Retain the existing fused conv+norm+step+gate structure as control.

**Correctness gate:** G0, G3, G5-G7, long recurrent sequences to expose state
drift, kDiv 0/2, and block plus full-decode A/B.

**Risk:** High numerical sensitivity and likely f32-MMA overhead; the earlier
state transpose was 16% slower and must not be repeated unchanged.

### 15. Adaptive split-K decode attention and context-length bucketing — medium EV

**Expected value:** Preserve short-context baseline while keeping the 1.36x
deep-context win; 1-10% on mixed-context serving where a long slot currently
sets the grid for all slots.

**Approach:** Measure scalar/split crossover, select per step or bucket slots by
context range, and dispatch only each bucket's live chunk count. Sweep reduction
fan-in and a TG-local multi-simd chunk reduction as a structural alternative,
without redoing the known-losing coalesced per-key loop or CK=64/256 sweep.

**Correctness gate:** G0, G4-G7, G9, context tails around every 256 boundary,
and slots with widely different lengths.

**Risk:** Medium. Extra dispatches for buckets can erase saved empty TGs; state
mapping must remain exact.

### 16. Router/select completion fusion — medium-low EV

**Expected value:** Under 1-2% decode/prefill, but removes one tiny serial stage
per layer if done without burdening expert consumers.

**Approach:** Have the final router-logit TG (atomic completion per token) run
the stable top-k/softmax and write SelT, or combine selection into a compact
follow-on region of the router kernel. Never inline selection into every expert
consumer—the prior version lost 39% from register pressure.

**Correctness gate:** G0-G7, exact selected ids/order/weights including ties and
both 256/512-expert models; end-to-end timing required.

**Risk:** Medium. Global atomic completion and device visibility can serialize;
small theoretical value.

### 17. Residual/normalization representation and fusion sweep — medium-low EV

**Expected value:** 1-3% prefill/decode if normalized-vector traffic and serial
one-TG reductions are removed.

**Approach:** Compare (a) raw residual + one inv-RMS scalar consumed by the next
projection kernels, (b) fused tail into router/down reduction, and (c) an
atomic-last-TG epilogue. Avoid recomputing RMS in every memory-bound consumer.
Measure the whole layer, not dispatch count.

**Correctness gate:** G0-G7 and isolated normalization error; full stream exact.

**Risk:** Medium-high. Previous stage-count fusion was perf-neutral; recompute
or extra registers can lose more than the saved 8 KB vector traffic.

### 18. GPU-resident multi-step command encoding / ICB — low-medium EV

**Expected value:** At most the measured ~2% submit/encode component for normal
decode, possibly more only if it keeps the device out of an idle state.

**Approach:** Let argmax write the next token/position buffers and encode
`chunkN` dependent steps in one command buffer, recording tokens to a ring;
truncate at first EOS on the host. Compare direct encoding, indirect command
buffers, and current per-step submission. Do not call this a fix for the
inside-CB barrier ramp without evidence.

**Correctness gate:** G0-G7, EOS at every step, slot completion/restart, and
token ring bounds.

**Risk:** Medium for only low EV; multi-slot EOS/finish masking is subtle.
Prior command-buffer pipelining already proved submit is not the main limit.

### 19. Activation/scratch arena aliasing and lazy allocation — medium memory EV

**Expected value:** Hundreds of MB of peak-memory reduction (not necessarily
speed), enabling larger context/spec scratch and reducing page pressure.

**Approach:** Perform a lifetime map of b*/bb* buffers; place non-overlapping
activations in one aligned arena or MTLHeap. Allocate full N*vocab logits only
for callers that request them; otherwise use fused top-1 scratch. Size verify
scratch to configured K. Avoid touching unused KV/state pages at open except
required MMA slack.

**Correctness gate:** G0-G7, guard/red-zone checks, maxB/context tails, split
stages, and RSS/footprint measurement.

**Risk:** Medium. Aliasing a live buffer creates silent corruption; Metal
resource-offset alignment and concurrent encoder overlap must be respected.

### 20. Prefix-cache snapshot compaction — medium memory/serving EV

**Expected value:** Potential multi-GB memory reduction and faster cache save
at large context/`QK_PCACHE=6`; small direct token-rate impact.

**Approach:** Store recurrent stripes in full but KV as compact live-length
per-head payloads with per-entry length metadata, instead of zero-initialized
full-context vectors. Grow entries lazily, preserve qkp3 state ABI, and measure
CPU copy/page-fault cost. Consider GPU blit/asynchronous copy only if CPU memcpy
is observed on the request critical path.

**Correctness gate:** G0, G5-G7, G10; restore entries at many lengths, overwrite
long with short and vice versa, six-entry eviction/LRU, split worker state ops,
RSS and copy time.

**Risk:** High production blast radius despite no math change. The live 80B
worker depends on exactly six entries and state ABI compatibility.

### 21. Optional f16 KV storage for capacity — low speed, medium memory EV

**Expected value:** About 1.8 GB saved in the measured configuration and roughly
2x KV capacity. No expected speed win at context 8000; revisit only at very deep
context where KV becomes bandwidth-significant.

**Approach:** Restore the prior complete fp16 storage/fp32 compute patch behind
an explicit precision flag, including all serial/batch/split readers/writers and
snapshot sizes. Benchmark at the deepest locally feasible context and use the
saved memory for a capacity demonstration, not a decode-speed claim.

**Correctness gate:** Self-consistent serial==batched/split argmax, coherent long
output, G4/G6/G7/G9 adapted to f16, cache save/restore, and explicit comparison
showing expected divergence from f32 at long context.

**Risk:** Known long-context greedy divergence and no measured speed gain.
Never default-on under the token-exact contract.

### 22. Optional reduced-precision DeltaNet/activation storage — low-medium EV

**Expected value:** State/activation memory and traffic reduction; possible
1-3% speed only if state traffic is limiting. Capacity value is modest relative
to weights/KV.

**Approach:** Test bf16/f16 recurrent state storage with f32 accumulation and
f16 transient activations in isolated opt-in variants. Quantify drift over long
sequences before any performance work.

**Correctness gate:** Self-consistency and coherent long output, serial/batched
handoff, state snapshot/restore, and quality-risk documentation.

**Risk:** Very high numerical/quality risk for recurrent state; likely hard
wall if drift compounds. Not eligible for token-exact default.

### 23. KV/cache layout and very-deep-context compression — low EV now

**Expected value:** Capacity and possible bandwidth gain only beyond the tested
8k context. Q8/fp8 KV could approach 4x capacity versus f32.

**Approach:** First profile 32k-65k on a feasible model/config. Only if KV
bandwidth becomes material, test blockwise Q8 KV with vectorized dequant in
split attention, or page/chunk layouts that match the split kernel. Do not
repeat the already-losing coalesced scalar inner loop.

**Correctness gate:** Precision-class self-consistency, G4/G6/G7/G9/G10,
coherent long output, capacity and speed measurements.

**Risk:** High quality/complexity risk and no evidence of a present bottleneck.

### 24. Offline weight layout/repacking and binary shader cache — low-medium EV

**Expected value:** Startup reduction and possible 1-3% kernel gains where
runtime dequant staging remains strided. No-copy RSS is a key feature and must
not be sacrificed casually.

**Approach:** Evaluate a versioned sidecar/GGUF packing for the few hot formats
that benefits both GEMV and GEMM without expanding weights; compare Metal
binary archives/offline libraries to runtime JIT for startup. Keep mmap windows
and tensor containment intact.

**Correctness gate:** G0-G7, byte/shape/version validation, forced multi-window
35B open, 80B open smoke, load time and RSS.

**Risk:** Medium-high operational complexity, duplicate storage, and potential
loss of zero-copy portability.

### 25. Prefill chunk/crossover autotuning — low-medium EV

**Expected value:** 1-5% long-prompt throughput and lower thermal sensitivity;
pp512 itself fixes N=512.

**Approach:** Sweep maxB/chunks {128,192,256,384,512,768,1024}, grouped-MoE
crossover, GEMM crossover, and DN chunk size under cool and soaked controls.
Select from device/model shape at open or retain explicit env knobs when the
optimum is workload-dependent.

**Correctness gate:** G0-G7, multi-chunk prompts across each boundary, exact
handoff, sustained thermal A/B.

**Risk:** Low correctness risk, medium benchmarking risk from thermals and
memory footprint.

### 26. Instrumentation: per-stage counters and reproducible perf ledger — enabling EV

**Expected value:** Indirect but high confidence: prevents chasing SLC/thermal
artifacts and exposes occupancy, bytes, and barrier idle time for ranks 4-17.

**Approach:** Add optional Metal counter samples/signposts at stage boundaries
where supported, a benchmark wrapper recording env/git hash/GPU probe/worker
state, and stage budgets for pp512, one-stream, B=8, and ctx=8k. Instrumentation
must be compiled/disabled out of normal runs or prove zero cost.

**Correctness gate:** G0-G7 with instrumentation off; identical token stream on;
cross-check total GPU time against command-buffer timestamps.

**Risk:** Low if opt-in, but counter sampling can perturb the schedule and is
not itself a performance lever.

## Conditional/external hard-wall investigations

These are real optimization families but cannot be completed without assets or
authority not currently present. Close them with evidence rather than silently
dropping them.

| Candidate | Expected value | Approach | Correctness gate | Risk / hard wall |
|---|---|---|---|---|
| Compatible external draft model | Potential 2x speculation | Inventory available GGUFs for exact 248,320-token vocab and benchmark draft cost; only add a second engine if draft/target economics are positive. | Distribution/greedy exact spec acceptance, G8, memory and end-to-end latency. | No compatible small draft is currently identified; a dense 27B draft is slower/larger than target. |
| Native MTP/Medusa/EAGLE head | Potential 1.5-3x | Load and execute only if the exact target checkpoint supplies compatible trained head tensors; share target command buffer and recurrent sandbox. | Head-reference validation, G8, quality policy for non-lossless modes. | Current 35B GGUF has no compatible MTP/EAGLE/Medusa weights; training/conversion is outside this repository task. |
| Early-exit/self-layer-skipping draft | Uncertain 1.1-1.5x | Measure intermediate-hidden output-head acceptance at several layer cutoffs before building stateful drafting. | Exact greedy verification, acceptance/cost threshold, G8. | The 417 MB head per draft step and low acceptance may make it strictly slower; recurrent scratch is complex. |
| MTL tensor operations / newer hardware features | Future | Query feature sets and implement only when the device exposes the API. | Full gates and hardware fallback. | M4 reports the needed tensor API unavailable; cannot optimize with absent hardware. |
| Cross-box network/worker pipeline | Potential latency win outside local engine | Use existing stage stats to distinguish network RTT, first-touch mapping, and state-copy costs; optimize only the locally owned measured component. | Split token refs, qkp3 ABI, production integration gate owned with tron. | Wired network/head scheduling require external coordination; never mutate the live deployment implicitly. |

## Measured dead ends: do not repeat unchanged

Fresh variants above are deliberately narrower than these closed experiments.

| Closed lever | Evidence / rule |
|---|---|
| f16 KV as a speed optimization | No win at ctx=8000 (f16 13424 ms vs f32 13404 ms), with long-context token divergence. Revisit only for memory or a demonstrated very-deep-context bandwidth wall. |
| Command-buffer pipelining as the decode fix | Encode+submit is about 2%; GPU work is about 96%. Only GPU-resident multi-step or speculation merits a bounded recheck. |
| More dispatch fusion by stage count alone | Removing about 200 stages/token moved under 2%. New fusion must remove material traffic/work or improve occupancy, not merely a barrier. |
| Decode-attention coalesced per-key inner loop | About 1% slower; CK=64 vs 256 was neutral. Do not repeat without a structural scheduling change. |
| DeltaNet state transpose | 16% slower due instruction/dependency cost. |
| Blocked 8x8 MMA forward solve | Correct but prep+solve was much slower than compact scalar; Neumann/doubling must beat the total chain. |
| Split-RHS scalar solve | Correct but slower than compact scalar. |
| Inline MoE re-selection in consumers | 39% slower from register pressure. |
| Grouped MoE read-once without hoisted dequant | Slower; weight decoding must be hoisted and reused. |
| Flash-attention exp2/causal-mask squeeze | Correct but perf-neutral. Geometry/data reuse is the remaining angle. |
| Plain f32 MMA GEMM substitution | Slower than scalar in early trials; packed half/f32-accumulate kernels are the proven prefill class. |

## Result log

Update this table in the same commit as each accepted lever or documented hard
wall. Record exact commands/config, control and candidate samples, thermal probe,
correctness gates, and commit id.

| Rank / lever | Status | Correctness | Performance result | Decision / commit |
|---|---|---|---|---|
| Plan and source audit | complete | Read all 1109 lines of `PORT.md`; audited Metal/Vulkan spec paths and all Metal kernel families | No benchmark; planning only | Initial plan commit |
| Clean campaign baseline | complete | G0-G7 green: prefillcmp 36/36 at 7.9e-7; dncmp all PASS; block/ablock PASS; decode exact; eight slots identical; handoff exact | Cool probe 546.8 GB/s; decode 8.54 ms/tok; B=8 195.2 tok/s; pp512 346.26 ms = 1479 tok/s | Control for subsequent A/B work |
| 1a. Metal oracle verifier | validated | Default hp path K={2,4,8,16,32} reproduced all 95 checked positions; every legacy G0-G7 gate re-run green. Scalar GEMM also exact at K={48,64,96}. Default hp K=64 had one near-tie mismatch, so exact default is capped at 32. | On one 96-token stream at 9.72 ms serial: K=2/4/8/16/32 verification was 7.82/5.85/4.93/4.55/4.36 ms/token = 1.24/1.66/1.97/2.13/2.23x oracle speedup. Scalar K=96 was 4.23 ms/token = 2.30x. | Keep `verify` harness; proceed to scratch rollback, then prompt lookup. This commit. |
| 1b. Metal recurrent scratch + rollback | validated | `speccmp` K=8 C={0,4,2,1} exactly reconstructed 128 serial tokens; C=1 covered 127 consecutive rejection/commit rounds. G0-G7, cache restore, and staggered two-slot serving all green after changing recurrent allocations. | Adds one shared 62.8 MiB recurrent scratch stripe. Full accept including live→scratch and promotion was 5.24 ms/token = 1.86x; forced avg accepts 4/2/1 cost 15.02/26.91/51.90 ms/token, confirming speculation must be trigger-gated. | Keep scratch/promotion/replay state machine; proceed to prompt-lookup integration. This commit. |
| 1c. Metal prompt-lookup policy | validated | Exact output versus serial on dormant ids3, partial-accept ids2, and echo fixture; G0-G8 plus cache/staggered-slot gates green with `QK_SPEC=1`. Multi-slot stays on serial for fairness. K serving is capped at 32 because of the measured hp near-tie; default remains L=6/K=8. | Echo A/B/A/B: 1301.7/1300.5 ms serial vs 949.0/951.6 ms spec = 1.37x, 12 rounds and 7.67 accepted. ids2 default L=6: 2246.0 -> 1953.5 ms = 1.15x, 7.62 accepted. ids3 had zero triggers and 2194.7 vs 2199.7 ms (neutral). Deliberately weak L=2 averaged 4.06 and slowed 8%, validating the gate. K={4,8,16,32} echo times were 1042.6/951.6/966.7/942.7 ms; K=8 retained for safer latency/partials. | Land exact opt-in `QK_SPEC`; keep collecting `[spec]` workload stats before any default-on decision. This commit. |
| 2a. Q8_0 dense slot batching | validated | New independent B={2,4,8} kernels are bit-exact to grid-z on distinct random RHS at both real shapes; TPR=32 changed bits and was rejected. G0-G7, cache, staggered B=2, and B=16 stress all green; 200-token B=8 output exact. The pre-existing shader prototypes remain uncommitted. | Isolated B8: qkv 170.1 -> 117.5 us (1.45x), output 85.7 -> 63.0 us (1.36x). End-to-end B=2 was 142.9/143.1 -> 145.7/145.1 tok/s; thermally stable B=4 was 126.0/128.3 -> 129.8/131.1; B=8 interleaves were 185-191 -> 194-208 tok/s (about 5-9%). B-width sweep favored B8. The B8 Q6_K head prototype was exact but 3-5% slower alone, so it is not wired. | Default exact-count Q8 path for compatible 35B slot counts; `QK_SLOT_BATCH=0` is rollback/A-B control. Proceed to MoE slot batching and a different head design. This commit. |
| 3a. Fused router slot batching | hard wall | B={2,4,8} reproduced the exact 200-token stream and eight-slot determinism. Implementation was removed after measurement. | At B8, adjacent dense-batched controls were 207.8/207.4 tok/s versus 205.4/203.4; B-width sweep at 100 tokens gave B2/B4/B8 186.8/184.7/181.4 against thermally bracketing controls 189.5/184.6. The B live RMS+dot accumulators reduce occupancy more than F32 router-row reuse saves. | Reject fused residual/RMS+router batching. A future router attempt must separate normalization or use MMA; proceed to quantized expert reuse. |
| 3b. Rank-wise gate/up slot batching | hard wall | IQ3 routed plus Q8 shared B={2,4,8} retained the exact 200-token stream and slot determinism, including a heterogeneous-route fallback. Implementation was removed after measurement. | B8 200-token pairs were 207.5/207.0 controls versus 208.3/203.1 (noise-to-loss). Width sweep was B2/B4/B8 187.2/185.7/183.3 between thermally falling controls 189.3/184.2. Sixteen live gate/up accumulators and serial RHS work erase same-rank route reuse. | Reject monolithic rank-wise batching. Revisit shared-only gate/up or compact expert-union grouping; proceed to the lower-register down projection. |
| 3c. Scalar rank-wise down batching | hard wall | IQ4/Q6 routed plus Q8 shared B={2,4,8} retained the exact token stream and determinism, with heterogeneous-route fallback. This implementation was removed. | B8 pairs: 207.4/207.0 controls versus 205.4/200.2. Width sweep B2/B4/B8 was 184.8/183.1/183.9 between falling controls 189.1/184.6. Even eight accumulators lose because one simdgroup serializes all RHS work. | Reject serial-RHS scheduling. Preserve one simdgroup per slot and share raw weights inside a threadgroup instead. |
| 3d. Threadgroup-staged down rows | hard wall | B8 IQ4 path kept one simdgroup per slot, preserved the exact 20-token stream, and left Q6 layers on the control. Implementation was removed. | Dense-batched controls were 207.3/207.0 tok/s versus 200.1/193.8. Raw quant-row copy plus a threadgroup barrier costs more than repeated cache-resident reads. | Reject explicit staging. Test compact expert grouping with existing MMA kernels, which changes compute efficiency rather than only cache level. |
| 3e. Compact live-expert grouped MoE | validated | The exact-class v4 path retained the complete 200-token control stream and eight-slot determinism; G0-G7, cache, staggered heterogeneous B=2, and equivalent B=16 stress are green. `moegcmp` measured v4 hidden-value drift at max relative 1.1e-3, but no accepted token changed. Exact prompt/cursor trajectory IDs guard the identical-slot fast path; arbitrary or sparse traffic falls back. | Interleaved B8 rollback/default pairs were 211.6/214.8 and 211.5/213.4 tok/s, averaging +1.2%. Compacting the launch from all 257 experts to the sorted live union was essential: full-grid v4 measured about 181 tok/s versus about 214 compact. The opt-in packed-f16 v5 path measured about 238 tok/s versus about 212 control (+12%) in the clean 200-token run and produced one identical 60-token stream at B={1,2,4,8}; it first diverges from the f32 control at generated token 42 (3023 versus 440), so it remains an explicit precision tier. | Default v4 only for equivalent B>=8 on the 256-expert 35B (`QK_MOE_SLOT_GROUPED=0` rollback). Keep v5 opt-in with its documented precision tradeoff; 512-expert 80B remains on the prior path. Live-list compaction also supplies the primitive needed by rank 4. This commit. |
| 4a. Packed-prefill non-empty work list | validated | The group pass emits bounded `(expert,32-assignment tile)` pairs; packed gate/up and both down formats consume them without grid-z overlaunch. Uniform routing at N={31,32,33,48,96,128,192,512,1024} and concentrated routing at N={31,32,33,511,512} were bit-identical to v5 for every gate/up and IQ4 down entry. The Q6 down twin compiles; the available 35B has IQ4 down throughout. Default G0-G7 are green, forced packed prefill is 36/36 argmax-consistent, and packed N=512 handoff is exact. | At N=512 uniform routing, gate/up fell 3.089 -> 3.010 ms/layer and down 1.648 -> 1.572 ms/layer. Full pp512 interleaves were 345.79/346.00 ms grid-z versus 340.53/340.89 ms compact: +1.5%, 1480-1481 -> 1502-1504 tok/s, with 562-563 GB/s bracketing probes. The host-known work bound is 400 pairs instead of the old 4,112-pair grid; sentinels avoid CPU synchronization or indirect-dispatch ordering. | Use compact work pairs behind existing packed `QK_MOE_GROUPED=5`; `QK_MOE_WORK=0` restores the exact v5 grid control. It retains v5's already-documented packed-f16 precision class and is bit-identical to v5 itself. Proceed to the resident multi-token-tile variant. This commit. |
| 4b. Resident 64-assignment MoE tiles | hard wall | A two-bank 64-assignment packed kernel kept every gate/up and down entry bit-identical to v5. A hybrid down path used the 32-wide kernel for counts <=32 and resident-64 only for larger experts; implementation was removed after measurement. | Uniform N=512 gate/up regressed 3.010 -> 5.610 ms because wider accumulator/X tiles double work for the many experts below 32 assignments and reduce occupancy. Even fully populated hot routing was 1.737 -> 1.787 ms. Resident down alone improved an artificial all-hot case 1.035 -> 0.833 ms; after adding the small/large hybrid needed for natural routes, uniform down was 1.647 -> 1.656 ms, so the extra dispatch erased shared-expert reuse. | Reject multi-token resident tiles at this geometry. The compact 32-wide work list is the rank-4 endpoint; revisit only with hardware/occupancy evidence or an indirect kernel split that avoids launching the high-register class for the natural routed distribution. |
| 5. DeltaNet resident-state step | validated | An 8-row/8-simdgroup threadgroup panel is bit-identical to the streamed kernel for output and final state at Tn={63,127,128,511,512}; both kDiv mappings pass the required dncmp matrix. G0-G7, packed-N512 handoff, cache restore, staggered heterogeneous slots, and all speculative rollback corruption modes are green. The pre-existing kDiv=2 Tn=128 reference threshold is unchanged because resident and streamed states match bit-for-bit. | Isolated Tn512 step fell from 582-583 to 285-289 us/layer and the full KQ+solve+step chain from 936-939 to 646-649 us. Interleaved pp512 controls were 342.56/340.90 ms versus 332.38/333.16 ms resident, about +2.7%, 1495-1502 -> 1537-1540 tok/s, with 560-562 GB/s probes. Geometry sweep rejected PW32/NSG4 530 us, PW16/NSG4 330 us, PW8/NSG4 296-313 us, PW8/NSG2 378 us, PW8/NSG16 312 us, PW16/NSG8 294 us, and PW32/NSG8 361 us. | Default the 8-row/8-simdgroup resident kernel; `QK_DN_STEP_RES=0` restores the streamed bit-exact control. Losing geometries were removed. Proceed to triangular solve. This commit. |
| 6a. DeltaNet exact SIMD decay staging | validated | One simdgroup issues the 64 decay/beta loads in parallel but reproduces the original left-to-right addition sequence through shuffles. The complete dncmp boundary matrix is unchanged for both kDiv mappings; G0-G7, packed-N512 handoff, cache restore, staggered heterogeneous slots, and every speculative rollback mode are green. A parallel prefix scan was rejected because reassociation raised Tn512 state error from 0.00490 to 0.00570. | Interleaved Tn512 controls were 252.4/251.3/254.3 us versus 239.5/235.1/239.3 us, about 5.9% faster for the solve. pp512 is below reliable whole-graph resolution: candidate 333.15/331.41/333.07 ms versus control 332.47/332.48/333.46 ms. Cached exponent factors were neutral at 254 us; 64/32-thread persistent-column variants regressed to 265/428 us; 16/32-byte triangular-row padding was noise. | Keep the exact load-parallel/scalar-order prefix as the rank-6 micro-win; it removes about 15 us per DeltaNet layer without changing any arithmetic result. This commit. |
| 6b. DeltaNet Neumann/doubling inverse | hard wall | The degree-63 product `(I+N)(I+N^2)...(I+N^32)`, N=-M, passed the full dncmp matrix in both a monolithic 32-KiB kernel and a split inverse/application design. Reordered f32 MMA changed small hidden values but remained inside every recurrent bound. The prototype was removed after measurement. | Five optimized strict-lower MMA squarings alone cost only 83.4 us, but the complete monolithic solve was 444.1 us. Parallelizing the 256 RHS columns in a second dispatch improved this to 394.9 us: inverse/prep 298.2 us plus application 93.0 us, versus 238-254 us scalar. The inverse/prep alone is already slower than the entire control. | Reject Neumann/doubling on M4 Max. Ten triangular matrix products, 32-KiB occupancy, and RHS application make the algebraic parallelism a net loss. The prior blocked diagonal-wavefront MMA solve and split-RHS scalar result close the remaining structurally distinct solve schedules. |
| 7. Grouped-down reduction + residual/RMS fusion | hard wall | The fused 256-thread geometry reproduced both raw residual and normalized output bit-for-bit at N={32,512}. A threadgroup-resident raw tile was also bit-exact. Native 128/512-thread reductions kept raw output exact but changed normalized values at only 2.8-2.9e-7; an emulated eight-simdgroup tree made 128 threads bit-exact. All prototypes were removed. | At N=512, separate reduce+add_rmsnorm measured 0.091-0.099 ms/layer. Fused native 128/256/512 measured 0.084-0.092 / 0.097-0.105 / 0.090-0.097 ms; exact 256 with an 8-KiB resident raw tile was 0.090-0.094 ms, while exact-tree 128 was 0.115 ms. Full pp512 A/B was control 332.60/332.91/331.39 ms versus fused 332.38/332.28/333.17 ms: no repeatable movement. | Reject the fusion. `dY` traffic dominates; the removed `bbMY` pass is already cache-efficient, and reducing eight dimension TGs to one TG/token offsets its traffic saving. Do not add atomic completion to decode when the lower-risk prefill form is neutral. |
| 8a. Exact f32 batched Q6_K head | validated | The 32-row x 8-token f32-MMA tile matched shipping-head argmax on distinct real-weight RHS at N={4,5,8,9,16,32}, chose id 0 for deliberate all-logit ties, and stayed within 4.3e-6 max absolute logit error. The full scalar prefill sweep is 36/36 exact with 7.9e-7 worst relative error. G0-G8, cache restore, staggered two-slot serving, eight-slot determinism, and handoff are green. | Head+argmax at N=4/8/16/32 fell from 2.739/5.519/11.030/22.025 ms to 2.426/2.669/5.128/10.025 ms: 1.13/2.07/2.15/2.20x. Oracle verifier A/B at K=4/8/16/32 improved from 5.80/4.97/4.75/4.43 to 5.63/4.68/4.43/4.15 ms/token (about 3-6%). N=2 is slower and remains on GEMV. BK=32 was 1.5% slower; BK=128 was 43% slower; f32 BN=16/32 only tied repeated BN=8 and were removed. | Default the tiled head only for per-position argmax batches N>=4; `QK_HEAD_GEMM=0` restores shipping GEMV and `QK_HEAD_GEMM_N` exposes the crossover. Full-logit single-row callers remain exact GEMV. This commit. |
| 8b. Single-row fused head top-1 | hard wall | Exact shipping-order spans {16,64,256} returned the same winner as materialized logits, including stable lower-id ties. The prototype was removed after decode A/B. | Span 16 was the isolated winner at about 0.84 ms, but real one-slot decode was flat: 200-token control/candidate 92.2/92.0 tok/s and 100-token control/candidate 85.5/85.6 tok/s. The head reads 398 MiB while logits are only 0.95 MiB, so suppressed stores and argmax cannot move the graph. | Reject B=1 fusion. Preserve full logits for sampling/debug and pursue batched per-tile candidate reduction only as a memory lever. |
| 8c. Batched fused head candidates + compact logits scratch | validated | The M32/M64 kernels preserve the exact f32 tile arithmetic per logit, return every real-weight row winner at N={4,8}, choose id 0 for all-zero ties, and materialize the final logit row bit-for-bit against the full tiled head. The strengthened scalar prefill gate now directly exercises fused argmax and is 36/36 exact at 2.9e-6 worst relative final-logit error. `stagecmp` proves final top-32 ordering/top-1 coherence; G0-G8, cache restore, staggered serving, eight-slot determinism, and handoff are green. | M64 fused candidate+reduce at N=8/16/32 measured 2.604/5.046/9.830 ms versus 2.672/5.134/10.021 ms for full tiled logits+argmax (about 1-3%); M32 remains the neutral/slightly faster N=4 geometry. M96 was exact but slower at N=8 (2.692 ms) and was removed. Default maxB=128 last-stage head scratch falls from about 121.25 MiB to 6.6 MiB; maxB=512 falls from about 485 MiB to 18 MiB, saving about 467 MiB. pp512 was 331.11 ms compact versus 331.94 ms rollback, so no regression. | Default per-tile candidates for N>=4, M32 below N=8 and M64 otherwise; retain only the final full-logit row when sampling/top-k needs it. `QK_HEAD_GEMM=0` restores shipping full logits. This commit. |
| 8d. Optional half-input/f32-accumulate verifier head | validated precision tier | Real-weight `headcmp` at N={4,5,7,8,9,16,17,32,33,64,128} retained every f32 winner and stable id-0 ties; random final-row drift stayed below 5.0e-4 absolute. The strengthened scalar prefill sweep is 36/36 argmax-consistent with 6.9e-4 worst relative final-logit drift. Oracle verification is exact on ids1/2/3/4, every K=8 rollback corruption mode is exact, top-32 agrees with the materialized final row, and all adapted G0-G8/cache/staggered-slot/handoff gates are green. | Fused N=4/8/16/32 head time fell from exact-f32 2.438/2.604/5.046/9.830 ms to 1.846/1.858/2.528/3.980 ms (1.3-2.5x). Bracketed ids3 oracle K=4/8/16/32 improved from 5.61/4.54/4.18/4.00 to 5.40/4.44/4.02/3.83 ms/token (about 2-4%). Using all 256 threads for half-row Q6 staging improved N=8 another 2.7%. A 128-row/16-simdgroup tile was exact but 6-8% slower; native B64/B128 accumulator widths hit a severe register cliff, so both were removed in favor of repeated B32. | Ship only behind `QK_HEAD_F16=1`; N<4 and the default path remain exact f32. The tier explicitly trades unseen near-tie risk for verifier speed and is not default-on. This commit. |
| 9. Prefill flash-attention geometry + two-head GQA reuse | validated | The exact Q8/K64/S16 two-head twin is bit-for-bit identical to shipping Q16/K64/S8 at N={1,7,9,17,65,128,512,1024} and nonzero bases {37,63,127,384,512}. The SIMD-softmax winner changes only probability-sum order and stayed within 6.0e-8 absolute / 5.9e-7 relative in `facmp`; the full scalar-projection sweep is 36/36 token-exact at 2.4e-6 worst final-logit relative drift. A 1,040-token multi-chunk prompt reproduced the exact 20-token Q16 continuation. G0-G8, handoff, top-k, cache restore, staggered serving, and eight-slot determinism are green. | At N=512/base=0, isolated attention fell from about 1.4-1.5 ms/layer shipping to 0.531 ms exact-GQA and 0.462 ms SIMD-GQA; at base=512 it fell from 3.27 to 1.30/1.14 ms. Cool 563 GB/s pp512 interleaves improved from 332.39/331.78 ms (1540-1543 tok/s) to 324.63/325.27 ms (1574-1577 tok/s), about +2.2%. The 1,040-token prompt run fell 2855.6 -> 2812.6 ms. Q8 unpaired was about 329-330 ms; K128 tied at base 0 and changed grouping; S4 and unpaired S16 lost; register-resident O was 0.598 versus 0.531 ms; all were removed. Four Q heads with QTM=8 require over 40 KiB TG memory, while QTM=4 needs packing/padded 8x8 MMA and cannot preserve the reuse without extra barriers. | Default Q8/K64/S16 GQA2 with SIMD softmax. `QK_FA_GEOM=exact` preserves the winning geometry with scalar-order softmax; `QK_FA_GEOM=q16` restores the original kernel. This commit. |
| 10a. Full-tile Q8_0 GEMM specialization | validated | The bounds-free 64x32 packed kernel is bit-for-bit identical to the safe kernel across all real M/K families at N={32,96,128,512}, including 4,194,304 compared outputs at 8192x2048x512. Runtime selection requires M%64=N%32=0; all tails retain the original kernel. The mandated scalar prefill sweep is 36/36 at 2.4e-6 worst relative drift; dncmp/block/ablock, exact 20-token decode, eight-slot determinism, and handoff are green. Packed-f16 prefill is the same pre-existing 35/36 near-tie with specialization on or off. | Isolated aligned real shapes improved about 1.5-4%: 8192x2048x512 1344 -> 1325 us, 2048x4096x512 710 -> 696 us, and 512x2048x512 131 -> 125 us. Cool 564 GB/s pp512 interleaves were 324.58/323.40/324.07 ms safe versus 322.94/321.91/322.71 ms aligned, about +0.5% (1580-1583 -> 1587-1591 tok/s). Existing scalar/sg/h/h2 variants all lost; packed 64x64, 128x32, 32x64, 128x16 and K64 prototypes ranged from neutral to 31% slower and were removed; direct transposed MMA stores were neutral. | Select the aligned specialization by default for full tiles; `QK_GEMM_ALIGNED=0` restores the safe bit-identical control. Continue rank 10 with isolated IQ4_XS tensors. This commit. |
| 10b. Full-tile IQ4_XS GEMM specialization | validated | The real-weight `iq4gemm` harness finds zero bit mismatches against the safe kernel for every 80B dense family MxK={8192x2048,4096x2048,2048x4096,512x2048,64x2048} and N={32,64,96,128,256,512,1024}; the largest comparison covers 8,388,608 outputs. Full 80B aligned and rollback prefill sweeps reproduce the identical pre-existing packed-f16 35/36 near-tie and every per-case drift. A 40-token 80B serving stream is exact across rollback, eight slots are identical, attention-block smoke passes, and N128 handoff is exact. The complete 35B G0-G7 matrix remains green: scalar prefill 36/36 at 2.4e-6, dncmp/block/ablock PASS, exact decode, eight-slot determinism, and handoff. | At N=512, safe/aligned us were 1477/1425 (8192x2048), 756/730 (4096x2048), 783/752 (2048x4096), 143/136 (512x2048), and 76/70 (64x2048), about 3-8%. The qkv winner stays 3.5-5% faster from N32 through N1024. Thermally annotated 80B pp512 A/B/B/A was safe 546.96 ms at 563.7 GB/s, aligned 539.29 at 564.3, aligned 533.52 at 558.2, safe 543.72 at 562.5: +1.4-1.9% in the adjacent brackets (936/942 -> 949/960 tok/s). Exact aligned 64x64 and 32x64 reuse tiles were 1467/1479 us at qkv versus 1425 us for 64x32; 64x64 regressed the K4096 shape to 826 versus 752 us, so both were removed. | Use the same full-tile selector and `QK_GEMM_ALIGNED=0` rollback for IQ4_XS on 80B. Retain `iq4gemm` for real-tensor regression and close rank 10. This commit. |
| 11a. Fixed-K Q8_0 decode GEMV + shape-specific TPR | validated | The real-weight `q8gemvcmp` harness covers qkv, gate, output, and shared-expert projections. Explicit unrolling changes 56-58% of K2048 result bit patterns through compiler reassociation, but max absolute drift is only 1.2-2.4e-7 (3.9-7.9e-7 of output RMS); K4096/TPR128 is bit-exact. The production TPR16 path passes scalar prefill 36/36 at 2.5e-6, all dncmp/block/ablock checks, the mandated exact 20-token decode, 200-token eight-slot determinism, and exact handoff. | Real fixed/safe isolated us were qkv 24.2/25.1, gate 13.6/14.3, output 18.6/19.2, and shared gate 3.7/4.1. With matched 559-563 GB/s probes, full 383-token A/B/B/A was rollback 4117.8/4134.2 ms (93.0/92.6 tok/s) versus fixed 4054.3/4061.3 ms (94.5/94.3), a repeatable +1.5-1.8%. Full-graph sweeps selected TPR16 for K2048 and retained TPR128 for K4096; TPR32/64/128 K2048 and smaller K4096 widths lost end to end despite some isolated wins. Bounds-only specialization was neutral, while a two-chain TPR16 accumulator was neutral/slower and was removed. | Default the K={2048,4096} specializations and TPR={16,128}; `QK_GEMV_FAST=0` restores the original dynamic K and TPR={64,128} path. Keep `QK_GEMV_TPR_A/O` for diagnostic sweeps and the real-weight comparison harness. This commit. |
| 11b. Fixed-K IQ4_XS decode GEMV | validated | `iq4gemvcmp` reports zero bit mismatches against the dynamic kernel on all 80B dense families tested: 8192x2048 qkv, 4096x2048 gate, 2048x4096 output, and 64x2048 SSM projection. Eight 80B slots remain identical and N128 handoff is exact. The 80B scalar-prefill sweep retains its identical pre-existing packed-f16 35/36 near-tie; the serial fixed path is bit-exact to rollback. The final 35B G0-G7 matrix is green: scalar prefill 36/36 at 2.5e-6, dncmp/block/ablock PASS, exact mandated decode, eight-slot determinism, and handoff. | Real safe/fixed bracketed us were 25.7-26.2/23.7-23.8 (qkv, +8-10%), 14.5-15.3/13.3-14.1 (gate, +8-13%), 16.4-16.7/14.1-14.5 (output, +13-15%), and 4.49-4.54/2.39-2.52 (64-row projection, +44-48%). Clean 80B 400-token brackets were fixed 5095-5105 ms (78.4-78.5 tok/s) versus rollback 5172-5178 ms (77.3), +1.3-1.6%; an earlier independent bracket measured +1.4-1.8%. One 5061 ms rollback outlier contradicted both adjacent controls and was retained in the ledger but excluded from the stable interval. NSG={1,2,4,8} was essentially tied on qkv, while NSG2 was safest across output and small shapes; dynamic NSG1 was end-to-end neutral. | Default fixed K2048/K4096 pipelines for IQ4 dense projections with `QK_IQ4_FIXED_K=0` as the bit-exact rollback. Retain NSG2 and the real-weight comparator. This commit. |
| 11c. Fixed-K Q6_K output-head GEMV | validated | `q6gemvcmp` compares all 248,320 real logits and reports zero bit mismatches for NSG={1,2,4,8}. The final default passes scalar prefill 36/36 at 2.5e-6, every dncmp/block/ablock check, the mandated exact decode, eight-slot determinism, and exact handoff. | At NSG2 the 397.9 MiB real head pass fell from 816-821 to 788-790 us, +3.5-4.0%; all NSG widths had the same roughly 790 us fixed result, so NSG2 remains. A 200-token single-command-buffer A/B/B/A resolved the whole-graph movement: rollback GPU time 1751.6/1754.2 ms versus fixed 1746.8/1747.0 ms, about 30 us/token or +0.34%. Short serving brackets were thermal-noise limited (94.3-94.9 tok/s) and showed no regression. | Default the bit-exact K2048 head specialization; `QK_HEAD_FIXED_K=0` restores the dynamic kernel. Keep `q6gemvcmp` as the full-logit regression. This commit. |
