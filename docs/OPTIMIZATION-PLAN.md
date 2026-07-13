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

**Approach:** Retile state columns from 64 to 32 (and sweep 16/32/64), allowing
a 128x32 state panel plus D scratch to fit in 32 KB threadgroup memory. Load the
panel once, run all chunks while it remains resident, and write once; remove
device barriers/reloads inside the chunk loop. Sweep TG/simdgroup geometry and
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

