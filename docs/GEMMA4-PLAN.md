# Gemma 4 26B-A4B QAT Q4_0 on Vulkan/RDNA3

Status: design and staged measurement plan only. This document does not authorize engine implementation.

## Scope and inspected facts

The target is `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, text only, on the RX 7900 XTX. The XT is a calibration and portability target, not a tensor-parallel partner. The local text GGUF at `/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf` is a real, fully allocated 14,439,363,584-byte file (13.45 GiB), and its tensor table parses. Do not load the vision projector.

This is a new graph, not a parameter substitution in qk's Qwen path. The current graph and shaders assume Qwen normalization/epilogues, Q/K dimensions centered on 128, full causal attention, a 512-lane MoE selector, and large experts. Gemma 4 needs native Q4_0, two attention geometries, a 1024-token KV ring, Gemma normalization/scaling, GELU experts, and much smaller routed matrices. It has no DeltaNet; all `dn_*.comp` shaders are out of scope.

Inspection of the GGUF and llama.cpp's `gemma4` graph corrects three supplied assumptions:

1. Full attention is at one-indexed layers 6, 12, 18, 24, and 30: **five full and 25 sliding layers**, not 6/24. The explicit list in the prompt was right; its prose count was not.
2. Full layers use 16 query heads of dimension 512 and two KV heads of dimension 512. Sliding layers use 16 query heads and eight KV heads, all dimension 256. A full layer has one raw 1024-wide K/V projection; learned K normalization and plain V normalization turn it into distinct cached K and V.
3. The tied `token_embd.weight`/LM head is **Q6_K**, not Q4_0. It occupies 605,552,640 bytes. The major attention/FFN weights are Q4_0 and the routers are f32.

All traffic counts below are decimal unless marked MiB/GiB. Q4_0 is exactly 18 bytes per 32 weights, or 0.5625 byte/weight. A bandwidth roof is a lower bound on time, not a prediction.

## 1. Corrected roofline and bandwidth calibration

### 1.1 Exact active weight traffic

The supplied 2.10 GB/token estimate is 13.3% low. Its attention estimate is close, but it prices the delivered 605.6 MB Q6_K head as a 415.2 MB Q4_0 head and uses the inconsistent layer split.

| Component | Derivation | Active parameters/token | Stored bytes/token |
|---|---:|---:|---:|
| 25 sliding attention layers | `25*2816*(4096+2048+2048+4096)` | 865,075,200 | 486,604,800 Q4_0 |
| 5 full attention layers | `5*2816*(8192+1024+8192)`; raw K is also V | 245,104,640 | 137,871,360 Q4_0 |
| Routed experts | `30*8*3*2816*704` | 1,427,374,080 | 802,897,920 Q4_0 |
| Shared FFN | `30*3*2816*2112` | 535,265,280 | 301,086,720 Q4_0 |
| Tied embedding/LM head | `262144*2816` | 738,197,504 | 605,552,640 Q6_K |
| Router | `30*2816*128` | 10,813,440 | 43,253,760 f32 |
| Norms, scales, biases, active embedding row | tensor ledger | — | 2,788,408 |
| **Total** | | **3,821,830,144 matrix weights** | **2,380,055,608** |

The format totals are 1,728,460,800 Q4_0 bytes, 605,552,640 Q6_K bytes, and 46,042,168 other bytes. At the nominal specifications:

| GPU | Nominal BW | Corrected zero-context roof | Supplied roof | Correction |
|---|---:|---:|---:|---:|
| RX 7900 XT | 800 GB/s | 336.1 tok/s | 381 tok/s | -11.8% |
| RX 7900 XTX | 960 GB/s | 403.4 tok/s | 457 tok/s | -11.7% |

The model still deserves the A4B label: about 3.82B matrix parameters are active. Stored bytes do not scale uniformly with that count because the head and routers use more expensive formats.

### 1.2 KV capacity and context-dependent traffic

A sliding layer stores `2*8*256*2 = 8192` bytes/token. Its capped allocation is:

`25*1024*8192 = 209,715,200 bytes = 200 MiB`.

A full layer stores `2*2*512*2 = 4096` bytes/token. Five layers use `20,480*context` bytes: 640 MiB at 32K and 5 GiB at 256K. Maximum f16 KV is therefore 5.1953125 GiB, not 12.6 GB. The reported 13.43-GiB model plus maximum KV is about 18.63 GiB, leaving roughly 5.37 GiB on a 24-GiB card for scratch and allocator overhead. Single-GPU execution fits, but maximum context plus batching or an assistant needs a peak-allocation gate.

Assuming the attention shader physically reads each live K/V value once and reuses it across GQA query heads:

| Prior context | Sliding KV read | Full KV read | Total traffic floor | XT nominal roof | XTX nominal roof |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 2.380 GB | 336.1 | 403.4 |
| 1,024 | 209.7 MB | 21.0 MB | 2.611 GB | 306.4 | 367.7 |
| 8,192 | 209.7 MB | 167.8 MB | 2.758 GB | 290.1 | 348.1 |
| 32,768 | 209.7 MB | 671.1 MB | 3.261 GB | 245.3 | 294.4 |
| 262,144 | 209.7 MB | 5.369 GB | 7.958 GB | 100.5 | 120.6 |

These are optimistic: they exclude cache writes, activation traffic, softmax reductions, dispatch floors, and imperfect coalescing. A per-query-head kernel rereads sliding KV twice and full KV eight times; avoiding that is first-order, especially at long context.

The supplied `tg128` baseline starts from a cleared context; it does not follow the separately reported pp512 test. Its average live length is 64.5, so perfect GQA reuse reads only 14.531 MB of KV/token (28.836 MB on the last token). A real decode after pp512 averages 576.5 positions and reads 129.874 MB/token. Keep both measurements; do not use the latter to explain the former.

### 1.3 Empirical bandwidth calibration

Use qk's existing measurements as anchors, not the spec sheet. `docs/amd-opt/REPORT.md` reports the following large-buffer rates:

| Format | XTX GB/s | XT GB/s |
|---|---:|---:|
| raw/f16 | 917.8 | 778.8 |
| Q8_0 | 927.8 | 775.0 |
| Q6_K | 800.4 | 606.5 |
| IQ4_XS | 729.0 | 583.3 |
| IQ3_XXS | 543.9 | 421.8 |

Add Q4_0 cases to the same 2,000-iteration harness. Use allocations above 96 MiB so Infinity Cache cannot dominate, exercise inner dimensions 704, 2112, and 2816, and include a `262144 x 2816`-sized stream. Bracket each run with raw/f16, Q8_0, Q6_K, and IQ4_XS controls. Pin the current cards by PCI address (`QK_DEVICE_PCI=1a:00.0` XTX, `03:00.0` XT), verify enumeration first, and drain persistent GPU workers. Record payload GB/s, shader time, workgroups, clocks, Mesa/RADV version, and hot/cold variance.

Q4_0 reads 18 bytes per 32 weights versus IQ4_XS's 17, but removes the codebook lookup and 6-bit subscale extraction. The first hypothesis is 850–900 GB/s on XTX and 700–760 GB/s on XT. The calibrated serial weight floor is:

`T_weight = 1.7284608/Q4_GBps + 0.60555264/Q6K_GBps + 0.046042168/raw_GBps` seconds.

At Q4=850–900, Q6_K=800.4, and raw=917.8 GB/s, the XTX floor is 2.84–2.73 ms, or 352–367 tok/s. The provisional XT bracket is 3.52–3.33 ms, or 284–300 tok/s. All later predictions must be replaced with this measured formula.

## 2. Per-kernel design and arithmetic

### 2.1 Native Q4_0 GEMV and GEMM

Q4_0 stores one f16 scale followed by 32 signed values packed as 16 nibbles; exact dequantization is `d*(nibble-8)`. Low nibbles encode the first 16 values and high nibbles the next 16. Preserve the QAT blocks verbatim. Do not route them through IQ4_XS.

For decode GEMV, start with subgroup reductions and f32 accumulation:

| Inner dimension | Q4 blocks/row | Encoded bytes/row | Initial threads/row | Rows/256-thread WG |
|---:|---:|---:|---:|---:|
| 704 | 22 | 396 | 32 | 8 |
| 2112 | 66 | 1,188 | 32 | 8 |
| 2816 | 88 | 1,584 | 32 | 8 |

Specialize those three inner dimensions so trip counts and tails are compile-time constants. Load each packed word and scale once, process its paired low/high nibbles, and reduce in the subgroup. At K=2816 each wave lane consumes two blocks and lanes 0-23 consume a third. Sweep TPR32/64; TPR128 wastes task lanes on 88 blocks and needs four waves to reduce one row.

For prefill and speculative verification, add weight-stationary Q4_0 GEMM. Start from the existing `gemm_q8_0`/`gemm_iq4_xs` BM128, BN64, BK64, local256 structure and retain BN32 for small batches. F32 LDS is about 33 KiB for weights plus 16 KiB for activations, allowing one workgroup/CU but not naive double buffering. Decode and GEMM must share the exact dequant routine. The pp target is to beat the measured 3505.04 tok/s; a point prediction waits for the Q4 tile benchmark rather than assuming an implausibly low 8–12 TFLOP/s.

The 1.728 GB Q4 portion costs 1.92–2.03 ms at 900–850 GB/s. Relative to the measured 729-GB/s IQ4_XS rate, 875 GB/s saves about 0.40 ms/token, roughly **29 tok/s** near the final budget. Simpler dequant buys lower VALU/VGPR pressure, not fewer QAT bytes.

Predicted short-context contribution: **0.82 ms attention projections + 0.43 ms shared FFN + the Q4 portion of routed time below**. Gate: exact CPU block dequant versus ggml; GEMV/GEMM agreement; f32-reference error corpus; SPIR-V validation; measured bandwidth sweep before graph work.

### 2.2 Router and expert selection

The router is a 128-logit f32 GEMV. Replace the current 512-lane selector with local size 128 (four wave32s) and an exact hierarchical top-8. Shrink `SelT` from the Qwen-oriented 16-entry/shared-gate layout to eight uint IDs plus eight f32 weights: 64 bytes, with C++/GLSL alignment assertions.

Gemma routing is not Qwen routing. Build a separately RMS-normalized router input, multiply by `1/sqrt(2816)` and the learned per-dimension scale, run the f32 `2816 x 128` router, select exact top-8 with llama.cpp's tie ordering, then softmax those eight. Keep router GEMV and selection separate until parity; fuse only after dispatch timestamps justify it.

Traffic is 1.442 MB/layer, 43.25 MB/token. Predicted contribution: **0.16 ms** for all 30 routers and selections. Gate: exact logits, IDs, ordering, weights, and adversarial ties before any expert shader consumes `SelT`.

### 2.3 Routed experts: two dispatches, not 16

The fused file layout matters. Per selected expert, gate/up is
`2816*1408*0.5625 = 2,230,272` bytes and down is
`704*2816*0.5625 = 1,115,136` bytes: **3,345,408 bytes/expert**. At 900 GB/s
that is only 3.72 microseconds and 34.8 KB/CU. A single expert cannot sustain
96 CUs. Eight experts are 26.763 MB/layer.

Keep qk's useful structure: one combined gate+up dispatch and one down dispatch
per layer. Use x for output-row tiles, y for eight selected ranks, z for slots,
and fetch physical expert ID through `SelT`.

Pair corresponding gate/up rows. A local256 WG computes eight pairs, one pair
per wave with two accumulators, then applies GELU(gate)*up. Across top-8 this is
704 WGs and 5,632 waves, **58.7 waves/CU**, reading 17.842 MB. Sweep four versus
eight pairs/WG because the two accumulators plus GELU may cross a VGPR cliff.

For down, one wave computes an output row and loops over
`8*(704/32)=176` Q4 blocks. Eight rows/local256 gives 352 WGs and 2,816 waves,
**29.3 waves/CU**, reading 8.921 MB. Apply both the softmax route weight and
`ffn_down_exps.scale[e]` before the final sum.

The pair's floor is 29.7 us/layer at 900 GB/s and 0.892 ms/model. Budget
**1.12 ms**, 717 GB/s or 80% of the large-Q4 target. The 90-97% achieved by
large qk streams is not credible for 1-18 MB phases with cold IDs, launch,
tails, GELU, and indirection; accept 65-80%, and retile below 60%.

A per-expert implementation is 16 dispatches/layer versus two. At qk's ~1.4-us
tiny-stage floor, fusion removes 420 dispatches and up to 0.588 ms/token,
roughly **37 tok/s** around the final point.

For prefill batch `B`, the expected union under uniform routing is `U(B)=128*(1-(120/128)^B)`: 29.1 experts at B=4, 51.6 at B=8, 82.4 at B=16, and essentially all 128 at B=128. Sort token/expert pairs once, run grouped expert GEMMs, then scatter-weight-reduce. Measure real histograms because clustering improves reuse but can hurt load balance.

Gate: router outputs, GELU gate/up tensors, each expert contribution, weighted down result, and a complete routed branch match llama.cpp debug tensors and a scalar oracle.

### 2.4 Shared dense expert and Gemma epilogues

The always-on shared branch contains three `2816 x 2112` Q4 matrices per layer and uses GELU. It reads 301.09 MB/token. At 800–880 GB/s plus activation/epilogue work, predict **0.43 ms**.

Implement the graph order explicitly:

1. scale token embedding by `sqrt(2816)`;
2. attention pre-norm, attention, post-norm, residual;
3. shared pre-norm, GELU gate/up/down, shared post-norm;
4. routed pre-norm, separately scaled router path, GELU experts, routed post-norm;
5. sum shared and routed branches, post-FFN norm, residual, and layer output scale;
6. final norm, tied head, multimodal-token suppression, and logit softcap.

The routed input is `RMS(pre_ffw_norm_2, attn_out)`, but the router input is a
different `unweighted_RMS(attn_out)/sqrt(2816)`, multiplied elementwise by F32
`ffn_gate_inp.scale` before the F32 router GEMV. The selected down contribution
is multiplied by F32 `ffn_down_exps.scale[e]` as well as its softmax weight.
Dense and MoE both run on every layer, receive `post_ffw_norm_1` and
`post_ffw_norm_2` separately, are added, then pass through `post_ffw_norm`, the
residual, and `layer_output_scale`. This ordering is copied from
`src/models/gemma4.cpp`, not inferred from names.

Fuse only graph-adjacent operations after their rounding order passes. Predicted norms/residuals/GELUs/cache writes: **0.20 ms**, with another **0.06 ms** miscellaneous command-buffer cost. Gate: every branch and layer boundary matches before full-model parity.

### 2.5 Sliding and full attention

Use two compile-time families:

| Type | Count | Q heads | KV heads | Head dim | Rotary scalar dims | Theta | Live span |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sliding | 25 | 16 | 8 | 256 | 256 | 10,000 | last 1,024 |
| Full | 5 | 16 | 2 | 512 | 128 effective | 1,000,000 + frequency factors | all |

Full layers are zero-indexed 5, 11, 17, 23, and 29. Derive the table from GGUF metadata and assert those five positions.

The existing `fa_attn_srv*` path is full-causal, uses Qwen-specific Q gating/scaling, and has arrays/tiles that top out around 256 elements. Gemma has no attention output gate, uses configured attention scale 1.0 rather than `1/sqrt(head_dim)`, and full `head_dim=512` breaks that tiling. In the first full variant, each of 256 lanes handles dimensions `t` and `t+256` for norm, dot, and output accumulation.

Sliding masking is exactly `[max(0, position+1-1024), position]`. Store K/V in a 1024-slot ring but use absolute position for RoPE. Full cache stays linearly addressed. Sliding Q/K have learned per-head RMS norms. Full raw K/V is normalized separately for learned K and plain V before both are cached.

The GGUF's 256 full-layer RoPE factors leave the first 64 pairs active and suppress the other 192, rotating 128 scalar dimensions. Generate factors per position/chunk; a resident `262144*512` table wastes hundreds of MiB.

Reuse one KV head across two sliding Q heads or eight full Q heads. Keep a one-Q-head variant for short contexts if the grouped form exposes too little parallelism. At long context, split the KV range and reduce so five full layers do not collapse to ten WGs. Select switch points by sweep.

Unique KV reads average 14.531 MB in standalone tg128, are 115.3 MB at depth
512, 230.7 MB at 1K, and 880.8 MB at 32K. The tg128 budget still uses **0.13
ms** because softmax/launch/low group count dominate its 0.016-ms byte floor.
A naive per-Q-head shader adds about 0.36 GB at 1K and 4.91 GB at 32K, moving
the 32K nominal roof from 294 tok/s toward 118 tok/s.

Gate: compare pre/post-RoPE Q/K, normalized V, cache writes, probabilities, and outputs at positions 0, 1, 1023, 1024, and 1025; at every full layer; and on both sides of each split threshold. Ring wrap must not change generated IDs.

### 2.6 Q6_K LM head and exact argmax

The `262144 x 2816` tied head reads 605.55 MB/token, **25.4%** of the weight floor. Exact greedy argmax cannot skip rows. A hierarchical vocabulary tree still reads every leaf unless it has certified upper bounds; an approximate shortlist fails qk's parity standard.

Tune the existing Q6_K path at M=262144, K=2816 (11 superblocks/row), then
apply required `30*tanh(logit/30)` over all logits before sampling or returning
them. Preserve llama.cpp's suppression and tie order. A fused candidate path may
reduce scratch, but it must retain softcap semantics for exposed logits.
Avoiding the 1-MiB materialized vector saves only 10-25 us, so it gets no
material B=1 credit. At the measured 800.4 GB/s, head weight time is 0.757 ms;
budget **0.78 ms**.

Cheaper output copies are quality experiments, not the default:

| Output path | Head bytes | Saved/token | Raw total | Ideal gain | Extra VRAM at max context |
|---|---:|---:|---:|---:|---:|
| Original Q6_K | 605.553 MB | - | 2.380 GB | - | none |
| Q5_K copy | 507.511 MB | 98.042 MB | 2.282 GB | 4.3% | 0.473 GiB; ~4.90 GiB remains |
| Q4_0 copy | 415.236 MB | 190.317 MB | 2.190 GB | 8.7% | 0.387 GiB; ~4.99 GiB remains |

The original Q6 tensor must remain for embedding lookup, so a repacked head is
additive VRAM. The softcap is monotone and therefore does not protect greedy
argmax from pre-cap requantization error. Q5/Q4 require separately named
quality tiers with perplexity/eval and token-divergence evidence; IQ4_XS would
also discard the QAT representation.

Batching is exact. If only the head is weight-stationary, its effective bytes/slot are `1.774503 GB + 0.605553/B`:

| Slots | Bytes/slot | Aggregate nominal roof | Gain over B=1 |
|---:|---:|---:|---:|
| 1 | 2.380 GB | 403.4 tok/s | — |
| 2 | 2.077 GB | 462.1 tok/s | 14.6% |
| 4 | 1.926 GB | 498.5 tok/s | 23.6% |
| 8 | 1.850 GB | 518.9 tok/s | 28.6% |
| 16 | 1.812 GB | 529.7 tok/s | 31.3% |

These are aggregate weight-only roofs and improve throughput, not single-request latency. Gate: fused and materialized logits produce identical token and tie choice, including suppressed IDs and near ties, for every slot.

### 2.7 Speculative/MTP decoding

The single-request mechanism that can amortize the head is the exact matching assistant, [`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant). It is an approximately 0.4B Gemma4Assistant intended to consume target hidden/KV state; it is not an independent small-model prefill path. The ordinary [`google/gemma-4-26B-A4B-it-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant) is a behavioral cross-check, but the QAT-matched assistant is the drafter to evaluate.

The accepted-token queue, batched target verification, per-position argmax,
prompt lookup, and parity cases in `docs/speculative-decoding.md` and
`docs/spec-decode-qk-plan.md` port. The assistant itself does not: qk needs a
Gemma assistant loader/ABI, export of the target final-norm hidden state, and
shared-target-KV handling. Gemma has no recurrent DeltaNet state, so reject and
rollback need only scratch KV positions/cursors.

A K-token target verifier reads the 1.577 GB non-routed target weights once per round. All three matrices for one expert across 30 layers cost 100.36 MB, so routed traffic is `100.36 MB*U(K)`. Reserve 0.8 GB/round for the assistant until its tensor ledger and kernels are measured:

| K | Expected expert union/layer | Target round bytes | Assistant allowance | Break-even committed tokens | Expected commits at provisional acceptance 0.918 |
|---:|---:|---:|---:|---:|---:|
| 2 | 15.5 | 3.13 GB | <=0.8 GB | >1.65 | 2.76 |
| 4 | 29.1 | 4.50 GB | <=0.8 GB | >2.23 | 4.25 |
| 8 | 51.6 | 6.76 GB | <=0.8 GB | >3.18 | 6.55 |

The 0.918 value is a sensitivity, not an assumption. Measure acceptance on qk
chat, prose, and coding fixtures. Start at K=4, then sweep 2/4/8. Do not publish
an emitted-tok/s prediction until assistant traffic, acceptance, and expert
union are measured. Stop if acceptance is below 75%, assistant time exceeds
30% of a round, or verification degenerates into K serial GEMVs.

Gate: forced all-reject, partial-accept, and all-accept cases emit exactly the same IDs as target-only greedy decode.

### 2.8 Existing qk kernel disposition

| Existing piece | Disposition for Gemma 4 |
|---|---|
| GGUF mapping/upload, descriptor views, pre-recorded command buffers, z-axis slots | Port unchanged as infrastructure; add a separate `gemma4` graph |
| `gemv_q6_k.comp`, `embed_q6k.comp` | Reuse after M/K/TPR validation; tied tensor is one allocation with two views |
| `rmsnorm.comp`, `vec_add.comp`, `argmax1/2.comp` | Primitive math ports; graph scheduling and mandatory softcap need variants |
| `gemv_q8_0.comp`, `gemv_iq4_xs.comp` | Templates/calibration references only; body weights need native Q4_0 |
| `gemm_q8_0.comp`, `gemm_iq4_xs.comp` | Reuse tile structure for new Q4_0 GEMM |
| `moe_select*`, `moe_gateup_*`, `moe_down_*` | New local128/top8 record and native-Q4 variants; retain ID indirection/grouping ideas |
| `fa_attn_srv*`, `fa_prep*` | Reuse online-softmax and `(acc,m,l)` reducer concepts; new SWA256 and full512 front ends |
| `add_rms3*` and Qwen attention output gate | Do not reuse semantically; Gemma norm/residual order differs |
| **all `dn_*.comp`** | **Dead: do not compile, bind, dispatch, or tune** |

### 2.9 Smaller siblings and local oracle

There is no smaller sparse A4B sibling. The fastest QAT Q4_0 bring-up targets
are [`google/gemma-4-E2B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
and [`google/gemma-4-E4B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf);
E4B is the better architecture smoke test. The 12B and 31B are dense
cross-checks. These siblings validate loader, Q4, norms, and parts of attention,
but not 128-expert routing, and none is the matching drafter.

Prefer target-shaped synthetic tensors before downloading a sibling. The existing `/mnt/data/archive/midnight/gemma-4/` BF16, Q8_0, and Q4_K_M 31B files are useful offline architecture oracles for names, metadata, normalization, and sliding/full behavior. They are dense and the wrong quantization for final parity.

## 3. Staged implementation plan and parity gates

### Stage 0 — freeze artifacts and measurements

1. Record the local GGUF SHA-256, exact size, tensor names/types/shapes, metadata, official repository revision, and vision tensors skipped. Reconcile any official/local byte difference; do not reject the valid local file merely because its filename differs.
2. Generate the exact tensor ledger above from GGUF metadata. An unexplained difference above 0.5% blocks implementation.
3. Measure Q4_0 bandwidth on both GPUs and run the frozen llama.cpp baseline below.
4. Freeze raw numeric input-ID fixtures: ordinary chat/coding, window positions 1023/1024/1025, full-attention contexts, and near-tie hidden states.

Gate: artifact ledger reconciles, both cards have repeatable bandwidth data,
the frozen llama.cpp comparator is recorded, and llama.cpp produces stable
reference IDs on two clean runs.

### Stage 1 — loader and standalone quant kernels

Add Gemma 4 tensor-role mapping, Q4_0 block recognition, and text-only skipping. Validate standalone Q4 GEMV/GEMM and Q6 head reduction; execute no model graph yet.

Gate: every tensor consumes exactly its declared range; CPU dequant, Vulkan GEMV/GEMM, and head argmax pass. Correctness precedes tuning.

### Stage 2 — dense Gemma graph primitives

Implement embedding scale, Gemma RMS variants, GELU, residual/output scales, suppression, tied head, and exact graph ordering on synthetic data and the local dense 31B oracle.

Gate: every primitive, branch boundary, and dense block matches the scalar/llama.cpp oracle; head argmax includes near ties.

### Stage 3 — sliding and full attention

Bring up the simple sliding ring first, then full dimension-512 attention, then GQA reuse and split-K variants.

Gate: Q/K/V, cache, probabilities, and hidden states match at positions 0/1/1023/1024/1025, all five full layers, and both sides of every kernel switch.

### Stage 4 — shared and routed MoE

Implement the 128-wide selector, 64-byte `SelT`, all-eight gate/up and down dispatches, shared GELU branch, merge, then grouped prefill GEMM.

Gate: exact top-8 IDs/weights and each branch tensor match; adversarial ties are stable; one complete sparse block passes before proceeding.

### Stage 5 — serial model parity

Assemble all 30 layers with separate ring/full cache state. Run one token, 128 tokens, window-boundary fixtures, and long contexts while retaining optional diagnostic tensor writes.

Gate: token-for-token identical greedy IDs against frozen llama.cpp runs at depth 0, 1K, 8K, and 32K, with at least 1,000 generated tokens cumulatively. Near-tie cases get diagnosis, not a parity waiver.

### Stage 6 — prefill and performance

Enable Q4 GEMM, grouped experts, attention tiling, and dimension specializations. Re-run pp128/512/2048, tg256 depth sweep, and parity after every material optimization.

Gate: prefill-to-decode IDs match pure stepwise decode; gains reproduce in three clean runs and neither card regresses over 5% without explanation.

### Stage 7 — exact multi-slot batching

Start with weight-stationary Q6 head, then extend small-B GEMM to projections/FFNs where timestamps pay. Preserve qk's z-axis slot convention and independent ring state.

Gate: B=1/2/4/8/16 slots each match isolated execution, including different ring-wrap positions. Report aggregate throughput and per-slot latency.

### Stage 8 — optional matching assistant

Only after serial parity/performance, ledger and quantize the QAT assistant, implement K=4 exact verification, and sweep K=2/4/8.

Gate: all rejection patterns reproduce target-only IDs; acceptance exceeds 75%; assistant is under 30% of round time; emitted throughput beats serial by at least 20%.

## 4. Exact llama.cpp baseline and parity harness

Reference: `/mnt/data/llama.cpp-master`, commit
`571d0d540df04f25298d0e159e520d9fc62ed121`. The exact supplied baseline
reproduction is:

```bash
GGML_VK_VISIBLE_DEVICES=2 \
  /mnt/data/llama.cpp-master/build/bin/llama-bench \
  -m /mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf \
  -ngl 99 -p 512 -n 128
```

The current decode reference (the context is cleared between depths) is:

```text
tg128 d0     = 137.73 +/- 0.92 tok/s
tg128 d4096  = 127.77 +/- 0.17 tok/s
tg128 d16384 = 120.76 +/- 0.20 tok/s
model  = 13.43 GiB / 25.23 B parameters
```

The 3505.04 tok/s pp512 result remains the historical prefill bar until a clean
current-revision prefill run replaces it; the recorded 571d0d5 pp samples were
started at 10% GPU busy and are not accepted as clean measurements.

For the expanded campaign, pin by PCI and verify that `Vulkan0` is the XTX:

```bash
export MODEL=/mnt/data/models/gemma-4-26B-A4B-qat/gemma-4-26B_q4_0-it.gguf
export BIN=/mnt/data/llama.cpp-master/build/bin
export DRI_PRIME='pci-0000_1a_00_0!'
git -C /mnt/data/llama.cpp-master rev-parse HEAD
"$BIN/llama-bench" --list-devices
```

Run seven warm repetitions, preserving JSONL. `llama-bench` at this commit uses `-fitt`, not `-fit off`; leave fitting off by default:

```bash
"$BIN/llama-bench" -m "$MODEL" -dev Vulkan0 -sm none -mg 0 -ngl 99 \
  -fa on -ctk f16 -ctv f16 -p 128,512,2048 -n 0 -b 8192 -ub 512 \
  -r 7 --delay 5 -o jsonl > llama-gemma4-xtx-pp.jsonl

"$BIN/llama-bench" -m "$MODEL" -dev Vulkan0 -sm none -mg 0 -ngl 99 \
  -fa on -ctk f16 -ctv f16 -p 0 -n 256 -d 0,1024,32768 \
  -b 8192 -ub 512 -r 7 --delay 5 -o jsonl > llama-gemma4-xtx-tg.jsonl

"$BIN/llama-bench" -m "$MODEL" -dev Vulkan0 -sm none -mg 0 -ngl 99 \
  -fa on -ctk f16 -ctv f16 -pg 512,256 -b 8192 -ub 512 \
  -r 7 --delay 5 -o jsonl > llama-gemma4-xtx-pg.jsonl
```

Measure independent and shared-prefix batching. `llama-batched-bench` does accept `-fit off`:

```bash
"$BIN/llama-batched-bench" -m "$MODEL" -dev Vulkan0 -sm none -mg 0 -ngl 99 \
  -fa on -ctk f16 -ctv f16 -fit off -c 32768 -b 8192 -ub 512 \
  -npp 512 -ntg 128 -npl 1,2,4,8,16 -tgs --output-format jsonl \
  > llama-gemma4-xtx-batched-independent.jsonl

"$BIN/llama-batched-bench" -m "$MODEL" -dev Vulkan0 -sm none -mg 0 -ngl 99 \
  -fa on -ctk f16 -ctv f16 -fit off -c 32768 -b 8192 -ub 512 \
  -npp 512 -ntg 128 -npl 1,2,4,8,16 -pps -tgs --output-format jsonl \
  > llama-gemma4-xtx-batched-shared.jsonl
```

Repeat with `DRI_PRIME='pci-0000_03_00_0!'` and XT output names. Record Mesa/RADV version, device UUID, clocks, wall temperature, peak VRAM, and median plus spread.

Readable greedy smoke test:

```bash
"$BIN/llama-cli" -m "$MODEL" -dev Vulkan0 -sm none -ngl 99 -fa on \
  -ctk f16 -ctv f16 -fit off -c 32768 -b 8192 -ub 512 \
  --temp 0 --seed 1 --no-display-prompt -f tests/gemma4-prompt.txt -n 128
```

Text output is not authoritative because tokenizer/template differences can hide model divergence. Tokenize once with llama.cpp and save the numeric IDs. Feed the exact same ID arrays to a single-slot `llama-server` `/completion` request and qk. Use greedy sampling, penalties disabled, `cache_prompt=false`, `ignore_eos=true`, and return token IDs. At each stage, teacher-force the same expected next token to find the first divergent layer; log top-2 values for diagnosis. The hard gate remains exact output IDs, with no near-tie exception.

## 5. Performance prediction and residual roofline gap

The measured comparator is llama.cpp 571d0d5 on this XTX: **137.73 +/- 0.92
tok/s for tg128 at depth zero**. The historical prefill bar is **3505.04
tok/s for pp512**. The decode result is 7.261 ms/token and only 327.8 GB/s of
unique active-weight traffic.

The XTX B=1 budget matches standalone tg128 (average depth 64.5):

| Work | Traffic/shape | Point time | Predicted contribution |
|---|---:|---:|---:|
| Q4 attention projections | 624.48 MB | 0.82 ms | large Q4 streams plus small full-layer shapes |
| Routed experts | 802.90 MB | 1.12 ms | combined eight-expert dispatches |
| Shared FFN | 301.09 MB | 0.43 ms | medium Q4 streams + GELU |
| f32 router/select | 43.25 MB | 0.16 ms | 30 small stages |
| Q6_K head/argmax | 605.55 MB | 0.78 ms | measured Q6 baseline + reduction |
| Attention core in tg128 | 14.531 MB average KV | 0.13 ms | latency/softmax bound; GQA reuse |
| Norms, residuals, cache writes | fragmented | 0.20 ms | launch-sensitive |
| Miscellaneous command cost | — | 0.06 ms | conservative remainder |
| **Total** | | **3.70 ms** | **270 tok/s** |

Prediction: **270 tok/s on XTX**, plausible range **250–290**; **222 tok/s on
XT**, range **210–235**. Against measured llama.cpp, the XTX point is +132.27
tok/s and **1.96x**; the range is **1.82-2.11x**. The acceptance target is at
least 1.3x the same-day llama median after exact parity. At 32K, KV traffic and
split reduction move XTX to roughly **200–225 tok/s**.

The measured-stream 920-GB/s roof is 386.5 tok/s (2.587 ms), so the central
residual to 3.704 ms is 1.117 ms:

| Residual source | Approx. ms |
|---|---:|
| Q4/Q6 format rates below raw stream | 0.20 |
| Tiny expert/dense granularity and tails | 0.22 |
| Attention beyond its 0.016-ms unique-byte floor | 0.11 |
| Router, norms, GELU, residuals, RoPE, softcap | 0.25 |
| Dispatch/barrier/DPM dependency gaps | 0.23 |
| Cache/unmodeled contingency | 0.11 |
| **Total** | **1.12** |

For pp512, do not publish a compute-only prediction before native Q4 GEMM is
measured; the acceptance gate is to exceed the measured 3505.04 tok/s while
preserving pp-to-tg token parity. Exact B=4 decode should move aggregate
throughput toward 400-500 tok/s as weights become stationary. MTP is excluded
from the 270 base claim; its speed is reported only after acceptance/union logs.

## 6. Risks and where the math may be wrong

1. **GGUF revision/type drift.** The local size differs slightly from a repository listing seen during research. SHA, revision, tensor ledger, and metadata are Stage 0 gates.
2. **Q4_0 bandwidth.** Eighteen-byte blocks may coalesce poorly; simpler dequant may not reach 850 GB/s. Compiler VGPRs and RADV changes matter.
3. **Full-attention semantics.** Dimension 512, two KV heads, raw K-as-V, separate K/V norms, partial RoPE factors, and scale 1.0 are all discontinuous correctness hazards.
4. **KV traffic.** The table assumes perfect reuse once per stored K/V. Cache lines, split rereads, and reductions add traffic; short sliding data may conversely benefit from cache.
5. **Expert routing.** `U(B)` assumes independent uniform routes. Real clustering changes both reuse and load balance. Log per-layer histograms and max/mean work.
6. **Occupancy model.** Wave count is not residency. VGPR/SGPR/LDS limits and a ~1.4 microsecond dispatch floor must be measured with pipeline timestamps.
7. **Router/top-k exactness.** Small f32 ordering differences can change an expert ID and amplify immediately. IDs/order are hard checks, not tolerance checks.
8. **Greedy near ties.** Vulkan and CPU reductions can flip top-1 despite small tensor error. Preserve a slow diagnostic order and exact ID gate.
9. **Q6 head bottleneck.** The delivered head is 46% larger than the prompt assumed. Its measured 800 GB/s may vary at the exact 262K-row geometry.
10. **Prefill estimate.** Standalone native Q4 GEMM is measured, but grouped-expert balance is not; use the 3505.04 llama result as the gate, not a guessed TFLOP/s.
11. **llama.cpp drift.** The measured 137.73 result is pinned to 571d0d5; rerun same-day before comparisons after future reference updates.
12. **Assistant economics.** Assistant traffic and acceptance are provisional. MTP is optional and abandoned if its explicit gates fail.
13. **Maximum-context memory.** Model plus KV leaves ~5.37 GiB before Vulkan overhead. Scratch, batching, duplicated tied weights, or an assistant can exhaust it.
14. **Sibling mismatch.** E4B/12B/31B passing does not validate sparse routing or the exact A4B attention mix.

Success remains qk's existing standard: identical greedy token IDs from identical numeric input IDs, followed by a reproducible throughput win. Performance before parity is diagnostic only.
