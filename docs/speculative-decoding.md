# Speculative Decoding for Qwen3.6-35B-A3B on RX 7900 XT / llama.cpp Vulkan

Scope: a practical, quantitative assessment of speculative decoding for **this specific rig** —
`qwen35moe` (Qwen3.6-35B-A3B, hybrid gated-DeltaNet + attention MoE) served by **llama.cpp build
`b8672` (tagged 2026-04-06)**, Vulkan/RADV backend, single AMD RX 7900 XT (20 GB, ~800 GB/s),
256K context, KV cache `q8_0`, in k8s. Also covers implications for the custom Vulkan GEMV kernels
in `/home/ryan/IdeaProjects/qwen-kernel`.

Ground-truth facts below tagged **[local]** were verified directly against the checkout at
`/home/ryan/intellij/ggerganov/llama.cpp` (file:line cited) and the two model GGUFs — these are
authoritative for your binary. Facts tagged **[web]** come from external sources (URLs inline);
where a number is post-cutoff, vendor-only, or single-source it is marked **[FLAG]** and should be
treated as directional.

---

## TL;DR — bottom line and the one next action

1. **On your current build (`b8672`), speculative decoding is silently DISABLED for this model.**
   `qwen35moe` is a hybrid recurrent (gated-DeltaNet) model. Its DeltaNet state cannot be partially
   rolled back, so llama.cpp's compatibility probe fails and `llama-server` logs
   *"speculative decoding not supported by this context"* and never initializes it — for **both**
   draft-model and n-gram modes. **[local]** `common/speculative.cpp:801-835`,
   `tools/server/server-context.cpp:767-794`.

2. **The `Qwen3.6-27B-Q5_K_M-mtp.gguf` file cannot accelerate your A3B.** It is a *different model*:
   a **dense** `qwen35` (n_embd **5120**, 64+1 layers), not the MoE (`qwen35moe`, n_embd **2048**).
   Its MTP/NextN head is dimensionally incompatible with the A3B and, on `b8672`, MTP is not executed
   at all (weights load into `llama_layer_nextn` but no graph consumes them; they're even flagged
   `TENSOR_SKIP`). **[local]** `src/llama-model.cpp:5441-5443`, GGUF metadata. It shares your A3B's
   248,320-token vocab, but as a 27B *dense* draft it is far **slower** than the 3B-active target, so
   it's useless as a draft too.

3. **Even after you fix #1, the economics are marginal.** Your target is a 3B-active MoE: decode is
   already memory-bandwidth-bound and cheap per token. Speculation's payoff scales as
   `accepted_tokens / (γ·c + verify_cost)`, and for an A3B the **verify pass over N draft tokens
   streams the *union* of experts across those tokens** — several× the weight traffic of a single
   decode — so realized speedup collapses toward (or below) 1× at the small draft lengths (2–8) that
   fit. Community llama.cpp benchmarks of exactly this model show **net slowdowns** for external-draft
   speculation even at 100% acceptance. **[web/FLAG]**

4. **The kernel consequence for `qwen-kernel`:** the moment you verify N≥2 draft tokens, ggml's Vulkan
   backend switches **every** weight from the GEMV path (`mul_mat_vec_id`, what your M1–M3 kernels
   replicate) to the tiled `matmul_id` coopmat/MMQ **GEMM** path. **[local]** `ggml-vulkan.cpp:1772-1773`.
   Routed-expert weights barely amortize at N=2–8 (~1 token/expert); the *dense/always-on* weights
   (shared expert, attention/DeltaNet, router, LM head) amortize fully. That split is exactly where a
   custom small-batch kernel would earn its keep.

### Single recommended next action

**Upgrade llama.cpp to a build after 2026-04-19 (includes PR #19493 "speculative checkpointing"), then
test *draftless* n-gram speculation on your code/reasoning workloads only:**

```
llama-server ... \
  --spec-type ngram-mod --spec-use-checkpoints on \
  --spec-ngram-size-n 24 --draft-min 48 --draft-max 64 --ctx-checkpoints 8
```

This needs **zero extra VRAM**, has **no vocab/draft-model dependency**, and is the mode PR #19493
specifically showed helping (~30% on repetitive code generation). Expect **~0–30% faster on repetitive
code/reasoning-repeat, roughly neutral on prose, and never load an external draft model** on this rig
(wrong vocab, no VRAM headroom, and the A3B target is too cheap to beat). Do **not** expect the dense
27B-mtp file to help. If you later want the strongest option — self-speculation via the model's own MTP
head — you need *both* a build ≥ b9190 (PR #22673, MTP merged 2026-05-16) *and* a **re-converted A3B GGUF
with `--mtp` kept**, and you must first clear the RDNA3/RADV MTP correctness bug (#23088). See §4.

---

## 1. State of the art (2025–2026)

### The governing formula

All (correctly-implemented) speculative methods are **lossless** — modified rejection sampling
reproduces the target's exact output distribution — with the notable exception of Medusa's "typical
acceptance." **[web]** Leviathan et al. 2023 give the controlling equations
(https://ar5iv.labs.arxiv.org/html/2211.17192):

- Expected tokens per iteration: **E = (1 − α^(γ+1)) / (1 − α)**
- Wall-clock speedup: **(1 − α^(γ+1)) / [(1 − α)(γ·c + 1)]**

where **α** = per-token acceptance rate, **γ** = draft tokens proposed per round, **c** = cost of one
draft call ÷ one target call. The two levers are **acceptance × draft-cheapness**; speedup saturates in
γ and is dragged down by draft cost c. A 2025 result (NAACL, "Decoding Speculative Decoding") stresses
that a *faster, lower-α* draft can beat a *slower, higher-α* one — optimize end-to-end latency, not α
(https://aclanthology.org/2025.naacl-long.328.pdf). Crucially, **all these gains assume a
memory-bound (small-batch / low-QPS) regime**; they shrink or invert once decoding becomes
compute-bound. This is the seed of the A3B problem in §4.

### Methods and measured speedups

| Family | Mechanism | Reported speedup (greedy, batch 1) | Notes / restrictions |
|---|---|---|---|
| **Classic draft model** (Leviathan '23; Chen '23) | small same-family model drafts, target verifies | **2–3.5×** (T5-XXL 3.4×; Chinchilla-70B 2–2.5×; TRT-LLM Llama-3.3-70B+1B **3.55×**) | needs same tokenizer/vocab; +1 resident draft model. Chen: HumanEval **2.46×** > XSum 1.92×. https://arxiv.org/abs/2302.01318 |
| **Medusa** (2024) | extra trained heads + tree attention, no 2nd model | **2.2–3.6×** (peak 3.62×) | **non-lossless** ("typical acceptance"). https://arxiv.org/abs/2401.10774 |
| **EAGLE-1** (2024) | autoregress at hidden-**feature** level | **2.7–3.5×**, τ≈3.6–4.5 | lossless; trained head per model. https://arxiv.org/abs/2401.15077 |
| **EAGLE-2** (2024) | + dynamic context-aware draft **trees** | **3.05–4.26×** (+20–40% over v1), up to 4.96× HumanEval | https://arxiv.org/abs/2406.16858 |
| **EAGLE-3** (2025) | multi-layer feature fusion + training-time test | **up to 6.5×** (τ **7.54** on code), mean ~4–5.5× | acceptance keeps scaling with training data. https://arxiv.org/abs/2503.01840 |
| **MTP heads** (DeepSeek-V3; Qwen3-Next/3.5/3.6) | model's own next-token-prediction module self-drafts | **~1.8×** (DeepSeek-V3, D=1, **85–90%** 2nd-token accept) | ships with the checkpoint; no separate vocab. τ≈2.2–2.7 (< EAGLE). https://arxiv.org/html/2412.19437v1 |
| **Prompt-lookup / n-gram** | copy tokens after an n-gram match in context | **2–4×** on edit/summarize/RAG; ~1.6× chat | zero params; needs literal input↔output overlap. https://github.com/apoorvumang/prompt-lookup-decoding |
| **Lookahead** (Jacobi) | parallel n-gram guess-and-verify | 1.5–2.3× (up to 4× on code, multi-GPU) | no draft model. https://arxiv.org/abs/2402.02057 |
| **Token trees** (SpecInfer, Sequoia, SpecExec) | verify many candidate continuations in one pass | SpecInfer 1.5–2.8×; Sequoia up to 4× on-chip, **~10–20× offloaded** | trees raise E[accepted/pass]; extra FLOPs "free" only when memory-bound. https://arxiv.org/abs/2305.09781, https://arxiv.org/abs/2402.12374 |

**MTP vs EAGLE (relevant to this model):** EAGLE reaches higher mean accepted length (τ≈5–7) than a
native MTP head (τ≈2.2–2.7), because MTP is a shallow depth-1 module recursively unrolled off its
training distribution, whereas EAGLE trains a dedicated multi-step drafter with dynamic trees. But
**MTP ships free with the base checkpoint; EAGLE needs a separately-trained head per model.** **[web]**
https://www.lmsys.org/blog/2025-07-17-mtp/

### Code vs prose (acceptance gap)

Robust, multi-source: **code ≥ structured/JSON ≥ prose**, in both acceptance and speedup — code has
predictable boilerplate, brackets, indentation, and repeated identifiers the drafter anticipates.
Chen 2023 (same draft/target): HumanEval **2.46×** vs XSum 1.92×. EAGLE-3: code is the highest-τ task
for every model (Vicuna-13B τ 7.54 code vs 6.47 summarization). n-gram methods show the widest spread
(code-edit ~7 accepted tok/step vs open-ended chat ~1.4). Practitioner ranges (single-source **[FLAG]**):
code 30–60%, chat 20–40%, structured 50–70%, repetitive edit/summarize 80%+.

---

## 2. llama.cpp support status — your `b8672` vs current master

**Timeline anchor [web, GitHub releases API + local git]:** your `b8672` is **2026-04-06**; today's
master is ~`b9873` (2026-07-04). Several headline features merged **after** your checkout, so most
current docs/blogs describe a codebase you don't have:

| Feature | Merged | In `b8672`? |
|---|---|---|
| Draft-model + n-gram speculative (server, HTTP) | 2024-11 (#10455), ngram-mod #19164 (2026-01-30) | **Yes** |
| **Speculative checkpointing for recurrent/hybrid** (PR #19493) | **2026-04-19** | **No** (13 days too early) |
| **MTP** self-speculation (PR #22673, `--spec-type draft-mtp`) | **2026-05-16** (~b9190) | **No** |
| **EAGLE-3** (`draft-eagle3`, #18039) | 2026-06-12 | **No** (enum exists but stubbed: `has_draft_eagle3 = false; // TODO PR-18039`) **[local]** `common/speculative.cpp:855` |
| DFlash (#22105) | 2026-06-28 | **No** |

### Speculative CLI flags present in `b8672` [local] `common/arg.cpp:3423-3568`

- `-md, --model-draft FNAME` — draft model (default: unused)
- `--draft, --draft-n, --draft-max N` — draft length, **default 16**
- `--draft-min, --draft-n-min N` — **default 0**
- `--draft-p-min P` — greedy accept threshold, **default 0.75**; `--draft-p-split P` (0.10)
- `-cd, --ctx-size-draft N`; `-devd, --device-draft`; `-ngld, --gpu-layers-draft` (auto/all);
  `-ctkd/-ctvd, --cache-type-k/v-draft`; `-otd, --override-tensor-draft`;
  `-cmoed/--cpu-moe-draft`, `-ncmoed/--n-cpu-moe-draft`
- `--spec-replace TARGET DRAFT` — *"translate the string in TARGET into DRAFT if the draft model and
  main model are not compatible"* — a **string-level** reconciliation for near-compatible vocabs (see §4)
- Draftless: `--spec-type [none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]`, with
  `--spec-ngram-size-n` (12), `--spec-ngram-size-m` (48), `--spec-ngram-min-hits` (1);
  static/dynamic lookup caches `-lcs/-lcd`. **[local]** + `docs/speculative.md`

Note: current master renamed all of these under a `--spec-draft-*` namespace and **changed
`--draft-max`'s default 16 → 3**; if you upgrade, your existing flags will error and need porting.

### Draft/target compatibility check [local] `common/speculative.cpp:50-105`

For the plain draft path, `common_speculative_are_compatible()` requires: (1) identical vocab **type**;
(2) matching `add_bos`/`add_eos` and BOS/EOS ids; (3) **vocab-size difference ≤
`SPEC_VOCAB_MAX_SIZE_DIFFERENCE = 128`**; (4) identical **token text** for ids 5…min(n_vocab). If the
vocabs are *not* compatible, `b8672` does **not** hard-fail — it sets `vocab_cmpt = false` and
**translates tokens by detokenizing the target token to text, applying `--spec-replace`, and
retokenizing into the draft vocab** (`speculative.cpp:198-268`). Useful but lossy/slower. (EAGLE3/MTP
in later builds *bypass* this — they share the target tokenizer and instead assert e.g. MTP
`n_embd_out == target hidden width`.)

### MTP status [local + web]

- **`b8672`: MTP is inert.** The spec enum has no `draft-mtp`; the converter drops/ignores MTP weights
  for Qwen (`convert_hf_to_gguf.py:4748` `if name.startswith("mtp"): return  # ignore MTP layers for
  now`) and DeepSeek (`skip_mtp = True`, "TODO … when we support MTP for deepseek"). GLM/Gemma keep
  NextN tensors as inert metadata. Nothing runs them.
- **Master (≥ b9190): MTP works for `qwen35`/`qwen35moe`, `step35`, `gemma4` only** (runtime modes in
  `common_speculative_impl_draft_mtp`), used via `--spec-type draft-mtp`; the refactored converter adds
  `--mtp/--no-mtp` (default keeps the head) "only supported for Qwen3.5/3.6 and Step3.5." **Not**
  supported: DeepSeek-V3/R1, GLM-4.5/4.6 (PR #15225 closed unmerged; open crash #24309), Qwen3-Next
  (PR #20533 closed unmerged). **[web]** PR #22673 (merged 2026-05-16).

### Per-request control & Vulkan/RADV caveats

- **Per-request speculative params (`speculative.n_max`, etc.) are NOT settable over the HTTP API** in
  either `b8672` or master — draft config is startup-CLI only (the request-schema fields are
  `#if 0`-disabled and echoed read-only). **[web]** `tools/server/server-schema.cpp`.
- **Vulkan/RDNA3 (your exact backend) has multiple relevant bugs [web]:**
  - **#23088** — MTP/NextN on RDNA3 + RADV produced **garbage (~0.7% accept vs ~81% on HIP/ROCm)**;
    closed "completed" (verify against whatever build you land on).
  - **#23126** — draft + target on the **same Vulkan device** serialize on one compute queue → draft
    per-token time explodes, no speedup. Mitigation: keep the draft off the Vulkan queue (`-ngld 0` /
    `-devd none`, i.e. CPU draft).
  - **#23430 (open)** — "core dumped running qwen 3.6 27b on vulkan"; **#23199 (open)** — Vulkan device
    selection falls back to ROCm for MTP.
  - These postdate `b8672` and mostly concern MTP (which `b8672` lacks), but the same-queue
    serialization is architectural and would also bite a draft+target-on-Vulkan setup.

---

## 3. The hybrid-architecture wrinkle: rolling back DeltaNet state

### Why it's hard

Speculative decoding must revert per-sequence state when draft tokens are rejected. For **softmax
attention this is O(1)** — the KV entries for rejected positions are simply discarded (nothing was
mutated). For **recurrent / SSM / linear-attention layers the state is a single fixed-size tensor
updated in place** each token, so there's nothing to "truncate": once you advance through N speculated
tokens, the state that produced the last *accepted* token is gone. **[web]** ("Snakes and Ladders,"
Wu et al. 2024, https://proceedings.mlr.press/v262/wu24a.html).

Your model makes this concrete. `qwen35moe` is built as `llm_build_qwen35moe : llm_build_delta_net_base`
and uses **`llama_memory_hybrid`** — a `llama_kv_cache` for the full-attention layers plus a
`llama_memory_recurrent` for the **gated-DeltaNet** layers. **[local]** `src/models/models.h:615`,
`src/llama-model.cpp:8289-8335`. Gated DeltaNet's state is **matrix-valued per head** (Yang/Kautz,
"Gated Delta Networks," https://arxiv.org/abs/2412.06464): `S_t = α_t(I − β_t k_t k_tᵀ)S_{t−1} +
β_t k_t v_tᵀ` — an erase-then-write on a d×d matrix (your GGUF carries `ssm_a/alpha/beta/conv1d/dt/
norm/out` per DeltaNet block; `SSM_INNER_SIZE = 64`). Snapshotting that matrix per speculated token is
memory-bandwidth-heavy, far more than Mamba's vector state. **[local]** GGUF `blk.0.ssm_*`.

### What `b8672` actually does: it refuses

`llama_memory_recurrent::seq_rm()` **returns `false` for any partial removal** — *"models like Mamba or
RWKV can't have a state partially erased at the end."* **[local]** `src/llama-memory-recurrent.cpp:154-167`.
`llama_memory_hybrid::seq_rm()` calls the recurrent one first and inherits that `false`. The speculative
subsystem probes for this at init: `common_speculative_is_compat()` decodes 2 tokens and tries
`llama_memory_seq_rm(mem, 0, 1, -1)`; the failure triggers *"the target context does not support partial
sequence removal"* → returns `false`. **[local]** `common/speculative.cpp:801-835`. The server uses that
as a hard gate: `can_spec = common_speculative_is_compat(ctx)`; if false it logs *"speculative decoding
not supported by this context"* and **never creates `slot.spec`** — disabling **all** speculative modes
(draft *and* n-gram). **[local]** `tools/server/server-context.cpp:767-794`. So on `b8672`, this model
gets no speculation of any kind.

### The fix (not in your build): checkpoint + re-decode

PR **#19493 "server: speculative checkpointing"** (merged **2026-04-19**, ~2 weeks after `b8672`) adds
**full context-state checkpoint/restore** for recurrent/hybrid targets: it snapshots the KV+recurrent
state before drafting and restores to the last accepted position on rejection, enabled by
`--spec-use-checkpoints on` (reusing the general `--ctx-checkpoints` slots from PR #15293, which *do*
exist in `b8672` for context management but aren't wired into speculation). It is explicitly slower than
`seq_rm`: *"in case of a partially accepted draft, we have to go back to the checkpoint and execute a
shorter batch"* — i.e. snapshot **+ partial recompute**. It reports **~30% on repetitive code**
(quicksort). **[web]** https://github.com/ggml-org/llama.cpp/pull/19493. (These commits are already in
your local object DB — `455d8e4be`, `bcb5eeb64` — but not on the `b8672` tag; a newer checkout gets them.)

### How the broader ecosystem handles it (design menu) [web]

- **Snapshot + restore** (llama.cpp #19493): copy whole KV+SSM/conv state, cheap restore, short
  re-decode on partial accept. Simplest; bandwidth-bound for DeltaNet's matrix state.
- **Activation replay / recompute** (Snakes-and-Ladders; STree, https://arxiv.org/abs/2505.14969; vLLM
  "ReplaySSM" proposal): keep only cheap inputs in a ring buffer, recompute state from the last accepted
  position. Low memory, scales to trees; research-preferred.
- **Per-candidate sandbox slots** (SGLang): give each speculated branch its own state cell; on accept
  promote it, on reject discard — *no rollback at all*.
- vLLM notes MTP is **safer** than n-gram here because MTP draft tokens are produced *after* the base
  model step, so they don't pre-evolve the shared state (issue #39273). This is a hint that MTP is the
  cleanest speculation style for hybrids.

---

## 4. Practical recommendation for this rig

### The two GGUFs, decided [local]

| | `Qwen3.6-35B-A3B-UD-Q3_K_M` (target) | `Qwen3.6-27B-Q5_K_M-mtp` |
|---|---|---|
| arch | **qwen35moe** (hybrid MoE) | **qwen35** (dense) |
| layers / n_embd | 40 / **2048** | 64+1 NextN / **5120** |
| FFN | 256 experts (top-8) + 1 shared, expert FFN 512 | dense FFN 17408 |
| MTP head | **none** (dropped at convert) | **present** (`nextn_predict_layers=1`, `blk.64.nextn.*`) |
| vocab | 248,320 | 248,320 (same tokenizer) |

- **The 27B-mtp is not usable to speed up your A3B.** Different architecture and hidden width (5120 vs
  2048) → its MTP head cannot attach to the MoE. As a *plain draft* it shares the vocab but is a **27B
  dense** model — dramatically slower than a 3B-active target (c ≫ 1, so speedup < 1 by the Leviathan
  formula) and it alone is ~19.7 GB, exceeding your remaining VRAM. It is only meaningful as *its own*
  model, and only on a build that executes MTP (≥ b9190) with the RDNA3 correctness bug cleared.
- **Your A3B GGUF has no MTP head** (converted on `b8672`, which drops Qwen MTP layers). Self-speculation
  via MTP would require a **re-converted GGUF with `--mtp` kept** *and* a build ≥ b9190.

### External draft models: don't

- **Vocab wall:** Qwen3 0.6B/1.7B/4B use vocab **151,936**; your target is **248,320** (Qwen3.6 expanded
  it, per web the multimodal/multilingual token additions **[web/FLAG]**). That 96k gap is far beyond
  llama.cpp's ±128 tolerance; `b8672` would fall back to lossy string-translation, and it's moot anyway
  because §3 disables speculation for this target. A vocab-matched small draft would have to be a
  Qwen3.5/3.6-family model at 248,320, or a `transplant-vocab` graft of a 0.6B onto that vocab.
- **VRAM:** the target is ~16.6 GB at Q3_K_M; with 256K context and `q8_0` KV you are near full on 20 GB.
  A resident draft forces a smaller quant or CPU-offload of target layers → a net loss *before*
  speculation runs.
- **The A3B economics (the core reason):** realized speedup ≈
  `a / ( γ·(T_draft/T_target) + T_verify(N)/T_target )`. On a 3B-active MoE, `T_target` is tiny
  (memory-bound, ~3B weights streamed), so (a) a 0.6–0.8B draft's `T_draft/T_target` is a large fraction,
  and (b) `T_verify(N)/T_target ≫ 1` because verifying N tokens activates the **union of top-8 experts
  across all N tokens**, streaming several× the expert weight of one decode (see §5). Both denominator
  terms blow up → net negative at feasible draft lengths.
- **Empirical corroboration [web/FLAG — single practitioner, consumer GPU]:** llama.cpp on
  Qwen3.6-35B-A3B (RTX 3090) measured baseline ~135.7 tok/s with **every** spec mode slower even at 100%
  acceptance (ngram-mod −3.4%, external draft w/ 0.8B −11%, aggressive −39…−54%), attributed to an
  "expert-saturation" point of ~90+ tokens/pass — far beyond any 2–8 draft length. Literature agrees the
  *relative* gain shrinks as active-params shrink (Qwen3-30B-A3B ~1.9–2.4× only *with* a learned EAGLE-3/
  MoE-Spec head on datacenter GPUs; smaller-active OLMoE lower still).

### What to actually run

1. **Upgrade past 2026-04-19** (PR #19493) so hybrid speculation is even permitted. Prefer ≥ b9190 if you
   also want to experiment with MTP — but first validate output correctness on Vulkan (bug #23088).
2. **Use draftless n-gram, checkpoint-enabled** (the TL;DR command): `--spec-type ngram-mod
   --spec-use-checkpoints on --spec-ngram-size-n 24 --draft-min 48 --draft-max 64`. The docs note
   *"MoEs require long drafts"* — hence the long m-gram. Zero extra VRAM, no vocab issue. Best on
   repetitive code, refactoring/rewrite loops (llama.vim-style), and reasoning models that repeat their
   scratchpad in the final answer; roughly neutral on prose.
3. **Strongest ceiling, most work: self-speculation MTP.** Build ≥ b9190 + re-convert the A3B with
   `--mtp` + clear #23088 on RADV. MTP is the cleanest style for a hybrid (draft tokens produced after
   the base step; no state pre-evolution) and community numbers for Qwen3.6 MTP are ~2.2–2.5×
   single-stream **[web/FLAG, community not vendor]** — but every one of those preconditions is a real
   gate on this rig, and MTP wins at batch-1 while *degrading* under concurrency, which matters for a
   shared k8s server.

**Acceptance expectations for you:** highest on code/structured output and on reasoning-repeat (n-gram
can exceed 70–80% accepted on verbatim-repeat spans); modest on chat; near-zero benefit on open-ended
prose. Given the A3B verification tax, treat "faster on code, neutral elsewhere" as the realistic
outcome — not the 2–3× headline figures from dense-model benchmarks.

---

## 5. Kernel implications for `qwen-kernel` (batched verification)

### The compute-shape shift: GEMV → grouped GEMM

At normal decode (batch 1) each activated expert does a matrix-**vector** product — a bandwidth-bound
GEMV dominated by streaming that expert's weights. This is exactly what your M1–M3 kernels
(`gemv_q8_0`, `gemv_q6_k`, `gemv_iq4_xs`, `gemv_iq3_xxs`, `gemv_f16`) and `moe_router.comp` implement
(single-token: `ids[8]`, `w[8]`, `wShared`). Speculative **verification processes N draft tokens in one
forward pass** (N typically 2–8), turning each expert's work into a **skinny grouped/segmented GEMM**:
tokens are routed independently, so the activated-expert set is the **union across the N tokens**, and
each expert multiplies only the *subset* of the N tokens routed to it (token-permutation + batched-GEMM-
per-expert, the Megablocks/grouped-GEMM pattern). **[web]** https://arxiv.org/html/2506.20675v1.

### Weight-load amortization at N=2–8 (why routed experts barely help)

With 256 experts and top-8, at these tiny N the union is almost the full sum — collisions are rare, so
routed-expert weights are streamed almost as many times as at N=1 (assuming ~uniform routing) **[web/FLAG
— uniform-routing calculation; real routers cluster somewhat, so slightly better]**:

| N draft tokens | distinct experts touched | tokens per activated expert | per-token expert-weight traffic vs N=1 |
|---|---|---|---|
| 1 | 8 | 1.00 | 1.00× |
| 2 | ~15.8 | ~1.02 | ~0.98× |
| 4 | ~30.5 | ~1.05 | ~0.95× |
| 8 | ~57.4 | ~1.11 | ~0.90× |

So even at N=8 you load ~57 experts to make 8 tokens (~7.1 experts/token vs 8) — only ~10% less
expert-bandwidth per token. **Routed-expert weights essentially do not amortize** at draft lengths.

**What *does* amortize fully** (used by *all* N tokens, so weight streamed once, GEMV→N-column GEMM,
up to N× arithmetic intensity): the **shared expert** (`ffn_*_shexp`, q8_0, always-on), the
**attention / gated-DeltaNet** weights (`attn_qkv` 2048→8192, `ssm_*`, `ssm_out`), the **router**
(`ffn_gate_inp`, f32), token embeddings, and the **LM head** (`output.weight`, q6_k, 248320 rows).
That is where the verify pass actually buys compute density on an A3B.

### ggml's Vulkan dispatch — the exact threshold [local] `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

- `mul_mat_vec_max_cols = 8` (line 260) — dense `MUL_MAT` uses the vector path for n ≤ 8.
- **MoE (`MUL_MAT_ID`) uses the GEMV path only when `src[2]->ne[1] == 1`** (ids has n_tokens == 1) —
  lines 1772-1773. Any **N ≥ 2** verification batch takes the **tiled `matmul_id`** path
  (`ggml_vk_mul_mat_id` → `ggml_vk_use_mul_mat_vec_id`, line 8730-8738): coopmat/coopmat2 or the
  integer-dot **MMQ** (`q8_1`) shaders — a completely different kernel from your `mul_mat_vec_id` GEMVs.
- Practically: your GEMV kernels are the *batch-1 decode* kernels; a spec-decode verify pass would run
  the coopmat/MMQ `matmul_id` GEMM for **every** expert and dense weight. Known issue: the coopmat tiles
  are sized for many-rows-per-expert, so at ~1 token/expert they're mostly padded — AMD KHR_coopmat MoE
  tuning (64×64 → 128×32 tiles) gave **+7.6–10.5%** prompt-processing on MoE (discussion #22598) **[web]**,
  evidence the default tiling is poorly matched to these skinny shapes.

### Guidance if you extend M4/M5 for verification

- **The batch sizes that matter are N = 2–8, with ~1–2 columns per routed expert.** Do **not** design
  around big per-expert tiles or Megablocks-style 128-row padding — that wastes bandwidth on near-empty
  blocks. Use a **weight-stationary, variable/segmented** inner loop: stream each expert's weight tile
  **once**, accumulate the handful of assigned token columns, then evict. Effectively an N-wide GEMV that
  reuses each weight load across a small variable RHS (PyTorch's persistent grouped-GEMM shows the L2-hit
  win from exactly this ordering). **[web]**
- **Make the always-on dense weights a proper N-column GEMM.** The shared expert, `attn_qkv`, DeltaNet
  and LM head are the parts that amortize N×; capturing their density is where the verify pass gets
  cheaper-per-token. Your M4 (fused expert FFN) and M5 (block megakernel) should treat "routed experts =
  skinny weight-stationary" and "dense/shared/attention/LM-head = tiled GEMM" as two distinct regimes.
- **Router (`moe_router.comp`) needs an N-token variant**: compute N independent top-8 selections, then a
  scatter/gather (argsort of flattened expert assignments) to group tokens per expert — the permutation
  that feeds grouped-GEMM.
- **DeltaNet layers in a verify pass** must advance the recurrent state through all N tokens (the chunked
  scan `build_delta_net_chunking`/`_fused` already does this for prefill) **and** be checkpointed for
  rollback (§3). Your M5 megakernel is the natural place to fuse the N-token chunked DeltaNet scan with a
  state snapshot. Note this only becomes relevant on a build where hybrid speculation is enabled at all
  (post-#19493).
- **Reality check:** because routed-expert weights don't amortize below ~90 tokens/pass, at N=2–8 the win
  from a custom verify kernel comes almost entirely from the dense path and from *not regressing* the
  expert path vs the batch-1 GEMV — consistent with the measured net-loss of naive spec-decode on this
  model. The kernel work is more valuable for **prompt processing / large-batch prefill** (where N is
  genuinely large) than for 2–8-token speculative verification.

---

## Sources

Primary (local, authoritative for your binary): `common/speculative.cpp` (`:50-105` compat, `:198-268`
vocab translation, `:801-835` `is_compat`, `:855` EAGLE3 stub); `tools/server/server-context.cpp:767-794`;
`common/arg.cpp:3423-3568`; `src/llama-memory-recurrent.cpp:154-167`; `src/llama-model.cpp:8289-8335`,
`:5441-5443`; `src/models/models.h:615`; `ggml/src/ggml-vulkan/ggml-vulkan.cpp:260,1772-1773,8730-8738`;
`convert_hf_to_gguf.py:4748`; `docs/speculative.md`; GGUF metadata of both model files.

External (URLs inline above). Load-bearing: Leviathan 2023 (2211.17192), Chen 2023 (2302.01318),
EAGLE-1/2/3 (2401.15077 / 2406.16858 / 2503.01840), DeepSeek-V3 MTP (2412.19437), Gated DeltaNet
(2412.06464), Snakes-and-Ladders (PMLR v262), STree (2505.14969), Cascade/MoE-verification (2506.20675),
llama.cpp PRs #19493 / #22673 / #15293 and Vulkan issues #23088 / #23126 / #23430.

**Uncertainty flags:** Qwen3.6-specific and post-2026-01 items (Qwen3.6 MTP acceptance/speedup numbers,
EAGLE-3.1, DSpark, the RTX-3090 A3B benchmark, the ~90-token saturation figure) are community/vendor or
single-source and post-date verifiable literature — treat as directional. The per-expert amortization
table assumes uniform routing (real routers cluster, so slightly more favorable). All llama.cpp code
facts and the b8672-vs-master timeline are verified against source.
