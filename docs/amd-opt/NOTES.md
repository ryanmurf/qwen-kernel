# AMD RDNA3 optimization experiment log

Append-only working log for the mission in `BRIEF.md`. Times are America/Denver.
Every GPU command in this log is explicitly pinned with `QK_DEVICE_PCI`.

## 2026-07-13

### 06:25 — Mission intake and immutable baseline

- Created branch `amd-opt` from `2663cac` (`Engine: split-K for the batch-path n==1 (decode) attention case`). `main` was seven local commits ahead of `origin/main`; no tracked worktree changes existed. `docs/amd-opt/BRIEF.md` was the only untracked input and is preserved.
- Read `/home/ryan/.codex/RTK.md` and `docs/amd-opt/BRIEF.md`. Hard rails adopted: PCI pin every GPU run; never use `qk token`; XTX DPM remains `auto`; XT manual pin remains unchanged; XTX deployment must be restored; correctness requires byte-identical `GEN:` streams on `big_a.ids`/256 and `big4.ids`/64 plus the built-in suite.
- Inputs verified read-only:
  - 35B GGUF: `/home/ryan/intellij/ggerganov/llama.cpp/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`, 15,831.7 MiB as reported by `ls`.
  - 80B GGUF: `/mnt/data/models/Qwen3-Next-80B-A3B-Instruct-IQ4_XS-qk.gguf`, 40,869.6 MiB as reported by `ls`.
  - `big_a.ids`: 5.7 KiB, mission states 1,213 tokens. `big4.ids`: 22.7 KiB, mission states 4,852 tokens.
- Hardware readback (no state changes):
  - `03:00.0` maps to `/sys/class/drm/card1`, 21,458,059,264 bytes VRAM total, 2,879,836,160 bytes used at intake, `power_dpm_force_performance_level=manual`, available sclk states 500/2025 MHz (intake readback displayed the 0 MHz idle marker).
  - `1a:00.0` maps to `/sys/class/drm/card2`, 25,753,026,560 bytes VRAM total, 28,004,352 bytes used at intake, `power_dpm_force_performance_level=auto`, available sclk states 500/2371 MHz.
  - `qk-server-split-80b` deployment was `1/1` and must be restored to one replica after the exclusive-XTX campaign.
- Parallel read-only audits started for roofline/model accounting, tuning candidates, and correctness/benchmark harness design. GPU work remains centrally serialized.

### 06:30 — Untouched 35B / 7900 XT correctness and profiler baseline

- Rebuilt commit `2663cac` with `/usr/bin/cmake --build build -j`: success (`qk` SHA-256 `6c2b9625701f83b88710ddf1a1cc93609cb40ad1575a6ae4ab76de34120f3d13`, `libqk.so` SHA-256 `db636c33fb83084675307bd759c471728eb6a59cc0c3805787a4dce9b264ea3f`). `git diff --check` passed.
- The first `serve-test` attempt used the brief's absolute scratchpad filename and failed before Vulkan initialization with `No such file or directory`; no GPU work occurred. The files are visible when the scratchpad is the process working directory, so all subsequent invocations use relative `big_a.ids`/`big4.ids` from that directory.
- Baseline A command/config: `QK_DEVICE_PCI=03:00.0 QK_STEP_PROF=1 QK_CHUNK=8 QK_GGUF=<35B> QK_SHADER_DIR=<build/shaders> build/qk serve-test big_a.ids 256 1 8192`.
  - Device selection proved in output: `AMD Radeon RX 7900 XT ... [0000:03:00.0]`.
  - Request: 1,213 prompt tokens, 256 generated tokens, 4,797.1 ms full request.
  - First pure-decode step: 8,643 us on GPU over 445 timestamped stages = 115.70 tok/s instantaneous reciprocal.
  - Largest aggregates: `at.split` 1,443.5 us; `dn.proj` 1,271.0 us; `dn.step` 797.1 us; `moe.gu` 774.0 us; `moe.dn` 667.4 us; `head` 635.3 us; `wo` 599.6 us; `moe.sel` 528.2 us; `add3` 432.4 us; `moe.route` 341.7 us.
  - Exact `GEN:` line SHA-256: `201f416edb24cc1f5c630bdfe66471d0f12442b53503e1f88da627853c3f60a4`.
- Baseline B command/config: same, `serve-test big4.ids 64 1 8192`.
  - Request: 4,852 prompt tokens, 64 generated tokens, 12,966.5 ms full request.
  - First pure-decode step: 11,811 us on GPU = 84.67 tok/s instantaneous reciprocal.
  - Largest aggregates: `at.split` 4,151.8 us; `dn.proj` 1,247.5 us; `dn.step` 814.3 us; `moe.gu` 797.6 us; `moe.dn` 702.6 us; `head` 641.2 us; `wo` 603.1 us; `moe.sel` 588.2 us.
  - Exact `GEN:` line SHA-256: `98b14c52ffa9f983059f81184b680b6a108a6ed84f3b1cb20bc314e33b4667ef`.
- Baseline profiler interpretation: at 1.2k context, non-attention fixed work is ~7.20 ms and split attention is 1.61 ms including reduction; by 4.9k, split attention grows to 4.61 ms and dominates. The fixed-work tuning campaign should therefore use `big_a`; the long-context gate separately prevents attention regressions.

### 06:32 — Untouched 35B / 7900 XTX baseline and exclusive-use handoff

- Scaled `gemma/qk-server-split-80b` from 1 to 0 replicas (explicitly allowed by the mission), waited for rollout completion, then verified XTX VRAM used = 28,004,352 bytes (<2 GB) and DPM remained `auto`. Restore target is recorded as 1 replica.
- Copied the volatile scratchpad inputs to `/tmp/amd-opt-big_a.ids` and `/tmp/amd-opt-big4.ids` for the campaign; token counts 1,213/4,852 and input SHA-256 values `79f318133b5db687bef0bc1a886f788393ca97b48a48e93b5bef48fa2226a251` / `5c7e187e8abf7008d8854fc013e745c67d1a669e078749cf461040efc1ec64e8`.
- Baseline A command/config: `QK_DEVICE_PCI=1a:00.0 QK_STEP_PROF=1 QK_CHUNK=8 ... serve-test big_a.ids 256 1 8192`.
  - Device selection proved in output: `AMD Radeon RX 7900 XTX ... [0000:1a:00.0]`.
  - Full request 3,436.4 ms; first pure-decode step 6,204 us = 161.19 tok/s.
  - Largest aggregates: `dn.proj` 1,039.6 us; `at.split` 828.6 us; `moe.gu` 539.5 us; `dn.step` 528.8 us; `wo` 504.4 us; `moe.dn` 491.5 us; `head` 458.0 us; `moe.sel` 377.2 us.
  - `GEN:` SHA-256 `201f416edb24cc1f5c630bdfe66471d0f12442b53503e1f88da627853c3f60a4`, identical to XT.
- Baseline B command/config: same, `serve-test big4.ids 64 1 8192`.
  - Full request 9,260.4 ms; first pure-decode step 9,180 us = 108.93 tok/s.
  - Largest aggregates: `at.split` 3,502.3 us; `dn.proj` 1,039.5 us; `moe.gu` 540.1 us; `dn.step` 527.3 us; `wo` 499.2 us; `moe.dn` 493.0 us; `head` 455.1 us; `at.attn` 433.6 us; `moe.sel` 378.1 us.
  - `GEN:` SHA-256 `98b14c52ffa9f983059f81184b680b6a108a6ed84f3b1cb20bc314e33b4667ef`, identical to XT.

### 06:36 — Measurement harness, Vulkan validity repair, and phase-separated baselines

- Read-only audit found that all live `dn_step` pipeline layouts declared 12 push-constant bytes while shader and dispatch use 16, and `dn_step_batch` declared 16 while shader/dispatch use 20. Vulkan requires the layout range to cover every pushed byte; RADV acceptance was undefined behavior. Corrected all current harness/engine declarations to 16/20. Also added the four omitted IQ4 pipeline destructors (`pGemvA4`, `pGemvO4`, `pMoeGU4`, `pGemmB4`). These are validity/lifetime repairs, not credited as speedups.
- Added `serve-bench`: times serving batch prefill separately, measures an exact chunk-divisible decode window with `max_gen` held one token past the stop point, checks every ABI return, and cancels before the completion snapshot. Added `stage-bench`: drives a standalone first-stage range through `stageRun`, separately times production-shaped prefill and repeated n=1 stateful calls, and hashes all returned hidden rows.
- Rebuild succeeded and `git diff --check` passed.
- Phase-separated 35B baseline, `QK_CHUNK=8`, `big_a`, 256 decode tokens, tmax 8192:
  - XT (`03:00.0`): batch prefill 1,212 tokens / 2,445.8 ms = **495.53 tok/s**; steady decode 2,238.3 ms = **114.37 tok/s**, 8.744 ms/token.
  - XTX (`1a:00.0`, DPM auto): batch prefill 1,212 / 1,684.9 ms = **719.31 tok/s**; steady decode 1,667.9 ms = **153.48 tok/s**, 6.515 ms/token.
  - Both `serve-bench` streams match the immutable `big_a` `GEN:` hash `201f416e...f60a4`.
- Phase-separated 80B head `[0,12)` baseline, `QK_MAXB=128` default, `big_a`, 256 decode-shaped calls, tmax 8192:
  - XT: prefill 1,213 / 949.5 ms = **1,277.55 stage-tok/s**; decode 649.9 ms = **393.89 stage-tok/s**, 2.539 ms/token.
  - XTX: prefill 1,213 / 666.4 ms = **1,820.11 stage-tok/s**; decode 481.4 ms = **531.79 stage-tok/s**, 1.880 ms/token.
  - Cross-card byte hashes agree: prefill `cde0335a3bb4f888`, decode `5a140a9a47ada3cf`.
- Harness semantics: the 35B prefill count is 1,212 because `slot_start` intentionally leaves the final prompt token for the first serial step; 80B stage prefill processes all 1,213 inputs. `serve-test` remains the required completion/snapshot correctness gate and is not used for steady performance.

### 06:41 — Trial 1: live-length split-K dispatch (`QK_ATTN_LIVE_DISPATCH`)

- Problem: recorded decode dispatched `hQ * ceil(tmax/32)` split workgroups on every attention layer; the shader returned immediately for splits beyond the live context. At tmax 8,192 / C=1,213 this was 4,096 capacity WGs versus 608 live WGs per layer. At production tmax 49,152 the waste is much larger.
- Implementation behind presence flag `QK_ATTN_LIVE_DISPATCH`: make split and q-head separate Vulkan grid dimensions, retain the fixed `splitMax` scratch stride, and use a coherent host-written indirect command `{ceil(max_live/chunk), hQ, maxZ}` in recorded decode. The dynamic batch n=1 path directly dispatches the known live split count. Flag-off still dispatches full capacity and is the kill-switch.
- XTX A/B at `big_a`, tmax 8192, chunk32:
  - Flag off in the same build: first step 6,360 us; `at.split` 884.5 us; 256-token window 148.99 tok/s (6.712 ms/token).
  - Flag on: first step **5,935 us**; `at.split` **490.7 us**; 256-token window **156.39 tok/s** (6.394 ms/token). Same-build first-step win 425 us / 6.7%; wall-window win 5.0% in this pair. Versus original uninstrumented window 153.48 tok/s, +1.9%.
- XTX long-context `big4`, flag on: first step 9,288 us; `at.split` 3,550.2 us; 64-token window 107.99 tok/s (9.260 ms/token). This is within run variation of the original 9,180 us / 108.93 instantaneous result because real attention work dominates once 152/256 splits are live.
- XT `big_a`, flag on: first step 8,841 us; `at.split` 1,683.7 us; window 111.46 tok/s. This trial did not beat the original XT baseline (8,643 us / 114.37 window); the XT needs a chunk/grid sweep before enabling this flag there.
- All three trial streams remained exact: `big_a` hash `201f416e...f60a4`; `big4` hash `98b14c52...67ef`.

### 06:44 — Trial 1b: live-dispatch chunk sweep

- Re-swept `QK_ATTN_CHUNK={16,32,64,128}` because removing capacity WGs changes the old scheduler/occupancy tradeoff. Each run used `QK_ATTN_LIVE_DISPATCH=1`, `big_a`, tmax8192, 8 measured decode tokens after the full prefill, and one-shot GPU timestamps. All produced the same first eight greedy IDs.
- XTX first-step totals / split / reduce (`at.attn`) in us:
  - chunk16: **5,862 / 412.8 / 163.6**
  - chunk32: 5,935 / 490.7 / 142.8 (earlier 256-token run)
  - chunk64: **5,841 / 428.8 / 100.9** (best total; 165.61 tok/s over the short wall window)
  - chunk128: 5,979 / 579.3 / 85.2
- XT first-step totals / split / reduce in us:
  - chunk16: 8,001 / 586.3 / 241.0
  - chunk32: 8,841 / 1,683.7 / 173.6 (earlier 256-token run; anomalously poor split scheduling)
  - chunk64: **7,664 / 497.1 / 86.6** (best; 130.12 tok/s short window)
  - chunk128: 7,792 / 673.7 / 79.5
- Decision: chunk64 is the common winner with live dispatch on both cards. Keep both settings opt-in while subsequent levers compose; final unprofiled 256/64-token windows will decide the shipped/default policy.

### 06:51 — Trial 2: 256-lane/subgroup MoE selector (`QK_MOE_SELECT_FAST`)

- First attempted a mechanically exact 256-lane emulation of the prior 512-lane selector. It preserved the first eight greedy IDs but made the selector itself slower: XT `moe.sel` 601.4 us versus 537 us in the comparable live-dispatch/chunk64 run, and XTX 432.9 us versus 382.2 us. First-step totals were 7,583 us versus 7,664 us on XT (unrelated aggregate variation) and 5,884 versus 5,841 us on XTX. Rejected and replaced; those shader contents are not retained.
- The retained opt-in implementation assigns two experts to each of 256 lanes, uses an exact shared-memory addition tree for the 512 expert gates, and uses subgroup max/min reductions for deterministic top-k selection with lower expert ID as the tie-breaker. It preserves the existing arithmetic and selected-expert ordering while halving the workgroup width.
- Composed with live attention/chunk64, first-step results were:
  - XT: total **7,396 us**, `moe.sel` **473.6 us / 11.84 us per layer**, 134.13 tok/s over the short eight-token wall window.
  - XTX: total **5,793 us**, `moe.sel` **343.9 us / 8.60 us per layer**, 166.66 tok/s over the short window.
- Relative to the exact-emulation attempt, selector time fell 21.3% on XT and 20.6% on XTX. Both cards generated the immutable first eight IDs `271 248068 198 8160 579 264 7047 1817`.

### 06:51 — Trial 3: register-resident delta-state step (`QK_DN_STEP_REG`)

- Problem: the two-pass delta update reread and rewrote every 128-float state row once per pass. The opt-in replacement loads the row into 32 `vec4` registers, performs both passes in the same invocation and arithmetic order, and stores once. Descriptor bindings and push constants remain unchanged; the old pipeline is the flag-off kill switch.
- Composed profiler results (`big_a`, live attention, chunk64, fast selector, eight decode tokens):
  - XT: first step **7,095 us**; `dn.step` **541.0 us / 18.03 us per layer**, down from 773.2 us in the preceding composed run (**-30.0%**); short wall window 139.06 tok/s.
  - XTX: first step **5,703 us**; `dn.step` **393.1 us / 13.10 us per layer**, down from 530.6 us (**-25.9%**); short wall window 170.58 tok/s.
- The whole first-step total improved 4.1% on XT and 1.6% on XTX versus the fast-selector composition. Both outputs remained byte-identical for the measured IDs. Retain behind `QK_DN_STEP_REG` pending the full correctness matrix.

### 06:54 — Long-run format bandwidth calibration (2,000 iterations)

- Every synthetic matrix exceeded the XTX's 96 MiB Infinity Cache: F16 `M=16384,K=8192` (256 MiB), Q8_0 `16384x8192` (136 MiB), Q6_K `24576x8192` (157.5 MiB), IQ4_XS `32768x8192` (136 MiB), and IQ3_XXS `49152x8192` (147 MiB). All ten correctness checks passed.
- Sustained results (GB/s, XT / XTX):
  - F16: **778.8 / 917.8**
  - Q8_0: **775.0 / 927.8**
  - Q6_K: **606.5 / 800.4**
  - IQ4_XS: **583.3 / 729.0**
  - IQ3_XXS: **421.8 / 543.9**
- These runs validate the nominal 800/960 GB/s roof for simple formats and quantify the practical dequantization tax used in the report's second, shader-aware roof. XTX remained `auto`; every invocation was PCI-pinned.

### 06:57 — Trial 4: selector fused into the last router workgroup (`QK_MOE_ROUTE_FUSED`)

- A router workgroup writes its exact `moe_logits` result with device-scope release semantics, then increments a per-request completion counter. The last workgroup acquires all logits and runs the retained subgroup selector in its epilogue. `add_rmsnorm_route` resets the counter without an extra dispatch. This removes 40 selector dispatches/barriers per 35B token. The counter lives in previously padded selector storage; layout remains 160 bytes.
- XTX composed `big_a` profiler: **5,449 us / 405 stages**, with combined `moe.r+s` 599.8 us versus separate route+select 653.3 us, and 177.50 tok/s over eight tokens. First eight IDs exact.
- XT first sample was noisy: 7,351 us, combined route+select 826.8 us versus 819.8 us separate; unrelated `dn.step`/conv variance dominated. The later 256-token A/B resolved the policy: identical **125.27 tok/s** either way on XT, while XTX improved 177.47 -> **178.97 tok/s**. Retain as an opt-in decode lever; batched `n>1` now uses the standalone fast selector because device-scope atomics are not amortized there.

### 07:00 — Trial 5: right-size underfilled routed IQ4 down workgroups (`QK_MOE_DOWN_128`)

- The 35B ordinary IQ4 down kernel has `top8 * (512/32) = 128` useful tasks but launched 256 lanes. Specialized the existing shader's local size and retained the 256-lane pipeline as the kill switch; Q6 layers remain at 256 lanes.
- Paired profiler result, 256 -> 128 lanes:
  - XT `moe.dn` 679.5 -> **644.6 us** (-5.1%); short wall 137.81 -> 138.45 tok/s.
  - XTX `moe.dn` 502.7 -> **471.6 us** (-6.2%); unrelated attention/conv variation hid the stage-local win in the eight-token wall sample.
- The first eight 35B IDs remained exact. On the 80B top10 shape, 160 tasks make the 128-lane variant change the reduction grouping; it is therefore a separately measured card-specific option, not a universal default.

### 07:02 — Trial 6: GQA-aware K/V reuse and context-adaptive dispatch

- Implemented `fa_attn_srv_split_gqa`: one workgroup handles 2/4/8 adjacent query heads that share a KV head, loads each K/V element once, and retains each head's score, max, sum, and value accumulation order. Manual `QK_ATTN_GQA_GROUP={2,4,8}` variants plus `QK_ATTN_GQA_AUTO` (group1 below 2,048 live tokens, group4 above) are behind flags. AUTO records group1/group4 indirect dispatches and gives exactly one a zero grid, allowing the host to switch without command-buffer rerecording.
- At `big_a` / 1,213 context on XTX, chunk64 `at.split` was group1 **446.9 us**, group2 499.0, group4 624.6, group8 927.7. Reuse loses at short context because fewer workgroups and extra head reductions reduce occupancy; AUTO correctly selects group1.
- At `big4` / 4,852 context, chunk64 results (first-step `at.split`, total):
  - XT group1 15,791.7 / 22,913 us; group2 1,262.3 / 8,143; **group4 968.8 / 7,858**; group8 1,291.7 / 8,277.
  - XTX group1 11,925.1 / 18,006 us; group2 995.6 / 6,317; **group4 840.9 / 6,145**; group8 1,056.4 / 6,447.
- Group4 chunk sweep at long context:
  - XT split/total: chunk32 1,304.7/8,392; **chunk64 968.8/7,858**; chunk128 1,173.0/7,936.
  - XTX: chunk32 1,042.2/6,536; **chunk64 840.9/6,145**; chunk128 948.4/6,206.
- AUTO reproduced the intended branches: `big_a` split 498.1/446.9 us (XT/XTX), `big4` 967.8/833.6 us. Every GQA/chunk sample generated the same first eight IDs as its baseline.

### 07:08 — Trial 7: dense projection workgroup packing (rejected)

- Tested half TPR for both Q8 and IQ4 dense projections: input projections 64 -> 32 threads/row (8 rows/WG) and output projections 128 -> 64 (4 rows/WG). The implementation was flag-gated during the experiment and removed after rejection.
- 35B paired profiler, normal -> half TPR:
  - XT: `dn.proj` 1,263.4 -> 1,293.5 us; `at.proj` 319.6 -> 338.0; `wo` 595.5 -> 621.4; total 7,100 -> 7,244 us.
  - XTX: 1,042.4 -> 1,074.3; 261.1 -> 279.6; 504.8 -> 518.7; total 5,646 -> 5,730 us.
- 80B head also failed to improve: XT 417.7 -> 419.0 stage-tok/s and XTX 617.1 -> 618.4 (noise-sized), while changing hidden-byte hashes. Reject and restore the original TPR pipelines.

### 07:11 — Trial 8: prefill batch width, submissions, and attention budget

- 35B `big_a` prefill sweep with composed kernels (tok/s XT / XTX):
  - `QK_MAXB=64`: 434.61 / 637.55
  - 128-era composed samples: about 507 / 727–737
  - 256: 553.92 / 780.91
  - 512: 560.69 / 788.25
  - **1024: 562.61 / 815.48**
- Long `big4`, MAXB1024: **463.04 / 680.28 tok/s**, versus composed MAXB128 samples 408.64 / 571.88. `QK_ATTN_BUDGET=4,194,304` removed the last-query tiling at 4.9k and nudged this to **465.54 / 685.00**; 4M already covers the largest query-key product, so 8M adds no work reduction.
- `QK_SUBMIT_LAYERS=40` versus default 8 at `big_a` was flat (562.87 / 812.49), proving queue waits are not the remaining prefill bottleneck and retaining the safer default ring-time bound.
- 80B head `big_a` MAXB sweep (XT / XTX): 128 baseline 1,277.55 / 1,820.11; 256 **1,377.76 / 1,859.77**; 512 1,363.41 / 1,863.32; 1024 samples 1,355–1,374 / 1,803–1,909. Use 256 as the stable common setting.
- Conclusion: wider batches recover 8–19%, but the large gap to the unique-expert cache roof is architectural. Current MoE dispatches are token-major; actually reaching that roof requires sorting token/expert pairs and a grouped expert GEMM that reuses each dequantized row across matching tokens. A naive expert-major scan would trade the bandwidth saving for `n_expert * n_token` selection scans and was not accepted without a production grouping primitive.

### 07:15 — Trial 9: fused register delta step + gate (`QK_DN_STEP_GATE_FUSED`)

- Extended the register-cached step with the exact 128-lane RMS reduction and SiLU gate epilogue. This removes the `o[]` write/read plus one dispatch/barrier per recurrent layer. The batch path uses it only for `n==1`; multi-token causal scans retain `dn_step_batch` + `dn_gate_batch`.
- 35B profiler:
  - XT combined `dn.s+g` **596.7 us** versus 639.4 us for separate register-step + gate, while stages fell 445 -> 415 before route fusion. The short total was noise-dominated (7,223 us).
  - XTX combined **394.8 us** versus 460.0 us, and total fell 5,646 -> **5,494 us**.
- 80B head `[0,12)` without routed-down resizing: XT 417.67 -> **423.27** stage-tok/s; XTX 617.10 -> **631.87**. Hidden hashes were unchanged relative to the same chunk/attention configuration (`3331a4356b41c1d4`). First eight 35B IDs remained exact.

### 07:18 — Final measured candidates

- 35B common config: `QK_MAXB=1024 QK_ATTN_CHUNK=64 QK_ATTN_LIVE_DISPATCH=1 QK_ATTN_GQA_AUTO=1 QK_MOE_SELECT_FAST=1 QK_MOE_ROUTE_FUSED=1 QK_DN_STEP_GATE_FUSED=1 QK_MOE_DOWN_128=1` (route fusion automatically falls back to the fast standalone selector for batch `n>1`).
  - `big_a`, 256-token steady window: XT **125.27 tok/s**, prefill **561.66 tok/s**; XTX **178.97 tok/s**, prefill **809.37 tok/s** in the route-fused A/B (best separate-selector prefill sample 813.24).
  - `big4`, 64-token steady window: XT **131.14 tok/s**, prefill **461.26 tok/s**; XTX **163.77 tok/s**, prefill **679.46 tok/s**. With 4M attention budget, prefill samples were 465.54 / 685.00.
  - The long-context decode becoming faster than XT `big_a` is intentional: AUTO activates four-head KV reuse only beyond the 2,048-token crossover.
- 80B head `[0,12)`, `big_a`, 256 decode calls:
  - XT card-best (`MAXB=256`, live/chunk64/AUTO, route fusion, fused step+gate, 256-lane IQ4 down): prefill **1,374.37**, decode **427.29 stage-tok/s**, hashes `cde0335a3bb4f888` / `3331a4356b41c1d4`.
  - XTX card-best (same except fast standalone selector + 128-lane IQ4 down): prefill **1,904.01**, decode **640.01 stage-tok/s**, hashes `d85e23e8adb10a02` / `d488fb24d6592dcd`.
  - Running either configuration on both cards produced matching cross-card hashes; the two configurations differ because top10/128-lane down changes the reduction grouping.

### 07:20 — Roofline accounting checkpoint

- GGUF row bytes proved from block layouts: F32 `4`, Q8_0 `34/32=1.0625`, Q6_K `210/256=0.8203125`, IQ4_XS `136/256=0.53125`, IQ3_XXS `98/256=0.3828125` bytes/weight.
- Exact active decode parameter payload:
  - 35B: Q8 1,492,910,080 + Q6 437,823,120 + IQ4 164,888,576 + IQ3 256,901,120 + F32 104,581,632 = **2,457,104,528 B/token**.
  - 80B head `[0,12)`: Q8 43,452,544 + IQ4 402,751,488 + F32 56,537,856 + 8,192 zero boundary norm = **502,750,080 B/token**.
- Minimum recurrent traffic is 4,390,912 B/layer (state and conv read+write): 131,727,360 B for 30 35B recurrent layers and 39,518,208 B for nine 80B-head recurrent layers. Unique GQA KV traffic is `4096*C` B/attention layer. Final chunk64 split scratch is `16512*ceil(C/64)` B/attention layer.
- Thus `B35(C)=2,457,104,528+131,727,360+10*(4096C+16512*ceil(C/64))`; `B80(C)=502,750,080+39,518,208+3*(4096C+16512*ceil(C/64))`. Raw 800/960 GB/s ceilings are approximately 302.84/363.41 tok/s (35B C=1213), 285.70/342.84 (35B C=4852), and 1,433/1,720 stage-tok/s (80B C=1213).
- FP work lower bounds: `F35(C)=6,000,640,000+163,840C`; `F80(C)=1,658,634,240+49,152C`. Peak FP32 is 43.5456 TF/s XT (`84*128*2*2.025GHz`) and 58.269696 TF/s XTX (`96*128*2*2.371GHz` maximum auto-DPM state). Compute-only ceilings are thousands to tens of thousands of tok/s, so bandwidth—not ALU—is proved binding.
- Prefill parameter floor for a batch is `P35=1,597,491,712+sum_l(U_l*R_l)+1680B`, with `R=1,359,872` ordinary / `1,662,976` Q6-down and `E[U]=256*(1-(248/256)^B)`; `P80=302,207,744+sum_12(U_l*1,671,168)+2176B`, `E[U]=512*(1-(502/512)^B)`. At B=128, expected parameter bytes/token are 121.19 MB (35B) and 76.16 MB (80B); including ideal state/KV for the actual `big_a` chunks yields 127.77/78.89 MB per token and nominal 800/960 roofs of 6,261/7,513 and 10,140/12,168 tok/s. These are explicitly probabilistic/cache-ideal; deterministic bounds use `n_used <= U_l <= n_expert`.

### 07:21 — Required correctness gate (pre-final cleanup build)

- Full `serve-test` on both cards with the final composed flags:
  - `big_a`, 256: XT 4,231.8 ms, XTX 2,981.9 ms; both `GEN:` SHA-256 **`201f416edb24cc1f5c630bdfe66471d0f12442b53503e1f88da627853c3f60a4`**.
  - `big4`, 64: XT 10,992.6 ms, XTX 7,585.7 ms; both SHA-256 **`98b14c52ffa9f983059f81184b680b6a108a6ed84f3b1cb20bc314e33b4667ef`**.
- Both hashes exactly equal the untouched build. Built-in five-format suite passed on both cards; representative post-change XT/XTX results were F16 770.0/913.4, Q8 cache-resident 1,327.5/1,371.4, Q6 670.5/715.6, IQ4 614.8/647.1, IQ3 424.7/429.8 GB/s. A final rerun follows the batch-route policy cleanup.

### 07:32 — Final cleanup, repeat measurements, and verification

- Removed the rejected dense half-TPR experiment completely. Kept fused route/select for decode (`n==1`) and made batched `n>1` work use the standalone fast selector, avoiding the unamortized device-scope atomic path. Rebuilt after both cleanups.
- Final 35B phase-separated `serve-bench` with `QK_CHUNK=8 QK_MAXB=1024 QK_ATTN_CHUNK=64 QK_ATTN_LIVE_DISPATCH=1 QK_ATTN_GQA_AUTO=1 QK_MOE_SELECT_FAST=1 QK_MOE_ROUTE_FUSED=1 QK_DN_STEP_GATE_FUSED=1 QK_MOE_DOWN_128=1`:
  - `big_a` / C=1,213: XT prefill 1,212 tokens in 2,142.6 ms = **565.66 tok/s**, decode 256 in 2,043.0 ms = **125.31 tok/s** (7.980 ms/token); XTX prefill 1,477.5 ms = **820.28 tok/s**, decode 1,432.8 ms = **178.67 tok/s** (5.597 ms/token).
  - `big4` / C=4,852: XT prefill 4,851 tokens in 10,388.1 ms = **466.98 tok/s**, decode 64 in 485.9 ms = **131.71 tok/s** (7.593 ms/token); XTX prefill 7,050.6 ms = **688.03 tok/s**, decode 390.1 ms = **164.06 tok/s** (6.095 ms/token).
- Final 80B-head repeated `stage-bench` medians at `big_a` / C=1,213:
  - XT card-best (MAXB256, live/chunk64/AUTO, fused route/select, fast selector, fused step+gate, down256): prefill samples 1,374.37 / 1,368.13 / 1,380.66, median **1,374.37 stage-tok/s**; decode 427.29 / 427.19 / 426.34, median **427.19**. Hashes `cde0335a3bb4f888` / `3331a4356b41c1d4`.
  - XTX card-best (MAXB256, live/chunk64/AUTO, standalone fast selector, fused step+gate, down128): prefill 1,922.88 / 1,895.49 / 1,929.10, median **1,922.88 stage-tok/s**; decode 633.66 / 630.33 / 631.72, median **631.72**. Hashes `d85e23e8adb10a02` / `d488fb24d6592dcd`.
- Final-build 35B `serve-test` correctness rerun with the common composed flags:
  - `big_a`, 256: XT 4,205.3 ms, XTX 2,967.7 ms; both SHA-256 **`201f416edb24cc1f5c630bdfe66471d0f12442b53503e1f88da627853c3f60a4`**.
  - `big4`, 64: XT 10,880.6 ms, XTX 7,511.8 ms; both SHA-256 **`98b14c52ffa9f983059f81184b680b6a108a6ed84f3b1cb20bc314e33b4667ef`**.
- Final built-in suite passed all five formats on both cards. XT/XTX results: F16 769.9/914.3, Q8 cache-resident 1,332.8/1,402.8, Q6 669.0/711.2, IQ4 627.4/661.1, IQ3 424.4/423.5 GB/s. All GPU commands were explicitly PCI-pinned, all allocations remained VRAM-backed, XT stayed manual-pinned, and every XTX readback remained `auto`.
- Fresh `/usr/bin/cmake --build build --parallel` passed. `spirv-val` passed all eight affected SPIR-V modules. `/usr/bin/ctest --test-dir build` reported no registered tests, as expected; `git diff --check` passed. `REPORT.md` now records the derivations, ceilings, achieved fractions, correctness hashes, and retained/rejected levers.

### 07:33 — XTX production restoration

- Before restoration, `gemma/qk-server-split-80b` was still the authorized campaign state `0/0`, XTX VRAM use was 28,004,352 bytes, XTX DPM read `auto`, and XT DPM read `manual`.
- Scaled only `deployment/qk-server-split-80b` in namespace `gemma` back to its original one replica. Rollout completed successfully; final deployment status was spec/updated/ready/available = **1/1/1/1**, pod `qk-server-split-80b-cfd797b74-7wn9x` was `1/1 Running`, and XTX DPM remained **`auto`**. No GPU benchmark was run after restoration.

### 07:39 — Independent final roofline audit and corrections

- The roofline subagent independently reproduced every chunk64 decode byte floor, cold-stream ceiling, format-calibrated ceiling, and printed start-context percentage. It flagged rigor/labeling issues rather than arithmetic errors in decode.
- Corrected the B128 prefill reference for the final partial chunks: 35B is `9×128+60`, 80B is `9×128+61`, not ten full-width chunks. With independent-uniform route unions, the exact 112 distinct embedding rows in `big_a`, and the retained ideal state/KV addends, expected payload is **133,035,436 B/token** (35B) and **81,237,788 B/token** (80B head). The corrected nominal XT/XTX ceilings are **6,013/7,216 tok/s** and **9,848/11,817 stage-tok/s**. This supersedes the representative B128 figures in the 07:20 checkpoint.
- Tightened the report's proof language: the q64 scratch and full cold-stream weight equations describe the retained implementation, while the parameter-minus-one-cache-capacity table is the formal implementation-independent bandwidth upper bound. The prefill min/full-union columns bound active unique payload under a one-fetch cache ideal, not observed DRAM traffic.
- Recomputed achieved fractions against roofs averaged over the measured growing-context windows. Practical-roof fractions are **49.10/56.74%** (35B short XT/XTX), **54.22/54.86%** (35B long), and **38.12/45.68%** (80B head).

## Phase 2 — residual decode gap, XTX priority

### 17:01 — Intake and merged-main handoff

- Ryan accepted Phase 1 and requested per-stage roof attribution, an XTX `auto`-DPM cold/warm residency measurement, and env-gated attempts on the top three remaining stages with the same byte-exact gates.
- The worktree was clean on `amd-opt` at `5f15bba`; local `main` still pointed to its parent `2663cac`. Fast-forwarded local `main` to the accepted Phase-1 commit and will place the Phase-2 commit directly on `main` as requested.
- Re-adopted all original rails: every GPU command explicitly sets `QK_DEVICE_PCI`; never use `qk token`; never write XTX DPM controls; preserve the XT manual pin; scale only `gemma/qk-server-split-80b` and restore one replica; read the 35B model in place without modifying llama.cpp.
- Stable inputs remain `/tmp/amd-opt-big_a.ids` (1,213 tokens, SHA-256 `79f318133b5db687bef0bc1a886f788393ca97b48a48e93b5bef48fa2226a251`) and `/tmp/amd-opt-big4.ids` (4,852 tokens, `5c7e187e8abf7008d8854fc013e745c67d1a669e078749cf461040efc1ec64e8`).
- Intake state: XTX deployment `1/1/1`, XTX DPM `auto` with advertised 500/2,371 MHz sclk states, and idle readback at the 0 MHz marker. Fresh `/usr/bin/cmake --build build --parallel` passed on `main`.
- Spawned independent read-only audits for exact per-stage roof accounting, safe XTX DPM methodology, and source-grounded ranking of the next kernel levers. GPU work remains centrally serialized.

### 17:04 — Phase-2 matched per-stage baseline

- Scaled only `gemma/qk-server-split-80b` from one replica to zero, waited for pod deletion, then verified XTX VRAM/GTT drained to 28,004,352/15,994,880 bytes and DPM remained `auto` before any model load.
- Profile command on each card used `QK_STEP_PROF=1 QK_CHUNK=8 QK_MAXB=1024 QK_ATTN_CHUNK=64 QK_ATTN_LIVE_DISPATCH=1 QK_ATTN_GQA_AUTO=1 QK_MOE_SELECT_FAST=1 QK_MOE_ROUTE_FUSED=1 QK_DN_STEP_GATE_FUSED=1 QK_MOE_DOWN_128=1`, `big_a`, eight generated tokens, and the required explicit PCI BDF.
- XT (`03:00.0`, manual) first pure-decode step: **7,645 us / 375 stages**. Aggregate stage times in us: `dn.proj` 1,260.6; `moe.dn` 878.8; `moe.r+s` 812.0; `moe.gu` 791.9; `head` 644.9; `dn.s+g` 629.2; `wo` 605.6; `at.split` 507.8; `addN` 429.8; `add3` 361.1; `at.proj` 314.4; `dn.conv` 195.9; `at.prep` 87.6; `at.attn` 86.5; `am1` 21.0; `emb` 8.0; `rms0` 7.6; `am2+copy` 2.8.
- XTX (`1a:00.0`, auto) matched step: **5,493 us / 375 stages**. Aggregate stage times in us: `dn.proj` 1,047.5; `moe.r+s` 606.5; `moe.gu` 544.1; `wo` 508.4; `moe.dn` 471.9; `head` 456.1; `at.split` 421.4; `dn.s+g` 396.9; `at.proj` 261.0; `addN` 223.0; `add3` 217.4; `dn.conv` 158.6; `at.attn` 82.4; `at.prep` 75.5; `am1` 10.8; `rms0` 6.3; `emb` 3.4; `am2+copy` 2.4.
- Both selected the intended device and generated the immutable first eight IDs `271 248068 198 8160 579 264 7047 1817`. These tables follow a full prompt prefill, so they are warm-load stage baselines; the dedicated auto-DPM experiment will force an idle interval before decode.

### 17:18 — Exact XTX stage-floor attribution

- The 35B start-context implementation floor is **2.7517 ms/token** at the XTX's nominal 960 GB/s, versus the measured **5.4936 ms** step: **2.7419 ms** remains. Grouped by cause, parameter-bearing stages account for **1.3360 ms / 48.7%** of that raw-roof gap, attention **0.5243 ms / 19.1%**, recurrent work **0.4183 ms / 15.3%**, and small serialized stages **0.4633 ms / 16.9%**. Against the format-calibrated 315.49 tok/s practical roof, the 2.3239 ms gap splits 39.9% parameter, 22.5% attention, 19.9% small/launch, and 17.7% recurrent.
- Principal per-stage XTX floors (decimal bytes and microseconds):

| Stage | Payload represented by floor | Floor | Observed | Floor/observed | Residual |
|---|---:|---:|---:|---:|---:|
| `dn.proj` | 817.90 MB Q8/F32 | 881.7 us | 1,047.5 us | 84.2% | 165.8 us |
| `moe.r+s` | 173.34 MB F32 + shared Q8 | 187.8 us | 606.5 us | 31.0% | **418.7 us** |
| `moe.gu` | 256.90 MB IQ3 | 472.3 us | 544.1 us | 86.8% | 71.8 us |
| `wo` | 356.52 MB Q8 | 384.3 us | 508.4 us | 75.6% | 124.1 us |
| `moe.dn` | 230.10 MB IQ4/Q6/shared Q8 | 300.0 us | 471.9 us | 63.6% | **171.9 us** |
| `head` | 417.18 MB Q6 | 434.6 us raw | 456.1 us | 95.3% | 21.5 us |
| `at.split` | 403.73 MB actually executed at C=1,213 | 420.6 us | 421.4 us | 99.8% | 0.8 us |
| `dn.s+g` | 127.82 MB unique F32 state | 139.3 us | 396.9 us | 35.1% | **257.6 us** |
| `at.proj` | 200.54 MB Q8 | 216.1 us | 261.0 us | 82.8% | 44.9 us |

- `head` and `at.split` are already within 5% of their nominal raw roofs; the latter's gap to the *unique-GQA* model is redundant group-1 KV execution, not low bandwidth. Prior group-4 trials lost to fewer workgroups/serial reduction, so it is not one of the top three under the retained executed-byte floor.
- The actionable order is therefore `moe.r+s`, `dn.s+g`, `moe.dn`. `moe.r+s` scales XT->XTX by 1.339x, essentially the FP/CU ratio rather than memory-bandwidth ratio, consistent with its selector barriers/device atomic and a shared-Q8 kernel with only 64 useful lanes in a local-256 group. `dn.s+g` launches only 32 four-wave groups per recurrent layer on 96 CUs and its fixed-`i` lanes touch row-major state 512 bytes apart. `moe.dn` has a shared-Q8 pass with only 16 useful lanes in local-256. Stable per-layer detail and the absence of periodic outliers do not support instruction-cache misses as the primary cause; launch granularity, synchronization, and underfilled memory access explain the observed scaling.

### 17:31 — XTX `auto`-DPM cold/warm residency

- Added harness-only controls: `QK_BENCH_PREFIX`, `QK_BENCH_IDLE_MS`, per-chunk trace, deferred one-shot profiler arming, 50 ms `gpu_metrics` v1.3 sampling (10 ms for the first-token study), and token-list suppression via deterministic FNV hashes. The sampler observer-effect gate was **161.25 tok/s off versus 161.22 on** (0.019%). XTX remained `auto`; no DPM sysfs control was written.
- Seven ABBA-counterbalanced process runs used the identical `big_a` prompt, 256 unmeasured generated tokens, then 128 measured tokens. Warm/continuous results were **161.064 ± 0.206 tok/s**; after a 3,000 ms idle, cold results were **159.607 ± 0.363 tok/s**. The cold-start loss is **1.457 tok/s / 0.913%**, or **7.255 ms per 128-token window**. Every prefix hash was `9194f8e9b036fd8e` and every measured hash `3e28d49873ce1eb4`.
- The cost is front-loaded: representative first 8-token chunks were **44.709 ms warm** versus **60.151 ms cold** (+15.442 ms); subsequent pre-threshold chunks converged to about 44 versus 43.7 ms. At the first cold sample UCLK was **96 MHz** and GFX **1,509 MHz**; by the next 50 ms sample UCLK was 1,249 MHz. Across full windows, warm UCLK-high residency was **100%**, cold **93.8%**; GFX-high residency was 93.8% in both because the first sample catches ramp-up.
- Deferred `QK_STEP_PROF` makes the mechanism explicit: the first warm token was **5.588 ms**, while the first token after idle was **12.824 ms**. The 10 ms telemetry series was UCLK 96 -> 1,249 MHz by ~18 ms and GFX 1,708 -> 2,526 -> 2,987 MHz by ~39 ms. Early parameter/state stages inflated 2-3x (`dn.proj` 1,043.7 -> 2,689.4 us, `dn.s+g` 404.9 -> 1,207.6, `moe.r+s` 609.6 -> 1,393.0), whereas the late head had already recovered (455.2 -> 458.4 us). Thus auto-DPM residency costs isolated decode bursts, but not the sustained ceiling gap. The later context-window slowdown appeared in both warm and cold traces after clocks were resident; it is not a cold-clock effect.

### 18:07 — Top-three env-gated kernel trials

- `moe.r+s` trial A, `QK_MOE_SHARED_GU_64=1`: exact local-64 shared Q8 gate/up preserves the two useful subgroup reductions and removes six zero waves. XTX `moe.r+s` sampled **606.5 -> 590.0 us** and the 256-token window reached 180.25 tok/s; the full decode FNV remained `9194f8e9b036fd8e`.
- `moe.r+s` trial B, `QK_MOE_SELECT_HIER=1`: each subgroup keeps its local top-8, then one subgroup selects from their union. A candidate omitted from a local top-8 already has eight candidates ahead of it, proving it cannot be in the global top-8; value ordering and lower-ID tie breaks are unchanged. This removes 24 whole-workgroup barriers. XTX `moe.r+s` sampled **606.5 -> 581.3 us**, 180.60 tok/s, exact FNV. Generalized the union scan to up to 128 candidates so the opt-in path also covers the model's supported top-10/512-expert shape.
- `moe.dn`, `QK_MOE_SHARED_DOWN_32=1`: exact local-32 shared-Q8 down keeps the sole useful subgroup and removes seven zero waves. XTX `moe.dn` sampled **471.9 -> 428.0 us** and 180.79 tok/s, exact FNV.
- `dn.s+g` was pursued through four independent flags, all byte-exact after correction but rejected on performance:
  - `QK_DN_STATE_TRANSPOSED=1` paired transposed batch/decode state kernels to make fixed-`i` lane accesses contiguous; `dn.s+g` regressed to **495.1 us** and decode to 175.10 tok/s. Row-major per-thread cache-line reuse outweighed coalescing.
  - `QK_DN_STEP_TILED=1` exposed 128 one-wave state-update groups then ran the unchanged gate; synchronization erased the occupancy gain (**403.5 us**, 178.33 tok/s).
  - `QK_DN_STEP_SCALAR=1` broadcast the head-uniform decay exponential/beta via the existing q/k barrier; the compiler was already effectively uniform (**401.5 us**, 179.36 tok/s).
  - `QK_DN_STEP_TILE_FUSED=1` used two local-64 update tiles and a monotonic atomic last-tile gate. The initial hard-coded two-subgroup reduction failed the hash gate and was immediately corrected to use `gl_NumSubgroups`; the corrected stream was exact, but **406.8 us / 177.57 tok/s**. None is retained in the final configuration.
- Retained composition (`QK_MOE_SELECT_HIER=1 QK_MOE_SHARED_DOWN_32=1`, with shared-GU64 separately available) stayed exact. Four ABBA XTX baseline/candidate 256-token samples averaged **182.08 -> 183.59 tok/s** (+0.83%); medians were **182.68 -> 184.30**. On the manual-pinned XT, a direct no-profiler pair was **125.36 -> 126.83 tok/s** (+1.17%). A composed XT profile reduced `moe.r+s` **812.0 -> 755.3 us** and `moe.dn` **878.8 -> 586.7 us**; the timestamp profiler itself perturbed wall throughput, so achieved rates use non-profiled windows.

### 18:46 — Cleanup-build medians and byte-exact gates

- Removed all four rejected DeltaNet shader/pipeline variants after recording their results. The final tree retains only the profiling/DPM harness plus `moe_route_select_hier`, `moe_gateup_q8_64`, and `moe_down_q8b_32`. Hierarchical global-candidate fan-in is specialization-controlled: two candidates/lane for top-8 and four for the supported top-10 shape, avoiding top-10 generality overhead on 35B.
- Final flags add `QK_MOE_SELECT_HIER=1 QK_MOE_SHARED_GU_64=1 QK_MOE_SHARED_DOWN_32=1` to the Phase-1 common configuration. Contemporaneous non-profiled three-run medians:
  - XTX `auto`: Phase-1 flags **178.65 tok/s** versus final **180.38 tok/s** (+0.97%). The longer five-sample final series was 181.36 / 181.66 / 180.77 / 180.20 / 179.80; auto-DPM/power variance is why the matched median is reported rather than the best sample.
  - XT manual: after one warm-up outlier, final samples were 126.61 / 126.63 tok/s versus baseline 125.07 / 125.13, so final achieved is **126.61 tok/s** (+about 1.2% contemporaneously; +1.04% versus the accepted 125.31 Phase-1 result).
  - Final long-context single samples: XT **133.04 tok/s** and XTX **164.80 tok/s**, with identical FNV `74f1a8a1435d4aa8`.
- Fractions over the exact growing windows are now: short XT **41.90% raw / 49.61% practical**, short XTX **49.74% / 57.28%**; long XT **46.59% / 54.77%**, long XTX **48.09% / 55.11%**.
- Final-build mandatory `serve-test` matrix, every command explicitly PCI-pinned and with all final flags:
  - `big_a`, 256: XT and XTX both exactly **`201f416edb24cc1f5c630bdfe66471d0f12442b53503e1f88da627853c3f60a4`**.
  - `big4`, 64: XT and XTX both exactly **`98b14c52ffa9f983059f81184b680b6a108a6ed84f3b1cb20bc314e33b4667ef`**.
- The top-10/512-expert compatibility gate compared baseline versus hierarchical 80B head `[0,12)` on XTX; both produced prefill/decode hidden hashes **`d85e23e8adb10a02` / `c5ed7d32532f02bd`**.
- Built-in five-format suite passed both cards. XT/XTX: F16 769.1/912.9, Q8 1,412.6/1,434.0, Q6 682.5/709.6, IQ4 643.9/665.1, IQ3 430.5/427.7 GB/s. All three new modules passed `spirv-val --target-env vulkan1.2`; CTest reported no registered tests.

### 18:58 — Production restoration

- After the final GPU command, XTX VRAM/GTT had drained to 27,947,008/15,970,304 bytes. Scaled only `gemma/deployment/qk-server-split-80b` from zero back to its original one replica and waited for rollout.
- Final deployment spec/updated/ready/available is **1/1/1/1**; pod `qk-server-split-80b-cfd797b74-ldrhl` is `1/1 Running` with zero restarts. Final DPM readbacks are XTX **`auto`** and XT **`manual`**. No GPU benchmark ran after restoration.

## 2026-07-14

### 00:01 — Architectural follow-on intake and checkpoint discipline

- Re-read this log and `REPORT.md` before source work and treated every recorded roof and negative result as closed. The live worktree contained the complete, documented 17:01–18:58 residual-stage patch plus its three retained shaders; provenance audit found no unrelated edits. Preserved it in checkpoint `793a76a` before the first test, then used small commits between every implementation/test boundary as requested.
- The mission levers were narrowed exactly to the report's remaining opportunity: counting-sorted grouped prefill MoE, plus a decode quant-kernel/fusion change. No DeltaNet layout/tile variants, old IQ3 dword repack, dense half-TPR, or short-context GQA dead ends were repeated.
- Stable inputs remained `/tmp/amd-opt-big_a.ids` (1,213 tokens, SHA-256 `79f318...a251`) and `/tmp/amd-opt-big4.ids` (4,852, `5c7e18...64e8`). Every GPU invocation below explicitly set `QK_DEVICE_PCI`; `qk token` was never used.

### 00:05 — Grouped prefill MoE implementation and XT isolation

- Added `QK_MOE_GROUP_PREFILL_GU`, `QK_MOE_GROUP_PREFILL_DOWN`, and umbrella `QK_MOE_GROUP_PREFILL`. One GPU counting-sort workgroup builds expert ranges over packed `(token,slot)` pairs. Expert-major IQ3 gate/up and IQ4/Q6 down workgroups dequantize each row once and loop all matching pairs; grouped down writes token/slot contributions and a grouped add/RMS tail folds them deterministically.
- Correctness audit before GPU use fixed three avoidable association changes: restored IQ3's alternating lower/upper term order, applied routed weights before subgroup reduction, and formed the routed slot sum before `residual + routed + shared`. Q6 retains `w*d*scale*dot`. Small speculative batches retain token-major dispatch when `n*n_used < n_expert`; empty buckets return before dequant; the grouped path is guarded to `n_ff=512`.
- First XT matched smoke, all with identical first-16 greedy tokens:
  - token-major **595.31 tok/s**;
  - grouped GU only **767.69** (+29.0%);
  - grouped GU+down **858.33** (+44.2%).
- Final XT `big_a` composed samples were prefill 849.61 / 854.38 / 848.62 (median **849.61 tok/s**) and decode 126.83 / 128.33 / 128.55 (median **128.33**). `big4` single was **645.72 prefill / 135.14 decode**.

### 00:11 — Decode IQ3 row-stationary trial

- Added decode-only `QK_MOE_GU_ROWTILE`: one workgroup computes consecutive IQ3 gate/up rows while reusing each lane's eight input values. This is not the old 25-u32 repack; weight layout is unchanged and each row preserves the original lane, term, subgroup, and serial-wave reduction order.
- Eight rows lost to register pressure: XT `moe.gu` was **918.4 us** and wall decode regressed. The four-row checkpoint reduced `moe.gu` from the recorded 791.9 to **726.2 us** (-8.3%) and stayed exact.
- Direct wall pairs: XT **127.41→128.05 tok/s**; XTX **180.18→181.52**. Later composed medians/singles are reported below. The gain is retained but small because routed GU is only one part of the serialized decode stream.

### 00:15 — Exclusive XTX campaign and final medians

- Intake anomaly was the same as the earlier handoff: the live 80B pod was `1/1`, but card2 showed 27.9 MB VRAM, 16.2 GB GTT, and an `EBUSY` DPM read. Scaled only `gemma/deployment/qk-server-split-80b` to zero, waited for pod deletion, then required VRAM/GTT drain to 27,947,008/15,921,152 bytes and DPM **`auto`** before the first XTX benchmark.
- XTX token-major short prefill samples were 863.23/863.16 tok/s. Composed grouped samples were 1,164.82 / 1,142.14 / 1,153.31, median **1,153.31 tok/s** (+33.6% matched; +35.97% over the prior report's 848.19). Composed decode samples were 181.16 / 181.44 / 180.96, median **181.16 tok/s**. `big4` reached **944.50 prefill / 166.72 decode**.
- Optional grouped IQ4 down on the 80B XTX head moved prefill **1,928.75→1,966.41 stage-tok/s** (+1.95%). Its hidden hashes changed (`d85e.../c5ed...` baseline versus `21a8.../3847...`) because the old local-128 kernel mixes two slots per subgroup; this flag is not added to production. A grouped IQ4 gate/up shader remains the material 80B extension.
- XTX remained `auto` throughout every readable benchmark-state check. No DPM control was written.

### 00:21 — Mandatory gates and production restoration anomaly

- Final composed flags add `QK_MOE_GROUP_PREFILL=1 QK_MOE_GU_ROWTILE=1` to the previous common set.
- Final `serve-test` matrix, every command PCI-pinned:
  - `big_a`, 256: XT and XTX both exactly **`201f416edb24cc1f5c630bdfe66471d0f12442b53503e1f88da627853c3f60a4`**.
  - `big4`, 64: XT and XTX both exactly **`98b14c52ffa9f983059f81184b680b6a108a6ed84f3b1cb20bc314e33b4667ef`**.
- Built-in five-format suite passed both cards. XT/XTX correctness-run rates: F16 769.2/913.1, Q8 1,343.6/1,331.7, Q6 677.1/626.0, IQ4 621.3/581.1, IQ3 431.9/410.3 GB/s.
- Restored only `qk-server-split-80b` to one replica; deployment and new pod reached spec/updated/ready/available **1/1/1/1** with zero restarts. No XTX benchmark ran afterward.
- The restored GPU state is not healthy: model load briefly reached 9.0 GB VRAM, then migrated about 16.2 GB to GTT; DPM remained unreadable. Kernel logs show repeated XTX PSP load/unload failures and suspend/resume cycles, and `fuser` identifies `ptyxis` PID 54767 holding `/dev/dri/renderD129`. This exactly reproduces the intake anomaly. The replica remains up; no unauthorized process kill, GPU reset, or second rollout was attempted.

## 2026-07-19

### Qwen 35B RDNA3 prefill gap — clean XTX baseline and retained row-paired grouped MoE

- Worked on branch `prefill-opt` at base `52e240f9c6d91750d0e5e692976cfb67fd9bc603` in an isolated sibling worktree because the requested repository worktree was actively dirty on `gemma4-port`. No commit was created and the original worktree was not touched.
- Every accepted qk sample was PCI-pinned to `1a:00.0` and read `/sys/class/drm/card2/device/gpu_busy_percent <= 2` immediately before the repetition. The concurrently active stage6 job repeatedly reclaimed card2; contaminated/aborted series were discarded. llama.cpp was run once as a five-repetition series after a 0% busy read; its tool does not expose a per-repetition hook, which is recorded explicitly in the JSONL.
- Clean XTX head-to-head medians (five samples, min–max in parentheses):
  - pp512: llama.cpp `571d0d5` **2,892.47 tok/s** (2,866.23–2,967.39), qk baseline **1,158.70** (1,152.43–1,162.75), a 2.50x llama/qk ratio.
  - pp1024: llama.cpp **2,857.38** (2,842.00–2,884.35), qk **1,246.83** (1,231.56–1,263.55), 2.29x.
  - pp2048: llama.cpp **2,822.22** (2,801.66–2,864.88), qk **1,149.93** (1,148.91–1,173.90), 2.45x.
  - qk uses the serving admission path without an all-row LM head; pp2048 is two maxB=1024 chunks. llama uses `llama-bench -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512`. This is the same qk admission surface used by the prior AMD campaign, but the surface difference is retained as a comparison caveat.
- Added opt-in Vulkan stage timestamps (`QK_COUNTERS=1`, deferred/raw modes, and `qk counters`). The clean baseline pp512 trace spanned **442.632 ms** versus 445.053 ms wall. RDNA3 attribution was: gate/up 93.011 ms / **21.01%**, routed down 78.605 / **17.76%**, recurrent dense projections+output 122.087 / **27.58%**, recurrent conv+step+gate 46.355 / **10.48%**, attention projections+output 33.421 / **7.55%**, attention prep+core 20.355 / **4.60%**, and route/select/pair/shared work 35.304 / 7.98%. This differs materially from the Metal split: attention core is larger and recurrent dense work dominates gate/up as a group.
- Retained two exact 35B grouped-prefill kernels. One workgroup now computes two adjacent expert gate/up rows or two adjacent down output rows, reusing each selected token's input values while preserving each row's original lane assignment, term order, subgroup reduction, and routed-weight placement. They default to row-2 when grouped prefill is enabled; `QK_MOE_GROUP_PREFILL_ROWS=1` and `QK_MOE_GROUP_PREFILL_DOWN_ROWS=1` are exact rollbacks. The row-paired down selector is intentionally restricted to the 35B IQ3-gate graph so the independently tuned 80B grouped-down path is unchanged.
- End-to-end pp512 experiments (five samples each): gate/up row-2 **1,183.94 tok/s** versus adjacent rollback control 1,157.07; gate/up row-4 1,176.74; pair-panel local-512 956.26. With gate row-2, down row-2 reached **1,223.76** (reverse bracket 1,220.25), while down row-4 was neutral at 1,223.42 with wider spread. Retained both row-2 schedules; removed pair-panel and row-4 production variants.
- DeltaNet row panels were bit-exact for output and final state at both k-head mappings and Tn={1,5,64,65,128,200,512}. Split-4 improved the isolated Tn512 step from 1,364.9 to 1,066.0 us at a 1% busy read, but composed pp512 median was only **1,217.59 tok/s**, below the MoE-only result. Production selection was dropped; the split shader remains only in standalone `dncmp` so the exact comparator is reproducible.
- Final cleaned-build medians (five repetitions): pp512 **1,205.66 tok/s** (1,197.39–1,226.78, +4.05%); pp1024 **1,292.27** (1,291.23–1,294.32, +3.65%); pp2048 **1,227.45** (1,198.53–1,229.55, +6.74%). Versus llama medians, qk now reaches 41.68% / 45.23% / 43.49%; llama remains 2.40x / 2.21x / 2.30x faster. The gap is narrowed, not closed.
- The final clean trace (`gpu_busy_before=2`) was 420.435 ms wall / 418.070 ms timestamp span. Gate/up fell **93.011 -> 84.365 ms (-9.3%)**, down **78.605 -> 64.718 ms (-17.7%)**, and total span fell **5.55%**; unrelated stages stayed close. This directly attributes the retained gain to the intended two largest individual RDNA3 stages.
- Correctness completed before final cleanup for each retained change: scalar prefill 42/42 at worst relative logit drift `9.9e-7`, `dncmp` all raw-bit cases PASS, exact 100-token `ids3` reference, eight 200-token slots identical, and N=128 handoff exact for all three seeds. `block` and `ablock` fail their pre-existing Vulkan CPU-reference thresholds, but the candidate output reproduces untouched main line-for-line (`block` state max-abs difference remains `2.38e-7`). A final cleaned/default-path rerun and ids1–ids4 reference sweep follow below.
- Full raw samples, busy reads, contaminated-run exclusions, commands/semantics, trace stages, and experiment decisions are in `bench/results-prefill-opt.jsonl`.

### Final cleaned/default-path gate disposition

- Rebuilt after removing rejected production variants and after restricting row-paired down to the 35B graph. All retained SPIR-V modules passed `spirv-val --target-env vulkan1.2`; the final scalar prefill sweep remained **42/42** with worst relative drift `9.9e-7`, and the standalone DeltaNet comparator remained raw-bit exact in all 28 cases.
- Final `serve-test` reference checks under the full tuned flag set: ids1 **100/100 exact**, ids2 **100/100 exact**, ids3 **100/100 exact**. ids4 differs from `ref4.txt` at generated index 32 (`735` versus `13914`), but this is pre-existing: untouched main and both row kernels rolled back produce the identical `735`. Most decisively, the retained candidate equals rollback for **all 64 ids4 tokens** both at maxB1024/full tuning and at maxB512 with only grouped prefill enabled. Thus the optimization preserves the current engine trajectory across both admission geometries, while the stale absolute ref4 divergence is not hidden or relabeled as a pass.
- Final eight-slot gate: eight 200-token ids3 streams were identical. Final `prefilldecode 128 24 2048`: all three seeds passed; seeds 7 and 99 matched all 24 decode tokens and the EOS seed matched its empty stream. **HANDOFF EXACT**.
- `block 0 3 200` and `ablock 3 3 200` retain their current-main Vulkan CPU-reference failures exactly. Candidate and untouched output are line-for-line identical; `block` retains state max-absolute difference `2.38e-7`. These are reported as pre-existing failures, not green gates.

### Qwen dense-prefill cooperative matrix (`qwen-prefill-coopmat`)

- Started from the clean requested base `cf899c1` in the isolated `qwen-kernel-qwen-prefill` worktree. Both RDNA3 cards advertise `VK_KHR_cooperative_matrix`, wave64, and the required 16x16x16 F16-input/F32-accumulator shape. Added a source-native Q8_0 dense GEMM; no SPIR-V or executable was copied or loaded from another tree. A 256-thread group produces a complete 128-row by 64-token output tile with eight subgroup cooperative accumulators. Q8 weights and F32 activations are staged as F16 in LDS, while the K reduction and stores remain F32.
- The initial structural probe exposed four wave64 subgroups, not the eight wave32 subgroups assumed by the first layout. The corrected mapping gives each subgroup two 16-row bands and four adjacent 16-token tiles. A synthetic comparison then confirmed the expected F16-input rounding. A split-half residual correction was slower and less accurate because it cannot reproduce the cooperative instruction's internal association; it was removed.
- Enabling cooperative projections in all layers improved an idle-XT pp512 median from 980.514 to 1,262.935 tok/s (+28.8%, five repetitions), but changed the established ids4 decode trajectory at generated token 32. This configuration was rejected. Layer bisection found first-cooperative-layer 3/8/9 unsafe and 10/12/16 rollback-exact for all 64 ids4 tokens. The retained default is therefore layer 10. Selection is additionally limited to Q8 projections, `N>=192`, and complete M%128/N%64/K%32 tiles. N=128 always uses the established scalar-order kernel, making the sensitive prefill/decode handoff an exact rollback rather than a cooperative numerical approximation.
- Final XTX performance, every accepted repetition preceded by `gpu_busy_percent<=2`, five samples each (median, min-max): pp512 **1,483.014 tok/s** (1,474.400-1,489.175), pp1024 **1,531.455** (1,529.708-1,535.411), pp2048 **1,404.067** (1,403.468-1,434.252). The contemporaneous pp512 scalar rollback was 1,203.954 (1,197.482-1,222.359), so the retained end-to-end win is **23.18%**. Against the recorded row-paired baselines, pp1024/2048 improve **18.51%/14.39%**.
- The pp512 timestamp span fell 418.070 -> **346.088 ms** (-17.22%). Dense projection time (`dn.proj + dn.out + at.proj + at.out`) fell 152.167 -> **82.692 ms** (-45.66%). The new residual distribution is gate/up 84.001 ms / 24.27%, down 63.678 / 18.40%, DeltaNet step 39.332 / 11.36%, shared/router 32.946 / 9.52%, and attention core 19.319 / 5.58%. Dense cooperative projection work is no longer the dominant individual group.
- Correctness on the retained layer-10 build: scalar-prefill comparator **42/42 argmax** (coop cases worst relative logit drift 0.0067; scalar fallback cases 9.9e-7), `dncmp` **28/28 raw-bit**, and the three-seed N=128 handoff **HANDOFF EXACT** for 24 decode tokens. The scalar-prefill harness must omit live/GQA indirect-dispatch flags because it submits recorded command buffers without the serving path's indirect update; with those flags it produces NaNs even at N=128 where cooperative GEMM is disabled. Final XT reference runs were ids1/ids2/ids3 **100/100 exact**; ids4 was **64/64 exact against the scalar rollback**, retaining only the known `ref4` divergence at generated index 32 (`735` versus `13914`). Eight ids3 slots were identical for 200 tokens. `serve-bench` retained the known ids2/ids4 `produced=0` behavior both with cooperative GEMM off and on. `block` and `ablock` retain their pre-existing Vulkan CPU-threshold failures (`block` state max-abs 2.38e-7); no new failure was hidden behind them.
- The thesis is directionally correct but does **not** close the gap. qk now reaches 51.27% / 53.60% / 49.75% of llama.cpp at pp512/1024/2048; llama remains 1.95x / 1.87x / 2.01x faster. Closing the rest requires batched/cooperative quantized work for the routed IQ3 gate/up and IQ4/Q6 down paths, plus DeltaNet step and shared/router improvements. A dense-only cooperative kernel has reached an honest end-to-end floor, and pp2048 also pays two maxB=1024 admissions.

### Qwen routed-MoE + shared-expert cooperative matrix (`qwen-prefill-coopmat`, continued)

- Continued from `2e02490` in the isolated `qwen-kernel-qwen-prefill` worktree. Development on the XT (`03:00.0`); every accepted XTX repetition was gated on `card2 gpu_busy_percent <= 2` read immediately before it via `QK_PB_BUSY_PATH`. All kernels are qk-native GLSL following the repo's own `gemm_q8_0_coopmat.comp` structure and the Gemma-4 grouped/prefetch schedule; no external SPIR-V or binary was copied, linked, or loaded.
- Cross-checked the target against llama.cpp `571d0d5` per-op timing (`GGML_VK_PERF_LOGGER=1`, XTX): `MUL_MAT_ID iq3_xxs m=512 k=2048` ~794-827 us per call (gate and up each, 80 calls), `iq4_xs down` ~822-873 us x37, `q6_K down` ~800 us x3, shared-expert dense Q8 GEMMs ~44 us each. Conclusion: llama's routed-expert cost is ~96 ms of its ~179 ms pp512, and qk's largest deficits were the scalar routed MoE, the scalar shared expert (~10x llama), and a dense coopmat GEMM running ~4x below llama's mul_mm throughput.
- **Routed cooperative MoE** (`QK_MOE_PREFILL_COOPMAT`, default first layer 4): expert-major (offsets, pairs) schedule retained from `moe_group_pairs`; workgroup = (64-row group, expert), looping the expert's pairs in 32-column tiles; two 16-column cooperative tiles per subgroup strip; IQ3_XXS gate/up and IQ4_XS/Q6_K down dequantized to f16 LDS with raw block bytes prefetched into registers one K step ahead; F32 accumulators; SiLU pairing elementwise on F32 accumulators; h staged and scattered in F32; routed weight applied after the complete K reduction at the scatter, where the scalar kernel applies it. Empty experts exit before any work, so no tile-list compaction pass is needed on this 256-expert grid (the per-expert loop plays that role). Selection requires n >= 192, ffE%64==0, nEmbd%64==0, and !guIq4; N=128 and decode stay on the exact scalar rollback, and the 80B IQ4 graph is untouched.
- **Shared-expert cooperative GEMMs** (enabled under the same umbrella): `moe_gateup_q8_coopmat` (64x64 tile, fused SiLU, strided h scatter, first layer 6) and `moe_down_q8_coopmat` (first layer 6). The shared down feeds the residual stream directly and single-f16 staging changed ids4 even when only layers 32-39 were cooperative; the shipped kernel splits BOTH operands into f16 hi/lo planes with the lo planes scaled by 2^11 (raw residuals are f16-subnormal and flush inside the cooperative product), takes three cooperative products per tile into a main and a correction accumulator, and folds `c + e/2048` at the store. That construction is ids4-exact from layer 6.
- **Found and fixed a real defect while building the split**: the first `moe_down_q8_coopmat` dequant mapping assigned 256 threads to 64 rows x 2 half-blocks (128 tasks) without a guard, so threads 128-255 wrote rowLocal 64-127 past `Wd` into the following LDS arrays. The fixed mapping is 64 rows x 4 quarter-blocks. This out-of-bounds write is what made the shared-down path look catastrophically non-deterministic in early bisection (0.8 rel at one layer).
- **Dense coopmat GEMM v2** (`gemm_q8_0_coopmat2`, default; `QK_PREFILL_COOPMAT_V1` restores the original): same 128x64 tile, dispatch, bindings, and — by construction — bit-identical output (identical ascending-k coopMatMulAdd sequence per accumulator and identical f16 staging expressions); only the memory schedule changed (BK=64, one block per thread, padded LDS strides, register prefetch). XT dense stage times fell dn.proj 58.9->47.5, dn.out 26.0->19.1, at.proj 13.0->9.8, at.out 8.1->5.6 ms.
- Rejected: LDS-resident IQ3 grid LUT (gate/up 91.7->98.5 ms, reverted); single-f16 shared down at any tested boundary (10/24/32 all change ids4); routed/sharedGU boundaries below their retained values (routed@3 and sharedGU@4 diverge; routed@4 and sharedGU@6 are exact at maxB=1024 and 512).
- Final XTX medians, 5 reps, every rep `gpu_busy<=2` (min-max): pp512 **2029.148 tok/s** (2024.728-2029.874), pp1024 **2153.718** (2148.279-2161.020), pp2048 **1928.736** (1925.736-1933.427). Contemporaneous XTX rollback (branch-HEAD config, `QK_PREFILL_COOPMAT_V1`, no MoE coop) pp512 median 1489.263 (1480.967-1494.155), so the retained end-to-end win is **+36.25%**. Versus the recorded llama.cpp `571d0d5` medians the ratios are **0.702x / 0.754x / 0.683x** (from 0.513x / 0.536x / 0.498x).
- Final XTX pp512 stage trace (timestamp span 223.1 ms): moe.gate+up 59.5 ms / 26.7%, dn.step 31.9 / 14.3%, moe.down 32.5 / 14.6%, dn.proj 30.4 / 13.6%, moe.route+shared_gu 18.2 / 8.2%, at.core 16.8 / 7.5%, dn.out 13.5 / 6.1%. The remaining gap to llama is now spread across the routed IQ3 gate/up (still ~1.4x llama's per-layer mul_mat_id), the DeltaNet recurrent step, and attention core — no single stage dominates.
- Correctness on the retained build (XT, full tuned flags): prefillcmp **45/45 argmax** (worst rel 0.035; the harness must omit `QK_ATTN_LIVE_DISPATCH/QK_ATTN_GQA_AUTO`, the pre-existing recorded-command-buffer incompatibility); `dncmp` **all PASS** raw-bit; ids1/ids2/ids3 **100/100 exact**; ids4 **64/64 exact vs rollback** at maxB=1024 and 512, retaining only the pre-existing `ref4` divergence at generated index 32 (`735` vs `13914`); eight 200-token ids3 slots identical; `prefilldecode 128 24 2048` **HANDOFF EXACT** all three seeds; serve-bench retains the known ids2/ids4 `produced=0`; `block`/`ablock` retain their pre-existing CPU-reference threshold failures with `block` state max-abs unchanged at 2.38e-07. All six new/changed SPIR-V modules pass `spirv-val --target-env vulkan1.2`.
- Assessment: parity is not closed — 0.68-0.75x — but the MoE thesis is now fully executed (routed + shared + dense). Reaching parity requires (1) the routed IQ3 gate/up to close its remaining ~1.4x against llama's mul_mat_id (padding at 16 pairs/expert-tile and dequant ALU), (2) a DeltaNet-step schedule change (31.9 ms, recurrent, previously resistant), and (3) attention-core work — each individually smaller than what this pass recovered.
