# AMD RDNA3 kernel optimization report

Date: 2026-07-13
Branch: `amd-opt`
Models: Qwen3.6-35B-A3B full model and Qwen3-Next-80B-A3B head stage `[0,12)`

## Outcome

The decode roof is bandwidth-bound on both cards. The proof below gives an implementation-independent cache-best bound, a tighter cold-stream DRAM model for the retained split-K path, and a measured dequant-aware target for the current kernels. The final 35B build reaches 49–57% of that measured target, depending on card/context, while remaining greedy-byte-exact. The 80B head reaches 38% on XT and 46% on XTX.

Phase-matched `big_a` results:

| Workload | Card | Baseline prefill | Final prefill | Baseline decode | Final decode | Decode gain |
|---|---:|---:|---:|---:|---:|---:|
| 35B full | 7900 XT | 495.53 tok/s | **565.66 tok/s** | 114.37 tok/s | **125.31 tok/s** | **+9.57%** |
| 35B full | 7900 XTX | 719.31 tok/s | **820.28 tok/s** | 153.48 tok/s | **178.67 tok/s** | **+16.41%** |
| 80B head `[0,12)` | 7900 XT | 1,277.55 stage-tok/s | **1,374.37** | 393.89 stage-tok/s | **427.19** | **+8.45%** |
| 80B head `[0,12)` | 7900 XTX | 1,820.11 stage-tok/s | **1,922.88** | 531.79 stage-tok/s | **631.72** | **+18.79%** |

At 4,852-token context, the final 35B steady decode is **131.71 tok/s on XT** and **164.06 tok/s on XTX**. This is faster than the XT short-context result because the adaptive kernel begins reusing each KV read across four query heads above 2,048 tokens.

## 1. Hardware model and conventions

| Card | CUs | Clock used for peak | FP32 peak | DRAM model | Infinity Cache |
|---|---:|---:|---:|---:|---:|
| RX 7900 XT | 84 | fixed 2.025 GHz | 43.5456 TFLOP/s | 800 GB/s | 80 MB |
| RX 7900 XTX | 96 | 2.371 GHz maximum auto-DPM state | 58.269696 TFLOP/s | 960 GB/s | 96 MB |

The peak calculation is

```text
peak = CU × 128 FP32 lanes/CU × 2 FLOP/FMA × clock
XT   = 84 × 128 × 2 × 2.025e9 = 43.5456e12 FLOP/s
XTX  = 96 × 128 × 2 × 2.371e9 = 58.269696e12 FLOP/s
```

The XTX was never manually pinned; every readback showed `power_dpm_force_performance_level=auto`. Every GPU command was explicitly selected by PCI (`03:00.0` XT, `1a:00.0` XTX). GB/s calculations use decimal bytes, matching the advertised 800/960 GB/s.

## 2. Exact active decode bytes

### Quantized row sizes

The GGUF block definitions and static assertions in [src/gguf.h](../../src/gguf.h) prove these encoded bytes per weight:

| Format | Block bytes / values | Bytes/weight |
|---|---:|---:|
| F32 | 4 / 1 | 4.000000 |
| Q8_0 | 34 / 32 | 1.062500 |
| Q6_K | 210 / 256 | 0.8203125 |
| IQ4_XS | 136 / 256 | 0.531250 |
| IQ3_XXS | 98 / 256 | 0.3828125 |

For routed experts, only the selected `top_k` expert slices are active for one decode token. Router, shared-expert, dense, normalization, embedding/head, and recurrent weights are counted in full. Tensor shapes and the 35B/80B shape split are checked by the loader in [src/main.cpp](../../src/main.cpp).

### Active parameter payload

| Workload | Q8_0 | Q6_K | IQ4_XS | IQ3_XXS | F32 | Total bytes/token |
|---|---:|---:|---:|---:|---:|---:|
| 35B full, 40 layers, top-8 | 1,492,910,080 | 437,823,120 | 164,888,576 | 256,901,120 | 104,581,632 | **2,457,104,528** |
| 80B head `[0,12)`, top-10 | 43,452,544 | 0 | 402,751,488 | 0 | 56,537,856 | **502,741,888** |

The 80B runtime also reads an 8,192-byte zero boundary norm on the non-final stage, so its runtime parameter floor is **502,750,080 B/token**.

Cross-check by active layer:

| Workload/layer kind | Active payload |
|---|---:|
| 35B ordinary recurrent | 52,650,752 B |
| 35B Q6-down recurrent | 55,075,584 B |
| 35B ordinary attention | 45,312,000 B |
| 35B Q6-down attention | 47,736,832 B |
| 80B top-10 recurrent | 42,754,816 B |
| 80B top-10 attention | 39,315,456 B |

The 35B model has 30 recurrent and 10 attention layers; the 80B head has 9 recurrent and 3 attention layers.

### Mandatory state and attention traffic

For each recurrent layer:

```text
delta state S = 32 heads × 128 × 128 × 4 B = 2,097,152 B
conv window    = 8,192 channels × 3 × 4 B = 98,304 B
minimum read + write = 2×S + 2×conv = 4,390,912 B/layer
```

That is 131,727,360 B/token for the 35B's 30 recurrent layers and 39,518,208 B/token for the 80B head's 9 layers. The register-resident kernel now performs the state read/write once, matching this lower bound.

For an attention layer at live context `C`, the unique GQA KV traffic is

```text
2 (K,V) × 2 KV heads × C × 256 dimensions × 4 B = 4,096 C bytes.
```

The retained split-K implementation with chunk `q=64` writes

```text
16 query heads × ceil(C/q) × (256 output + max + sum) × 4 B
= 16,512 ceil(C/64) bytes/layer
```

of partial scratch. Smaller activation reads/writes are intentionally omitted; therefore the result is an optimistic byte floor and its reciprocal is a valid upper ceiling.

The retained chunk-64 split-K path's cold-stream byte model is:

```text
B35(C) = 2,457,104,528 + 131,727,360
         + 10 × [4,096C + 16,512 ceil(C/64)]

B80(C) = 502,750,080 + 39,518,208
         + 3 × [4,096C + 16,512 ceil(C/64)]
```

### Cold-stream DRAM ceilings

If each active stream reaches DRAM once, `time/token ≥ bytes/token ÷ bandwidth`. This is the relevant model for the current weight streams, but it is not universal: another implementation can remove split scratch, and Infinity Cache can retain part of a prior token's working set. Section 4 therefore gives the looser implementation-independent cache-best bound.

| Workload | Context | Byte floor | XT latency floor / ceiling | XTX latency floor / ceiling |
|---|---:|---:|---:|---:|
| 35B | 1,213 | 2,641,653,648 B | 3.3021 ms / **302.84 tok/s** | 2.7517 ms / **363.41 tok/s** |
| 35B | 4,852 | 2,800,118,928 B | 3.5001 ms / **285.70 tok/s** | 2.9168 ms / **342.84 tok/s** |
| 80B head | 1,213 | 558,114,816 B | 0.6976 ms / **1,433.40 stage-tok/s** | 0.5814 ms / **1,720.08** |
| 80B head | 4,852 | 605,654,400 B | 0.7571 ms / **1,320.89 stage-tok/s** | 0.6309 ms / **1,585.06** |

Parameter-only ceilings, useful as a cross-check, are 325.59/390.70 tok/s for 35B and 1,591.25/1,909.50 stage-tok/s for 80B (XT/XTX).

## 3. ALU/dequant proof

The minimum core floating-point work per decode token, including attention's context term, is

```text
F35(C) = 6,000,640,000 + 163,840 C FLOP
F80(C) = 1,658,634,240 +  49,152 C FLOP
```

The underlying active MAC counts are 2,945,269,760 for 35B (including the output head) and 812,802,048 for the 80B head. At `C=1,213`:

| Workload | FLOP/token | Intensity at byte floor | XT compute ceiling | XTX compute ceiling |
|---|---:|---:|---:|---:|
| 35B | 6,199,377,920 | 2.347 FLOP/B | 7,024 tok/s | 9,399 tok/s |
| 80B head | 1,718,255,616 | 3.079 FLOP/B | 25,343 stage-tok/s | 33,912 |

The ridge points are `43.5456 TF / 800 GB = 54.43 FLOP/B` and `58.269696 TF / 960 GB = 60.70 FLOP/B`. Both workloads are over an order of magnitude below the ridge point, proving that raw FP32 throughput cannot bind before DRAM.

Dequantization adds integer extraction, scale arithmetic, codebook lookup, and SFU pressure. A low-bound FP operation count per weight is approximately 2.0625 Q8, 2.1875 Q6, 2.125 IQ4, and 2.625 IQ3; this still leaves the workloads far below the ridge point, but does not faithfully price integer/table pipelines. The stronger practical proof is measurement.

### Measured dequant-aware roof

Every calibration matrix exceeded 96 MB and ran for 2,000 iterations:

| Format | Matrix weight size | XT | XTX |
|---|---:|---:|---:|
| F16/raw proxy | 256 MiB | 778.8 GB/s | 917.8 GB/s |
| Q8_0 | 136 MiB | 775.0 GB/s | 927.8 GB/s |
| Q6_K | 157.5 MiB | 606.5 GB/s | 800.4 GB/s |
| IQ4_XS | 136 MiB | 583.3 GB/s | 729.0 GB/s |
| IQ3_XXS | 147 MiB | 421.8 GB/s | 543.9 GB/s |

Summing `bytes_format ÷ measured_bandwidth_format` gives parameter-only practical ceilings of 272.16/336.86 tok/s for 35B and 1,220.79/1,513.05 stage-tok/s for 80B. Pricing state/KV/scratch at measured raw bandwidth gives:

| Workload | Context | XT practical ceiling | XTX practical ceiling |
|---|---:|---:|---:|
| 35B | 1,213 | **255.68 tok/s** | **315.49 tok/s** |
| 35B | 4,852 | **243.03 tok/s** | **299.20 tok/s** |
| 80B head | 1,213 | **1,123.30 stage-tok/s** | **1,386.50** |
| 80B head | 4,852 | **1,051.22 stage-tok/s** | **1,293.60** |

This is not a new theoretical limit—the kernels can improve their format throughput—but it is the most useful current-code ceiling.

## 4. Launch/barrier and cache bounds

With `S` serialized stages and the measured 1.4 µs stage floor:

```text
T/token ≥ max(B/BW, F/peak, S × 1.4 µs)
```

The original 35B profile had 445 barrier-delimited stages, a 0.623 ms floor. Selector-route fusion removes 40 and step-gate fusion removes 30, leaving approximately 375 stages or 0.525 ms. The 80B head's original 134 stages imply 0.188 ms; its final per-card paths remove 9–21 stages. Every launch floor is below the corresponding DRAM/dequant floor.

At prefill batch 128, the 35B and 80B paths have approximately 432 and 131 stages, only 4.725 and 1.433 µs/token after amortization. Queue-flush sweeps were flat, confirming that launch is not the remaining prefill limiter.

The full active payload cannot reside in cache: 2.46 GB is 31× the XT cache, while the 80B head's 503 MB is 5–6× cache. Even an impossible best case that permanently removes one full cache capacity from the active parameter stream only raises the bandwidth ceilings to:

| Workload | XT cache-best | XTX cache-best |
|---|---:|---:|
| 35B | 336.54 tok/s | 406.59 tok/s |
| 80B head | 1,892.37 stage-tok/s | 2,360.17 stage-tok/s |

These are the formal implementation-independent bandwidth ceilings: every token needs the full active parameter set, and at most one cache capacity can already be resident. They deliberately omit state, KV, scratch, and activation traffic, making them more permissive. They still sit far below the compute-only ceilings in Section 3, which proves bandwidth is the first possible roof on both cards.

Layer payloads (39–55 MB) do fit individually. That does not help n=1 weights, which are consumed once before advancing to a different layer, but it is central to prefill expert reuse. At `C=4,852`, one layer's unique KV is 19.9 MB; the old eight-query-head read pattern is about 159 MB and exceeds cache. The group-4 kernel reads it twice (~39.7 MB), fitting both cards and explaining the long-context gain. At much longer contexts this working set will again exceed cache.

## 5. Prefill floor and ceiling

Let `B` be batch width, `U_l` the number of unique routed experts selected in layer `l`, and `D_B` the number of distinct input-token embedding rows in that batch (`1 ≤ D_B ≤ B`). Under the cache-ideal model that counts every dense row and every distinct routed/embedding row once, the exact active parameter payload is

```text
P35(B) = 1,597,491,712 + Σ_l U_l R_l + 1,680 D_B
R_l    = 1,359,872 B normally; 1,662,976 B for Q6-down layers

P80(B) = 302,207,744 + Σ_12 U_l × 1,671,168 + 2,176 D_B
```

These equations count dense/shared/router weights once per batch and each routed expert slice once. Deterministically,

```text
n_used ≤ U_l ≤ min(n_expert, B × n_used).
```

Under independent uniform routing,

```text
E[U] = E × [1 - (1 - k/E)^B].
```

At `B=128`, `E[U35]=251.601` of 256 and `E[U80]=470.999` of 512. The tiny embedding term is shown using one issued row per token in this generic table:

| Workload | Best-case parameter B/token | Expected | Full-union worst case |
|---|---:|---:|---:|
| 35B | 15.94 MB | **121.19 MB** | 123.09 MB |
| 80B head | 3.93 MB | **76.16 MB** | 82.58 MB |

For the actual 1,213-token prompt under the baseline 128-token chunking, the 35B prefill sequence is `9×128 + 60` and the 80B sequence is `9×128 + 61`. The final partial chunk has materially less expert-union amortization than a full chunk. Recomputing it, using the prompt's exact 112 distinct embedding rows across the ten chunks, and adding ideal recurrent/KV traffic gives expected payloads of **133,035,436 B/token** for 35B and **81,237,788 B/token** for the 80B head. Therefore the B128 cache-ideal expected ceilings are:

| Workload | XT 800 GB/s | XTX 960 GB/s |
|---|---:|---:|
| 35B prefill | **6,013 tok/s** | **7,216 tok/s** |
| 80B-head prefill | **9,848 stage-tok/s** | **11,817 stage-tok/s** |

These expected ceilings are explicitly probabilistic and assume each unique expert row is fetched once per batch. The min/full-union columns are deterministic bounds on active unique parameter payload under that one-fetch model, not bounds on actual DRAM traffic. Exact prompt-specific DRAM bytes require route-union and cache/DRAM counters.

The prefill compute lower bounds for a 128-token chunk beginning at context `P` are approximately

```text
F35_prefill(P) = 4,983,521,280 + 163,840(P + 64.5) FLOP/token
F80_prefill(P) = 1,658,634,240 +  49,152(P + 64.5) FLOP/token.
```

Even at `P=4,852`, compute ceilings remain 7,522/10,066 tok/s for 35B and 22,915/30,664 stage-tok/s for 80B (XT/XTX). Compute remains above the expected bandwidth roof.

The final benchmarks use `QK_MAXB=1024` for 35B and 256 for 80B, which reduces repeated dense/union reads and raises the ideal ceiling further. Current token-major MoE dispatch does not realize the unique-expert assumption; reaching it requires sorted token/expert pairs and grouped expert GEMM/dequant reuse.

## 6. Tuning results

### Retained

- **Live split dispatch** (`QK_ATTN_LIVE_DISPATCH`): indirect grid uses `ceil(live/chunk)` instead of capacity `ceil(tmax/chunk)`.
- **Chunk 64**: common best after live-grid sweeps on both cards.
- **Adaptive GQA reuse** (`QK_ATTN_GQA_AUTO`): group1 below 2,048 tokens, group4 above. At 4,852 context, `at.split` fell from 15.79 to 0.97 ms on XT and 11.93 to 0.83 ms on XTX in paired samples.
- **256-lane selector** (`QK_MOE_SELECT_FAST`): subgroup top-k, deterministic lower-ID ties, exact shared-gate addition tree.
- **Router/selector fusion** (`QK_MOE_ROUTE_FUSED`): last router workgroup selects with device-scope release/acquire semantics, removing one stage per layer. Batch `n>1` automatically uses the standalone fast selector.
- **Register state** (`QK_DN_STEP_REG`): state row loaded once; `dn.step` fell 30.0% XT and 25.9% XTX.
- **Step+gate fusion** (`QK_DN_STEP_GATE_FUSED`): removes intermediate output and one stage per recurrent layer.
- **128-lane IQ4 down** (`QK_MOE_DOWN_128`): 5–6% routed-down stage win for 35B; used on the 80B XTX only after card-specific measurement.
- **Wider prefill**: MAXB1024 improves 35B prefill 13–19%; MAXB256 is the stable 80B point.

### Rejected or bounded

- Exact-emulation 256-lane selector: 20–21% slower selector; replaced.
- Dense GEMV half-TPR: every 35B projection regressed; no meaningful 80B gain; code removed.
- GQA group2/4/8 at short context: reuse overhead/under-occupancy loses; hence adaptive policy.
- GQA group8 at long context: too few workgroups; group4 wins.
- Submission depth 8→40: flat; retain safer ring-time default.
- Attention budget 2M→4M: only ~1% long-prefill gain; 4M is optional.
- `dn.proj`, `wo`, and head are already near their format-specific bandwidth roofs. The TPR experiment confirmed there is no simple occupancy win.
- Prior repack experiments (`9c4d3a`) measured 5.81 versus 5.82 ms and are not repeated; RoPE, host chunk sync, and already-fused addN/add3 were likewise excluded by prior evidence in the brief/history.
- True grouped prefill MoE remains the largest architectural lever. An expert-major scan without a sorted pair list merely exchanges weight traffic for `expert × token` selection work, so it was not shipped as a nominal optimization.

Full experiment chronology and all intermediate numbers are in [NOTES.md](NOTES.md).

## 7. Final configurations and achieved fraction of roof

35B common configuration:

```bash
QK_MAXB=1024
QK_ATTN_CHUNK=64
QK_ATTN_LIVE_DISPATCH=1
QK_ATTN_GQA_AUTO=1
QK_MOE_SELECT_FAST=1
QK_MOE_ROUTE_FUSED=1
QK_DN_STEP_GATE_FUSED=1
QK_MOE_DOWN_128=1
```

80B XT adds the same live/chunk/AUTO, fast selector, route fusion, fused step/gate, and `QK_MAXB=256`, but leaves IQ4 down at 256 lanes. The XTX uses the fast standalone selector and 128-lane IQ4 down.

The table uses the average roof over each measured, growing decode window (`C=1,213…1,468` for the 256-token windows and `C=4,852…4,915` for the 64-token windows), rather than comparing a window-average rate with only its starting context.

| Workload | Card | Achieved | Cold-stream roof | % cold | Measured practical roof | % practical |
|---|---:|---:|---:|---:|---:|---:|
| 35B decode C=1,213…1,468 | XT | 125.31 | 302.20 | 41.47% | 255.20 | **49.10%** |
| 35B decode C=1,213…1,468 | XTX | 178.67 | 362.64 | 49.27% | 314.89 | **56.74%** |
| 35B decode C=4,852…4,915 | XT | 131.71 | 285.56 | 46.12% | 242.92 | **54.22%** |
| 35B decode C=4,852…4,915 | XTX | 164.06 | 342.67 | 47.88% | 299.06 | **54.86%** |
| 80B head decode C=1,213…1,468 | XT | 427.19 | 1,429.08 | 29.89% | 1,120.57 | **38.12%** |
| 80B head decode C=1,213…1,468 | XTX | 631.72 | 1,714.89 | 36.84% | 1,382.98 | **45.68%** |

Prefill reaches 9.41/11.37% of the B128 cache-ideal expected ceiling for 35B and 13.96/16.27% for the 80B head (XT/XTX). Because final batch widths are larger than 128, those percentages are generous; the remaining gap is predominantly the unrealized grouped-expert reuse assumption.

## 8. Correctness and verification

Final-build `serve-test` was run on both cards with all common 35B flags:

| Gate | Immutable baseline SHA-256 | XT | XTX |
|---|---|---|---|
| `big_a`, 256 generated | `201f416edb24cc1f5c630bdfe66471d0f12442b53503e1f88da627853c3f60a4` | exact | exact |
| `big4`, 64 generated | `98b14c52ffa9f983059f81184b680b6a108a6ed84f3b1cb20bc314e33b4667ef` | exact | exact |

Final request times were 4,205.3/2,967.7 ms (`big_a`, XT/XTX) and 10,880.6/7,511.8 ms (`big4`). Both cards passed all five built-in format correctness cases. The system CTest binary reports no registered tests. `cmake --build build -j` and `git diff --check` pass.

After the campaign, `gemma/qk-server-split-80b` was restored to its original single replica and verified `1/1` ready/available. The XTX remained on `auto`; no GPU benchmark ran after restoration.

The 80B stage has no LM head in `[0,12)`, so verification uses byte hashes of all exported hidden rows. Repeated runs were deterministic; XT final hashes are `cde0335a3bb4f888` / `3331a4356b41c1d4`, and XTX card-best hashes are `d85e23e8adb10a02` / `d488fb24d6592dcd` (prefill/decode). The difference is expected because XTX enables the top10 128-lane down reduction; running either configuration on both cards yields matching hashes.

## 9. Remaining opportunity

The next material step is a grouped prefill MoE primitive: generate and sort `(expert, token, slot)` pairs, dispatch one expert-row tile over all matching tokens, and reuse dequantized rows. That directly attacks the 84–92% prefill roof gap. For decode, remaining fixed work is mostly format-limited expert/dense GEMV plus launch-separated residual/norm stages; closing much more than the current 38–57% practical-roof fraction requires broader multi-op fusion or a weight layout/kernel that raises IQ3/IQ4/Q6 effective bandwidth.
