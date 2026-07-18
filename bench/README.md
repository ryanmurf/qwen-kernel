# CLI-path benchmark: qk-server vs llama.cpp Vulkan (2026-07-08)

> **SUPERSEDED — read `#refresh-2026-07-18` below before quoting any number here.**
> The llama.cpp baseline in this document is commit `25eec6f32` (tag b8672),
> authored **2026-04-05** — three months before the benchmark date, despite
> being labelled "master". llama.cpp's Vulkan backend improved substantially in
> that window. A same-day re-measurement against llama.cpp `571d0d5` puts the
> decode advantage at **1.34x (7900 XT) / 1.44x (7900 XTX)**, not the ~5x this
> table implies. The end-to-end CLI numbers below have **not** been re-run and
> should not be quoted.

End-to-end comparison of the two switchable backends on this box, measured
through the **real client path** — the Claude Code CLI (`claude-qwen` wrapper,
Claude Code v2.1.204, `-p --output-format json`) — not raw curl. That means
every number includes the wrapper's system prompt, tool schemas, SSE
streaming, and the CLI's concurrent small-model side-requests. Zero errors on
either backend; tool calls (`num_turns=2`) and `--resume` multi-turn worked on
both.

## Result

| scenario (median wall, cold in parens) | qk-server | llama.cpp | qk advantage |
|---|---|---|---|
| S0 ping — overhead/TTFT proxy | 0.8s (3.5) | 0.8s (10.9) | ~equal warm |
| S1 generate ~350 tok | **5.3s** (8.1) | 32.5s (35.8) | **6.1×** |
| S2 Bash tool call, 2-turn loop | **1.5s** (4.8) | 6.3s (11.0) | **4.3×** |
| S3 multi-turn ×4, per turn | **2.3s** (6.7) | 14.3s (23.7) | **6.2×** |
| warm decode through the CLI | **65 tok/s** | 12 tok/s | **5.4×** |

Raw single-stream decode on qk (curl, no CLI side-traffic): **~140 tok/s**
(400 tok / 2.85 s). The CLI-path number is lower because the wrapper routes
its auxiliary model calls to the same engine, so two streams share the GPU —
qk overlaps them across its 2 slots; llama.cpp (`--parallel 1`) serializes
them, which is part (not most) of its S2/S3 gap.

Peak junction: qk 104 °C, llama 98 °C (qk runs hotter because it keeps the
GPU busier; throttle point is 110 °C).

## Configuration

Same model file for both: `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (~15.8 GB), single
RX 7900 XT (20 GB, RADV), one backend scaled up at a time (`deploy/switch.sh`).

- **qk-server**: this repo @ `cf55162`, slots=2, ctx=16384, chunk=8,
  `QK_FORK=1` (cross-turn prefix KV reuse), `QK_MAX_TOOL_CHARS=4000`.
  17.6 GB VRAM. Greedy decode.
- **llama.cpp**: master `25eec6f32` (built 2026-07-08, image
  `llama-server:vulkan-anthropic` — rebuilt because the previous b8671 image
  predates llama.cpp's Anthropic `/v1/messages` endpoint, without which the
  CLI cannot talk to it). Production args: `-ngl 99 -fa on -ctk q8_0 -ctv
  q8_0 -c 262144 --jinja --temp 0.2 --top-p 0.95 --top-k 20`. 20.1 GB VRAM.

Fairness caveats, in llama.cpp's favor and against: it ran with a 16× larger
context reservation (262144 vs 16384) and quantized KV — its production
config, not a tuned-for-benchmark one; it also ran single-slot where qk had
two. It was NOT current master: `25eec6f32` is tag b8672, authored 2026-04-05.
The previously deployed build was b8671 - the immediately preceding tag,
authored hours earlier. The claim that it was "3 months newer" was false;
only the *build* was three months newer than the source. Sampling differed (temp 0.2 vs greedy), which
affects output length slightly, not tok/s.

The cold-vs-warm spread on qk is its prefix cache (`[pcache]` hit on repeated
system prompt + history); llama.cpp's slot prompt cache plays the same role
but the gap is dominated by decode speed, where the specialized kernels win
~5× (see `docs/speculative-decoding.md` §5 for why generic Vulkan MoE paths
struggle at batch 1 on this hybrid DeltaNet+MoE architecture).

## Reproduce

```bash
# backend A
./deploy/switch.sh qk
python3 bench/qbench.py qk-server outdir/
# backend B
./deploy/switch.sh gemma       # needs the :vulkan-anthropic image
python3 bench/qbench.py llamacpp outdir/
```

Raw per-call records: `results-qk-server.jsonl`, `results-llamacpp.jsonl`
(wall time, `duration_api_ms`, tokens, turns, junction temp per call).

---

## Refresh (2026-07-18) {#refresh-2026-07-18}

The July 8 table above compared against a llama.cpp source tree from **April 5**.
This section re-measures decode against llama.cpp `571d0d5` (authored
2026-07-18, the same day), same model file, same cards, matched KV precision.

### Engine-level decode, same model, same day

`Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (15.45 GiB, 34.66B params), f16 KV both sides.
llama.cpp: `llama-bench -ngl 99 -fa on -ctk f16 -ctv f16 -n 128 -p 0 -r 5`.
qk: `qk serve-bench tests/ids3.txt 128` with the tuned flag set below.

| card | qk (tuned) | llama.cpp `571d0d5` | qk advantage |
|---|---:|---:|---:|
| RX 7900 XTX | **190.7 tok/s** | 132.25 +/- 0.91 | **1.44x** |
| RX 7900 XT | **147.1 tok/s** | 109.73 +/- 0.18 | **1.34x** |

qk samples: XTX 191.40 / 189.91; XT 147.25 / 146.88. Untuned qk on XT is
139.58 (140.01 / 139.62 / 139.11), so the tuned flags are worth ~5.4%:

```
QK_CHUNK=8 QK_MAXB=1024 QK_ATTN_CHUNK=64 QK_ATTN_LIVE_DISPATCH=1 \
QK_ATTN_GQA_AUTO=1 QK_MOE_SELECT_FAST=1 QK_MOE_ROUTE_FUSED=1 \
QK_DN_STEP_GATE_FUSED=1 QK_MOE_DOWN_128=1
```

llama.cpp depth sensitivity on XT: 109.73 (d0), 108.98 (d1213), 105.61 (d4852).

Raw records: `results-2026-07-18-refresh.jsonl`.

### What changed, and why the advantage shrank

llama.cpp's Vulkan backend got substantially faster between April and July.
The April baseline is what produced the ~5x CLI-path gap; a like-for-like
decode comparison today is **1.34-1.44x**. qk also improved over the same
period (XTX 178.67 in the July notes vs 190.7 today), so both moved - but
llama.cpp moved more.

### Caveats that still apply

- **These are decode-only, near-zero context** (21-token prompt). They are not
  the end-to-end CLI numbers in the table above, which have not been re-run.
- The tuned-flag qk numbers use a flag set documented only in
  `docs/amd-opt/NOTES.md`, not in the README defaults.
- The GPUs are shared with other workloads on this box. `gpu_busy_percent` was
  confirmed 0-1% immediately before each recorded run; an earlier XTX
  llama.cpp attempt during contention returned 126.01 +/- 11.21 and was
  discarded. Always check contention before recording.
- The `172 tok/s` figure in the top-level README is an **XTX** number
  (`docs/amd-opt/NOTES.md`: XTX 178.67, XT 125.31 at C=1213), but the README
  attributes it to the 7900 XT. The cards differ by ~20% of memory bandwidth.
- The `big_a` / `big4` fixtures behind the C=1,213 and C=4,852 numbers in
  `docs/amd-opt/NOTES.md` are **not in this repo**, so those specific figures
  are not externally reproducible. `tests/ids3.txt` is, and is what the refresh
  above uses. `tests/ids2.txt` and `tests/ids4.txt` currently produce zero
  tokens under `serve-bench` (`produced=0`) and need investigation.
