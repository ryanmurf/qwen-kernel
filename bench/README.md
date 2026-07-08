# CLI-path benchmark: qk-server vs llama.cpp Vulkan (2026-07-08)

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
two. It benefited from being current master (3 months newer than the
previously deployed build). Sampling differed (temp 0.2 vs greedy), which
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
