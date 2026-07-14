# Prod 80B: standalone on midnight (2026-07-14)

The 80B is no longer split across tron and midnight. midnight's M4 Max serves
all 48 layers by itself, faster than the two boxes together did, and faster than
llama.cpp does on the same machine:

| | split (tron head + midnight worker) | **standalone (midnight)** | llama.cpp Metal |
|---|---:|---:|---:|
| decode, warm, end-to-end | 50.7 tok/s | **71.5 tok/s** | — |
| decode, engine | — | 92.5 tok/s | 70.6 tok/s (tg128) |
| prefill | — | 1029 tok/s | 1017 tok/s (pp512) |

The split existed because the model "fits neither box". That was false for a
64 GB M4 Max — 42.9 GB of IQ4_XS weights fit, and the wire was never the
bottleneck (a WiFi/ethernet A/B came out identical).

## Launch (midnight, tmux `qk80-server`)

```sh
cd ~/IdeaProjects/nes-experimental/qk-metal/qwen-kernel
QK_MLOCK=1 QK_MAXB=512 QK_PCACHE=3 QK_PCACHE_LOG=1 \
QK_TEMP_DEFAULT=0.7 QK_TEMP_CAP=0.8 QK_STATS_FILE=/tmp/qk-stats.log \
caffeinate -i ./server/target/release/server \
  --model ../models/Qwen3-Next-80B-A3B-Instruct-IQ4_XS-qk.gguf \
  --engine-lib ./build/libqk.dylib \
  --host 0.0.0.0 --port 8080 --slots 2 --ctx 49152 --chunk 8 \
  --local-driver
```

Why each non-obvious knob:

- `QK_MLOCK=1` — **not optional.** Without it macOS reclaims the weight mapping
  during idle: after ~30 min the server's pages were all "inactive" (45 GB), and
  requests re-faulted them from disk — decode DECAYED under load (43 -> 24 ->
  16 tok/s) instead of warming up, while `/health` stayed 200. With it, 49.9 GB
  stays wired and decode holds. MTLResidencySet (the fix for the 2026-07-13
  panic) governs GPU-side residency; it does not stop the kernel from evicting
  the CPU mapping, which is what mlock is for.
- `--local-driver` — **sampling**. `qk_step_chunk` picks the next token inside
  the engine's fused argmax chain, so a driver cannot inject a sampled one and
  the single-box path was greedy-only. The local driver runs the same engine one
  position at a time through `qk_stage_run` + `qk_stage_topk`, which puts the
  choice back on the server (temperature/top-p work again — task #44's
  loop-breaker, without which agent turns relapse into byte-identical tool
  trajectories). Costs ~6% decode (75.9 -> 71.5 tok/s). Greedy is byte-identical
  either way.
- `QK_MAXB=512` — the grouped-MoE prefill path only engages at N >= 192. Unset,
  cold prefill runs ~250 tok/s instead of ~520.
- `QK_PCACHE=3` — cross-turn snapshots. In `--local-driver` these are the
  DRIVER's `[split-cache]` entries (qk_state_save/load), not the engine's
  internal pcache. **Do not set `QK_FORK`** here: it only gates the engine's own
  `qk_slot_start` path, which the driver never calls, and both index the same
  entries (qk.h). Without the driver (plain `qk_step_chunk` serving) the
  opposite holds — the engine only snapshots under `QK_FORK`, and leaving it
  unset makes every agent turn re-prefill its whole history.
- `caffeinate -i` — the lid/idle sleep would otherwise take prod down.

Measured on a 4,745-token agent-shaped prompt: cold turn ~19 s, warm turns
**0.26 s** (reuse=4742 prefill=3), with occasional ~2.8 s outliers — those track
memory pressure, see below.

Decode is **55-67 tok/s (median ~62)** on a machine the user is actively working
on (GatherV2/Slack/WindowServer competing for the same GPU), and 71-73 tok/s when
it is quiet. The FIRST request after an idle gap is slow (13-20 tok/s) even with
mlock — Metal re-warms its residency — and then it settles. Compare: the split it
replaced was a steady 50.7.

## The front door

`gemma-server` (REDACTED-CLUSTER-IP) still fronts everything, so no client changes.
It has NO selector any more; `deploy/gemma-server-midnight.yaml` gives it
hand-written Endpoints pointing at tron:18080, where `qk80-tunnel.service`
forwards to midnight's localhost:8080.

The tunnel is not architecture. midnight's macOS application firewall blocks
inbound connections to the unsigned qk-server binary, and allowlisting it needs
an interactive sudo on that box (ssh cannot answer the prompt):

```sh
B=~/IdeaProjects/nes-experimental/qk-metal/qwen-kernel/server/target/release/server
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$B"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$B"
```

After that, point the Endpoints at REDACTED-LAN-IP:8080 and disable the unit.

## What this costs, and the ceiling

midnight holds 42.9 GB of weights plus KV and snapshots on a 64 GB machine that
is also the dev laptop. Observed during testing: ~150 MB free, 3.7 GB of swap in
use — which is where the occasional 2.8 s warm turn comes from (a snapshot
restore that pages). If those outliers start to hurt, `QK_PCACHE=2` is the first
knob; a lower `--ctx` is the second.

Prod's fate now rides on that one laptop. It kernel-panicked on 2026-07-13 under
the standalone load — root-caused (PORT.md) as watchdogd starvation from the
42.9 GB run paging against the 32 GB-wired split worker, i.e. BOTH engines
resident at once, and fixed by an MTLResidencySet over the weight mapping. Do
not start the pipe-worker and the standalone server together.

The 128 GB node arriving end of July 2026 is the natural home for this: same
topology, none of the memory pressure.
