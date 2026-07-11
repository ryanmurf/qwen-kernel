# Split-model serving — design (task #19)

> **STATUS 2026-07-10: P0–P2 shipped.** `--split-next` + the split driver
> (`server/src/split.rs`) are live; offline coverage via the stub
> (`tests/split_driver.rs`). All four gates passed on the 7900 XT
> (head `QK_LAYERS=0:20` + `qk pipe-worker 20:40`, ctx 8192, slots 2):
> (1) split `/v1/messages` byte-identical to unsplit (text, usage,
> stop_reason); (2) streamed ≡ non-streamed; (3) worker killed mid-stream →
> partial tokens + clean SSE `error` / HTTP 500, head survives, next request
> after worker restart reconnects lazily and succeeds; (4) two concurrent
> sequences both match their unsplit references.
>
> **P3 CROSS-BOX: PASSED 2026-07-10.** tron (RX 7900 XT, Vulkan, layers
> [0,22)) drove midnight (M4 Max, Metal engine from the metal-port branch,
> layers [22,40), `qk pipe-worker :18100`, ctx 32768) over WiFi:
> `qk pipe` GEN **token-exact** vs the single-GPU reference, and the full
> `qk-server --split-next` HTTP path **byte-identical** (text + usage).
> Decode 17.2 ms/tok engine-level (s1 5.1 + s2+net 12.1; worker compute
> ~5 ms, WiFi RTT ~4.3 ms avg is the tax — wire the Mac for ~0.3 ms), 50
> tok/s streamed end-to-end at 200 tokens. Two GPU vendors, two APIs, two
> engines — one bit-exact model. Descoped: per-stage recorded step CBs
> (~2 ms/tok single-stream on the head vs the async driver's aggregate win).
>
> **Clean sweep 2026-07-10 (flash kernel + async driver, Mac idle, WiFi
> RTT ~3.7 ms):**
>
> | S (head layers) | single ms/tok (s1 + s2+net) | 2-stream aggregate |
> |---|---|---|
> | 22 | **17.0** (4.4 + 12.6) | **12.5 ms/tok = 80 tok/s** |
> | 33 | 21.9 (11.1 + 10.8) | 13.6 ms/tok |
>
> S=22 wins both axes: the worker's ~7 ms fixed per-frame cost (Metal
> encode overhead, layer-count-insensitive) + WiFi RTT dominate, so
> shifting layers to tron only inflates s1 (which also grows
> super-linearly through the batched-n=1 path). The async driver fully
> hides the head's work at 2 streams (aggregate = worker+net bound).
>
> **WIRED (2026-07-10, RTT 4.3→1.26 ms; WiFi had degraded to 18 ms avg
> that day):**
>
> | link | single (engine) | single (HTTP) | 2-stream aggregate |
> |---|---|---|---|
> | WiFi | 17.0 ms/tok | ~19 ms/tok | 12.5 ms/tok (80 tok/s) |
> | wired | **10.7 ms/tok** (s1 4.2 + s2+net 6.6) | ~13.4 ms/tok | **9.5 ms/tok (105 tok/s)** |
>
> Token-exact throughout. Still worker-bound under load (~8-9 ms effective
> per frame back-to-back), so slots=4 stays pointless until midnight's
> per-frame overhead shrinks — that's the single remaining lever.
> qk-server-split.yaml now targets the wired IP (REDACTED-LAN-IP).

Serve one model as N pipeline stages on N devices, behind the existing
qk-server HTTP/Anthropic layer. Builds on the engine's pipeline split
(`QK_LAYERS=a:b` + `qk_stage_run`, commit 240c63e), which is validated
token-exact against the unsplit engine at every tested boundary, in-process
and over TCP.

**Format compatibility = split GGUF files** (llama.cpp `-%05d-of-%05d.gguf`,
already supported by `Gguf::open`). llama.cpp's RPC wire protocol is
explicitly out of scope — our stage transport is our own.

## Why (and why now)

- **One box = staging only.** Two stages on the same GPU still hold the full
  weight set between them (~15.4 GB + ~0.5 GB overhead) and the same total
  KV — there is **no capacity win on a single card**, and both stages
  serialize on the same hardware, so no throughput win either. The
  single-box shape exists to validate the choreography (which it did:
  token-exact, all failure drills green) at ~6.4–6.7 ms/tok vs 5.8 serial.
- **The win is per-NODE, i.e. cross-box**: tron (RDNA3/Vulkan) + midnight
  (M4 Max/Metal, port at M5 — same ABI, server unmodified). Each node then
  holds ~half the weights (~8.5 GB), freeing ~7 GB per node for KV: slots/ctx
  can grow well past today's slots=2/ctx=16384 ceiling. The engine is
  hard-wired to Qwen3.6-35B-A3B, so cross-box split first means more
  slots/ctx/headroom for *this* model; larger models additionally need a
  second hard-wired architecture, which is its own project.

## Topology

```
client ── HTTP ──> qk-server (HEAD)                     qk pipe-worker (WORKER)
                   tokenizer, template, SSE,            engine only, no Rust:
                   admission, slot bookkeeping          QK_LAYERS=20:40
                   engine stage QK_LAYERS=0:20  ──TCP──>  stage_run per frame
                   (embedding side)             <──ids──  (head side owns EOS)
```

- **Head** = the stage that owns the embedding (`lFirst == 0`). It keeps the
  entire serving brain: tokenizer, chat template, admission, stop conditions,
  SSE. Only the forward pass is distributed.
- **Worker** = `qk pipe-worker <port> <a:b>` from the engine CLI. No Rust on
  the worker node — one C++ binary + shaders + its GGUF shard(s). (A
  `--worker` mode inside qk-server can come later if we want auth/metrics;
  it is not needed for correctness.)
- v1 is exactly **two stages**. N>2 would chain workers (each forwards hidden
  rows to the next; last returns ids to the head) — deferred until a third
  device exists.

## Stage transport (v1 = the `qk pipe` frame, unchanged)

16-byte header `{u32 op, u32 slot, u32 n, u32 base}`, little-endian, then:

| dir | eng stage | payload |
|---|---|---|
| head→worker | op=1 | `n * 2048` f32 hidden rows (positions `[base, base+n)`) |
| worker→head | reply | `n` u32 greedy ids (worker is the last stage) |
| head→worker | op=2 | none — connection close |

- One TCP connection, `TCP_NODELAY`, blocking, single in-flight request —
  matches the engine-thread model on both ends. 8 KB/token each way is
  nothing on loopback (~0.1 ms/tok measured) and fine on LAN (2.5 GbE ≈
  0.03 ms serialization + RTT ~0.2 ms).
- `slot` is carried in the frame and maps 1:1 to the worker's slot — the
  worker's engine is opened with the same `--slots/--ctx`, so per-slot state
  (KV, DeltaNet S) stays coherent by construction as long as the head sends
  every position exactly once per slot in order. `base==0` resets the slot on
  the worker (that IS slot_start); cancel needs no message (the next sequence
  on that slot starts at base 0 and resets).
- Connection hello (added ahead of the Metal worker): the client sends magic
  `0x716b7031` ("qkp1"); the worker replies `{magic, layer_first, layer_end,
  n_layer, n_embd, n_slots, n_ctx}`. The head requires a contiguous split
  reaching the final layer, matching n_embd, and worker slots/ctx covering
  its own — so mixed builds or a mis-launched worker fail loudly at connect,
  never mid-stream. Bump the magic on any wire change.

## Head-side server changes (the actual work)

`server/src/engine.rs` today drives `qk_slot_start`/`qk_step_chunk` on one
engine thread. Split mode replaces those two calls with stage choreography —
everything above (admission, slots, trimming, SSE) is untouched.

1. **ffi.rs**: bind `qk_stage_run`, `qk_layer_first/end`, `qk_n_layer`,
   `qk_n_embd` — via `Library::get` at load, *optional* (old libqk.so still
   serves unsplit; split mode errors cleanly if symbols are missing).
2. **Config**: `--split-next <host:port>` enables split mode; the engine lib
   is opened with `QK_LAYERS` already in the env (deployment sets it, e.g.
   `0:20`). Sanity-check at startup: `qk_layer_first == 0` (head must embed),
   `qk_layer_end < qk_n_layer` (there must be a remote tail).
3. **Split driver** (new, inside the engine thread loop):
   - *start(slot, prompt)*: `stage_run(slot, toks, n, base=0, hidden_out)`
     locally (chunks internally at maxB), send frames to the worker as each
     ≤maxB hidden chunk lands (don't buffer the whole prompt), keep the last
     id of the final reply = first generated token.
   - *step(slot)*: local `stage_run(n=1)` → frame → id. Round-robin active
     slots per loop pass; emit per token into the existing chunk-emission
     plumbing (counts of 1).
   - *stop*: EOS/max_gen/stop-string handled on the head exactly as today.
   - Worker socket: one per server, owned by the engine thread; reconnect
     with backoff on error, failing active requests (5xx) but not the server.
4. **Explicitly OFF in split mode v1** (each is a follow-up with a real
   design, not silently skipped): prefix cache (`QK_FORK`), history-boundary
   snapshots (`snap_prefix` → pass 0), spec decode (`QK_SPEC` — verify
   rounds would need scratch-stripe copies mirrored on the worker).
   Deployment must not set those envs on either stage; the server warns if
   it sees them alongside `--split-next`.

   Design note for cross-turn reuse in split mode (task #30): driver-side
   "warm slot" continuation does NOT work — the gated-DeltaNet state cannot
   rewind, and the fed history is never an exact token prefix of the next
   turn's prompt (the re-rendered `<|im_end|>` was never generated;
   retokenization of generated text can drift). Reuse requires the
   history-boundary snapshot mirrored on the worker: additive engine ABI
   `qk_state_save/qk_state_load(slot, idx)` over the existing copyStripes /
   pcache-entry machinery, wire ops op3/op4 carrying `{slot, idx}` (hello
   magic bumps to "qkp2"), and a head-side token-key table mapping cached
   prefixes to snapshot indices. The head snapshots both sides at
   `snap_prefix` during prefill; the next turn restores both sides and
   prefills only its delta.

## Correctness gates (mirror the engine harness)

1. `qk pipe` GEN parity already proves the math; the server gate is
   end-to-end: same prompt through unsplit qk-server vs split qk-server →
   byte-identical non-streamed response body (greedy).
2. Streamed vs non-streamed identical through the split path.
3. Kill the worker mid-generation → request fails 5xx, server stays up,
   next request after worker restart succeeds (fresh base=0 reset).
4. Two concurrent sequences (slots=2) interleave correctly — distinct
   prompts, both streams match their unsplit references.

## Performance expectations (set now, measure then)

- Single stream, one box, split 20: ~150 tok/s equivalent (6.6 ms/tok) vs
  172 unsplit — the price of two half-model passes through the batched-n=1
  path plus a hop. Cross-box adds LAN RTT (~0.2–0.5 ms/tok).
- Balance rule for heterogeneous devices: pick the boundary so per-stage
  wall time matches (`s1 ≈ s2` in `qk pipe` output), not layer counts —
  the M4 Max stage will want fewer layers than the 7900 XT stage.
- Throughput under load: v1 drives stages sequentially per token, so the
  pipeline is idle half the time per stream; two active slots naturally
  overlap (slot B's stage-1 while slot A is on the worker) only with an
  async driver — that is the first optimization after v1, not part of it.

## Plan

| phase | work | gate |
|---|---|---|
| P0 | design doc (this file) + ffi bindings for the stage ABI | server builds against stub with new optional symbols |
| P1 | `--split-next` + split driver in engine.rs, worker = `qk pipe-worker` | localhost 2-stage: parity gates 1–3 |
| P2 | slots=2 interleave + failure drills + timing table in this doc | gate 4, numbers recorded |
| P3 | deploy manifest for the 2-stage-on-one-box shape (bigger ctx/slots experiment) | slots/ctx sweep vs unsplit baseline |

## Two-node deployment (runbook)

Pieces (all in `deploy/`): `qk-server-split.yaml` (the tron head,
`QK_LAYERS=0:22` + `--split-next midnight:18100`, replicas 0 by default),
`midnight-qk-worker.plist` (launchd template for the Metal worker —
KeepAlive, caffeinate, args `port layers ctx slots`), and `switch.sh split`
to flip client traffic (scales the other backends to 0 first; the
`gemma-server` Service keeps routing by the shared `gpu-llm: server` label).

- **Startup order does not matter.** The head connects lazily; a request
  against a down worker returns a clean 5xx and the head stays up.
- **Worker restart** (crash, launchd relaunch): in-flight requests fail 5xx,
  the head voids its snapshot registry (a restarted worker lost its half),
  reconnects on the next request, and new sequences start at base 0 —
  no head bounce needed.
- **Sizing contract**: worker `slots`/`ctx` must cover the head's, and the
  layer ranges must be contiguous to layer 40 — the qkp2 hello rejects any
  mismatch at connect with a specific error. Keep `QK_PCACHE` equal on both
  sides (it is the snapshot entry count for split cross-turn reuse).
- **Applying the yaml resets replicas to 0** (deliberate, switch.sh owns
  scaling) — always `./switch.sh split` (or `kubectl scale ... --replicas=1`)
  after `kubectl apply`.
- **Probe**: the hello doubles as a health check —
  `python3 -c "import socket,struct; s=socket.create_connection((HOST,18100),5); s.sendall(struct.pack('<I',0x716b7032)); print(struct.unpack('<7I', s.recv(28)))"`
- Split mode v1 serves without `QK_FORK`/`QK_SPEC` (the server warns if
  set); cross-turn reuse comes from the split driver's own boundary
  snapshots (`[split-cache]` log lines).

## Qwen3-Next-80B-A3B-Instruct (the second split backend, 2026-07-10)

The 42.8 GB IQ4_XS-qk repack (`/mnt/data/models/`, byte-identical copy in
midnight's `models/`) cannot fit either box alone: tron head takes layers
`[0,12)` (~12 GB VRAM incl. embd + 3 attn layers' KV at ctx 32768), midnight
takes `[12,48)` (~32 GB wired) plus the final norm/head. Same qkp2 contract
as the 35B split; everything above applies unchanged, plus:

- Backend: `deploy/qk-server-split-80b.yaml`, flipped by `./switch.sh split80`.
  Worker prereq on midnight: `QK_MLOCK=1 qk pipe-worker 18200 12:48 32768 2`.
  The 35B :18100 worker is a separate process; both can be resident.
- The server auto-detects the model shape: `qwen3next` arch, `qwen2`
  pre-tokenizer, optional bos, and — user-visible — the generation cue:
  Instruct templates get a plain `<|im_start|>assistant\n` (no think
  scaffold). Hermes `<tool_call>` parsing is unchanged.
- **Arch trap (cost us the first gate)**: qwen3next's DeltaNet pairs v-heads
  *consecutively* with k-heads (k-head g serves v-heads 2g/2g+1) where
  qwen35moe tiles modulo. `dn_step*`'s `kDiv` push constant carries this;
  it is set from `general.architecture` at open. A wrong pairing produces
  *coherent but wrong-context* generations (recurrent state compounds a
  per-token error) — bisect with the QK_DUMP_* hooks against llama.cpp
  eval-callback dumps (see 64192d5).

### Gate semantics and results (commit 64192d5 + midnight ce3ef7f)

Token-exactness vs llama.cpp greedy references is NOT byte-guaranteed for
this model: llama quantizes activations (Q8_K) for its IQ4_XS matmuls while
qk reads f32 activations (per-op checks put qk ~1e-6 from exact math, llama
~1% off it). At a genuine near-tie the two legitimately pick different
tokens. The gate is therefore: greedy prefix-exact up to a *certified*
near-tie — every divergence must land on a position where llama's own top-2
logprob gap is small (< ~0.15), with qk picking one of llama's top choices.

Measured through the real split (in-cluster head, WiFi to midnight):
- ref3: 100/100 tokens exact; ref1: exact to token 35 then a 0.006-logprob
  tie (' the' vs ' Russia'); ref2: exact to 41 then a 0.11 tie. Text-prompt
  path byte-identical to token-ids path (tokenizer parity).
- Determinism x2 exact; /v1/messages round trip clean (`end_turn`).
- Steady decode ~30 tok/s single-stream (256-token run, prefill excluded);
  ~33 tok/s e2e on short runs.
