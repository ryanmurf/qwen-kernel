# qk-server — Rust inference server specification

## Context

This repo contains a from-scratch Vulkan compute inference engine (C++) for
Qwen3.6-35B-A3B (`qwen35moe`) on an AMD RX 7900 XT. The engine already decodes
batched multi-request streams with greedy sampling, token-for-token identical
to llama.cpp. What's missing is a **server**: HTTP surface, tokenizer, chat
templating, slot scheduling, streaming.

**You are building that server, in Rust, as the crate `server/` (binary
`qk-server`).** The GPU engine is consumed through a tiny C ABI
(`include/qk.h`) loaded at runtime via `dlopen` (libloading). A deterministic
stub implementation (`stub/qk_stub.c`) exists so the entire crate builds and
tests **without any GPU**.

Safety rationale (this is the point of the exercise): everything that touches
untrusted input — HTTP parsing, JSON, tokenization, template rendering, UTF-8
handling — lives in safe Rust. The `unsafe` FFI surface is one module, one
thread, five functions.

## Hard requirements

1. `cargo build --release` and `cargo test` pass **offline after initial
   `cargo fetch`**, with no GPU, no model file, no network at test time.
2. `cargo clippy --all-targets -- -D warnings` is clean.
3. `unsafe` code only in `src/ffi.rs`.
4. Tokenizer passes every fixture in `tests/fixtures/tokenizer_fixtures.json`
   (repo root; see "Fixtures") **exactly**.
5. No panics on any request input: malformed JSON, wrong types, huge bodies,
   invalid UTF-8, out-of-range token ids, oversized prompts → structured JSON
   errors with proper status codes. `unwrap`/`expect` only where infallible by
   construction (justify with a comment) or at startup.
6. Rust edition 2024, stable toolchain (cargo 1.96 installed).

Suggested deps: axum 0.8, tokio (full), serde/serde_json, minijinja (latest,
with `loader` + `builtins` as needed), fancy-regex, aho-corasick, memmap2,
libloading, clap (derive), tracing + tracing-subscriber, anyhow/thiserror,
uuid, futures. Avoid heavyweight extras; no openssl (plain HTTP).

## CLI

```
qk-server --model /path/model.gguf [--host 0.0.0.0] [--port 8090]
          [--slots 8] [--ctx 1024] [--chunk 8]
          [--engine-lib ../build/libqk.so] [--queue 64]
          [--chat-template auto|builtin] [--log-level info]
```

`--engine-lib` also via env `QK_ENGINE_LIB`. Model path is the real GGUF; the
server reads **only the KV metadata section** (tokenizer + template); tensor
data is the engine's business.

## GGUF v3 metadata reader (`src/gguf.rs`)

Read-only, `memmap2`, little-endian. Layout from file start:

```
u32 magic = 0x46554747 ("GGUF")   u32 version (=3)
u64 n_tensors                      u64 n_kv
then n_kv entries: { string key; u32 type; value }
string = { u64 len; len bytes UTF-8 (not NUL-terminated) }
types: 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool(1B),8 string,
       9 array { u32 elem_type; u64 count; count elems }, 10 u64,11 i64,12 f64
```

Stop after the KV section — tensor info is not needed. Validate magic/version;
bound every length against file size (mmap of a 16 GB file — lengths are
attacker-ish data, don't trust). Keys needed:

| key | type | value in our model |
|---|---|---|
| `general.architecture` | str | `qwen35moe` (warn if different) |
| `general.name` | str | `Qwen3.6-35B-A3B` (model alias) |
| `tokenizer.ggml.model` | str | must be `gpt2` |
| `tokenizer.ggml.pre` | str | must be `qwen35` |
| `tokenizer.ggml.tokens` | [str] | 248320 entries |
| `tokenizer.ggml.token_type` | [i32] | 248320 entries |
| `tokenizer.ggml.merges` | [str] | 247587 entries, `"left right"` |
| `tokenizer.ggml.eos_token_id` | u32 | 248046 (`<|im_end|>`) |
| `tokenizer.ggml.bos_token_id` | u32 | 248044 (`<|endoftext|>`) |
| `tokenizer.ggml.add_bos_token` | bool | **false** — never prepend BOS |
| `tokenizer.chat_template` | str | ~8 kB Jinja |

Token types: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED, 6=BYTE.
This vocab: 248044 normal, 27 control, 6 user_defined (`<tool_call>`,
`</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>`),
243 unused `[PAD…]`.

## Tokenizer (`src/tokenizer.rs`) — byte-level BPE, must match llama.cpp

### 1. Special-token partition

Specials = tokens with type 3 (CONTROL) or 4 (USER_DEFINED).

- `parse_special == true`: match both kinds against raw input text.
- `parse_special == false`: match **only USER_DEFINED** (CONTROL text like
  `<|im_start|>` stays plain text and goes through BPE).

Use `aho-corasick` with `MatchKind::LeftmostLongest` over the applicable
special strings; matched spans become single tokens, the text between spans
goes to step 2. (Longest matters: `<tts_text_bos>` vs `<tts_text_bos_single>`.)

### 2. Pretokenizer regex (qwen35)

Apply per fragment with `fancy_regex` (needs the lookahead), exact pattern:

```
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Every char of the fragment lands in exactly one match (concatenation of
matches == fragment). Iterate matches in order.

### 3. Byte mapping (GPT-2 byte-level)

Vocab/merge strings live in "byte-mapped" space. Build the classic GPT-2
byte↔unicode table: bytes `0x21..=0x7E`, `0xA1..=0xAC`, `0xAE..=0xFF` map to
themselves (as chars); every other byte `b` maps to `char(0x100 + n)` where
`n` is its index in ascending order of those remaining bytes (so 0x00→Ā,
0x20→Ġ, 0x0A→Ċ, …). Encode each regex piece: UTF-8 bytes → mapped chars.

### 4. BPE merges

Merges file line k (0-based) = rank k, format `"left right"` (split on the
single space; sides are byte-mapped strings). Start from single-char symbols;
repeatedly merge the adjacent pair with the **lowest rank**; among equal-rank
candidates merge the **leftmost occurrence first**; repeat until no adjacent
pair has a rank. Then map each resulting symbol via exact vocab lookup
(`HashMap<String, u32>` built from `tokenizer.ggml.tokens`). Every single
mapped char exists in this vocab, so lookup cannot fail on well-formed data;
return an internal error rather than panicking if it somehow does.

Implementation note: the classic linked-list + binary-heap of (rank, pos)
algorithm is fine; correctness bar is the fixtures, perf bar is "tokenizes a
100 kB prompt in well under a second".

### 5. Detokenize

`id → tokens[id]`, reverse the byte map char-by-char, concatenate bytes, then
UTF-8-decode. Rules:

- NORMAL and USER_DEFINED tokens always render their text.
- CONTROL tokens render only when `render_special` (the `/detokenize`
  endpoint uses `render_special = false`; internal streaming never emits
  control ids anyway). USER_DEFINED strings (`<think>`…) are plain text — not
  byte-mapped — same for CONTROL when rendered.
- **Streaming-safe incremental decoding**: a token can end mid-UTF-8-sequence
  (emoji split across tokens). Keep a per-stream byte buffer; only flush the
  longest valid-UTF-8 prefix; hold incomplete trailing bytes for the next
  token. Never emit U+FFFD for bytes that merely haven't completed yet; emit
  U+FFFD only for genuinely invalid sequences at end-of-stream.

### Fixtures (`tests/fixtures/tokenizer_fixtures.json`, repo root)

Captured from the live llama.cpp (build b8672) serving this exact GGUF:

- `tokenize`: `[{text, tokens, detok}]` — 28 cases (unicode, emoji, CJK,
  contractions, whitespace runs, code, 100×'a', …). Semantics:
  `add_special=false`, `parse_special=true` (none of the texts contain
  special strings, so the flag is moot here). Your `tokenize(text)` must
  equal `tokens`; your `detokenize(tokens)` must equal `detok` (== `text`).
- `tokenize_special`: same text with `parse_special` true (10 tokens) and
  false (25 tokens) — asserts partition semantics.
- `chat`: `[{messages, prompt}]` — 3 cases; rendered chat template output.

## Chat template (`src/template.rs`)

Load `tokenizer.chat_template` from GGUF into minijinja
(`add_messages: messages, add_generation_prompt: true` context conventions —
match the fixtures). Register `raise_exception(msg)` as a function returning a
template error. The template uses `namespace()`, macros, `is string` /
`is mapping` / `is iterable` tests, loops — current minijinja handles these
(enable `minijinja` features as needed, e.g. `builtins`). The context for
rendering: `{ messages: [...], add_generation_prompt: true }` where message
`content` is a plain string (multimodal array content: if a client sends
OpenAI-style `content: [{type:"text", text:"…"}]`, concatenate the text parts
into one string before rendering; reject image/audio parts with 400).

Must reproduce the 3 `chat` fixtures byte-exactly. If rendering the GGUF
template fails at startup, or `--chat-template builtin` is set, fall back to a
hardcoded ChatML formatter that also reproduces the fixtures (derive its exact
shape from the fixture `prompt` strings — including the trailing
`<|im_start|>assistant` opening and any `<think>` handling present there).

## FFI (`src/ffi.rs`) — the only unsafe module

Mirror `include/qk.h` exactly (repo root). Load via `libloading` at startup
from `--engine-lib`. Wrap in:

```rust
pub struct Engine { /* lib + raw ptr, private */ }
impl Engine {
    pub fn open(lib: &Path, model: &Path, cfg: QkConfig) -> Result<Self>;
    pub fn slot_start(&mut self, slot: u32, prompt: &[u32], max_gen: u32) -> Result<()>;
    pub fn slot_cancel(&mut self, slot: u32);
    pub fn step_chunk(&mut self, out: &mut StepOut) -> Result<u32>; // active count
    pub fn n_vocab(&self) -> u32; /* etc */
}
```

`Engine` is `Send` but NOT `Sync`; it lives on one dedicated OS thread
(std::thread), created at startup, which owns all engine calls until shutdown
(`qk_close` on drop). Buffer sizes for `step_chunk` come from
`n_slots`/`chunk` getters — allocate once. Validate `out_counts[s] <= chunk`
and token ids `< n_vocab` after each call (defend against a buggy engine lib
— this crosses a trust boundary).

## Engine thread & scheduler (`src/engine.rs`)

- Requests arrive as `Job { prompt_ids: Vec<u32>, max_gen: u32, events: tokio::sync::mpsc::Sender<SlotEvent> }`
  via an `std::sync::mpsc::Sender<Cmd>` handed to HTTP handlers
  (`Cmd::Submit(Job)`; cancellation via dropping the event receiver — see
  below).
- Engine thread loop:
  1. Drain pending `Cmd`s without blocking; if no slot is active, block on
     `recv()` (parks the thread; zero GPU work when idle).
  2. Admit queued jobs to free slots FIFO (`qk_slot_start`). Queue bound =
     `--queue`; enforcement happens on the HTTP side (semaphore/counter) so
     the handler can return 503 immediately.
  3. If any slot active: `qk_step_chunk`; for each slot with tokens →
     `events.blocking_send(SlotEvent::Tokens(vec))`; finished bit →
     `SlotEvent::Done { reason }` (Eos or Limit) and free the slot record.
  4. A failed/closed `blocking_send` (client gone) → `qk_slot_cancel(slot)`.
- `max_gen` for a job = `min(n_predict_requested, n_ctx - n_prompt - margin)`;
  jobs whose prompt alone is `>= n_ctx` are rejected at the HTTP layer with
  400 and llama.cpp-style message.
- Stop strings are enforced on the HTTP side (it owns detok text); on hit it
  just drops the receiver — engine notices on next send and cancels the slot.
- Graceful shutdown (SIGINT/SIGTERM): stop accepting, drop all jobs, join
  engine thread, `qk_close`.

## HTTP API (`src/http.rs`) — llama.cpp-server-compatible subset

Content-Type `application/json`; body limit 4 MB; errors as
`{"error":{"code":<http>,"message":"…","type":"…"}}` (llama.cpp shape).

- `GET /health` → `{"status":"ok"}`.
- `POST /tokenize` `{content, add_special?=false, parse_special?=true,
  with_pieces?=false}` → `{"tokens":[u32]}` or with_pieces:
  `{"tokens":[{"id":u32,"piece":str}]}`. (add_special is a no-op for this
  model: add_bos=false.)
- `POST /detokenize` `{tokens:[u32]}` → `{"content":str}`
  (render_special=false; out-of-range ids → 400).
- `GET /props` → `{"model_path":…, "model_alias":…, "chat_template":…,
  "total_slots":n_slots, "default_generation_settings":{"n_ctx":ctx}, "modalities":{"vision":false,"audio":false}}`.
- `GET /v1/models` → OpenAI list, one entry, `id` = model alias.
- `POST /completion` (alias `/completions`):
  request: `prompt` (string → tokenize with parse_special=true, or array of
  token ids; mixed arrays not required), `n_predict`/`max_tokens` (default:
  fill to ctx), `stream?=false`, `stop?:[str]`, `return_tokens?=false`,
  `id_slot?` ignored, sampling params accepted-and-ignored (greedy engine —
  log once per request at debug). Non-stream response:
  `{content, tokens? , stop:true, model, tokens_predicted, tokens_evaluated,
  stop_type:"eos"|"limit"|"word", stopping_word:"…", timings:{prompt_n,
  prompt_ms, predicted_n, predicted_ms, predicted_per_second}}`.
  Stream (`text/event-stream`): chunks `data: {"content":"…delta…",
  "stop":false, "tokens":[…]?}` then a final
  `data: {"content":"", "stop":true, …full fields as non-stream…}`. No
  `[DONE]` sentinel (llama.cpp doesn't send one here).
- `POST /v1/completions`: OpenAI text-completion shape over the same path
  (`choices:[{text, index:0, finish_reason:"stop"|"length"}]`, `usage`).
  Stream: OpenAI chunks + terminal `data: [DONE]`.
- `POST /v1/chat/completions`: messages → template → prompt → generate.
  Stop at EOS (`<|im_end|>`). Response/stream per OpenAI: first stream chunk
  carries `delta:{role:"assistant"}`, then content deltas, final chunk
  `finish_reason:"stop"|"length"`, then `data: [DONE]`. `usage` on non-stream
  (and on stream final chunk when `stream_options.include_usage`).

Streaming mechanics: SSE via axum; flush per event; heartbeat not required.
Stop-string handling must hold back a suffix of emitted text equal to the
longest stop string minus 1 chars (llama.cpp behavior) so a stop string
arriving across token boundaries is caught before reaching the client; on
hit, truncate at the match, `stop_type:"word"`, `stopping_word` set.

Client disconnect (stream or not) must free the slot promptly (drop of the
event receiver / axum connection close → abort the forwarding task).

## Testing

- **Unit**: byte map table (spot: 0x20→'Ġ', 0x0A→'Ċ', 'A'→'A'); regex split
  cases; BPE on tiny hand-built vocab; incremental UTF-8 holdback (emoji split
  across two tokens); stop-string holdback across chunks.
- **Fixture tests**: all three fixture groups, exact.
- **Integration** (`tests/server.rs`): `build.rs` compiles
  `stub/qk_stub.c` → `$OUT_DIR/libqk_stub.so` (plain `cc -shared -fPIC`,
  re-run-if-changed) and exports the path via `cargo:rustc-env=QK_STUB_LIB`.
  Tests boot the full server (random port, tiny synthetic GGUF written to a
  tempdir by test code — build a minimal valid GGUF v3 with a ~300-token toy
  vocab, merges, template; the tokenizer under test is constructed from GGUF
  bytes, so this also tests the reader) against the stub lib and assert:
  - `/health`, `/tokenize`↔`/detokenize` roundtrip on the toy vocab;
  - `/completion` non-stream: prompt of token ids `[5,6,8]`, `n_predict:4` →
    exactly the stub recurrence from seed 8, 4 ids, `stop_type:"limit"`;
  - `/completion` stream: same ids arrive as SSE, final chunk `stop:true`;
  - EOS: prompt `[7,9]` → 3 generated ids then `stop_type:"eos"`;
  - concurrency: 4 simultaneous streaming requests with distinct seeds each
    get exactly their own recurrence (no cross-stream leakage), while
    `total_slots` may be 2 — i.e. 2 queue then run;
  - 503 when queue cap exceeded; 400 on malformed JSON / oversized prompt /
    bad token ids.
  For fixture-dependent tests (real vocab), locate the fixtures file via
  `CARGO_MANIFEST_DIR/../tests/fixtures/tokenizer_fixtures.json`; these run
  without the model file (they need no GGUF — construct the tokenizer from
  the arrays embedded in a helper that parses fixtures? No: fixtures don't
  carry the vocab). → Real-vocab fixture tests DO need the real GGUF; gate
  them behind `env QK_MODEL_GGUF` (skip with a clear message when unset).
  Everything else must run without it.

## Definition of done

`cargo fmt` applied; `cargo clippy --all-targets -- -D warnings` clean;
`cargo test` green (model-gated tests skipped without `QK_MODEL_GGUF`, all
run green with it); `server/README.md` documents CLI, endpoints, and the
dlopen contract; no `unsafe` outside `src/ffi.rs`.

## Facts appendix

- Model file (for gated tests / manual runs):
  `/home/ryan/intellij/ggerganov/llama.cpp/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`
- vocab 248320; merges 247587; eos 248046 `<|im_end|>`; bos 248044
  `<|endoftext|>` (add_bos=false); pad 248055.
- Control tokens (type 3), ids 248044–248076 range:
  `<|endoftext|>, <|im_start|>, <|im_end|>, <|object_ref_start|>,
  <|object_ref_end|>, <|box_start|>, <|box_end|>, <|quad_start|>,
  <|quad_end|>, <|vision_start|>, <|vision_end|>, <|vision_pad|>,
  <|image_pad|>, <|video_pad|>, <|fim_prefix|>, <|fim_middle|>,
  <|fim_suffix|>, <|fim_pad|>, <|repo_name|>, <|file_sep|>, <|audio_start|>,
  <|audio_end|>, <tts_pad>, <tts_text_bos>, <tts_text_eod>,
  <tts_text_bos_single>, <|audio_pad|>` — read them from the GGUF, don't
  hardcode.
- User-defined (type 4): `<tool_call> </tool_call> <tool_response>
  </tool_response> <think> </think>` (ids 248058/248059/248066/248067/248068/248069).
- Engine step timing (real lib): ~6 ms/step at 1 active slot to ~42 ms at 16;
  size HTTP timeouts accordingly (none by default; rely on disconnect).
