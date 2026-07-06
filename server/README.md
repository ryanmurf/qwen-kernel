# qk-server

Rust HTTP inference server for the qwen-kernel engine.

## Build

```bash
cargo build --release
```

The binary is `target/release/qk-server`.

## Run

```bash
qk-server --model /path/model.gguf \
  --engine-lib ../build/libqk.so \
  --host 0.0.0.0 \
  --port 8090 \
  --slots 8 \
  --ctx 1024 \
  --chunk 8 \
  --queue 64
```

`--engine-lib` can also be supplied as `QK_ENGINE_LIB`.

The server reads tokenizer metadata and the chat template from the GGUF file.
Tensor data is handled only by the engine library.

## Endpoints

- `GET /health`
- `GET /props`
- `GET /v1/models`
- `POST /tokenize`
- `POST /detokenize`
- `POST /completion`
- `POST /completions`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/messages` (Anthropic Messages API)
- `POST /v1/messages/count_tokens`

All request bodies must use `Content-Type: application/json`. Error responses
use the llama.cpp-style JSON shape:

```json
{"error":{"code":400,"message":"...","type":"invalid_request_error"}}
```

### Anthropic Messages (`/v1/messages`)

Speaks enough of the Anthropic Messages API for the Claude CLI: `system`
(string or text blocks), `messages` with `text`/`tool_use`/`tool_result`
content blocks, `tools`, `stop_sequences`, assistant prefill, and SSE
streaming (`message_start` / `content_block_*` / `message_delta` /
`message_stop`). Requests are rendered into the Qwen3 ChatML format (tools as
a hermes-style `<tools>` system block, tool results as `<tool_response>`);
generated `<tool_call>` JSON blocks are parsed back into Anthropic `tool_use`
content blocks, and stray `<think>` blocks are dropped. `stop_reason` maps to
`end_turn` / `max_tokens` / `stop_sequence` / `tool_use`. Errors from this
endpoint use the Anthropic error shape
(`{"type":"error","error":{"type":"...","message":"..."}}`). Image content is
rejected.

## Engine Library Contract

The engine library is loaded at startup with `dlopen` through `libloading`.
It must export the C ABI in `../include/qk.h`.

All engine calls are made from one dedicated OS thread. HTTP, JSON parsing,
tokenization, chat rendering, stop-string handling, and streaming text handling
remain in Rust before requests cross into the engine.

## Tests

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```

The build script compiles `../stub/qk_stub.c` into a shared library for tests,
so normal test runs do not need a GPU or model file.

The real Qwen tokenizer fixtures are gated by `QK_MODEL_GGUF`:

```bash
QK_MODEL_GGUF=/path/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf cargo test real_vocab
```
