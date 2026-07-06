//! Anthropic Messages API (`POST /v1/messages`) on top of the chat-completions
//! machinery: render the Anthropic conversation into the Qwen ChatML/hermes
//! tool format, generate, and map the output back to Anthropic content blocks
//! (`text` / `tool_use`), including the Anthropic SSE streaming protocol.

use std::time::Duration;

use async_stream::stream;
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::Stream;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::engine::FinishReason;
use crate::http::{
    AppState, FinishKind, Generation, Prompt, StopDetector, collect_generation,
    decode_tokens_lossless, parse_json, prepare_generation, require_json, submit_generation,
};
use crate::{Result, ServerError};

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Deserialize)]
pub struct MessagesReq {
    model: Option<String>,
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    system: Option<SystemField>,
    #[serde(default)]
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    tools: Vec<ToolDef>,
    #[serde(default)]
    stop_sequences: Vec<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SystemField {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Deserialize)]
struct SystemBlock {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<Block>),
}

#[derive(Deserialize)]
struct Block {
    #[serde(rename = "type")]
    kind: String,
    // text
    text: Option<String>,
    // tool_use
    name: Option<String>,
    input: Option<Value>,
    // tool_result
    content: Option<ToolResultContent>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ToolResultContent {
    Text(String),
    Blocks(Vec<Block>),
}

#[derive(Deserialize)]
struct ToolDef {
    name: String,
    description: Option<String>,
    input_schema: Option<Value>,
}

/// Render the Anthropic request into the Qwen3 ChatML prompt, following the
/// upstream Qwen chat template: tools advertised in the system turn inside
/// `<tools>` tags, assistant tool calls as `<tool_call>` JSON, and tool
/// results wrapped in `<tool_response>` inside a user turn.
fn render_prompt(req: &MessagesReq) -> Result<String> {
    let system = match &req.system {
        None => None,
        Some(SystemField::Text(text)) => Some(text.clone()),
        Some(SystemField::Blocks(blocks)) => Some(
            blocks
                .iter()
                .filter(|block| block.kind == "text")
                .filter_map(|block| block.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n\n"),
        ),
    };

    let mut out = String::new();
    if req.tools.is_empty() {
        if let Some(system) = &system {
            out.push_str("<|im_start|>system\n");
            out.push_str(system);
            out.push_str("<|im_end|>\n");
        }
    } else {
        out.push_str("<|im_start|>system\n");
        if let Some(system) = &system {
            out.push_str(system);
            out.push_str("\n\n");
        }
        out.push_str(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        );
        for tool in &req.tools {
            out.push('\n');
            out.push_str(
                &json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description.as_deref().unwrap_or(""),
                        "parameters": tool.input_schema.clone().unwrap_or_else(|| json!({})),
                    }
                })
                .to_string(),
            );
        }
        out.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function name and \
             arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
        );
    }

    let prefill = req
        .messages
        .last()
        .filter(|message| message.role == "assistant")
        .is_some();
    let turns = if prefill {
        &req.messages[..req.messages.len() - 1]
    } else {
        &req.messages[..]
    };
    for message in turns {
        match message.role.as_str() {
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(&render_user_content(&message.content)?);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                out.push_str(&render_assistant_content(&message.content)?);
                out.push_str("<|im_end|>\n");
            }
            role => {
                return Err(ServerError::bad_request(format!(
                    "unsupported message role: {role}"
                )));
            }
        }
    }

    out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    if prefill {
        let last = req.messages.last().expect("prefill checked non-empty");
        out.push_str(render_assistant_content(&last.content)?.trim_end());
    }
    Ok(out)
}

fn render_user_content(content: &AnthropicContent) -> Result<String> {
    let blocks = match content {
        AnthropicContent::Text(text) => return Ok(text.clone()),
        AnthropicContent::Blocks(blocks) => blocks,
    };
    let mut parts = Vec::new();
    for block in blocks {
        match block.kind.as_str() {
            "text" => parts.push(block.text.clone().unwrap_or_default()),
            "tool_result" => parts.push(format!(
                "<tool_response>\n{}\n</tool_response>",
                tool_result_text(block.content.as_ref())?
            )),
            "image" => {
                return Err(ServerError::bad_request(
                    "image content blocks are not supported",
                ));
            }
            _ => {}
        }
    }
    Ok(parts.join("\n"))
}

fn tool_result_text(content: Option<&ToolResultContent>) -> Result<String> {
    match content {
        None => Ok(String::new()),
        Some(ToolResultContent::Text(text)) => Ok(text.clone()),
        Some(ToolResultContent::Blocks(blocks)) => {
            let mut parts = Vec::new();
            for block in blocks {
                match block.kind.as_str() {
                    "text" => parts.push(block.text.clone().unwrap_or_default()),
                    "image" => {
                        return Err(ServerError::bad_request(
                            "image content blocks are not supported",
                        ));
                    }
                    _ => {}
                }
            }
            Ok(parts.join("\n"))
        }
    }
}

fn render_assistant_content(content: &AnthropicContent) -> Result<String> {
    let blocks = match content {
        AnthropicContent::Text(text) => return Ok(text.clone()),
        AnthropicContent::Blocks(blocks) => blocks,
    };
    let mut out = String::new();
    for block in blocks {
        match block.kind.as_str() {
            "text" => out.push_str(block.text.as_deref().unwrap_or("")),
            "tool_use" => {
                let call = json!({
                    "name": block.name.as_deref().unwrap_or(""),
                    "arguments": block.input.clone().unwrap_or_else(|| json!({})),
                });
                out.push_str("\n<tool_call>\n");
                out.push_str(&call.to_string());
                out.push_str("\n</tool_call>");
            }
            // thinking / redacted_thinking are not replayable into the prompt
            _ => {}
        }
    }
    Ok(out)
}

/// Incremental parser for generated text: passes plain text through, extracts
/// `<tool_call>{...}</tool_call>` blocks, and drops `<think>...</think>`
/// blocks. Text that could be the start of a tag is held back until resolved,
/// so it never emits a partial tag as text.
pub struct OutputParser {
    buf: String,
    state: ParseState,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ParseState {
    Text,
    Think,
    Tool,
}

pub enum ParsedEvent {
    Text(String),
    ToolCall { name: String, input: Value },
}

impl Default for OutputParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            state: ParseState::Text,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<ParsedEvent> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.state {
                ParseState::Text => {
                    let tool = self.buf.find(TOOL_OPEN).map(|pos| (pos, TOOL_OPEN, ParseState::Tool));
                    let think =
                        self.buf.find(THINK_OPEN).map(|pos| (pos, THINK_OPEN, ParseState::Think));
                    let first = match (tool, think) {
                        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
                        (a, b) => a.or(b),
                    };
                    if let Some((pos, marker, next)) = first {
                        if pos > 0 {
                            out.push(ParsedEvent::Text(self.buf[..pos].to_owned()));
                        }
                        self.buf.drain(..pos + marker.len());
                        self.state = next;
                        continue;
                    }
                    let keep = holdback_len(&self.buf, &[TOOL_OPEN, THINK_OPEN]);
                    let ready = self.buf.len() - keep;
                    if ready > 0 {
                        out.push(ParsedEvent::Text(self.buf[..ready].to_owned()));
                        self.buf.drain(..ready);
                    }
                    break;
                }
                ParseState::Think => {
                    if let Some(pos) = self.buf.find(THINK_CLOSE) {
                        self.buf.drain(..pos + THINK_CLOSE.len());
                        self.state = ParseState::Text;
                        continue;
                    }
                    break;
                }
                ParseState::Tool => {
                    if let Some(pos) = self.buf.find(TOOL_CLOSE) {
                        let raw = self.buf[..pos].trim().to_owned();
                        self.buf.drain(..pos + TOOL_CLOSE.len());
                        self.state = ParseState::Text;
                        out.push(parse_tool_call(&raw));
                        continue;
                    }
                    break;
                }
            }
        }
        out
    }

    pub fn finish(self) -> Vec<ParsedEvent> {
        match self.state {
            ParseState::Text if !self.buf.is_empty() => vec![ParsedEvent::Text(self.buf)],
            ParseState::Text | ParseState::Think => Vec::new(),
            // Generation ended (token limit) inside an unterminated tool call;
            // salvage it if the JSON happens to be complete.
            ParseState::Tool => vec![parse_tool_call(self.buf.trim())],
        }
    }
}

/// Longest suffix of `buf` that is a proper prefix of one of `markers`.
fn holdback_len(buf: &str, markers: &[&str]) -> usize {
    let max = markers
        .iter()
        .map(|marker| marker.len() - 1)
        .max()
        .unwrap_or(0)
        .min(buf.len());
    for take in (1..=max).rev() {
        if !buf.is_char_boundary(buf.len() - take) {
            continue;
        }
        let suffix = &buf[buf.len() - take..];
        if markers.iter().any(|marker| marker.starts_with(suffix)) {
            return take;
        }
    }
    0
}

fn parse_tool_call(raw: &str) -> ParsedEvent {
    #[derive(Deserialize)]
    struct Call {
        name: String,
        #[serde(default)]
        arguments: Value,
    }
    if let Ok(call) = serde_json::from_str::<Call>(raw) {
        // Some models emit arguments as a JSON-encoded string.
        let input = match call.arguments {
            Value::Null => json!({}),
            Value::String(text) => {
                serde_json::from_str(&text).unwrap_or(Value::String(text))
            }
            other => other,
        };
        return ParsedEvent::ToolCall {
            name: call.name,
            input,
        };
    }
    ParsedEvent::Text(format!("{TOOL_OPEN}\n{raw}\n{TOOL_CLOSE}"))
}

fn stop_reason(finish: FinishKind, tool_calls: usize) -> &'static str {
    if tool_calls > 0 {
        return "tool_use";
    }
    match finish {
        FinishKind::Eos => "end_turn",
        FinishKind::Limit => "max_tokens",
        FinishKind::Word => "stop_sequence",
    }
}

fn anthropic_error(err: ServerError) -> Response {
    let (status, kind) = match &err {
        ServerError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        ServerError::QueueFull(_) => (
            StatusCode::from_u16(529).unwrap_or(StatusCode::SERVICE_UNAVAILABLE),
            "overloaded_error",
        ),
        ServerError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
    };
    (
        status,
        Json(json!({
            "type": "error",
            "error": { "type": kind, "message": err.to_string() }
        })),
    )
        .into_response()
}

pub async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle_messages(state, headers, body).await {
        Ok(response) => response,
        Err(err) => anthropic_error(err),
    }
}

async fn handle_messages(state: AppState, headers: HeaderMap, body: Bytes) -> Result<Response> {
    require_json(&headers)?;
    let req: MessagesReq = parse_json(&body)?;
    if req.messages.is_empty() {
        return Err(ServerError::bad_request("messages must not be empty"));
    }
    let model = req.model.clone().unwrap_or_else(|| state.model_alias.clone());
    let prompt = render_prompt(&req)?;
    let generation = prepare_generation(&state, Prompt::Text(prompt), req.max_tokens)?;
    let rx = submit_generation(&state, &generation.prompt_ids, generation.max_gen)?;
    if req.stream {
        let stream = stream_messages(state, rx, generation, req.stop_sequences, model);
        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response());
    }
    let completed = collect_generation(&state, rx, &req.stop_sequences).await?;
    let mut parser = OutputParser::new();
    let mut events = parser.push(&completed.content);
    events.extend(parser.finish());

    let mut content = Vec::new();
    let mut tool_calls = 0usize;
    for event in events {
        match event {
            ParsedEvent::Text(text) => match content.last_mut() {
                Some(Value::Object(block)) if block["type"] == "text" => {
                    let existing = block["text"].as_str().unwrap_or("").to_owned();
                    block.insert("text".to_owned(), Value::String(existing + &text));
                }
                _ => content.push(json!({ "type": "text", "text": text })),
            },
            ParsedEvent::ToolCall { name, input } => {
                tool_calls += 1;
                content.push(json!({
                    "type": "tool_use",
                    "id": format!("toolu_{}", Uuid::new_v4().simple()),
                    "name": name,
                    "input": input,
                }));
            }
        }
    }
    // Drop whitespace-only text blocks (spacing around tool calls) and trim a
    // trailing text block, which Anthropic clients never expect to see.
    content.retain(|block| {
        block["type"] != "text" || !block["text"].as_str().unwrap_or("").trim().is_empty()
    });
    if let Some(Value::Object(block)) = content.last_mut()
        && block["type"] == "text"
    {
        let trimmed = block["text"].as_str().unwrap_or("").trim_end().to_owned();
        block.insert("text".to_owned(), Value::String(trimmed));
    }

    let reason = stop_reason(completed.finish, tool_calls);
    let stop_sequence = if completed.finish == FinishKind::Word {
        Value::String(completed.stopping_word.clone())
    } else {
        Value::Null
    };
    Ok(Json(json!({
        "id": format!("msg_{}", Uuid::new_v4().simple()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": reason,
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": generation.prompt_ids.len(),
            "output_tokens": completed.tokens.len(),
        }
    }))
    .into_response())
}

/// Streaming content-block bookkeeping: lazily opens a text block on the
/// first non-whitespace text (whitespace before that is buffered so spacing
/// between tool calls does not become an empty text block), and closes it
/// when a tool_use block starts.
struct BlockEmitter {
    index: usize,
    text_open: bool,
    pending_ws: String,
    tool_calls: usize,
}

impl BlockEmitter {
    fn new() -> Self {
        Self {
            index: 0,
            text_open: false,
            pending_ws: String::new(),
            tool_calls: 0,
        }
    }

    fn on_event(&mut self, event: ParsedEvent) -> Vec<Event> {
        match event {
            ParsedEvent::Text(text) => {
                if !self.text_open {
                    if text.trim().is_empty() {
                        self.pending_ws.push_str(&text);
                        return Vec::new();
                    }
                    let delta = std::mem::take(&mut self.pending_ws) + &text;
                    self.text_open = true;
                    return vec![
                        sse(
                            "content_block_start",
                            json!({
                                "type": "content_block_start",
                                "index": self.index,
                                "content_block": { "type": "text", "text": "" }
                            }),
                        ),
                        text_delta(self.index, &delta),
                    ];
                }
                vec![text_delta(self.index, &text)]
            }
            ParsedEvent::ToolCall { name, input } => {
                self.pending_ws.clear();
                self.tool_calls += 1;
                let mut events = Vec::new();
                if self.text_open {
                    events.push(block_stop(self.index));
                    self.index += 1;
                    self.text_open = false;
                }
                events.push(sse(
                    "content_block_start",
                    json!({
                        "type": "content_block_start",
                        "index": self.index,
                        "content_block": {
                            "type": "tool_use",
                            "id": format!("toolu_{}", Uuid::new_v4().simple()),
                            "name": name,
                            "input": {}
                        }
                    }),
                ));
                events.push(sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": self.index,
                        "delta": { "type": "input_json_delta", "partial_json": input.to_string() }
                    }),
                ));
                events.push(block_stop(self.index));
                self.index += 1;
                events
            }
        }
    }

    fn finish(&mut self) -> Vec<Event> {
        if self.text_open {
            self.text_open = false;
            let index = self.index;
            self.index += 1;
            return vec![block_stop(index)];
        }
        Vec::new()
    }
}

fn sse(name: &str, data: Value) -> Event {
    Event::default().event(name).data(data.to_string())
}

fn text_delta(index: usize, text: &str) -> Event {
    sse(
        "content_block_delta",
        json!({
            "type": "content_block_delta",
            "index": index,
            "delta": { "type": "text_delta", "text": text }
        }),
    )
}

fn block_stop(index: usize) -> Event {
    sse(
        "content_block_stop",
        json!({ "type": "content_block_stop", "index": index }),
    )
}

fn stream_messages(
    state: AppState,
    mut rx: tokio::sync::mpsc::Receiver<crate::engine::SlotEvent>,
    generation: Generation,
    stops: Vec<String>,
    model: String,
) -> impl Stream<Item = std::result::Result<Event, axum::Error>> {
    stream! {
        let msg_id = format!("msg_{}", Uuid::new_v4().simple());
        yield Ok(sse("message_start", json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": { "input_tokens": generation.prompt_ids.len(), "output_tokens": 0 }
            }
        })));
        yield Ok(sse("ping", json!({ "type": "ping" })));

        let mut stop = StopDetector::new(&stops);
        let mut parser = OutputParser::new();
        let mut emitter = BlockEmitter::new();
        let mut finish = FinishKind::Limit;
        let mut stopping_word = String::new();
        let mut output_tokens = 0usize;
        while let Some(event) = rx.recv().await {
            match event {
                crate::engine::SlotEvent::Tokens(ids) => {
                    output_tokens += ids.len();
                    let text = decode_tokens_lossless(&state, &ids).unwrap_or_default();
                    if let Some(hit) = stop.push(&text) {
                        finish = FinishKind::Word;
                        stopping_word = hit;
                        break;
                    }
                    for parsed in parser.push(&stop.take_ready()) {
                        for event in emitter.on_event(parsed) {
                            yield Ok(event);
                        }
                    }
                }
                crate::engine::SlotEvent::Done { reason } => {
                    finish = if reason == FinishReason::Eos { FinishKind::Eos } else { FinishKind::Limit };
                    break;
                }
                crate::engine::SlotEvent::Error(message) => {
                    yield Ok(sse("error", json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": message }
                    })));
                    return;
                }
            }
        }

        let mut tail = parser.push(&stop.finish());
        tail.extend(parser.finish());
        for parsed in tail {
            for event in emitter.on_event(parsed) {
                yield Ok(event);
            }
        }
        for event in emitter.finish() {
            yield Ok(event);
        }

        let reason = stop_reason(finish, emitter.tool_calls);
        let stop_sequence = if finish == FinishKind::Word {
            Value::String(stopping_word)
        } else {
            Value::Null
        };
        yield Ok(sse("message_delta", json!({
            "type": "message_delta",
            "delta": { "stop_reason": reason, "stop_sequence": stop_sequence },
            "usage": { "output_tokens": output_tokens }
        })));
        yield Ok(sse("message_stop", json!({ "type": "message_stop" })));
    }
}

pub async fn count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle_count_tokens(state, headers, body) {
        Ok(response) => response,
        Err(err) => anthropic_error(err),
    }
}

fn handle_count_tokens(state: AppState, headers: HeaderMap, body: Bytes) -> Result<Response> {
    require_json(&headers)?;
    let req: MessagesReq = parse_json(&body)?;
    let prompt = render_prompt(&req)?;
    let tokens = state.tokenizer.tokenize(&prompt, true)?;
    Ok(Json(json!({ "input_tokens": tokens.len() })).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_from_json(body: Value) -> MessagesReq {
        serde_json::from_value(body).expect("request parses")
    }

    #[test]
    fn renders_system_tools_and_tool_flow() {
        let req = req_from_json(json!({
            "model": "claude-test",
            "max_tokens": 128,
            "system": [{ "type": "text", "text": "Be brief." }],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
            }],
            "messages": [
                { "role": "user", "content": "Weather in Paris?" },
                { "role": "assistant", "content": [
                    { "type": "text", "text": "Checking." },
                    { "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": { "city": "Paris" } }
                ]},
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "toolu_1", "content": "18C, sunny" }
                ]}
            ]
        }));
        let prompt = render_prompt(&req).expect("renders");
        assert!(prompt.starts_with("<|im_start|>system\nBe brief.\n\n# Tools"));
        assert!(prompt.contains("<tools>\n{"));
        assert!(prompt.contains("\"name\":\"get_weather\""));
        assert!(prompt.contains("\n</tools>\n\nFor each function call"));
        assert!(prompt.contains("<|im_start|>assistant\nChecking.\n<tool_call>\n{"));
        assert!(prompt.contains("\"city\":\"Paris\""));
        assert!(prompt.contains("}\n</tool_call><|im_end|>\n"));
        assert!(prompt.contains(
            "<|im_start|>user\n<tool_response>\n18C, sunny\n</tool_response><|im_end|>\n"
        ));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn renders_plain_system_string_and_prefill() {
        let req = req_from_json(json!({
            "max_tokens": 16,
            "system": "You are terse.",
            "messages": [
                { "role": "user", "content": "List one color as JSON." },
                { "role": "assistant", "content": "{\"color\":" }
            ]
        }));
        let prompt = render_prompt(&req).expect("renders");
        assert!(prompt.starts_with("<|im_start|>system\nYou are terse.<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n{\"color\":"));
    }

    #[test]
    fn rejects_images_and_unknown_roles() {
        let image = req_from_json(json!({
            "max_tokens": 16,
            "messages": [{ "role": "user", "content": [
                { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "" } }
            ]}]
        }));
        assert!(render_prompt(&image).is_err());
        let role = req_from_json(json!({
            "max_tokens": 16,
            "messages": [{ "role": "tool", "content": "x" }]
        }));
        assert!(render_prompt(&role).is_err());
    }

    fn collect(events: Vec<ParsedEvent>) -> Vec<(String, String)> {
        events
            .into_iter()
            .map(|event| match event {
                ParsedEvent::Text(text) => ("text".to_owned(), text),
                ParsedEvent::ToolCall { name, input } => (name, input.to_string()),
            })
            .collect()
    }

    #[test]
    fn parser_passes_plain_text() {
        let mut parser = OutputParser::new();
        let mut events = parser.push("hello ");
        events.extend(parser.push("world"));
        events.extend(parser.finish());
        let text: String = collect(events)
            .into_iter()
            .map(|(_, text)| text)
            .collect();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn parser_extracts_tool_calls_and_drops_think() {
        let output = "<think>reasoning</think>I will check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>";
        // Feed at every possible split point to exercise holdback.
        for split in 0..=output.len() {
            if !output.is_char_boundary(split) {
                continue;
            }
            let mut parser = OutputParser::new();
            let mut events = parser.push(&output[..split]);
            events.extend(parser.push(&output[split..]));
            events.extend(parser.finish());
            let got = collect(events);
            let text: String = got
                .iter()
                .filter(|(kind, _)| kind == "text")
                .map(|(_, text)| text.as_str())
                .collect();
            assert_eq!(text.trim(), "I will check.", "split at {split}");
            let tools: Vec<_> = got.iter().filter(|(kind, _)| kind != "text").collect();
            assert_eq!(tools.len(), 2, "split at {split}");
            assert_eq!(tools[0].0, "get_weather");
            assert_eq!(tools[0].1, "{\"city\":\"Paris\"}");
            assert_eq!(tools[1].0, "get_time");
        }
    }

    #[test]
    fn parser_falls_back_on_malformed_tool_json() {
        let mut parser = OutputParser::new();
        let mut events = parser.push("<tool_call>\nnot json\n</tool_call>");
        events.extend(parser.finish());
        let got = collect(events);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "text");
        assert!(got[0].1.contains("not json"));
    }

    #[test]
    fn parser_salvages_unterminated_tool_call() {
        let mut parser = OutputParser::new();
        let mut events =
            parser.push("<tool_call>\n{\"name\": \"ls\", \"arguments\": {\"dir\": \"/\"}}");
        events.extend(parser.finish());
        let got = collect(events);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "ls");
    }

    #[test]
    fn parser_unwraps_string_encoded_arguments() {
        let mut parser = OutputParser::new();
        let mut events = parser.push(
            "<tool_call>{\"name\": \"ls\", \"arguments\": \"{\\\"dir\\\": \\\"/tmp\\\"}\"}</tool_call>",
        );
        events.extend(parser.finish());
        let got = collect(events);
        assert_eq!(got[0].0, "ls");
        assert_eq!(got[0].1, "{\"dir\":\"/tmp\"}");
    }
}
