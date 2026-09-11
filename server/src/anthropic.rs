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

use crate::engine::{FinishReason, Sampling};
use crate::http::{
    AppState, FinishKind, Generation, Prompt, StopDetector, collect_generation,
    decode_tokens_lossless, parse_json, prepare_generation, require_json, resolve_sampling,
    submit_generation,
};
use crate::{Result, ServerError};

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Token offset of the conversation-history boundary — the prompt with the
/// trailing generation cue (`state.chat_template.gen_cue()`, the exact scaffold
/// `render_prompt` appends) stripped. Snapshotting the KV there (rather
/// than after the whole prompt) makes turn N's cache entry a genuine prefix of
/// turn N+1, which appends the reply + next turn *after the same history*, so
/// N+1 restores it and prefills only its delta. Returns 0 (disabled) when the
/// prompt doesn't end in the scaffold (e.g. an assistant-prefill continuation),
/// which is always safe: matchPrefix verifies tokens before reusing, so an
/// off/zero boundary only forgoes reuse, never corrupts output.
fn history_boundary(state: &AppState, prompt: &str) -> u32 {
    match prompt.strip_suffix(state.chat_template.gen_cue()) {
        Some(history) => state
            .tokenizer
            .tokenize(history, true)
            .map(|ids| ids.len() as u32)
            .unwrap_or(0),
        None => 0,
    }
}

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
    temperature: Option<f32>,
    top_p: Option<f32>,
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

/// Serializes JSON the way Python's `json.dumps` does (`", "` and `": "`
/// separators). The Qwen chat template renders tools and past tool calls with
/// `| tojson`, so this is the shape the model saw in training and the shape it
/// emits inside `<tool_call>` — history must round-trip byte-identically or
/// the model starts imitating the degraded shape mid-session.
struct SpacedFormatter;

impl serde_json::ser::Formatter for SpacedFormatter {
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }

    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
}

fn spaced_json(value: &Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, SpacedFormatter);
    serde::Serialize::serialize(value, &mut ser).expect("serializing Value cannot fail");
    String::from_utf8(out).expect("serde_json output is UTF-8")
}

/// Tool name plus its top-level input-schema property names, used to recover
/// a generated tool call whose JSON is missing the `name` field.
#[derive(Clone)]
pub struct ToolSpec {
    name: String,
    keys: Vec<String>,
    properties: Value,
}

fn tool_specs(tools: &[ToolDef]) -> Vec<ToolSpec> {
    tools
        .iter()
        .map(|tool| ToolSpec {
            name: tool.name.clone(),
            properties: tool
                .input_schema
                .as_ref()
                .and_then(|schema| schema.get("properties"))
                .cloned()
                .unwrap_or(Value::Null),
            keys: tool
                .input_schema
                .as_ref()
                .and_then(|schema| schema.get("properties"))
                .and_then(Value::as_object)
                .map(|props| props.keys().cloned().collect())
                .unwrap_or_default(),
        })
        .collect()
}

/// If every argument key belongs to exactly one advertised tool's schema
/// (or only one tool is advertised at all), that tool must be the target.
fn infer_tool_name(input: &Value, tools: &[ToolSpec]) -> Option<String> {
    if let [only] = tools {
        return Some(only.name.clone());
    }
    let args = input.as_object()?;
    if args.is_empty() {
        return None;
    }
    let mut matches = tools
        .iter()
        .filter(|tool| args.keys().all(|key| tool.keys.contains(key)));
    let hit = matches.next()?;
    matches.next().is_none().then(|| hit.name.clone())
}

/// Render the Anthropic request into the Qwen3 ChatML prompt, following the
/// upstream Qwen chat template: tools advertised in the system turn inside
/// `<tools>` tags, assistant tool calls as `<tool_call>` JSON, and tool
/// results wrapped in `<tool_response>` inside a user turn.
fn render_prompt(req: &MessagesReq, gen_cue: &str) -> Result<String> {
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
            out.push_str(&spaced_json(&json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description.as_deref().unwrap_or(""),
                    "parameters": tool.input_schema.clone().unwrap_or_else(|| json!({})),
                }
            })));
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
            // Claude Code >= 2.2 sends system reminders as in-messages system
            // turns (older CLIs only used the top-level `system` field).
            "system" => {
                out.push_str("<|im_start|>system\n");
                out.push_str(&render_user_content(&message.content)?);
                out.push_str("<|im_end|>\n");
            }
            role => {
                return Err(ServerError::bad_request(format!(
                    "unsupported message role: {role}"
                )));
            }
        }
    }

    out.push_str(gen_cue);
    if prefill {
        let last = req.messages.last().expect("prefill checked non-empty");
        out.push_str(render_assistant_content(&last.content)?.trim_end());
    }
    Ok(out)
}

/// Flash's GGUF template owns the XML tool format and open reasoning cue.
/// Preserve system priority when converting Claude's in-message reminders:
/// this template accepts system content only in the initial turn.
fn render_flash_prompt(
    req: &MessagesReq,
    template: &crate::template::ChatTemplate,
) -> Result<String> {
    let mut system = match &req.system {
        None => String::new(),
        Some(SystemField::Text(text)) => text.clone(),
        Some(SystemField::Blocks(blocks)) => blocks
            .iter()
            .filter(|b| b.kind == "text")
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n\n"),
    };
    let mut messages = Vec::new();
    let prefill = req.messages.last().is_some_and(|m| m.role == "assistant");
    let count = req.messages.len() - usize::from(prefill);
    for message in &req.messages[..count] {
        match message.role.as_str() {
            "system" => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(&render_user_content(&message.content)?);
            }
            "user" => messages
                .push(json!({"role":"user", "content":render_user_content(&message.content)?})),
            "assistant" => {
                let mut text = String::new();
                let mut calls = Vec::new();
                match &message.content {
                    AnthropicContent::Text(value) => text.push_str(value),
                    AnthropicContent::Blocks(blocks) => {
                        for block in blocks {
                            match block.kind.as_str() {
                            "text" => text.push_str(block.text.as_deref().unwrap_or("")),
                            "tool_use" => calls.push(json!({"type":"function", "function":{
                                "name":block.name.as_ref().ok_or_else(|| ServerError::bad_request("tool_use is missing name"))?,
                                "arguments":block.input.clone().unwrap_or_else(|| json!({}))
                            }})),
                            "image" => return Err(ServerError::bad_request("image content blocks are not supported")),
                            _ => {}
                        }
                        }
                    }
                }
                messages.push(json!({"role":"assistant", "content":text, "tool_calls":calls}));
            }
            role => {
                return Err(ServerError::bad_request(format!(
                    "unsupported message role: {role}"
                )));
            }
        }
    }
    if !system.is_empty() {
        messages.insert(0, json!({"role":"system","content":system}));
    }
    let tools: Vec<_> = req
        .tools
        .iter()
        .map(|tool| {
            json!({"type":"function","function":{
                "name":tool.name, "description":tool.description.as_deref().unwrap_or(""),
                "parameters":tool.input_schema.clone().unwrap_or_else(|| json!({}))
            }})
        })
        .collect();
    let mut out = template.render_flash(json!(messages), json!(tools))?;
    if prefill {
        // An answer prefill is not a partial private reasoning block.
        out.push_str("</think>\n\n");
        out.push_str(render_assistant_content(&req.messages.last().unwrap().content)?.trim_end());
    }
    Ok(out)
}

// Cap any single rendered tool_result. A giant tool dump (whole POM, dependency
// tree, tarball/file listing) is the usual cause of context overflow, and it is
// safe to truncate to head+tail — the agent keeps its own earlier findings and
// can re-query narrowly. ~12k chars ≈ 3–4k tokens: generous for a focused real
// output, decisive against a whole-file dump. Tunable via QK_MAX_TOOL_CHARS.
fn max_tool_result_chars() -> usize {
    // ~5k chars ≈ 1.3k tokens: enough for a focused tool output (package.json, a
    // targeted grep/jq), small enough that many turns accumulate slowly and stay
    // under the 16k window so the prefix cache keeps working without any trimming.
    std::env::var("QK_MAX_TOOL_CHARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n >= 1024)
        .unwrap_or(5_000)
}

fn floor_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_boundary(s: &str, mut idx: usize) -> usize {
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

fn cap_tool_output(s: String) -> String {
    let max = max_tool_result_chars();
    if s.len() <= max {
        return s;
    }
    let half = max / 2;
    let head = &s[..floor_boundary(&s, half)];
    let tail = &s[ceil_boundary(&s, s.len().saturating_sub(half))..];
    format!(
        "{head}\n\n[tool output truncated to fit context — {} of {} chars shown; \
         re-query more narrowly (grep/jq/head) if you need the rest]\n\n{tail}",
        max,
        s.len()
    )
}

// Render the conversation so it fits n_ctx. Stage-1 tool-result capping (above)
// usually suffices; if a long session still overflows, drop the OLDEST droppable
// turn — never messages[0] (the task) and never the most recent KEEP_RECENT (the
// agent's own findings, which it needs to emit its final JSON) — and retry. This
// is a graceful last resort: a server that trims to fit beats hard-erroring the
// whole item after minutes of work ("Prompt is too long").
fn fit_to_context(state: &AppState, req: &mut MessagesReq) -> Result<String> {
    const KEEP_RECENT: usize = 6;
    const CTX_RESERVE: u32 = 512; // room for at least a short generation
    let n_ctx = state.engine.n_ctx;
    // Purely a quality heuristic — prepare_generation clamps max_gen to the
    // real capacity — so scale it down on small-ctx servers (tests,
    // experiments) instead of reserving the whole window away.
    let reserve = CTX_RESERVE.min(n_ctx / 8);
    let hard = n_ctx.saturating_sub(reserve); // the server would reject beyond this
    // When we MUST trim, cut down to a lower target so the session has headroom to
    // grow a few more turns with its prefix cache intact — instead of pinning at
    // the ceiling and cold-prefilling ~n_ctx tokens EVERY turn (each trim that
    // drops a turn shifts the prefix and forces a full re-prefill that also blocks
    // the single engine thread). Trims then happen every few turns, not every one.
    let target = (n_ctx as f32 * 0.72) as u32;
    let mut trimming = false;
    loop {
        let prompt = if state.chat_template.is_flash() {
            render_flash_prompt(req, &state.chat_template)?
        } else {
            render_prompt(req, state.chat_template.gen_cue())?
        };
        let len = state.tokenizer.tokenize(&prompt, true)?.len() as u32;
        let limit = if trimming { target } else { hard };
        if len <= limit {
            return Ok(prompt); // fits (with headroom once we started trimming)
        }
        if req.messages.len() <= 1 + KEEP_RECENT {
            // Can't drop more. If still over the HARD cap, the task + a few recent
            // turns alone exceed the window — genuinely unfittable. Fail FAST (a
            // clean error in ~one turn, seconds) rather than submit a giant prefill
            // that grinds to the 2400s timeout and blocks the engine thread.
            if len > hard {
                return Err(ServerError::bad_request(
                    "conversation exceeds the context window even after trimming; \
                     keep tool outputs small (grep/jq/head)",
                ));
            }
            return Ok(prompt); // between target and hard: acceptable, no more to drop
        }
        trimming = true;
        req.messages.remove(1); // drop the oldest turn after the task
    }
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
                cap_tool_output(tool_result_text(block.content.as_ref())?)
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
                // Byte-shape-identical to the emission format the <tools>
                // preamble demands: `{"name": ..., "arguments": ...}` with
                // `name` FIRST. (A serde_json map would sort keys and put
                // `arguments` first — the model then imitates that degraded
                // history shape and eventually drops `name` entirely.)
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str("<tool_call>\n{\"name\": ");
                out.push_str(&spaced_json(&Value::String(
                    block.name.clone().unwrap_or_default(),
                )));
                out.push_str(", \"arguments\": ");
                out.push_str(&spaced_json(
                    &block.input.clone().unwrap_or_else(|| json!({})),
                ));
                out.push_str("}\n</tool_call>");
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
/// so it never emits a partial tag as text. A `<tool_call>` block that cannot
/// be parsed (even after repair/inference against the advertised tools)
/// surfaces as `ParsedEvent::Malformed` so the caller can retry generation
/// instead of ending the turn.
pub struct OutputParser {
    buf: String,
    state: ParseState,
    tools: Vec<ToolSpec>,
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
    Malformed { raw: String, error: String },
}

impl Default for OutputParser {
    fn default() -> Self {
        Self::new(&[])
    }
}

impl OutputParser {
    pub fn new(tools: &[ToolSpec]) -> Self {
        Self {
            buf: String::new(),
            state: ParseState::Text,
            tools: tools.to_vec(),
        }
    }

    pub fn for_prompt(tools: &[ToolSpec], prompt: &str) -> Self {
        let mut parser = Self::new(tools);
        let tail = prompt
            .rsplit("<|im_start|>assistant\n")
            .next()
            .unwrap_or("");
        if let Some(open) = tail.rfind(THINK_OPEN) {
            if tail.rfind(THINK_CLOSE).is_none_or(|close| close < open) {
                parser.state = ParseState::Think;
            }
        }
        parser
    }

    pub fn push(&mut self, text: &str) -> Vec<ParsedEvent> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.state {
                ParseState::Text => {
                    let tool = self
                        .buf
                        .find(TOOL_OPEN)
                        .map(|pos| (pos, TOOL_OPEN, ParseState::Tool));
                    let think = self
                        .buf
                        .find(THINK_OPEN)
                        .map(|pos| (pos, THINK_OPEN, ParseState::Think));
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
                    // Reasoning can be very long at xhigh. Retain only a
                    // possible split closing tag, never the entire block.
                    let ready = self.buf.len() - holdback_len(&self.buf, &[THINK_CLOSE]);
                    self.buf.drain(..ready);
                    break;
                }
                ParseState::Tool => {
                    if let Some(pos) = self.buf.find(TOOL_CLOSE) {
                        let raw = self.buf[..pos].trim().to_owned();
                        self.buf.drain(..pos + TOOL_CLOSE.len());
                        self.state = ParseState::Text;
                        out.push(parse_tool_call(&raw, &self.tools));
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
            ParseState::Tool => vec![parse_tool_call(self.buf.trim(), &self.tools)],
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

/// Repair JSON corruptions Qwen has been observed to emit inside
/// `<tool_call>` blocks — a dropped `":"` after a key, as in
/// `{"arguments{"command": ...}, "name": "Bash"}`.
fn repair_tool_json(raw: &str) -> String {
    raw.replace("\"arguments\"{", "\"arguments\": {")
        .replace("\"arguments{", "\"arguments\": {")
        .replace("\"name\"{", "\"name\": {")
        .replace("\"name{", "\"name\": {")
}

fn parse_tool_call(raw: &str, tools: &[ToolSpec]) -> ParsedEvent {
    if raw.starts_with("<function=") {
        return match parse_xml_tool_call(raw, tools) {
            Ok((name, input)) => ParsedEvent::ToolCall { name, input },
            Err(error) => ParsedEvent::Malformed {
                raw: raw.to_owned(),
                error,
            },
        };
    }
    #[derive(Deserialize)]
    struct Call {
        name: Option<String>,
        #[serde(default)]
        arguments: Value,
    }
    let call = serde_json::from_str::<Call>(raw)
        .or_else(|_| serde_json::from_str::<Call>(&repair_tool_json(raw)));
    let call = match call {
        Ok(call) => call,
        Err(err) => {
            return ParsedEvent::Malformed {
                raw: raw.to_owned(),
                error: format!("the content between <tool_call> tags is not valid JSON ({err})"),
            };
        }
    };
    // Some models emit arguments as a JSON-encoded string.
    let input = match call.arguments {
        Value::Null => json!({}),
        Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
        other => other,
    };
    let name = call.name.filter(|name| !name.is_empty()).or_else(|| {
        let inferred = infer_tool_name(&input, tools);
        if let Some(name) = &inferred {
            tracing::warn!("tool call missing \"name\"; inferred {name:?} from argument keys");
        }
        inferred
    });
    match name {
        Some(name) => ParsedEvent::ToolCall { name, input },
        None => ParsedEvent::Malformed {
            raw: raw.to_owned(),
            error: "the JSON object is missing the required \"name\" field".to_owned(),
        },
    }
}

/// Parse only Qwen's function/parameter wire syntax. This is not a general XML
/// parser: no entities, DTDs, file references, or implicit tool-name inference.
fn parse_xml_tool_call(
    raw: &str,
    tools: &[ToolSpec],
) -> std::result::Result<(String, Value), String> {
    fn header<'a>(text: &'a str, prefix: &str) -> std::result::Result<(&'a str, &'a str), String> {
        let body = text
            .strip_prefix(prefix)
            .ok_or("expected XML tool header")?;
        let (name, rest) = body.split_once('>').ok_or("unterminated XML tool header")?;
        if name.is_empty()
            || name
                .chars()
                .any(|c| c.is_whitespace() || matches!(c, '<' | '=' | '"' | '\''))
        {
            return Err("invalid XML tool identifier".into());
        }
        Ok((name, rest))
    }
    let (name, mut rest) = header(raw, "<function=")?;
    let spec = tools.iter().find(|tool| tool.name == name);
    if !tools.is_empty() && spec.is_none() {
        return Err("unadvertised XML tool name".into());
    }
    let mut input = serde_json::Map::new();
    loop {
        rest = rest.trim_start();
        if let Some(end) = rest.strip_prefix("</function>") {
            if !end.trim().is_empty() {
                return Err("trailing XML tool content".into());
            }
            break;
        }
        let (key, body) = header(rest, "<parameter=")?;
        if input.contains_key(key) {
            return Err("duplicate XML tool parameter".into());
        }
        let (value, tail) = body
            .split_once("</parameter>")
            .ok_or("unterminated XML tool parameter")?;
        if value.contains("<parameter=") || value.contains("</function>") {
            return Err("ambiguous nested XML tool delimiters".into());
        }
        // Remove exactly the template's wrapper newlines, not argument spaces.
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let kind = spec
            .and_then(|s| s.properties.get(key))
            .and_then(|p| p.get("type"))
            .and_then(Value::as_str);
        let parsed = if kind == Some("string") {
            Value::String(value.into())
        } else {
            serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.into()))
        };
        let valid = match kind {
            Some("integer") => parsed.is_i64() || parsed.is_u64(),
            Some("number") => parsed.is_number(),
            Some("boolean") => parsed.is_boolean(),
            Some("object") => parsed.is_object(),
            Some("array") => parsed.is_array(),
            Some("null") => parsed.is_null(),
            _ => true,
        };
        if !valid {
            return Err(format!("invalid type for XML tool parameter {key}"));
        }
        input.insert(key.into(), parsed);
        rest = tail;
    }
    Ok((name.into(), Value::Object(input)))
}

/// Last-resort passthrough when a malformed tool call cannot be retried.
fn malformed_fallback(raw: &str) -> String {
    format!("{TOOL_OPEN}\n{raw}\n{TOOL_CLOSE}")
}

/// How many times one request may re-prompt the model after it emitted an
/// unusable `<tool_call>` (no valid call alongside it).
const MAX_TOOL_RETRIES: usize = 2;

/// Feed a malformed tool call back to the model as a synthetic error
/// `<tool_response>` and reopen the assistant turn — the same shape as a real
/// tool round-trip — so the agentic loop continues instead of the turn ending
/// with the raw `<tool_call>` text.
fn continuation_prompt(prompt: &str, generated: &str, error: &str, gen_cue: &str) -> String {
    format!(
        "{prompt}{generated}<|im_end|>\n<|im_start|>user\n<tool_response>\n\
         ERROR: your tool call was malformed and was NOT executed: {error}. \
         Re-emit the corrected tool call as a single JSON object with both fields, exactly:\n\
         <tool_call>\n{{\"name\": \"<function-name>\", \"arguments\": {{<args-json-object>}}}}\n</tool_call>\n\
         </tool_response><|im_end|>\n{gen_cue}"
    )
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

pub async fn messages(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    match handle_messages(state, headers, body).await {
        Ok(response) => response,
        Err(err) => anthropic_error(err),
    }
}

async fn handle_messages(state: AppState, headers: HeaderMap, body: Bytes) -> Result<Response> {
    require_json(&headers)?;
    let mut req: MessagesReq = parse_json(&body)?;
    if req.messages.is_empty() {
        return Err(ServerError::bad_request("messages must not be empty"));
    }
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.model_alias.clone());
    let prompt = fit_to_context(&state, &mut req)?;
    let specs = tool_specs(&req.tools);
    let sampling = resolve_sampling(req.temperature, req.top_p, None);
    let mut generation = prepare_generation(&state, Prompt::Text(prompt.clone()), req.max_tokens)?;
    generation.snap_prefix = history_boundary(&state, &prompt);
    let rx = submit_generation(
        &state,
        &generation.prompt_ids,
        generation.max_gen,
        generation.snap_prefix,
        sampling,
    )?;
    if req.stream {
        let job = StreamJob {
            generation,
            prompt,
            specs,
            max_tokens: req.max_tokens,
            stops: req.stop_sequences,
            model,
            sampling,
        };
        let stream = stream_messages(state, rx, job);
        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response());
    }

    let input_tokens = generation.prompt_ids.len();
    let mut rx = rx;
    let mut prompt_text = prompt;
    let mut attempts = 0usize;
    let mut output_tokens = 0usize;
    let mut events: Vec<ParsedEvent> = Vec::new();
    let (finish, stopping_word) = loop {
        let completed = collect_generation(&state, rx, &req.stop_sequences).await?;
        output_tokens += completed.tokens.len();
        let mut parser = OutputParser::for_prompt(&specs, &prompt_text);
        let mut batch = parser.push(&completed.content);
        batch.extend(parser.finish());
        let has_tool_call = batch
            .iter()
            .any(|event| matches!(event, ParsedEvent::ToolCall { .. }));
        let malformed_error = batch.iter().find_map(|event| match event {
            ParsedEvent::Malformed { error, .. } => Some(error.clone()),
            _ => None,
        });
        // A malformed tool call with no valid one alongside would end the
        // turn as plain text; retry generation with a synthetic error result
        // instead. (On a token-limit finish there is no budget to retry.)
        if let Some(error) = malformed_error
            && !has_tool_call
            && completed.finish == FinishKind::Eos
            && attempts < MAX_TOOL_RETRIES
        {
            let next_prompt = continuation_prompt(
                &prompt_text,
                &completed.content,
                &error,
                state.chat_template.gen_cue(),
            );
            let resubmit =
                prepare_generation(&state, Prompt::Text(next_prompt.clone()), req.max_tokens)
                    .and_then(|mut next| {
                        next.snap_prefix = history_boundary(&state, &next_prompt);
                        submit_generation(
                            &state,
                            &next.prompt_ids,
                            next.max_gen,
                            next.snap_prefix,
                            sampling,
                        )
                    });
            if let Ok(next_rx) = resubmit {
                tracing::warn!(
                    "malformed <tool_call> ({error}); re-prompting (retry {})",
                    attempts + 1
                );
                attempts += 1;
                prompt_text = next_prompt;
                rx = next_rx;
                events.extend(
                    batch
                        .into_iter()
                        .filter(|event| !matches!(event, ParsedEvent::Malformed { .. })),
                );
                continue;
            }
        }
        events.extend(batch);
        break (completed.finish, completed.stopping_word);
    };

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
            // Retries exhausted (or a valid call rode alongside): pass the
            // raw block through as text so nothing is silently lost.
            ParsedEvent::Malformed { raw, .. } => {
                content.push(json!({ "type": "text", "text": malformed_fallback(&raw) }));
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

    let reason = stop_reason(finish, tool_calls);
    let stop_sequence = if finish == FinishKind::Word {
        Value::String(stopping_word)
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
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
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
            // Callers hold Malformed events back for the retry path; if one
            // reaches the emitter anyway, pass it through as text.
            ParsedEvent::Malformed { raw, .. } => {
                self.on_event(ParsedEvent::Text(malformed_fallback(&raw)))
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

/// Everything `stream_messages` needs beyond the state and the first
/// generation's event channel; `prompt`/`max_tokens` allow re-prompting after
/// a malformed tool call.
struct StreamJob {
    generation: Generation,
    prompt: String,
    specs: Vec<ToolSpec>,
    max_tokens: Option<u32>,
    stops: Vec<String>,
    model: String,
    sampling: Sampling,
}

fn stream_messages(
    state: AppState,
    rx: tokio::sync::mpsc::Receiver<crate::engine::SlotEvent>,
    job: StreamJob,
) -> impl Stream<Item = std::result::Result<Event, axum::Error>> {
    stream! {
        let msg_id = format!("msg_{}", Uuid::new_v4().simple());
        yield Ok(sse("message_start", json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": job.model.clone(),
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": { "input_tokens": job.generation.prompt_ids.len(), "output_tokens": 0 }
            }
        })));
        yield Ok(sse("ping", json!({ "type": "ping" })));

        let mut rx = rx;
        let mut prompt_text = job.prompt;
        let mut attempts = 0usize;
        let mut emitter = BlockEmitter::new();
        let mut finish;
        let mut stopping_word = String::new();
        let mut output_tokens = 0usize;
        // Raw text of malformed tool calls that could not be retried away;
        // emitted as plain text at the end so nothing is silently lost.
        let mut leftover: Vec<String> = Vec::new();
        loop {
            let mut stop = StopDetector::new(&job.stops);
            let mut parser = OutputParser::for_prompt(&job.specs, &prompt_text);
            let mut raw_text = String::new();
            let mut malformed: Vec<(String, String)> = Vec::new();
            finish = FinishKind::Limit;
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
                        let ready = stop.take_ready();
                        raw_text.push_str(&ready);
                        for parsed in parser.push(&ready) {
                            match parsed {
                                ParsedEvent::Malformed { raw, error } => malformed.push((raw, error)),
                                other => {
                                    for event in emitter.on_event(other) {
                                        yield Ok(event);
                                    }
                                }
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

            let ready = stop.finish();
            raw_text.push_str(&ready);
            let mut tail = parser.push(&ready);
            tail.extend(parser.finish());
            for parsed in tail {
                match parsed {
                    ParsedEvent::Malformed { raw, error } => malformed.push((raw, error)),
                    other => {
                        for event in emitter.on_event(other) {
                            yield Ok(event);
                        }
                    }
                }
            }

            // Same retry rule as the non-streaming path: a malformed tool
            // call with no valid one alongside gets a synthetic error result
            // and another chance, instead of ending the turn.
            if !malformed.is_empty()
                && emitter.tool_calls == 0
                && finish == FinishKind::Eos
                && attempts < MAX_TOOL_RETRIES
            {
                let error = malformed[0].1.clone();
                let next_prompt = continuation_prompt(
                    &prompt_text,
                    &raw_text,
                    &error,
                    state.chat_template.gen_cue(),
                );
                let resubmit =
                    prepare_generation(&state, Prompt::Text(next_prompt.clone()), job.max_tokens)
                        .and_then(|mut next| {
                            next.snap_prefix = history_boundary(&state, &next_prompt);
                            submit_generation(&state, &next.prompt_ids, next.max_gen, next.snap_prefix, job.sampling)
                        });
                match resubmit {
                    Ok(next_rx) => {
                        tracing::warn!(
                            "malformed <tool_call> ({error}); re-prompting (retry {})",
                            attempts + 1
                        );
                        attempts += 1;
                        prompt_text = next_prompt;
                        rx = next_rx;
                        continue;
                    }
                    Err(_) => {
                        leftover = malformed.into_iter().map(|(raw, _)| raw).collect();
                        break;
                    }
                }
            }
            if emitter.tool_calls == 0 {
                leftover = malformed.into_iter().map(|(raw, _)| raw).collect();
            }
            break;
        }

        for raw in leftover {
            for event in emitter.on_event(ParsedEvent::Text(malformed_fallback(&raw))) {
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
    let prompt = render_prompt(&req, state.chat_template.gen_cue())?;
    let tokens = state.tokenizer.tokenize(&prompt, true)?;
    Ok(Json(json!({ "input_tokens": tokens.len() })).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::{CUE_PLAIN, CUE_THINK};

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
        let prompt = render_prompt(&req, CUE_THINK).expect("renders");
        assert!(prompt.starts_with("<|im_start|>system\nBe brief.\n\n# Tools"));
        assert!(prompt.contains(
            "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\""
        ));
        assert!(prompt.contains("\n</tools>\n\nFor each function call"));
        assert!(prompt.contains(
            "<|im_start|>assistant\nChecking.\n<tool_call>\n\
             {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n\
             </tool_call><|im_end|>\n"
        ));
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
        let prompt = render_prompt(&req, CUE_THINK).expect("renders");
        assert!(prompt.starts_with("<|im_start|>system\nYou are terse.<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n{\"color\":"));
    }

    #[test]
    fn renders_in_messages_system_turns() {
        // Claude Code >= 2.2 injects system reminders as system-role entries
        // inside `messages`, alongside the top-level `system` field.
        let req = req_from_json(json!({
            "max_tokens": 16,
            "system": "You are terse.",
            "messages": [
                { "role": "user", "content": "hello" },
                { "role": "system", "content": [
                    { "type": "text", "text": "<system-reminder>stay on task</system-reminder>" }
                ]},
                { "role": "user", "content": "continue" }
            ]
        }));
        let prompt = render_prompt(&req, CUE_THINK).expect("renders");
        assert!(prompt.contains(
            "<|im_start|>system\n<system-reminder>stay on task</system-reminder><|im_end|>\n\
             <|im_start|>user\ncontinue<|im_end|>\n"
        ));
    }

    #[test]
    fn rejects_images_and_unknown_roles() {
        let image = req_from_json(json!({
            "max_tokens": 16,
            "messages": [{ "role": "user", "content": [
                { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "" } }
            ]}]
        }));
        assert!(render_prompt(&image, CUE_THINK).is_err());
        let role = req_from_json(json!({
            "max_tokens": 16,
            "messages": [{ "role": "tool", "content": "x" }]
        }));
        assert!(render_prompt(&role, CUE_THINK).is_err());
    }

    fn collect(events: Vec<ParsedEvent>) -> Vec<(String, String)> {
        events
            .into_iter()
            .map(|event| match event {
                ParsedEvent::Text(text) => ("text".to_owned(), text),
                ParsedEvent::ToolCall { name, input } => (name, input.to_string()),
                ParsedEvent::Malformed { raw, .. } => ("malformed".to_owned(), raw),
            })
            .collect()
    }

    fn cli_like_specs() -> Vec<ToolSpec> {
        let tools: Vec<ToolDef> = serde_json::from_value(json!([
            {
                "name": "Bash",
                "input_schema": { "type": "object", "properties": {
                    "command": {}, "description": {}, "timeout": {}
                }}
            },
            {
                "name": "Read",
                "input_schema": { "type": "object", "properties": {
                    "file_path": {}, "offset": {}, "limit": {}
                }}
            }
        ]))
        .expect("tools parse");
        tool_specs(&tools)
    }

    #[test]
    fn parser_honors_open_reasoning_and_bounds_discard_buffer() {
        let prompt = crate::template::CUE_THINK_OPEN;
        for split in 0..="hidden</think>answer".len() {
            let output = "hidden</think>answer";
            let mut parser = OutputParser::for_prompt(&[], prompt);
            let mut events = parser.push(&output[..split]);
            events.extend(parser.push(&output[split..]));
            events.extend(parser.finish());
            let text: String = collect(events).into_iter().map(|(_, text)| text).collect();
            assert_eq!(text, "answer", "split {split}");
        }
        let mut parser = OutputParser::for_prompt(&[], prompt);
        for _ in 0..100 {
            assert!(parser.push(&"reasoning".repeat(1024)).is_empty());
            assert!(parser.buf.len() < THINK_CLOSE.len());
        }
        assert_eq!(collect(parser.push("</think>done"))[0].1, "done");
        let mut closed = OutputParser::for_prompt(&[], CUE_THINK);
        assert_eq!(collect(closed.push("answer"))[0].1, "answer");
    }

    #[test]
    fn parser_streams_flash_xml_and_preserves_argument_types() {
        let tools: Vec<ToolDef> = serde_json::from_value(json!([{
            "name":"Write", "input_schema":{"type":"object","properties":{
                "content":{"type":"string"},"count":{"type":"integer"},
                "enabled":{"type":"boolean"},"options":{"type":"object"}
            }}
        }]))
        .unwrap();
        let specs = tool_specs(&tools);
        let output = "<tool_call>\n<function=Write>\n<parameter=content>\n  123\n\n</parameter>\n<parameter=count>\n3\n</parameter>\n<parameter=enabled>\nfalse\n</parameter>\n<parameter=options>\n{\"x\":1}\n</parameter>\n</function>\n</tool_call>";
        for split in 0..=output.len() {
            let mut parser = OutputParser::new(&specs);
            let mut events = parser.push(&output[..split]);
            events.extend(parser.push(&output[split..]));
            events.extend(parser.finish());
            assert_eq!(events.len(), 1, "split {split}");
            let ParsedEvent::ToolCall { name, input } = &events[0] else {
                panic!("not a call")
            };
            assert_eq!(name, "Write");
            assert_eq!(
                input,
                &json!({"content":"  123\n","count":3,"enabled":false,"options":{"x":1}})
            );
        }
        for bad in [
            "<function=Unknown></function>",
            "<function=Write><parameter=count>wrong</parameter></function>",
            "<function=Write><parameter=count>1</parameter><parameter=count>2</parameter></function>",
            "<function=Write><parameter=count>1</function>",
            "<function=Write></function>junk",
            "<function=Write><parameter=content>x<parameter=count>1</parameter></function>",
        ] {
            assert!(
                matches!(parse_tool_call(bad, &specs), ParsedEvent::Malformed { .. }),
                "{bad}"
            );
        }
    }

    #[test]
    fn flash_claude_history_uses_checkpoint_template() {
        let Some(model) = std::env::var_os("QK_FLASH_GGUF") else {
            return;
        };
        let meta = crate::gguf::read_metadata(std::path::Path::new(&model)).unwrap();
        let template = crate::template::ChatTemplate::new(
            meta.chat_template,
            crate::template::TemplateMode::Auto,
        )
        .for_architecture(meta.architecture.as_deref());
        let req = req_from_json(json!({"system":"global policy", "tools":[{
            "name":"Read","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}
        }],"messages":[
            {"role":"user","content":"read it"},
            {"role":"system","content":"system reminder"},
            {"role":"assistant","content":[{"type":"tool_use","name":"Read","input":{"path":"hi.txt"}}]},
            {"role":"user","content":[{"type":"tool_result","content":"hello"}]}
        ]}));
        let prompt = render_flash_prompt(&req, &template).unwrap();
        assert_eq!(prompt.matches("<|im_start|>system\n").count(), 1);
        assert!(prompt.contains("global policy\n\nsystem reminder"));
        assert!(
            prompt.contains("<function=Read>\n<parameter=path>\nhi.txt\n</parameter>\n</function>")
        );
        assert!(prompt.contains("<tool_response>\nhello\n</tool_response>"));
        assert!(prompt.ends_with(crate::template::CUE_THINK_OPEN));
        let prefill = req_from_json(json!({"messages":[
            {"role":"user","content":"say it"},{"role":"assistant","content":"The answer is"}
        ]}));
        let prompt = render_flash_prompt(&prefill, &template).unwrap();
        assert!(prompt.ends_with("<think>\n</think>\n\nThe answer is"));
        assert!(OutputParser::for_prompt(&[], &prompt).state == ParseState::Text);
    }

    #[test]
    fn parser_passes_plain_text() {
        let mut parser = OutputParser::new(&[]);
        let mut events = parser.push("hello ");
        events.extend(parser.push("world"));
        events.extend(parser.finish());
        let text: String = collect(events).into_iter().map(|(_, text)| text).collect();
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
            let mut parser = OutputParser::new(&[]);
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
    fn parser_reports_malformed_tool_json() {
        let mut parser = OutputParser::new(&[]);
        let mut events = parser.push("<tool_call>\nnot json\n</tool_call>");
        events.extend(parser.finish());
        let got = collect(events);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "malformed");
        assert!(got[0].1.contains("not json"));
    }

    #[test]
    fn tool_use_round_trip_matches_emission_shape() {
        // The re-rendered history must be byte-identical to the shape the
        // <tools> preamble demands: "name" FIRST, json.dumps-style spacing,
        // and argument keys in their original (non-alphabetical) order.
        let req = req_from_json(json!({
            "max_tokens": 8,
            "messages": [
                { "role": "user", "content": "write it" },
                { "role": "assistant", "content": [
                    { "type": "tool_use", "id": "t1", "name": "Write",
                      "input": { "file_path": "/tmp/x", "content": "hi" } }
                ]},
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "t1", "content": "ok" }
                ]}
            ]
        }));
        let prompt = render_prompt(&req, CUE_THINK).expect("renders");
        assert!(prompt.contains(
            "<|im_start|>assistant\n<tool_call>\n\
             {\"name\": \"Write\", \"arguments\": {\"file_path\": \"/tmp/x\", \"content\": \"hi\"}}\n\
             </tool_call><|im_end|>\n"
        ));
    }

    #[test]
    fn parser_infers_missing_name_from_argument_keys() {
        let specs = cli_like_specs();
        let mut parser = OutputParser::new(&specs);
        let mut events = parser.push(
            "<tool_call>\n{\"arguments\": {\"command\": \"ls\", \"description\": \"x\"}}\n</tool_call>",
        );
        events.extend(parser.finish());
        let got = collect(events);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "Bash");
        assert_eq!(got[0].1, "{\"command\":\"ls\",\"description\":\"x\"}");
    }

    #[test]
    fn parser_reports_ambiguous_missing_name_as_malformed() {
        // Empty arguments match every tool: no unique candidate, no guess.
        let specs = cli_like_specs();
        let mut parser = OutputParser::new(&specs);
        let mut events = parser.push("<tool_call>{\"arguments\": {}}</tool_call>");
        events.extend(parser.finish());
        let got = collect(events);
        assert_eq!(got[0].0, "malformed");
    }

    #[test]
    fn parser_repairs_missing_colon_corruption() {
        // Observed in the wild: {"arguments{"command": ...}, "name": "Bash"}
        let specs = cli_like_specs();
        let mut parser = OutputParser::new(&specs);
        let mut events = parser.push(
            "<tool_call>\n{\"arguments{\"command\":\"unzip -p x.jar\",\"description\":\"Read\"},\"name\":\"Bash\"}\n</tool_call>",
        );
        events.extend(parser.finish());
        let got = collect(events);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "Bash");
        assert!(got[0].1.contains("unzip -p x.jar"));
    }

    #[test]
    fn continuation_prompt_reopens_assistant_turn() {
        let prompt = continuation_prompt(
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<tool_call>\n{\"arguments\": {}}\n</tool_call>",
            "the JSON object is missing the required \"name\" field",
            CUE_THINK,
        );
        assert!(prompt.contains("<tool_call>\n{\"arguments\": {}}\n</tool_call><|im_end|>\n"));
        assert!(prompt.contains("<tool_response>\nERROR: your tool call was malformed"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn plain_cue_renders_without_think_scaffold() {
        // Qwen3-Next-Instruct shape: generation cue opens the assistant turn
        // with no think block, for both fresh turns and tool-retry
        // continuations.
        let req = req_from_json(json!({
            "messages": [{ "role": "user", "content": "hi" }]
        }));
        let prompt = render_prompt(&req, CUE_PLAIN).expect("renders");
        assert!(prompt.ends_with("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"));
        assert!(!prompt.contains("<think>"));
        let cont = continuation_prompt(&prompt, "<tool_call>bad</tool_call>", "err", CUE_PLAIN);
        assert!(cont.ends_with("</tool_response><|im_end|>\n<|im_start|>assistant\n"));
        assert!(!cont.contains("<think>"));
    }

    #[test]
    fn parser_salvages_unterminated_tool_call() {
        let mut parser = OutputParser::new(&[]);
        let mut events =
            parser.push("<tool_call>\n{\"name\": \"ls\", \"arguments\": {\"dir\": \"/\"}}");
        events.extend(parser.finish());
        let got = collect(events);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "ls");
    }

    #[test]
    fn parser_unwraps_string_encoded_arguments() {
        let mut parser = OutputParser::new(&[]);
        let mut events = parser.push(
            "<tool_call>{\"name\": \"ls\", \"arguments\": \"{\\\"dir\\\": \\\"/tmp\\\"}\"}</tool_call>",
        );
        events.extend(parser.finish());
        let got = collect(events);
        assert_eq!(got[0].0, "ls");
        assert_eq!(got[0].1, "{\"dir\":\"/tmp\"}");
    }
}
