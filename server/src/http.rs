use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use async_stream::stream;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, header};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::Stream;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{Semaphore, mpsc as tokio_mpsc};
use tower_http::limit::RequestBodyLimitLayer;
use uuid::Uuid;

use crate::engine::{EngineHandle, FinishReason, Job, SlotEvent};
use crate::template::{ChatMessage, ChatTemplate};
use crate::tokenizer::Tokenizer;
use crate::{Result, ServerError};

#[derive(Clone)]
pub struct AppState {
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Arc<ChatTemplate>,
    pub engine: EngineHandle,
    pub model_path: PathBuf,
    pub model_alias: String,
    pub queue: Arc<Semaphore>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/props", get(props))
        .route("/v1/models", get(models))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/completion", post(completion))
        .route("/completions", post(completion))
        .route("/v1/completions", post(openai_completion))
        .route("/v1/chat/completions", post(openai_chat_completion))
        .route("/v1/messages", post(crate::anthropic::messages))
        .route("/v1/messages/count_tokens", post(crate::anthropic::count_tokens))
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(4 * 1024 * 1024))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn props(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "model_path": state.model_path.display().to_string(),
        "model_alias": state.model_alias,
        "chat_template": state.chat_template.source(),
        "total_slots": state.engine.n_slots,
        "default_generation_settings": { "n_ctx": state.engine.n_ctx },
        "modalities": { "vision": false, "audio": false },
    }))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_alias,
            "object": "model",
            "created": 0,
            "owned_by": "qk-server"
        }]
    }))
}

#[derive(Deserialize)]
struct TokenizeReq {
    content: String,
    #[serde(default)]
    add_special: bool,
    #[serde(default = "default_true")]
    parse_special: bool,
    #[serde(default)]
    with_pieces: bool,
}

#[derive(Deserialize)]
struct DetokenizeReq {
    tokens: Vec<u32>,
}

async fn tokenize(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response> {
    require_json(&headers)?;
    let req: TokenizeReq = parse_json(&body)?;
    let _ = req.add_special;
    let tokens = state.tokenizer.tokenize(&req.content, req.parse_special)?;
    if req.with_pieces {
        let pieces = tokens
            .iter()
            .map(|id| {
                state
                    .tokenizer
                    .token_piece(*id, true)
                    .map(|piece| json!({ "id": id, "piece": piece }))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Json(json!({ "tokens": pieces })).into_response())
    } else {
        Ok(Json(json!({ "tokens": tokens })).into_response())
    }
}

async fn detokenize(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<Value>> {
    require_json(&headers)?;
    let req: DetokenizeReq = parse_json(&body)?;
    let content = state.tokenizer.detokenize(&req.tokens, false)?;
    Ok(Json(json!({ "content": content })))
}

#[derive(Deserialize)]
struct CompletionReq {
    prompt: Prompt,
    n_predict: Option<u32>,
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Vec<String>,
    #[serde(default)]
    return_tokens: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
}

async fn completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response> {
    require_json(&headers)?;
    let req: CompletionReq = parse_json(&body)?;
    let generation = prepare_generation(&state, req.prompt, req.n_predict.or(req.max_tokens))?;
    if req.stream {
        let rx = submit_generation(&state, &generation.prompt_ids, generation.max_gen)?;
        Ok(sse_response(stream_llama(
            state,
            rx,
            generation,
            req.stop,
            req.return_tokens,
        ))
        .into_response())
    } else {
        let rx = submit_generation(&state, &generation.prompt_ids, generation.max_gen)?;
        let completed = collect_generation(&state, rx, &req.stop).await?;
        Ok(Json(llama_completion_body(
            &state,
            &generation,
            completed,
            req.return_tokens,
        ))
        .into_response())
    }
}

async fn openai_completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response> {
    require_json(&headers)?;
    let req: CompletionReq = parse_json(&body)?;
    let generation = prepare_generation(&state, req.prompt, req.n_predict.or(req.max_tokens))?;
    if req.stream {
        let rx = submit_generation(&state, &generation.prompt_ids, generation.max_gen)?;
        Ok(sse_response(stream_openai_completion(state, rx, generation)).into_response())
    } else {
        let rx = submit_generation(&state, &generation.prompt_ids, generation.max_gen)?;
        let completed = collect_generation(&state, rx, &req.stop).await?;
        let finish = if completed.finish == FinishKind::Limit {
            "length"
        } else {
            "stop"
        };
        Ok(Json(json!({
            "id": format!("cmpl-{}", Uuid::new_v4()),
            "object": "text_completion",
            "model": state.model_alias,
            "choices": [{ "text": completed.content, "index": 0, "finish_reason": finish }],
            "usage": {
                "prompt_tokens": generation.prompt_ids.len(),
                "completion_tokens": completed.tokens.len(),
                "total_tokens": generation.prompt_ids.len() + completed.tokens.len()
            }
        }))
        .into_response())
    }
}

#[derive(Deserialize)]
struct ChatReq {
    messages: Vec<RawChatMessage>,
    n_predict: Option<u32>,
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Vec<String>,
    stream_options: Option<StreamOptions>,
}

#[derive(Deserialize)]
struct StreamOptions {
    include_usage: Option<bool>,
}

#[derive(Deserialize)]
struct RawChatMessage {
    role: String,
    content: RawContent,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

async fn openai_chat_completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response> {
    require_json(&headers)?;
    let req: ChatReq = parse_json(&body)?;
    let messages = normalize_messages(req.messages)?;
    let prompt = state.chat_template.render(&messages, true)?;
    let generation = prepare_generation(
        &state,
        Prompt::Text(prompt),
        req.n_predict.or(req.max_tokens),
    )?;
    if req.stream {
        let rx = submit_generation(&state, &generation.prompt_ids, generation.max_gen)?;
        let include_usage = req
            .stream_options
            .as_ref()
            .and_then(|opts| opts.include_usage)
            .unwrap_or(false);
        Ok(sse_response(stream_openai_chat(state, rx, generation, include_usage)).into_response())
    } else {
        let rx = submit_generation(&state, &generation.prompt_ids, generation.max_gen)?;
        let completed = collect_generation(&state, rx, &req.stop).await?;
        let finish = if completed.finish == FinishKind::Limit {
            "length"
        } else {
            "stop"
        };
        Ok(Json(json!({
            "id": format!("chatcmpl-{}", Uuid::new_v4()),
            "object": "chat.completion",
            "model": state.model_alias,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": completed.content },
                "finish_reason": finish
            }],
            "usage": {
                "prompt_tokens": generation.prompt_ids.len(),
                "completion_tokens": completed.tokens.len(),
                "total_tokens": generation.prompt_ids.len() + completed.tokens.len()
            }
        }))
        .into_response())
    }
}

#[derive(Clone)]
pub(crate) struct Generation {
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) max_gen: u32,
    started: Instant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FinishKind {
    Eos,
    Limit,
    Word,
}

pub(crate) struct Completed {
    pub(crate) content: String,
    pub(crate) tokens: Vec<u32>,
    pub(crate) finish: FinishKind,
    pub(crate) stopping_word: String,
    predicted_ms: f64,
}

pub(crate) fn prepare_generation(
    state: &AppState,
    prompt: Prompt,
    requested: Option<u32>,
) -> Result<Generation> {
    let prompt_ids = match prompt {
        Prompt::Text(text) => state.tokenizer.tokenize(&text, true)?,
        Prompt::Tokens(tokens) => tokens,
    };
    if prompt_ids.is_empty() {
        return Err(ServerError::bad_request("prompt must not be empty"));
    }
    let n_vocab = usize::try_from(state.engine.n_vocab).unwrap_or(state.tokenizer.len());
    if prompt_ids
        .iter()
        .any(|id| usize::try_from(*id).map_or(true, |id| id >= n_vocab))
    {
        return Err(ServerError::bad_request(
            "prompt contains token id out of range",
        ));
    }
    let prompt_len = u32::try_from(prompt_ids.len())
        .map_err(|_| ServerError::bad_request("prompt is too large"))?;
    if prompt_len >= state.engine.n_ctx {
        return Err(ServerError::bad_request(format!(
            "the prompt is too long ({prompt_len} tokens, max {})",
            state.engine.n_ctx.saturating_sub(1)
        )));
    }
    let capacity = state.engine.n_ctx.saturating_sub(prompt_len);
    let max_gen = requested.unwrap_or(capacity).min(capacity).max(1);
    Ok(Generation {
        prompt_ids,
        max_gen,
        started: Instant::now(),
    })
}

pub(crate) fn submit_generation(
    state: &AppState,
    prompt_ids: &[u32],
    max_gen: u32,
) -> Result<tokio_mpsc::Receiver<SlotEvent>> {
    let permit = state
        .queue
        .clone()
        .try_acquire_owned()
        .map_err(|_| ServerError::QueueFull("request queue is full".to_owned()))?;
    // Headroom so a briefly-busy SSE consumer doesn't force the engine onto
    // its buffered-retry path; a stalled consumer is handled there regardless.
    let (tx, rx) = tokio_mpsc::channel(64);
    state
        .engine
        .submit(Job {
            prompt_ids: prompt_ids.to_vec(),
            max_gen,
            events: tx,
            permit,
        })
        .map_err(|_| ServerError::internal("engine thread is not running"))?;
    Ok(rx)
}

pub(crate) async fn collect_generation(
    state: &AppState,
    mut rx: tokio_mpsc::Receiver<SlotEvent>,
    stops: &[String],
) -> Result<Completed> {
    let start = Instant::now();
    let mut tokens = Vec::new();
    let mut stop = StopDetector::new(stops);
    let mut finish = FinishKind::Limit;
    while let Some(event) = rx.recv().await {
        match event {
            SlotEvent::Tokens(ids) => {
                tokens.extend(ids.iter().copied());
                let text = decode_tokens_lossless(state, &ids)?;
                if let Some(hit) = stop.push(&text) {
                    finish = FinishKind::Word;
                    return Ok(Completed {
                        content: stop.finish(),
                        tokens,
                        finish,
                        stopping_word: hit,
                        predicted_ms: start.elapsed().as_secs_f64() * 1000.0,
                    });
                }
            }
            SlotEvent::Done { reason } => {
                finish = match reason {
                    FinishReason::Eos => FinishKind::Eos,
                    FinishReason::Limit => FinishKind::Limit,
                };
                break;
            }
            SlotEvent::Error(message) => return Err(ServerError::internal(message)),
        }
    }
    Ok(Completed {
        content: stop.finish(),
        tokens,
        finish,
        stopping_word: String::new(),
        predicted_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

fn stream_llama(
    state: AppState,
    mut rx: tokio_mpsc::Receiver<SlotEvent>,
    generation: Generation,
    stops: Vec<String>,
    return_tokens: bool,
) -> impl Stream<Item = std::result::Result<Event, axum::Error>> {
    stream! {
        let mut tokens = Vec::new();
        let mut stop = StopDetector::new(&stops);
        let mut finish = FinishKind::Limit;
        while let Some(event) = rx.recv().await {
            match event {
                SlotEvent::Tokens(ids) => {
                    tokens.extend(ids.iter().copied());
                    let text = decode_tokens_lossless(&state, &ids).unwrap_or_default();
                    if let Some(hit) = stop.push(&text) {
                        finish = FinishKind::Word;
                        let delta = stop.take_ready();
                        if !delta.is_empty() {
                            yield Ok(Event::default().data(json!({"content": delta, "stop": false, "tokens": if return_tokens { json!(ids) } else { Value::Null }}).to_string()));
                        }
                        let completed = Completed { content: stop.finish(), tokens: tokens.clone(), finish, stopping_word: hit, predicted_ms: generation.started.elapsed().as_secs_f64() * 1000.0 };
                        yield Ok(Event::default().data(llama_completion_body(&state, &generation, completed, return_tokens).to_string()));
                        return;
                    }
                    let delta = stop.take_ready();
                    if !delta.is_empty() {
                        let body = if return_tokens {
                            json!({"content": delta, "stop": false, "tokens": ids})
                        } else {
                            json!({"content": delta, "stop": false})
                        };
                        yield Ok(Event::default().data(body.to_string()));
                    }
                }
                SlotEvent::Done { reason } => {
                    finish = if reason == FinishReason::Eos { FinishKind::Eos } else { FinishKind::Limit };
                    break;
                }
                SlotEvent::Error(message) => {
                    yield Ok(Event::default().data(json!({"error":{"code":500,"message":message,"type":"internal_error"}}).to_string()));
                    return;
                }
            }
        }
        let completed = Completed { content: stop.finish(), tokens, finish, stopping_word: String::new(), predicted_ms: generation.started.elapsed().as_secs_f64() * 1000.0 };
        yield Ok(Event::default().data(llama_completion_body(&state, &generation, completed, return_tokens).to_string()));
    }
}

fn stream_openai_completion(
    state: AppState,
    mut rx: tokio_mpsc::Receiver<SlotEvent>,
    generation: Generation,
) -> impl Stream<Item = std::result::Result<Event, axum::Error>> {
    stream! {
        let id = format!("cmpl-{}", Uuid::new_v4());
        while let Some(event) = rx.recv().await {
            match event {
                SlotEvent::Tokens(ids) => {
                    let text = decode_tokens_lossless(&state, &ids).unwrap_or_default();
                    yield Ok(Event::default().data(json!({"id": id, "object":"text_completion.chunk", "model": state.model_alias, "choices":[{"text":text,"index":0,"finish_reason":null}]}).to_string()));
                }
                SlotEvent::Done { reason } => {
                    let finish = if reason == FinishReason::Limit { "length" } else { "stop" };
                    yield Ok(Event::default().data(json!({"id": id, "object":"text_completion.chunk", "model": state.model_alias, "choices":[{"text":"","index":0,"finish_reason":finish}], "usage": {"prompt_tokens": generation.prompt_ids.len()}}).to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                SlotEvent::Error(message) => {
                    yield Ok(Event::default().data(json!({"error":{"message":message}}).to_string()));
                    return;
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

fn stream_openai_chat(
    state: AppState,
    mut rx: tokio_mpsc::Receiver<SlotEvent>,
    generation: Generation,
    include_usage: bool,
) -> impl Stream<Item = std::result::Result<Event, axum::Error>> {
    stream! {
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        yield Ok(Event::default().data(json!({"id": id, "object":"chat.completion.chunk", "model": state.model_alias, "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}).to_string()));
        let mut completion_tokens = 0usize;
        while let Some(event) = rx.recv().await {
            match event {
                SlotEvent::Tokens(ids) => {
                    completion_tokens += ids.len();
                    let text = decode_tokens_lossless(&state, &ids).unwrap_or_default();
                    yield Ok(Event::default().data(json!({"id": id, "object":"chat.completion.chunk", "model": state.model_alias, "choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]}).to_string()));
                }
                SlotEvent::Done { reason } => {
                    let finish = if reason == FinishReason::Limit { "length" } else { "stop" };
                    let mut body = json!({"id": id, "object":"chat.completion.chunk", "model": state.model_alias, "choices":[{"index":0,"delta":{},"finish_reason":finish}]});
                    if include_usage {
                        body["usage"] = json!({"prompt_tokens": generation.prompt_ids.len(), "completion_tokens": completion_tokens, "total_tokens": generation.prompt_ids.len() + completion_tokens});
                    }
                    yield Ok(Event::default().data(body.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                SlotEvent::Error(message) => {
                    yield Ok(Event::default().data(json!({"error":{"message":message}}).to_string()));
                    return;
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

fn llama_completion_body(
    state: &AppState,
    generation: &Generation,
    completed: Completed,
    return_tokens: bool,
) -> Value {
    let stop_type = match completed.finish {
        FinishKind::Eos => "eos",
        FinishKind::Limit => "limit",
        FinishKind::Word => "word",
    };
    let mut body = json!({
        "content": completed.content,
        "stop": true,
        "model": state.model_alias,
        "tokens_predicted": completed.tokens.len(),
        "tokens_evaluated": generation.prompt_ids.len(),
        "stop_type": stop_type,
        "stopping_word": completed.stopping_word,
        "timings": {
            "prompt_n": generation.prompt_ids.len(),
            "prompt_ms": 0.0,
            "predicted_n": completed.tokens.len(),
            "predicted_ms": completed.predicted_ms,
            "predicted_per_second": if completed.predicted_ms > 0.0 { completed.tokens.len() as f64 / (completed.predicted_ms / 1000.0) } else { 0.0 }
        }
    });
    if return_tokens {
        body["tokens"] = json!(completed.tokens);
    }
    body
}

pub(crate) fn decode_tokens_lossless(state: &AppState, ids: &[u32]) -> Result<String> {
    let mut out = String::new();
    for id in ids {
        if let Ok(piece) = state.tokenizer.token_piece(*id, false) {
            out.push_str(&piece);
        }
    }
    Ok(out)
}

pub(crate) struct StopDetector {
    stops: Vec<String>,
    held: String,
    ready: String,
    hold_chars: usize,
}

impl StopDetector {
    pub(crate) fn new(stops: &[String]) -> Self {
        let hold_chars = stops
            .iter()
            .map(|stop| stop.chars().count().saturating_sub(1))
            .max()
            .unwrap_or(0);
        Self {
            stops: stops
                .iter()
                .filter(|stop| !stop.is_empty())
                .cloned()
                .collect(),
            held: String::new(),
            ready: String::new(),
            hold_chars,
        }
    }

    pub(crate) fn push(&mut self, text: &str) -> Option<String> {
        self.held.push_str(text);
        for stop in &self.stops {
            if let Some(pos) = self.held.find(stop) {
                self.ready.push_str(&self.held[..pos]);
                self.held.clear();
                return Some(stop.clone());
            }
        }
        let len = self.held.chars().count();
        if len > self.hold_chars {
            let split_chars = len - self.hold_chars;
            let split_byte = self
                .held
                .char_indices()
                .nth(split_chars)
                .map(|(idx, _)| idx)
                .unwrap_or(self.held.len());
            self.ready.push_str(&self.held[..split_byte]);
            self.held = self.held[split_byte..].to_owned();
        }
        None
    }

    pub(crate) fn take_ready(&mut self) -> String {
        std::mem::take(&mut self.ready)
    }

    pub(crate) fn finish(mut self) -> String {
        self.ready.push_str(&self.held);
        self.ready
    }
}

pub(crate) fn require_json(headers: &HeaderMap) -> Result<()> {
    let Some(value) = headers.get(header::CONTENT_TYPE) else {
        return Err(ServerError::bad_request(
            "Content-Type must be application/json",
        ));
    };
    let Ok(value) = value.to_str() else {
        return Err(ServerError::bad_request(
            "Content-Type must be application/json",
        ));
    };
    if !value.to_ascii_lowercase().starts_with("application/json") {
        return Err(ServerError::bad_request(
            "Content-Type must be application/json",
        ));
    }
    Ok(())
}

pub(crate) fn parse_json<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T> {
    serde_json::from_slice(body)
        .map_err(|err| ServerError::bad_request(format!("malformed JSON: {err}")))
}

fn normalize_messages(messages: Vec<RawChatMessage>) -> Result<Vec<ChatMessage>> {
    messages
        .into_iter()
        .map(|message| {
            let content = match message.content {
                RawContent::Text(text) => text,
                RawContent::Parts(parts) => {
                    let mut text = String::new();
                    for part in parts {
                        if part.kind == "text" {
                            text.push_str(part.text.as_deref().unwrap_or(""));
                        } else {
                            return Err(ServerError::bad_request(
                                "image and audio chat parts are not supported",
                            ));
                        }
                    }
                    text
                }
            };
            Ok(ChatMessage {
                role: message.role,
                content,
            })
        })
        .collect()
}

fn sse_response<S>(stream: S) -> Response
where
    S: Stream<Item = std::result::Result<Event, axum::Error>> + Send + 'static,
{
    Sse::new(stream).into_response()
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_holdback_across_chunks() {
        let mut stop = StopDetector::new(&["abc".to_owned()]);
        assert_eq!(stop.push("x"), None);
        assert_eq!(stop.take_ready(), "");
        assert_eq!(stop.push("ya"), None);
        assert_eq!(stop.take_ready(), "x");
        assert_eq!(stop.push("bczz"), Some("abc".to_owned()));
        assert_eq!(stop.finish(), "y");
    }
}
