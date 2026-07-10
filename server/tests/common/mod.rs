use std::path::{Path, PathBuf};
use std::sync::Arc;

use server::engine::EngineThread;
use server::ffi::QkConfig;
use server::gguf;
use server::http::{AppState, router};
use server::template::{ChatTemplate, TemplateMode};
use server::tokenizer::{TOKEN_CONTROL, TOKEN_NORMAL, Tokenizer, build_byte_maps};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

pub struct TestServer {
    pub base_url: String,
    pub engine: Option<EngineThread>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
        if let Some(engine) = self.engine.take() {
            engine.shutdown();
        }
    }
}

pub async fn spawn_server(model: &Path, queue: usize, slots: u32) -> anyhow::Result<TestServer> {
    let metadata = gguf::read_metadata(model)?;
    let tokenizer = Arc::new(Tokenizer::from_config(metadata.tokenizer)?);
    let engine = EngineThread::start(
        Path::new(env!("QK_STUB_LIB")),
        model,
        QkConfig {
            n_slots: slots,
            n_ctx: 64,
            chunk: 4,
        },
        None,
    )?;
    let state = AppState {
        tokenizer,
        chat_template: Arc::new(ChatTemplate::new(
            metadata.chat_template,
            TemplateMode::Builtin,
        )),
        engine: engine.handle(),
        model_path: model.to_owned(),
        model_alias: metadata.name,
        queue: Arc::new(Semaphore::new(queue.saturating_add(slots as usize))),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}", listener.local_addr()?);
    let app = router(state);
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok(TestServer {
        base_url,
        engine: Some(engine),
        task,
    })
}

pub fn write_toy_gguf(path: &Path) -> anyhow::Result<PathBuf> {
    let (byte_to_char, _) = build_byte_maps();
    let mut tokens = byte_to_char.iter().map(char::to_string).collect::<Vec<_>>();
    tokens.push("ab".to_owned());
    tokens.push("abc".to_owned());
    tokens.push("<|im_start|>".to_owned());
    tokens.push("<|im_end|>".to_owned());
    while tokens.len() < 300 {
        tokens.push(format!("[PAD{}]", tokens.len()));
    }
    let mut token_types = vec![TOKEN_NORMAL; tokens.len()];
    token_types[258] = TOKEN_CONTROL;
    token_types[259] = TOKEN_CONTROL;
    let merges = vec!["a b".to_owned(), "ab c".to_owned()];
    let mut bytes = Vec::new();
    bytes.extend(0x4655_4747u32.to_le_bytes());
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(11u64.to_le_bytes());
    write_kv_string(&mut bytes, "general.architecture", "qwen35moe");
    write_kv_string(&mut bytes, "general.name", "toy-qk");
    write_kv_string(&mut bytes, "tokenizer.ggml.model", "gpt2");
    write_kv_string(&mut bytes, "tokenizer.ggml.pre", "qwen35");
    write_kv_string_array(&mut bytes, "tokenizer.ggml.tokens", &tokens);
    write_kv_i32_array(&mut bytes, "tokenizer.ggml.token_type", &token_types);
    write_kv_string_array(&mut bytes, "tokenizer.ggml.merges", &merges);
    write_kv_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 259);
    write_kv_u32(&mut bytes, "tokenizer.ggml.bos_token_id", 258);
    write_kv_bool(&mut bytes, "tokenizer.ggml.add_bos_token", false);
    write_kv_string(&mut bytes, "tokenizer.chat_template", "");
    std::fs::write(path, bytes)?;
    Ok(path.to_owned())
}

fn write_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend((value.len() as u64).to_le_bytes());
    bytes.extend(value.as_bytes());
}

fn write_key(bytes: &mut Vec<u8>, key: &str, ty: u32) {
    write_str(bytes, key);
    bytes.extend(ty.to_le_bytes());
}

fn write_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
    write_key(bytes, key, 8);
    write_str(bytes, value);
}

fn write_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
    write_key(bytes, key, 4);
    bytes.extend(value.to_le_bytes());
}

fn write_kv_bool(bytes: &mut Vec<u8>, key: &str, value: bool) {
    write_key(bytes, key, 7);
    bytes.push(u8::from(value));
}

fn write_kv_string_array(bytes: &mut Vec<u8>, key: &str, values: &[String]) {
    write_key(bytes, key, 9);
    bytes.extend(8u32.to_le_bytes());
    bytes.extend((values.len() as u64).to_le_bytes());
    for value in values {
        write_str(bytes, value);
    }
}

fn write_kv_i32_array(bytes: &mut Vec<u8>, key: &str, values: &[i32]) {
    write_key(bytes, key, 9);
    bytes.extend(5u32.to_le_bytes());
    bytes.extend((values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
}
