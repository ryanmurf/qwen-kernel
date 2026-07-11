use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use server::engine::EngineThread;
use server::ffi::QkConfig;
use server::gguf;
use server::http::{AppState, router};
use server::template::{ChatTemplate, TemplateMode};
use server::tokenizer::Tokenizer;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "qk-server")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8090)]
    port: u16,
    #[arg(long, default_value_t = 8)]
    slots: u32,
    #[arg(long, default_value_t = 1024)]
    ctx: u32,
    #[arg(long, default_value_t = 8)]
    chunk: u32,
    #[arg(long, env = "QK_ENGINE_LIB", default_value = "../build/libqk.so")]
    engine_lib: PathBuf,
    /// Split serving (docs/split-serving.md): host:port of the worker stage
    /// (`qk pipe-worker`) holding the remaining layers. Requires this process
    /// to load the head stage (QK_LAYERS=0:S in the environment).
    #[arg(long)]
    split_next: Option<String>,
    #[arg(long, default_value_t = 64)]
    queue: usize,
    #[arg(long, value_enum, default_value_t = CliTemplateMode::Auto)]
    chat_template: CliTemplateMode,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Clone, Debug, ValueEnum)]
enum CliTemplateMode {
    Auto,
    Builtin,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(&cli.log_level))
        .init();

    let metadata = gguf::read_metadata(&cli.model)?;
    if !matches!(
        metadata.architecture.as_deref(),
        Some("qwen35moe") | Some("qwen3next")
    ) {
        tracing::warn!("unexpected model architecture: {:?}", metadata.architecture);
    }
    let tokenizer = Arc::new(Tokenizer::from_config(metadata.tokenizer)?);
    let template_mode = match cli.chat_template {
        CliTemplateMode::Auto => TemplateMode::Auto,
        CliTemplateMode::Builtin => TemplateMode::Builtin,
    };
    let chat_template = Arc::new(ChatTemplate::new(metadata.chat_template, template_mode));
    tracing::info!(
        "generation cue: {}",
        if chat_template.gen_cue().contains("<think>") {
            "think scaffold (Qwen3.6 shape)"
        } else {
            "plain assistant turn (instruct shape)"
        }
    );
    tracing::info!(
        "sampling policy: QK_TEMP_DEFAULT={} QK_TEMP_CAP={} (explicit temperature wins; 0/absent-with-no-default = greedy)",
        std::env::var("QK_TEMP_DEFAULT").unwrap_or_else(|_| "0 (greedy)".into()),
        std::env::var("QK_TEMP_CAP").unwrap_or_else(|_| "none".into()),
    );
    let engine_thread = EngineThread::start(
        &cli.engine_lib,
        &cli.model,
        QkConfig {
            n_slots: cli.slots,
            n_ctx: cli.ctx,
            chunk: cli.chunk,
        },
        cli.split_next.clone(),
    )?;
    let state = AppState {
        tokenizer,
        chat_template,
        engine: engine_thread.handle(),
        model_path: cli.model.clone(),
        model_alias: metadata.name,
        queue: Arc::new(Semaphore::new(cli.queue.saturating_add(cli.slots as usize))),
    };
    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("qk-server listening on http://{}", listener.local_addr()?);
    let app = router(state);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    engine_thread.shutdown();
    Ok(())
}
