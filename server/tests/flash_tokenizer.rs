use serde::Deserialize;
use server::{
    gguf,
    template::{ChatMessage, ChatTemplate, TemplateMode},
    tokenizer::Tokenizer,
};
use std::path::Path;

#[derive(Deserialize)]
struct Case {
    text: String,
    tokens: Vec<u32>,
}
#[derive(Deserialize)]
struct Chat {
    messages: Vec<ChatMessage>,
    prompt: serde_json::Value,
}
#[derive(Deserialize)]
struct Fixtures {
    tokenize: Vec<Case>,
    chat: Chat,
}

#[test]
fn flash_next_tokenizer_and_template_match_reference() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("warn")
        .with_test_writer()
        .try_init();
    let Some(model) = std::env::var_os("QK_FLASH_GGUF") else {
        return Ok(());
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/flash_tokenizer.json");
    let fixture: Fixtures = serde_json::from_slice(&std::fs::read(path)?)?;
    let meta = gguf::read_metadata(Path::new(&model))?;
    assert_eq!(meta.architecture.as_deref(), Some("qwen4exp"));
    let tokenizer = Tokenizer::from_config(meta.tokenizer)?;
    for case in fixture.tokenize {
        assert_eq!(
            tokenizer.tokenize(&case.text, true)?,
            case.tokens,
            "{:?}",
            case.text
        );
    }
    let template = ChatTemplate::new(meta.chat_template, TemplateMode::Auto)
        .for_architecture(meta.architecture.as_deref());
    let prompt = template.render(&fixture.chat.messages, true)?;
    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
    assert_eq!(template.gen_cue(), "<|im_start|>assistant\n<think>\n");
    if std::env::var("QK_REASONING_EFFORT").as_deref() == Ok("xhigh") {
        assert_eq!(prompt, fixture.chat.prompt["prompt"].as_str().unwrap());
    }
    Ok(())
}
