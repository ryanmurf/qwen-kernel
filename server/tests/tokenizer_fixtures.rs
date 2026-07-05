use std::path::Path;

use serde::Deserialize;
use server::gguf;
use server::template::{ChatMessage, ChatTemplate, TemplateMode};
use server::tokenizer::Tokenizer;

#[derive(Deserialize)]
struct Fixtures {
    tokenize: Vec<TokenCase>,
    tokenize_special: Vec<SpecialCase>,
    chat: Vec<ChatCase>,
}

#[derive(Deserialize)]
struct TokenCase {
    text: String,
    tokens: Vec<u32>,
    detok: String,
}

#[derive(Deserialize)]
struct SpecialCase {
    text: String,
    parse_special: bool,
    tokens: Vec<u32>,
}

#[derive(Deserialize)]
struct ChatCase {
    messages: Vec<ChatMessage>,
    prompt: String,
}

#[test]
fn real_vocab_fixtures_match_llama_cpp() -> anyhow::Result<()> {
    let Some(model) = std::env::var_os("QK_MODEL_GGUF") else {
        println!("skipping real-vocab fixtures: QK_MODEL_GGUF is unset");
        return Ok(());
    };
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("tokenizer_fixtures.json");
    let fixtures: Fixtures = serde_json::from_slice(&std::fs::read(fixture_path)?)?;
    let metadata = gguf::read_metadata(Path::new(&model))?;
    let tokenizer = Tokenizer::from_config(metadata.tokenizer)?;

    for case in fixtures.tokenize {
        let tokens = tokenizer.tokenize(&case.text, true)?;
        assert_eq!(tokens, case.tokens, "tokenize failed for {:?}", case.text);
        let detok = tokenizer.detokenize(&case.tokens, false)?;
        assert_eq!(detok, case.detok, "detokenize failed for {:?}", case.text);
    }

    for case in fixtures.tokenize_special {
        let tokens = tokenizer.tokenize(&case.text, case.parse_special)?;
        assert_eq!(tokens, case.tokens, "special tokenize failed");
    }

    let metadata = gguf::read_metadata(Path::new(&model))?;
    let template = ChatTemplate::new(metadata.chat_template, TemplateMode::Auto);
    for case in fixtures.chat {
        assert_eq!(template.render(&case.messages, true)?, case.prompt);
    }
    Ok(())
}
