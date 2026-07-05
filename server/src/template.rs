use minijinja::{Environment, ErrorKind};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{Result, ServerError};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemplateMode {
    Auto,
    Builtin,
}

#[derive(Clone)]
pub struct ChatTemplate {
    source: Option<String>,
}

impl ChatTemplate {
    pub fn new(source: Option<String>, mode: TemplateMode) -> Self {
        let source = if mode == TemplateMode::Builtin {
            None
        } else {
            source
        };
        Self { source }
    }

    pub fn source(&self) -> &str {
        self.source.as_deref().unwrap_or("builtin")
    }

    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        if let Some(source) = &self.source {
            match render_minijinja(source, messages, add_generation_prompt) {
                Ok(rendered) => return Ok(rendered),
                Err(err) => {
                    // Some GGUF templates (e.g. Qwen's) use Jinja features minijinja
                    // lacks; the builtin ChatML formatter is fixture-faithful. Warn once,
                    // not per request.
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| {
                        tracing::warn!("GGUF chat template unsupported ({err}); using builtin formatter")
                    });
                }
            }
        }
        Ok(render_builtin(messages, add_generation_prompt))
    }
}

fn render_minijinja(
    source: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    let mut env = Environment::new();
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
        },
    );
    let tmpl = env
        .template_from_str(source)
        .map_err(|err| ServerError::internal(format!("chat template parse failed: {err}")))?;
    tmpl.render(json!({
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }))
    .map_err(|err| ServerError::internal(format!("chat template render failed: {err}")))
}

pub fn render_builtin(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for message in messages {
        out.push_str("<|im_start|>");
        out.push_str(&message.role);
        out.push('\n');
        out.push_str(&message.content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_matches_qwen_shape() {
        let rendered = render_builtin(
            &[ChatMessage {
                role: "user".to_owned(),
                content: "What is the capital of France?".to_owned(),
            }],
            true,
        );
        assert_eq!(
            rendered,
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
