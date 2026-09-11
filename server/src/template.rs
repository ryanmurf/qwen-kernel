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

/// Generation cue for models whose chat template opens the assistant turn with
/// a pre-closed think block (Qwen3.6 shape) vs plain instruct models
/// (Qwen3-Next-Instruct). history_boundary() strips this exact suffix, so the
/// anthropic renderer and the cache-boundary logic must both go through
/// [`ChatTemplate::gen_cue`].
pub const CUE_THINK: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
pub const CUE_PLAIN: &str = "<|im_start|>assistant\n";
pub const CUE_THINK_OPEN: &str = "<|im_start|>assistant\n<think>\n";

#[derive(Clone)]
pub struct ChatTemplate {
    source: Option<String>,
    think_cue: bool,
    flash: bool,
}

impl ChatTemplate {
    pub fn new(source: Option<String>, mode: TemplateMode) -> Self {
        // The cue is a property of the MODEL (does its GGUF template scaffold a
        // think block?), not of which renderer ends up used — detect it before
        // Builtin mode drops the source. No template at all keeps the think cue
        // (the pre-template fixture behavior).
        let think_cue = source.as_deref().map_or(true, |s| s.contains("<think>"));
        let source = if mode == TemplateMode::Builtin {
            None
        } else {
            source
        };
        Self {
            source,
            think_cue,
            flash: false,
        }
    }

    pub fn for_architecture(mut self, architecture: Option<&str>) -> Self {
        self.flash = architecture == Some("qwen4exp");
        self
    }

    pub fn is_flash(&self) -> bool {
        self.flash
    }

    pub fn source(&self) -> &str {
        self.source.as_deref().unwrap_or("builtin")
    }

    pub fn gen_cue(&self) -> &'static str {
        if self.flash {
            CUE_THINK_OPEN
        } else if self.think_cue {
            CUE_THINK
        } else {
            CUE_PLAIN
        }
    }

    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        if let Some(source) = &self.source {
            let context =
                json!({"messages":messages,"add_generation_prompt":add_generation_prompt});
            match render_minijinja(source, context, self.flash) {
                Ok(rendered) => return Ok(rendered),
                Err(err) => {
                    if self.flash {
                        return Err(err);
                    } // never silently change this model's prompt format
                    // Some GGUF templates (e.g. Qwen's) use Jinja features minijinja
                    // lacks; the builtin ChatML formatter is fixture-faithful. Warn once,
                    // not per request.
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| {
                        tracing::warn!(
                            "GGUF chat template unsupported ({err}); using builtin formatter"
                        )
                    });
                }
            }
        }
        if self.flash {
            return Err(ServerError::internal(
                "Flash rendering requires its GGUF template",
            ));
        }
        Ok(render_builtin(
            messages,
            add_generation_prompt,
            self.gen_cue(),
        ))
    }

    pub fn render_flash(
        &self,
        messages: serde_json::Value,
        tools: serde_json::Value,
    ) -> Result<String> {
        let source = self.source.as_ref().ok_or_else(|| {
            ServerError::internal("Flash tool rendering requires its GGUF template")
        })?;
        render_minijinja(
            source,
            json!({"messages":messages,"tools":tools,"add_generation_prompt":true}),
            true,
        )
    }
}

fn render_minijinja(source: &str, mut context: serde_json::Value, flash: bool) -> Result<String> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(|_, value, method, args| {
        if let Some(text) = value.as_str() {
            if method == "startswith" || method == "endswith" {
                let (pattern,): (&str,) = minijinja::value::from_args(args)?;
                return Ok((if method == "startswith" {
                    text.starts_with(pattern)
                } else {
                    text.ends_with(pattern)
                })
                .into());
            }
        }
        Err(minijinja::Error::from(ErrorKind::UnknownMethod))
    });
    // The current Flash checkpoint expects open reasoning and supports these
    // named effort levels. Older architectures keep their existing defaults.
    if flash {
        let effort = std::env::var("QK_REASONING_EFFORT").unwrap_or_else(|_| "xhigh".into());
        if !matches!(effort.as_str(), "low" | "medium" | "high" | "xhigh") {
            return Err(ServerError::internal("invalid QK_REASONING_EFFORT"));
        }
        context["reasoning_effort"] = effort.into();
        context["enable_thinking"] = true.into();
        context["preserve_thinking"] = true.into();
    }
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
        },
    );
    let tmpl = env
        .template_from_str(source)
        .map_err(|err| ServerError::internal(format!("chat template parse failed: {err}")))?;
    tmpl.render(context)
        .map_err(|err| ServerError::internal(format!("chat template render failed: {err}")))
}

pub fn render_builtin(
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    gen_cue: &str,
) -> String {
    let mut out = String::new();
    for message in messages {
        out.push_str("<|im_start|>");
        out.push_str(&message.role);
        out.push('\n');
        out.push_str(&message.content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str(gen_cue);
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
            CUE_THINK,
        );
        assert_eq!(
            rendered,
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn cue_follows_template_think_scaffold() {
        // Qwen3.6-style template (mentions <think>) keeps the think cue.
        let think = ChatTemplate::new(Some("{{ '<think>\\n' }}".to_owned()), TemplateMode::Auto);
        assert_eq!(think.gen_cue(), CUE_THINK);
        // Instruct template (no think block) gets the plain cue — even in
        // Builtin mode, where the source is dropped for rendering.
        let plain_src =
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";
        for mode in [TemplateMode::Auto, TemplateMode::Builtin] {
            let plain = ChatTemplate::new(Some(plain_src.to_owned()), mode);
            assert_eq!(plain.gen_cue(), CUE_PLAIN);
        }
        // No template KV at all: fixture status quo (think cue).
        assert_eq!(
            ChatTemplate::new(None, TemplateMode::Auto).gen_cue(),
            CUE_THINK
        );
    }
}
