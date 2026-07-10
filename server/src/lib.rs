pub mod anthropic;
pub mod engine;
pub mod ffi;
pub mod gguf;
pub mod http;
pub mod split;
pub mod template;
pub mod tokenizer;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    QueueFull(String),
    #[error("{0}")]
    Internal(String),
}

impl ServerError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    fn status_and_type(&self) -> (StatusCode, &'static str) {
        match self {
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            Self::QueueFull(_) => (StatusCode::SERVICE_UNAVAILABLE, "server_overloaded"),
            Self::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        }
    }
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorShape<'a>,
}

#[derive(Serialize)]
struct ErrorShape<'a> {
    code: u16,
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, kind) = self.status_and_type();
        let message = self.to_string();
        (
            status,
            Json(ErrorBody {
                error: ErrorShape {
                    code: status.as_u16(),
                    message: &message,
                    kind,
                },
            }),
        )
            .into_response()
    }
}

impl From<anyhow::Error> for ServerError {
    fn from(value: anyhow::Error) -> Self {
        Self::Internal(value.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ServerError>;
