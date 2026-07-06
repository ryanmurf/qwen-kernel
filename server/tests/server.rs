mod common;

use futures::StreamExt;
use reqwest::StatusCode;
use serde_json::{Value, json};

use common::{spawn_server, write_toy_gguf};

fn recur(seed: u32, n: usize) -> Vec<u32> {
    let mut prev = seed;
    let mut out = Vec::new();
    for _ in 0..n {
        let next = prev.wrapping_mul(1_103_515_245).wrapping_add(12_345) % 248_000;
        out.push(next);
        prev = next;
    }
    out
}

#[tokio::test]
async fn health_tokenize_detokenize_and_completion() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let model = write_toy_gguf(&dir.path().join("toy.gguf"))?;
    let server = spawn_server(&model, 8, 2).await?;
    let client = reqwest::Client::new();

    let health: Value = client
        .get(format!("{}/health", server.base_url))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(health["status"], "ok");

    let tok: Value = client
        .post(format!("{}/tokenize", server.base_url))
        .json(&json!({"content":"abc"}))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(tok["tokens"], json!([257]));

    let detok: Value = client
        .post(format!("{}/detokenize", server.base_url))
        .json(&json!({"tokens":[257]}))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(detok["content"], "abc");

    let completion: Value = client
        .post(format!("{}/completion", server.base_url))
        .json(&json!({"prompt":[5,6,8],"n_predict":4,"return_tokens":true}))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(completion["stop_type"], "limit");
    assert_eq!(completion["tokens"], json!(recur(8, 4)));
    Ok(())
}

#[tokio::test]
async fn completion_stream_and_eos() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let model = write_toy_gguf(&dir.path().join("toy.gguf"))?;
    let server = spawn_server(&model, 8, 2).await?;
    let client = reqwest::Client::new();

    let mut stream = client
        .post(format!("{}/completion", server.base_url))
        .json(&json!({"prompt":[5,6,8],"n_predict":4,"stream":true,"return_tokens":true}))
        .send()
        .await?
        .bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        body.extend(chunk?);
    }
    let text = String::from_utf8(body)?;
    assert!(text.contains("\"stop\":true"));
    assert!(text.contains(&recur(8, 4)[0].to_string()));

    let eos: Value = client
        .post(format!("{}/completion", server.base_url))
        .json(&json!({"prompt":[7,9],"n_predict":8,"return_tokens":true}))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(eos["stop_type"], "eos");
    assert_eq!(eos["tokens"], json!(recur(9, 3)));
    Ok(())
}

#[tokio::test]
async fn malformed_and_bad_ids_are_structured_errors() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let model = write_toy_gguf(&dir.path().join("toy.gguf"))?;
    let server = spawn_server(&model, 1, 1).await?;
    let client = reqwest::Client::new();

    let bad_json = client
        .post(format!("{}/tokenize", server.base_url))
        .header("content-type", "application/json")
        .body("{")
        .send()
        .await?;
    assert_eq!(bad_json.status(), StatusCode::BAD_REQUEST);
    let body: Value = bad_json.json().await?;
    assert_eq!(body["error"]["type"], "invalid_request_error");

    let bad_ids = client
        .post(format!("{}/detokenize", server.base_url))
        .json(&json!({"tokens":[999999]}))
        .send()
        .await?;
    assert_eq!(bad_ids.status(), StatusCode::BAD_REQUEST);
    Ok(())
}

#[tokio::test]
async fn anthropic_messages_non_streaming() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let model = write_toy_gguf(&dir.path().join("toy.gguf"))?;
    let server = spawn_server(&model, 8, 2).await?;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("{}/v1/messages", server.base_url))
        .json(&json!({
            "model": "claude-test",
            "max_tokens": 4,
            "system": "be brief",
            "messages": [{"role": "user", "content": "abc"}]
        }))
        .send()
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await?;
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["model"], "claude-test");
    assert_eq!(body["stop_reason"], "max_tokens");
    assert!(body["content"].is_array());
    assert!(body["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(body["usage"]["output_tokens"], 4);

    let count: Value = client
        .post(format!("{}/v1/messages/count_tokens", server.base_url))
        .json(&json!({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "abc"}]
        }))
        .send()
        .await?
        .json()
        .await?;
    assert!(count["input_tokens"].as_u64().unwrap() > 0);

    let bad = client
        .post(format!("{}/v1/messages", server.base_url))
        .json(&json!({"model": "claude-test", "max_tokens": 4, "messages": []}))
        .send()
        .await?;
    assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
    let bad_body: Value = bad.json().await?;
    assert_eq!(bad_body["type"], "error");
    assert_eq!(bad_body["error"]["type"], "invalid_request_error");
    Ok(())
}

#[tokio::test]
async fn anthropic_messages_streaming_protocol() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let model = write_toy_gguf(&dir.path().join("toy.gguf"))?;
    let server = spawn_server(&model, 8, 2).await?;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("{}/v1/messages", server.base_url))
        .json(&json!({
            "model": "claude-test",
            "max_tokens": 4,
            "stream": true,
            "messages": [{"role": "user", "content": "abc"}]
        }))
        .send()
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .starts_with("text/event-stream")
    );
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        body.extend(chunk?);
    }
    let text = String::from_utf8(body)?;
    let events: Vec<&str> = text
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect();
    assert_eq!(events.first(), Some(&"message_start"));
    assert!(events.contains(&"message_delta"));
    assert_eq!(events.last(), Some(&"message_stop"));
    let start_data = text
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("first data line");
    let start: Value = serde_json::from_str(start_data)?;
    assert_eq!(start["type"], "message_start");
    assert!(start["message"]["usage"]["input_tokens"].as_u64().unwrap() > 0);
    let delta_data = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str::<Value>(data).unwrap())
        .find(|value| value["type"] == "message_delta")
        .expect("message_delta event");
    assert_eq!(delta_data["delta"]["stop_reason"], "max_tokens");
    assert_eq!(delta_data["usage"]["output_tokens"], 4);
    Ok(())
}

#[tokio::test]
async fn concurrent_requests_do_not_mix_tokens() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let model = write_toy_gguf(&dir.path().join("toy.gguf"))?;
    let server = spawn_server(&model, 4, 2).await?;
    let client = reqwest::Client::new();
    let mut tasks = Vec::new();
    for seed in [8u32, 9, 10, 11] {
        let client = client.clone();
        let url = format!("{}/completion", server.base_url);
        tasks.push(tokio::spawn(async move {
            let body: Value = client
                .post(url)
                .json(&json!({"prompt":[seed],"n_predict":4,"return_tokens":true}))
                .send()
                .await?
                .json()
                .await?;
            anyhow::Ok((seed, body["tokens"].clone()))
        }));
    }
    for task in tasks {
        let (seed, tokens) = task.await??;
        assert_eq!(tokens, json!(recur(seed, 4)));
    }
    Ok(())
}
