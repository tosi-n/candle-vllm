use std::env;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use candle_vllm::openai::embedded_runtime::{
    EmbeddedCandleVllmHost, EmbeddedChatOutput, EmbeddedRuntimeConfig,
};
use candle_vllm::openai::lora::RegisterAdapterRequest;
use candle_vllm::openai::requests::{
    ChatCompletionRequest, ChatMessage, MessageContentType, Messages,
};

fn usage() -> ! {
    eprintln!(
        "usage: cargo run --example check_lora --features metal -- <model_dir> <adapter_dir> [question]"
    );
    std::process::exit(2);
}

fn build_request(model_id: &str, question: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: Some(model_id.to_string()),
        messages: Messages::Chat(vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(MessageContentType::PureText(
                    "Answer briefly and directly.".to_string(),
                )),
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContentType::PureText(question.to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
        ]),
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(-1),
        n: Some(1),
        max_tokens: Some(64),
        stream: Some(false),
        ..Default::default()
    }
}

async fn run_chat(
    host: &EmbeddedCandleVllmHost,
    request: ChatCompletionRequest,
    adapter_id: Option<String>,
) -> Result<String> {
    match host
        .chat(request, adapter_id, None)
        .await
        .context("embedded runtime chat failed")?
    {
        EmbeddedChatOutput::Completion(response) => Ok(response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default()),
        EmbeddedChatOutput::Streaming(_) => {
            bail!("streaming output is not supported in this probe")
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let model_dir = args.next().unwrap_or_else(|| usage());
    let adapter_dir = args.next().unwrap_or_else(|| usage());
    let question = args
        .next()
        .unwrap_or_else(|| "What does Project Blue Orchard do?".to_string());

    let model_id = env::var("CANDLE_VLLM_MODEL_ID")
        .unwrap_or_else(|_| "Qwen/Qwen3-4B-Instruct-2507".to_string());
    let dtype = env::var("CANDLE_VLLM_DTYPE").unwrap_or_else(|_| "bf16".to_string());
    let block_size = env::var("CANDLE_VLLM_BLOCK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(32);

    let adapter_dir = PathBuf::from(adapter_dir);
    let adapter_path = adapter_dir.join("adapter_model.safetensors");
    let legacy_adapter_path = adapter_dir.join("adapter.safetensors");
    let config_path = adapter_dir.join("adapter_config.json");

    let adapter_path = if adapter_path.exists() {
        adapter_path
    } else if legacy_adapter_path.exists() {
        legacy_adapter_path
    } else {
        bail!(
            "no adapter weights found in {} (expected adapter_model.safetensors or adapter.safetensors)",
            adapter_dir.display()
        );
    };

    if !config_path.exists() {
        bail!(
            "missing adapter config in {} (expected adapter_config.json)",
            adapter_dir.display()
        );
    }

    let runtime = EmbeddedCandleVllmHost::new(EmbeddedRuntimeConfig {
        // Local folder loads use weight_path only; runtime_canonical_model_id
        // keeps adapter compatibility checks aligned with the public model id.
        model_id: None,
        weight_path: Some(model_dir),
        dtype: Some(dtype),
        block_size,
        record_conversation: false,
        runtime_local_only_strict: true,
        runtime_canonical_model_id: Some(model_id.clone()),
        ..Default::default()
    })
    .await
    .context("failed to create embedded runtime")?;

    let request = build_request(&model_id, &question);
    let baseline = run_chat(&runtime, request.clone(), None).await?;

    let adapter_id = format!(
        "probe-{}",
        adapter_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("adapter")
    );
    runtime
        .register_adapter(RegisterAdapterRequest {
            id: adapter_id.clone(),
            adapter_path: adapter_path.display().to_string(),
            config_path: Some(config_path.display().to_string()),
            version: None,
            checksum_sha256: None,
            pinned: true,
            base_model: Some(model_id.clone()),
            artifact_refs: Vec::new(),
            scope: None,
            backend: None,
        })
        .context("failed to register adapter")?;
    runtime
        .load_adapter(&adapter_id, true)
        .await
        .context("failed to load adapter")?;

    let adapted = run_chat(&runtime, request, Some(adapter_id.clone())).await?;

    println!("model: {model_id}");
    println!("adapter: {adapter_id}");
    println!("question: {question}");
    println!("\n--- baseline ---\n{baseline}");
    println!("\n--- adapted ---\n{adapted}");

    Ok(())
}
