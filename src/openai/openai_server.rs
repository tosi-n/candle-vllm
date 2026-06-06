use super::logger::ChatCompletionLogger;
use super::requests::Messages;
use super::requests::{
    normalize_empty_openai_tool_results, validate_openai_tool_messages, ChatCompletionRequest,
    EmbeddingRequest, EmbeddingType, EncodingFormat,
};
use super::responses::{APIError, ChatCompletionResponse, ChatResponder};
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::streaming::{ChatResponse, Streamer, StreamingStatus};
use super::OpenAIServerData;
use crate::openai::lora::{LoadAdapterRequest, RegisterAdapterRequest, UnloadAdapterRequest};
use crate::openai::multimodal::{build_messages_and_images, ImageData};
use crate::openai::{resolve_tools_for_request, ResolvedToolConfig};
use crate::tools::helpers::{
    build_invalid_tool_call_feedback, build_tool_schema_map, filter_tool_calls,
};
use crate::tools::stream_parser::{
    detect_prefilled_reasoning_end_marker, extract_reasoning_content,
};
use axum::response::sse::KeepAlive;
use axum::{
    extract::{Json, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
};
use flume;
use serde_json::json;
use std::env;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Notify;
use tokio::time::Duration;
use tracing::debug;
use uuid::Uuid;

const REQUEST_ADMISSION_DECODE_BUDGET_TOKENS: usize = 4096;

fn current_model_name(data: &OpenAIServerData) -> Result<String, APIError> {
    let model = data.model.read();
    let model_name = model.model_name();
    if model_name.is_empty() {
        Err(APIError::new("Missing pipeline".to_string()))
    } else {
        Ok(model_name.to_string())
    }
}

fn resolve_response_model_name(requested: Option<&str>, current: &str) -> String {
    match requested.map(str::trim) {
        None | Some("") | Some("default") => current.to_string(),
        Some(name) => name.to_string(),
    }
}

// Get prompt, roles
async fn get_gen_prompt(
    data: &OpenAIServerData,
    request: &ChatCompletionRequest,
    tool_config: &ResolvedToolConfig,
) -> Result<(String, Option<ImageData>), APIError> {
    let model = data.model.read();
    let mut conversation = model.conversation();
    let image_config = model.image_config();
    drop(model);
    let mut image_data = None;

    match &request.messages {
        Messages::Literal(msg) => {
            conversation.append_message("user".to_string(), msg.clone());
        }
        Messages::Chat(messages) => {
            let (render_messages, images) =
                build_messages_and_images(messages, image_config.as_ref())
                    .map_err(APIError::from)?;
            image_data = images;
            for message in render_messages {
                let role = message.role.as_str();
                if role == "system" {
                    conversation.set_system_message(Some(message.content.clone()));
                    continue;
                }
                conversation.append_template_message(message);
            }
        }
        Messages::Map(messages) => {
            for message in messages {
                let role = message
                    .get("role")
                    .ok_or(APIError::new("Message key `role` not found.".to_string()))?;
                let content = message
                    .get("content")
                    .ok_or(APIError::new(
                        "Message key `content` not found.".to_string(),
                    ))?
                    .clone();

                if role == "system" {
                    conversation.set_system_message(Some(content.clone()));
                } else {
                    use crate::openai::conversation::Message;
                    conversation.append_template_message(Message {
                        role: role.to_string(),
                        content,
                        num_images: 0,
                        reasoning_content: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
        }
    }

    let enable_thinking = request.thinking.unwrap_or(true);
    let prompt = conversation.get_prompt(enable_thinking, &tool_config.tools);

    Ok((prompt, image_data))
}

async fn check_length(
    request: &ChatCompletionRequest,
    prompt: String,
    data: &OpenAIServerData,
) -> Result<Vec<u32>, APIError> {
    let token_ids = {
        let model = data.model.read();
        model
            .tokenizer()
            .encode_fast(prompt, true)
            .map_err(APIError::from)?
            .get_ids()
            .to_vec()
    };

    let max_gen_tokens = request
        .max_tokens
        .unwrap_or(data.pipeline_config.default_max_tokens);

    if token_ids.len() >= data.pipeline_config.max_model_len {
        Err(APIError::new(format!(
            "This model's maximum context length is {} tokens. \
            However, you requested {} tokens ({} in the messages, \
            {} in the completion). \nPlease clear the chat history or reduce the length of the \
            messages.",
            data.pipeline_config.max_model_len,
            max_gen_tokens + token_ids.len(),
            token_ids.len(),
            max_gen_tokens
        )))
    } else {
        Ok(token_ids)
    }
}

fn parse_adapter_id(headers: &HeaderMap, request: &ChatCompletionRequest) -> Option<String> {
    if let Some(adapter_id) = headers
        .get("x-runtime-adapter-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(adapter_id.to_string());
    }

    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.runtime.as_ref())
        .and_then(|runtime| runtime.adapter_id.clone())
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
}

fn parse_adapter_schedule(
    request: &ChatCompletionRequest,
) -> Option<Vec<crate::openai::requests::AdapterScheduleStep>> {
    let mut schedule = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.runtime.as_ref())
        .and_then(|runtime| runtime.adapter_schedule.clone())?;

    schedule.retain(|step| !step.adapter_id.trim().is_empty());
    if schedule.is_empty() {
        return None;
    }

    schedule.sort_by_key(|step| step.start_step);
    for step in &mut schedule {
        step.adapter_id = step.adapter_id.trim().to_string();
    }
    Some(schedule)
}

fn parse_session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-runtime-session-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct WarmupAdaptersRequest {
    pub adapter_ids: Vec<String>,
    #[serde(default)]
    pub wait: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WarmupAdaptersResponse {
    pub requested: usize,
    pub queued: usize,
    pub loaded: usize,
    pub failed: Vec<String>,
}

fn ensure_local_adapter_scope(raw: Option<&str>) -> Result<(), APIError> {
    let Some(raw) = raw else {
        return Ok(());
    };
    let scope = raw.trim().to_lowercase();
    if scope.is_empty() || scope == "local" {
        return Ok(());
    }
    Err(APIError::new(format!(
        "Invalid adapter scope '{}'. This runtime only supports local adapter operations.",
        raw
    )))
}

#[utoipa::path(
    post,
    tag = "candle-vllm",
    path = "/v1/chat/completions",
    request_body = ChatCompletionRequest,
    responses((status = 200, description = "Chat completions"))
)]
pub async fn chat_completions(
    State(data): State<Arc<OpenAIServerData>>,
    headers: HeaderMap,
    request: Json<ChatCompletionRequest>,
) -> ChatResponder {
    let mut request = request.0;
    let logger = ChatCompletionLogger::new();
    if let Some(ref l) = logger {
        l.log_request(&request);
    }
    if let Messages::Chat(messages) = &mut request.messages {
        normalize_empty_openai_tool_results(messages);
        if let Err(err) = validate_openai_tool_messages(messages) {
            return ChatResponder::ValidationError(APIError::new(err));
        }
    }

    #[cfg(feature = "nccl")]
    use crate::openai::communicator::DaemonManager;
    #[cfg(feature = "nccl")]
    if !DaemonManager::is_master_rank() {
        return ChatResponder::ModelError(APIError::from(
            "Daemon process unable to generate response, please request server port of the main process!",
        ));
    }

    if request.logit_bias.as_ref().is_some()
        && request.logit_bias.as_ref().is_some_and(|x| !x.is_empty())
    {
        return ChatResponder::ValidationError(APIError::new_str(
            "`logit_bias` is not currently supported.",
        ));
    }

    let session_id = parse_session_id(&headers);
    let mut adapter_id = parse_adapter_id(&headers, &request);
    let adapter_schedule = parse_adapter_schedule(&request);
    if adapter_id.is_none() {
        if let Some(session_id) = session_id.as_ref() {
            adapter_id = data.session_adapters.read().get(session_id).cloned();
        }
    }
    if let (Some(session_id), Some(adapter_id)) = (session_id.as_ref(), adapter_id.as_ref()) {
        data.session_adapters
            .write()
            .insert(session_id.clone(), adapter_id.clone());
    }
    if let Some(adapter) = adapter_id.as_deref() {
        if let Err(err) = data.lora_manager.ensure_loaded_async(adapter).await {
            return ChatResponder::ValidationError(APIError::new(format!(
                "Failed to prepare adapter '{}': {}",
                adapter, err
            )));
        }
    }

    let tool_config = match resolve_tools_for_request(
        &request.tools,
        &request.tool_choice,
        data.mcp_manager.as_ref(),
    ) {
        Ok(config) => config,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let (prompt, image_data) = match get_gen_prompt(&data, &request, &tool_config).await {
        Ok(p) => p,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let token_ids = match check_length(&request, prompt.clone(), &data).await {
        Ok(ids) => ids,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    debug!("\n\n\nPrompt {:?}", prompt);
    if let Some(ref l) = logger {
        l.log_prompt(&prompt);
    }

    let request_id = format!("cmpl-{}", Uuid::new_v4());

    let mut max_request_tokens = request
        .max_tokens
        .unwrap_or(data.pipeline_config.default_max_tokens);

    let max_model_decode_tokens = data
        .pipeline_config
        .max_model_len
        .saturating_sub(token_ids.len())
        .saturating_sub(10);
    if max_request_tokens > max_model_decode_tokens {
        tracing::warn!(
            "Requested max_tokens {} exceeds remaining model context {}, max_tokens changed to {}.",
            max_request_tokens,
            max_model_decode_tokens,
            max_model_decode_tokens
        );
        max_request_tokens = max_model_decode_tokens;
    }
    if max_request_tokens == 0 {
        return ChatResponder::ValidationError(APIError::new(format!(
            "Requested prompt({} tokens) leaves no room for generated tokens within maximum model context {}.",
            token_ids.len(),
            data.pipeline_config.max_model_len
        )));
    }

    // Query prefix cache to determine how many prompt tokens are already cached
    let mut cached_tokens = {
        let mut model = data.model.write();
        model.query_prefix_cache_match_tokens(&token_ids)
    };
    let mut new_tokens = token_ids.len().saturating_sub(cached_tokens);
    let minimum_decode_budget_tokens =
        max_request_tokens.min(REQUEST_ADMISSION_DECODE_BUDGET_TOKENS);
    let mut target_required_tokens = new_tokens.saturating_add(max_request_tokens);
    let mut minimum_required_tokens = new_tokens.saturating_add(minimum_decode_budget_tokens);

    let mut available_tokens = {
        let mut model = data.model.write();
        let (available_tokens, evicted) = model.ensure_available_kv_tokens(target_required_tokens);
        if evicted > 0 {
            tracing::warn!(
                "Evicted {} prefix cache block(s) to reserve {} KV tokens for request admission ({} new prompt + {} requested decode).",
                evicted,
                target_required_tokens,
                new_tokens,
                max_request_tokens
            );
        }
        available_tokens
    };
    loop {
        let refreshed_cached_tokens = {
            let mut model = data.model.write();
            model.query_prefix_cache_match_tokens(&token_ids)
        };
        if refreshed_cached_tokens == cached_tokens {
            break;
        }

        cached_tokens = refreshed_cached_tokens;
        new_tokens = token_ids.len().saturating_sub(cached_tokens);
        target_required_tokens = new_tokens.saturating_add(max_request_tokens);
        minimum_required_tokens = new_tokens.saturating_add(minimum_decode_budget_tokens);
        let (refreshed_available_tokens, evicted) = {
            let mut model = data.model.write();
            model.ensure_available_kv_tokens(target_required_tokens)
        };
        if evicted > 0 {
            tracing::warn!(
                "Evicted {} additional prefix cache block(s) after prefix-cache hit changed; reserving {} KV tokens ({} new prompt + {} requested decode).",
                evicted,
                target_required_tokens,
                new_tokens,
                max_request_tokens
            );
        }
        available_tokens = refreshed_available_tokens;
        if evicted == 0 {
            break;
        }
    }

    if minimum_required_tokens > available_tokens {
        if available_tokens <= new_tokens {
            return ChatResponder::ValidationError(APIError::new(format!(
                "Requested prompt({} tokens, {} new after prefix cache) is  \
                larger than available kvcache (maximum {} tokens).\n \
                You can increase kvcache by setting `--gpu-memory-fraction` (default 0.5) to a larger value!",
                token_ids.len(),
                new_tokens,
                available_tokens
            )));
        }
        return ChatResponder::ValidationError(APIError::new(format!(
            "Requested prompt({} tokens, {} new after prefix cache) plus {} decode budget tokens is \
            larger than available kvcache (maximum {} tokens).\n \
            You can increase kvcache by setting `--gpu-memory-fraction` (default 0.5) to a larger value!",
            token_ids.len(),
            new_tokens,
            minimum_decode_budget_tokens,
            available_tokens
        )));
    }

    if target_required_tokens > available_tokens {
        tracing::warn!(
            "Request admitted with {} KV tokens available, below requested reservation {} tokens but enough for {} new prompt tokens plus {} decode budget tokens ({} cached prompt tokens).",
            available_tokens,
            target_required_tokens,
            new_tokens,
            minimum_decode_budget_tokens,
            cached_tokens
        );
    }

    let generation_cfg = data.pipeline_config.generation_cfg.as_ref().unwrap();
    let mut sampling_params = match SamplingParams::new(
        request.n.unwrap_or(1),
        request.best_of,
        request
            .presence_penalty
            .unwrap_or(generation_cfg.presence_penalty.unwrap_or(0.0)),
        request
            .frequency_penalty
            .unwrap_or(generation_cfg.frequency_penalty.unwrap_or(0.0)),
        request.repeat_last_n,
        request.temperature.or(generation_cfg.temperature),
        request.top_p.or(generation_cfg.top_p),
        request.min_p.or(generation_cfg.min_p),
        request.top_k.or(generation_cfg.top_k),
        request.use_beam_search.unwrap_or(false),
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        request.stop.clone(),
        request.stop_token_ids.clone().unwrap_or_default(),
        request.ignore_eos.unwrap_or(false),
        max_request_tokens,
        None,
        None,
        request.skip_special_tokens.unwrap_or(true),
        request.thinking,
    ) {
        Ok(params) => params,
        Err(e) => return ChatResponder::ValidationError(e),
    };
    let has_tools = !tool_config.tools.is_empty();
    sampling_params.mcp_mode = if has_tools { Some(true) } else { None };

    let prefilled_reasoning_end = detect_prefilled_reasoning_end_marker(&prompt);

    let (response_tx, rx) = flume::unbounded();
    tracing::info!("{:?}", sampling_params);

    let data_clone = data.clone();
    let request_id_clone = request_id.clone();
    let stream_request = request.stream.is_some_and(|x| x);
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let model_name = match current_model_name(&data) {
        Ok(current) => resolve_response_model_name(request.model.as_deref(), &current),
        Err(e) => return ChatResponder::ModelError(e),
    };
    let sync_notify = Arc::new(Notify::new());
    let sync_completion_notify = if stream_request {
        None
    } else {
        Some(Arc::clone(&sync_notify))
    };
    let adapter_id_for_engine = adapter_id.clone();
    let adapter_schedule_for_engine = adapter_schedule.clone();
    let request_tools_for_engine = tool_config.tools.clone();
    let response_tools = tool_config.tools.clone();

    let _ = tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(async move {
            {
                let mut model = data.model.write();
                model.add_request(
                    token_ids,
                    request_id.clone(),
                    SystemTime::now(),
                    sampling_params,
                    request.logprobs.unwrap_or(false),
                    false,
                    EncodingFormat::default(),
                    EmbeddingType::default(),
                    request_tools_for_engine.clone(),
                    image_data,
                    if stream_request {
                        Some(Arc::new(response_tx))
                    } else {
                        None
                    },
                    sync_completion_notify,
                    include_usage,
                    prefilled_reasoning_end,
                    adapter_id_for_engine,
                    adapter_schedule_for_engine,
                );
                model.notify.notify_one();
            }
        });
    });

    if stream_request {
        if let Some(ref l) = logger {
            l.log_start_response();
        }
        ChatResponder::Streamer(
            Sse::new(Streamer {
                rx,
                status: StreamingStatus::Uninitialized,
                logger,
            })
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_millis(
                        env::var("KEEP_ALIVE_INTERVAL")
                            .map(|val| val.parse::<u64>().unwrap_or(100))
                            .unwrap_or(100),
                    ))
                    .text("keep-alive-text"),
            ),
        )
    } else {
        // wait until current response finished
        tracing::warn!("waiting response for sync request {}", request_id_clone);
        sync_notify.as_ref().notified().await;
        // Re-acquire read lock to get the response
        // Note: we need to drop the lock later
        let (choices, usage) = {
            let model = data_clone.model.read();
            if !model.completion_records.contains_key(&request_id_clone) {
                return ChatResponder::ModelError(APIError::from(format!(
                    "Unable to generate response for request {request_id_clone}"
                )));
            }
            let record = &model.completion_records[&request_id_clone];
            (record.0.clone(), record.1.clone())
        };

        let mut final_choices = choices.clone();

        // Extract reasoning content BEFORE tool parsing so that reasoning
        // blocks are preserved even when tool calls consume the remaining
        // content.  Without this, tool parsing sets content=None and the
        // subsequent reasoning extraction finds nothing.
        if crate::stream_as_reasoning_content() {
            for choice in &mut final_choices {
                if let Some(text) = choice.message.content.take() {
                    match extract_reasoning_content(&text) {
                        Some((reasoning, remaining)) => {
                            choice.message.content = if remaining.is_empty() {
                                None
                            } else {
                                Some(remaining)
                            };
                            choice.message.reasoning_content = Some(reasoning);
                        }
                        None => {
                            choice.message.content = Some(text);
                        }
                    }
                }
            }
        }

        if has_tools {
            let parser = crate::tools::parser::ToolParser::new();
            let tool_schemas = build_tool_schema_map(&response_tools);
            for choice in &mut final_choices {
                let parsed_calls = if let Some(calls) = choice.message.tool_calls.take() {
                    calls
                } else if let Some(content) = &choice.message.content {
                    parser.parse(content)
                } else {
                    Vec::new()
                };

                if parsed_calls.is_empty() {
                    continue;
                }

                let (valid_calls, invalid_calls) = filter_tool_calls(&parsed_calls, &tool_schemas);
                if !invalid_calls.is_empty() {
                    tracing::warn!(
                        "Dropped {} invalid tool call(s) before response",
                        invalid_calls.len()
                    );
                }
                if valid_calls.is_empty() {
                    if let Some(feedback) =
                        build_invalid_tool_call_feedback(&invalid_calls, &tool_schemas, None)
                    {
                        choice.message.content = Some(feedback);
                    }
                    choice.finish_reason = Some("stop".to_string());
                    continue;
                }

                choice.message.tool_calls = Some(valid_calls);
                choice.message.content = None;
                choice.finish_reason = Some("tool_calls".to_string());
            }
        }

        let response = ChatCompletionResponse {
            id: request_id_clone,
            choices: final_choices,
            created: usage.created,
            model: model_name,
            object: "chat.completion",
            usage: usage,
        };
        if let Some(ref l) = logger {
            l.log_response(&response);
        }
        ChatResponder::Completion(response)
    }
}

#[utoipa::path(
    post,
    tag = "candle-vllm",
    path = "/v1/embeddings",
    request_body = EmbeddingRequest,
    responses((status = 200, description = "Embeddings"))
)]
pub async fn create_embeddings(
    State(data): State<Arc<OpenAIServerData>>,
    request: Json<EmbeddingRequest>,
) -> ChatResponder {
    let input = request.input.clone();
    let prompts = input.into_vec();

    //For now only support single prompt for simplicity, loop if multiple
    if prompts.len() != 1 {
        return ChatResponder::ValidationError(APIError::new_str(
            "Currently only support single string or token array input.",
        ));
    }

    let prompt_str = prompts[0].clone();

    //TODO: Reuse check_length or similar logic. For now simplified.
    let token_ids = {
        let model = data.model.read();
        match model.tokenizer().encode_fast(prompt_str, true) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(e) => return ChatResponder::ValidationError(APIError::from(e)),
        }
    };

    let available_tokens = {
        let mut model = data.model.write();
        let (available_tokens, evicted) = model.ensure_available_kv_tokens(token_ids.len());
        if evicted > 0 {
            tracing::warn!(
                "Evicted {} prefix cache block(s) before embedding length check.",
                evicted
            );
        }
        available_tokens
    };

    if token_ids.len() >= available_tokens {
        return ChatResponder::ValidationError(APIError::new_str("Prompt too long."));
    }

    let request_id = format!("embd-{}", Uuid::new_v4());

    // Create sampling params for embedding (max_tokens=0, etc)
    // We reuse SamplingParams but most fields irrelevant.
    let _generation_cfg = data.pipeline_config.generation_cfg.as_ref().unwrap();
    let sampling_params = match SamplingParams::new(
        1,
        None,
        0.0,
        0.0,
        None,
        None,
        None,
        None,
        None,
        false,
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        None,
        Vec::new(),
        false,
        1,
        None,
        None,
        true,
        None,
    ) {
        Ok(params) => params,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let (response_tx, rx) = flume::unbounded();

    let request_id_clone = request_id.clone();

    let _ = tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(async move {
            {
                let mut model = data.model.write();
                model.add_request(
                    token_ids,
                    request_id_clone,
                    SystemTime::now(),
                    sampling_params,
                    false,
                    true, //is_embedding
                    request.encoding_format.clone(),
                    request.embedding_type.clone(),
                    Vec::new(),
                    None,
                    Some(Arc::new(response_tx)),
                    None,
                    false,
                    None,
                    None,
                    None,
                );
                model.notify.notify_one();
            }
        });
    });

    // Wait for response from channel
    // Embedding is strictly one response.
    match rx.recv_async().await {
        Ok(ChatResponse::Embedding(resp)) => ChatResponder::Embedding(resp),
        Ok(ChatResponse::ModelError(e)) => ChatResponder::ModelError(APIError::new_str(&e)),
        Ok(_) => ChatResponder::InternalError(APIError::new(format!("Unexpected response type"))),
        Err(_) => ChatResponder::InternalError(APIError::new("Channel closed".to_string())),
    }
}

#[utoipa::path(
    post,
    tag = "candle-vllm",
    path = "/v1/adapters",
    request_body = RegisterAdapterRequest,
    responses((status = 200, description = "Registered adapter"))
)]
pub async fn register_adapter(
    State(data): State<Arc<OpenAIServerData>>,
    request: Json<RegisterAdapterRequest>,
) -> Response {
    let mut request = request.0;
    if let Err(err) = ensure_local_adapter_scope(request.scope.as_deref()) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response();
    }
    request.scope = None;
    request.backend = None;
    match data.lora_manager.register(request) {
        Ok(record) => (StatusCode::OK, Json(record)).into_response(),
        Err(err) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

#[utoipa::path(
    get,
    tag = "candle-vllm",
    path = "/v1/adapters",
    responses((status = 200, description = "List adapters"))
)]
pub async fn list_adapters(State(data): State<Arc<OpenAIServerData>>) -> Response {
    (StatusCode::OK, Json(data.lora_manager.list())).into_response()
}

#[utoipa::path(
    get,
    tag = "candle-vllm",
    path = "/v1/adapters/{id}",
    params(("id" = String, Path, description = "Adapter id")),
    responses((status = 200, description = "Get adapter"))
)]
pub async fn get_adapter(
    State(data): State<Arc<OpenAIServerData>>,
    Path(id): Path<String>,
) -> Response {
    match data.lora_manager.get(&id) {
        Some(record) => (StatusCode::OK, Json(record)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("Adapter '{}' not found", id) })),
        )
            .into_response(),
    }
}

#[utoipa::path(
    post,
    tag = "candle-vllm",
    path = "/v1/adapters/{id}/load",
    request_body = LoadAdapterRequest,
    params(("id" = String, Path, description = "Adapter id")),
    responses((status = 200, description = "Load adapter"))
)]
pub async fn load_adapter(
    State(data): State<Arc<OpenAIServerData>>,
    Path(id): Path<String>,
    request: Option<Json<LoadAdapterRequest>>,
) -> Response {
    let request = request.map(|json| json.0).unwrap_or_default();
    if let Err(err) = ensure_local_adapter_scope(request.scope.as_deref()) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response();
    }

    let pin_override = request.pinned;
    let wait = request.wait.unwrap_or(false);

    if wait {
        return match data.lora_manager.load_async(&id, pin_override).await {
            Ok(record) => (StatusCode::OK, Json(record)).into_response(),
            Err(err) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response(),
        };
    }

    let (record, should_start) = match data.lora_manager.enqueue_load(&id, pin_override) {
        Ok(v) => v,
        Err(err) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response();
        }
    };

    if should_start {
        let manager = Arc::clone(&data.lora_manager);
        let id_for_task = id.clone();
        tokio::spawn(async move {
            if let Err(err) = manager.load_async(&id_for_task, None).await {
                tracing::warn!(
                    "Background LoRA load failed for adapter '{}': {}",
                    id_for_task,
                    err
                );
            }
        });
    }

    (StatusCode::ACCEPTED, Json(record)).into_response()
}

#[utoipa::path(
    post,
    tag = "candle-vllm",
    path = "/v1/adapters/{id}/unload",
    params(("id" = String, Path, description = "Adapter id")),
    responses((status = 200, description = "Unload adapter"))
)]
pub async fn unload_adapter(
    State(data): State<Arc<OpenAIServerData>>,
    Path(id): Path<String>,
    request: Option<Json<UnloadAdapterRequest>>,
) -> Response {
    let request = request.map(|json| json.0).unwrap_or_default();
    if let Err(err) = ensure_local_adapter_scope(request.scope.as_deref()) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response();
    }

    match data.lora_manager.unload(&id) {
        Ok(record) => (StatusCode::OK, Json(record)).into_response(),
        Err(err) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

#[utoipa::path(
    get,
    tag = "candle-vllm",
    path = "/v1/adapters/status",
    responses((status = 200, description = "Adapter status"))
)]
pub async fn adapters_status(State(data): State<Arc<OpenAIServerData>>) -> Response {
    (StatusCode::OK, Json(data.lora_manager.status())).into_response()
}

#[utoipa::path(
    post,
    tag = "candle-vllm",
    path = "/v1/adapters/warmup",
    request_body = WarmupAdaptersRequest,
    responses((status = 200, description = "Warmup adapters"))
)]
pub async fn warmup_adapters(
    State(data): State<Arc<OpenAIServerData>>,
    request: Json<WarmupAdaptersRequest>,
) -> Response {
    let mut queued = 0usize;
    let mut loaded = 0usize;
    let mut failed = Vec::new();

    for raw_id in &request.adapter_ids {
        let adapter_id = raw_id.trim();
        if adapter_id.is_empty() {
            continue;
        }

        if request.wait {
            match data.lora_manager.load_async(adapter_id, None).await {
                Ok(_) => loaded += 1,
                Err(err) => failed.push(format!("local/{adapter_id}: {err}")),
            }
        } else {
            match data.lora_manager.enqueue_load(adapter_id, None) {
                Ok((_, should_start)) => {
                    queued += 1;
                    if should_start {
                        let manager = Arc::clone(&data.lora_manager);
                        let id = adapter_id.to_string();
                        tokio::spawn(async move {
                            if let Err(err) = manager.load_async(&id, None).await {
                                tracing::warn!("Warmup load failed for adapter '{}': {}", id, err);
                            }
                        });
                    }
                }
                Err(err) => failed.push(format!("local/{adapter_id}: {err}")),
            }
        }
    }

    (
        StatusCode::OK,
        Json(WarmupAdaptersResponse {
            requested: request.adapter_ids.len(),
            queued,
            loaded,
            failed,
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_scope_accepts_local_or_empty() {
        assert!(ensure_local_adapter_scope(None).is_ok());
        assert!(ensure_local_adapter_scope(Some("local")).is_ok());
        assert!(ensure_local_adapter_scope(Some("")).is_ok());
    }

    #[test]
    fn adapter_scope_rejects_non_local_values() {
        let err = ensure_local_adapter_scope(Some("cloud")).expect_err("cloud scope should fail");
        assert!(err.to_string().contains("only supports local"));
    }

    #[test]
    fn adapter_schedule_is_sorted_and_trimmed() {
        let request = ChatCompletionRequest {
            metadata: Some(crate::openai::requests::ChatCompletionMetadata {
                runtime: Some(crate::openai::requests::RuntimeRequestMetadata {
                    adapter_id: Some("base".to_string()),
                    adapter_schedule: Some(vec![
                        crate::openai::requests::AdapterScheduleStep {
                            start_step: 10,
                            adapter_id: "  role_b ".to_string(),
                        },
                        crate::openai::requests::AdapterScheduleStep {
                            start_step: 2,
                            adapter_id: "role_a".to_string(),
                        },
                        crate::openai::requests::AdapterScheduleStep {
                            start_step: 12,
                            adapter_id: " ".to_string(),
                        },
                    ]),
                }),
            }),
            ..ChatCompletionRequest::default()
        };

        let schedule = parse_adapter_schedule(&request).expect("schedule should parse");
        assert_eq!(schedule.len(), 2);
        assert_eq!(schedule[0].start_step, 2);
        assert_eq!(schedule[0].adapter_id, "role_a");
        assert_eq!(schedule[1].start_step, 10);
        assert_eq!(schedule[1].adapter_id, "role_b");
    }
}
