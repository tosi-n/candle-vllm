use super::requests::Messages;
use super::requests::{ChatCompletionRequest, EmbeddingRequest, EmbeddingType, EncodingFormat};
use super::responses::{APIError, ChatCompletionResponse, ChatResponder};
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::streaming::{ChatResponse, Streamer, StreamingStatus};
use super::OpenAIServerData;
use crate::openai::backend_router::{
    BackendOperationResult, CloudAdapterGetResult, CloudAdapterListResult,
    CloudBackendStatusSnapshot, RoutingMode, SelectedBackend,
};
use crate::openai::lora::{
    AdapterRecord, AdapterStatusResponse, LoadAdapterRequest, RegisterAdapterRequest,
    UnloadAdapterRequest,
};
use crate::openai::{resolve_tools_for_request, ResolvedToolConfig};
use crate::tools::ToolFormat;
use axum::body::Body;
use axum::response::sse::KeepAlive;
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response, Sse},
    Json,
};
use bytes::Bytes;
use flume;
use futures::Stream;
use serde_json::json;
use std::env;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Instant, SystemTime};
use tokio::sync::Notify;
use tokio::time::Duration;
use tracing::debug;
use uuid::Uuid;

// Get prompt, roles
async fn get_gen_prompt(
    data: &OpenAIServerData,
    request: &ChatCompletionRequest,
    tool_config: &ResolvedToolConfig,
) -> Result<String, APIError> {
    let mut model = data.model.write();
    let pipeline = model
        .get_mut_pipeline(0)
        .ok_or(APIError::new("Missing pipeline".to_string()))?;
    let mut conversation = pipeline.0.get_conversation().clone();

    match &request.messages {
        Messages::Literal(msg) => {
            return Ok(msg.clone());
        }
        Messages::Chat(messages) => {
            for message in messages {
                let role = message.role.as_str();
                if role == "system" {
                    if let Some(content) = &message.content {
                        conversation.set_system_message(Some(content.clone()));
                    }
                    continue;
                }

                if role == "tool" {
                    let tool_call_id = message.tool_call_id.as_deref().unwrap_or("unknown");
                    let content = message.content.clone().unwrap_or_default();
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        let prompt = format!("[Tool Result for {}]: {}", tool_call_id, trimmed);
                        conversation.append_message(role.to_string(), prompt);
                    }
                    continue;
                }

                if let Some(content) = &message.content {
                    conversation.append_message(role.to_string(), content.clone());
                }
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
                    conversation.append_message(role.to_string(), content)
                }
            }
        }
    }

    if !tool_config.tools.is_empty() {
        let mut tools_prompt = ToolFormat::get_tool_prompt(&pipeline.0.tool_config);

        // Enforce tool_choice=function by prepending a mandatory instruction
        if let crate::openai::ToolChoiceKind::Function(name) = &tool_config.choice {
            tools_prompt = format!(
                "IMPORTANT: You MUST call the tool \"{}\". Do not respond with plain text.\n\n{}",
                name, tools_prompt
            );
        }

        let current_system = conversation.get_system_message().unwrap_or_default();
        let new_system = if current_system.is_empty() {
            tools_prompt
        } else {
            format!("{}\n\n{}", current_system, tools_prompt)
        };
        conversation.set_system_message(Some(new_system));
    }

    Ok(conversation.get_prompt(request.thinking.unwrap_or(false), &tool_config.tools))
}

async fn check_length(
    request: &ChatCompletionRequest,
    prompt: String,
    data: &OpenAIServerData,
) -> Result<(Vec<u32>, usize), APIError> {
    let (token_ids, available_kv_tokens) = {
        let model = data.model.read();
        let available_kv_tokens = model.get_available_kv_tokens();
        let pipeline = model
            .get_pipeline(0)
            .ok_or(APIError::new("Missing pipeline".to_string()))?;
        (
            pipeline
                .0
                .tokenizer()
                .encode_fast(prompt, false)
                .map_err(APIError::from)?
                .get_ids()
                .to_vec(),
            available_kv_tokens,
        )
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
    } else if token_ids.len() >= available_kv_tokens {
        Err(APIError::new(format!(
            "Requested prompt({} tokens) is  \
            larger than available kvcache (maximum {} tokens).\n \
            You can increase kvcache by setting `--mem` to a larger value!",
            token_ids.len(),
            available_kv_tokens
        )))
    } else {
        let max_valid_request_tokens =
            std::cmp::min(available_kv_tokens, data.pipeline_config.max_model_len) - 10;
        Ok((token_ids, max_valid_request_tokens))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExecutionMode {
    Cloud,
    Local,
    Hybrid,
}

impl ExecutionMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cloud => "cloud",
            Self::Local => "local",
            Self::Hybrid => "hybrid",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "cloud" => Some(Self::Cloud),
            "local" => Some(Self::Local),
            "hybrid" => Some(Self::Hybrid),
            _ => None,
        }
    }
}

fn parse_execution_mode(
    headers: &HeaderMap,
    model: Option<&str>,
    runtime_local_only_strict: bool,
) -> Result<ExecutionMode, APIError> {
    if let Some(mode) = headers
        .get("x-hybrie-execution-mode")
        .and_then(|value| value.to_str().ok())
    {
        let selected = ExecutionMode::parse(mode).ok_or_else(|| {
            APIError::new(format!(
                "Invalid x-hybrie-execution-mode '{}'. Expected cloud|local|hybrid.",
                mode
            ))
        })?;
        if runtime_local_only_strict && selected != ExecutionMode::Local {
            return Err(APIError::new(
                "Runtime is configured as local-only strict. x-hybrie-execution-mode must be 'local' or omitted.".to_string(),
            ));
        }
        return Ok(selected);
    }

    if let Some(model) = model {
        if model.starts_with("cloud/") {
            if runtime_local_only_strict {
                return Err(APIError::new(
                    "Runtime is configured as local-only strict. Model prefix 'cloud/' is not allowed."
                        .to_string(),
                ));
            }
            return Ok(ExecutionMode::Cloud);
        }
        if model.starts_with("local/") {
            return Ok(ExecutionMode::Local);
        }
        if model.starts_with("hybrid/") {
            if runtime_local_only_strict {
                return Err(APIError::new(
                    "Runtime is configured as local-only strict. Model prefix 'hybrid/' is not allowed."
                        .to_string(),
                ));
            }
            return Ok(ExecutionMode::Hybrid);
        }
    }

    Ok(ExecutionMode::Local)
}

fn parse_adapter_id(headers: &HeaderMap, request: &ChatCompletionRequest) -> Option<String> {
    if let Some(adapter_id) = headers
        .get("x-hybrie-adapter-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(adapter_id.to_string());
    }

    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.hybrie.as_ref())
        .and_then(|hybrie| hybrie.adapter_id.clone())
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
}

fn parse_adapter_timeline(
    request: &ChatCompletionRequest,
) -> Option<Vec<crate::openai::requests::HybrieAdapterStep>> {
    let mut timeline = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.hybrie.as_ref())
        .and_then(|hybrie| hybrie.adapter_timeline.clone())?;

    timeline.retain(|step| !step.adapter_id.trim().is_empty());
    if timeline.is_empty() {
        return None;
    }

    timeline.sort_by_key(|step| step.start_step);
    for step in &mut timeline {
        step.adapter_id = step.adapter_id.trim().to_string();
    }
    Some(timeline)
}

fn parse_session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-hybrie-session-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
}

fn strip_mode_prefix(model: Option<String>) -> Option<String> {
    model.map(|model_id| {
        for prefix in ["local/", "cloud/", "hybrid/"] {
            if let Some(stripped) = model_id.strip_prefix(prefix) {
                return stripped.to_string();
            }
        }
        model_id
    })
}

fn execution_mode_to_routing_mode(mode: ExecutionMode) -> RoutingMode {
    match mode {
        ExecutionMode::Local => RoutingMode::LocalOnly,
        ExecutionMode::Cloud => RoutingMode::CloudOnly,
        ExecutionMode::Hybrid => RoutingMode::Hybrid,
    }
}

fn with_hybrie_headers(
    mut response: Response,
    mode: ExecutionMode,
    backend: &str,
    adapter_id: Option<&str>,
) -> Response {
    let selected_mode =
        HeaderValue::from_str(mode.as_str()).unwrap_or_else(|_| HeaderValue::from_static("local"));
    response
        .headers_mut()
        .insert("x-hybrie-selected-mode", selected_mode);

    let selected_backend =
        HeaderValue::from_str(backend).unwrap_or_else(|_| HeaderValue::from_static("local"));
    response
        .headers_mut()
        .insert("x-hybrie-selected-backend", selected_backend);

    let adapter_header = HeaderValue::from_str(adapter_id.unwrap_or("none"))
        .unwrap_or_else(|_| HeaderValue::from_static("none"));
    response
        .headers_mut()
        .insert("x-hybrie-adapter-id", adapter_header);

    response
}

struct MeteredLocalStreamer {
    inner: Streamer,
    router: Arc<crate::openai::backend_router::BackendRouter>,
    started: Instant,
    done: bool,
}

impl MeteredLocalStreamer {
    fn new(
        inner: Streamer,
        router: Arc<crate::openai::backend_router::BackendRouter>,
        started: Instant,
    ) -> Self {
        Self {
            inner,
            router,
            started,
            done: false,
        }
    }

    fn finish_once(&mut self, success: bool) {
        if self.done {
            return;
        }
        self.done = true;
        self.router.end_local_request(self.started, success);
    }
}

impl Stream for MeteredLocalStreamer {
    type Item = Result<axum::response::sse::Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(None) => {
                self.finish_once(true);
                Poll::Ready(None)
            }
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for MeteredLocalStreamer {
    fn drop(&mut self) {
        self.finish_once(true);
    }
}

struct MeteredCloudByteStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    router: Arc<crate::openai::backend_router::BackendRouter>,
    backend_id: String,
    started: Instant,
    done: bool,
}

impl MeteredCloudByteStream {
    fn new(
        inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
        router: Arc<crate::openai::backend_router::BackendRouter>,
        backend_id: String,
        started: Instant,
    ) -> Self {
        Self {
            inner,
            router,
            backend_id,
            started,
            done: false,
        }
    }

    fn finish_once(&mut self, success: bool) {
        if self.done {
            return;
        }
        self.done = true;
        self.router
            .end_cloud_request(&self.backend_id, self.started, success);
    }
}

impl Stream for MeteredCloudByteStream {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => Poll::Ready(Some(Ok(chunk))),
            Poll::Ready(Some(Err(err))) => {
                self.finish_once(false);
                Poll::Ready(Some(Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    err.to_string(),
                ))))
            }
            Poll::Ready(None) => {
                self.finish_once(true);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for MeteredCloudByteStream {
    fn drop(&mut self) {
        self.finish_once(true);
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct WarmupAdaptersRequest {
    pub adapter_ids: Vec<String>,
    #[serde(default)]
    pub wait: bool,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WarmupAdaptersResponse {
    pub requested: usize,
    pub queued: usize,
    pub loaded: usize,
    pub failed: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct AdapterQuery {
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterScope {
    Local,
    Cloud,
    All,
}

fn parse_adapter_scope(
    raw: Option<&str>,
    runtime_local_only_strict: bool,
) -> Result<AdapterScope, APIError> {
    let Some(raw) = raw else {
        return Ok(AdapterScope::Local);
    };
    let scope = match raw.trim().to_lowercase().as_str() {
        "local" => Ok(AdapterScope::Local),
        "cloud" => Ok(AdapterScope::Cloud),
        "all" => Ok(AdapterScope::All),
        other => Err(APIError::new(format!(
            "Invalid adapter scope '{}'. Expected local|cloud|all.",
            other
        ))),
    }?;
    if runtime_local_only_strict && scope != AdapterScope::Local {
        return Err(APIError::new(
            "Runtime is configured as local-only strict. Adapter scope must be 'local' or omitted."
                .to_string(),
        ));
    }
    Ok(scope)
}

#[derive(Debug, Clone, serde::Serialize)]
struct AdapterStatusEnvelope {
    local: AdapterStatusResponse,
    cloud: Vec<CloudBackendStatusSnapshot>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AdapterOperationEnvelope {
    adapter_id: String,
    scope: String,
    local: Option<AdapterRecord>,
    cloud: Vec<BackendOperationResult>,
    errors: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AdapterListEnvelope {
    scope: String,
    local: Option<Vec<AdapterRecord>>,
    cloud: Vec<CloudAdapterListResult>,
    errors: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AdapterLookupEnvelope {
    adapter_id: String,
    scope: String,
    local: Option<AdapterRecord>,
    cloud: Vec<CloudAdapterGetResult>,
    errors: Vec<String>,
}

async fn proxy_chat_completion_to_cloud(
    data: Arc<OpenAIServerData>,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
    selected_mode: ExecutionMode,
    backend_id: &str,
    adapter_id: Option<&str>,
) -> ChatResponder {
    let backend = match data.backend_router.get_cloud_backend(backend_id) {
        Some(backend) => backend,
        None => {
            return ChatResponder::ValidationError(APIError::new(format!(
                "Unknown cloud backend '{}'",
                backend_id
            )));
        }
    };

    let mut outbound_request = request.clone();
    outbound_request.model = strip_mode_prefix(outbound_request.model.clone());

    let url = format!("{}/v1/chat/completions", backend.base_url);
    let started = Instant::now();
    data.backend_router.begin_cloud_request(&backend.id);

    let mut req_builder = data
        .backend_router
        .client()
        .post(url)
        .header("x-hybrie-execution-mode", "local")
        .json(&outbound_request);
    if let Some(adapter_id) = adapter_id {
        req_builder = req_builder.header("x-hybrie-adapter-id", adapter_id);
    }
    if let Some(auth) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    {
        req_builder = req_builder.header(axum::http::header::AUTHORIZATION.as_str(), auth);
    }
    if let Some(session_id) = headers
        .get("x-hybrie-session-id")
        .and_then(|value| value.to_str().ok())
    {
        req_builder = req_builder.header("x-hybrie-session-id", session_id);
    }

    let response = match req_builder.send().await {
        Ok(response) => response,
        Err(err) => {
            data.backend_router
                .end_cloud_request(&backend.id, started, false);
            return ChatResponder::ModelError(APIError::new(format!(
                "Cloud backend '{}' request failed: {}",
                backend.id, err
            )));
        }
    };

    let status =
        StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response_builder = Response::builder().status(status);
    if let Some(content_type) = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    {
        response_builder = response_builder.header(axum::http::header::CONTENT_TYPE, content_type);
    }
    if let Some(cache_control) = response
        .headers()
        .get(axum::http::header::CACHE_CONTROL)
        .and_then(|value| value.to_str().ok())
    {
        response_builder =
            response_builder.header(axum::http::header::CACHE_CONTROL, cache_control);
    }

    let is_stream = request.stream.unwrap_or(false);
    if is_stream {
        let stream = MeteredCloudByteStream::new(
            Box::pin(response.bytes_stream()),
            Arc::clone(&data.backend_router),
            backend.id.clone(),
            started,
        );
        let raw = match response_builder.body(Body::from_stream(stream)) {
            Ok(raw) => raw,
            Err(err) => {
                data.backend_router
                    .end_cloud_request(&backend.id, started, false);
                return ChatResponder::InternalError(APIError::new(format!(
                    "Failed to build cloud stream response: {}",
                    err
                )));
            }
        };
        return ChatResponder::Raw(with_hybrie_headers(
            raw,
            selected_mode,
            backend.id.as_str(),
            adapter_id,
        ));
    }

    let is_success = status.is_success();
    let body = match response.bytes().await {
        Ok(body) => body,
        Err(err) => {
            data.backend_router
                .end_cloud_request(&backend.id, started, false);
            return ChatResponder::InternalError(APIError::new(format!(
                "Failed to read cloud response body: {}",
                err
            )));
        }
    };
    data.backend_router
        .end_cloud_request(&backend.id, started, is_success);

    let raw = match response_builder.body(Body::from(body)) {
        Ok(raw) => raw,
        Err(err) => {
            return ChatResponder::InternalError(APIError::new(format!(
                "Failed to build cloud response: {}",
                err
            )));
        }
    };

    ChatResponder::Raw(with_hybrie_headers(
        raw,
        selected_mode,
        backend.id.as_str(),
        adapter_id,
    ))
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

    let selected_mode = match parse_execution_mode(
        &headers,
        request.model.as_deref(),
        data.runtime_local_only_strict,
    ) {
        Ok(mode) => mode,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let session_id = parse_session_id(&headers);
    let mut adapter_id = parse_adapter_id(&headers, &request);
    let adapter_timeline = parse_adapter_timeline(&request);
    if adapter_id.is_none() {
        if let Some(session_id) = session_id.as_ref() {
            adapter_id = data.sticky_adapters.read().get(session_id).cloned();
        }
    }
    if let (Some(session_id), Some(adapter_id)) = (session_id.as_ref(), adapter_id.as_ref()) {
        data.sticky_adapters
            .write()
            .insert(session_id.clone(), adapter_id.clone());
    }

    let routing_mode = execution_mode_to_routing_mode(selected_mode);
    let local_status = data.lora_manager.status();
    let selected_backend = match data
        .backend_router
        .select_backend(routing_mode, adapter_id.as_deref(), &local_status)
        .await
    {
        Ok(backend) => backend,
        Err(err) => return ChatResponder::ValidationError(err),
    };

    if selected_mode == ExecutionMode::Hybrid {
        if let Some(adapter) = adapter_id.as_deref() {
            let router = Arc::clone(&data.backend_router);
            let adapter_id = adapter.to_string();
            let selected_cloud = selected_backend.cloud_id().map(str::to_string);
            tokio::spawn(async move {
                router
                    .maybe_autosync_cloud_adapter(&adapter_id, selected_cloud.as_deref())
                    .await;
            });
        }
    }

    if let Some(adapter) = adapter_id.as_deref() {
        match &selected_backend {
            SelectedBackend::Local => {
                if let Err(err) = data.lora_manager.ensure_loaded_async(adapter).await {
                    return ChatResponder::ValidationError(APIError::new(format!(
                        "Failed to prepare adapter '{}': {}",
                        adapter, err
                    )));
                }
            }
            SelectedBackend::Cloud(_) => {
                if selected_mode == ExecutionMode::Hybrid {
                    if let Ok((_, should_start)) = data.lora_manager.enqueue_load(adapter, None) {
                        if should_start {
                            let manager = Arc::clone(&data.lora_manager);
                            let adapter_id = adapter.to_string();
                            tokio::spawn(async move {
                                if let Err(err) = manager.load_async(&adapter_id, None).await {
                                    tracing::warn!(
                                        "Background local load failed for adapter '{}': {}",
                                        adapter_id,
                                        err
                                    );
                                }
                            });
                        }
                    }
                }
            }
        }
    }

    if let SelectedBackend::Cloud(backend_id) = &selected_backend {
        return proxy_chat_completion_to_cloud(
            data,
            &headers,
            &request,
            selected_mode,
            backend_id,
            adapter_id.as_deref(),
        )
        .await;
    }

    let tool_config = match resolve_tools_for_request(
        &request.tools,
        &request.tool_choice,
        data.mcp_manager.as_ref(),
    ) {
        Ok(config) => config,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let prompt = match get_gen_prompt(&data, &request, &tool_config).await {
        Ok(p) => p,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let (token_ids, available_tokens): (Vec<u32>, usize) =
        match check_length(&request, prompt.clone(), &data).await {
            Ok(ids) => ids,
            Err(e) => return ChatResponder::ValidationError(e),
        };

    debug!("\n\n\nPrompt {:?}", prompt);

    let request_id = format!("cmpl-{}", Uuid::new_v4());

    let mut max_request_tokens = request
        .max_tokens
        .unwrap_or(data.pipeline_config.default_max_tokens);

    if max_request_tokens + token_ids.len() > available_tokens {
        tracing::warn!(
            "Requested max tokens + prompt length {} larger than available tokens {}, \
        max_tokens changed to {} ({} tokens reserved for prompt)!",
            max_request_tokens + token_ids.len(),
            available_tokens,
            available_tokens - token_ids.len(),
            token_ids.len()
        );
        max_request_tokens = if available_tokens > token_ids.len() {
            available_tokens - token_ids.len()
        } else {
            return ChatResponder::ValidationError(APIError::new(format!(
                "Requested prompt({} tokens) is  \
                larger than available kvcache (maximum {} tokens).\n \
                You can increase kvcache by setting `--mem` to a larger value!",
                token_ids.len(),
                available_tokens
            )));
        }
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

    let (response_tx, rx) = flume::unbounded();
    tracing::info!("{:?}", sampling_params);

    let data_clone = data.clone();
    let request_id_clone = request_id.clone();
    let stream_request = request.stream.is_some_and(|x| x);
    let model_name = strip_mode_prefix(request.model.clone()).unwrap_or("default".to_string());
    let request_logprobs = request.logprobs.unwrap_or(false);
    let sync_notify = Arc::new(Notify::new());
    let sync_completion_notify = if stream_request {
        None
    } else {
        Some(Arc::clone(&sync_notify))
    };
    let data_for_engine = data.clone();
    let adapter_id_for_engine = adapter_id.clone();
    let adapter_timeline_for_engine = adapter_timeline.clone();
    let local_started = Instant::now();
    data.backend_router.begin_local_request();

    let _ = tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(async move {
            {
                //send completion request to inference engine
                let mut model = data_for_engine.model.write();
                model.add_request(
                    token_ids,
                    request_id.clone(),
                    SystemTime::now(),
                    sampling_params,
                    request_logprobs,
                    false,
                    EncodingFormat::default(),
                    EmbeddingType::default(),
                    adapter_id_for_engine,
                    adapter_timeline_for_engine,
                    if stream_request {
                        Some(Arc::new(response_tx))
                    } else {
                        None
                    },
                    sync_completion_notify,
                );
                model.notify.notify_one();
            }
        });
    });

    if stream_request {
        let metered_streamer = MeteredLocalStreamer::new(
            Streamer {
                rx,
                status: StreamingStatus::Uninitialized,
            },
            Arc::clone(&data.backend_router),
            local_started,
        );
        let response = Sse::new(metered_streamer)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_millis(
                        env::var("KEEP_ALIVE_INTERVAL")
                            .map(|val| val.parse::<u64>().unwrap_or(100))
                            .unwrap_or(100),
                    ))
                    .text("keep-alive-text"),
            )
            .into_response();
        ChatResponder::Raw(with_hybrie_headers(
            response,
            selected_mode,
            "local",
            adapter_id.as_deref(),
        ))
    } else {
        // wait until current response finished
        tracing::warn!("waiting response for sync request {}", request_id_clone);
        sync_notify.as_ref().notified().await;
        // Re-acquire read lock to get the response
        // Note: we need to drop the lock later
        let (choices, usage) = {
            let model = data_clone.model.read();
            if !model.completion_records.contains_key(&request_id_clone) {
                data.backend_router.end_local_request(local_started, false);
                return ChatResponder::ModelError(APIError::from(format!(
                    "Unable to generate response for request {request_id_clone}"
                )));
            }
            let record = &model.completion_records[&request_id_clone];
            (record.0.clone(), record.1.clone())
        };

        // Check for tool calls in the output
        let mut final_choices = choices.clone();
        if has_tools {
            let parser = crate::tools::parser::ToolParser::new();
            for choice in &mut final_choices {
                if choice.message.tool_calls.is_some() {
                    continue;
                }
                if let Some(content) = &choice.message.content {
                    let calls = parser.parse(content);
                    if !calls.is_empty() {
                        choice.message.tool_calls = Some(calls);
                        choice.message.content = None;
                        choice.finish_reason = Some("tool_calls".to_string());
                    }
                }
            }
        }

        let response = Json(ChatCompletionResponse {
            id: request_id_clone,
            choices: final_choices,
            created: usage.created,
            model: model_name,
            object: "chat.completion",
            usage: usage,
        })
        .into_response();
        data.backend_router.end_local_request(local_started, true);
        ChatResponder::Raw(with_hybrie_headers(
            response,
            selected_mode,
            "local",
            adapter_id.as_deref(),
        ))
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
    let (token_ids, available_tokens) = {
        let model = data.model.read();
        let available_kv_tokens = model.get_available_kv_tokens();
        let pipeline = model
            .get_pipeline(0)
            .ok_or(APIError::new("Missing pipeline".to_string()));

        match pipeline {
            Ok(pipeline) => match pipeline.0.tokenizer().encode_fast(prompt_str, false) {
                Ok(encoding) => (encoding.get_ids().to_vec(), available_kv_tokens),
                Err(e) => return ChatResponder::ValidationError(APIError::from(e)),
            },
            Err(e) => return ChatResponder::ModelError(e),
        }
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
                    None,
                    None,
                    Some(Arc::new(response_tx)),
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
    let request = request.0;
    let scope = match parse_adapter_scope(request.scope.as_deref(), data.runtime_local_only_strict)
    {
        Ok(scope) => scope,
        Err(err) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response()
        }
    };

    if scope == AdapterScope::Local {
        let mut local_request = request.clone();
        local_request.scope = None;
        local_request.backend = None;
        return match data.lora_manager.register(local_request) {
            Ok(record) => (StatusCode::OK, Json(record)).into_response(),
            Err(err) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response(),
        };
    }

    let mut local = None;
    let mut errors = Vec::new();

    if scope == AdapterScope::All {
        let mut local_request = request.clone();
        local_request.scope = None;
        local_request.backend = None;
        match data.lora_manager.register(local_request) {
            Ok(record) => local = Some(record),
            Err(err) => errors.push(format!("local: {}", err)),
        }
    }

    let cloud = data
        .backend_router
        .register_adapter_on_cloud(&request, request.backend.as_deref())
        .await;
    for result in &cloud {
        if !result.ok {
            errors.push(format!("{}: {}", result.backend_id, result.message));
        }
    }

    let status = if errors.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::UNPROCESSABLE_ENTITY
    };
    (
        status,
        Json(AdapterOperationEnvelope {
            adapter_id: request.id,
            scope: match scope {
                AdapterScope::Local => "local".to_string(),
                AdapterScope::Cloud => "cloud".to_string(),
                AdapterScope::All => "all".to_string(),
            },
            local,
            cloud,
            errors,
        }),
    )
        .into_response()
}

#[utoipa::path(
    get,
    tag = "candle-vllm",
    path = "/v1/adapters",
    responses((status = 200, description = "List adapters"))
)]
pub async fn list_adapters(
    State(data): State<Arc<OpenAIServerData>>,
    Query(query): Query<AdapterQuery>,
) -> Response {
    let scope = match parse_adapter_scope(query.scope.as_deref(), data.runtime_local_only_strict) {
        Ok(scope) => scope,
        Err(err) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response()
        }
    };

    if scope == AdapterScope::Local {
        return (StatusCode::OK, Json(data.lora_manager.list())).into_response();
    }

    let local = if scope == AdapterScope::All {
        Some(data.lora_manager.list())
    } else {
        None
    };
    let cloud = data
        .backend_router
        .list_adapters_on_cloud(query.backend.as_deref())
        .await;
    let errors = cloud
        .iter()
        .filter_map(|result| {
            result
                .error
                .as_ref()
                .map(|err| format!("{}: {}", result.backend_id, err))
        })
        .collect::<Vec<_>>();
    let status = if errors.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::PARTIAL_CONTENT
    };

    (
        status,
        Json(AdapterListEnvelope {
            scope: match scope {
                AdapterScope::Local => "local".to_string(),
                AdapterScope::Cloud => "cloud".to_string(),
                AdapterScope::All => "all".to_string(),
            },
            local,
            cloud,
            errors,
        }),
    )
        .into_response()
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
    Query(query): Query<AdapterQuery>,
) -> Response {
    let scope = match parse_adapter_scope(query.scope.as_deref(), data.runtime_local_only_strict) {
        Ok(scope) => scope,
        Err(err) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response()
        }
    };

    if scope == AdapterScope::Local {
        return match data.lora_manager.get(&id) {
            Some(record) => (StatusCode::OK, Json(record)).into_response(),
            None => (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("Adapter '{}' not found", id) })),
            )
                .into_response(),
        };
    }

    let local = if scope == AdapterScope::All {
        data.lora_manager.get(&id)
    } else {
        None
    };
    let cloud = data
        .backend_router
        .get_adapter_on_cloud(&id, query.backend.as_deref())
        .await;
    let errors = cloud
        .iter()
        .filter_map(|result| {
            result
                .error
                .as_ref()
                .map(|err| format!("{}: {}", result.backend_id, err))
        })
        .collect::<Vec<_>>();
    let any_found = local.is_some() || cloud.iter().any(|result| result.adapter.is_some());
    let status = if any_found {
        if errors.is_empty() {
            StatusCode::OK
        } else {
            StatusCode::PARTIAL_CONTENT
        }
    } else {
        StatusCode::NOT_FOUND
    };

    (
        status,
        Json(AdapterLookupEnvelope {
            adapter_id: id,
            scope: match scope {
                AdapterScope::Local => "local".to_string(),
                AdapterScope::Cloud => "cloud".to_string(),
                AdapterScope::All => "all".to_string(),
            },
            local,
            cloud,
            errors,
        }),
    )
        .into_response()
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
    let scope = match parse_adapter_scope(request.scope.as_deref(), data.runtime_local_only_strict)
    {
        Ok(scope) => scope,
        Err(err) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response()
        }
    };

    let pin_override = request.pinned;
    let wait = request.wait.unwrap_or(false);

    if scope == AdapterScope::Local {
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
        return (StatusCode::ACCEPTED, Json(record)).into_response();
    }

    let mut local = None;
    let mut errors = Vec::new();

    if scope == AdapterScope::All {
        if wait {
            match data.lora_manager.load_async(&id, pin_override).await {
                Ok(record) => local = Some(record),
                Err(err) => errors.push(format!("local: {}", err)),
            }
        } else {
            match data.lora_manager.enqueue_load(&id, pin_override) {
                Ok((record, should_start)) => {
                    local = Some(record);
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
                }
                Err(err) => errors.push(format!("local: {}", err)),
            }
        }
    }

    let cloud = data
        .backend_router
        .load_adapter_on_cloud(&id, pin_override, wait, request.backend.as_deref())
        .await;
    for result in &cloud {
        if !result.ok {
            errors.push(format!("{}: {}", result.backend_id, result.message));
        }
    }

    let status = if errors.is_empty() {
        StatusCode::ACCEPTED
    } else {
        StatusCode::UNPROCESSABLE_ENTITY
    };
    (
        status,
        Json(AdapterOperationEnvelope {
            adapter_id: id,
            scope: match scope {
                AdapterScope::Local => "local".to_string(),
                AdapterScope::Cloud => "cloud".to_string(),
                AdapterScope::All => "all".to_string(),
            },
            local,
            cloud,
            errors,
        }),
    )
        .into_response()
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
    let scope = match parse_adapter_scope(request.scope.as_deref(), data.runtime_local_only_strict)
    {
        Ok(scope) => scope,
        Err(err) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response()
        }
    };

    if scope == AdapterScope::Local {
        return match data.lora_manager.unload(&id) {
            Ok(record) => (StatusCode::OK, Json(record)).into_response(),
            Err(err) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response(),
        };
    }

    let local = if scope == AdapterScope::All {
        match data.lora_manager.unload(&id) {
            Ok(record) => Some(record),
            Err(err) => {
                tracing::warn!("Local adapter unload failed for '{}': {}", id, err);
                None
            }
        }
    } else {
        None
    };

    let cloud = data
        .backend_router
        .unload_adapter_on_cloud(&id, request.backend.as_deref())
        .await;
    let errors = cloud
        .iter()
        .filter(|result| !result.ok)
        .map(|result| format!("{}: {}", result.backend_id, result.message))
        .collect::<Vec<_>>();

    let status = if errors.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::UNPROCESSABLE_ENTITY
    };
    (
        status,
        Json(AdapterOperationEnvelope {
            adapter_id: id,
            scope: match scope {
                AdapterScope::Local => "local".to_string(),
                AdapterScope::Cloud => "cloud".to_string(),
                AdapterScope::All => "all".to_string(),
            },
            local,
            cloud,
            errors,
        }),
    )
        .into_response()
}

#[utoipa::path(
    get,
    tag = "candle-vllm",
    path = "/v1/adapters/status",
    responses((status = 200, description = "Adapter status"))
)]
pub async fn adapters_status(State(data): State<Arc<OpenAIServerData>>) -> Response {
    let local = data.lora_manager.status();
    let cloud = if data.runtime_local_only_strict {
        Vec::new()
    } else {
        data.backend_router.cloud_status_snapshots().await
    };
    (StatusCode::OK, Json(AdapterStatusEnvelope { local, cloud })).into_response()
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
    let scope = match parse_adapter_scope(request.scope.as_deref(), data.runtime_local_only_strict)
    {
        Ok(scope) => scope,
        Err(err) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response()
        }
    };

    let mut queued = 0usize;
    let mut loaded = 0usize;
    let mut failed = Vec::new();

    for raw_id in &request.adapter_ids {
        let adapter_id = raw_id.trim();
        if adapter_id.is_empty() {
            continue;
        }

        if scope == AdapterScope::Local || scope == AdapterScope::All {
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
                                    tracing::warn!(
                                        "Warmup load failed for adapter '{}': {}",
                                        id,
                                        err
                                    );
                                }
                            });
                        }
                    }
                    Err(err) => failed.push(format!("local/{adapter_id}: {err}")),
                }
            }
        }

        if scope == AdapterScope::Cloud || scope == AdapterScope::All {
            let outcomes = data
                .backend_router
                .load_adapter_on_cloud(adapter_id, None, request.wait, request.backend.as_deref())
                .await;
            for result in outcomes {
                if result.ok {
                    if request.wait {
                        loaded += 1;
                    } else {
                        queued += 1;
                    }
                } else {
                    failed.push(format!(
                        "cloud/{}/{}: {}",
                        result.backend_id, adapter_id, result.message
                    ));
                }
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
    fn execution_mode_rejects_cloud_header_in_strict_runtime() {
        let mut headers = HeaderMap::new();
        headers.insert("x-hybrie-execution-mode", HeaderValue::from_static("cloud"));
        let err = parse_execution_mode(&headers, None, true).expect_err("cloud should be rejected");
        assert!(err.to_string().contains("local-only strict"));
    }

    #[test]
    fn execution_mode_accepts_cloud_header_when_not_strict() {
        let mut headers = HeaderMap::new();
        headers.insert("x-hybrie-execution-mode", HeaderValue::from_static("cloud"));
        let mode = parse_execution_mode(&headers, None, false).expect("cloud should be accepted");
        assert_eq!(mode, ExecutionMode::Cloud);
    }

    #[test]
    fn execution_mode_rejects_hybrid_prefix_in_strict_runtime() {
        let headers = HeaderMap::new();
        let err = parse_execution_mode(&headers, Some("hybrid/qwen"), true)
            .expect_err("hybrid prefix should be rejected");
        assert!(err.to_string().contains("local-only strict"));
    }

    #[test]
    fn adapter_scope_rejects_cloud_in_strict_runtime() {
        let err =
            parse_adapter_scope(Some("cloud"), true).expect_err("cloud scope should be rejected");
        assert!(err.to_string().contains("local-only strict"));
    }

    #[test]
    fn adapter_scope_defaults_to_local() {
        let scope = parse_adapter_scope(None, true).expect("default scope should parse");
        assert_eq!(scope, AdapterScope::Local);
    }

    #[test]
    fn adapter_timeline_is_sorted_and_trimmed() {
        let request = ChatCompletionRequest {
            metadata: Some(crate::openai::requests::ChatCompletionMetadata {
                hybrie: Some(crate::openai::requests::HybrieMetadata {
                    adapter_id: Some("base".to_string()),
                    adapter_timeline: Some(vec![
                        crate::openai::requests::HybrieAdapterStep {
                            start_step: 10,
                            adapter_id: "  role_b ".to_string(),
                        },
                        crate::openai::requests::HybrieAdapterStep {
                            start_step: 2,
                            adapter_id: "role_a".to_string(),
                        },
                        crate::openai::requests::HybrieAdapterStep {
                            start_step: 12,
                            adapter_id: " ".to_string(),
                        },
                    ]),
                }),
            }),
            ..ChatCompletionRequest::default()
        };

        let timeline = parse_adapter_timeline(&request).expect("timeline should parse");
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline[0].start_step, 2);
        assert_eq!(timeline[0].adapter_id, "role_a");
        assert_eq!(timeline[1].start_step, 10);
        assert_eq!(timeline[1].adapter_id, "role_b");
    }
}
