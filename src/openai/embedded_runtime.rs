use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use candle_core::{DType, Device, Result as CandleResult};
use flume::Receiver;
use parking_lot::RwLock;
use tokio::sync::Notify;
use uuid::Uuid;

use crate::openai::backend_router::BackendRouter;
use crate::openai::lora::{
    set_global_lora_manager, AdapterRecord, AdapterStatusResponse, LoRAManager,
    RegisterAdapterRequest,
};
use crate::openai::pipelines::llm_engine::{InteractiveSessionSnapshot, LLMEngine};
use crate::openai::pipelines::pipeline::DefaultLoader;
use crate::openai::requests::{ChatCompletionRequest, EmbeddingType, EncodingFormat, Messages};
use crate::openai::responses::{APIError, ChatCompletionResponse};
use crate::openai::runtime_internal::{RuntimeInternalService, RuntimeKvProfile};
use crate::openai::sampling_params::{EarlyStoppingCondition, GenerationConfig, SamplingParams};
use crate::openai::streaming::ChatResponse;
use crate::openai::{resolve_tools_for_request, OpenAIServerData, PipelineConfig, ToolChoiceKind};
use crate::scheduler::cache_engine::CacheEngine;
use crate::scheduler::prefix_cache::PrefixCacheConfig;
use crate::scheduler::sequence::{InteractiveSessionControl, InteractiveSessionEvent};
use crate::scheduler::SchedulerConfig;
use crate::tools::ToolFormat;

#[derive(Debug, Clone)]
pub struct EmbeddedRuntimeConfig {
    pub model_id: Option<String>,
    pub weight_path: Option<String>,
    pub weight_file: Option<String>,
    pub hf_token: Option<String>,
    pub hf_token_path: Option<String>,
    pub dtype: Option<String>,
    pub isq: Option<String>,
    pub device_ids: Vec<usize>,
    pub block_size: usize,
    pub max_num_seqs: usize,
    pub kvcache_mem_gpu: usize,
    pub kvcache_mem_cpu: usize,
    pub holding_time: usize,
    pub record_conversation: bool,
    pub runtime_local_only_strict: bool,
    pub runtime_canonical_model_id: Option<String>,
    pub fp8_kvcache: bool,
    pub prefix_cache: bool,
    pub prefix_cache_max_tokens: Option<usize>,
    pub prefill_chunk_size: Option<usize>,
    pub generation_cfg: Option<GenerationConfig>,
    pub max_active_loras: usize,
    pub max_lora_rank: usize,
    pub lora_mode: String,
}

impl Default for EmbeddedRuntimeConfig {
    fn default() -> Self {
        Self {
            model_id: None,
            weight_path: None,
            weight_file: None,
            hf_token: None,
            hf_token_path: None,
            dtype: Some("bf16".to_string()),
            isq: None,
            device_ids: vec![0],
            block_size: 64,
            max_num_seqs: 16,
            kvcache_mem_gpu: 4096,
            kvcache_mem_cpu: 128,
            holding_time: 500,
            record_conversation: false,
            runtime_local_only_strict: true,
            runtime_canonical_model_id: None,
            fp8_kvcache: false,
            prefix_cache: false,
            prefix_cache_max_tokens: None,
            prefill_chunk_size: None,
            generation_cfg: Some(GenerationConfig {
                temperature: None,
                top_k: None,
                top_p: None,
                min_p: None,
                frequency_penalty: None,
                presence_penalty: None,
            }),
            max_active_loras: 8,
            max_lora_rank: 64,
            lora_mode: "fallback".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddedModelInfo {
    pub id: String,
    pub max_active_loras: usize,
    pub loaded_loras: usize,
    pub lora_mode: String,
}

pub enum EmbeddedChatOutput {
    Completion(ChatCompletionResponse),
    Streaming(Receiver<ChatResponse>),
}

#[derive(Debug, Clone)]
pub struct EmbeddedSessionInfo {
    pub session_id: String,
    pub prompt_len: usize,
    pub cached_len: usize,
    pub generated_tokens: usize,
    pub adapter_id: Option<String>,
}

#[derive(Clone)]
pub struct EmbeddedCandleVllmHost {
    data: Arc<OpenAIServerData>,
    runtime_internal: RuntimeInternalService,
    model_id: String,
}

fn normalize_model_id_arg(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn derive_runtime_model_id(
    runtime_canonical_model_id: Option<&str>,
    model_id: Option<&str>,
    weight_path: Option<&str>,
    pipeline_name: &str,
) -> String {
    if let Some(model_id) = runtime_canonical_model_id.and_then(normalize_model_id_arg) {
        return model_id;
    }
    if let Some(model_id) = model_id.and_then(normalize_model_id_arg) {
        return model_id;
    }
    if let Some(weight_path) = weight_path.and_then(normalize_model_id_arg) {
        if let Some(name) = Path::new(&weight_path)
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            return name.to_string();
        }
    }
    pipeline_name.trim().to_string()
}

fn parse_adapter_id(request: &ChatCompletionRequest, adapter_override: Option<String>) -> Option<String> {
    adapter_override
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
        .or_else(|| {
            request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.hybrie.as_ref())
                .and_then(|hybrie| hybrie.adapter_id.clone())
                .map(|id| id.trim().to_string())
                .filter(|id| !id.is_empty())
        })
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

async fn build_prompt(
    data: &OpenAIServerData,
    request: &ChatCompletionRequest,
    tool_config: &crate::openai::ResolvedToolConfig,
) -> std::result::Result<String, APIError> {
    let mut model = data.model.write();
    let pipeline = model
        .get_mut_pipeline(0)
        .ok_or_else(|| APIError::new("Missing pipeline".to_string()))?;
    let mut conversation = pipeline.0.get_conversation().clone();

    match &request.messages {
        Messages::Literal(msg) => return Ok(msg.clone()),
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
                    .ok_or_else(|| APIError::new("Message key `role` not found.".to_string()))?;
                let content = message
                    .get("content")
                    .ok_or_else(|| APIError::new("Message key `content` not found.".to_string()))?
                    .clone();
                if role == "system" {
                    conversation.set_system_message(Some(content.clone()));
                } else {
                    conversation.append_message(role.to_string(), content);
                }
            }
        }
    }

    if !tool_config.tools.is_empty() {
        let mut tools_prompt = ToolFormat::get_tool_prompt(&pipeline.0.tool_config);
        if let ToolChoiceKind::Function(name) = &tool_config.choice {
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

    Ok(conversation.get_prompt(
        request.thinking.unwrap_or(false),
        &tool_config.tools,
    ))
}

async fn check_length(
    request: &ChatCompletionRequest,
    prompt: String,
    data: &OpenAIServerData,
) -> std::result::Result<(Vec<u32>, usize), APIError> {
    let (token_ids, available_kv_tokens) = {
        let model = data.model.read();
        let available_kv_tokens = model.get_available_kv_tokens();
        let pipeline = model
            .get_pipeline(0)
            .ok_or_else(|| APIError::new("Missing pipeline".to_string()))?;
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
            "This model's maximum context length is {} tokens. However, you requested {} tokens.",
            data.pipeline_config.max_model_len,
            max_gen_tokens + token_ids.len(),
        )))
    } else if token_ids.len() >= available_kv_tokens {
        Err(APIError::new(format!(
            "Requested prompt({} tokens) is larger than available kvcache (maximum {} tokens).",
            token_ids.len(),
            available_kv_tokens
        )))
    } else {
        let max_valid_request_tokens =
            std::cmp::min(available_kv_tokens, data.pipeline_config.max_model_len) - 10;
        Ok((token_ids, max_valid_request_tokens))
    }
}

impl EmbeddedCandleVllmHost {
    pub async fn new(config: EmbeddedRuntimeConfig) -> CandleResult<Self> {
        if config.prefill_chunk_size.is_some_and(|size| size % 1024 != 0) {
            candle_core::bail!("prefill_chunk_size must be divisible by 1024");
        }

        let loader = Box::new(DefaultLoader::new(
            config.model_id.clone(),
            config.weight_path.clone(),
            config.weight_file.clone(),
        ));
        let (paths, gguf) =
            loader.prepare_model_weights(config.hf_token.clone(), config.hf_token_path.clone())?;

        let dtype = crate::get_dtype(config.dtype.clone());
        let kv_cache_dtype = if config.fp8_kvcache { DType::U8 } else { dtype };
        let device_ids = if config.device_ids.is_empty() {
            vec![0usize]
        } else {
            config.device_ids.clone()
        };
        let num_shards = device_ids.len();

        #[cfg(not(feature = "nccl"))]
        if num_shards > 1 {
            candle_core::bail!("multiple shards require the nccl feature");
        }

        let (default_pipelines, mut pipeline_config) = {
            #[cfg(feature = "nccl")]
            let loaded = loader
                .load_model(
                    paths,
                    dtype,
                    kv_cache_dtype,
                    gguf,
                    config.isq.clone(),
                    config.block_size,
                    config.max_num_seqs,
                    device_ids,
                    None,
                    Some(0),
                    Some(1),
                    None,
                    None,
                )
                .await;

            #[cfg(not(feature = "nccl"))]
            let loaded = loader
                .load_model(
                    paths,
                    dtype,
                    kv_cache_dtype,
                    gguf,
                    config.isq.clone(),
                    config.block_size,
                    config.max_num_seqs,
                    device_ids,
                    Some(0),
                    Some(1),
                )
                .await;

            match loaded {
                Ok(value) => value,
                Err(err) => return Err(err),
            }
        };

        let mut runtime_model_config = None;
        let mut cache_config = None;
        let pipelines = default_pipelines
            .into_iter()
            .map(|pipeline| {
                let cfg = pipeline.get_model_config();
                let local_cache_cfg = crate::get_cache_config(
                    config.kvcache_mem_gpu,
                    config.kvcache_mem_cpu,
                    config.block_size,
                    &cfg,
                    kv_cache_dtype,
                    num_shards,
                );
                let cache_engine = CacheEngine::new(
                    &cfg,
                    &local_cache_cfg,
                    local_cache_cfg.dtype,
                    pipeline.device(),
                    num_shards,
                )?;
                if runtime_model_config.is_none() {
                    runtime_model_config = Some(cfg.clone());
                }
                if cache_config.is_none() {
                    cache_config = Some(local_cache_cfg.clone());
                }
                Ok((pipeline.rank(), (pipeline, cache_engine)))
            })
            .collect::<CandleResult<HashMap<_, _>>>()?;

        let cache_config = cache_config.expect("cache config should be initialized");
        let model_config = runtime_model_config.expect("model config should be initialized");
        let total_gpu_blocks = cache_config.num_gpu_blocks.unwrap_or(0);
        let default_prefix_cache_blocks = if total_gpu_blocks > 0 {
            std::cmp::max(1, total_gpu_blocks / 4)
        } else {
            0
        };
        let prefix_cache_max_blocks = if config.prefix_cache {
            let max_blocks = config
                .prefix_cache_max_tokens
                .map(|tokens| tokens / cache_config.block_size)
                .unwrap_or(default_prefix_cache_blocks);
            std::cmp::min(max_blocks, total_gpu_blocks)
        } else {
            0
        };
        let prefix_cache_config = PrefixCacheConfig {
            enabled: config.prefix_cache,
            max_cached_blocks: prefix_cache_max_blocks,
        };

        let llm_engine = LLMEngine::new(
            pipelines,
            SchedulerConfig {
                max_num_seqs: config.max_num_seqs,
                prefix_cache: prefix_cache_config,
            },
            &cache_config,
            &model_config,
            Arc::new(Notify::new()),
            config.holding_time,
            num_shards,
            false,
            #[cfg(feature = "nccl")]
            None,
            config.prefill_chunk_size,
        )?;

        if config.generation_cfg.is_some() || pipeline_config.generation_cfg.is_none() {
            pipeline_config.generation_cfg = config.generation_cfg.clone();
        }

        let pipeline_model_name = {
            let engine = llm_engine.read();
            let (pipeline, _) = engine
                .get_pipeline(0)
                .ok_or_else(|| candle_core::Error::msg("Missing pipeline at rank 0"))?;
            pipeline.name().to_string()
        };
        let runtime_model_name = derive_runtime_model_id(
            config.runtime_canonical_model_id.as_deref(),
            config.model_id.as_deref(),
            config.weight_path.as_deref(),
            &pipeline_model_name,
        );

        let backend_router = Arc::new(BackendRouter::new(
            Vec::new(),
            1.0,
            1.0,
            std::time::Duration::from_secs(5),
            false,
            Some(runtime_model_name.clone()),
        )
        .map_err(|err| candle_core::Error::msg(err.to_string()))?);
        let lora_manager = Arc::new(LoRAManager::new(
            Some(runtime_model_name.clone()),
            config.max_active_loras,
            config.max_lora_rank,
            config.lora_mode.clone(),
        ));
        set_global_lora_manager(Arc::clone(&lora_manager));

        let runtime_dtype = format!("{:?}", cache_config.dtype).to_lowercase();
        let runtime_kv_profile = RuntimeKvProfile {
            model_id: runtime_model_name.clone(),
            model_hash: RuntimeInternalService::model_hash(
                &runtime_model_name,
                &runtime_dtype,
                cache_config.block_size,
                model_config.num_hidden_layers,
                model_config
                    .num_key_value_heads
                    .unwrap_or(model_config.num_attention_heads)
                    / num_shards,
                model_config.k_head_dim(),
            ),
            dtype: runtime_dtype,
            block_size: cache_config.block_size,
            num_layers: model_config.num_hidden_layers,
            kv_heads: model_config
                .num_key_value_heads
                .unwrap_or(model_config.num_attention_heads)
                / num_shards,
            head_dim: model_config.k_head_dim(),
        };

        let data = Arc::new(OpenAIServerData {
            model: llm_engine,
            pipeline_config: PipelineConfig {
                max_model_len: pipeline_config.max_model_len,
                default_max_tokens: pipeline_config.default_max_tokens,
                generation_cfg: pipeline_config.generation_cfg,
            },
            record_conversation: config.record_conversation,
            device: Device::Cpu,
            runtime_local_only_strict: config.runtime_local_only_strict,
            mcp_manager: None,
            lora_manager: Arc::clone(&lora_manager),
            backend_router,
            sticky_adapters: Arc::new(RwLock::new(HashMap::new())),
        });
        let runtime_internal = RuntimeInternalService::new(
            Arc::clone(&data.model),
            Arc::clone(&lora_manager),
            runtime_kv_profile,
            1024 * 1024,
        );

        Ok(Self {
            data,
            runtime_internal,
            model_id: runtime_model_name,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn models(&self) -> Vec<EmbeddedModelInfo> {
        let status = self.data.lora_manager.status();
        vec![EmbeddedModelInfo {
            id: self.model_id.clone(),
            max_active_loras: status.max_active_loras,
            loaded_loras: status.loaded_loras,
            lora_mode: status.lora_mode,
        }]
    }

    pub fn health(&self) -> AdapterStatusResponse {
        self.data.lora_manager.status()
    }

    pub fn runtime_internal_service(&self) -> RuntimeInternalService {
        self.runtime_internal.clone()
    }

    pub fn register_adapter(&self, request: RegisterAdapterRequest) -> CandleResult<AdapterRecord> {
        self.data.lora_manager.register(request)
    }

    pub async fn load_adapter(
        &self,
        adapter_id: &str,
        pin: bool,
    ) -> CandleResult<AdapterRecord> {
        self.data.lora_manager.load_async(adapter_id, Some(pin)).await
    }

    pub fn unload_adapter(&self, adapter_id: &str) -> CandleResult<AdapterRecord> {
        self.data.lora_manager.unload(adapter_id)
    }

    pub fn adapter_status(&self) -> AdapterStatusResponse {
        self.data.lora_manager.status()
    }

    pub fn get_adapter(&self, adapter_id: &str) -> Option<AdapterRecord> {
        self.data.lora_manager.get(adapter_id)
    }

    fn default_session_sampling_params(&self) -> CandleResult<SamplingParams> {
        let generation_cfg = self
            .data
            .pipeline_config
            .generation_cfg
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("generation config missing"))?;
        SamplingParams::new(
            1,
            None,
            generation_cfg.presence_penalty.unwrap_or(0.0),
            generation_cfg.frequency_penalty.unwrap_or(0.0),
            None,
            generation_cfg.temperature,
            generation_cfg.top_p,
            generation_cfg.min_p,
            generation_cfg.top_k,
            false,
            1.0,
            EarlyStoppingCondition::UnlikelyBetterCandidates,
            None,
            vec![],
            false,
            self.data.pipeline_config.default_max_tokens,
            None,
            None,
            true,
            None,
        )
        .map_err(candle_core::Error::msg)
    }

    #[allow(clippy::too_many_arguments)]
    async fn prefill_session_inner(
        &self,
        prompt_ids: Vec<u32>,
        sampling_params: Option<SamplingParams>,
        adapter_id: Option<String>,
        adapter_timeline: Option<Vec<crate::openai::requests::HybrieAdapterStep>>,
        session_id: Option<String>,
        import_only: bool,
    ) -> CandleResult<EmbeddedSessionInfo> {
        if prompt_ids.is_empty() {
            candle_core::bail!("prompt_ids must not be empty");
        }
        if let Some(adapter) = adapter_id.as_deref() {
            self.data.lora_manager.ensure_loaded_async(adapter).await?;
        }

        let sampling_params = match sampling_params {
            Some(params) => params,
            None => self.default_session_sampling_params()?,
        };
        let request_id = session_id.unwrap_or_else(|| format!("sess-{}", Uuid::new_v4()));
        let control = if import_only {
            InteractiveSessionControl::new_import_only()
        } else {
            InteractiveSessionControl::new()
        };

        {
            let mut model = self.data.model.write();
            model.add_interactive_request(
                prompt_ids,
                request_id.clone(),
                SystemTime::now(),
                sampling_params,
                false,
                false,
                EncodingFormat::default(),
                EmbeddingType::default(),
                adapter_id,
                adapter_timeline,
                control.clone(),
            );
            model.notify.notify_one();
        }

        match control.next_event().await {
            InteractiveSessionEvent::PrefillReady => {}
            InteractiveSessionEvent::Error(message) => candle_core::bail!("{}", message),
            event => {
                candle_core::bail!("unexpected interactive event during prefill: {:?}", event);
            }
        }

        let info = self
            .session_info(&request_id)?
            .ok_or_else(|| candle_core::Error::msg("interactive session disappeared"))?;
        Ok(info)
    }

    pub async fn prefill_session(
        &self,
        prompt_ids: Vec<u32>,
        sampling_params: Option<SamplingParams>,
        adapter_id: Option<String>,
        adapter_timeline: Option<Vec<crate::openai::requests::HybrieAdapterStep>>,
    ) -> CandleResult<EmbeddedSessionInfo> {
        self.prefill_session_inner(
            prompt_ids,
            sampling_params,
            adapter_id,
            adapter_timeline,
            None,
            false,
        )
        .await
    }

    pub async fn prepare_import_session(
        &self,
        session_id: String,
        prompt_ids: Vec<u32>,
        sampling_params: Option<SamplingParams>,
        adapter_id: Option<String>,
    ) -> CandleResult<EmbeddedSessionInfo> {
        self.prefill_session_inner(
            prompt_ids,
            sampling_params,
            adapter_id,
            None,
            Some(session_id),
            true,
        )
        .await
    }

    pub fn session_info(&self, session_id: &str) -> CandleResult<Option<EmbeddedSessionInfo>> {
        let model = self.data.model.read();
        Ok(model.get_session_lookup(session_id).map(|lookup| EmbeddedSessionInfo {
            session_id: lookup.request_id,
            prompt_len: lookup.prompt_len,
            cached_len: lookup.total_len,
            generated_tokens: lookup.total_len.saturating_sub(lookup.prompt_len),
            adapter_id: lookup.adapter_id,
        }))
    }

    pub async fn decode_next(
        &self,
        session_id: &str,
        input_ids: Vec<u32>,
    ) -> CandleResult<Vec<u32>> {
        if !input_ids.is_empty() {
            candle_core::bail!("appending input_ids during decode is not supported");
        }
        let control = {
            let mut model = self.data.model.write();
            let Some(control) = model.interactive_control(session_id) else {
                candle_core::bail!("session '{}' not found", session_id);
            };
            let granted = model.grant_decode_steps(session_id, 1);
            if !granted {
                candle_core::bail!("session '{}' is not interactive", session_id);
            }
            model.notify.notify_one();
            control
        };

        match control.next_event().await {
            InteractiveSessionEvent::Token(token) => Ok(vec![token]),
            InteractiveSessionEvent::Finished => Ok(Vec::new()),
            InteractiveSessionEvent::Error(message) => candle_core::bail!("{}", message),
            InteractiveSessionEvent::PrefillReady => {
                candle_core::bail!("unexpected prefill-ready event during decode")
            }
        }
    }

    pub fn export_session_snapshot(
        &self,
        session_id: &str,
    ) -> CandleResult<Option<InteractiveSessionSnapshot>> {
        let model = self.data.model.read();
        model.export_interactive_session_snapshot(session_id)
    }

    pub async fn import_session_snapshot(
        &self,
        snapshot: InteractiveSessionSnapshot,
    ) -> CandleResult<EmbeddedSessionInfo> {
        let info = self
            .prepare_import_session(
                snapshot.session_id.clone(),
                snapshot.prompt_ids.clone(),
                Some(snapshot.sampling_params.clone()),
                snapshot.adapter_id.clone(),
            )
            .await?;
        let mut model = self.data.model.write();
        model.import_session_payload(snapshot.kv_payload)?;
        Ok(info)
    }

    pub fn release_session(&self, session_id: &str) -> bool {
        self.data.model.write().release_session(session_id)
    }

    pub async fn chat(
        &self,
        request: ChatCompletionRequest,
        adapter_override: Option<String>,
        session_id: Option<String>,
    ) -> std::result::Result<EmbeddedChatOutput, APIError> {
        if request.logit_bias.as_ref().is_some_and(|x| !x.is_empty()) {
            return Err(APIError::new_str("`logit_bias` is not currently supported."));
        }

        let mut adapter_id = parse_adapter_id(&request, adapter_override);
        let adapter_timeline = parse_adapter_timeline(&request);
        if adapter_id.is_none() {
            if let Some(session_id) = session_id.as_ref() {
                adapter_id = self.data.sticky_adapters.read().get(session_id).cloned();
            }
        }
        if let (Some(session_id), Some(adapter_id)) = (session_id.as_ref(), adapter_id.as_ref()) {
            self.data
                .sticky_adapters
                .write()
                .insert(session_id.clone(), adapter_id.clone());
        }

        if let Some(adapter) = adapter_id.as_deref() {
            self.data
                .lora_manager
                .ensure_loaded_async(adapter)
                .await
                .map_err(APIError::from)?;
        }

        let tool_config = resolve_tools_for_request(
            &request.tools,
            &request.tool_choice,
            self.data.mcp_manager.as_ref(),
        )?;
        let prompt = build_prompt(&self.data, &request, &tool_config).await?;
        let (token_ids, available_tokens) = check_length(&request, prompt, &self.data).await?;

        let mut max_request_tokens = request
            .max_tokens
            .unwrap_or(self.data.pipeline_config.default_max_tokens);
        if max_request_tokens + token_ids.len() > available_tokens {
            max_request_tokens = available_tokens.saturating_sub(token_ids.len());
            if max_request_tokens == 0 {
                return Err(APIError::new(format!(
                    "Requested prompt({} tokens) is larger than available kvcache (maximum {} tokens).",
                    token_ids.len(),
                    available_tokens
                )));
            }
        }

        let generation_cfg = self
            .data
            .pipeline_config
            .generation_cfg
            .as_ref()
            .ok_or_else(|| APIError::new("generation config missing".to_string()))?;
        let mut sampling_params = SamplingParams::new(
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
        )?;
        let has_tools = !tool_config.tools.is_empty();
        sampling_params.mcp_mode = if has_tools { Some(true) } else { None };

        let request_id = format!("cmpl-{}", Uuid::new_v4());
        let stream_request = request.stream.unwrap_or(false);
        let request_logprobs = request.logprobs.unwrap_or(false);
        let model_name = strip_mode_prefix(request.model.clone()).unwrap_or_else(|| self.model_id.clone());
        let (response_tx, rx) = flume::unbounded();
        let sync_notify = Arc::new(Notify::new());
        let sync_completion_notify = if stream_request {
            None
        } else {
            Some(Arc::clone(&sync_notify))
        };
        let data_for_engine = Arc::clone(&self.data);
        let adapter_for_engine = adapter_id.clone();
        let timeline_for_engine = adapter_timeline.clone();
        let request_id_for_read = request_id.clone();

        let _ = tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async move {
                let mut model = data_for_engine.model.write();
                model.add_request(
                    token_ids,
                    request_id,
                    SystemTime::now(),
                    sampling_params,
                    request_logprobs,
                    false,
                    EncodingFormat::default(),
                    EmbeddingType::default(),
                    adapter_for_engine,
                    timeline_for_engine,
                    if stream_request {
                        Some(Arc::new(response_tx))
                    } else {
                        None
                    },
                    sync_completion_notify,
                );
                model.notify.notify_one();
            });
        });

        if stream_request {
            return Ok(EmbeddedChatOutput::Streaming(rx));
        }

        sync_notify.notified().await;
        let (choices, usage) = {
            let model = self.data.model.read();
            let Some(record) = model.completion_records.get(&request_id_for_read) else {
                return Err(APIError::new(format!(
                    "Unable to generate response for request {}",
                    request_id_for_read
                )));
            };
            (record.0.clone(), record.1.clone())
        };

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

        Ok(EmbeddedChatOutput::Completion(ChatCompletionResponse {
            id: request_id_for_read,
            choices: final_choices,
            created: usage.created,
            model: model_name,
            object: "chat.completion",
            usage,
        }))
    }
}
