use axum::{
    extract::State,
    http::{self, Method},
    routing::{get, post},
    Json, Router,
};
use candle_core::{DType, Device, Result};
#[cfg(feature = "nccl")]
use candle_vllm::backend::heartbeat;
use candle_vllm::openai::backend_router::{BackendRouter, CloudBackendConfig};
use candle_vllm::openai::models::Config;
use candle_vllm::openai::openai_server::{
    adapters_status, chat_completions, create_embeddings, get_adapter, list_adapters, load_adapter,
    register_adapter, unload_adapter, warmup_adapters,
};
use candle_vllm::openai::pipelines::llm_engine::LLMEngine;
use candle_vllm::openai::pipelines::pipeline::DefaultLoader;
use candle_vllm::openai::runtime_internal::proto::runtime_internal_server::RuntimeInternalServer;
use candle_vllm::openai::runtime_internal::{RuntimeInternalService, RuntimeKvProfile};
use candle_vllm::openai::sampling_params::GenerationConfig;
use candle_vllm::openai::OpenAIServerData;
use candle_vllm::scheduler::cache_engine::{CacheConfig, CacheEngine};
use candle_vllm::scheduler::prefix_cache::PrefixCacheConfig;
use candle_vllm::scheduler::SchedulerConfig;
use clap::Parser;
use colored::*;
use local_ip_address::local_ip;
use rustchatui::start_ui_server;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tonic::transport::Server;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Huggingface token environment variable (optional). If not specified, load using hf_token_path.
    #[arg(long)]
    hf_token: Option<String>,

    /// Huggingface token file (optional). If neither `hf_token` or `hf_token_path` are specified this is used with the value
    /// of `~/.cache/huggingface/token`
    #[arg(long)]
    hf_token_path: Option<String>,

    /// Host address to bind to, to serve on host:port
    #[arg(long = "h", default_value = "0.0.0.0")]
    host: String,

    /// Port to serve on (host:port)
    #[arg(long = "p", default_value_t = 2000)]
    port: u16,

    /// Set verbose mode (print all requests)
    #[arg(long)]
    verbose: bool,

    /// Maximum number of sequences to allow
    #[arg(long, default_value_t = 16)]
    max_num_seqs: usize,

    /// Size of a block
    #[arg(long, default_value_t = 64)]
    block_size: usize,

    /// if weight_path is passed, it will ignore the model_id
    #[arg(long = "m")]
    model_id: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online), path must include last "/"
    #[arg(long = "w")]
    weight_path: Option<String>,

    /// The quantized weight file name (for gguf/ggml file)
    #[arg(long = "f")]
    weight_file: Option<String>,

    #[arg(long)]
    dtype: Option<String>,

    #[arg(long)]
    isq: Option<String>,

    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Available GPU memory for kvcache (MB)
    #[arg(long = "mem", default_value_t = 4096)]
    kvcache_mem_gpu: usize,

    /// Available CPU memory for kvcache (MB)
    #[arg(long, default_value_t = 128)]
    kvcache_mem_cpu: usize,

    /// Record conversation (default false, the client need to record chat history)
    #[arg(long)]
    record_conversation: bool,

    #[arg(long = "d", value_delimiter = ',')]
    device_ids: Option<Vec<usize>>,

    /// Maximum waiting time for processing parallel requests (in milliseconds).
    /// A larger value means the engine can hold more requests and process them in a single generation call.
    #[arg(long, default_value_t = 500)]
    holding_time: usize,

    //Whether the program is forced running in multithread model for parallel inference (for debug)
    #[arg(long, default_value_t = false)]
    multithread: bool,

    #[arg(long, default_value_t = false)]
    log: bool,

    #[arg(long)]
    temperature: Option<f32>,

    #[arg(long)]
    top_p: Option<f32>,

    #[arg(long)]
    min_p: Option<f32>,

    #[arg(long)]
    top_k: Option<isize>,

    #[arg(long)]
    frequency_penalty: Option<f32>,

    #[arg(long)]
    presence_penalty: Option<f32>,

    #[arg(long)]
    prefill_chunk_size: Option<usize>,

    #[arg(long, default_value_t = false)]
    fp8_kvcache: bool,

    /// Enable prefix cache to reuse KV cache for repeated prompt prefixes.
    #[arg(long, default_value_t = false)]
    prefix_cache: bool,

    /// Prefix cache size limit in tokens (rounded down to block size).
    #[arg(long)]
    prefix_cache_max_tokens: Option<usize>,

    #[arg(long, default_value_t = false)]
    ui_server: bool, //start candle-vllm with built-in web server

    /// MCP server command (single server mode)
    #[arg(long)]
    mcp_command: Option<String>,

    /// MCP server arguments (comma-separated)
    #[arg(long, value_delimiter = ',')]
    mcp_args: Option<Vec<String>>,

    /// Path to MCP config file (multi-server mode)
    #[arg(long)]
    mcp_config: Option<String>,

    /// Comma-separated cloud backends in `id=https://host:port` or `https://host:port` form.
    #[arg(long, value_delimiter = ',')]
    cloud_backends: Option<Vec<String>>,

    /// Relative weight for local backend score in hybrid mode.
    #[arg(long, default_value_t = 1.0)]
    hybrid_local_weight: f64,

    /// Relative weight for cloud backend score in hybrid mode.
    #[arg(long, default_value_t = 1.0)]
    hybrid_cloud_weight: f64,

    /// Refresh interval for cloud adapter status in hybrid/cloud routing.
    #[arg(long, default_value_t = 5000)]
    hybrid_status_ttl_ms: u64,

    /// Automatically warm adapters on the non-selected cloud backends in hybrid mode.
    #[arg(long, default_value_t = false)]
    hybrid_adapter_autosync: bool,

    /// Enforce runtime-node boundary: public API accepts only local execution mode/scope.
    #[arg(long, default_value_t = true)]
    runtime_local_only_strict: bool,

    /// Canonical model id to advertise in runtime metadata, adapter compatibility, and KV handoff.
    #[arg(long)]
    runtime_canonical_model_id: Option<String>,

    /// Enable internal runtime gRPC APIs for KV handoff and adapter runtime ops.
    #[arg(long, default_value_t = false)]
    runtime_internal_api: bool,

    /// Internal runtime gRPC bind host.
    #[arg(long, default_value = "127.0.0.1")]
    runtime_internal_grpc_host: String,

    /// Internal runtime gRPC bind port.
    #[arg(long, default_value_t = 51051)]
    runtime_internal_grpc_port: u16,

    /// Max chunk size (bytes) for KV export/import streaming.
    #[arg(long, default_value_t = 1_048_576)]
    runtime_internal_chunk_bytes: usize,
}

fn config_log(logger: ftail::Ftail, log_enable: bool, log_file: String) -> Result<()> {
    if !log_enable {
        return Ok(());
    }
    use tracing::log::LevelFilter;
    let mut cfg_filter = LevelFilter::Warn;
    if let Ok(level) = std::env::var("RUST_LOG") {
        let log_level_names: [&str; 6] = ["OFF", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"];
        let log_levels: [LevelFilter; 6] = [
            LevelFilter::Off,
            LevelFilter::Error,
            LevelFilter::Warn,
            LevelFilter::Info,
            LevelFilter::Debug,
            LevelFilter::Trace,
        ];
        let level = level.to_uppercase();
        for (i, name) in log_level_names.iter().copied().enumerate() {
            if level.contains(name) {
                cfg_filter = log_levels[i]
            }
        }
    };
    if std::fs::exists(&log_file).is_ok() {
        let _ = std::fs::remove_file(&log_file);
    }
    logger
        .console(cfg_filter)
        .single_file(log_file.as_str(), true, cfg_filter)
        .init()
        .map_err(candle_core::Error::wrap)
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
        let basename = Path::new(&weight_path)
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .or_else(|| weight_path.rsplit('/').next().map(str::to_string));
        if let Some(name) = basename.filter(|name| !name.trim().is_empty()) {
            return name;
        }
    }

    normalize_model_id_arg(pipeline_name).unwrap_or_else(|| pipeline_name.to_string())
}

#[cfg(test)]
mod tests {
    use super::derive_runtime_model_id;

    #[test]
    fn derive_runtime_model_id_prefers_canonical_override() {
        let resolved = derive_runtime_model_id(
            Some("Qwen/Qwen3.5-9B-Instruct"),
            Some("ignored-model"),
            Some("/models/ignored-weights"),
            "ignored-pipeline",
        );
        assert_eq!(resolved, "Qwen/Qwen3.5-9B-Instruct");
    }

    #[test]
    fn derive_runtime_model_id_falls_back_in_documented_order() {
        let from_model = derive_runtime_model_id(
            None,
            Some("Qwen/Qwen3.5-9B"),
            Some("/models/weights-dir"),
            "pipeline-name",
        );
        assert_eq!(from_model, "Qwen/Qwen3.5-9B");

        let from_weights = derive_runtime_model_id(
            None,
            None,
            Some("/models/Qwen3.5-9B-Instruct/"),
            "pipeline-name",
        );
        assert_eq!(from_weights, "Qwen3.5-9B-Instruct");

        let from_pipeline = derive_runtime_model_id(None, None, None, "qwen-pipeline");
        assert_eq!(from_pipeline, "qwen-pipeline");
    }
}

#[tokio::main]
#[allow(unused_mut)]
async fn main() -> Result<()> {
    let args = Args::parse();
    let runtime_canonical_model_id = args.runtime_canonical_model_id.clone();
    let requested_model_id = args.model_id.clone();
    let requested_weight_path = args.weight_path.clone();
    if !args.log {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    let loader = Box::new(DefaultLoader::new(
        args.model_id,
        args.weight_path,
        args.weight_file,
    ));

    let (paths, gguf) = loader.prepare_model_weights(args.hf_token, args.hf_token_path)?;

    let dtype = candle_vllm::get_dtype(args.dtype);
    let kv_cache_dtype = if args.fp8_kvcache { DType::U8 } else { dtype };

    if cfg!(feature = "flash-decoding") {
        assert!(
            !args.fp8_kvcache,
            "fp8 kvcache is not compatible with `flash-decoding` feature!"
        );
    }

    let device_ids: Vec<usize> = match args.device_ids {
        Some(ids) => ids,
        _ => vec![0usize],
    };
    let local_world_size = device_ids.len();
    let mut num_shards = local_world_size;
    #[cfg(not(feature = "nccl"))]
    assert!(
        num_shards == 1,
        "More than one shard was given, but NCCL is not enabled for parallel inference!"
    );

    if gguf && num_shards > 1 {
        panic!("Multiple device-ids detected: ggml/gguf model is not supported for multi-rank inference! \n\t*** Tips: use unquantized safetensors models (`--w`) with ISQ (e.g., `--isq q4k`) for multi-gpu inference!");
    }

    if gguf && args.isq.is_some() {
        panic!("Quantized gguf/ggml model does not support isq option!");
    }

    assert!(
        args.prefill_chunk_size.is_none() || args.prefill_chunk_size.unwrap() % 1024 == 0,
        "Error: prefill_chunk_size must be divisible by 1024!"
    );

    let multi_process = if num_shards > 1 {
        if args.multithread {
            tracing::warn!("The program is forced running under multithread mode (for debug purpose), which may not stable!");
            false
        } else {
            tracing::warn!("Multi-process mode is automatically enabled for multi-rank inference!");
            true
        }
    } else {
        !args.multithread
    };

    #[cfg(all(feature = "cuda", feature = "graph"))]
    {
        assert!(
            multi_process,
            "Graph capture is only available under multi process mode!"
        );
        if args.max_num_seqs > 16 {
            tracing::warn!("Higher GPU memory required for capturing large batch!");
        }
    }

    let logger: ftail::Ftail = ftail::Ftail::new();
    let host = args.host.clone();
    let mut port = args.port;
    #[cfg(feature = "nccl")]
    let (pipelines, global_rank, daemon_manager) = if multi_process {
        use candle_vllm::openai::communicator::init_subprocess;
        let (id, local_rank, global_rank, global_world_size, daemon_manager) =
            init_subprocess(device_ids.clone()).unwrap();
        if global_rank != 0 {
            port = port + global_rank as u16; //processes other than rank 0 use fake server port since they do not perform response
        }
        num_shards = global_world_size;
        let log_file = format!("candle-vllm-rank-{}.log", global_rank);
        let _ = config_log(logger, args.log, log_file);

        warn!("subprocess rank {} started!", global_rank);
        heartbeat::heartbeat_worker(Some(local_world_size - 1)).await;

        (
            loader
                .load_model(
                    paths,
                    dtype,
                    kv_cache_dtype,
                    gguf,
                    args.isq.clone(),
                    args.block_size,
                    args.max_num_seqs,
                    vec![device_ids[local_rank]],
                    Some(id),
                    Some(local_rank),
                    Some(local_world_size),
                    Some(global_rank),
                    Some(global_world_size),
                )
                .await,
            global_rank,
            Some(daemon_manager),
        )
    } else {
        use candle_vllm::openai::communicator::DaemonManager;
        DaemonManager::set_master_rank(true); //master rank default for multithreaded mode
        let log_file = format!("candle-vllm-{}ranks.log", device_ids.len());
        let _ = config_log(logger, args.log, log_file);
        (
            loader
                .load_model(
                    paths,
                    dtype,
                    kv_cache_dtype,
                    gguf,
                    args.isq.clone(),
                    args.block_size,
                    args.max_num_seqs,
                    device_ids,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await,
            0,
            None,
        )
    };

    #[cfg(feature = "nccl")]
    info!(
        "parallel model: {}!",
        if multi_process {
            "multiprocess"
        } else {
            "multithread"
        }
    );

    #[cfg(not(feature = "nccl"))]
    let (pipelines, global_rank) = {
        let log_file = "candle-vllm.log".to_string();
        let _ = config_log(logger, args.log, log_file);
        (
            loader
                .load_model(
                    paths,
                    dtype,
                    kv_cache_dtype,
                    gguf,
                    args.isq.clone(),
                    args.block_size,
                    args.max_num_seqs,
                    device_ids,
                    None,
                    None,
                )
                .await,
            0,
        )
    };

    let (default_pipelines, mut pipeline_config) = match pipelines {
        Err(e) => panic!("{e:?}"),
        Ok((p, c)) => (p, c),
    };
    let mut config: Option<Config> = None;
    let mut cache_config: Option<CacheConfig> = None;

    let pipelines = default_pipelines
        .into_iter()
        .map(|pipeline| {
            let cfg = pipeline.get_model_config();
            let cache_cfg = candle_vllm::get_cache_config(
                args.kvcache_mem_gpu,
                args.kvcache_mem_cpu, //dummy 512MB for cpu
                args.block_size,
                &cfg,
                kv_cache_dtype,
                num_shards,
            );
            let cache_engine = CacheEngine::new(
                &cfg,
                &cache_cfg,
                cache_cfg.dtype,
                pipeline.device(),
                num_shards,
            )
            .unwrap();
            if config.is_none() {
                config = Some(cfg.clone());
            }
            if cache_config.is_none() {
                cache_config = Some(cache_cfg.clone());
            }
            (pipeline.rank(), (pipeline, cache_engine))
        })
        .collect();

    let cache_config = cache_config.as_ref().unwrap().clone();
    let config = config.as_ref().unwrap().clone();
    info!("Cache config {:?}", cache_config);

    let total_gpu_blocks = cache_config.num_gpu_blocks.unwrap_or(0);
    let default_prefix_cache_blocks = if total_gpu_blocks > 0 {
        std::cmp::max(1, total_gpu_blocks / 4)
    } else {
        0
    };
    let prefix_cache_max_blocks = if args.prefix_cache {
        let max_blocks = args
            .prefix_cache_max_tokens
            .map(|tokens| tokens / cache_config.block_size)
            .unwrap_or(default_prefix_cache_blocks);
        std::cmp::min(max_blocks, total_gpu_blocks)
    } else {
        0
    };
    let prefix_cache_config = PrefixCacheConfig {
        enabled: args.prefix_cache,
        max_cached_blocks: prefix_cache_max_blocks,
    };

    let llm_engine = LLMEngine::new(
        pipelines,
        SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
            prefix_cache: prefix_cache_config,
        },
        &cache_config,
        &config,
        Arc::new(Notify::new()),
        args.holding_time,
        num_shards,
        multi_process,
        #[cfg(feature = "nccl")]
        daemon_manager,
        args.prefill_chunk_size,
    )?;

    if args.temperature.is_some() || pipeline_config.generation_cfg.is_none() {
        //overwrite the generation config when temperature (and others) specified in arguments
        //disable multinomial sampling (generation randomness) by setting `temperature` as 0
        pipeline_config.generation_cfg = Some(GenerationConfig {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            min_p: args.min_p,
            frequency_penalty: args.frequency_penalty,
            presence_penalty: args.presence_penalty,
        })
    } else {
        pipeline_config
            .generation_cfg
            .as_mut()
            .unwrap()
            .frequency_penalty = args.frequency_penalty;
        pipeline_config
            .generation_cfg
            .as_mut()
            .unwrap()
            .presence_penalty = args.presence_penalty;
        pipeline_config.generation_cfg.as_mut().unwrap().min_p = args.min_p;
    }

    info!("Pipeline config {:?}", pipeline_config);

    let max_model_len = pipeline_config.max_model_len;
    let kvcached_tokens = cache_config.num_gpu_blocks.unwrap() * cache_config.block_size;

    let mcp_manager_config = if let Some(path) = &args.mcp_config {
        match candle_vllm::mcp::McpManagerConfig::from_file(path) {
            Ok(cfg) => Some(cfg),
            Err(err) => {
                tracing::error!("Failed to load MCP config file: {:?}", err);
                None
            }
        }
    } else if let Some(command) = args.mcp_command.clone() {
        Some(candle_vllm::mcp::McpManagerConfig::from_single(
            candle_vllm::mcp::manager::McpToolConfig::new(
                command,
                args.mcp_args.clone().unwrap_or_default(),
            ),
        ))
    } else {
        None
    };

    // Initialize MCP Manager
    let mcp_manager = if let Some(cfg) = mcp_manager_config {
        match candle_vllm::mcp::McpClientManager::new(cfg) {
            Ok(manager) => Some(Arc::new(manager)),
            Err(err) => {
                tracing::error!("Failed to start MCP client manager: {:?}", err);
                None
            }
        }
    } else {
        None
    };

    let pipeline_model_name = {
        let engine = llm_engine.read();
        let (pipeline, _) = engine
            .get_pipeline(0)
            .ok_or_else(|| candle_core::Error::msg("Missing pipeline at rank 0"))?;
        pipeline.name().to_string()
    };
    let runtime_model_name = derive_runtime_model_id(
        runtime_canonical_model_id.as_deref(),
        requested_model_id.as_deref(),
        requested_weight_path.as_deref(),
        &pipeline_model_name,
    );

    let cloud_specs = args.cloud_backends.clone().unwrap_or_default();
    let mut cloud_backends =
        CloudBackendConfig::parse_specs(&cloud_specs).map_err(candle_core::Error::msg)?;
    if args.runtime_local_only_strict && !cloud_backends.is_empty() {
        warn!(
            "runtime_local_only_strict=true; ignoring configured cloud backends for external routing."
        );
        cloud_backends.clear();
    }
    if !cloud_backends.is_empty() {
        let rendered = cloud_backends
            .iter()
            .map(|b| format!("{}={}", b.id, b.base_url))
            .collect::<Vec<_>>()
            .join(", ");
        info!("Configured cloud backends: {}", rendered);
    }

    let backend_router = Arc::new(
        BackendRouter::new(
            cloud_backends,
            args.hybrid_local_weight,
            args.hybrid_cloud_weight,
            Duration::from_millis(args.hybrid_status_ttl_ms),
            args.hybrid_adapter_autosync,
            Some(runtime_model_name.clone()),
        )
        .map_err(|err| candle_core::Error::msg(err.to_string()))?,
    );

    let lora_manager = Arc::new(candle_vllm::openai::lora::LoRAManager::new(
        Some(runtime_model_name.clone()),
        8,
        64,
        "fallback",
    ));
    candle_vllm::openai::lora::set_global_lora_manager(Arc::clone(&lora_manager));

    let runtime_dtype = format!("{:?}", cache_config.dtype).to_lowercase();
    let runtime_kv_profile = RuntimeKvProfile {
        model_id: runtime_model_name.clone(),
        model_hash: RuntimeInternalService::model_hash(
            &runtime_model_name,
            &runtime_dtype,
            cache_config.block_size,
            config.num_hidden_layers,
            config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads)
                / num_shards,
            config.k_head_dim(),
        ),
        dtype: runtime_dtype,
        block_size: cache_config.block_size,
        num_layers: config.num_hidden_layers,
        kv_heads: config
            .num_key_value_heads
            .unwrap_or(config.num_attention_heads)
            / num_shards,
        head_dim: config.k_head_dim(),
    };

    let server_data = OpenAIServerData {
        pipeline_config,
        model: llm_engine,
        record_conversation: args.record_conversation,
        device: Device::Cpu,
        runtime_local_only_strict: args.runtime_local_only_strict,
        mcp_manager: mcp_manager.clone(),
        lora_manager,
        backend_router,
        sticky_adapters: Arc::new(parking_lot::RwLock::new(HashMap::new())),
    };

    if let Some(manager) = &mcp_manager {
        info!("Waiting for MCP tools to be available...");
        if manager.wait_for_available(std::time::Duration::from_secs(30)) {
            info!("MCP tools available.");
        } else {
            warn!("MCP tools wait timed out.");
        }
    }

    if global_rank != 0 {
        info!("\nDaemon service started at rank {}.", global_rank);
    }

    #[cfg(feature = "nccl")]
    if multi_process {
        let e = server_data.model.read();
        let mut daemon_manager = e.daemon_manager.write();
        daemon_manager.as_mut().unwrap().mpi_sync();
    }

    #[cfg(all(feature = "cuda", feature = "graph"))]
    LLMEngine::graph_capture(&server_data.model).unwrap();

    if global_rank == 0 {
        warn!(
            "Maximum Model Length (affected by `--mem` (kvcache-mem-gpu) and the number of ranks):"
        );
        for batch in [1, 8] {
            println!(
                "-> Batch {}: {}",
                batch,
                std::cmp::min(kvcached_tokens / batch, max_model_len)
            );
        }
        let ip = local_ip().unwrap_or("127.0.0.1".parse().unwrap());
        let local_url = format!("http://localhost:{port}/v1/");
        let lan_url = format!("http://{ip}:{port}/v1/");

        println!(
            "\n🧠 API server running at:\n\t{} (Local Access) \n\t{} (Remote Access)\n",
            local_url.cyan().bold(),
            lan_url.cyan().bold(),
        );

        println!("");
        println!(
            "🛑 {}",
            format!("EXIT: Ctrl+C to quit. If unresponsive: Ctrl+P → Ctrl+Q (last resort).")
                .bold()
                .red()
        );
    }

    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(Any) // same as "*"
        .allow_methods(Any)
        .allow_headers(Any);

    let shared_server_data = Arc::new(server_data);
    let app = Router::new()
        .route(
            "/v1/health",
            get(|State(data): State<Arc<OpenAIServerData>>| async move {
                let lora_status = data.lora_manager.status();
                Json(json!({
                    "status": "ok",
                    "runtime_local_only_strict": data.runtime_local_only_strict,
                    "max_active_loras": lora_status.max_active_loras,
                    "loaded_loras": lora_status.loaded_loras,
                    "lora_mode": lora_status.lora_mode
                }))
            }),
        )
        .route(
            "/v1/models",
            get(|State(data): State<Arc<OpenAIServerData>>| async move {
                let (model_name, lora_status) = {
                    let engine = data.model.read();
                    let (pipeline, _) = engine.get_pipeline(0).unwrap();
                    (pipeline.name().to_string(), data.lora_manager.status())
                };
                let cloud_backends = data.backend_router.cloud_backend_ids();
                let mut execution_modes = vec!["local"];
                if !data.runtime_local_only_strict && !cloud_backends.is_empty() {
                    execution_modes.push("cloud");
                    execution_modes.push("hybrid");
                }
                Json(json!({
                    "object": "list",
                    "data": [
                        {
                            "id": model_name,
                            "object": "model",
                            "created": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as i64,
                            "owned_by": "candle-vllm",
                            "permission": [],
                            "max_active_loras": lora_status.max_active_loras,
                            "loaded_loras": lora_status.loaded_loras,
                            "lora_mode": lora_status.lora_mode,
                            "execution_modes": execution_modes,
                            "cloud_backends": cloud_backends
                        }
                    ]
                }))
            }),
        )
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(create_embeddings))
        .route("/v1/adapters", post(register_adapter).get(list_adapters))
        .route("/v1/adapters/:id", get(get_adapter))
        .route("/v1/adapters/:id/load", post(load_adapter))
        .route("/v1/adapters/:id/unload", post(unload_adapter))
        .route("/v1/adapters/status", get(adapters_status))
        .route("/v1/adapters/warmup", post(warmup_adapters))
        .layer(cors_layer)
        .with_state(Arc::clone(&shared_server_data));

    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}"))
        .await
        .map_err(candle_core::Error::wrap)?;

    let mut tasks = Vec::new();
    tasks.push(tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("Chat API server error: {e:?}");
        }
    }));

    if args.runtime_internal_api && global_rank == 0 {
        let grpc_addr = format!(
            "{}:{}",
            args.runtime_internal_grpc_host, args.runtime_internal_grpc_port
        )
        .parse()
        .map_err(candle_core::Error::wrap)?;
        let internal_service = RuntimeInternalService::new(
            Arc::clone(&shared_server_data.model),
            Arc::clone(&shared_server_data.lora_manager),
            runtime_kv_profile,
            args.runtime_internal_chunk_bytes,
        );
        tasks.push(tokio::spawn(async move {
            if let Err(err) = Server::builder()
                .add_service(RuntimeInternalServer::new(internal_service))
                .serve(grpc_addr)
                .await
            {
                eprintln!("Runtime internal gRPC error: {err:?}");
            }
        }));
    }

    // Usage example: https://github.com/guoqingbao/rustchatui/blob/main/ReadMe.md
    if args.ui_server && global_rank == 0 {
        tasks.push(tokio::spawn(async move {
            start_ui_server((args.port - 1) as u16, Some(args.port as u16), None, None)
                .await
                .unwrap();
        }));
    }

    futures::future::try_join_all(tasks)
        .await
        .map_err(candle_core::Error::wrap)?;

    Ok(())
}
