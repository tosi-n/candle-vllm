use candle_core::{DType, Device, Result, Tensor};
use half::{bf16, f16};
use parking_lot::RwLock;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;
use tokio::sync::Notify;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdapterLoadState {
    Registered,
    Loading,
    Loaded,
    Evicting,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdapterConfig {
    #[serde(default, alias = "base_model_name_or_path")]
    pub base_model: Option<String>,
    #[serde(default, alias = "r")]
    pub rank: Option<usize>,
    #[serde(default, alias = "lora_alpha")]
    pub alpha: Option<f64>,
    #[serde(default)]
    pub target_modules: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterAdapterRequest {
    pub id: String,
    pub adapter_path: String,
    #[serde(default)]
    pub config_path: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub checksum_sha256: Option<String>,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub base_model: Option<String>,
    #[serde(default)]
    pub artifact_refs: Vec<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LoadAdapterRequest {
    #[serde(default)]
    pub pinned: Option<bool>,
    #[serde(default)]
    pub wait: Option<bool>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct UnloadAdapterRequest {
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterRecord {
    pub id: String,
    pub version: Option<String>,
    pub adapter_path: String,
    pub config_path: String,
    pub checksum_sha256: Option<String>,
    pub state: AdapterLoadState,
    pub pinned: bool,
    pub base_model: Option<String>,
    pub rank: Option<usize>,
    pub alpha: Option<f64>,
    pub target_modules: Vec<String>,
    pub artifact_refs: Vec<String>,
    pub loaded_modules: usize,
    pub loaded_at: Option<SystemTime>,
    pub last_used_at: Option<SystemTime>,
    pub last_error: Option<String>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterStatusResponse {
    pub max_active_loras: usize,
    pub loaded_loras: usize,
    pub lora_mode: String,
    pub slots_used: usize,
    pub slots_free: usize,
    pub adapters: Vec<AdapterRecord>,
}

#[derive(Debug, Clone, Copy)]
pub enum ShardSpec {
    Column { rank: usize, world_size: usize },
    Row { rank: usize, world_size: usize },
}

#[derive(Clone)]
struct LoRAModule {
    a: Tensor,
    b: Tensor,
    scale: f64,
}

#[derive(Clone)]
struct LoadedAdapter {
    modules: HashMap<String, LoRAModule>,
    loaded_at: SystemTime,
    last_used: SystemTime,
}

struct AdapterEntry {
    id: String,
    version: Option<String>,
    adapter_path: PathBuf,
    config_path: PathBuf,
    checksum_sha256: Option<String>,
    pinned: bool,
    state: AdapterLoadState,
    config: AdapterConfig,
    artifact_refs: Vec<String>,
    loaded: Option<LoadedAdapter>,
    last_error: Option<String>,
    created_at: SystemTime,
    updated_at: SystemTime,
}

impl AdapterEntry {
    fn to_record(&self) -> AdapterRecord {
        AdapterRecord {
            id: self.id.clone(),
            version: self.version.clone(),
            adapter_path: self.adapter_path.display().to_string(),
            config_path: self.config_path.display().to_string(),
            checksum_sha256: self.checksum_sha256.clone(),
            state: self.state.clone(),
            pinned: self.pinned,
            base_model: self.config.base_model.clone(),
            rank: self.config.rank,
            alpha: self.config.alpha,
            target_modules: self.config.target_modules.clone().unwrap_or_default(),
            artifact_refs: self.artifact_refs.clone(),
            loaded_modules: self
                .loaded
                .as_ref()
                .map(|loaded| loaded.modules.len())
                .unwrap_or(0),
            loaded_at: self.loaded.as_ref().map(|loaded| loaded.loaded_at),
            last_used_at: self.loaded.as_ref().map(|loaded| loaded.last_used),
            last_error: self.last_error.clone(),
            created_at: self.created_at,
            updated_at: self.updated_at,
        }
    }
}

struct LoRAManagerInner {
    adapters: HashMap<String, AdapterEntry>,
    lru: VecDeque<String>,
    load_notifiers: HashMap<String, Arc<Notify>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedRegistry {
    version: u32,
    adapters: Vec<PersistedAdapter>,
}

impl Default for PersistedRegistry {
    fn default() -> Self {
        Self {
            version: 1,
            adapters: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedAdapter {
    id: String,
    version: Option<String>,
    adapter_path: String,
    config_path: String,
    checksum_sha256: Option<String>,
    pinned: bool,
    config: AdapterConfig,
    artifact_refs: Vec<String>,
    created_at: SystemTime,
    updated_at: SystemTime,
}

pub struct LoRAManager {
    inner: RwLock<LoRAManagerInner>,
    expected_base_model: Option<String>,
    max_active_loras: usize,
    max_lora_rank: usize,
    lora_mode: String,
    registry_path: PathBuf,
    artifact_root: PathBuf,
}

impl LoRAManager {
    pub fn new(
        expected_base_model: Option<String>,
        max_active_loras: usize,
        max_lora_rank: usize,
        lora_mode: impl Into<String>,
    ) -> Self {
        Self::new_with_registry_path(
            expected_base_model,
            max_active_loras,
            max_lora_rank,
            lora_mode,
            registry_path_from_env(),
        )
    }

    pub fn new_with_registry_path(
        expected_base_model: Option<String>,
        max_active_loras: usize,
        max_lora_rank: usize,
        lora_mode: impl Into<String>,
        registry_path: PathBuf,
    ) -> Self {
        let manager = Self {
            inner: RwLock::new(LoRAManagerInner {
                adapters: HashMap::new(),
                lru: VecDeque::new(),
                load_notifiers: HashMap::new(),
            }),
            expected_base_model,
            max_active_loras,
            max_lora_rank,
            lora_mode: lora_mode.into(),
            artifact_root: artifact_root_from_env()
                .unwrap_or_else(|| default_artifact_root_for_registry(&registry_path)),
            registry_path,
        };
        manager.restore_registry_best_effort();
        manager
    }

    pub fn max_active_loras(&self) -> usize {
        self.max_active_loras
    }

    pub fn lora_mode(&self) -> &str {
        &self.lora_mode
    }

    pub fn loaded_loras(&self) -> usize {
        let inner = self.inner.read();
        inner
            .adapters
            .values()
            .filter(|entry| entry.state == AdapterLoadState::Loaded)
            .count()
    }

    pub fn list(&self) -> Vec<AdapterRecord> {
        let inner = self.inner.read();
        let mut records = inner
            .adapters
            .values()
            .map(|entry| entry.to_record())
            .collect::<Vec<_>>();
        records.sort_by(|a, b| a.id.cmp(&b.id));
        records
    }

    pub fn get(&self, id: &str) -> Option<AdapterRecord> {
        let inner = self.inner.read();
        inner.adapters.get(id).map(|entry| entry.to_record())
    }

    pub fn status(&self) -> AdapterStatusResponse {
        let loaded = self.loaded_loras();
        AdapterStatusResponse {
            max_active_loras: self.max_active_loras,
            loaded_loras: loaded,
            lora_mode: self.lora_mode.clone(),
            slots_used: loaded,
            slots_free: self.max_active_loras.saturating_sub(loaded),
            adapters: self.list(),
        }
    }

    fn resolve_registration_paths(
        &self,
        request: &RegisterAdapterRequest,
    ) -> Result<(PathBuf, PathBuf)> {
        let adapter_path = PathBuf::from(request.adapter_path.trim());
        let requested_config = request
            .config_path
            .as_ref()
            .map(|path| PathBuf::from(path.trim()));

        if adapter_path.exists() {
            if !adapter_path.is_file() {
                candle_core::bail!("adapter_path must be a file: {}", adapter_path.display());
            }
            let config_path = requested_config.unwrap_or_else(|| {
                adapter_path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join("adapter_config.json")
            });
            if config_path.exists() {
                if !config_path.is_file() {
                    candle_core::bail!("config_path must be a file: {}", config_path.display());
                }
                return Ok((adapter_path, config_path));
            }

            if request.artifact_refs.is_empty() {
                candle_core::bail!("config_path does not exist: {}", config_path.display());
            }

            let stage_dir = self.stage_artifact_dir(&request.id, request.version.as_deref());
            fs::create_dir_all(&stage_dir)?;
            let resolved_config = self.materialize_config_from_refs(
                &request.artifact_refs,
                stage_dir.join("adapter_config.json"),
            )?;
            return Ok((adapter_path, resolved_config));
        }

        if request.artifact_refs.is_empty() {
            candle_core::bail!("adapter_path does not exist: {}", adapter_path.display());
        }

        let stage_dir = self.stage_artifact_dir(&request.id, request.version.as_deref());
        fs::create_dir_all(&stage_dir)?;

        let staged_adapter = self.materialize_adapter_from_refs(
            &request.artifact_refs,
            stage_dir.join("adapter.safetensors"),
        )?;
        let config_target =
            requested_config.unwrap_or_else(|| stage_dir.join("adapter_config.json"));
        let staged_config = if config_target.exists() {
            if !config_target.is_file() {
                candle_core::bail!("config_path must be a file: {}", config_target.display());
            }
            config_target
        } else {
            self.materialize_config_from_refs(&request.artifact_refs, config_target)?
        };

        Ok((staged_adapter, staged_config))
    }

    fn stage_artifact_dir(&self, adapter_id: &str, version: Option<&str>) -> PathBuf {
        let safe_id = sanitize_path_component(adapter_id);
        let safe_version = sanitize_path_component(version.unwrap_or("latest"));
        self.artifact_root.join(safe_id).join(safe_version)
    }

    fn materialize_adapter_from_refs(&self, refs: &[String], dest: PathBuf) -> Result<PathBuf> {
        if dest.exists() {
            return Ok(dest);
        }
        let Some(source) = select_artifact_source(refs, ArtifactKind::Adapter) else {
            candle_core::bail!(
                "No adapter artifact found in artifact_refs (expected .safetensors or named adapter)"
            );
        };
        materialize_artifact(&source, &dest)?;
        Ok(dest)
    }

    fn materialize_config_from_refs(&self, refs: &[String], dest: PathBuf) -> Result<PathBuf> {
        if dest.exists() {
            return Ok(dest);
        }
        let Some(source) = select_artifact_source(refs, ArtifactKind::Config) else {
            candle_core::bail!(
                "No adapter config artifact found in artifact_refs (expected adapter_config.json or named config)"
            );
        };
        materialize_artifact(&source, &dest)?;
        Ok(dest)
    }

    pub fn register(&self, request: RegisterAdapterRequest) -> Result<AdapterRecord> {
        let (adapter_path, config_path) = self.resolve_registration_paths(&request)?;

        let mut config: AdapterConfig = serde_json::from_str(&fs::read_to_string(&config_path)?)
            .map_err(candle_core::Error::msg)?;
        if request.base_model.is_some() {
            config.base_model = request.base_model;
        }

        self.validate_base_model_compatibility(config.base_model.as_deref())?;

        if let Some(rank) = config.rank {
            if rank > self.max_lora_rank {
                candle_core::bail!(
                    "Adapter rank {} exceeds max_lora_rank {}",
                    rank,
                    self.max_lora_rank
                );
            }
        }

        let expected_checksum = request.checksum_sha256.clone();
        if let Some(expected) = expected_checksum.as_deref() {
            let actual = compute_sha256(&adapter_path)?;
            if !actual.eq_ignore_ascii_case(expected) {
                candle_core::bail!(
                    "checksum mismatch for {}: expected {}, got {}",
                    adapter_path.display(),
                    expected,
                    actual
                );
            }
        }

        // Validate tensor shapes and module pairing at registration time.
        let modules = parse_lora_modules(&adapter_path, &config, self.max_lora_rank)?;
        if modules.is_empty() {
            candle_core::bail!("No LoRA module weights found in {}", adapter_path.display());
        }

        let now = SystemTime::now();
        let (record, snapshot) = {
            let mut inner = self.inner.write();
            if let Some(existing) = inner.adapters.get(&request.id) {
                let existing_version = existing
                    .version
                    .clone()
                    .unwrap_or_else(|| "<none>".to_string());
                let requested_version = request
                    .version
                    .clone()
                    .unwrap_or_else(|| "<none>".to_string());
                candle_core::bail!(
                    "Adapter '{}' already exists (existing version {}, requested version {}). Adapter ids are immutable.",
                    request.id,
                    existing_version,
                    requested_version
                );
            }

            let entry = AdapterEntry {
                id: request.id.clone(),
                version: request.version,
                adapter_path,
                config_path,
                checksum_sha256: expected_checksum,
                pinned: request.pinned,
                state: AdapterLoadState::Registered,
                config,
                artifact_refs: request.artifact_refs,
                loaded: None,
                last_error: None,
                created_at: now,
                updated_at: now,
            };

            let record = entry.to_record();
            inner.adapters.insert(request.id, entry);
            (record, snapshot_registry(&inner))
        };

        self.persist_registry_best_effort(snapshot);
        Ok(record)
    }

    pub fn enqueue_load(
        &self,
        id: &str,
        pin_override: Option<bool>,
    ) -> Result<(AdapterRecord, bool)> {
        let (record, should_start, snapshot) = {
            let mut inner = self.inner.write();
            let (record, should_start) = {
                let entry = inner
                    .adapters
                    .get_mut(id)
                    .ok_or_else(|| candle_core::Error::msg(format!("Adapter {} not found", id)))?;

                if let Some(pin_override) = pin_override {
                    entry.pinned = pin_override;
                }

                let should_start = !matches!(
                    entry.state,
                    AdapterLoadState::Loaded | AdapterLoadState::Loading
                );
                if should_start {
                    entry.state = AdapterLoadState::Loading;
                    entry.last_error = None;
                    entry.updated_at = SystemTime::now();
                }

                (entry.to_record(), should_start)
            };
            if should_start {
                inner
                    .load_notifiers
                    .entry(id.to_string())
                    .or_insert_with(|| Arc::new(Notify::new()));
            }
            (record, should_start, snapshot_registry(&inner))
        };

        self.persist_registry_best_effort(snapshot);
        Ok((record, should_start))
    }

    pub async fn ensure_loaded_async(&self, id: &str) -> Result<()> {
        let _ = self.load_async(id, None).await?;
        Ok(())
    }

    pub async fn load_async(&self, id: &str, pin_override: Option<bool>) -> Result<AdapterRecord> {
        enum Plan {
            Ready(AdapterRecord),
            Wait(Arc<Notify>),
            Start {
                notify: Arc<Notify>,
                adapter_path: PathBuf,
                config: AdapterConfig,
            },
        }

        loop {
            let (plan, snapshot) = {
                let mut inner = self.inner.write();
                let mut should_touch_lru = false;
                let mut start_payload: Option<(PathBuf, AdapterConfig)> = None;
                let base_plan = {
                    let entry = inner.adapters.get_mut(id).ok_or_else(|| {
                        candle_core::Error::msg(format!("Adapter {} not found", id))
                    })?;

                    if let Some(pin_override) = pin_override {
                        entry.pinned = pin_override;
                    }

                    match entry.state {
                        AdapterLoadState::Loaded => {
                            entry.updated_at = SystemTime::now();
                            if let Some(loaded) = entry.loaded.as_mut() {
                                loaded.last_used = SystemTime::now();
                            }
                            should_touch_lru = true;
                            Plan::Ready(entry.to_record())
                        }
                        AdapterLoadState::Loading => Plan::Wait(Arc::new(Notify::new())),
                        _ => {
                            entry.state = AdapterLoadState::Loading;
                            entry.last_error = None;
                            entry.updated_at = SystemTime::now();
                            start_payload =
                                Some((entry.adapter_path.clone(), entry.config.clone()));
                            Plan::Wait(Arc::new(Notify::new()))
                        }
                    }
                };

                if should_touch_lru {
                    Self::touch_lru_locked(&mut inner, id);
                }

                let plan = match base_plan {
                    Plan::Ready(record) => Plan::Ready(record),
                    Plan::Wait(_) => {
                        let notify = inner
                            .load_notifiers
                            .entry(id.to_string())
                            .or_insert_with(|| Arc::new(Notify::new()))
                            .clone();
                        if let Some((adapter_path, config)) = start_payload {
                            Plan::Start {
                                notify,
                                adapter_path,
                                config,
                            }
                        } else {
                            Plan::Wait(notify)
                        }
                    }
                    Plan::Start { .. } => unreachable!(),
                };

                (plan, snapshot_registry(&inner))
            };

            self.persist_registry_best_effort(snapshot);

            match plan {
                Plan::Ready(record) => return Ok(record),
                Plan::Wait(notify) => {
                    notify.notified().await;
                    continue;
                }
                Plan::Start {
                    notify,
                    adapter_path,
                    config,
                } => {
                    let max_lora_rank = self.max_lora_rank;
                    let load_result = tokio::task::spawn_blocking(move || {
                        parse_lora_modules(&adapter_path, &config, max_lora_rank)
                    })
                    .await
                    .map_err(|err| {
                        candle_core::Error::msg(format!("Load task join error: {err}"))
                    })?;

                    let (finalize_result, snapshot) = {
                        let mut inner = self.inner.write();
                        let finalize_result = if !inner.adapters.contains_key(id) {
                            Err(candle_core::Error::msg("Missing adapter"))
                        } else {
                            match load_result {
                                Ok(modules) => {
                                    let mut slot_error = None;
                                    while inner
                                        .adapters
                                        .values()
                                        .filter(|entry| entry.state == AdapterLoadState::Loaded)
                                        .count()
                                        >= self.max_active_loras
                                    {
                                        if !self.evict_one_lru_locked(&mut inner)? {
                                            slot_error = Some(candle_core::Error::msg(
                                                "No evictable LoRA slots available (all loaded adapters are pinned)",
                                            ));
                                            break;
                                        }
                                    }

                                    if let Some(err) = slot_error {
                                        let entry = inner.adapters.get_mut(id).unwrap();
                                        entry.loaded = None;
                                        entry.state = AdapterLoadState::Failed;
                                        entry.last_error = Some(err.to_string());
                                        entry.updated_at = SystemTime::now();
                                        Err(err)
                                    } else {
                                        let record = {
                                            let entry = inner.adapters.get_mut(id).unwrap();
                                            entry.loaded = Some(LoadedAdapter {
                                                modules,
                                                loaded_at: SystemTime::now(),
                                                last_used: SystemTime::now(),
                                            });
                                            entry.state = AdapterLoadState::Loaded;
                                            entry.last_error = None;
                                            entry.updated_at = SystemTime::now();
                                            entry.to_record()
                                        };
                                        Self::touch_lru_locked(&mut inner, id);
                                        Ok(record)
                                    }
                                }
                                Err(err) => {
                                    let entry = inner.adapters.get_mut(id).unwrap();
                                    entry.loaded = None;
                                    entry.state = AdapterLoadState::Failed;
                                    entry.last_error = Some(err.to_string());
                                    entry.updated_at = SystemTime::now();
                                    Err(err)
                                }
                            }
                        };

                        inner.load_notifiers.remove(id);
                        (finalize_result, snapshot_registry(&inner))
                    };

                    self.persist_registry_best_effort(snapshot);

                    notify.notify_waiters();
                    return finalize_result;
                }
            }
        }
    }

    pub fn unload(&self, id: &str) -> Result<AdapterRecord> {
        let (record, snapshot) = {
            let mut inner = self.inner.write();
            let record = {
                let entry = inner
                    .adapters
                    .get_mut(id)
                    .ok_or_else(|| candle_core::Error::msg(format!("Adapter {} not found", id)))?;

                entry.state = AdapterLoadState::Evicting;
                entry.updated_at = SystemTime::now();

                entry.loaded = None;
                entry.state = AdapterLoadState::Registered;
                entry.last_error = None;
                entry.updated_at = SystemTime::now();
                entry.to_record()
            };
            inner.lru.retain(|adapter_id| adapter_id != id);
            if let Some(notify) = inner.load_notifiers.remove(id) {
                notify.notify_waiters();
            }
            (record, snapshot_registry(&inner))
        };

        self.persist_registry_best_effort(snapshot);
        Ok(record)
    }

    pub fn touch_usage(&self, id: &str) {
        let mut inner = self.inner.write();
        let should_touch_lru = {
            if let Some(entry) = inner.adapters.get_mut(id) {
                if let Some(loaded) = entry.loaded.as_mut() {
                    loaded.last_used = SystemTime::now();
                }
                true
            } else {
                false
            }
        };
        if should_touch_lru {
            Self::touch_lru_locked(&mut inner, id);
        }
    }

    pub fn compute_delta(
        &self,
        adapter_id: &str,
        module_name: &str,
        x: &Tensor,
        out_dtype: DType,
        shard: Option<ShardSpec>,
    ) -> Result<Option<Tensor>> {
        let normalized_module = normalize_module_name(module_name);

        let module = {
            let inner = self.inner.read();
            let entry = match inner.adapters.get(adapter_id) {
                Some(entry) => entry,
                None => return Ok(None),
            };
            if entry.state != AdapterLoadState::Loaded {
                return Ok(None);
            }
            let loaded = match entry.loaded.as_ref() {
                Some(loaded) => loaded,
                None => return Ok(None),
            };
            resolve_module(&loaded.modules, &normalized_module).cloned()
        };

        let Some(module) = module else {
            return Ok(None);
        };

        self.touch_usage(adapter_id);

        let mut a = module.a.to_device(x.device())?;
        let mut b = module.b.to_device(x.device())?;

        if a.dtype() != x.dtype() {
            a = a.to_dtype(x.dtype())?;
        }
        if b.dtype() != x.dtype() {
            b = b.to_dtype(x.dtype())?;
        }

        if let Some(shard) = shard {
            match shard {
                ShardSpec::Column { rank, world_size } => {
                    if world_size > 1 {
                        let out_dim = b.dim(0)?;
                        if out_dim % world_size == 0 {
                            let chunk = out_dim / world_size;
                            b = b.narrow(0, rank * chunk, chunk)?;
                        }
                    }
                }
                ShardSpec::Row { rank, world_size } => {
                    if world_size > 1 {
                        let in_dim = a.dim(1)?;
                        if in_dim % world_size == 0 {
                            let chunk = in_dim / world_size;
                            a = a.narrow(1, rank * chunk, chunk)?;
                        }
                    }
                }
            }
        }

        let x_in = x
            .dims()
            .last()
            .copied()
            .ok_or_else(|| candle_core::Error::msg("Invalid input rank for LoRA"))?;
        if x_in != a.dim(1)? {
            candle_core::bail!(
                "LoRA input dim mismatch for module {}: x={}, lora_a_in={}",
                module_name,
                x_in,
                a.dim(1)?
            );
        }

        let mut delta = x.matmul(&a.t()?)?;
        delta = delta.matmul(&b.t()?)?;

        if (module.scale - 1.0).abs() > f64::EPSILON {
            let scale = Tensor::new(module.scale as f32, x.device())?.to_dtype(delta.dtype())?;
            delta = delta.broadcast_mul(&scale)?;
        }

        if delta.dtype() != out_dtype {
            delta = delta.to_dtype(out_dtype)?;
        }

        Ok(Some(delta))
    }

    fn evict_one_lru_locked(&self, inner: &mut LoRAManagerInner) -> Result<bool> {
        let max_scan = inner.lru.len();
        let mut scanned = 0usize;
        while scanned < max_scan {
            let Some(candidate) = inner.lru.pop_front() else {
                break;
            };
            scanned += 1;
            let Some(entry) = inner.adapters.get_mut(&candidate) else {
                continue;
            };
            if entry.state != AdapterLoadState::Loaded {
                continue;
            }
            if entry.pinned {
                inner.lru.push_back(candidate);
                continue;
            }
            entry.state = AdapterLoadState::Evicting;
            entry.updated_at = SystemTime::now();
            entry.loaded = None;
            entry.state = AdapterLoadState::Registered;
            entry.updated_at = SystemTime::now();
            return Ok(true);
        }
        Ok(false)
    }

    fn touch_lru_locked(inner: &mut LoRAManagerInner, id: &str) {
        inner.lru.retain(|adapter_id| adapter_id != id);
        inner.lru.push_back(id.to_string());
    }

    fn validate_base_model_compatibility(&self, adapter_base_model: Option<&str>) -> Result<()> {
        let Some(expected) = self.expected_base_model.as_deref() else {
            return Ok(());
        };
        let Some(adapter_base_model) = adapter_base_model else {
            return Ok(());
        };

        if model_ids_compatible(expected, adapter_base_model) {
            return Ok(());
        }

        candle_core::bail!(
            "Adapter base model '{}' is not compatible with runtime model '{}'.",
            adapter_base_model,
            expected
        );
    }

    fn restore_registry_best_effort(&self) {
        if !self.registry_path.exists() {
            return;
        }

        let load_result = (|| -> Result<PersistedRegistry> {
            let text = fs::read_to_string(&self.registry_path)?;
            let registry: PersistedRegistry =
                serde_json::from_str(&text).map_err(candle_core::Error::msg)?;
            Ok(registry)
        })();

        match load_result {
            Ok(registry) => {
                let mut inner = self.inner.write();
                for persisted in registry.adapters {
                    if inner.adapters.contains_key(&persisted.id) {
                        continue;
                    }
                    let entry = AdapterEntry {
                        id: persisted.id.clone(),
                        version: persisted.version,
                        adapter_path: PathBuf::from(persisted.adapter_path),
                        config_path: PathBuf::from(persisted.config_path),
                        checksum_sha256: persisted.checksum_sha256,
                        pinned: persisted.pinned,
                        state: AdapterLoadState::Registered,
                        config: persisted.config,
                        artifact_refs: persisted.artifact_refs,
                        loaded: None,
                        last_error: None,
                        created_at: persisted.created_at,
                        updated_at: persisted.updated_at,
                    };
                    inner.adapters.insert(persisted.id, entry);
                }
            }
            Err(err) => {
                tracing::warn!(
                    "Failed to restore adapter registry from {}: {}",
                    self.registry_path.display(),
                    err
                );
            }
        }
    }

    fn persist_registry_best_effort(&self, snapshot: PersistedRegistry) {
        if let Err(err) = persist_registry(&self.registry_path, &snapshot) {
            tracing::warn!(
                "Failed to persist adapter registry to {}: {}",
                self.registry_path.display(),
                err
            );
        }
    }
}

fn snapshot_registry(inner: &LoRAManagerInner) -> PersistedRegistry {
    let mut adapters = inner
        .adapters
        .values()
        .map(|entry| PersistedAdapter {
            id: entry.id.clone(),
            version: entry.version.clone(),
            adapter_path: entry.adapter_path.display().to_string(),
            config_path: entry.config_path.display().to_string(),
            checksum_sha256: entry.checksum_sha256.clone(),
            pinned: entry.pinned,
            config: entry.config.clone(),
            artifact_refs: entry.artifact_refs.clone(),
            created_at: entry.created_at,
            updated_at: entry.updated_at,
        })
        .collect::<Vec<_>>();
    adapters.sort_by(|a, b| a.id.cmp(&b.id));
    PersistedRegistry {
        version: 1,
        adapters,
    }
}

fn persist_registry(path: &Path, snapshot: &PersistedRegistry) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension("tmp");
    let bytes = serde_json::to_vec_pretty(snapshot).map_err(candle_core::Error::msg)?;
    fs::write(&tmp_path, bytes)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArtifactKind {
    Adapter,
    Config,
}

fn select_artifact_source(refs: &[String], kind: ArtifactKind) -> Option<String> {
    let mut ranked = Vec::<(usize, String)>::new();
    for item in refs {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = if let Some((_, uri)) = trimmed.split_once('=') {
            uri.trim()
        } else {
            trimmed
        };
        if value.is_empty() {
            continue;
        }
        let lower = value.to_lowercase();
        let score = match kind {
            ArtifactKind::Adapter => {
                if lower.ends_with(".safetensors") {
                    100
                } else if lower.contains("adapter") {
                    50
                } else {
                    0
                }
            }
            ArtifactKind::Config => {
                if lower.ends_with("adapter_config.json") {
                    100
                } else if lower.ends_with(".json") || lower.contains("config") {
                    50
                } else {
                    0
                }
            }
        };
        if score > 0 {
            ranked.push((score, value.to_string()));
        }
    }

    ranked.sort_by(|a, b| b.0.cmp(&a.0));
    ranked.first().map(|(_, value)| value.clone())
}

fn materialize_artifact(source: &str, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }

    if source.starts_with("http://") || source.starts_with("https://") {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(candle_core::Error::msg)?;
        let response = client
            .get(source)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .map_err(candle_core::Error::msg)?;
        let bytes = response.bytes().map_err(candle_core::Error::msg)?;
        fs::write(destination, &bytes)?;
        return Ok(());
    }

    let source_path = if let Some(path) = source.strip_prefix("file://") {
        PathBuf::from(path)
    } else {
        PathBuf::from(source)
    };
    if !source_path.exists() {
        candle_core::bail!("artifact source does not exist: {}", source_path.display());
    }
    if !source_path.is_file() {
        candle_core::bail!("artifact source must be a file: {}", source_path.display());
    }
    fs::copy(source_path, destination)?;
    Ok(())
}

fn sanitize_path_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "default".to_string()
    } else {
        trimmed
    }
}

fn registry_path_from_env() -> PathBuf {
    if let Ok(path) = std::env::var("CANDLE_VLLM_ADAPTER_REGISTRY_PATH") {
        return PathBuf::from(path);
    }
    PathBuf::from(".candle-vllm/adapter-registry.json")
}

fn artifact_root_from_env() -> Option<PathBuf> {
    std::env::var("CANDLE_VLLM_ADAPTER_ARTIFACT_DIR")
        .ok()
        .map(PathBuf::from)
}

fn default_artifact_root_for_registry(registry_path: &Path) -> PathBuf {
    let parent = registry_path.parent().unwrap_or(Path::new(".candle-vllm"));
    parent.join("artifacts")
}

fn resolve_module<'a>(
    modules: &'a HashMap<String, LoRAModule>,
    normalized_module: &str,
) -> Option<&'a LoRAModule> {
    if let Some(module) = modules.get(normalized_module) {
        return Some(module);
    }

    let with_model_prefix = format!("model.{}", normalized_module);
    if let Some(module) = modules.get(&with_model_prefix) {
        return Some(module);
    }

    modules.iter().find_map(|(key, module)| {
        if key.ends_with(normalized_module) || normalized_module.ends_with(key.as_str()) {
            Some(module)
        } else {
            None
        }
    })
}

fn compute_sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn parse_lora_modules(
    path: &Path,
    config: &AdapterConfig,
    max_lora_rank: usize,
) -> Result<HashMap<String, LoRAModule>> {
    let bytes = fs::read(path)?;
    let safetensors = SafeTensors::deserialize(&bytes).map_err(candle_core::Error::msg)?;

    let mut a_map = HashMap::<String, Tensor>::new();
    let mut b_map = HashMap::<String, Tensor>::new();

    for name in safetensors.names() {
        if let Some(base) = name.strip_suffix(".lora_A.weight") {
            let tensor =
                tensor_from_view(safetensors.tensor(name).map_err(candle_core::Error::msg)?)?;
            a_map.insert(normalize_module_name(base), tensor);
        } else if let Some(base) = name.strip_suffix(".lora_B.weight") {
            let tensor =
                tensor_from_view(safetensors.tensor(name).map_err(candle_core::Error::msg)?)?;
            b_map.insert(normalize_module_name(base), tensor);
        }
    }

    let mut modules = HashMap::<String, LoRAModule>::new();
    for (module_name, a) in a_map {
        let b = b_map.remove(&module_name).ok_or_else(|| {
            candle_core::Error::msg(format!("Missing LoRA B weight for {}", module_name))
        })?;

        if a.rank() != 2 || b.rank() != 2 {
            candle_core::bail!("LoRA tensors must be rank-2 for module {}", module_name);
        }

        let rank = a.dim(0)?;
        let b_rank = b.dim(1)?;

        if rank != b_rank {
            candle_core::bail!(
                "LoRA rank mismatch for module {}: A rank {} != B rank {}",
                module_name,
                rank,
                b_rank
            );
        }

        if rank > max_lora_rank {
            candle_core::bail!(
                "LoRA rank {} exceeds max_lora_rank {} for module {}",
                rank,
                max_lora_rank,
                module_name
            );
        }

        if let Some(config_rank) = config.rank {
            if config_rank != rank {
                candle_core::bail!(
                    "LoRA rank mismatch with adapter_config for module {}: config rank {} != tensor rank {}",
                    module_name,
                    config_rank,
                    rank
                );
            }
        }

        let alpha = config.alpha.unwrap_or(rank as f64);
        let scale = alpha / rank as f64;

        modules.insert(module_name, LoRAModule { a, b, scale });
    }

    Ok(modules)
}

fn tensor_from_view(view: safetensors::tensor::TensorView<'_>) -> Result<Tensor> {
    let shape = view.shape();
    if shape.len() != 2 {
        candle_core::bail!("Expected rank-2 tensor in LoRA safetensors, got shape {shape:?}");
    }
    let rows = shape[0];
    let cols = shape[1];

    let data = view.data();
    let decoded = match view.dtype() {
        safetensors::Dtype::F32 => decode_f32(data)?,
        safetensors::Dtype::F16 => decode_f16(data),
        safetensors::Dtype::BF16 => decode_bf16(data),
        other => candle_core::bail!("Unsupported LoRA tensor dtype: {other:?}"),
    };

    Tensor::from_vec(decoded, (rows, cols), &Device::Cpu)
}

fn decode_f32(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % 4 != 0 {
        candle_core::bail!("Invalid f32 byte length in safetensors: {}", data.len());
    }
    Ok(data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn decode_f16(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn decode_bf16(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn normalize_module_name(name: &str) -> String {
    let mut normalized = name.trim().to_string();
    for prefix in ["base_model.model.", "base_model.", "model.model."] {
        if normalized.starts_with(prefix) {
            normalized = normalized[prefix.len()..].to_string();
        }
    }
    normalized
}

fn normalize_model_id(model_id: &str) -> String {
    let raw = model_id.trim().trim_end_matches('/');
    let tail = raw.rsplit('/').next().unwrap_or(raw).to_ascii_lowercase();
    let mut normalized = String::with_capacity(tail.len());
    let mut prev_dash = false;
    for ch in tail.chars() {
        let mapped = if ch.is_ascii_alphanumeric() {
            ch
        } else if ch == '_' || ch == '.' || ch == '-' {
            '-'
        } else {
            '-'
        };
        if mapped == '-' {
            if !prev_dash {
                normalized.push(mapped);
            }
            prev_dash = true;
        } else {
            prev_dash = false;
            normalized.push(mapped);
        }
    }
    normalized.trim_matches('-').to_string()
}

fn normalize_model_family(model_id: &str) -> String {
    let mut family = normalize_model_id(model_id);
    for suffix in ["-instruct", "-chat"] {
        if family.ends_with(suffix) {
            family.truncate(family.len() - suffix.len());
            break;
        }
    }
    family
}

fn model_ids_compatible(expected: &str, candidate: &str) -> bool {
    let expected = normalize_model_family(expected);
    let candidate = normalize_model_family(candidate);
    expected == candidate || expected.contains(&candidate) || candidate.contains(&expected)
}

static GLOBAL_LORA_MANAGERS: OnceLock<RwLock<Vec<Arc<LoRAManager>>>> = OnceLock::new();

thread_local! {
    static ACTIVE_ADAPTER_ID: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub fn set_global_lora_manager(manager: Arc<LoRAManager>) {
    let registry = GLOBAL_LORA_MANAGERS.get_or_init(|| RwLock::new(Vec::new()));
    let mut managers = registry.write();
    if managers
        .iter()
        .any(|existing| Arc::ptr_eq(existing, &manager))
    {
        return;
    }
    managers.push(manager);
}

pub fn global_lora_manager() -> Option<Arc<LoRAManager>> {
    GLOBAL_LORA_MANAGERS
        .get()
        .and_then(|registry| registry.read().first().cloned())
}

pub fn global_lora_manager_for_adapter(adapter_id: &str) -> Option<Arc<LoRAManager>> {
    GLOBAL_LORA_MANAGERS.get().and_then(|registry| {
        registry
            .read()
            .iter()
            .find(|manager| manager.get(adapter_id).is_some())
            .cloned()
    })
}

pub fn set_active_adapter(adapter_id: Option<String>) {
    ACTIVE_ADAPTER_ID.with(|active| {
        *active.borrow_mut() = adapter_id;
    });
}

pub fn clear_active_adapter() {
    set_active_adapter(None)
}

pub fn active_adapter_id() -> Option<String> {
    ACTIVE_ADAPTER_ID.with(|active| active.borrow().clone())
}

pub fn compute_active_lora_delta(
    module_name: &str,
    x: &Tensor,
    out_dtype: DType,
    shard: Option<ShardSpec>,
) -> Result<Option<Tensor>> {
    let Some(adapter_id) = active_adapter_id() else {
        return Ok(None);
    };
    let Some(manager) = global_lora_manager_for_adapter(&adapter_id).or_else(global_lora_manager)
    else {
        return Ok(None);
    };
    manager.compute_delta(&adapter_id, module_name, x, out_dtype, shard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{tensor::TensorView, Dtype};
    use tempfile::TempDir;

    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes().to_vec())
            .collect()
    }

    fn write_adapter_files(dir: &TempDir, id: &str, include_b: bool) -> (PathBuf, PathBuf) {
        let adapter_path = dir.path().join(format!("{id}.safetensors"));
        let config_path = dir.path().join(format!("{id}.json"));

        let a_values = vec![0.1f32, 0.2, 0.3, 0.4];
        let b_values = vec![0.5f32, 0.6, 0.7, 0.8];
        let a_bytes = f32_to_bytes(&a_values);
        let b_bytes = f32_to_bytes(&b_values);

        let a_view = TensorView::new(Dtype::F32, vec![2, 2], &a_bytes).unwrap();
        if include_b {
            let b_view = TensorView::new(Dtype::F32, vec![2, 2], &b_bytes).unwrap();
            safetensors::tensor::serialize_to_file(
                vec![
                    ("model.layers.0.self_attn.q_proj.lora_A.weight", a_view),
                    ("model.layers.0.self_attn.q_proj.lora_B.weight", b_view),
                ],
                &None,
                &adapter_path,
            )
            .unwrap();
        } else {
            safetensors::tensor::serialize_to_file(
                vec![("model.layers.0.self_attn.q_proj.lora_A.weight", a_view)],
                &None,
                &adapter_path,
            )
            .unwrap();
        }

        fs::write(
            &config_path,
            r#"{"base_model":"qwen3-4b","r":2,"lora_alpha":2.0,"target_modules":["q_proj"]}"#,
        )
        .unwrap();

        (adapter_path, config_path)
    }

    #[test]
    fn registration_validation_rejects_shape_mismatch() {
        let dir = TempDir::new().unwrap();
        let (adapter_path, config_path) = write_adapter_files(&dir, "bad", false);
        let manager = LoRAManager::new_with_registry_path(
            Some("qwen3-4b".to_string()),
            8,
            64,
            "fallback",
            dir.path().join("registry.json"),
        );

        let result = manager.register(RegisterAdapterRequest {
            id: "bad".to_string(),
            adapter_path: adapter_path.display().to_string(),
            config_path: Some(config_path.display().to_string()),
            version: Some("v1".to_string()),
            checksum_sha256: None,
            pinned: false,
            base_model: None,
            artifact_refs: Vec::new(),
            scope: None,
            backend: None,
        });

        assert!(result.is_err());
    }

    #[test]
    fn registration_stages_artifacts_when_adapter_path_is_missing() {
        let dir = TempDir::new().unwrap();
        let (source_adapter, source_config) = write_adapter_files(&dir, "stage-src", true);
        let missing_adapter = dir.path().join("missing").join("adapter.safetensors");
        let manager = LoRAManager::new_with_registry_path(
            Some("qwen3-4b".to_string()),
            8,
            64,
            "fallback",
            dir.path().join("registry.json"),
        );

        let record = manager
            .register(RegisterAdapterRequest {
                id: "stage".to_string(),
                adapter_path: missing_adapter.display().to_string(),
                config_path: None,
                version: Some("v1".to_string()),
                checksum_sha256: None,
                pinned: false,
                base_model: None,
                artifact_refs: vec![
                    source_adapter.display().to_string(),
                    source_config.display().to_string(),
                ],
                scope: None,
                backend: None,
            })
            .unwrap();

        assert!(PathBuf::from(&record.adapter_path).exists());
        assert!(PathBuf::from(&record.config_path).exists());
        assert!(record.adapter_path.contains("artifacts"));
    }

    #[test]
    fn hot_load_unload_works_under_active_runtime() {
        let dir = TempDir::new().unwrap();
        let (adapter_path, config_path) = write_adapter_files(&dir, "ok", true);
        let manager = Arc::new(LoRAManager::new_with_registry_path(
            Some("qwen3-4b".to_string()),
            8,
            64,
            "fallback",
            dir.path().join("registry.json"),
        ));

        manager
            .register(RegisterAdapterRequest {
                id: "ok".to_string(),
                adapter_path: adapter_path.display().to_string(),
                config_path: Some(config_path.display().to_string()),
                version: Some("v1".to_string()),
                checksum_sha256: None,
                pinned: false,
                base_model: None,
                artifact_refs: Vec::new(),
                scope: None,
                backend: None,
            })
            .unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            manager.ensure_loaded_async("ok").await.unwrap();
        });

        assert_eq!(manager.status().loaded_loras, 1);
        manager.unload("ok").unwrap();
        assert_eq!(manager.status().loaded_loras, 0);
    }

    #[test]
    fn slot_pressure_respects_pinning_and_reports_failure() {
        let dir = TempDir::new().unwrap();
        let (a_path, a_cfg) = write_adapter_files(&dir, "a", true);
        let (b_path, b_cfg) = write_adapter_files(&dir, "b", true);
        let manager = Arc::new(LoRAManager::new_with_registry_path(
            Some("qwen3-4b".to_string()),
            1,
            64,
            "fallback",
            dir.path().join("registry.json"),
        ));

        manager
            .register(RegisterAdapterRequest {
                id: "a".to_string(),
                adapter_path: a_path.display().to_string(),
                config_path: Some(a_cfg.display().to_string()),
                version: Some("v1".to_string()),
                checksum_sha256: None,
                pinned: true,
                base_model: None,
                artifact_refs: Vec::new(),
                scope: None,
                backend: None,
            })
            .unwrap();

        manager
            .register(RegisterAdapterRequest {
                id: "b".to_string(),
                adapter_path: b_path.display().to_string(),
                config_path: Some(b_cfg.display().to_string()),
                version: Some("v1".to_string()),
                checksum_sha256: None,
                pinned: false,
                base_model: None,
                artifact_refs: Vec::new(),
                scope: None,
                backend: None,
            })
            .unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            manager.ensure_loaded_async("a").await.unwrap();
            let err = manager.load_async("b", None).await.err().unwrap();
            assert!(err
                .to_string()
                .contains("No evictable LoRA slots available"));

            manager.load_async("a", Some(false)).await.unwrap();
            manager.ensure_loaded_async("b").await.unwrap();
        });

        assert_eq!(manager.status().loaded_loras, 1);
        assert_eq!(
            manager.get("a").unwrap().state,
            AdapterLoadState::Registered
        );
        assert_eq!(manager.get("b").unwrap().state, AdapterLoadState::Loaded);
    }

    #[test]
    fn concurrent_waiters_resume_after_single_load() {
        let dir = TempDir::new().unwrap();
        let (adapter_path, config_path) = write_adapter_files(&dir, "wait", true);
        let manager = Arc::new(LoRAManager::new_with_registry_path(
            Some("qwen3-4b".to_string()),
            8,
            64,
            "fallback",
            dir.path().join("registry.json"),
        ));

        manager
            .register(RegisterAdapterRequest {
                id: "wait".to_string(),
                adapter_path: adapter_path.display().to_string(),
                config_path: Some(config_path.display().to_string()),
                version: Some("v1".to_string()),
                checksum_sha256: None,
                pinned: false,
                base_model: None,
                artifact_refs: Vec::new(),
                scope: None,
                backend: None,
            })
            .unwrap();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            let m1 = manager.clone();
            let m2 = manager.clone();
            let t1 = tokio::spawn(async move { m1.ensure_loaded_async("wait").await });
            let t2 = tokio::spawn(async move { m2.ensure_loaded_async("wait").await });
            t1.await.unwrap().unwrap();
            t2.await.unwrap().unwrap();
        });

        assert_eq!(manager.get("wait").unwrap().state, AdapterLoadState::Loaded);
    }

    #[test]
    fn load_failure_isolated_from_other_adapters() {
        let dir = TempDir::new().unwrap();
        let (good_path, good_cfg) = write_adapter_files(&dir, "good", true);
        let (bad_path, bad_cfg) = write_adapter_files(&dir, "badload", true);

        let manager = Arc::new(LoRAManager::new_with_registry_path(
            Some("qwen3-4b".to_string()),
            8,
            64,
            "fallback",
            dir.path().join("registry.json"),
        ));

        manager
            .register(RegisterAdapterRequest {
                id: "good".to_string(),
                adapter_path: good_path.display().to_string(),
                config_path: Some(good_cfg.display().to_string()),
                version: Some("v1".to_string()),
                checksum_sha256: None,
                pinned: false,
                base_model: None,
                artifact_refs: Vec::new(),
                scope: None,
                backend: None,
            })
            .unwrap();

        manager
            .register(RegisterAdapterRequest {
                id: "badload".to_string(),
                adapter_path: bad_path.display().to_string(),
                config_path: Some(bad_cfg.display().to_string()),
                version: Some("v1".to_string()),
                checksum_sha256: None,
                pinned: false,
                base_model: None,
                artifact_refs: Vec::new(),
                scope: None,
                backend: None,
            })
            .unwrap();

        fs::remove_file(&bad_path).unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            let bad = manager.load_async("badload", None).await;
            assert!(bad.is_err());
            manager.ensure_loaded_async("good").await.unwrap();
        });

        assert_eq!(
            manager.get("badload").unwrap().state,
            AdapterLoadState::Failed
        );
        assert_eq!(manager.get("good").unwrap().state, AdapterLoadState::Loaded);
    }

    #[test]
    fn model_id_normalization_handles_qwen35_family_aliases() {
        assert!(super::model_ids_compatible(
            "Qwen/Qwen3.5-9B-Instruct",
            "qwen3_5-9b"
        ));
        assert!(super::model_ids_compatible(
            "qwen/qwen3_5.9b",
            "Qwen3.5-9B-Instruct"
        ));
        assert!(!super::model_ids_compatible(
            "qwen3.5-9b",
            "llama3.1-8b-instruct"
        ));
    }
}
