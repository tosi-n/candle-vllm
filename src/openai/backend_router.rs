use crate::openai::lora::{
    AdapterLoadState, AdapterRecord, AdapterStatusResponse, RegisterAdapterRequest,
};
use crate::openai::responses::APIError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingMode {
    LocalOnly,
    CloudOnly,
    Hybrid,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectedBackend {
    Local,
    Cloud(String),
}

impl SelectedBackend {
    pub fn label(&self) -> &str {
        match self {
            Self::Local => "local",
            Self::Cloud(id) => id.as_str(),
        }
    }

    pub fn cloud_id(&self) -> Option<&str> {
        match self {
            Self::Cloud(id) => Some(id.as_str()),
            Self::Local => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloudBackendConfig {
    pub id: String,
    pub base_url: String,
    #[serde(default = "default_backend_weight")]
    pub weight: f64,
}

fn default_backend_weight() -> f64 {
    1.0
}

impl CloudBackendConfig {
    pub fn parse_specs(specs: &[String]) -> Result<Vec<Self>, String> {
        let mut parsed = Vec::new();
        let mut seen = HashSet::new();

        for (idx, spec) in specs.iter().enumerate() {
            let raw = spec.trim();
            if raw.is_empty() {
                continue;
            }

            let (id, url) = if let Some((id, url)) = raw.split_once('=') {
                (id.trim().to_string(), url.trim().to_string())
            } else {
                (format!("cloud-{}", idx + 1), raw.to_string())
            };

            if id.is_empty() {
                return Err(format!("Invalid cloud backend spec '{}': missing id", raw));
            }
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Err(format!(
                    "Invalid cloud backend url '{}': must start with http:// or https://",
                    url
                ));
            }
            if !seen.insert(id.clone()) {
                return Err(format!("Duplicate cloud backend id '{}'", id));
            }

            parsed.push(Self {
                id,
                base_url: url.trim_end_matches('/').to_string(),
                weight: 1.0,
            });
        }
        Ok(parsed)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CloudBackendStatusSnapshot {
    pub id: String,
    pub base_url: String,
    pub outstanding: usize,
    pub latency_ewma_ms: f64,
    pub max_active_loras: Option<usize>,
    pub loaded_loras: Option<usize>,
    pub slots_free: Option<usize>,
    pub loaded_adapters: Vec<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendOperationResult {
    pub backend_id: String,
    pub ok: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CloudAdapterListResult {
    pub backend_id: String,
    pub adapters: Vec<AdapterRecord>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CloudAdapterGetResult {
    pub backend_id: String,
    pub adapter: Option<AdapterRecord>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
struct CloudBackendState {
    outstanding: usize,
    latency_ewma_ms: f64,
    last_error: Option<String>,
    last_status_refresh: Option<Instant>,
    max_active_loras: Option<usize>,
    loaded_loras: Option<usize>,
    slots_free: Option<usize>,
    loaded_adapters: HashSet<String>,
    loading_adapters: HashSet<String>,
    failed_adapters: HashSet<String>,
}

impl Default for CloudBackendState {
    fn default() -> Self {
        Self {
            outstanding: 0,
            latency_ewma_ms: 120.0,
            last_error: None,
            last_status_refresh: None,
            max_active_loras: None,
            loaded_loras: None,
            slots_free: None,
            loaded_adapters: HashSet::new(),
            loading_adapters: HashSet::new(),
            failed_adapters: HashSet::new(),
        }
    }
}

pub struct BackendRouter {
    client: reqwest::Client,
    cloud_backends: Vec<CloudBackendConfig>,
    cloud_states: RwLock<HashMap<String, CloudBackendState>>,
    local_outstanding: RwLock<usize>,
    local_latency_ewma_ms: RwLock<f64>,
    local_weight: f64,
    cloud_weight: f64,
    status_ttl: Duration,
    adapter_autosync: bool,
    runtime_model_id: Option<String>,
}

impl BackendRouter {
    pub fn new(
        cloud_backends: Vec<CloudBackendConfig>,
        local_weight: f64,
        cloud_weight: f64,
        status_ttl: Duration,
        adapter_autosync: bool,
        runtime_model_id: Option<String>,
    ) -> Result<Self, APIError> {
        let mut cloud_states = HashMap::new();
        for backend in &cloud_backends {
            cloud_states.insert(backend.id.clone(), CloudBackendState::default());
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(APIError::from)?;

        Ok(Self {
            client,
            cloud_backends,
            cloud_states: RwLock::new(cloud_states),
            local_outstanding: RwLock::new(0),
            local_latency_ewma_ms: RwLock::new(80.0),
            local_weight: if local_weight <= 0.0 {
                1.0
            } else {
                local_weight
            },
            cloud_weight: if cloud_weight <= 0.0 {
                1.0
            } else {
                cloud_weight
            },
            status_ttl: if status_ttl.is_zero() {
                Duration::from_secs(5)
            } else {
                status_ttl
            },
            adapter_autosync,
            runtime_model_id,
        })
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    pub fn has_cloud_backends(&self) -> bool {
        !self.cloud_backends.is_empty()
    }

    pub fn cloud_backend_ids(&self) -> Vec<String> {
        self.cloud_backends.iter().map(|b| b.id.clone()).collect()
    }

    pub fn get_cloud_backend(&self, id: &str) -> Option<CloudBackendConfig> {
        self.cloud_backends.iter().find(|b| b.id == id).cloned()
    }

    pub fn begin_local_request(&self) {
        let mut outstanding = self.local_outstanding.write();
        *outstanding += 1;
    }

    pub fn end_local_request(&self, started: Instant, success: bool) {
        {
            let mut outstanding = self.local_outstanding.write();
            if *outstanding > 0 {
                *outstanding -= 1;
            }
        }
        let mut latency = self.local_latency_ewma_ms.write();
        *latency = ewma(*latency, started.elapsed().as_secs_f64() * 1000.0);
        if !success {
            *latency = (*latency * 1.1).max(10.0);
        }
    }

    pub fn begin_cloud_request(&self, backend_id: &str) {
        let mut states = self.cloud_states.write();
        if let Some(state) = states.get_mut(backend_id) {
            state.outstanding += 1;
        }
    }

    pub fn end_cloud_request(&self, backend_id: &str, started: Instant, success: bool) {
        let mut states = self.cloud_states.write();
        if let Some(state) = states.get_mut(backend_id) {
            if state.outstanding > 0 {
                state.outstanding -= 1;
            }
            state.latency_ewma_ms = ewma(
                state.latency_ewma_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
            if !success {
                state.last_error = Some("last request failed".to_string());
            }
        }
    }

    pub async fn select_backend(
        &self,
        mode: RoutingMode,
        adapter_id: Option<&str>,
        local_status: &AdapterStatusResponse,
    ) -> Result<SelectedBackend, APIError> {
        if matches!(mode, RoutingMode::CloudOnly | RoutingMode::Hybrid) {
            self.refresh_cloud_statuses(false).await;
        }

        match mode {
            RoutingMode::LocalOnly => Ok(SelectedBackend::Local),
            RoutingMode::CloudOnly => self
                .pick_best_cloud(adapter_id)
                .map(|(id, _)| SelectedBackend::Cloud(id))
                .ok_or_else(|| {
                    APIError::new(
                        "Cloud mode requested but no cloud backends are configured.".to_string(),
                    )
                }),
            RoutingMode::Hybrid => {
                let local_score = self.score_local(local_status, adapter_id);
                let cloud = self.pick_best_cloud(adapter_id);
                if let Some((cloud_id, cloud_score)) = cloud {
                    if cloud_score < local_score {
                        Ok(SelectedBackend::Cloud(cloud_id))
                    } else {
                        Ok(SelectedBackend::Local)
                    }
                } else {
                    Ok(SelectedBackend::Local)
                }
            }
        }
    }

    pub async fn refresh_cloud_statuses(&self, force: bool) {
        if self.cloud_backends.is_empty() {
            return;
        }

        for backend in &self.cloud_backends {
            let _ = self.refresh_cloud_status(backend.clone(), force).await;
        }
    }

    async fn refresh_cloud_status(
        &self,
        backend: CloudBackendConfig,
        force: bool,
    ) -> Result<(), APIError> {
        let should_refresh = {
            let states = self.cloud_states.read();
            if let Some(state) = states.get(&backend.id) {
                if force {
                    true
                } else {
                    state
                        .last_status_refresh
                        .map(|ts| ts.elapsed() >= self.status_ttl)
                        .unwrap_or(true)
                }
            } else {
                true
            }
        };
        if !should_refresh {
            return Ok(());
        }

        let url = format!("{}/v1/adapters/status", backend.base_url);
        let started = Instant::now();
        let response = self.client.get(&url).send().await.map_err(APIError::from)?;
        let status: AdapterStatusResponse = response
            .error_for_status()
            .map_err(APIError::from)?
            .json()
            .await
            .map_err(APIError::from)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

        let mut loaded = HashSet::new();
        let mut loading = HashSet::new();
        let mut failed = HashSet::new();
        for record in &status.adapters {
            match record.state {
                AdapterLoadState::Loaded => {
                    loaded.insert(record.id.clone());
                }
                AdapterLoadState::Loading => {
                    loading.insert(record.id.clone());
                }
                AdapterLoadState::Failed => {
                    failed.insert(record.id.clone());
                }
                _ => {}
            }
        }

        let mut states = self.cloud_states.write();
        if let Some(state) = states.get_mut(&backend.id) {
            state.last_error = None;
            state.latency_ewma_ms = ewma(state.latency_ewma_ms, elapsed_ms);
            state.last_status_refresh = Some(Instant::now());
            state.max_active_loras = Some(status.max_active_loras);
            state.loaded_loras = Some(status.loaded_loras);
            state.slots_free = Some(status.slots_free);
            state.loaded_adapters = loaded;
            state.loading_adapters = loading;
            state.failed_adapters = failed;
        }
        Ok(())
    }

    pub async fn cloud_status_snapshots(&self) -> Vec<CloudBackendStatusSnapshot> {
        self.refresh_cloud_statuses(false).await;
        let states = self.cloud_states.read();
        let mut snapshots = Vec::new();
        for backend in &self.cloud_backends {
            if let Some(state) = states.get(&backend.id) {
                let mut loaded_adapters = state.loaded_adapters.iter().cloned().collect::<Vec<_>>();
                loaded_adapters.sort();
                snapshots.push(CloudBackendStatusSnapshot {
                    id: backend.id.clone(),
                    base_url: backend.base_url.clone(),
                    outstanding: state.outstanding,
                    latency_ewma_ms: state.latency_ewma_ms,
                    max_active_loras: state.max_active_loras,
                    loaded_loras: state.loaded_loras,
                    slots_free: state.slots_free,
                    loaded_adapters,
                    last_error: state.last_error.clone(),
                });
            }
        }
        snapshots
    }

    pub async fn load_adapter_on_cloud(
        &self,
        adapter_id: &str,
        pinned: Option<bool>,
        wait: bool,
        backend_id: Option<&str>,
    ) -> Vec<BackendOperationResult> {
        self.adapter_op_on_cloud(adapter_id, pinned, wait, backend_id, true)
            .await
    }

    pub async fn unload_adapter_on_cloud(
        &self,
        adapter_id: &str,
        backend_id: Option<&str>,
    ) -> Vec<BackendOperationResult> {
        self.adapter_op_on_cloud(adapter_id, None, false, backend_id, false)
            .await
    }

    pub async fn register_adapter_on_cloud(
        &self,
        request: &RegisterAdapterRequest,
        backend_id: Option<&str>,
    ) -> Vec<BackendOperationResult> {
        let backends: Vec<CloudBackendConfig> = if let Some(id) = backend_id {
            self.get_cloud_backend(id).into_iter().collect()
        } else {
            self.cloud_backends.clone()
        };

        let mut body = request.clone();
        body.scope = Some("local".to_string());
        body.backend = None;

        let mut results = Vec::new();
        for backend in backends {
            if let Some(expected_model) = self.runtime_model_id.as_deref() {
                match self.fetch_cloud_model_id(&backend).await {
                    Ok(cloud_model_id) => {
                        if !model_ids_compatible(expected_model, &cloud_model_id) {
                            results.push(BackendOperationResult {
                                backend_id: backend.id.clone(),
                                ok: false,
                                message: format!(
                                    "model mismatch: local='{}' cloud='{}'",
                                    expected_model, cloud_model_id
                                ),
                            });
                            continue;
                        }
                    }
                    Err(err) => {
                        results.push(BackendOperationResult {
                            backend_id: backend.id.clone(),
                            ok: false,
                            message: format!("failed model handshake: {}", err),
                        });
                        continue;
                    }
                }
            }

            let url = format!("{}/v1/adapters", backend.base_url);
            let req_started = Instant::now();
            self.begin_cloud_request(&backend.id);
            let response = self.client.post(&url).json(&body).send().await;

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    self.end_cloud_request(&backend.id, req_started, status.is_success());
                    results.push(BackendOperationResult {
                        backend_id: backend.id.clone(),
                        ok: status.is_success(),
                        message: if status.is_success() {
                            "registered".to_string()
                        } else if text.is_empty() {
                            status.to_string()
                        } else {
                            format!("{}: {}", status, text)
                        },
                    });
                }
                Err(err) => {
                    self.end_cloud_request(&backend.id, req_started, false);
                    results.push(BackendOperationResult {
                        backend_id: backend.id.clone(),
                        ok: false,
                        message: err.to_string(),
                    });
                }
            }
        }
        results
    }

    async fn fetch_cloud_model_id(&self, backend: &CloudBackendConfig) -> Result<String, String> {
        #[derive(Debug, Deserialize)]
        struct CloudModelEntry {
            id: String,
        }
        #[derive(Debug, Deserialize)]
        struct CloudModelsResponse {
            data: Vec<CloudModelEntry>,
        }

        let url = format!("{}/v1/models", backend.base_url);
        let req_started = Instant::now();
        self.begin_cloud_request(&backend.id);
        let result = async {
            let response = self
                .client
                .get(url)
                .send()
                .await
                .map_err(|err| err.to_string())?;
            let response = response.error_for_status().map_err(|err| err.to_string())?;
            let parsed = response
                .json::<CloudModelsResponse>()
                .await
                .map_err(|err| err.to_string())?;
            parsed
                .data
                .into_iter()
                .map(|entry| entry.id.trim().to_string())
                .find(|id| !id.is_empty())
                .ok_or_else(|| "cloud /v1/models response contained no model id".to_string())
        }
        .await;
        self.end_cloud_request(&backend.id, req_started, result.is_ok());
        result
    }

    pub async fn maybe_autosync_cloud_adapter(
        &self,
        adapter_id: &str,
        selected_cloud_backend: Option<&str>,
    ) {
        if !self.adapter_autosync || self.cloud_backends.is_empty() {
            return;
        }

        for backend in &self.cloud_backends {
            if selected_cloud_backend.is_some_and(|id| id == backend.id) {
                continue;
            }
            let should_skip = {
                let states = self.cloud_states.read();
                states
                    .get(&backend.id)
                    .map(|state| state.loaded_adapters.contains(adapter_id))
                    .unwrap_or(false)
            };
            if should_skip {
                continue;
            }
            let id = backend.id.clone();
            let _ = self
                .load_adapter_on_cloud(adapter_id, None, false, Some(&id))
                .await;
        }
    }

    pub async fn list_adapters_on_cloud(
        &self,
        backend_id: Option<&str>,
    ) -> Vec<CloudAdapterListResult> {
        let backends: Vec<CloudBackendConfig> = if let Some(id) = backend_id {
            self.get_cloud_backend(id).into_iter().collect()
        } else {
            self.cloud_backends.clone()
        };

        let mut results = Vec::new();
        for backend in backends {
            let url = format!("{}/v1/adapters", backend.base_url);
            let req_started = Instant::now();
            self.begin_cloud_request(&backend.id);
            let response = self.client.get(&url).send().await;
            match response {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<Vec<AdapterRecord>>().await {
                            Ok(adapters) => {
                                self.end_cloud_request(&backend.id, req_started, true);
                                results.push(CloudAdapterListResult {
                                    backend_id: backend.id.clone(),
                                    adapters,
                                    error: None,
                                });
                            }
                            Err(err) => {
                                self.end_cloud_request(&backend.id, req_started, false);
                                results.push(CloudAdapterListResult {
                                    backend_id: backend.id.clone(),
                                    adapters: Vec::new(),
                                    error: Some(err.to_string()),
                                });
                            }
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        self.end_cloud_request(&backend.id, req_started, false);
                        results.push(CloudAdapterListResult {
                            backend_id: backend.id.clone(),
                            adapters: Vec::new(),
                            error: Some(if body.is_empty() {
                                status.to_string()
                            } else {
                                format!("{}: {}", status, body)
                            }),
                        });
                    }
                }
                Err(err) => {
                    self.end_cloud_request(&backend.id, req_started, false);
                    results.push(CloudAdapterListResult {
                        backend_id: backend.id.clone(),
                        adapters: Vec::new(),
                        error: Some(err.to_string()),
                    });
                }
            }
        }
        results
    }

    pub async fn get_adapter_on_cloud(
        &self,
        adapter_id: &str,
        backend_id: Option<&str>,
    ) -> Vec<CloudAdapterGetResult> {
        let backends: Vec<CloudBackendConfig> = if let Some(id) = backend_id {
            self.get_cloud_backend(id).into_iter().collect()
        } else {
            self.cloud_backends.clone()
        };

        let mut results = Vec::new();
        for backend in backends {
            let url = format!("{}/v1/adapters/{}", backend.base_url, adapter_id);
            let req_started = Instant::now();
            self.begin_cloud_request(&backend.id);
            let response = self.client.get(&url).send().await;
            match response {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<AdapterRecord>().await {
                            Ok(adapter) => {
                                self.end_cloud_request(&backend.id, req_started, true);
                                results.push(CloudAdapterGetResult {
                                    backend_id: backend.id.clone(),
                                    adapter: Some(adapter),
                                    error: None,
                                });
                            }
                            Err(err) => {
                                self.end_cloud_request(&backend.id, req_started, false);
                                results.push(CloudAdapterGetResult {
                                    backend_id: backend.id.clone(),
                                    adapter: None,
                                    error: Some(err.to_string()),
                                });
                            }
                        }
                    } else if status == reqwest::StatusCode::NOT_FOUND {
                        self.end_cloud_request(&backend.id, req_started, true);
                        results.push(CloudAdapterGetResult {
                            backend_id: backend.id.clone(),
                            adapter: None,
                            error: None,
                        });
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        self.end_cloud_request(&backend.id, req_started, false);
                        results.push(CloudAdapterGetResult {
                            backend_id: backend.id.clone(),
                            adapter: None,
                            error: Some(if body.is_empty() {
                                status.to_string()
                            } else {
                                format!("{}: {}", status, body)
                            }),
                        });
                    }
                }
                Err(err) => {
                    self.end_cloud_request(&backend.id, req_started, false);
                    results.push(CloudAdapterGetResult {
                        backend_id: backend.id.clone(),
                        adapter: None,
                        error: Some(err.to_string()),
                    });
                }
            }
        }
        results
    }

    async fn adapter_op_on_cloud(
        &self,
        adapter_id: &str,
        pinned: Option<bool>,
        wait: bool,
        backend_id: Option<&str>,
        load: bool,
    ) -> Vec<BackendOperationResult> {
        let backends: Vec<CloudBackendConfig> = if let Some(id) = backend_id {
            self.get_cloud_backend(id).into_iter().collect()
        } else {
            self.cloud_backends.clone()
        };

        let mut results = Vec::new();
        for backend in backends {
            let url = if load {
                format!("{}/v1/adapters/{}/load", backend.base_url, adapter_id)
            } else {
                format!("{}/v1/adapters/{}/unload", backend.base_url, adapter_id)
            };
            let req_started = Instant::now();
            let response = if load {
                let body = serde_json::json!({
                    "pinned": pinned,
                    "wait": wait
                });
                self.client.post(&url).json(&body).send().await
            } else {
                self.client.post(&url).send().await
            };

            match response {
                Ok(resp) => {
                    self.end_cloud_request(&backend.id, req_started, resp.status().is_success());
                    if resp.status().is_success() {
                        results.push(BackendOperationResult {
                            backend_id: backend.id.clone(),
                            ok: true,
                            message: if load {
                                "load requested".to_string()
                            } else {
                                "unloaded".to_string()
                            },
                        });
                    } else {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        results.push(BackendOperationResult {
                            backend_id: backend.id.clone(),
                            ok: false,
                            message: format!("{}: {}", status, body),
                        });
                    }
                }
                Err(err) => {
                    self.end_cloud_request(&backend.id, req_started, false);
                    results.push(BackendOperationResult {
                        backend_id: backend.id.clone(),
                        ok: false,
                        message: err.to_string(),
                    });
                }
            }
        }
        results
    }

    fn pick_best_cloud(&self, adapter_id: Option<&str>) -> Option<(String, f64)> {
        let states = self.cloud_states.read();
        let mut best: Option<(String, f64)> = None;
        for backend in &self.cloud_backends {
            let Some(state) = states.get(&backend.id) else {
                continue;
            };
            let score = self.score_cloud(backend, state, adapter_id);
            match &best {
                Some((_, best_score)) if score >= *best_score => {}
                _ => {
                    best = Some((backend.id.clone(), score));
                }
            }
        }
        best
    }

    fn score_local(&self, local_status: &AdapterStatusResponse, adapter_id: Option<&str>) -> f64 {
        let outstanding = *self.local_outstanding.read() as f64;
        let latency = *self.local_latency_ewma_ms.read() / 100.0;
        let slot_pressure = if local_status.max_active_loras == 0 {
            0.0
        } else {
            local_status.loaded_loras as f64 / local_status.max_active_loras as f64
        };

        let adapter_adjustment = adapter_id
            .and_then(|id| {
                local_status
                    .adapters
                    .iter()
                    .find(|adapter| adapter.id == id)
                    .map(|adapter| adapter.state.clone())
            })
            .map(adapter_state_penalty)
            .unwrap_or(0.8);

        (outstanding + latency + slot_pressure + adapter_adjustment) * self.local_weight
    }

    fn score_cloud(
        &self,
        backend: &CloudBackendConfig,
        state: &CloudBackendState,
        adapter_id: Option<&str>,
    ) -> f64 {
        let outstanding = state.outstanding as f64;
        let latency = state.latency_ewma_ms / 100.0;
        let slot_pressure = match (state.loaded_loras, state.max_active_loras) {
            (Some(loaded), Some(max)) if max > 0 => loaded as f64 / max as f64,
            _ => 0.25,
        };

        let adapter_adjustment = if let Some(adapter_id) = adapter_id {
            if state.loaded_adapters.contains(adapter_id) {
                -2.8
            } else if state.loading_adapters.contains(adapter_id) {
                -1.2
            } else if state.failed_adapters.contains(adapter_id) {
                4.0
            } else {
                1.1
            }
        } else {
            0.0
        };

        let err_penalty = if state.last_error.is_some() { 1.5 } else { 0.0 };
        let weighted = outstanding + latency + slot_pressure + adapter_adjustment + err_penalty;
        weighted * (self.cloud_weight / backend.weight.max(0.1))
    }
}

fn adapter_state_penalty(state: AdapterLoadState) -> f64 {
    match state {
        AdapterLoadState::Loaded => -3.0,
        AdapterLoadState::Loading => -1.5,
        AdapterLoadState::Failed => 4.0,
        AdapterLoadState::Registered => 1.0,
        AdapterLoadState::Evicting => 2.5,
    }
}

fn ewma(current: f64, latest: f64) -> f64 {
    if current <= 0.0 {
        return latest;
    }
    (current * 0.8) + (latest * 0.2)
}

fn normalize_model_id(value: &str) -> String {
    let raw = value.trim().trim_end_matches('/');
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

fn normalize_model_family(value: &str) -> String {
    let mut family = normalize_model_id(value);
    for suffix in ["-instruct", "-chat"] {
        if family.ends_with(suffix) {
            family.truncate(family.len() - suffix.len());
            break;
        }
    }
    family
}

fn model_ids_compatible(local: &str, remote: &str) -> bool {
    let local = normalize_model_family(local);
    let remote = normalize_model_family(remote);
    local == remote || local.contains(&remote) || remote.contains(&local)
}

#[cfg(test)]
mod tests {
    use super::{BackendRouter, CloudBackendConfig, RoutingMode};
    use crate::openai::lora::{AdapterLoadState, AdapterRecord, AdapterStatusResponse};
    use std::time::SystemTime;

    fn local_status_with(adapter_id: Option<(&str, AdapterLoadState)>) -> AdapterStatusResponse {
        let adapters = adapter_id
            .map(|(id, state)| {
                vec![AdapterRecord {
                    id: id.to_string(),
                    version: None,
                    adapter_path: "/tmp/a.safetensors".to_string(),
                    config_path: "/tmp/adapter_config.json".to_string(),
                    checksum_sha256: None,
                    state,
                    pinned: false,
                    base_model: None,
                    rank: None,
                    alpha: None,
                    target_modules: Vec::new(),
                    artifact_refs: Vec::new(),
                    loaded_modules: 0,
                    loaded_at: None,
                    last_used_at: None,
                    last_error: None,
                    created_at: SystemTime::now(),
                    updated_at: SystemTime::now(),
                }]
            })
            .unwrap_or_default();
        AdapterStatusResponse {
            max_active_loras: 8,
            loaded_loras: adapters
                .iter()
                .filter(|adapter| adapter.state == AdapterLoadState::Loaded)
                .count(),
            lora_mode: "fallback".to_string(),
            slots_used: adapters
                .iter()
                .filter(|adapter| adapter.state == AdapterLoadState::Loaded)
                .count(),
            slots_free: 8 - adapters
                .iter()
                .filter(|adapter| adapter.state == AdapterLoadState::Loaded)
                .count(),
            adapters,
        }
    }

    #[tokio::test]
    async fn local_only_always_selects_local() {
        let router = BackendRouter::new(
            Vec::new(),
            1.0,
            1.0,
            std::time::Duration::from_secs(5),
            false,
            None,
        )
        .expect("router");
        let selected = router
            .select_backend(RoutingMode::LocalOnly, None, &local_status_with(None))
            .await
            .expect("select");
        assert!(matches!(selected, super::SelectedBackend::Local));
    }

    #[tokio::test]
    async fn cloud_only_errors_without_cloud_nodes() {
        let router = BackendRouter::new(
            Vec::new(),
            1.0,
            1.0,
            std::time::Duration::from_secs(5),
            false,
            None,
        )
        .expect("router");
        let err = router
            .select_backend(RoutingMode::CloudOnly, None, &local_status_with(None))
            .await
            .expect_err("should fail");
        assert!(err.to_string().contains("Cloud mode requested"));
    }

    #[test]
    fn cloud_backend_specs_parse_and_validate() {
        let parsed = CloudBackendConfig::parse_specs(&vec![
            "edge-a=http://127.0.0.1:2001".to_string(),
            "https://host-two:3000".to_string(),
        ])
        .expect("valid specs");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].id, "edge-a");
        assert_eq!(parsed[1].id, "cloud-2");

        let duplicate = CloudBackendConfig::parse_specs(&vec![
            "same=http://127.0.0.1:1".to_string(),
            "same=http://127.0.0.1:2".to_string(),
        ]);
        assert!(duplicate.is_err());
    }

    #[test]
    fn model_id_compatibility_is_normalized() {
        assert!(super::model_ids_compatible("Qwen3_4B", "qwen3-4b"));
        assert!(super::model_ids_compatible(
            "org/qwen3-4b-instruct",
            "qwen3_4b"
        ));
        assert!(super::model_ids_compatible(
            "Qwen/Qwen3.5-9B-Instruct",
            "qwen3_5-9b"
        ));
        assert!(!super::model_ids_compatible("qwen3-4b", "llama3-8b"));
    }
}
