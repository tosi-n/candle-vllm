use std::pin::Pin;
use std::sync::Arc;

use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use tokio_stream::StreamExt;
use tokio_stream::{self as stream, Stream};
use tonic::{Request, Response, Status};

use crate::openai::lora::{AdapterLoadState, LoRAManager};
use crate::openai::pipelines::llm_engine::{KvSessionPayload, LLMEngine};

pub mod proto {
    tonic::include_proto!("hybrie.runtime.v1");
}

use proto::runtime_internal_server::RuntimeInternal;

#[derive(Debug, Clone)]
pub struct RuntimeKvProfile {
    pub model_id: String,
    pub model_hash: String,
    pub dtype: String,
    pub block_size: usize,
    pub num_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Clone)]
pub struct RuntimeInternalService {
    engine: Arc<RwLock<LLMEngine>>,
    lora_manager: Arc<LoRAManager>,
    profile: RuntimeKvProfile,
    chunk_bytes: usize,
}

type ExportSessionKvStream =
    Pin<Box<dyn Stream<Item = Result<proto::KvTransferChunk, Status>> + Send + 'static>>;

impl RuntimeInternalService {
    pub fn new(
        engine: Arc<RwLock<LLMEngine>>,
        lora_manager: Arc<LoRAManager>,
        profile: RuntimeKvProfile,
        chunk_bytes: usize,
    ) -> Self {
        Self {
            engine,
            lora_manager,
            profile,
            chunk_bytes: std::cmp::max(chunk_bytes, 64 * 1024),
        }
    }

    pub fn model_hash(
        model_id: &str,
        dtype: &str,
        block_size: usize,
        num_layers: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!(
            "{}|{}|{}|{}|{}|{}",
            model_id, dtype, block_size, num_layers, kv_heads, head_dim
        ));
        format!("{:x}", hasher.finalize())
    }

    fn build_metadata(
        &self,
        payload: &KvSessionPayload,
        checksum: String,
    ) -> proto::SessionCompatibilityMetadata {
        let (adapter_id, adapter_version) = if let Some(adapter_id) = payload.adapter_id.as_deref()
        {
            let version = self
                .lora_manager
                .get(adapter_id)
                .and_then(|record| record.version)
                .unwrap_or_else(|| "latest".to_string());
            (adapter_id.to_string(), version)
        } else {
            (String::new(), String::new())
        };

        proto::SessionCompatibilityMetadata {
            model_id: self.profile.model_id.clone(),
            model_hash: self.profile.model_hash.clone(),
            adapter_id,
            adapter_version,
            dtype: self.profile.dtype.clone(),
            block_size: self.profile.block_size as u64,
            num_layers: self.profile.num_layers as u64,
            kv_heads: self.profile.kv_heads as u64,
            head_dim: self.profile.head_dim as u64,
            prompt_len: payload.prompt_len as u64,
            session_id: payload.session_id.clone(),
            checksum,
        }
    }

    fn validate_metadata_compatibility(
        &self,
        metadata: &proto::SessionCompatibilityMetadata,
    ) -> Result<(), Status> {
        if metadata.model_id != self.profile.model_id {
            return Err(Status::failed_precondition(format!(
                "model_id mismatch: expected '{}' got '{}'",
                self.profile.model_id, metadata.model_id
            )));
        }
        if metadata.model_hash != self.profile.model_hash {
            return Err(Status::failed_precondition(format!(
                "model_hash mismatch: expected '{}' got '{}'",
                self.profile.model_hash, metadata.model_hash
            )));
        }
        if metadata.dtype.to_lowercase() != self.profile.dtype.to_lowercase() {
            return Err(Status::failed_precondition(format!(
                "dtype mismatch: expected '{}' got '{}'",
                self.profile.dtype, metadata.dtype
            )));
        }
        if metadata.block_size as usize != self.profile.block_size {
            return Err(Status::failed_precondition(format!(
                "block_size mismatch: expected '{}' got '{}'",
                self.profile.block_size, metadata.block_size
            )));
        }
        if metadata.num_layers as usize != self.profile.num_layers {
            return Err(Status::failed_precondition(format!(
                "num_layers mismatch: expected '{}' got '{}'",
                self.profile.num_layers, metadata.num_layers
            )));
        }
        if metadata.kv_heads as usize != self.profile.kv_heads {
            return Err(Status::failed_precondition(format!(
                "kv_heads mismatch: expected '{}' got '{}'",
                self.profile.kv_heads, metadata.kv_heads
            )));
        }
        if metadata.head_dim as usize != self.profile.head_dim {
            return Err(Status::failed_precondition(format!(
                "head_dim mismatch: expected '{}' got '{}'",
                self.profile.head_dim, metadata.head_dim
            )));
        }
        Ok(())
    }

    async fn fail_fast_release(&self, session_id: &str) {
        let mut engine = self.engine.write();
        let _ = engine.release_session(session_id);
    }
}

#[tonic::async_trait]
impl RuntimeInternal for RuntimeInternalService {
    type ExportSessionKvStream = ExportSessionKvStream;

    async fn export_session_kv(
        &self,
        request: Request<proto::ExportSessionKvRequest>,
    ) -> Result<Response<Self::ExportSessionKvStream>, Status> {
        let session_id = request.into_inner().session_id.trim().to_string();
        if session_id.is_empty() {
            return Err(Status::invalid_argument("session_id is required"));
        }

        let payload = {
            let engine = self.engine.read();
            engine
                .export_session_payload(&session_id)
                .map_err(|err| Status::internal(err.to_string()))?
        }
        .ok_or_else(|| Status::not_found(format!("session '{}' not found", session_id)))?;

        let encoded = bincode::serialize(&payload)
            .map_err(|err| Status::internal(format!("failed to serialize payload: {}", err)))?;
        let checksum = {
            let mut hasher = Sha256::new();
            hasher.update(&encoded);
            format!("{:x}", hasher.finalize())
        };
        let metadata = self.build_metadata(&payload, checksum.clone());

        let total_chunks = std::cmp::max(1, encoded.len().div_ceil(self.chunk_bytes));
        let mut chunks = Vec::with_capacity(total_chunks);
        if encoded.is_empty() {
            chunks.push(proto::KvTransferChunk {
                session_id: payload.session_id,
                sequence_no: 0,
                payload: Vec::new(),
                is_last: true,
                checksum,
                metadata: Some(metadata),
            });
        } else {
            for (idx, chunk) in encoded.chunks(self.chunk_bytes).enumerate() {
                chunks.push(proto::KvTransferChunk {
                    session_id: payload.session_id.clone(),
                    sequence_no: idx as u64,
                    payload: chunk.to_vec(),
                    is_last: idx + 1 == total_chunks,
                    checksum: if idx + 1 == total_chunks {
                        checksum.clone()
                    } else {
                        String::new()
                    },
                    metadata: if idx == 0 {
                        Some(metadata.clone())
                    } else {
                        None
                    },
                });
            }
        }

        let out = stream::iter(chunks.into_iter().map(Ok));
        Ok(Response::new(Box::pin(out) as Self::ExportSessionKvStream))
    }

    async fn import_session_kv(
        &self,
        request: Request<tonic::Streaming<proto::KvTransferChunk>>,
    ) -> Result<Response<proto::ImportSessionKvResponse>, Status> {
        let mut stream = request.into_inner();
        let mut expected_seq_no = 0u64;
        let mut bytes = Vec::<u8>::new();
        let mut session_id = String::new();
        let mut metadata: Option<proto::SessionCompatibilityMetadata> = None;
        let mut final_checksum: Option<String> = None;
        let mut chunks_received = 0u64;

        while let Some(item) = stream.next().await {
            let chunk = item?;
            if chunk.sequence_no != expected_seq_no {
                if !session_id.is_empty() {
                    self.fail_fast_release(&session_id).await;
                }
                return Err(Status::failed_precondition(format!(
                    "out-of-order chunk sequence: expected {} got {}",
                    expected_seq_no, chunk.sequence_no
                )));
            }
            expected_seq_no += 1;
            chunks_received += 1;

            if session_id.is_empty() {
                session_id = chunk.session_id.clone();
            }
            if session_id != chunk.session_id {
                self.fail_fast_release(&session_id).await;
                return Err(Status::failed_precondition(
                    "mixed session_id in import stream",
                ));
            }
            if metadata.is_none() {
                metadata = chunk.metadata.clone();
            }
            bytes.extend_from_slice(&chunk.payload);
            if chunk.is_last {
                if !chunk.checksum.is_empty() {
                    final_checksum = Some(chunk.checksum.clone());
                }
                break;
            }
        }

        if session_id.is_empty() {
            return Err(Status::invalid_argument("import stream had no chunks"));
        }

        let Some(metadata) = metadata else {
            self.fail_fast_release(&session_id).await;
            return Err(Status::failed_precondition(
                "missing transfer metadata in first chunk",
            ));
        };
        self.validate_metadata_compatibility(&metadata)?;

        if !metadata.adapter_id.is_empty() {
            let Some(local) = self.lora_manager.get(&metadata.adapter_id) else {
                self.fail_fast_release(&session_id).await;
                return Err(Status::failed_precondition(format!(
                    "adapter '{}' is not registered locally",
                    metadata.adapter_id
                )));
            };
            let local_version = local.version.unwrap_or_else(|| "latest".to_string());
            let expected_version = if metadata.adapter_version.is_empty() {
                "latest".to_string()
            } else {
                metadata.adapter_version.clone()
            };
            if local_version != expected_version {
                self.fail_fast_release(&session_id).await;
                return Err(Status::failed_precondition(format!(
                    "adapter version mismatch for '{}': expected '{}' got '{}'",
                    metadata.adapter_id, expected_version, local_version
                )));
            }
        }

        let computed_checksum = {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            format!("{:x}", hasher.finalize())
        };

        let expected_checksum = final_checksum
            .or_else(|| (!metadata.checksum.is_empty()).then(|| metadata.checksum.clone()))
            .ok_or_else(|| Status::failed_precondition("missing checksum in transfer"))?;

        if computed_checksum != expected_checksum {
            self.fail_fast_release(&session_id).await;
            return Err(Status::failed_precondition(format!(
                "checksum mismatch: expected {} got {}",
                expected_checksum, computed_checksum
            )));
        }

        if metadata.session_id != session_id {
            self.fail_fast_release(&session_id).await;
            return Err(Status::failed_precondition(format!(
                "session_id mismatch: metadata='{}' stream='{}'",
                metadata.session_id, session_id
            )));
        }

        let payload: KvSessionPayload = bincode::deserialize(&bytes)
            .map_err(|err| Status::failed_precondition(format!("invalid payload: {}", err)))?;

        if payload.session_id != session_id {
            self.fail_fast_release(&session_id).await;
            return Err(Status::failed_precondition(format!(
                "payload session mismatch: payload='{}' stream='{}'",
                payload.session_id, session_id
            )));
        }

        if payload.prompt_len as u64 != metadata.prompt_len {
            self.fail_fast_release(&session_id).await;
            return Err(Status::failed_precondition(format!(
                "prompt_len mismatch: payload={} metadata={}",
                payload.prompt_len, metadata.prompt_len
            )));
        }

        let adapter_mismatch = {
            let engine = self.engine.read();
            engine.get_session_lookup(&session_id).and_then(|target| {
                let target_adapter = target.adapter_id.unwrap_or_default();
                let expected_adapter = metadata.adapter_id.clone();
                if target_adapter != expected_adapter {
                    Some((target_adapter, expected_adapter))
                } else {
                    None
                }
            })
        };
        if let Some((target_adapter, expected_adapter)) = adapter_mismatch {
            self.fail_fast_release(&session_id).await;
            return Err(Status::failed_precondition(format!(
                "adapter_id mismatch: target='{}' imported='{}'",
                target_adapter, expected_adapter
            )));
        }

        {
            let mut engine = self.engine.write();
            if let Err(err) = engine.import_session_payload(payload) {
                let _ = engine.release_session(&session_id);
                return Err(Status::failed_precondition(format!(
                    "KV import failed for session '{}': {}",
                    session_id, err
                )));
            }
        }

        Ok(Response::new(proto::ImportSessionKvResponse {
            success: true,
            message: "imported".to_string(),
            chunks_received,
            bytes_received: bytes.len() as u64,
        }))
    }

    async fn get_session_info(
        &self,
        request: Request<proto::GetSessionInfoRequest>,
    ) -> Result<Response<proto::GetSessionInfoResponse>, Status> {
        let session_id = request.into_inner().session_id.trim().to_string();
        if session_id.is_empty() {
            return Err(Status::invalid_argument("session_id is required"));
        }

        let lookup = {
            let engine = self.engine.read();
            engine.get_session_lookup(&session_id)
        };

        let Some(lookup) = lookup else {
            return Ok(Response::new(proto::GetSessionInfoResponse {
                found: false,
                message: "not_found".to_string(),
                session_id,
                prompt_len: 0,
                total_len: 0,
                block_count: 0,
                block_ids: Vec::new(),
                adapter_id: String::new(),
                metadata: None,
            }));
        };

        let metadata = proto::SessionCompatibilityMetadata {
            model_id: self.profile.model_id.clone(),
            model_hash: self.profile.model_hash.clone(),
            adapter_id: lookup.adapter_id.clone().unwrap_or_default(),
            adapter_version: lookup
                .adapter_id
                .as_deref()
                .and_then(|id| self.lora_manager.get(id))
                .and_then(|record| record.version)
                .unwrap_or_else(|| "latest".to_string()),
            dtype: self.profile.dtype.clone(),
            block_size: self.profile.block_size as u64,
            num_layers: self.profile.num_layers as u64,
            kv_heads: self.profile.kv_heads as u64,
            head_dim: self.profile.head_dim as u64,
            prompt_len: lookup.prompt_len as u64,
            session_id: lookup.request_id.clone(),
            checksum: String::new(),
        };

        Ok(Response::new(proto::GetSessionInfoResponse {
            found: true,
            message: "ok".to_string(),
            session_id: lookup.request_id,
            prompt_len: lookup.prompt_len as u64,
            total_len: lookup.total_len as u64,
            block_count: lookup.block_ids.len() as u64,
            block_ids: lookup.block_ids.into_iter().map(|id| id as u64).collect(),
            adapter_id: lookup.adapter_id.unwrap_or_default(),
            metadata: Some(metadata),
        }))
    }

    async fn load_adapter(
        &self,
        request: Request<proto::AdapterLoadRequest>,
    ) -> Result<Response<proto::AdapterLoadResponse>, Status> {
        let request = request.into_inner();
        let adapter_id = request.adapter_id.trim();
        if adapter_id.is_empty() {
            return Err(Status::invalid_argument("adapter_id is required"));
        }

        if request.wait {
            self.lora_manager
                .load_async(adapter_id, Some(request.pinned))
                .await
                .map_err(|err| Status::failed_precondition(err.to_string()))?;
            return Ok(Response::new(proto::AdapterLoadResponse {
                accepted: true,
                message: "loaded".to_string(),
                state: "loaded".to_string(),
            }));
        }

        let (notifier, should_start) = self
            .lora_manager
            .enqueue_load(adapter_id, Some(request.pinned))
            .map_err(|err| Status::failed_precondition(err.to_string()))?;

        if should_start {
            let manager = Arc::clone(&self.lora_manager);
            let adapter_id = adapter_id.to_string();
            tokio::spawn(async move {
                if let Err(err) = manager.load_async(&adapter_id, None).await {
                    tracing::warn!(
                        "Background adapter load failed for '{}': {}",
                        adapter_id,
                        err
                    );
                }
            });
        } else {
            drop(notifier);
        }

        Ok(Response::new(proto::AdapterLoadResponse {
            accepted: true,
            message: "queued".to_string(),
            state: "loading".to_string(),
        }))
    }

    async fn unload_adapter(
        &self,
        request: Request<proto::AdapterUnloadRequest>,
    ) -> Result<Response<proto::AdapterUnloadResponse>, Status> {
        let adapter_id = request.into_inner().adapter_id.trim().to_string();
        if adapter_id.is_empty() {
            return Err(Status::invalid_argument("adapter_id is required"));
        }

        self.lora_manager
            .unload(&adapter_id)
            .map_err(|err| Status::failed_precondition(err.to_string()))?;

        Ok(Response::new(proto::AdapterUnloadResponse {
            success: true,
            message: "unloaded".to_string(),
        }))
    }

    async fn get_adapter_status(
        &self,
        _request: Request<proto::AdapterStatusRequest>,
    ) -> Result<Response<proto::AdapterStatusResponse>, Status> {
        let status = self.lora_manager.status();
        let adapters = status
            .adapters
            .into_iter()
            .map(|record| proto::AdapterState {
                adapter_id: record.id,
                state: adapter_state_name(&record.state).to_string(),
                pinned: record.pinned,
                rank: record.rank.unwrap_or_default() as u64,
                last_error: record.last_error.unwrap_or_default(),
            })
            .collect::<Vec<_>>();

        Ok(Response::new(proto::AdapterStatusResponse {
            max_active_loras: status.max_active_loras as u64,
            loaded_loras: status.loaded_loras as u64,
            lora_mode: status.lora_mode,
            adapters,
            slots_free: status.slots_free as u64,
        }))
    }
}

fn adapter_state_name(state: &AdapterLoadState) -> &'static str {
    match state {
        AdapterLoadState::Registered => "registered",
        AdapterLoadState::Loading => "loading",
        AdapterLoadState::Loaded => "loaded",
        AdapterLoadState::Evicting => "evicting",
        AdapterLoadState::Failed => "failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_hash_is_stable() {
        let a = RuntimeInternalService::model_hash("qwen", "bf16", 64, 48, 8, 128);
        let b = RuntimeInternalService::model_hash("qwen", "bf16", 64, 48, 8, 128);
        let c = RuntimeInternalService::model_hash("qwen", "f16", 64, 48, 8, 128);
        let d = RuntimeInternalService::model_hash("qwen", "bf16", 64, 48, 4, 128);
        let e = RuntimeInternalService::model_hash("qwen", "bf16", 32, 48, 8, 128);
        let f = RuntimeInternalService::model_hash("qwen", "bf16", 64, 36, 8, 128);
        let g = RuntimeInternalService::model_hash("qwen-alt", "bf16", 64, 48, 8, 128);
        let h = RuntimeInternalService::model_hash("qwen", "bf16", 64, 48, 8, 256);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(a, e);
        assert_ne!(a, f);
        assert_ne!(a, g);
        assert_ne!(a, h);
    }
}
