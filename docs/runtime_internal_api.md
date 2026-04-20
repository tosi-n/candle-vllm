# Runtime Internal API

This document covers the generic internal runtime interface for session/KV handoff and adapter runtime operations.

## Scope Boundary

- `candle-vllm` runtime node:
  - Executes local inference.
  - Owns scheduler/KV cache/adapter runtime.
  - Exposes internal gRPC for KV/session and adapter runtime operations.
- External orchestrators can use this API to coordinate session handoff, adapter lifecycle, and runtime compatibility checks.

## CLI Flags

Runtime boundary and internal API flags:

- `--runtime-local-only-strict` (default: `true`)
  - Preserves a local-only public runtime boundary.
- `--runtime-internal-api` (default: `false`)
  - Enables internal gRPC server for KV handoff and adapter runtime ops.
- `--runtime-internal-grpc-host` (default: `127.0.0.1`)
  - Bind host for internal gRPC.
- `--runtime-internal-grpc-port` (default: `51051`)
  - Bind port for internal gRPC.
- `--runtime-internal-chunk-bytes` (default: `1048576`)
  - Chunk size for KV transfer streaming.
- `--runtime-canonical-model-id` (optional)
  - Canonical model id used in KV metadata/model hash/adapter compatibility.
  - Resolution order for runtime identity is:
    1. `--runtime-canonical-model-id`
    2. `--m`
    3. basename of `--w`
    4. pipeline-derived name

Example:

```bash
candle-vllm \
  --w /models/Qwen3.5-9B-Instruct \
  --runtime-canonical-model-id Qwen/Qwen3.5-9B-Instruct \
  --p 2000 \
  --runtime-local-only-strict true \
  --runtime-internal-api true \
  --runtime-internal-grpc-host 127.0.0.1 \
  --runtime-internal-grpc-port 51051
```

## Public HTTP Notes

- Added `GET /v1/health`:
  - `status`
  - `runtime_local_only_strict`
  - `max_active_loras`
  - `loaded_loras`
  - `lora_mode`
- `GET /v1/models` includes:
  - `max_active_loras`
  - `loaded_loras`
  - `lora_mode`
  
## Internal gRPC Service

Proto: [`proto/runtime_internal.proto`](../proto/runtime_internal.proto)

Service: `candle.vllm.runtime.v1.RuntimeInternal`

Methods:

- `ExportSessionKv(ExportSessionKvRequest) -> stream KvTransferChunk`
- `ImportSessionKv(stream KvTransferChunk) -> ImportSessionKvResponse`
- `GetSessionInfo(GetSessionInfoRequest) -> GetSessionInfoResponse`
- `LoadAdapter(AdapterLoadRequest) -> AdapterLoadResponse`
- `UnloadAdapter(AdapterUnloadRequest) -> AdapterUnloadResponse`
- `GetAdapterStatus(AdapterStatusRequest) -> AdapterStatusResponse`

## KV Handoff Safety/Validation

`ImportSessionKv` validates and fails fast on:

- `model_id`, `model_hash`
- `dtype`, `block_size`, `num_layers`, `kv_heads`, `head_dim`
- `adapter_id` + `adapter_version`
- `prompt_len`
- transfer checksum
- chunk ordering (`sequence_no`)

On failure, runtime releases the target session (`fail-fast` behavior).

## Adapter Step Scheduling (Request Metadata)

Runtime supports decode step-level adapter switching via:

- request header adapter (existing): `x-runtime-adapter-id`
- request body metadata schedule:

```json
{
  "metadata": {
    "runtime": {
      "adapter_id": "adapter_planner_router_v1",
      "adapter_schedule": [
        { "start_step": 0, "adapter_id": "adapter_planner_router_v1" },
        { "start_step": 64, "adapter_id": "adapter_answerer_domain_v1" },
        { "start_step": 128, "adapter_id": "adapter_verifier_quality_v1" }
      ]
    }
  }
}
```

Semantics:

- Exactly one active adapter per sequence at a time.
- Switch happens at decode step boundaries.
- Runtime batches mixed adapters in the same decode cycle by sub-batching.
