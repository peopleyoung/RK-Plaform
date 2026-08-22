# RTSP SEI Overlay Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the JPEG/MJPEG preview path with authenticated RTSP publication through ZLMediaKit and direct browser WS-FLV playback whose schema-v2 SEI metadata is synchronized and rendered in Canvas.

**Architecture:** RK3588 keeps the original encoded video access units and adds one bounded user-data-unregistered SEI NAL. The platform manages ZLMediaKit gateways, role-bound credentials, task media bindings, health, and lifecycle cleanup without relaying media. A single frontend player adapter consumes ZLMediaKit WS-FLV, validates the fixed RKNode SEI contract, aligns it by PTS, and renders detection, OCR, analytics, and segmentation overlays.

**Tech Stack:** FastAPI, Pydantic v2, SQLAlchemy, SQLite, React, TypeScript, mpegts.js 1.8.2, Vitest 4.1.11, Canvas 2D, Web Workers, C++17, FFmpeg, OpenCV, RKNN, ZLMediaKit, Docker Compose, pytest, Playwright.

---

## Execution Rules

- Work task by task. Start every implementation task with the named failing test and run the focused test before editing production code.
- Do not restore a JPEG, MJPEG, transcoding, WebRTC, WS-TS, audio, or platform-media-proxy fallback.
- Never log or return a media URL query string, raw publication/play token, ZLMediaKit API secret, or gateway Hook identity.
- Keep existing JSONL, HTTP, Kafka, and event schema-v2 field names unchanged.
- Do not claim RK3588 acceptance until the hardware gate in Task 14 has run on an authorized board.
- The current workspace has no Git metadata. Run each listed `git commit` only in the version-controlled checkout used for implementation; do not initialize or rewrite repository history merely to satisfy these checkpoints.

## Contract Snapshot

The public task media object is:

```json
{
  "zlmSei": {
    "enabled": true,
    "gatewayId": "gateway_builtin",
    "streamName": "camera_01",
    "reconnectMs": 1000
  }
}
```

The node-only desired-state object replaces `gatewayId` and `streamName` with a secret publication URL:

```json
{
  "zlmSei": {
    "enabled": true,
    "publishUri": "rtsp://192.168.1.10:8554/live/camera_01?publishToken=opaque",
    "reconnectMs": 1000
  }
}
```

The browser playback descriptor is:

```json
{
  "streamUrl": "ws://192.168.1.10:8081/live/camera_01.live.flv?playToken=opaque",
  "expiresAt": "2026-08-20T12:01:00Z",
  "taskId": "task-id",
  "revision": 7,
  "gatewayId": "gateway_builtin",
  "app": "live",
  "streamName": "camera_01",
  "codec": "h264",
  "reconnectMs": 1000
}
```

The preview capability is independent of inference health:

```json
{
  "state": "available",
  "reason": null
}
```

Allowed capability states are `available`, `unsupported`, `migration_required`, and `gateway_offline`. Allowed gateway states are `disabled`, `probing`, `online`, and `error`. Allowed player states are `unsupported`, `waiting_publish`, `connecting`, `live`, `metadata_degraded`, `reconnecting`, `unauthorized`, `codec_unsupported`, and `stopped`.

### Task 1: Add the persistent gateway and credential model

**Files:**

- Create: `backend/platform_api/media_contracts.py`
- Create: `backend/platform_api/media_secrets.py`
- Modify: `backend/platform_api/db_models.py`
- Modify: `backend/platform_api/database.py`
- Modify: `backend/platform_api/context.py`
- Modify: `backend/platform_api/settings.py`
- Test: `tests/test_media_gateway_store.py`
- Test: `tests/test_database_migrations.py`

- [ ] **Step 1: Write failing model, secret-permission, and additive-migration tests**

Cover these exact invariants:

- `MediaGatewayRecord` stores separate publish, playback, and API origins; fixed `app`; built-in/enabled flags; state; bounded last error; last probe and authenticated keepalive times.
- `MediaCredentialRecord` stores only SHA-256 token hashes and the role/binding/expiry/use/revocation metadata.
- `InferenceMediaBindingRecord` projects task, gateway, app, and stream for active-stream conflict queries.
- `InferenceTaskRecord.media_migration_required` is additive and defaults false.
- Existing rows whose `media_json.zlmSei` has `outputUri` but no `gatewayId` are marked for migration without changing the old JSON.
- gateway API and Hook identity files use directory mode `0700` and file mode `0600`; API responses expose booleans only.

Run:

```bash
pytest -q tests/test_media_gateway_store.py tests/test_database_migrations.py
```

Expected: failures for missing media tables, contracts, secret store, and migration column.

- [ ] **Step 2: Add exact enums and request/response contracts**

Implement in `media_contracts.py`:

```python
class MediaGatewayStatus(StrEnum):
    DISABLED = "disabled"
    PROBING = "probing"
    ONLINE = "online"
    ERROR = "error"

class MediaCredentialRole(StrEnum):
    PUBLISH = "publish"
    PLAY = "play"

class PreviewCapabilityState(StrEnum):
    AVAILABLE = "available"
    UNSUPPORTED = "unsupported"
    MIGRATION_REQUIRED = "migration_required"
    GATEWAY_OFFLINE = "gateway_offline"
```

Use Pydantic aliases consistent with existing camel-case API output. Validate hosts as hostnames or IP literals without schemes, ports from 1 through 65535, `app` and `streamName` with `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`, and `reconnectMs` from 1000 through 4000.

- [ ] **Step 3: Add SQLAlchemy records and migration logic**

Use these records:

```text
media_gateways:
  id, name, builtin, enabled,
  publish_host, rtsp_port, playback_host, ws_port,
  api_host, api_port, app, status,
  last_probe_at, last_hook_at, last_error, created_at, updated_at

media_credentials:
  id, token_hash, role, gateway_id, task_id, revision,
  app, stream_name, principal, expires_at, used_at, revoked_at, created_at

inference_media_bindings:
  task_id, gateway_id, app, stream_name, created_at, updated_at
```

Index credential hash, gateway/role/revocation, and binding gateway/app/stream. Do not put raw tokens or secrets in SQLite. Migration must be idempotent and tested by calling `create_schema()` twice.

- [ ] **Step 4: Add `MediaSecretStore` and context wiring**

Mirror the permission and atomic-write discipline of `NodeSecretStore`, but expose only:

```python
write_api_secret(gateway_id: str, secret: str) -> None
write_hook_identity(gateway_id: str, identity: str) -> None
api_secret(gateway_id: str) -> str
hook_identity(gateway_id: str) -> str
write_publication_token(credential_id: str, token: str) -> None
publication_token(credential_id: str) -> str
delete_publication_token(credential_id: str) -> None
delete(gateway_id: str) -> None
configured(gateway_id: str) -> tuple[bool, bool]
```

Add `media_secret_dir` to settings, instantiate the store in `create_app`, and add it to `AppContext`.

- [ ] **Step 5: Run focused tests and commit**

```bash
pytest -q tests/test_media_gateway_store.py tests/test_database_migrations.py
git add backend/platform_api/media_contracts.py backend/platform_api/media_secrets.py backend/platform_api/db_models.py backend/platform_api/database.py backend/platform_api/context.py backend/platform_api/settings.py tests/test_media_gateway_store.py tests/test_database_migrations.py
git commit -m "feat(media): add gateway and credential persistence"
```

Expected: focused tests pass; no secret value appears in serialized responses.

### Task 2: Implement ZLMediaKit control, credentials, and default-deny Hooks

**Files:**

- Create: `backend/platform_api/zlm_client.py`
- Create: `backend/platform_api/media_service.py`
- Create: `backend/platform_api/media_routes.py`
- Modify: `backend/platform_api/app.py`
- Test: `tests/test_media_auth.py`
- Test: `tests/test_media_gateway_api.py`

- [ ] **Step 1: Write failing tests for role separation and gateway health**

Test all of the following:

- tokens are generated with `secrets.token_urlsafe(32)` and only `sha256(token)` is persisted;
- publish tokens are revision-bound and revocable; play tokens expire after 60 seconds, bind to principal `admin`, and are consumed once;
- `on_publish` accepts only RTSP plus exactly one `publishToken` and matching gateway/task/revision/app/stream;
- `on_play` accepts only HTTP/WS-FLV plus exactly one `playToken` and matching binding;
- swapping roles, replaying a play token, using a stale revision, disabling a gateway, or sending a foreign stream returns a nonzero ZLM Hook code;
- the Hook request body `mediaServerId` is compared in constant time with the restricted gateway Hook identity before token lookup;
- a recent authenticated `on_server_keepalive` updates `last_hook_at`; a gateway is online only when control probing succeeds, Hook config matches, and keepalive age is at most 30 seconds;
- failures redact URL queries and bound `lastError` to 500 characters.

Run:

```bash
pytest -q tests/test_media_auth.py tests/test_media_gateway_api.py
```

Expected: import/route failures before implementation.

- [ ] **Step 2: Implement the bounded ZLM control client**

Use `urllib.request` like the existing node client, with 2-second connect and 5-second request limits. Implement only:

```python
get_server_config()
get_media_info(app: str, stream: str)
close_streams(app: str, stream: str)
```

Call `/index/api/getServerConfig`, `/index/api/getMediaList`, and `/index/api/close_streams` with the API secret. For cleanup, send `vhost=__defaultVhost__`, the configured app/stream, and `force=1`; do not restrict `schema`, so the media source and derived viewers close together. Error objects may include endpoint origin and status code, never query strings or response bodies containing secrets.

- [ ] **Step 3: Implement the media service transaction boundaries**

`MediaService` owns gateway CRUD/probe, task binding conflict checks, credential issue/consume/revoke, public capability calculation, node-only publication URL generation, playback descriptor generation, keepalive receipt, and best-effort stream closure.

Construct URLs only from validated stored fields:

```python
publish_uri = f"rtsp://{host}:{port}/{app}/{stream}?publishToken={quote(token)}"
stream_url = f"ws://{host}:{port}/{app}/{stream}.live.flv?playToken={quote(token)}"
```

Use bracketed formatting for IPv6 literals. Never accept a complete media URL from an API caller.

- [ ] **Step 4: Implement authenticated management routes and unauthenticated Hook routes**

Add admin routes:

```text
GET    /api/v1/media-gateways
POST   /api/v1/media-gateways
PUT    /api/v1/media-gateways/{gateway_id}
POST   /api/v1/media-gateways/{gateway_id}/probe
DELETE /api/v1/media-gateways/{gateway_id}
POST   /api/v1/inference-tasks/{task_id}/playback-session
```

Add ZLM routes without bearer auth because ZLM supplies its own body identity:

```text
POST /api/v1/media-hooks/zlm/{gateway_id}/on-publish
POST /api/v1/media-hooks/zlm/{gateway_id}/on-play
POST /api/v1/media-hooks/zlm/{gateway_id}/on-server-keepalive
```

Every Hook always returns HTTP 200 with ZLM JSON. Success is `{"code": 0, "msg": "success"}`; denial is `{"code": -1, "msg": "denied"}`. Do not reveal which credential check failed.

ZLM's `admin_params` is not the Hook identity: matching it bypasses the Hook entirely. Keep it empty in the managed config and authenticate the documented `mediaServerId` body field instead.

- [ ] **Step 5: Run tests and commit**

```bash
pytest -q tests/test_media_auth.py tests/test_media_gateway_api.py
git add backend/platform_api/zlm_client.py backend/platform_api/media_service.py backend/platform_api/media_routes.py backend/platform_api/app.py tests/test_media_auth.py tests/test_media_gateway_api.py
git commit -m "feat(media): authorize ZLM publication and playback"
```

Expected: all allow/deny matrix cases pass and logs contain no test token values.

### Task 3: Replace task `outputUri` with a managed media binding

**Files:**

- Modify: `backend/platform_api/contracts.py`
- Modify: `backend/platform_api/inference_service.py`
- Modify: `backend/platform_api/inference_routes.py`
- Modify: `backend/platform_api/media_service.py`
- Test: `tests/test_inference_api.py`
- Test: `tests/test_media_task_lifecycle.py`

- [ ] **Step 1: Write failing task migration and lifecycle tests**

Test the canonical public media object, ASCII grammar, reconnect range, gateway availability, and active-stream uniqueness within one gateway/app. Add cases proving:

- RTSP input plus RKMPP decoder plus enabled ZLM on an online gateway is `available`;
- image/file/OpenCV/disabled media is `unsupported` without changing inference health;
- an offline gateway is `gateway_offline`;
- a legacy `outputUri` task is `migration_required` and cannot create a new desired revision;
- updating that task with `gatewayId` and `streamName` clears migration state;
- desired state contains `publishUri` but never `gatewayId`, API secret, Hook identity, or browser token;
- task stop, replacement revision, retirement, and gateway disable revoke credentials and attempt `close_streams` without failing task state if ZLM is unreachable.

Run:

```bash
pytest -q tests/test_inference_api.py tests/test_media_task_lifecycle.py
```

Expected: old output-URI behavior causes failures.

- [ ] **Step 2: Make the task contract typed and deterministic**

Replace raw media validation with a discriminated Pydantic contract. Reject `outputUri` on create/update with error code `media_migration_required`. Keep existing stored legacy JSON readable. Add `previewCapability` to every task response and keep it out of node health calculations.

- [ ] **Step 3: Generate publication credentials at revision activation**

When a task receives a new `config_revision`, revoke its prior publication credential, create one credential for the new revision, and persist its hash. Reuse that credential's protected clear value in the permission-restricted desired-revision artifact; do not mint a new publisher token on each desired-state poll. If the current design does not persist desired-state artifacts, add a `secret_value` file under the media secret root named by credential ID, mode `0600`, and delete it on revocation.

Build `AgentTaskDescriptor.media` through `MediaService.node_media(...)` instead of passing `task.media_json` through directly.

- [ ] **Step 4: Enforce lifecycle cleanup centrally**

Create one idempotent helper:

```python
revoke_task_publication(session, task, *, close_stream: bool) -> None
```

Call it from stop, retire, revision replacement, rollback, and gateway disable paths. Commit the database revocation even when the external close request fails; store only a redacted bounded diagnostic.

- [ ] **Step 5: Run focused tests and commit**

```bash
pytest -q tests/test_inference_api.py tests/test_media_task_lifecycle.py
git add backend/platform_api/contracts.py backend/platform_api/inference_service.py backend/platform_api/inference_routes.py backend/platform_api/media_service.py tests/test_inference_api.py tests/test_media_task_lifecycle.py
git commit -m "feat(inference): bind task media to managed gateways"
```

### Task 4: Remove the platform and Agent JPEG/MJPEG chain

**Files:**

- Delete: `backend/platform_api/inference_preview.py`
- Modify: `backend/platform_api/inference_routes.py`
- Modify: `backend/platform_api/inference_service.py`
- Modify: `backend/platform_api/settings.py`
- Modify: `workers/inference_agent/agent.py`
- Modify: `workers/inference_agent/client.py`
- Modify: `workers/inference_agent/runtime_adapter.py`
- Delete: `third_party/nv_video_pipeline/src/nodes/PreviewOutputNode.cpp`
- Delete: `third_party/nv_video_pipeline/src/nodes/PreviewOutputNode.h`
- Modify: `third_party/nv_video_pipeline/CMakeLists.txt`
- Test: `tests/test_inference_agent.py`
- Test: `tests/test_inference_runtime_adapter.py`
- Test: `tests/test_inference_api.py`
- Test: `tests/test_no_legacy_preview.py`

- [ ] **Step 1: Add a repository-level failure test for forbidden legacy symbols**

`tests/test_no_legacy_preview.py` must scan runtime source, Compose, environment examples, and current operator docs and fail on:

```text
PreviewOutputNode
RKNODE_PREVIEW_
preview.jpg
preview.mjpeg
upload_preview
InferencePreviewStore
useInferencePreviewStream
```

Exclude historical plans/specs and archived Trellis task records from this scan.

Run:

```bash
pytest -q tests/test_no_legacy_preview.py tests/test_inference_agent.py tests/test_inference_runtime_adapter.py tests/test_inference_api.py
```

Expected: the new guard identifies all current legacy references.

- [ ] **Step 2: Remove backend upload/session/image routes and settings**

Delete the old Agent preview upload route, old preview-session implementation, JPEG/MJPEG responses, preview store, and preview settings. The only session route left is the new playback descriptor from Task 2.

- [ ] **Step 3: Remove Agent polling/upload and runtime preview graph generation**

Delete preview uploader state, methods, client calls, environment parsing, filesystem polling, JPEG limits, cleanup, and `PreviewOutputNode` YAML generation. Change `_task_media` to require node-only `publishUri`, validate RTSP without userinfo, validate `reconnectMs` as 1000..4000, and emit only `ZlmSeiOutputNode` for enabled media.

- [ ] **Step 4: Delete the C++ preview node and build entry**

Remove both files and the CMake source line. Do not leave a disabled registration or OpenCV JPEG dependency for preview.

- [ ] **Step 5: Run focused tests and commit**

```bash
pytest -q tests/test_no_legacy_preview.py tests/test_inference_agent.py tests/test_inference_runtime_adapter.py tests/test_inference_api.py
git add -A backend/platform_api workers/inference_agent third_party/nv_video_pipeline/src/nodes/PreviewOutputNode.cpp third_party/nv_video_pipeline/src/nodes/PreviewOutputNode.h third_party/nv_video_pipeline/CMakeLists.txt tests/test_no_legacy_preview.py tests/test_inference_agent.py tests/test_inference_runtime_adapter.py tests/test_inference_api.py
git commit -m "refactor(preview): remove JPEG and MJPEG pipeline"
```

### Task 5: Implement official DeepLab post-processing for every sink

**Files:**

- Create: `third_party/nv_video_pipeline/src/rknn_instance/DeepLabPostprocess.h`
- Create: `third_party/nv_video_pipeline/src/rknn_instance/DeepLabPostprocess.cpp`
- Modify: `third_party/nv_video_pipeline/src/rknn_instance/RknnStructuredInstance.cpp`
- Modify: `third_party/nv_video_pipeline/CMakeLists.txt`
- Create: `third_party/nv_video_pipeline/tests/DeepLabPostprocessTest.cpp`
- Create: `tests/test_deeplab_postprocess_cpp.py`
- Modify: `tests/test_deeplab_rknn_infer.py`

- [ ] **Step 1: Add failing layout, interpolation, argmax, and tie tests**

Use one `NCHW` and one equivalent `NHWC` tensor with three classes and a 2x2 logit grid resized to 5x3. Calculate the expected mask with `cv::resize(..., INTER_LINEAR)` on each class plane followed by a strict class-axis argmax. Include a tie and assert the lowest class index wins, matching NumPy `argmax`.

The host test must compile without RKNN libraries:

```bash
pytest -q tests/test_deeplab_postprocess_cpp.py tests/test_deeplab_rknn_infer.py
```

Expected: the current low-resolution argmax implementation disagrees at interpolation boundaries.

- [ ] **Step 2: Extract a pure official post-processor**

Expose:

```cpp
bool deeplab_logits_to_mask(
    const float* logits,
    size_t classes,
    size_t height,
    size_t width,
    bool nhwc,
    int source_width,
    int source_height,
    cv::Mat& class_mask);
```

Reject null/zero/oversized dimensions. Normalize each class to a `CV_32F` plane, resize every plane to source dimensions with `cv::INTER_LINEAR`, then perform argmax. Return a source-resolution `CV_32S` mask. Never resize a class-ID mask.

- [ ] **Step 3: Route all structured sinks through the corrected mask**

Replace `RknnStructuredInstance::decode_deeplab` internals with the helper before constructing `FrameSegmentationResult`. Because JSONL, HTTP, Kafka, event media, and SEI all serialize that shared result, this changes all sinks together without schema changes.

- [ ] **Step 4: Run tests and commit**

```bash
pytest -q tests/test_deeplab_postprocess_cpp.py tests/test_deeplab_rknn_infer.py
git add third_party/nv_video_pipeline/src/rknn_instance/DeepLabPostprocess.h third_party/nv_video_pipeline/src/rknn_instance/DeepLabPostprocess.cpp third_party/nv_video_pipeline/src/rknn_instance/RknnStructuredInstance.cpp third_party/nv_video_pipeline/CMakeLists.txt third_party/nv_video_pipeline/tests/DeepLabPostprocessTest.cpp tests/test_deeplab_postprocess_cpp.py tests/test_deeplab_rknn_infer.py
git commit -m "fix(deeplab): resize logits before argmax"
```

### Task 6: Harden SEI generation and publishing limits

**Files:**

- Modify: `third_party/nv_video_pipeline/src/utils/SeiPacket.h`
- Modify: `third_party/nv_video_pipeline/src/utils/SeiPacket.cpp`
- Create: `third_party/nv_video_pipeline/src/utils/MediaUrl.h`
- Create: `third_party/nv_video_pipeline/src/utils/MediaUrl.cpp`
- Modify: `third_party/nv_video_pipeline/src/nodes/ZlmSeiOutputNode.h`
- Modify: `third_party/nv_video_pipeline/src/nodes/ZlmSeiOutputNode.cpp`
- Modify: `third_party/nv_video_pipeline/src/objects/FrameInferenceResult.h`
- Modify: `third_party/nv_video_pipeline/src/objects/FrameInferenceResult.cpp`
- Modify: `third_party/nv_video_pipeline/CMakeLists.txt`
- Create: `third_party/nv_video_pipeline/tests/SeiPacketTest.cpp`
- Create: `third_party/nv_video_pipeline/tests/MediaUrlTest.cpp`
- Create: `tests/test_sei_packet_cpp.py`

- [ ] **Step 1: Write failing byte-level and limit tests**

Verify exact H.264/H.265 NAL headers, payload type 5, the 16 UUID bytes, payload-size continuation bytes, rbsp trailing bits, and emulation-prevention insertion after `00 00` before `00`, `01`, `02`, or `03`. Parse the resulting EBSP back to RBSP in the test and compare the entire UUID+JSON payload.

Also verify:

- user payload above 1 MiB returns no SEI;
- source dimensions above 3840x2160 or RLE above 262144 runs skip SEI only;
- URL redaction retains scheme/host/port/path and removes query and fragment;
- no test log contains a supplied token sentinel.

Run:

```bash
pytest -q tests/test_sei_packet_cpp.py
```

Expected: emulation-prevention and limit cases fail initially.

- [ ] **Step 2: Make SEI creation a checked operation**

Change the helper to return `std::optional<std::vector<uint8_t>>`. Define shared constants:

```cpp
constexpr size_t kMaxSeiUserPayloadBytes = 1024 * 1024;
constexpr size_t kMaxSegmentationRuns = 262144;
constexpr int kMaxSegmentationWidth = 3840;
constexpr int kMaxSegmentationHeight = 2160;
```

Build payload type/size plus UUID/JSON/trailing bits as RBSP, convert RBSP to EBSP, and prepend the codec-specific Annex-B start code and NAL header. H.264 uses NAL type 6; H.265 uses prefix SEI NAL type 39.

- [ ] **Step 3: Skip only the SEI on overflow**

Expose segmentation dimensions/run count from `FrameSegmentationResult`. In `ZlmSeiOutputNode`, build and publish the original access unit even when metadata exceeds a bound; prepend SEI only when all bounds pass. Increment bounded counters and emit at most one redacted warning per 60 seconds. Do not mutate or suppress JSONL, HTTP, Kafka, or event outputs.

Change reconnect validation to 1000..4000 and use capped exponential delays of 1/2/4 seconds after consecutive failures. Reset the delay after a successful write.

- [ ] **Step 4: Redact all publication URL logs**

Use `redact_media_url(output_uri_)` in allocation, connect, publish, and write errors. Do not pass a raw FFmpeg error containing the URL through a log field.

- [ ] **Step 5: Run tests and commit**

```bash
pytest -q tests/test_sei_packet_cpp.py tests/test_deeplab_postprocess_cpp.py
git add third_party/nv_video_pipeline/src/utils/SeiPacket.h third_party/nv_video_pipeline/src/utils/SeiPacket.cpp third_party/nv_video_pipeline/src/utils/MediaUrl.h third_party/nv_video_pipeline/src/utils/MediaUrl.cpp third_party/nv_video_pipeline/src/nodes/ZlmSeiOutputNode.h third_party/nv_video_pipeline/src/nodes/ZlmSeiOutputNode.cpp third_party/nv_video_pipeline/src/objects/FrameInferenceResult.h third_party/nv_video_pipeline/src/objects/FrameInferenceResult.cpp third_party/nv_video_pipeline/CMakeLists.txt third_party/nv_video_pipeline/tests/SeiPacketTest.cpp third_party/nv_video_pipeline/tests/MediaUrlTest.cpp tests/test_sei_packet_cpp.py
git commit -m "fix(media): bound and redact SEI publication"
```

### Task 7: Pin mpegts.js and build the browser metadata core

**Files:**

- Modify: `package.json`
- Modify: `package-lock.json`
- Create: `src/media/contracts.ts`
- Create: `src/media/mpegtsAdapter.ts`
- Create: `src/media/syncQueue.ts`
- Create: `src/media/geometry.ts`
- Create: `src/media/pascalPalette.ts`
- Create: `src/media/contracts.test.ts`
- Create: `src/media/syncQueue.test.ts`
- Create: `src/media/geometry.test.ts`
- Create: `src/media/mpegtsAdapter.test.ts`

- [ ] **Step 1: Pin reviewed dependencies and add a unit-test command**

Use exact versions, not ranges:

```json
{
  "scripts": {
    "test:unit": "vitest run"
  },
  "dependencies": {
    "mpegts.js": "1.8.2"
  },
  "devDependencies": {
    "vitest": "4.1.11"
  }
}
```

Run `npm install`, then assert the lock entry has mpegts.js git head `791bac89e2ea8cbebe7f7ba247c3b342c09418cc` and integrity `sha512-cZYMa5muASH55wrS6JK5IGkUqnCA96c0f+WdnW91jIK9s/N6NzjmTQDJNYg9A5u/G2Q5nj2b3w1Zwzze/9guxA==`. Keep the package license in the production image through normal npm bundling/license collection.

- [ ] **Step 2: Write failing schema, queue, transform, and adapter tests**

Cover exact UUID bytes, UTF-8/JSON/schema-v2/task/revision checks, positive source dimensions, 1 MiB payload limit, 262144 RLE run limit, run sum equality, and rejection of unknown structured-result shapes.

Queue tests must prove max age 2 seconds, max length 120, selection of the newest entry whose PTS is not later than the current video time plus one frame tolerance, and clearing after 1 second without valid metadata. Geometry tests must cover pillarbox and letterbox `object-fit: contain` transforms. Adapter tests must prove upstream millisecond PTS becomes seconds.

Run:

```bash
npm run test:unit -- src/media
```

Expected: module-not-found failures.

- [ ] **Step 3: Implement strict project-owned contracts**

Decode `mpegts.Events.SEI_ARRIVED` only when type is user-data-unregistered and UUID exactly matches `9451ef8f-d241-496a-80ba-6818e24dc04e`. `TextDecoder` must use `{ fatal: true }`. Return a typed immutable envelope or a bounded diagnostic code; never throw raw metadata into React or logs.

Define typed results for primary/secondary detections, OCR detection/recognition, segmentation, and analytics while tolerating absent optional arrays. Bound labels/text to 256 UTF-8 characters and result arrays to 10000 items.

- [ ] **Step 4: Implement the isolated mpegts adapter and PTS queue**

Create the player with:

```ts
mpegts.createPlayer(
  { type: 'flv', isLive: true, url: descriptor.streamUrl },
  {
    enableWorker: true,
    enableStashBuffer: false,
    lazyLoad: false,
    liveBufferLatencyChasing: true,
  },
)
```

The adapter is the only module importing `mpegts.js`. It exposes normalized media errors, codec metadata, lifecycle methods, and SEI `{ uuid, userData, ptsSeconds }` events.

- [ ] **Step 5: Run tests, verify the package, and commit**

```bash
npm run test:unit -- src/media
npm view mpegts.js@1.8.2 gitHead dist.integrity
git add package.json package-lock.json src/media
git commit -m "feat(web): add validated SEI playback core"
```

Expected: tests pass and npm prints the reviewed git head and exact integrity above.

### Task 8: Render synchronized overlays in a Web Worker and Canvas

**Files:**

- Create: `src/media/segmentation.worker.ts`
- Create: `src/media/segmentationWorkerClient.ts`
- Create: `src/media/renderOverlay.ts`
- Create: `src/media/segmentation.worker.test.ts`
- Create: `src/media/renderOverlay.test.ts`
- Create: `src/media/fixtures/schema-v2-yolo.json`
- Create: `src/media/fixtures/schema-v2-ocr.json`
- Create: `src/media/fixtures/schema-v2-deeplab.json`

- [ ] **Step 1: Add failing deterministic rendering tests**

Create small schema-v2 fixtures for:

- primary and secondary YOLO boxes with class, score, track ID, and parent linkage;
- PPOCR quadrilaterals plus recognition text/confidence;
- configured areas and lines with analytics counts;
- an 8x4 `class-rle-v1` segmentation mask.

Test Pascal VOC colors for class IDs 0, 1, 2, 15, and 255; RLE run sum validation; source-size validation; and the 0.5 segmentation alpha. Use a fake Canvas 2D context to assert transformed coordinates, stable draw order, and no state leakage between frames.

Run:

```bash
npm run test:unit -- src/media/segmentation.worker.test.ts src/media/renderOverlay.test.ts
```

Expected: missing renderer and worker failures.

- [ ] **Step 2: Decode segmentation off the React/UI thread**

The worker accepts only a validated segmentation result and returns an `ImageBitmap` when supported or transferable RGBA bytes otherwise. Expand each `[classId, count]` run exactly once into the source-resolution raster, color it with the deterministic Pascal VOC palette, and make class 0 transparent. Abort and return a diagnostic code if dimensions, run count, class ID, or total pixels violate the contract. The browser must never perform logits resize or argmax.

- [ ] **Step 3: Implement one Canvas renderer for all result types**

Draw in this order:

1. segmentation mask at alpha 0.5;
2. configured areas and lines;
3. primary detections;
4. secondary detections and parent linkage;
5. OCR polygons and text;
6. analytics counts.

Use the `object-fit: contain` transform from Task 7 for every coordinate. Scale strokes and labels for readability but keep screen-space line width bounded from 1 through 4 CSS pixels. Clamp text boxes inside the visible video rectangle and never write inference text into HTML.

- [ ] **Step 4: Run tests and commit**

```bash
npm run test:unit -- src/media
git add src/media/segmentation.worker.ts src/media/segmentationWorkerClient.ts src/media/renderOverlay.ts src/media/segmentation.worker.test.ts src/media/renderOverlay.test.ts src/media/fixtures
git commit -m "feat(web): render schema-v2 inference overlays"
```

### Task 9: Build the shared player and replace both preview surfaces

**Files:**

- Create: `src/media/useInferenceStreamPlayer.ts`
- Create: `src/components/InferenceStreamPlayer.tsx`
- Create: `src/components/InferenceStreamPlayer.test.tsx`
- Create: `src/components/MediaGatewaySettings.tsx`
- Delete: `src/api/useInferencePreviewStream.ts`
- Delete: `src/components/InferencePreview.tsx`
- Modify: `src/types.ts`
- Modify: `src/api/client.ts`
- Modify: `src/pages/InferencePage.tsx`
- Modify: `src/pages/VideoWallPage.tsx`
- Modify: `src/pages/SettingsPage.tsx`
- Modify: `src/styles.css`
- Modify: `scripts/smoke.mjs`

- [ ] **Step 1: Write failing player state and lifecycle tests**

Use fake API, mpegts adapter, `requestVideoFrameCallback`, clock, and worker implementations. Prove:

- HTTPS page plus `ws://` descriptor enters `unsupported` before constructing mpegts;
- unavailable capability maps to explicit non-playing UI;
- each connection/reconnection obtains a fresh 60-second playback descriptor;
- the default retry sequence is 1, 2, 4, 4 seconds and stops on unmount/task stop;
- 401/403 enters `unauthorized`, unsupported H.265 enters `codec_unsupported`, and missing publication enters `waiting_publish`;
- valid SEI changes state to `live`; video without valid metadata for one second becomes `metadata_degraded` and clears Canvas;
- queue selection uses `requestVideoFrameCallback`'s `mediaTime`, never `Date.now()`;
- one failed video-wall tile does not recreate or pause another tile.

Run:

```bash
npm run test:unit -- src/components/InferenceStreamPlayer.test.tsx src/media
```

Expected: component/hook missing.

- [ ] **Step 2: Implement the player hook as an explicit state machine**

The hook owns descriptor acquisition, mpegts attach/load/play, SEI validation, bounded PTS queue, frame callback loop, segmentation worker, exponential reconnect, and full cleanup. It accepts task ID, revision, capability, and analytics geometry. It returns only state, diagnostic code, video/canvas refs, and retry/stop controls.

Detect H.264 as mandatory. Permit H.265 only when both mpegts media info and `MediaSource.isTypeSupported(...)` confirm support; otherwise show `codec_unsupported`. Do not transcode or switch transport.

- [ ] **Step 3: Implement the stable video/Canvas component**

Use one aspect-ratio-constrained media surface with an unframed `<video muted playsInline>` and an absolute, pointer-events-none `<canvas>`. A `ResizeObserver` keeps device-pixel backing dimensions synchronized without changing layout. Use compact status overlays that do not cover player controls or other page content.

- [ ] **Step 4: Add gateway management and typed task selection**

Add gateway types and client methods matching Task 2. `MediaGatewaySettings` supports list, create/update, enable/disable, secret replacement, probe, configured-state display, and deletion. It displays separate node publish, browser playback, and platform API host/port inputs. Secret fields are write-only and never repopulated.

In the inference task editor, replace arbitrary `outputUri` with an online-gateway selector, validated stream name, enabled toggle, and reconnect delay. Show `media_migration_required` as an administrator action, not a generic task failure.

- [ ] **Step 5: Replace single and wall preview callers**

Use `InferenceStreamPlayer` on the task detail and every four/six-view wall tile. Remove `<img>` polling and the old hook/component imports. Keep each wall player's resources and retry state independent.

- [ ] **Step 6: Run frontend verification and commit**

```bash
npm run test:unit
npm run build
npm run test:ui
git add -A package.json package-lock.json src scripts/smoke.mjs
git commit -m "feat(web): replace MJPEG with direct SEI player"
```

Expected: unit, type-check, production build, and UI smoke tests pass.

### Task 10: Build the managed ZLMediaKit image and online Compose service

**Files:**

- Create: `deploy/media/Dockerfile`
- Create: `deploy/media/config.ini.template`
- Create: `deploy/media/render_config.py`
- Create: `deploy/media/zlm-base-image.lock`
- Create: `deploy/media/zlm-candidate-verification.json`
- Create: `scripts/lock_zlm_base_image.py`
- Create: `scripts/verify_zlm_candidate.py`
- Create: `scripts/configure_media_secrets.py`
- Modify: `scripts/build_offline_images.sh`
- Modify: `deploy/compose.yaml`
- Modify: `deploy/.env.example`
- Modify: `backend/platform_api/settings.py`
- Modify: `backend/platform_api/app.py`
- Test: `tests/test_media_image.py`
- Test: `tests/test_self_contained_deploy.py`

- [ ] **Step 1: Add failing image-lock, config, and Compose tests**

Require:

- a full `zlmediakit/zlmediakit@sha256:` digest in `zlm-base-image.lock`, never `:master` in a Docker `FROM` or Compose service;
- wrapper image `rknode-platform-media:2026.08.20` with the same required OCI labels as other offline images;
- config with HTTP-FLV/WS-FLV and RTSP enabled, HLS/recording disabled, bounded logs, `admin_params` empty, and only `on_publish`, `on_play`, and `on_server_keepalive` Hooks enabled;
- rendered `general.mediaServerId` from the Hook identity and a separate API secret;
- the `media` service on the platform network with host ports 8554/TCP and 8081/TCP, health check, restart policy, and bounded Docker logs;
- Web default port 5173 in online and offline central deployment;
- API startup seeds/updates only `gateway_builtin` from explicit environment values and stores secrets through `MediaSecretStore`.

Run:

```bash
pytest -q tests/test_media_image.py tests/test_self_contained_deploy.py
```

Expected: missing lock/image/service failures.

- [ ] **Step 2: Implement the candidate verifier, test an immutable image, then lock it**

The upstream `master` tag is mutable. Use it only to discover a candidate digest:

```bash
docker pull zlmediakit/zlmediakit:master
candidate_image="$(docker image inspect --format '{{index .RepoDigests 0}}' zlmediakit/zlmediakit:master)"
```

Implement `verify_zlm_candidate.py` to accept `--image` and `--record`, then pass the immutable RepoDigest to it. The script starts only the named disposable ZLM/API/Vite fixture, publishes generated H.264 plus the fixed UUID/schema-v2 SEI over authenticated RTSP, opens direct WS-FLV in Playwright through the Task 9 adapter, and requires both nonblank video pixels and the expected SEI event. It preserves logs on failure and removes only its own temporary resources on success. Its machine-readable JSON record contains the tested RepoDigest, UTC time, fixture hash, and pass/fail checks, but no secrets.

Only after that end-to-end check passes, run:

```bash
python scripts/verify_zlm_candidate.py \
  --image "$candidate_image" \
  --record deploy/media/zlm-candidate-verification.json
python scripts/lock_zlm_base_image.py \
  --image "$candidate_image" \
  --verification deploy/media/zlm-candidate-verification.json \
  --output deploy/media/zlm-base-image.lock
```

The lock script must reject tag-only inspect output, require the candidate verification record produced by `verify_zlm_candidate.py`, and atomically write one full immutable RepoDigest. This generated release lock is intentional; do not put an unverified digest or a textual placeholder in source. Task 13 broadens this same fixture into the full authorization, reconnect, rendering, and latency matrix; any later failure invalidates the candidate and blocks the release.

- [ ] **Step 3: Build a labeled project wrapper image**

`deploy/media/Dockerfile` accepts a required `ZLM_BASE_IMAGE` build argument, copies the maintained template and renderer, preserves `/opt/media/bin` as workdir, and executes:

```text
./MediaServer -s default.pem -c ../conf/config.ini -l 0
```

`build_offline_images.sh platform` reads and validates the lock, builds API, Web, and Media images, and passes the existing release/source/offline-ready OCI labels. The renderer uses Python already present in the upstream image, validates every required environment value, writes `/opt/media/conf/config.ini` without echoing secrets, sets mode `0600`, then `exec`s MediaServer.

- [ ] **Step 4: Configure built-in gateway bootstrap without address inference**

Add required central deployment variables:

```text
RKNODE_MEDIA_PUBLISH_HOST=192.168.1.10
RKNODE_MEDIA_PLAYBACK_HOST=192.168.1.10
RKNODE_MEDIA_RTSP_PORT=8554
RKNODE_MEDIA_WS_PORT=8081
```

Generate the two 256-bit secrets directly into the permission-restricted deployment environment instead of documenting sample secret values:

```bash
cp deploy/.env.example deploy/.env
python scripts/configure_media_secrets.py --env-file deploy/.env
```

The script sets `umask 077`, preserves unrelated environment entries, creates `RKNODE_ZLM_API_SECRET` and `RKNODE_ZLM_HOOK_IDENTITY` with `secrets.token_hex(32)` only when absent, and never prints either value.

The example IP is documentation only; Compose must fail validation if either externally reachable host remains unset. API control uses internal host `media` and port 80. On startup, create `gateway_builtin` if absent; update only its deployment-owned connectivity fields if it already exists, preserving task bindings and operator-visible identity. The media service and API receive the same two secrets, while the Web service receives neither.

- [ ] **Step 5: Add the maintained ZLM template**

Use `/opt/media/conf/config.ini`, the path loaded by the official image command. Configure:

```ini
[api]
secret=${RKNODE_ZLM_API_SECRET}
apiDebug=0

[general]
mediaServerId=${RKNODE_ZLM_HOOK_IDENTITY}

[hook]
enable=1
admin_params=
alive_interval=10
timeoutSec=5
retry=1
retry_delay=1
on_publish=http://api:8000/api/v1/media-hooks/zlm/gateway_builtin/on-publish
on_play=http://api:8000/api/v1/media-hooks/zlm/gateway_builtin/on-play
on_server_keepalive=http://api:8000/api/v1/media-hooks/zlm/gateway_builtin/on-server-keepalive
```

Complete the template with explicit RTSP 554 and HTTP 80 container ports, cross-origin HTTP enabled for LAN browser playback, FLV enabled, and unused recording/HLS/shell features disabled. Do not enable TLS because the approved deployment has no domain/certificate.

- [ ] **Step 6: Run deploy tests and commit**

```bash
pytest -q tests/test_media_image.py tests/test_self_contained_deploy.py tests/test_offline_deploy.py
python scripts/configure_media_secrets.py --env-file deploy/.env
docker compose --env-file deploy/.env -f deploy/compose.yaml config >/dev/null
git add deploy/media scripts/lock_zlm_base_image.py scripts/verify_zlm_candidate.py scripts/configure_media_secrets.py scripts/build_offline_images.sh deploy/compose.yaml deploy/.env.example backend/platform_api/settings.py backend/platform_api/app.py tests/test_media_image.py tests/test_self_contained_deploy.py
git commit -m "feat(deploy): add pinned ZLMediaKit service"
```

Expected: tests pass and Compose expands without an unpinned image or secret in Web configuration.

### Task 11: Update node/central Compose and the offline bundle

**Files:**

- Modify: `deploy/rk3588/compose.yaml`
- Modify: `deploy/rk3588/.env.example`
- Modify: `deploy/nodes/rk3588/compose.yaml`
- Modify: `deploy/nodes/rk3588/.env.example`
- Modify: `deploy/offline/rk3588/compose.inference.yaml`
- Modify: `deploy/offline/rk3588/inference.env.example`
- Modify: `deploy/offline/rk3588/node.env.example`
- Modify: `deploy/offline/platform/compose.yaml`
- Modify: `deploy/offline/platform/.env.example`
- Modify: `scripts/package_offline_bundle.py`
- Modify: `deploy/offline/common/verify.sh`
- Test: `tests/test_offline_deploy.py`
- Test: `tests/test_self_contained_deploy.py`

- [ ] **Step 1: Extend failing deployment matrix tests**

Assert all RK3588 inference Compose variants have no preview environment or preview directory and still mount permission-restricted desired revision data. Assert platform offline bundles include API, Web, and Media images with architecture/version/offline-ready labels and SHA-256 archive checksums. Verify no offline script pulls or builds on the target.

Run:

```bash
pytest -q tests/test_offline_deploy.py tests/test_self_contained_deploy.py
```

Expected: old preview variables and two-image platform bundle fail.

- [ ] **Step 2: Remove every node preview variable and path**

Delete `RKNODE_PREVIEW_*` from online/offline RK3588 Compose and examples. Keep node platform URL, enrollment, runtime state, output sinks, devices, and models intact. Do not expose publisher credentials as environment variables; they stay in desired revision files.

- [ ] **Step 3: Package three central images in one platform archive**

Change the platform `BundleSpec.images` to:

```python
(
    "rknode-platform-api:{version}",
    "rknode-platform-web:{version}",
    "rknode-platform-media:{version}",
)
```

Keep one `rknode-platform-amd64-2026.08.20.tar` bundle containing the three image archives and manifests. The total release now has eight images in six role tar bundles: three central images, four trainer variants, and one shared RK3588 image used by converter and inference roles.

- [ ] **Step 4: Make offline verification include media health and ports**

Verify platform readiness through port 5173, MediaServer container health, host TCP ports 8554 and 8081, and the built-in gateway's `online` state after authenticated keepalive. `deploy.sh` remains `--pull never --no-build`.

- [ ] **Step 5: Run tests and commit**

```bash
pytest -q tests/test_offline_deploy.py tests/test_self_contained_deploy.py tests/test_no_legacy_preview.py
git add deploy/rk3588 deploy/nodes/rk3588 deploy/offline/rk3588 deploy/offline/platform scripts/package_offline_bundle.py deploy/offline/common/verify.sh tests/test_offline_deploy.py tests/test_self_contained_deploy.py
git commit -m "feat(offline): ship media gateway in platform bundle"
```

### Task 12: Rewrite operator documentation and technical deck content

**Files:**

- Modify: `docs/system-guide.md`
- Modify: `docs/simple-node-deployment.md`
- Modify: `docs/offline-deployment.md`
- Modify: `deploy/rk3588/runtime-adapter/README.md`
- Modify: `scripts/generate_platform_ppt.py`
- Modify: `scripts/validate_platform_ppt.py`
- Regenerate: platform PPT output selected by `scripts/generate_platform_ppt.py`
- Test: `tests/test_no_legacy_preview.py`

- [ ] **Step 1: Add failing documentation assertions**

Extend the legacy guard and PPT validator to reject JPEG/MJPEG preview claims, arbitrary ZLM `outputUri`, seven-image wording, platform port 8080, and diagrams that route media bytes through the API.

- [ ] **Step 2: Document first-deployment address selection and security**

Explain with concrete commands that:

- for an example central host at `192.168.1.10`, platform Web is `http://192.168.1.10:5173`;
- nodes publish to `rtsp://192.168.1.10:8554` through a platform-issued credential;
- browsers play `ws://192.168.1.10:8081` through a 60-second credential;
- the platform controls the built-in ZLM internally at `media:80`;
- external gateways may have three different hosts because node, browser, and API routes differ;
- plain HTTP/WS is restricted to a trusted LAN, VPN, or SSH tunnel; an HTTPS page cannot use plain WS;
- firewall checks use `ss`, `curl`, and `nc` for 5173, 8554, and 8081;
- the platform node token remains independent from media publish/play tokens.

Include exact gateway creation/probe UI/API flow, task migration procedure, player-state troubleshooting, ZLM keepalive/auth checks, rollback, and secret rotation without displaying secret values.

- [ ] **Step 3: Update release/image/offline explanations**

Document API/Web/Media `2026.08.20`, unchanged trainer `2026.08.15` images, and RK3588 `2026.08.20-business`. State clearly why eight images still produce six tar bundles. Explain that the exact pinned upstream ZLM digest is recorded in `deploy/media/zlm-base-image.lock` and verified before packaging.

- [ ] **Step 4: Update and regenerate the technical deck**

Replace the old preview path with this flow:

```text
RTSP camera -> RKMPP/inference -> original H264/H265 + SEI
            -> authenticated RTSP -> ZLMediaKit
            -> authenticated WS-FLV -> browser video + Canvas
```

Show the API as control plane only, the three gateway origins, separate publisher/player tokens, built-in/external gateway modes, and current ports 5173/8554/8081. Do not claim H.265 universal support, audio, public Internet TLS, or completed board acceptance.

- [ ] **Step 5: Validate docs and commit**

```bash
python scripts/generate_platform_ppt.py
python scripts/validate_platform_ppt.py
pytest -q tests/test_no_legacy_preview.py tests/test_offline_deploy.py
git add docs deploy/rk3588/runtime-adapter/README.md scripts/generate_platform_ppt.py scripts/validate_platform_ppt.py
git commit -m "docs: describe RTSP SEI deployment and operation"
```

### Task 13: Add real ZLM/WS-FLV/SEI browser compatibility and latency gates

**Files:**

- Create: `tests/media/compose.yaml`
- Create: `tests/media/config.ini`
- Create: `tests/media/fixture-page.html`
- Create: `tests/media/fixture-player.ts`
- Create: `scripts/run_media_e2e.py`
- Create: `scripts/media_latency_report.py`
- Modify: `package.json`
- Modify: `scripts/smoke.mjs`

- [ ] **Step 1: Create a failing one-command end-to-end test**

Add:

```json
{
  "scripts": {
    "test:media-e2e": "python scripts/run_media_e2e.py"
  }
}
```

The runner uses the immutable ZLM base digest, an isolated temporary Compose project, the real FastAPI Hook routes, FFmpeg, the built frontend modules, and Playwright Chromium. It must clean up only its named containers/network/temporary directory and preserve logs on failure.

- [ ] **Step 2: Generate a deterministic H.264+SEI source at test time**

Use FFmpeg `testsrc2`, `libx264`, and the `h264_metadata=sei_user_data` bitstream filter with UUID `9451ef8f-d241-496a-80ba-6818e24dc04e`. Inject a schema-v2 fixture on frames with known PTS, publish it by RTSP with a valid platform publication token, and use stream copy into ZLM. Do not commit a large opaque video binary.

The fixture must exercise a detection box and small segmentation mask. A second run injects a foreign UUID and proves no overlay appears.

- [ ] **Step 3: Verify the actual direct browser path**

Playwright must connect to ZLM over WS-FLV, not an API media endpoint. Assert nonblank video pixels, received `SEI_ARRIVED`, matching Canvas pixels/coordinates, task/revision rejection, 401/403 behavior, and recovery after forcibly closing then republishing the stream. Inspect network requests and fail if FLV response bytes come from port 5173/API.

- [ ] **Step 4: Measure synchronized latency**

Embed source PTS/frame index and measure at `requestVideoFrameCallback`. Produce machine-readable JSON with sample count, p50, p95, max, overlay skew in frames, and reconnect duration. Gate:

- single YOLO/OCR p95 end-to-overlay latency at most 1.0 second;
- each tile in four/six wall p95 at most 1.5 seconds;
- overlay skew at most one frame;
- reconnect at most 5 seconds;
- no queue above 120 entries or older than 2 seconds.

DeepLab is reported separately at source resolution; it must preserve official semantics even if its measured limit requires documented task resolution/frame-rate constraints.

- [ ] **Step 5: Run the compatibility matrix and commit**

```bash
npm run test:media-e2e
npm run test:unit
npm run build
git add tests/media scripts/run_media_e2e.py scripts/media_latency_report.py package.json package-lock.json scripts/smoke.mjs
git commit -m "test(media): verify ZLM SEI browser pipeline"
```

Repeat browser acceptance on current and immediately previous major desktop Chrome and Edge. Firefox, Safari, iOS, and Android results are informative and do not block this release.

### Task 14: Run the full quality, release, offline, rollback, and RK3588 hardware gates

**Files:**

- Modify as failures require: files already owned by Tasks 1-13
- Create: `release/2026.08.20/media-acceptance.json`
- Create: `release/2026.08.20/image-manifest.json`
- Create: `release/2026.08.20/SHA256SUMS`

- [ ] **Step 1: Run the complete repository verification**

```bash
pytest -q
npm run test:unit
npm run build
npm run test:ui
npm run test:media-e2e
python scripts/validate_platform_ppt.py
docker compose --env-file deploy/.env -f deploy/compose.yaml config >/dev/null
```

Expected: every command exits 0. Fix regressions at their owning task and rerun both focused and complete tests.

- [ ] **Step 2: Build only changed release images under new date-based tags**

Do not overwrite or delete rollback images. Build:

```bash
RKNODE_RELEASE_VERSION=2026.08.20 RKNODE_SOURCE_REVISION=rtsp-sei-2026.08.20 scripts/build_offline_images.sh platform
```

On an authorized arm64/RK3588 build host, build:

```bash
RKNODE_RELEASE_VERSION=2026.08.20-business RKNODE_SOURCE_REVISION=rtsp-sei-business-2026.08.20 scripts/build_offline_images.sh rk3588
```

This yields new API, Web, Media, and RK3588 images. Trainer code is unchanged, so retain the four verified `2026.08.15` trainer images. Inspect architecture, version, source revision, offline-ready labels, entrypoints, and image IDs; record exact values in `image-manifest.json`.

- [ ] **Step 3: Package and verify six offline role archives**

Package the `2026.08.20` central bundle, four retained trainer bundles, and `2026.08.20-business` RK3588 bundle. Generate SHA-256 checksums, load them on clean compatible hosts with network access disabled, run each bundle's `verify.sh`, and record the result. The platform archive must contain three images; the RK3588 archive still serves converter and inference containers.

- [ ] **Step 4: Run an authorized RK3588 board test before claiming acceptance**

Obtain current explicit authorization and connection details from the operator; do not reuse credentials from an earlier conversation. On the board:

1. preserve the currently running image tag/ID, Compose config, desired revision, and logs;
2. load the new RK3588 archive without deleting the old image;
3. start an RTSP/RKMPP H.264 task for each supported model family;
4. verify publication Hook, direct browser WS-FLV, schema-v2 SEI, and correct overlays;
5. compare DeepLab output pixel-for-pixel with `scripts/deeplabv3_rknn_infer.py` on the same input;
6. verify multi-context/multi-worker scheduling remains healthy;
7. run reconnect, stop/restart, credential revocation, and six-tile tests;
8. record device/kernel/RKNN/runtime/image IDs and measurements in `media-acceptance.json`.

H.264, YOLO, OCR, official DeepLab semantics, resource stability, and lifecycle revocation are blocking. H.265 is recorded as supported or `codec_unsupported` for each accepted browser; it is not transcoded.

- [ ] **Step 5: Prove full-set rollback before cleanup**

Back up `platform-data`, media secret storage, Compose environment, and image IDs. Roll back API/Web/Media/RK3588 together to the recorded old set, restore the database backup when schema compatibility requires it, and verify old service readiness. Then redeploy the accepted new set and rerun health/player checks.

Do not mix the new task media contract with an old API/frontend/node image. Do not remove old images, dangling layers, old preview data, or backups until new deployment and rollback have both passed and the operator separately authorizes cleanup.

- [ ] **Step 6: Commit release evidence**

```bash
git add release/2026.08.20/media-acceptance.json release/2026.08.20/image-manifest.json release/2026.08.20/SHA256SUMS
git commit -m "chore(release): record RTSP SEI acceptance"
```

## Final Acceptance Checklist

- [ ] Original H.264 access units reach the browser through RTSP -> ZLM -> WS-FLV without re-encoding.
- [ ] H.265 either plays unchanged on a proven browser or shows `codec_unsupported`.
- [ ] The platform never carries FLV response bodies.
- [ ] Publish, play, API, and Hook identities are separate, role-bound, redacted, and default-deny.
- [ ] A new 60-second play token is issued on every reconnect.
- [ ] Only the fixed UUID/schema/task/revision metadata reaches the overlay queue.
- [ ] Queue, payload, source-dimension, and RLE limits are enforced.
- [ ] DeepLab resizes every logit plane bilinearly before argmax and every sink receives the same corrected mask.
- [ ] Single task and four/six wall use one player implementation and meet their measured latency gates.
- [ ] Legacy JPEG/MJPEG code, routes, settings, environment, docs, and UI are absent.
- [ ] Online and offline platform deployments use the same accepted pinned ZLM digest.
- [ ] The release has eight images in six tar bundles, with exact labels and checksums.
- [ ] Full-set rollback and authorized RK3588 hardware acceptance evidence are recorded before cleanup.
