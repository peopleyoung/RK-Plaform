# Runtime adapter contract

The inference agent mounts this directory at `/opt/rknode/runtime-adapter`.
Production nodes configure four commands, for example:

```text
RKNODE_SELF_TEST_COMMAND=/opt/rknode/runtime-adapter/self-test.sh
RKNODE_MODEL_PROBE_COMMAND=/opt/rknode/runtime-adapter/probe-model.sh
RKNODE_RUNTIME_COMMAND=/opt/rknode/runtime-adapter/activate-task.sh
RKNODE_RUNTIME_HEALTH_COMMAND=/opt/rknode/runtime-adapter/health.sh
```

`self-test.sh` validates the NPU device, required binaries and advertised
adapter registry. It also recovers the persisted `current` revision after an
agent/container restart. `probe-model.sh` validates the staged artifact,
manifest, labels, output contract and task configuration without opening a
second live NPU inference context.

The activation command is executed once per desired node revision after every
release has been downloaded, checksummed and probed. Its primary contract is:

- `RKNODE_DESIRED_REVISION`
- `RKNODE_RELEASE_CONFIGS` (JSON array of release/model/task groups)

Each release group contains `releaseId`, `adapter`, `modelPath`,
`manifestPath`, and `tasks`. The older single-release fields remain available
to model probes and compatibility integrations:

- `RKNODE_TASK_ID`
- `RKNODE_RELEASE_ID`
- `RKNODE_MODEL_PATH`
- `RKNODE_MANIFEST_PATH`
- `RKNODE_ADAPTER`
- `RKNODE_INPUT_URI`
- `RKNODE_TASK_CONFIG`
- `RKNODE_TASK_CONFIGS` (JSON array of all tasks sharing this release/context)

`activate-task.sh` generates immutable JSON/YAML-compatible configs under
`RKNODE_RUNTIME_STATE_DIR/revisions/<revision>`, groups tasks with identical
release, adapter, threshold, NPU core and context/worker parameters into one RKNN instance, stops the old
pipeline, runs `rknn_instance_probe` against every generated instance, switches
`current`, starts `rknn_pipeline`, and waits for its readiness file. Startup or
probe failure restores the old `current` revision and process. The successful
old revision is retained as `previous`.

`health.sh` verifies the persisted revision, PID start time and readiness file.
With `RKNODE_AUTO_RECOVER=true` it restarts the current revision when the
pipeline exits. `RKNODE_RELEASE_CONFIGS=[]` writes an empty revision and stops
the old process. JSONL files are written under `RKNODE_INFERENCE_OUTPUT_DIR`.
Pipeline stdout/stderr is inherited by the inference service container and is
therefore bounded by the Compose `json-file` log rotation settings. Do not
redirect it to an append-only file inside the persistent model/state volume.

`deploy/rk3588/Dockerfile.node` compiles the bundled
`third_party/nv_video_pipeline` source during its builder stage. No separate
source checkout or prebuilt pipeline image is required. The resulting image contains Python, the agent,
`rknn_pipeline`, `rknn_instance_probe`, and the matching `librknnrt`; it does
not require the Docker socket. Outside explicit staging-only mode, any missing
command or non-zero exit prevents a healthy report.

Each task may include `npuCoreMask` (`auto`, `core0`, `core1`, `core2`, `core0_1`,
or `core0_1_2`) and `npuCorePolicy` (`shared` or `exclusive`). The generated
instance YAML carries these as `core_mask` and `core_policy`. The C++ RKNN
adapter applies the mask with `rknn_set_core_mask` immediately after
`rknn_init`; an invalid mask or runtime failure aborts activation and preserves
the previous revision. Exclusive assignments are rejected when their masks
overlap another runtime context.

Each primary task also accepts `contextCount` and `workerCount`. Both default to
`1`, must be positive integers, and `workerCount` cannot exceed
`contextCount`. Every `analytics.secondaryModels` item accepts an independent
pair with the same rules:

```json
{
  "contextCount": 3,
  "workerCount": 2,
  "analytics": {
    "secondaryModels": [
      {
        "releaseId": "secondary-release-id",
        "sourceClassIds": [0],
        "confidenceThreshold": 0.25,
        "contextCount": 2,
        "workerCount": 1
      }
    ]
  }
}
```

The generated instance contains `context_count`, `worker_count`, and a derived
`queue_capacity=max(8, worker_count*2)`. The first context is created with
`rknn_init`; the remaining contexts use `rknn_dup_context`. Each worker leases
one context for the complete inference and decode operation, so one context is
never used concurrently. Every context receives the pool's configured core
mask.

Equivalent primary or secondary configurations share one pool. Changing either
count creates a distinct pool in the next immutable revision. Node
`maxModelInstances` capacity counts the sum of actual contexts once per unique
pool; it does not count tasks or workers. A partial context creation failure
aborts activation, destroys the new pool, and leaves the previous revision
active.

## RK3588 media graph

The optional task `media` object is converted into independent graph branches:

```json
{
  "decoder": "rkmpp",
  "tracking": {"enabled": true, "trackBuffer": 30},
  "kafka": {
    "enabled": true,
    "brokers": "kafka-1:9092,kafka-2:9092",
    "topic": "sei_msg",
    "key": ""
  },
  "zlmSei": {
    "enabled": true,
    "publishUri": "rtsp://192.168.1.10:8554/live/line-a-result?publishToken=opaque",
    "reconnectMs": 1000
  }
}
```

`rkmpp` accepts RTSP H.264/H.265 and uses Rockchip MPP for decode. The encoded
access unit is kept beside the decoded frame, so the ZLM branch remuxes the
original compressed stream and does not encode video again. `tracking` is only
valid for YOLO detection adapters. Kafka and ZLM consume the same `anhuan_v1`
JSON envelope after tracking; an empty Kafka key is derived from the last two
input URI path segments. Both destinations use bounded/retry behavior and do
not stop local JSON or inference when the remote service is down. The platform
creates `publishUri` from the selected managed gateway and sends it only in the
node desired state; operators do not enter a complete publication URL.

ZLM SEI requires `decoder=rkmpp` because OpenCV capture does not retain the
original encoded packet. Activation rejects incompatible combinations before
the active revision is replaced.

## Compile inside the inference container

The production inference image intentionally includes the compiler, CMake,
development headers, the pinned source tree, RKNN SDK snapshot, and Rockchip
MPP snapshot. Rebuild into a temporary directory without modifying the running
binary:

```bash
/opt/rknode/runtime-adapter/build-runtime.sh
```

The result stays in `/tmp/rknode-build`. Replacing `/usr/local/bin` or
`/usr/local/lib` in a running container is not an upgrade procedure; rebuild
and retag the image, then recreate the service through Compose. Set
`RKNODE_RUNTIME_SOURCE_DIR`, `RKNODE_RUNTIME_BUILD_DIR`, or
`RKNODE_BUILD_JOBS` when a bind-mounted source tree or another output path is
required. The script does not access the network.

When the verified previous node image is already loaded but the upstream
Toolkit image is unavailable, rebuild the complete node image without APT/Pip
or registry access:

```bash
docker build --platform linux/arm64 \
  -f deploy/rk3588/Dockerfile.node.rebuild \
  --build-arg RKNODE_BASE_IMAGE=rknode-rk3588-node:2026.08.20-business \
  --build-arg RKNODE_RELEASE_VERSION=2026.08.20-business \
  --build-arg RKNODE_BUILD_JOBS=2 \
  -t rknode-rk3588-node:2026.08.20-business .
```

The rebuild file removes the old source/application trees before copying the
current workspace, recompiles all native binaries, and emits another flattened
image. It never changes or deletes the base tag.
