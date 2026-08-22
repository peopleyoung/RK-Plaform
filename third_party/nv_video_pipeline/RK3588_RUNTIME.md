# RK3588 C++ Deployment

RK3588/RKNN is the repository's sole production runtime. It supports the
OpenCV compatibility capture path and an RTSP H.264/H.265 Rockchip MPP hardware
decode path, executes converted models through `librknnrt`, and writes typed
inference results.

## Model Conversion

Run conversion with RKNN Toolkit2 2.3.2:

```bash
python3 tools/rknn/convert_models.py \
  --source-dir bin \
  --output-dir models/rk3588
```

The converter reads `bin/bytetrack_s.onnx` and `bin/aqm.onnx`, exports FP16
models, and writes `manifest.json`. The manifest records Toolkit version,
source/output hashes, preprocessing, and the expected flat output contract.

## Native Build

The validated RKNN 2.3.2 header, ARM64 runtime, and upstream license are under
`3rdparty/rknpu2`:

```bash
cmake -S . -B build-rknn \
  -DRKNN_SDK_ROOT="$PWD/3rdparty/rknpu2" \
  -DROCKCHIP_MPP_ROOT="$PWD/3rdparty/rockchip-mpp" \
  -DRKNODE_WITH_RKMPP=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rknn --parallel 4
```

Build outputs are written to `build-rknn/bin`. Inspect both converted models
on the board before starting the graph:

```bash
LD_LIBRARY_PATH=3rdparty/rknpu2/Linux/aarch64 \
  build-rknn/bin/rknn_smoke \
  models/rk3588/bytetrack_s.rknn \
  models/rk3588/aqm.rknn
```

For an activation preflight, initialize and warm every generated instance after
the old runtime has drained:

```bash
build-rknn/bin/rknn_instance_probe \
  /data/runtime/revisions/12/instances.yaml \
  release-0 release-1
```

## Container

The multi-stage image uses an ARM64 Toolkit2 build image and a Debian 12
runtime:

```bash
docker build -f docker/Dockerfile.rk3588 \
  --build-arg RKNN_TOOLKIT_IMAGE=rknn_toolkit2:2.3.2-debian12-cp311-aarch64 \
  --build-arg RKNN_RUNTIME_IMAGE=debian:bookworm \
  -t nv-video-pipeline:rk3588 .
```

`DEBIAN_MIRROR` and `DEBIAN_SECURITY_MIRROR` may override package download
endpoints without changing package versions.

Validate both models through the production image:

```bash
docker run --rm \
  --device /dev/dri/card0:/dev/dri/card0 \
  -v "$PWD/models/rk3588:/models:ro" \
  --entrypoint /usr/local/bin/rknn_smoke \
  nv-video-pipeline:rk3588 \
  /models/bytetrack_s.rknn /models/aqm.rknn
```

The compose file maps `/dev/dri/card0` for the NPU plus the board's media
devices, mounts models and the deterministic test image read-only, uses host
networking for RTSP, and writes results outside the container. On the validated
board, `renderD128` alone does not expose the NPU; privileged mode is not
required.

```bash
mkdir -p output/rk3588
docker compose -f docker/compose.rk3588.yml build
docker compose -f docker/compose.rk3588.yml up --abort-on-container-exit
```

With an empty output directory, the finite image run appends one JSONL row for
each configured model and exits after the configured duration. Replace the
test image path with a file or RTSP URL for live deployment. `JsonOutputNode`
also accepts an HTTP result sink:

```yaml
output:
  type: http
  url: http://127.0.0.1:8080/inference/results
  connect_timeout_ms: 500
  request_timeout_ms: 2000
  authorization_env: RKNODE_RESULT_SINK_TOKEN
```

The node posts the same structured JSON envelope as `RKNN_RESULT` once per
completed frame. The optional bearer token is read from the named environment
variable. Timeouts are bounded, and non-2xx or transport errors are logged
without terminating the pipeline; consumers should deduplicate by
`frame_index`. Keep the JSONL sink when a durable local audit copy is needed.

## RTSP SEI Publication

`ZlmSeiOutputNode` remuxes the original RKMPP H.264/H.265 access unit and adds
the schema-v2 structured result as user-data-unregistered SEI. It publishes to
the managed RTSP URL supplied in node desired state and does not re-encode or
draw on the video pixels. The browser plays the derived WS-FLV stream directly
from the media gateway and renders synchronized overlays in Canvas.

## Platform YOLO Releases

Platform releases with output contract `rknn_yolo_dfl_split_heads_v1` use the
`YOLO_DFL_SPLIT` instance type shown in
`config/rk3588/instances.platform.example.yaml`. The adapter discovers DFL box
and class tensors by grid and channel count, accepts both paired six-output
graphs and Rockchip nine-output graphs with score-sum auxiliaries, applies DFL
decode and class-aware NMS, and preserves the existing JSONL detection schema.
Rockchip's official export already emits sigmoid class probabilities, so the
default does not apply sigmoid again. Set `class_scores_logits: true` only for
a verified non-official graph that exposes raw class logits.

Do not advertise `yolo_dfl_split_v1` in the node agent until this build is
installed on the board. The current production default is one serial RKNN
context per model release; pipelines referencing the same instance share that
context and its bounded queue.

Generated platform instances may set `core_mask` to `auto`, `core0`, `core1`,
`core2`, `core0_1`, or `core0_1_2`. Both RKNN instance implementations call
`rknn_set_core_mask` immediately after `rknn_init`; startup fails if the
installed runtime rejects the requested mask. `core_policy` is `shared` or
`exclusive` metadata used by the platform scheduler. Tasks share a context only
when release, post-processing parameters, core mask, and core policy all match.

## MPP, ByteTrack, Kafka and ZLM SEI

The platform-generated graph can select `RkMppCaptureNode` for RTSP H.264/H.265
hardware decode. It sends Annex-B access units to Rockchip MPP while retaining
the original compressed packet beside the decoded BGR frame. `ByteTrackNode`
then adds stable `track_id` values to YOLO detections. Structured segmentation
and OCR adapters reject tracking.

`KafkaOutputNode` and `ZlmSeiOutputNode` share one `anhuan_v1` serializer, so
box coordinates, labels, confidence, tracking ID, frame index, and creation
time have one schema. Kafka defaults to topic `sei_msg`; when no key is given,
it uses the final two path segments of the source URI. The producer is
asynchronous and bounded. ZLM output inserts unregistered user-data SEI with
the reference UUID into the unchanged H.264/H.265 access unit and remuxes it to
RTSP without video re-encoding. On reconnect it waits for the next key frame.
Kafka delivery and ZLM reconnect failures are isolated from local JSON and
preview branches.

Run the protocol compatibility probe after every image build:

```bash
rknn_protocol_probe
```

The inference image is also a build environment. Its pinned source and SDK
paths are `/opt/rknode/src/nv_video_pipeline`, `/opt/rknn-sdk`, and
`/opt/rockchip-mpp`; use the same CMake options shown in the native build
section and build under `/tmp`.

## DeepLabV3+ and PPOCR Releases

The `rknn_structured` instance supports the platform adapters below:

| Platform adapter | Instance type | Output |
| --- | --- | --- |
| `deeplab_logits_v1` | `DEEPLAB_LOGITS` | semantic class mask |
| `ppocr_db_det_v1` | `PPOCR_DB` | quadrilateral text regions |
| `ppocr_ctc_rec_v1` | `PPOCR_CTC` | decoded text and confidence |

Use `config/rk3588/instances.platform.example.yaml` as the generated instance
template. Startup rejects a non-batch-one tensor, an unknown layout, a DeepLab
class-channel mismatch, a PPOCR DB output with more than one channel, or a CTC
dictionary/blank-index mismatch before the graph starts.

DeepLab JSON uses `class-rle-v1`. `width` and `height` are the network output
mask dimensions; `source_width` and `source_height` identify the input frame.
Consumers reconstruct the class-ID mask from `[class_id, run_length]` pairs and
resize it to the source dimensions with nearest-neighbor interpolation. Keeping
the mask at network output resolution avoids a costly CPU resize and very large
JSON records on the board.

PPOCR DB uses a thresholded probability map, contour scoring, an
area/perimeter-derived rotated-rectangle expansion, and quadrilaterals mapped
to source coordinates. `max_candidates` and `max_regions` bound per-frame work
and output size. PPOCR CTC supports probability tensors by default; the
platform `ppocr_ctc_logits_v1` generator sets `ctc_scores_logits: true` because
that contract explicitly contains raw logits.

Structured inference must use `interval: 1`; reusing a segmentation or OCR
result for later frames is rejected. Only advertise these adapter names from
the inference agent after this exact runtime build and its labels are installed
and the board-side probe succeeds.
