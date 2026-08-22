# Bundled source snapshots

This directory contains the source snapshots required to build every platform
node from this repository. Docker builds do not clone framework or inference
repositories. Keep the upstream license files in each snapshot when updating
them.

| Path | Upstream/source | Revision | Used by |
| --- | --- | --- | --- |
| `training/yolov5` | `https://github.com/airockchip/yolov5.git` | `d25a07534c14f44296f9444bab2aa5c601cdaaab` | Torch trainer |
| `training/yolov6` | `https://github.com/airockchip/yolov6.git` | `0e7c2d5a93f6d49ed5ab6f005ccdd9d9bbd3db9b` | Torch trainer |
| `training/yolov7` | `https://github.com/airockchip/yolov7.git` | `c2d39f4db6b82800ce6c61740be5ef82854c5d3e` | Torch trainer |
| `training/yolov8` | `https://github.com/airockchip/ultralytics_yolov8.git` | `4674fe6e003dfbc5f2250d3b39dd31faaf7a9877` | Torch trainer |
| `training/yolov10` | `https://github.com/airockchip/yolov10.git` | `81f32c4ee396e679b489ff786faa0f9fa0eec298` | Torch trainer |
| `training/yolo11` | `https://github.com/airockchip/ultralytics_yolo11.git` | `0692e9297670acf4cc6d0cec773d7a9493cb8a5f` | Torch trainer |
| `training/segmentation_models_pytorch` | `https://github.com/qubvel-org/segmentation_models.pytorch.git` | `v0.5.0` / `420ce84b0c2df0286fa9bb2bd1499eea625c9b33` | DeepLabV3+ trainer |
| `training/paddleocr` | `https://github.com/PaddlePaddle/PaddleOCR.git` | `8cce9b6fd7ccb50226d0c38f94054d81c29b8184` | Paddle trainer |
| `training/weights/yolov8n.pt` | `https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt` | SHA256 below | Torch trainer pretrained YOLOv8n |
| `nv_video_pipeline` | internal `nv_video_pipeline` source snapshot | platform-integrated snapshot, 2026-08-12 | RK3588 inference image |
| `nv_video_pipeline/src/nodes/track/bytetrack` | internal `video_pipeline_anhuan` portable ByteTrack source | snapshot, 2026-08-13 | RK3588 target tracking |
| `nv_video_pipeline/3rdparty/rockchip-mpp` | Rockchip MPP Debian package on RK3588 node | 1.5.0-1, snapshot 2026-08-13 | RK3588 hardware decode |

`training/SOURCES.lock` is copied into trainer images and checked against the
Docker build arguments. The semantic-segmentation snapshot is installed from
the local project path, so the DeepLabV3+ implementation is not downloaded
from PyPI during the image build. The RK3588 inference snapshot includes the aarch64
RKNN Runtime header and library used by its C++ build. Their SHA256 values are:

```text
c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a  rknn_api.h
d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8  librknnrt.so
f03882324fddb343ff714e77b1fe37587ab70a42d3e40057327e5fa9afa46261  librockchip_mpp.so.1
31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95  yolov8n.pt
```

The original local sources were not removed or modified when these snapshots
were created. Package installation still requires access to the configured
APT/Python/Torch registries unless the corresponding base images and packages
are supplied by an offline mirror.
