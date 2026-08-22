# RK3588 视频推理流水线

本项目是一个面向 Rockchip RK3588 的 C++17 视频分析运行时。程序使用
OpenCV 在 CPU 侧读取图片、视频或 RTSP 流，通过 RKNN C API 调用 NPU，
再把检测、语义分割或 OCR 结果写入 JSONL 文件，或者通过 HTTP POST
发送给上游服务。

RKNN 是当前唯一生产推理后端。仓库不再包含 CUDA、TensorRT、NVDEC、
NVENC 或 CNStream 运行路径。

## 1. 当前状态

默认配置启用两路目标检测：

| 实例 | 模型来源 | 类型 | 默认输出 |
| --- | --- | --- | --- |
| `bytetrack` | `bytetrack_s.rknn` | `ByteTrack` 平坦检测张量 | `/output/bytetrack.jsonl` |
| `hat` | `aqm.rknn` | `V5` 平坦检测张量 | `/output/hat.jsonl` |

代码还支持下列平台模型适配器，但它们不是默认启用项：

| 任务 | 实例工厂 | `type` | 结果类型 |
| --- | --- | --- | --- |
| YOLO DFL 分头检测 | `rknn_yolo` | `YOLO_DFL_SPLIT` | 检测框列表 |
| DeepLabV3+ 语义分割 | `rknn_structured` | `DEEPLAB_LOGITS` | `class-rle-v1` 类别掩码 |
| PPOCR DB 文本检测 | `rknn_structured` | `PPOCR_DB` | 四边形文本区域 |
| PPOCR CTC 文本识别 | `rknn_structured` | `PPOCR_CTC` | 文本和置信度 |

平台扩展配置模板位于
`config/rk3588/instances.platform.example.yaml`。模板中的 release ID 是
占位符，不能直接作为生产配置使用。

截至 2026-08-10，板端保存的 `nv-video-pipeline-rk3588:20260807` 和
`nv-video-pipeline-rk3588:20260807-cleanup` 是两次检测流水线验证镜像。
当前工作树后续增加了结构化推理、HTTP 输出和实例探针；要在板端使用
这些新能力，必须从当前源码重新构建新标签，不能把旧镜像视为已包含
这些功能。

## 2. 运行架构

```text
base.yaml
  |-- instances.yaml ------------> InstancesManager
  |                                  |-- rknn_yolo
  |                                  `-- rknn_structured
  |
  `-- pipelines/*.yaml ----------> Pipeline
                                     |
图片 / 视频 / RTSP -> VideoCaptureNode -> InferNode -> JsonOutputNode
                          CPU frame       RK3588 NPU     JSONL / HTTP
```

主要运行流程如下：

1. `rknn_pipeline` 读取 `base.yaml`，初始化日志、实例文件和流水线目录。
2. `InstancesManager` 创建所有 `enable != 0` 的 RKNN 实例并校验模型、
   标签、输入输出张量和后处理参数。
3. 每个流水线 YAML 的每个 `inputs` 元素创建一条独立流水线。
4. 下游节点先启动，源节点最后启动，避免首帧在消费者就绪前丢失。
5. `VideoCaptureNode` 产生包含 CPU `cv::Mat` 和帧序号的帧数据。
6. `InferNode` 将任务提交给有界 RKNN 实例队列，并异步等待结果。
7. `JsonOutputNode` 将检测或结构化结果写入文件或发送到 HTTP 服务。
8. 到达运行时长，或收到 `SIGINT`/`SIGTERM` 后，程序先停止流水线，
   再停止推理线程并释放 RKNN context。

每个模型实例默认只创建一个串行 RKNN context。同一实例可被多条流水线
引用，但它们共享该实例的队列和 NPU 执行上下文。

## 3. 目录说明

```text
CMakeLists.txt                 C++17 目标、依赖和 RKNN SDK 定位
src/common/                    帧数据包、工厂、回调和通用队列
src/compat/                    线程、定时器等小型兼容层
src/nodes/                     采集、推理和结果输出节点
src/objects/                   帧、检测框、分割和 OCR 结果对象
src/pipelines/                 流水线和实例生命周期管理
src/rknn_instance/             RKNN 推理、张量校验和后处理
src/utils/                     日志与线程安全队列
samples/RknnPipelineMain.cpp   YAML 驱动的生产入口
samples/RknnSmoke.cpp          模型加载和张量检查工具
samples/RknnInstanceProbe.cpp  生成实例的初始化与预热探针
config/rk3588/                 默认配置及平台实例示例
tools/rknn/                    ONNX 到 RKNN 的离线转换脚本
tests/rknn_stub/               仅用于主机测试的 RKNN C API stub 和夹具
3rdparty/rknpu2/               已验证的 RKNN 2.3.2 头文件、ARM64 库和许可证
docker/                        RK3588 多阶段镜像和 Compose 文件
bin/                           默认 ONNX 转换输入及标签
```

生成的 CMake 产物应位于构建目录，例如 `build-rknn/bin`。不要把编译
产物写回源码目录 `bin/`。

## 4. 环境要求

### 4.1 已验证板端基线

- RK3588，Linux ARM64；
- Debian 12；
- RKNN Toolkit2、runtime 和 server 2.3.2；
- RKNPU 驱动 0.9.8；
- Docker 24.x，`linux/arm64`；
- NPU 设备 `/dev/dri/card0`。

`3rdparty/rknpu2` 中的头文件、`librknnrt.so`、Toolkit2、板端 runtime、
server 和 NPU 驱动属于同一个兼容栈。升级时必须整体校验，不能混用
不同版本。SDK 文件来源和 SHA-256 记录见
`3rdparty/rknpu2/README.md`。

### 4.2 原生构建依赖

- CMake 3.15 或更高版本；
- 支持 C++17 的 GCC/G++；
- OpenCV：`core`、`imgproc`、`imgcodecs`、`videoio`；
- yaml-cpp、jsoncpp、spdlog、fmt；
- libcurl；
- pthread；
- RKNN C API 头文件和 ARM64 `librknnrt.so`。

仓库内的 RKNN 动态库是 AArch64 ELF。普通 x86_64 主机可运行 stub
编译测试，但不能直接链接后运行真实 RKNN NPU 程序。

## 5. 默认模型转换

默认转换脚本固定要求 RKNN Toolkit2 2.3.2，并把两个 ONNX 模型转换成
RK3588 FP16 模型：

```bash
python3 tools/rknn/convert_models.py \
  --source-dir bin \
  --output-dir models/rk3588 \
  --target rk3588 \
  --toolkit-version 2.3.2
```

建议在板端已有的
`rknn_toolkit2:2.3.2-debian12-cp311-aarch64` 容器内执行转换，或者在
版本完全一致的 Toolkit2 环境中执行。

转换结果：

```text
models/rk3588/bytetrack_s.rknn
models/rk3588/aqm.rknn
models/rk3588/manifest.json
```

默认模型契约：

| 模型 | ONNX 输入 | RKNN 输入 | 输出 |
| --- | --- | --- | --- |
| ByteTrack | `[1,3,608,1088]` | RGB/NHWC/UINT8 | `[1,13566,6]` |
| 安全帽检测 | `[1,3,640,640]` | RGB/NHWC/UINT8 | `[1,25200,7]` |

`manifest.json` 记录 Toolkit 版本、源文件与输出文件哈希、预处理参数、
张量契约和输出大小。部署时应同时校验模型文件与 manifest，避免配置和
模型版本错配。

`tools/rknn/convert_models.py` 只负责上述两个默认检测模型。平台生成的
YOLO DFL、DeepLabV3+ 和 PPOCR release 应由平台转换流程生成，再将
`.rknn`、标签和生成的实例配置挂载到运行容器；不要把它们添加到默认
转换表中冒充固定模型。

## 6. 原生构建

在 RK3588 板端或兼容 ARM64 环境中执行：

```bash
cmake -S . -B build-rknn \
  -DRKNN_SDK_ROOT="$PWD/3rdparty/rknpu2" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-rknn --parallel 4
```

`RKNN_SDK_ROOT` 必须显式指定，并且至少包含：

```text
include/rknn_api.h
Linux/aarch64/librknnrt.so
```

构建输出：

| 文件 | 用途 |
| --- | --- |
| `build-rknn/bin/libVideoPipeline_RKNN.so` | 图运行时和 RKNN 实例共享库 |
| `build-rknn/bin/rknn_pipeline` | 生产流水线入口 |
| `build-rknn/bin/rknn_smoke` | 加载模型并打印输入输出张量 |
| `build-rknn/bin/rknn_instance_probe` | 初始化指定实例并执行一次预热推理 |

### 6.1 模型张量检查

```bash
LD_LIBRARY_PATH="$PWD/3rdparty/rknpu2/Linux/aarch64" \
  build-rknn/bin/rknn_smoke \
  models/rk3588/bytetrack_s.rknn \
  models/rk3588/aqm.rknn
```

此工具会真实执行 `rknn_init` 和张量查询。在板端运行时需要映射或访问
`/dev/dri/card0`。

### 6.2 实例预检

```bash
build-rknn/bin/rknn_instance_probe \
  /data/runtime/revisions/12/instances.yaml \
  release-0 release-1
```

探针会加载实例文件，启动指定实例，使用一张 64x64 黑图执行一次推理，
然后完整停止。发布切换时，应先让旧推理进程退出，再执行新版本探针；
不要同时启动第二组真实 NPU context 作为“预热”。

### 6.3 启动主程序

```bash
build-rknn/bin/rknn_pipeline config/rk3588/base.yaml 10
```

第一个参数是基础配置，第二个参数是运行秒数：

- 大于 0：到期后正常退出；
- 为 0 或省略：持续运行，直到收到终止信号。

设置 `RKNODE_READY_FILE=/path/to/ready` 后，程序只会在全部实例和流水线
启动成功后写入 `ready` 文件，可用于发布系统的就绪判定。

## 7. 配置说明

### 7.1 配置层级

`config/rk3588/base.yaml`：

- `instances`：实例配置文件，相对路径按 base 文件所在目录解析；
- `pipelines`：流水线目录，相对路径按 base 文件所在目录解析；
- `log_config.log_level`：spdlog 日志等级数值；
- `pipe_perf_interval`：节点性能统计周期；
- `instance_perf_interval`：实例性能统计周期。

`config/rk3588/instances.yaml` 定义默认模型实例。常用字段：

| 字段 | 含义 |
| --- | --- |
| `instance_name` | 工厂名：`rknn_yolo` 或 `rknn_structured` |
| `model_path` | 容器或进程可访问的 RKNN 模型绝对路径 |
| `label_path` | 标签文件路径；检测、DeepLab 和 CTC 必需 |
| `type` | 模型输出契约和后处理类型 |
| `queue_capacity` | 实例有界任务队列容量 |
| `enable` | `0` 禁用，其他值或缺省表示启用 |
| `perf` | `1` 输出该实例性能统计 |

检测模型还支持：

- `confidence_threshold`；
- `nms_threshold`；
- `max_detections`；
- `class_scores_logits`，仅用于确认输出是原始 logits 的 DFL 模型。

PPOCR DB 还支持：

- `binary_threshold`；
- `box_threshold`；
- `unclip_ratio`；
- `min_size`；
- `max_candidates`；
- `max_regions`。

PPOCR CTC 使用 `blank_index` 和 `ctc_scores_logits`。标签数、blank 位置、
输出类别轴或 DeepLab 类别通道不一致时，实例会在启动阶段失败。

### 7.2 流水线 YAML

默认流水线结构：

```yaml
inputs:
  - /assets/test.jpg

capture:
  node: VideoCaptureNode
  loop: false
  perf: 1

detect:
  node: InferNode
  instance: bytetrack
  interval: 1
  perf: 1
  link_to:
    - capture

result:
  node: JsonOutputNode
  instance: bytetrack
  output: /output/bytetrack.jsonl
  link_to:
    - detect
```

规则：

- `inputs` 必填，每个元素生成一条流水线；
- 每个非源节点必须有且只有一个有效上游 `link_to`；
- 当前运行时不支持多输入同步节点；
- `enable: 0` 会在连图前删除节点，引用关系也必须同步修改；
- `interval > 1` 只允许复用检测结果；分割和 OCR 必须使用
  `interval: 1`；
- 多输入需要不同输出时，可配置与 `inputs` 等长的 `outputs` 数组，
  运行时会按输入索引注入 `output`。

### 7.3 输入类型

`VideoCaptureNode` 支持：

- `.jpg`、`.jpeg`、`.png`、`.bmp` 单图；
- OpenCV/FFmpeg 可读取的本地视频文件；
- `rtsp://` 流。

单图在 `loop: false` 时只产生一帧；`loop: true` 时每秒产生一帧。
视频结束且 `loop: false` 时退出采集。RTSP 或循环视频读取失败后，节点
按 `reconnect_ms` 等待并重新打开；打开和读取超时均为 5 秒。

### 7.4 输出到文件或 HTTP

兼容的简写文件配置：

```yaml
output: /output/result.jsonl
```

显式 JSONL 配置：

```yaml
output:
  type: jsonl
  path: /output/result.jsonl
```

HTTP 配置：

```yaml
output:
  type: http
  url: http://127.0.0.1:8080/results
  connect_timeout_ms: 1000
  request_timeout_ms: 3000
  authorization_env: RESULT_API_TOKEN
```

HTTP sink 使用 `Content-Type: application/json`。若配置
`authorization_env`，程序从对应环境变量读取 token，并发送
`Authorization: Bearer ...`；不要把 token 直接写入 YAML 或仓库。
HTTP 返回非 2xx 或超时会记录警告，不会把凭据打印到日志。

## 8. Docker 镜像构建

### 8.1 镜像角色

| 镜像/阶段 | 作用 | 是否项目运行镜像 |
| --- | --- | --- |
| `rknn_toolkit2:2.3.2-debian12-cp311-aarch64` | ARM64 编译及 Toolkit2 基础环境 | 否 |
| Dockerfile `builder` 阶段 | 安装开发包并编译四个目标 | 否，不应长期打标签保留 |
| `debian:bookworm` | 精简运行阶段基础 | 否 |
| `nv-video-pipeline:rk3588` | 最终 RK3588 运行镜像 | 是 |
| `nv-video-pipeline-rknn-stub-dev:latest` | x86 主机 stub 测试环境 | 否，禁止用于生产 |

镜像是只读构建产物；容器是镜像的运行实例。使用 `docker run --rm`
执行 smoke 或有限图测试后，测试容器会自动删除，但镜像仍然存在。

### 8.2 多阶段构建内容

`docker/Dockerfile.rk3588` 执行以下步骤：

1. 以 ARM64 Toolkit2 2.3.2 镜像作为 builder；
2. 安装编译器、CMake、OpenCV、yaml-cpp、spdlog、jsoncpp 和 curl；
3. 将精简的 RKNN SDK 复制到 `/opt/rknn-sdk`；
4. 以 Release 模式编译共享库及三个可执行程序；
5. 以 Debian 12 创建运行层，只安装运行时依赖；
6. 复制程序、`librknnrt.so`、许可证、默认配置和默认标签；
7. 创建 `/models`、`/assets`、`/output` 约定目录；
8. 设置入口为 `/usr/local/bin/rknn_pipeline`。

`.dockerignore` 采用默认拒绝策略，只把构建必需的源码、配置、SDK 和
标签发送给 Docker daemon。ONNX 原文件、测试图片、输出和本地构建目录
不会被复制进生产镜像。

### 8.3 在 RK3588 板端构建

推荐在板端 `/userdata` 等空间充足的数据分区创建独立构建目录。根分区
空间较小时，不要把源码归档、模型和输出放在 `/root` 或根文件系统。

```bash
docker build \
  -f docker/Dockerfile.rk3588 \
  --build-arg RKNN_TOOLKIT_IMAGE=rknn_toolkit2:2.3.2-debian12-cp311-aarch64 \
  --build-arg RKNN_RUNTIME_IMAGE=debian:bookworm \
  -t nv-video-pipeline:rk3588 \
  .
```

网络受限时可只替换 Debian 下载镜像，不改变依赖版本：

```bash
docker build \
  -f docker/Dockerfile.rk3588 \
  --build-arg DEBIAN_MIRROR=http://mirrors.aliyun.com/debian \
  --build-arg DEBIAN_SECURITY_MIRROR=http://mirrors.aliyun.com/debian-security \
  -t nv-video-pipeline:rk3588 \
  .
```

不要用相同标签静默覆盖正在发布的镜像。建议先构建唯一版本标签，例如
`nv-video-pipeline-rk3588:20260810-structured`，通过 smoke、实例探针和
有限图测试后再更新部署引用。

### 8.4 容器内模型 smoke

```bash
docker run --rm \
  --name nv-video-pipeline-smoke \
  --device /dev/dri/card0:/dev/dri/card0 \
  -v "$PWD/models/rk3588:/models:ro" \
  --entrypoint /usr/local/bin/rknn_smoke \
  nv-video-pipeline:rk3588 \
  /models/bytetrack_s.rknn /models/aqm.rknn
```

在已验证板端，仅映射 `renderD128` 不能完成 `rknn_init`；NPU 必须能
访问 `/dev/dri/card0`。不需要 `--privileged`。

### 8.5 容器内运行完整流水线

```bash
mkdir -p output/rk3588

docker run --rm \
  --name nv-video-pipeline-rk3588 \
  --network host \
  --device /dev/dri/card0:/dev/dri/card0 \
  -v "$PWD/models/rk3588:/models:ro" \
  -v "$PWD/test.jpg:/assets/test.jpg:ro" \
  -v "$PWD/output/rk3588:/output" \
  nv-video-pipeline:rk3588 \
  /opt/video-pipeline/config/base.yaml 10
```

图片和模型使用只读挂载，输出目录可写。使用 RTSP 或硬件媒体路径时，
按实际需要增加 `/dev/dri/renderD128`、`/dev/rga` 和 host network；不要
直接扩大到特权容器。

### 8.6 使用 Docker Compose

`docker/compose.rk3588.yml` 定义一个服务：

- 服务名：`rk3588_video_pipeline`；
- 容器名：`nv-video-pipeline-rk3588`；
- 镜像：`nv-video-pipeline:rk3588`；
- 网络：host；
- 默认挂载：模型、`test.jpg` 和输出目录；
- 默认设备：`card0`、`renderD128`、`rga`；
- 重启策略：`no`。

```bash
mkdir -p models/rk3588 output/rk3588

docker compose -f docker/compose.rk3588.yml build
docker compose -f docker/compose.rk3588.yml up --abort-on-container-exit
```

Dockerfile 默认命令是
`/opt/video-pipeline/config/base.yaml 10`，因此默认有限图运行约 10 秒后
正常退出。长期运行时应在 Compose 中覆盖 command，将时长设为 `0`，
并按运维策略设置重启行为。

停止和删除 Compose 容器不会删除镜像：

```bash
docker compose -f docker/compose.rk3588.yml down
```

清理开发中间镜像前先检查容器引用，并按明确标签删除。不要在共享板端
直接执行宽泛的 `docker system prune -a`。

## 9. 输出数据格式

JSONL 文件采用追加写入。重复测试前若需要精确统计，应使用新的输出
目录或明确归档旧结果，避免把历史行误认为本次输出。

### 9.1 检测结果

```json
{"frame_index":0,"width":1920,"height":1080,"instance":"bytetrack","detections":[{"x":431,"y":357,"w":1081,"h":722,"label":"人","confidence":0.635473,"class_id":0,"track_id":-1}]}
```

坐标已经从网络 letterbox 空间映射回原始帧。当前 `track_id` 默认为
`-1`，本运行时没有额外的跨帧跟踪器。

### 9.2 DeepLab 分割结果

```json
{"frame_index":0,"width":1920,"height":1080,"instance":"segment","result_type":"segmentation","result":{"width":128,"height":128,"source_width":1920,"source_height":1080,"encoding":"class-rle-v1","labels":["background","ng","scratch"],"runs":[[0,120],[1,16],[0,80]]}}
```

`runs` 按行优先顺序保存 `[class_id, run_length]`。消费者依次展开即可
恢复 `result.width × result.height` 的类别 ID mask，再用最近邻插值恢复到
`source_width × source_height`。运行时保留网络输出分辨率，避免在板端
执行大尺寸 CPU resize 并写出巨大的 JSON。

### 9.3 PPOCR DB 文本检测

```json
{"frame_index":0,"instance":"ocr_det","result_type":"ocr_detection","result":{"regions":[{"confidence":0.91,"points":[[10,20],[120,20],[120,48],[10,48]]}]}}
```

### 9.4 PPOCR CTC 文本识别

```json
{"frame_index":0,"instance":"ocr_rec","result_type":"ocr_recognition","result":{"text":"ABC","confidence":0.94}}
```

HTTP sink 发送的请求体与对应 JSONL 单行内容相同。

## 10. 主机 stub 验证

stub 只用于验证编译、张量契约、图连线、JSON/HTTP 输出和线程退出，
不能证明真实 NPU 可用。

先构建 x86 测试环境：

```bash
docker build \
  -f tests/Dockerfile.rknn-stub-build \
  -t nv-video-pipeline-rknn-stub-dev:latest \
  .
```

再在临时目录生成测试用 `librknnrt.so` 并构建：

```bash
docker run --rm \
  -v "$PWD:/src" \
  -w /src \
  nv-video-pipeline-rknn-stub-dev:latest \
  bash -lc '
    set -e
    sdk=/tmp/rknn-stub-sdk
    mkdir -p "$sdk/include" "$sdk/Linux/aarch64"
    cp tests/rknn_stub/include/rknn_api.h "$sdk/include/"
    g++ -std=c++17 -fPIC -shared \
      tests/rknn_stub/rknn_stub.cpp \
      -I"$sdk/include" \
      -o "$sdk/Linux/aarch64/librknnrt.so"
    cmake -S . -B /tmp/nv-video-build \
      -DRKNN_SDK_ROOT="$sdk" \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build /tmp/nv-video-build --parallel 4
    /tmp/nv-video-build/bin/rknn_smoke \
      tests/rknn_stub/fixtures/split.rknn \
      tests/rknn_stub/fixtures/deeplab.rknn \
      tests/rknn_stub/fixtures/ppocr-det.rknn \
      tests/rknn_stub/fixtures/ppocr-rec.rknn
  '
```

真实发布仍必须在 RK3588 上完成模型转换、`rknn_smoke`、实例探针和完整
流水线验证。

## 11. 板端发布检查清单

1. 确认目标是 `linux/arm64`，Toolkit/runtime/server/driver 版本匹配。
2. 校验 SDK、模型、标签和 manifest 的 SHA-256。
3. 使用唯一镜像标签构建，不覆盖当前可回滚镜像。
4. 用 `rknn_smoke` 检查每个模型的真实输入输出张量。
5. 旧推理进程退出后，用 `rknn_instance_probe` 验证生成实例。
6. 仅映射必要设备，至少包含 `/dev/dri/card0`。
7. 用有限图片跑完整图，检查每个实例都有预期 JSON 结果。
8. 检查输出数量、坐标或 mask 尺寸、类别 ID 和标签映射。
9. 检查实例线程正常停止，测试容器使用 `--rm` 后无残留。
10. 对可执行程序和共享库执行 `ldd`，确认无缺失依赖，也无 CUDA、
    TensorRT、NVIDIA codec 或 CNStream 依赖。
11. 验证通过后再更新正式部署引用；探针或启动失败时保留旧镜像回滚。

## 12. 常见问题

### CMake 提示必须指定 `RKNN_SDK_ROOT`

必须显式传入 SDK 根目录：

```bash
-DRKNN_SDK_ROOT="$PWD/3rdparty/rknpu2"
```

不要让 CMake 在主机全局目录中隐式找到其他版本的 `librknnrt.so`。

### 在 x86 主机链接或运行失败

仓库内真实 RKNN 库是 ARM64。x86 主机只能使用 `tests/rknn_stub`；真实
运行必须在 RK3588/ARM64 上完成。

### `rknn_init` 无法打开设备

检查容器是否映射 `/dev/dri/card0`、宿主机设备权限、RKNN server/runtime
和驱动版本。只映射 `/dev/dri/renderD128` 不足以访问已验证板端 NPU。

### 实例初始化时标签或张量不匹配

检查 `label_path`、标签行数、模型 release 和实例 `type`。DeepLab 输出
类别通道必须等于标签数；CTC 输出必须包含标签类别加 blank；普通检测
输出特征数必须等于 `5 + 类别数`。

### 结构化实例拒绝 `interval > 1`

分割和 OCR 结果不能安全复用于后续帧。把对应 `InferNode.interval` 设为
`1`。只有检测型 `FrameTargetList` 支持间隔复用。

### 输出文件行数持续增加

JSONL 采用 append 模式。为每次验收创建独立输出目录，或者在运行前
明确归档旧结果。

### RTSP 打不开或频繁重连

确认容器使用 host network、板端能访问 RTSP 地址、OpenCV 的 FFmpeg
后端可用，并检查 5 秒打开/读取超时和 `reconnect_ms` 配置。有限图片
测试通过不代表外部 RTSP 网络可达。

### HTTP 输出没有结果

确认 URL 使用 `http://` 或 `https://`，超时范围合法，授权环境变量已
设置，接收端返回 2xx。程序会记录 curl 错误和 HTTP 状态，但不会在
日志中打印 bearer token。

### Docker 构建占用空间过大

构建目录放在 `/userdata` 等数据分区，保留生产和回滚运行镜像，删除
明确的 builder 标签和无容器引用的项目中间层。共享板端不要使用无范围
限制的全局清理命令。

## 13. 安全与运维约束

- SSH 密码、HTTP token、模型服务凭据不得写入仓库、YAML 或日志；
- 模型和输入资产默认只读挂载，只有输出目录可写；
- 不使用 `--privileged`，按需映射 NPU 和媒体设备；
- 生产、候选和回滚镜像使用不同标签；
- 不删除共享 Toolkit2 基础镜像或其他服务正在使用的镜像；
- 不把 test stub、ONNX 原文件或构建缓存复制到生产运行层；
- 主机 stub 结果和 RK3588 真机 NPU 结果必须分开记录。

精简的英文板端部署参考见 `docs/rk3588.md`。
