# RKNode 完整使用与部署手册

适用版本：平台与训练镜像 `2026.08.20`，RK3588 镜像 `2026.08.20-business`。

适用对象：第一次部署本项目的运维人员、平台管理员和业务操作人员。本文以当前仓库中的 Compose、环境模板和已验证运行拓扑为准。

## 旧版静态 Token 迁移

旧节点可在维护窗口暂时保留静态 Token；新节点必须使用统一 enrollment。迁移时保留原数据卷和 endpoint，领取一次性注册码后再切换到长期 Token 文件，具体步骤见[节点部署手册](simple-node-deployment.md)。

安全边界：节点控制端口不得暴露到公网；跨网段优先使用 VPN，必要时使用 HTTPS 或 SSH 隧道。平台和文档不得保存 SSH 密码。

## 1. 系统能力与边界

RKNode 提供以下完整流程：

```text
数据集上传
  -> CPU/CUDA 模型训练
  -> 固定分辨率 ONNX 与部署清单
  -> RK3588 转换为 RKNN 并做板端初始化验证
  -> 登记并发布模型版本
  -> 创建 RK3588 推理任务
  -> 下发/滚动部署
  -> 原始 RTSP + SEI 输出
  -> 浏览器解析 SEI 并绘制检测框、分割掩码和 OCR 结果
```

| Profile | 用途 | 训练框架 | RKNN 精度 |
| --- | --- | --- | --- |
| `yolo-detect` | 目标检测 | Torch / Rockchip YOLO | INT8、FP16 |
| `deeplabv3plus` | 语义分割 | Torch / segmentation-models-pytorch | INT8、FP16 |
| `ppocr-det` | OCR 文本检测 | PaddleOCR | INT8、FP16 |
| `ppocr-rec` | OCR 文本识别 | PaddleOCR | FP16 |

平台不会在中央服务器上执行 RKNN 转换或 NPU 推理。转换和推理必须运行在 RK3588 节点。一个训练任务只选择一个 CPU 或 CUDA 节点，不支持单任务跨节点分布式训练。

## 2. 系统架构

```text
                       +-----------------------------+
浏览器 ----------------> Web :5173                   |
                       |   -> API :8000（内部）      |
                       |   -> Media :8081（WS-FLV）  |
                       | Media :8554（RTSP 发布）    |
                       +--------------+--------------+
                                      |
             平台主动访问节点 / 节点主动访问平台
                    +-----------------+------------------+
                    |                                    |
             Trainer :10081                      RK3588 板端
             CPU 或 CUDA                  Converter :10081
                                          Inference :10082
```

中央平台包含 API、Web、Media 三个容器。Web 同时反向代理 `/api/v1` 到 Compose 内部 API；Media 接收板端 RTSP + SEI，并向浏览器提供 WS-FLV。新节点统一使用 direct 模式，旧版 pull/static-token 只作为兼容路径保留。

## 3. 版本、镜像与交付物

| 角色 | 镜像 | 架构 | 说明 |
| --- | --- | --- | --- |
| API | `rknode-platform-api:2026.08.20` | amd64 | 中央 API |
| Web | `rknode-platform-web:2026.08.20` | amd64 | Nginx + 前端 |
| Media | `rknode-platform-media:2026.08.20` | amd64 | 固定摘要 ZLMediaKit |
| Torch CPU | `rknode-trainer-torch-cpu:2026.08.20` | amd64 | YOLO、DeepLab |
| Paddle CPU | `rknode-trainer-paddle-cpu:2026.08.20` | amd64 | PPOCR Det/Rec |
| Torch CUDA | `rknode-trainer-torch-cuda12.4:2026.08.20` | amd64 | NVIDIA CUDA 12.4 |
| Paddle CUDA | `rknode-trainer-paddle-cuda12.6:2026.08.20` | amd64 | NVIDIA CUDA 12.6 |
| RK3588 | `rknode-rk3588-node:2026.08.20-business` | arm64 | converter + inference |

标准离线交付只有两个 tar：`platform-amd64` 包含三个平台镜像；`rk3588-node-arm64` 包含一个统一 RK3588 镜像。四个训练镜像已统一版本，但不进入标准离线包。

部署前核对镜像身份：

```bash
docker image inspect rknode-platform-api:2026.08.20 \
  --format '{{.Id}} {{.Architecture}} {{index .Config.Labels "org.opencontainers.image.version"}} {{index .Config.Labels "org.opencontainers.image.revision"}}'

docker image inspect rknode-rk3588-node:2026.08.20-business \
  --format '{{.Id}} {{.Architecture}} {{.Size}} {{index .Config.Labels "org.opencontainers.image.version"}} {{index .Config.Labels "org.opencontainers.image.revision"}}'
```

禁止部署 `latest`、`local` 或 `<none>` 镜像。`source revision` 是构建源码快照标识，用于审计镜像来源；它不是容器 ID，也不是运行配置。

## 4. 地址方向与端口规划

### 4.1 两个方向不能混淆

```text
平台系统设置中的 endpoint = 中央平台 -> 节点宿主机 IP/DNS:发布端口
节点的 RKNODE_PLATFORM_URL = 节点 -> 中央 Web/API 根地址
```

`RKNODE_PLATFORM_URL` 不附加 `/api/v1`，当前为：

```dotenv
RKNODE_PLATFORM_URL=http://172.16.66.249:5173
```

### 4.2 当前已验证地址

| 服务 | 当前地址 | 说明 |
| --- | --- | --- |
| Web/API 入口 | `172.16.66.249:5173` | 浏览器和节点访问 |
| RTSP 发布 | `172.16.66.249:8554` | 板端发布 H.264/H.265 + SEI |
| WS-FLV 播放 | `172.16.66.249:8081` | 浏览器直连 Media |
| CPU Torch 训练 | `172.16.66.249:10081` | 当前与中央平台同机 |
| RK3588 转换 | `172.29.0.1:11081` | SSH 隧道到板端 `127.0.0.1:10081` |
| RK3588 推理 | `172.29.0.1:11082` | SSH 隧道到板端 `127.0.0.1:10082` |
| RK3588 SSH | `172.30.82.12:124` | 管理与隧道，不是平台 endpoint |

目标直连地址是 `172.30.82.12:10081/10082`，但只有中央服务器能访问这两个端口并得到 HTTP 401/200 后才可切换。连接超时或拒绝都表示直连条件未满足。

### 4.3 防火墙最小开放范围

| 端口 | 来源 | 目标 | 用途 |
| --- | --- | --- | --- |
| 5173/TCP | 浏览器、节点 | 中央平台 | Web/API |
| 8554/TCP | RK3588 节点 | 中央平台 | RTSP 发布 |
| 8081/TCP | 浏览器 | 中央平台 | WS-FLV 播放 |
| 10081/TCP | 中央平台 | 训练/转换宿主机 | 节点控制接口 |
| 10082/TCP | 中央平台 | 推理宿主机 | 节点控制接口 |
| 124/TCP | 受控运维主机 | RK3588 | 当前 SSH 管理入口 |

节点控制端口不得暴露到公网。跨网段优先使用 VPN；没有域名和证书时，只能在可信 LAN/VPN 内使用 HTTP/WS。

## 5. 部署前准备

### 5.1 中央平台主机

- Linux amd64、Docker Engine 24+、Docker Compose v2。
- 固定 IP/DNS，节点和浏览器均能访问 5173。
- 磁盘能容纳三个平台镜像、`platform-data` 和媒体数据。
- 已安装 `curl`；离线验证还需要 `nc`、`sha256sum`、`gzip`。

```bash
uname -m
docker version
docker compose version
df -h
ss -lntp | grep -E ':(5173|8554|8081|10081|10082|11081|11082)\b' || true
```

### 5.2 训练主机

CPU 节点需要 Linux amd64 和 Docker Compose v2。CUDA 节点还需兼容的 NVIDIA 驱动与 NVIDIA Container Toolkit。节点必须能访问中央 5173，中央必须能访问训练主机 10081。

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 5.3 RK3588 主机

RK3588 需要 Linux arm64、Docker Compose v2、最终 arm64 镜像以及以下设备：

```bash
uname -m
ls -l /dev/dri/card0 /dev/dri/renderD128 /dev/mpp_service /dev/rga
ls -ld /dev/dma_heap /sys/firmware/devicetree/base
docker image inspect rknode-rk3588-node:2026.08.20-business
```

## 6. 部署中央平台

### 6.1 准备配置

```bash
cp deploy/.env.example deploy/.env
chmod 600 deploy/.env
```

至少修改：

```dotenv
RKNODE_API_IMAGE=rknode-platform-api:2026.08.20
RKNODE_WEB_IMAGE=rknode-platform-web:2026.08.20
RKNODE_MEDIA_IMAGE=rknode-platform-media:2026.08.20
RKNODE_ADMIN_TOKEN=<管理员强随机令牌>
RKNODE_WORKER_TOKEN=<不同的兼容Worker令牌>
RKNODE_WEB_PORT=5173
RKNODE_CORS_ORIGINS=http://<中央服务器IP>:5173
RKNODE_PUBLIC_API_URL=http://<中央服务器IP>:5173/api/v1
RKNODE_MEDIA_PUBLISH_HOST=<中央服务器IP>
RKNODE_MEDIA_PLAYBACK_HOST=<中央服务器IP>
RKNODE_MEDIA_RTSP_PORT=8554
RKNODE_MEDIA_WS_PORT=8081
```

可用 `openssl rand -hex 32` 分别生成管理员和 Worker 兼容令牌，两者必须不同。新 direct 节点不会获得全局 Worker 令牌。

从 `deploy/media/zlm-base-image.lock` 复制固定摘要到 `RKNODE_ZLM_BASE_IMAGE`，再生成媒体密钥：

```bash
sed -n '1p' deploy/media/zlm-base-image.lock
python3 scripts/configure_media_secrets.py --env-file deploy/.env
chmod 600 deploy/.env
```

脚本只补充缺失的 `RKNODE_ZLM_API_SECRET` 和 `RKNODE_ZLM_HOOK_IDENTITY`，不会打印密钥原文。

### 6.2 启动与验证

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml config --quiet
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --no-build
docker compose --env-file deploy/.env -f deploy/compose.yaml ps

curl -fsS http://127.0.0.1:5173/api/v1/ready
curl -I http://127.0.0.1:5173
nc -z 127.0.0.1 8554
nc -z 127.0.0.1 8081
docker compose --env-file deploy/.env -f deploy/compose.yaml logs --tail=100 api frontend media
```

期望 `/api/v1/ready` 返回 `{"status":"ready"}`，API 和 Media 为 `healthy`，Web 为 `Up`。浏览器访问 `http://<中央服务器IP>:5173`，首次打开输入 `RKNODE_ADMIN_TOKEN`；令牌只保存在当前浏览器会话的 `sessionStorage` 中。

## 7. 注册并部署节点

所有 trainer、converter、inference 使用相同流程。具体环境变量和命令见[节点部署手册](simple-node-deployment.md)。

### 7.1 平台登记

进入“系统设置” -> “新增节点”：

| 字段 | 训练 | 转换 | 推理 |
| --- | --- | --- | --- |
| 节点类型 | 模型训练 | 模型转换 | 模型推理 |
| 接入模式 | 直连调度 | 直连调度 | 直连调度（固定） |
| 加速器 | CPU 或 CUDA | RK3588 | RK3588 |
| 当前地址 | 训练宿主机 IP:10081 | `172.29.0.1:11081` | `172.29.0.1:11082` |
| 能力 | Torch/Paddle profile | 模型 profile ID | 推理 adapter ID |


保存后平台只在当前窗口显示 Endpoint ID 和一次性注册码。Endpoint ID 写入节点 `.env`；注册码立即下载或复制到节点权限为 `0600` 的 secret 文件。

统一接入索引：平台登记字段是“节点宿主机 IP / 域名”和服务端口；节点环境使用 `RKNODE_ENDPOINT_ID`、`RKNODE_PLATFORM_URL`、`RKNODE_ENROLLMENT_TOKEN_FILE` 和 `RKNODE_NODE_TOKEN_FILE`。状态顺序为 `pending -> claimed -> enrolled`。当前训练地址为 `172.16.66.249:10081`，RK3588 隧道地址为 `172.29.0.1:11081`、`172.29.0.1:11082`；板端直连验收地址为 `172.30.82.12:10081`、`172.30.82.12:10082`。

### 7.2 首次认领

```bash
install -d -m 700 ./secrets
printf '%s\n' '<一次性注册码>' > ./secrets/node-enrollment-token
chmod 600 ./secrets/node-enrollment-token

docker compose -p <项目名> --env-file .env \
  -f compose.yaml -f compose.enrollment.yaml config --quiet
docker compose -p <项目名> --env-file .env \
  -f compose.yaml -f compose.enrollment.yaml up -d --no-build
```

| 状态 | 含义 | 能否调度 |
| --- | --- | --- |
| `pending` | 平台已登记，节点未领取 | 否 |
| `claimed` | 长期 Token 已发放，首次探测未通过 | 否 |
| `enrolled + online` | 身份和健康探测通过 | 是 |
| `enrolled + offline/error` | 已注册但当前不可用 | 否 |

### 7.3 切换稳态

达到 `enrolled + online` 后，去掉 enrollment overlay 重建：

```bash
docker compose -p <项目名> --env-file .env \
  -f compose.yaml up -d --no-build --force-recreate

docker compose -p <项目名> --env-file .env -f compose.yaml exec -T <服务名> \
  stat -c '%a:%s' /data/state/node-token
```

期望权限为 `600` 且文件非空。确认平台仍在线后删除一次性注册码，再重启一次。不要删除节点数据卷，否则长期 Token、模型缓存和运行状态会一起丢失。

## 8. 平台业务使用

### 8.1 数据集

进入“数据集”上传压缩包，填写名称、版本、任务类型、格式和类别：

| 任务 | 格式 |
| --- | --- |
| 目标检测 | YOLO、COCO Detection、Pascal VOC Detection |
| 语义分割 | mask pairs、COCO Segmentation、Pascal VOC Segmentation |
| OCR 检测 | PPOCR Detection |
| OCR 识别 | PPOCR Recognition |

状态达到 `ready` 后才能训练。`failed` 时先修正数据集错误，不要重复提交同一坏包。

### 8.2 训练

进入“模型训练”新建任务，选择 ready 数据集、profile、variant、CPU/CUDA、固定分辨率、epoch、batch、优化器、学习率和随机种子。平台只调度到能力、加速器和健康状态都匹配的节点。

| Profile | 宽高范围 | 对齐 |
| --- | --- | --- |
| YOLO | 160..2048 | 32 的倍数 |
| DeepLabV3+ | 128..2048 | 32 的倍数 |
| PPOCR Det | 128..2048 | 32 的倍数 |
| PPOCR Rec | 宽 32..2048，高 32..128 | 8 的倍数 |

任务成功后检查 ONNX、部署清单、日志和校验产物。失败任务“重试”会创建新任务，原失败记录保留。

### 8.3 RKNN 转换

进入“模型转换”，选择 ONNX 产物和精度。INT8 必须选择校准数据集，PPOCR Rec 只支持 FP16。converter 会执行图检查、构建、导出和板端 runtime 初始化。`converted` 不等于 `validated`；只有验证报告标记部署就绪的 RKNN 才能发布。

### 8.4 模型发布、推理和部署

进入“推理下发”：

1. 从成功转换任务登记模型版本。
2. 对 `qualified` 版本执行“发布”。
3. 创建推理任务，选择 published 模型、健康板卡、输入 URI、NPU 核心和结果出口。
4. 配置媒体、跟踪、区域/越线、事件输出和二级推理。
5. 创建 `canary`、`rolling` 或 `all_at_once` 部署批次。
6. 查看目标阶段与事件，成功后任务进入 `running`。

推理支持 NPU 核心自动/指定、共享/独占策略、JSONL 或 HTTP 结果出口、RKMPP、ByteTrack、区域越线、事件快照/录像和二级 YOLO。主模型和二级模型可分别设置 `contextCount` 与 `workerCount`，且 worker 不得超过 context。

`RKNODE_MAX_MODEL_INSTANCES` 限制的是唯一运行池实际占用的 RKNN context 总数，不是任务数或 worker 线程数。

### 8.5 视频监控与 SEI 叠加

```text
RK3588 保留原始 H.264/H.265
  -> 写入固定 UUID 的 SEI 元数据
  -> RTSP 发布到中央 Media :8554
  -> 浏览器用 60 秒播放令牌连接 Media :8081
  -> 前端按时间戳绘制目标框、分割掩码、OCR 和业务规则
```

进入“视频监控”选择运行中的推理任务。播放令牌是短期、单任务凭据，不是节点 Token。DeepLab 使用官方 logits 后处理：通道 argmax 生成类别图，再按 palette/任务类别绘制半透明掩码。

## 9. 日常运维

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
docker compose --env-file deploy/.env -f deploy/compose.yaml logs -f --tail=200 api
docker compose --env-file deploy/.env -f deploy/compose.yaml logs -f --tail=200 frontend
docker compose --env-file deploy/.env -f deploy/compose.yaml logs -f --tail=200 media
docker compose --env-file deploy/.env -f deploy/compose.yaml restart api frontend media
docker compose --env-file deploy/.env -f deploy/compose.yaml down
```

`down` 不删除命名卷；禁止使用 `down -v`，除非明确批准永久删除平台业务数据。默认日志轮转为单文件 20 MB、保留 3 个文件。

## 10. 备份与恢复

平台状态、数据库、数据集和模型产物位于 `platform-data` 卷。备份前停止写入：

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml stop api frontend
docker run --rm \
  -v deploy_platform-data:/source:ro \
  -v "$PWD/backup":/backup \
  alpine:3.20 sh -c 'cd /source && tar -czf /backup/platform-data.tgz .'
docker compose --env-file deploy/.env -f deploy/compose.yaml start api frontend
```

Compose 项目名改变时卷名也会改变，先用 `docker volume ls` 确认。单独加密备份权限为 `0600` 的 `deploy/.env`。节点卷包含长期 Token、任务工作区、模型缓存和运行状态，升级时必须保留。

## 11. 升级与回滚

1. 确认没有活动训练、转换或部署批次。
2. 备份 `platform-data`、节点卷和 `.env`。
3. 加载明确版本的新镜像，核对架构、version、revision 和 digest。
4. 只修改 `.env` 镜像标签，不删除旧标签。
5. 运行 `docker compose config --quiet`。
6. `up -d --no-build --force-recreate` 并完成健康、转换和推理验收。
7. 验收后才能清理旧镜像。

回滚时改回已验证标签并重建容器。数据库结构不兼容时必须同时恢复对应备份，不能只回滚容器。

## 12. 安全要求

- 管理员、媒体和节点凭据使用强随机值；管理员与兼容 Worker 令牌必须不同。
- 一次性注册码只写入 `0600` secret，稳态切换后删除。
- 长期节点 Token 只保存在节点数据卷，不复制到日志、截图和工单。
- SSH 使用密钥和 `BatchMode=yes`；平台不保存 SSH 密码或私钥。
- 节点控制端口只允许中央平台来源地址。
- 无可信 TLS 时，不向公网开放 5173、8081、8554 或节点端口。
- RK3588 inference 使用显式设备映射，不使用 `privileged`。
- 不执行 `docker system prune -a` 或批量镜像删除，除非逐项确认容器引用和回滚要求。

## 13. 常见故障

| 现象 | 主要检查 |
| --- | --- |
| Web 5173 无法访问 | frontend 状态、端口冲突、主机防火墙、`RKNODE_WEB_PORT` |
| Web 打开但 API 401 | 管理员令牌；清除错误 sessionStorage 后重试 |
| API 不 ready | API 日志、卷权限、环境变量、磁盘空间 |
| Media 不 healthy | ZLM 固定摘要、媒体密钥、8554/8081、Media 日志 |
| 节点一直 pending | enrollment overlay、Endpoint ID、secret 路径/有效期、节点到 5173 的路由 |
| 节点 claimed 不 enrolled | 中央到 endpoint 不可达，或身份/能力不一致 |
| 节点 enrolled 但 offline | 容器日志、`/health`、设备映射、平台地址、SSH 隧道 |
| RK3588 端口拒绝 | 容器、发布端口、路由；当前环境检查 11081/11082 隧道 |
| CUDA 节点离线 | `nvidia-smi`、Container Toolkit、镜像 CUDA 与驱动兼容性 |
| 转换任务 queued | converter 必须 `enrolled + online` 且 capability 匹配 |
| INT8 创建失败 | 需要匹配且 ready 的校准集；PPOCR Rec 不支持 INT8 |
| 推理部署失败 | published 状态、adapter/feature、context 容量、NPU 核心冲突 |
| 视频有画面无叠加 | 任务状态、SEI/时间戳、浏览器 Worker、播放会话和输出诊断 |
| 重启后 Token 丢失 | 是否删除/更换命名卷，Token 路径是否为 `/data/state/node-token` |

外部直接访问节点 `/health` 返回 HTTP 401，可以证明网络和端口已可达；完整健康内容必须带节点 Token。

## 14. 上线验收清单

- [ ] API、Web、Media 使用 `2026.08.20` 最终镜像。
- [ ] 5173、8554、8081 按规划开放，API/Media healthy。
- [ ] 管理员令牌和媒体密钥已生成且未泄露。
- [ ] trainer、converter、inference 均为 `enrolled + online`。
- [ ] 一次性注册码已解除挂载并删除，长期 Token 权限为 600。
- [ ] 代表性训练产出固定形状 ONNX。
- [ ] RK3588 转换返回部署就绪验证报告。
- [ ] published 模型可完成推理部署。
- [ ] 真实 RTSP 可播放，检测框/掩码/OCR 与 SEI 同步。
- [ ] 备份和回滚命令已按实际 Compose 项目名演练。

## 15. 构建与质量验证

平台与训练镜像在 amd64 构建：

```bash
RKNODE_RELEASE_VERSION=2026.08.20 RKNODE_SOURCE_REVISION=<源码标识> \
  bash scripts/build_offline_images.sh platform
bash scripts/build_offline_images.sh trainer-torch-cpu
bash scripts/build_offline_images.sh trainer-paddle-cpu
bash scripts/build_offline_images.sh trainer-torch-cuda
bash scripts/build_offline_images.sh trainer-paddle-cuda
```

RK3588 镜像必须在 arm64 板端原生构建：

```bash
RKNODE_RELEASE_VERSION=2026.08.20-business RKNODE_SOURCE_REVISION=<源码标识> \
  bash scripts/build_offline_images.sh rk3588
```

脚本默认把 RKNN 工具链运行时镜像压平为约 2.23 GB 的最终独立镜像。不要交付约 4.6 GB 的构建中间镜像。

```bash
.venv/bin/ruff check backend workers tests scripts
.venv/bin/pyright backend workers
.venv/bin/pytest -q
npm run build
npm run test:unit
npm run test:ui
```

离线包制作与安装见[离线部署手册](offline-deployment.md)。
