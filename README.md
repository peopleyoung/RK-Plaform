# RK3588 Platform

本项目提供中央平台、训练节点、RK3588 转换节点和 RK3588 推理节点。部署配置只使用 Compose YAML：所有运行环境变量都在对应 `compose.yaml` 的 `environment` 中，首次注册凭证使用 Compose `secrets` 挂载。仓库不再需要 `.env`、`bundle.env` 或 `docker compose --env-file`。

## 1. 部署拓扑

中央平台示例地址为 `172.16.66.249:5173`，API 在容器内监听 8000，Media 对外提供 RTSP `8554` 和 WS-FLV `8081`。节点服务统一监听节点宿主机地址映射的 10081/10082：训练节点示例 `172.16.66.249:10081`，板端直连地址为 `172.30.82.12:10081`、`172.30.82.12:10082`。当前现网仍通过中央主机 SSH 隧道 `172.29.0.1:11081`、`172.29.0.1:11082` 探活；直连端口恢复后再切换 Endpoint，隧道只作为应急连接方式维护。

节点的服务地址填写节点宿主机 IP 和端口；`RKNODE_PLATFORM_URL` 填中央平台地址，不是节点自身地址。平台创建 Endpoint 后取得 `RKNODE_ENDPOINT_ID` 和一次性注册码，注册码写入对应的 `./secrets/*-enrollment-token` 文件，权限设为 0600。

基础 Compose 使用 `RKNODE_NODE_TOKEN_FILE` 保存注册后的长期凭证；首次启动 overlay 通过 `RKNODE_ENROLLMENT_TOKEN_FILE` 只读挂载一次性注册码。

## 2. 推理媒体链路

RK3588 推理镜像内置 10 类可注册算子：`VideoCaptureNode`、`RkMppCaptureNode`、`InferNode`、`ByteTrackNode`、`SecondaryInferNode`、`AnalyticsNode`、`EventOutputNode`、`JsonOutputNode`、`KafkaOutputNode` 和 `ZlmSeiOutputNode`。

单个任务不会固定启用全部算子。创建任务时，用户从服务端算子目录选择节点并编辑默认参数；平台校验受限 DAG、模型适配器、节点能力和媒体约束，再保存完整图快照：

~~~text
Capture -> 主推理 -> [ByteTrack] -> [二级推理] -> [区域/越线] -> [事件]
                                      └-> JSONL/HTTP
                                      └-> [Kafka]
                                      └-> [ZLM SEI]
~~~

其中 `capture.rkmpp` 才能使用事件录像和 `output.zlm_sei`；ByteTrack、区域/越线分析和二级推理仅适用于 YOLO 检测任务。`AnalyticsNode` 同时承载区域和越线规则，`EventOutputNode` 同时承载事件抓拍和录像，Kafka 与 ZLM SEI 是独立输出分支。板端 YOLO 推理只接受 `YOLO_DFL_SPLIT`，旧 `V5` 和旧 `ByteTrack` 模型推理类型已移除；`ByteTrackNode` 跟踪算子仍保留。

新任务只接受 `graph` 契约，不接受顶层 `releaseId`、`media` 或 `analytics`。保存任务只生成草稿和不可变图修订，必须另行创建部署批次才能下发。旧任务不会自动转换，升级前按 [系统指南](docs/system-guide.md#旧推理任务迁移) 执行备份和显式清理。

当前在线和离线模板版本为 API/Web/Media/训练 `2026.08.26`、RK3588 转换/推理 `2026.08.26-business`，对应 C++ pipeline 为 `2026.08.26-business`；生成或部署离线包前仍以包内 `VERSION` 和 `manifest.json` 为准。

## 3. 中央平台

在仓库根目录编辑 [deploy/compose.yaml](deploy/compose.yaml)：

- `admin-token` 是前端登录令牌，默认示例为 `admin`，生产环境应直接在 YAML 中改成随机长字符串。
- `worker-token`、`zlm-api-secret`、`zlm-hook-identity` 也必须直接改在 YAML 中；不要提交真实值。
- 修改 `172.16.66.249` 为中央服务器可被浏览器和节点访问的宿主机 IP。不能提供域名或证书时使用 `http://IP:5173`。

检查并启动：

~~~bash
docker compose -f deploy/compose.yaml config --quiet
docker compose -f deploy/compose.yaml up -d --no-build
docker compose -f deploy/compose.yaml ps
~~~

浏览器打开 `http://<中央服务器IP>:5173`，登录令牌就是 Compose 中 `admin-token` 的值。升级时使用 `up -d --no-build --force-recreate`，不要删除 `platform-data`、`media-data` 和 `media-logs` 卷。

## 4. 节点注册

在平台的节点管理页创建 Endpoint，记录 Endpoint ID、节点类型、加速器和一次性注册码。将 Endpoint ID 写入节点 Compose 的 `RKNODE_ENDPOINT_ID`，将中央平台写入 `RKNODE_PLATFORM_URL`。节点端口由 Compose 的 `ports` 决定，平台登记的服务地址应填写节点宿主机 IP 加端口，例如训练 `172.16.66.249:10081`，板端转换 `172.30.82.12:10081`、推理 `172.30.82.12:10082`。

首次注册以训练节点为例：

~~~bash
cd deploy/nodes/trainer
mkdir -p secrets
umask 077
printf '%s' '<平台一次性注册码>' > secrets/trainer-enrollment-token
chmod 600 secrets/trainer-enrollment-token
docker compose -f compose.yaml -f compose.enrollment.yaml up -d --no-build
~~~

板端 RK3588：

~~~bash
cd deploy/nodes/rk3588
mkdir -p secrets output
umask 077
printf '%s' '<转换注册码>' > secrets/converter-enrollment-token
printf '%s' '<推理注册码>' > secrets/inference-enrollment-token
chmod 600 secrets/*-enrollment-token
docker compose -p rknode-rk3588 -f compose.yaml -f compose.enrollment.yaml up -d --no-build
~~~

平台状态依次为 `pending`、`claimed`、`enrolled`；两个 Endpoint 都显示 `enrolled` 且在线后，停止并移除 enrollment overlay，后续只使用基础 Compose：

~~~bash
docker compose -p rknode-rk3588 -f compose.yaml down
docker compose -p rknode-rk3588 -f compose.yaml up -d --no-build
~~~

## 5. 训练运行时选择

默认训练 Compose 使用 Torch CPU。需要其他运行时，叠加一个固定配置文件：

~~~bash
docker compose -f deploy/nodes/trainer/compose.yaml \
  -f deploy/nodes/trainer/compose.torch-cuda.yaml up -d --no-build
~~~

可选文件为 `compose.torch-cpu.yaml`、`compose.torch-cuda.yaml`、`compose.paddle-cpu.yaml`、`compose.paddle-cuda.yaml`。CUDA 主机必须安装 NVIDIA Container Toolkit；Paddle 版本和设备能力以对应 YAML 为准。

## 6. 运维命令

~~~bash
docker compose -f deploy/compose.yaml logs -f --tail=200 api frontend media
docker compose -f deploy/nodes/trainer/compose.yaml logs -f --tail=200 trainer
docker compose -p rknode-rk3588 -f deploy/nodes/rk3588/compose.yaml logs -f --tail=200 converter inference
docker compose -f deploy/compose.yaml down
~~~

节点不能直接暴露公网，优先使用同一内网或 VPN。没有证书时使用受控 IP:端口访问；临时 SSH 隧道只能由跳板机建立，禁止把 SSH 密码写入脚本或镜像，且不得保存 SSH 密码。

## 7. 离线部署

使用 `scripts/package_offline_bundle.py` 生成归档。归档包含固定版本镜像、`manifest.json`、Compose 文件和脚本，不包含 `.env` 或注册码。目标机解包后直接编辑 Compose 中的 `replace-with-*` 和 `CENTRAL_SERVER_IP`，再运行：

~~~bash
./load-images.sh
./configure-media-secrets.py --compose-file compose.yaml   # 仅平台包
./deploy.sh --enroll
./verify.sh --enroll
~~~

待注册完成后再运行 `./deploy.sh` 和 `./verify.sh`。离线包默认拒绝拉取和构建，镜像必须已通过 `load-images.sh` 导入。

## 旧版静态 Token 迁移

旧部署如果仍保留静态 Token，应先在平台为节点生成一次性注册码。将注册码写入固定 secret 文件，使用 enrollment overlay 启动一次；确认 `enrolled` 后移除 overlay，再清空 Compose 中旧的静态 Token 并重建容器。旧 Token 不得继续复制到日志、镜像或公网主机。

## 安全边界

节点宿主机 IP / 域名必须能访问中央平台，但服务端口不得暴露到公网。生产环境建议 VPN；无证书环境至少限制防火墙来源。平台与节点之间可使用 HTTPS 反向代理，临时 SSH 隧道必须设置 `BatchMode=yes`、密钥权限 0600、自动过期，并不得保存 SSH 密码。

更多步骤见 [docs/system-guide.md](docs/system-guide.md)、[docs/simple-node-deployment.md](docs/simple-node-deployment.md) 和 [docs/offline-deployment.md](docs/offline-deployment.md)。
