# 系统部署与运维指南

本文面向首次部署运维人员。当前版本只使用 Docker Compose YAML，不读取 .env 或 bundle.env。Compose 的 environment 是最终生效值，节点注册码通过 Compose secrets 注入。

## 1. 地址和职责

中央平台示例地址为 http://172.16.66.249:5173。平台 API 在 Compose 网络内为 http://api:8000，Media 对外暴露 RTSP 172.16.66.249:8554 和 WS-FLV 172.16.66.249:8081。

节点服务地址必须填写节点宿主机 IP 和端口：

| 节点 | 服务地址示例 | Compose 映射 |
| --- | --- | --- |
| Torch 训练 | 172.16.66.249:10081 | 10081:10081 |
| RK3588 转换 | 172.30.82.12:10081 | 10081:10081 |
| RK3588 推理 | 172.30.82.12:10082 | 10082:10081 |
| 临时 SSH 隧道 | 172.29.0.1:11081、172.29.0.1:11082 | 仅应急验证 |

RKNODE_PLATFORM_URL 始终填写中央平台地址；节点服务地址填写节点宿主机地址。节点宿主机 IP / 域名需要能访问中央平台，不能提供域名或证书时直接使用 IP:端口。

## 2. 中央平台首次启动

编辑 [deploy/compose.yaml](../deploy/compose.yaml) 的 x-rknode-platform-config：

- admin-token 为前端登录令牌，示例为 admin。
- worker-token 为节点调用 API 的共享令牌。
- zlm-api-secret 和 zlm-hook-identity 必须使用随机长字符串。
- 将所有 172.16.66.249 改为中央服务器实际内网 IP。

Media 密钥可以由仓库脚本直接写回 Compose anchor：

~~~bash
python3 scripts/configure_media_secrets.py --compose-file deploy/compose.yaml
chmod 600 deploy/compose.yaml
docker compose -f deploy/compose.yaml config --quiet
docker compose -f deploy/compose.yaml up -d --no-build
docker compose -f deploy/compose.yaml ps
~~~

浏览器访问 http://<中央服务器IP>:5173，使用 Compose 中 admin-token 登录。健康检查：

~~~bash
curl -fsS http://127.0.0.1:5173/api/v1/ready
docker compose -f deploy/compose.yaml logs --tail=100 api frontend media
~~~

## 3. 平台注册 Endpoint

在平台节点管理页逐个创建训练、转换、推理 Endpoint，保存 Endpoint ID 和一次性注册码。注册状态为 pending、claimed、enrolled；只有显示 enrolled 且心跳在线才可进入稳定运行。

把 Endpoint ID 直接写入节点 Compose 的 RKNODE_ENDPOINT_ID，把中央平台写入 RKNODE_PLATFORM_URL。注册码只落盘到权限 0600 的 secret 文件，不写入环境变量或命令历史。

RKNODE_ENROLLMENT_TOKEN_FILE 是首次启动时的 secret 挂载路径；RKNODE_NODE_TOKEN_FILE 是注册成功后保存在数据卷中的长期凭证路径。

## 4. 训练节点

编辑 [deploy/nodes/trainer/compose.yaml](../deploy/nodes/trainer/compose.yaml)：

- RKNODE_ENDPOINT_ID 使用平台创建的训练 Endpoint ID。
- RKNODE_PLATFORM_URL 使用中央平台，例如 http://172.16.66.249:5173。
- RKNODE_NODE_NAME、能力、端口和宿主机服务地址按实际机器修改。
- 运行时可叠加 compose.torch-cuda.yaml、compose.paddle-cpu.yaml 或 compose.paddle-cuda.yaml。

首次注册：

~~~bash
cd deploy/nodes/trainer
mkdir -p secrets
umask 077
printf '%s' '<训练 Endpoint 一次性注册码>' > secrets/trainer-enrollment-token
chmod 600 secrets/trainer-enrollment-token
docker compose -f compose.yaml -f compose.enrollment.yaml up -d --no-build
docker compose -f compose.yaml -f compose.enrollment.yaml ps
~~~

注册完成后切换基础文件：

~~~bash
docker compose -f compose.yaml -f compose.enrollment.yaml down
docker compose -f compose.yaml up -d --no-build
~~~

## 5. RK3588 转换和推理节点

编辑 [deploy/nodes/rk3588/compose.yaml](../deploy/nodes/rk3588/compose.yaml) 中两个服务：

- 转换 RKNODE_ENDPOINT_ID 对应平台转换 Endpoint，服务地址为 172.30.82.12:10081。
- 推理 RKNODE_ENDPOINT_ID 对应平台推理 Endpoint，服务地址为 172.30.82.12:10082。
- 两个服务的 RKNODE_PLATFORM_URL 都指向中央平台，而不是 172.30.82.12。
- 推理服务保留 NPU、RGA、MPP、DMA 和设备树挂载。

首次注册：

~~~bash
cd deploy/nodes/rk3588
mkdir -p secrets output
umask 077
printf '%s' '<转换一次性注册码>' > secrets/converter-enrollment-token
printf '%s' '<推理一次性注册码>' > secrets/inference-enrollment-token
chmod 600 secrets/*-enrollment-token
docker compose -p rknode-rk3588 -f compose.yaml -f compose.enrollment.yaml up -d --no-build
~~~

两个 Endpoint 都处于 enrolled 后：

~~~bash
docker compose -p rknode-rk3588 -f compose.yaml -f compose.enrollment.yaml down
docker compose -p rknode-rk3588 -f compose.yaml up -d --no-build
~~~

## 6. 日常运维和网络

~~~bash
docker compose -f deploy/compose.yaml ps
docker compose -f deploy/nodes/trainer/compose.yaml logs -f --tail=200 trainer
docker compose -p rknode-rk3588 -f deploy/nodes/rk3588/compose.yaml logs -f --tail=200 converter inference
~~~

端口只开放给内网或 VPN，服务不得暴露到公网。无证书部署使用受控 IP:端口；有条件时使用 HTTPS 反向代理。当前现网通过中央主机 SSH 隧道 `172.29.0.1:11081` 和 `172.29.0.1:11082` 探活；直连端口恢复后应切换 Endpoint。隧道只作为应急连接方式维护，跳板机应使用密钥、BatchMode=yes 和自动过期，不得保存 SSH 密码。

## 7. 推理图编排

推理容器当前注册 10 类可选算子：

| 类别 | 业务算子 ID | 运行时算子 | 作用 |
| --- | --- | --- | --- |
| 输入 | `capture.opencv` | `VideoCaptureNode` | OpenCV 兼容输入 |
| 输入 | `capture.rkmpp` | `RkMppCaptureNode` | RTSP H.264/H.265 的 MPP 硬解码，并保留原始编码包 |
| 推理 | `inference.primary` | `InferNode` | 主 RKNN 模型推理 |
| 跟踪 | `processing.bytetrack` | `ByteTrackNode` | YOLO 检测结果跟踪 |
| 推理 | `inference.secondary` | `SecondaryInferNode` | 对指定类别执行二级 YOLO 推理 |
| 分析 | `processing.analytics` | `AnalyticsNode` | 区域进入/离开和越线规则 |
| 事件 | `processing.events` | `EventOutputNode` | JPEG 抓拍和原码流录像 |
| 输出 | `output.json` | `JsonOutputNode` | JSONL 或 HTTP 结构化结果 |
| 输出 | `output.kafka` | `KafkaOutputNode` | 异步 Kafka 结果输出 |
| 输出 | `output.zlm_sei` | `ZlmSeiOutputNode` | 将 schema-v2 结果写入原始 RTSP 码流的 SEI |

这些是镜像支持的算子，不代表每个任务都会实例化全部节点。Web 端从 `GET /api/v1/inference-operator-catalog` 读取目录和默认参数，并调用 `POST /api/v1/inference-graphs/validate` 校验。创建任务保存为草稿；编辑时提交 `baseRevisionId` 做乐观并发控制。只有图语义改变才生成新的不可变 revision，画布坐标不参与语义哈希。典型链路为：

~~~text
Capture -> Infer -> [ByteTrack] -> [SecondaryInfer 0..N]
                                  -> [Analytics]
                                  -> [EventOutput]
                                  -> JsonOutput
                                  -> [KafkaOutput]
                                  -> [ZlmSeiOutput]
~~~

方括号表示按任务配置启用的分支，JSON、Kafka 和 ZLM SEI 是独立输出，不是互相串联的必经节点。

配置约束：

- 图必须恰好包含一个 Capture、一个主推理和至少一个输出；每个运行时节点只允许一个上游，二级推理形成单链。
- `capture.rkmpp` 只接受 RTSP 输入；ZLM SEI 和事件录像必须使用 RKMPP。
- ByteTrack 只适用于 YOLO 检测模型；区域/越线规则必须同时启用 ByteTrack。
- 二级推理只能引用已发布的 YOLO 检测版本，最多四个独立版本。
- `AnalyticsNode` 同时处理区域和越线；`EventOutputNode` 同时处理抓拍和录像。
- Kafka/ZLM sink 失败不会改变本地推理任务健康状态，失败分支会重试或丢弃，不阻塞 NPU 主链。
- 选择 `output.zlm_sei` 时在算子参数中直接选择在线媒体网关并填写流名称；平台只在 Agent desired state 中注入带短期凭证的 `publishUri`。
- 板端 YOLO 只接受 `yolo_dfl_split_v1` / `YOLO_DFL_SPLIT`。训练目录可以保留其他导出变体，但旧 V5/平坦 ByteTrack 输出不能登记为可部署推理版本。

任务列表约每 3 秒自动刷新，部署详情约每 2 秒刷新；不需要手动刷新页面。保存任务不会重启当前运行实例，操作员需要选择任务创建部署批次。部署目标固定任务当时的 `graphRevisionId`、图快照和语义哈希，回滚也使用目标保存的上一份图快照。

检查当前任务实际链路时，以板端 `/data/runtime/revisions/<revision>/pipelines/` 中的任务图为准，而不是只看节点能力列表。节点健康检查和版本核对：

~~~bash
curl -fsS -H "Authorization: Bearer <node-token>" http://<node-host>:10082/health
docker compose -p rknode-rk3588 -f compose.yaml ps
docker compose -p rknode-rk3588 -f compose.yaml logs --tail=200 inference
~~~

当前在线发布线为 API/Web `2026.08.25`、Media `2026.08.24`、RK3588 转换/推理 `2026.08.25-business`；训练角色仍为 `2026.08.24`。离线包尚未切换到这条在线发布线，使用前必须核对包内 `VERSION` 与镜像清单。

### 旧推理任务迁移

新 API 不兼容旧任务请求和旧数据库任务。先停止推理 Agent，再做只读预览：

~~~bash
python scripts/migrate_inference_graph_v1.py --database /path/to/platform.db
~~~

确认统计后，指定一个不存在的备份文件执行清理：

~~~bash
python scripts/migrate_inference_graph_v1.py \
  --database /path/to/platform.db \
  --backup /path/to/platform.before-graph-v1.db \
  --execute
~~~

脚本先用 SQLite 在线备份 API 生成完整备份，再清除旧推理任务、图修订和部署历史，并递增受影响节点的 desired revision。脚本不会自动执行，不支持内存数据库或 PostgreSQL。需要恢复时先停止 API，把当前数据库移走，再用备份文件恢复原路径；旧服务只能读取旧数据库，新服务要求重新编排任务。

## 旧版静态 Token 迁移

旧节点仍使用静态 Token 时，在平台为同一 Endpoint 重新生成一次性注册码，写入对应 secret 文件并使用 compose.enrollment.yaml 启动。确认 enrolled 后停止 overlay，清空 Compose 中旧 Token 字段并重建服务。旧 Token 不得写入日志、镜像、Git 或公网主机。

## 8. 备份和升级

备份 Compose YAML、平台数据卷、节点数据卷和权限为 0600 的 secret 文件。升级时先执行 docker compose ... config --quiet，再使用 up -d --no-build --force-recreate；不要删除持久化卷。升级后检查平台 ready、节点 enrolled + online、Media RTSP/WS-FLV 和推理设备挂载。
