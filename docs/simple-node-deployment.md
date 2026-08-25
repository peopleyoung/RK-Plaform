# 节点快速部署

本文只描述当前 Compose-only 节点流程。所有节点采用统一接入方式：平台分配 Endpoint ID 和一次性注册码，节点通过 RKNODE_PLATFORM_URL 回连中央平台；节点服务地址则是节点宿主机 IP 加端口。

## 1. 先准备中央平台

中央平台示例为 172.16.66.249:5173。确认 deploy/compose.yaml 中的 admin-token、worker-token、Media 两个 secret 和服务器 IP 已填写：

~~~bash
python3 scripts/configure_media_secrets.py --compose-file deploy/compose.yaml
docker compose -f deploy/compose.yaml config --quiet
docker compose -f deploy/compose.yaml up -d --no-build
~~~

在浏览器进入 http://172.16.66.249:5173，创建节点 Endpoint。保存 Endpoint ID 和一次性注册码，注册码只显示一次。

## 2. 配置字段

节点 Compose 中必须确认这些字段：

| 字段 | 含义 |
| --- | --- |
| RKNODE_ENDPOINT_ID | 平台创建的 Endpoint ID |
| RKNODE_PLATFORM_URL | 中央平台地址，例如 http://172.16.66.249:5173 |
| RKNODE_NODE_NAME | 节点唯一名称 |
| RKNODE_NODE_PORT | 容器监听端口，训练/转换为 10081，推理容器为 10081 |
| RKNODE_NODE_TOKEN_FILE | 已注册后保存节点 Token 的容器路径 |
| RKNODE_ENROLLMENT_TOKEN_FILE | 首次注册时的 Compose secret 路径 |

服务地址不是 RKNODE_PLATFORM_URL：平台登记训练地址为 172.16.66.249:10081，板端直连地址为 172.30.82.12:10081/10082。当前运行环境仍通过中央主机上的 SSH 隧道 `172.29.0.1:11081`（转换）和 `172.29.0.1:11082`（推理）探活；这是现网连接方式，隧道断开时节点会离线。直连端口恢复后，应将平台 Endpoint 切换到板端直连地址。

当前在线镜像矩阵为：API/Web `2026.08.25`、Media `2026.08.24`、RK3588 转换/推理 `2026.08.25-business`、Torch CPU 训练 `2026.08.24`。离线包仍按包内 `deploy/offline/VERSION` 和 `manifest.json` 管理，不要把在线标签直接填入旧离线包。

## 3. Torch/Paddle 训练节点

默认文件是 [deploy/nodes/trainer/compose.yaml](../deploy/nodes/trainer/compose.yaml)，直接编辑 YAML 中的 Endpoint ID、平台 URL、节点名称和能力。需要 CUDA/Paddle 时叠加对应文件：

~~~bash
# Torch CPU
docker compose -f deploy/nodes/trainer/compose.yaml up -d --no-build
# Torch CUDA
docker compose -f deploy/nodes/trainer/compose.yaml -f deploy/nodes/trainer/compose.torch-cuda.yaml up -d --no-build
# Paddle CPU
docker compose -f deploy/nodes/trainer/compose.yaml -f deploy/nodes/trainer/compose.paddle-cpu.yaml up -d --no-build
~~~

首次注册使用固定 secret 文件：

~~~bash
cd deploy/nodes/trainer
mkdir -p secrets
umask 077
printf '%s' '<训练注册码>' > secrets/trainer-enrollment-token
chmod 600 secrets/trainer-enrollment-token
docker compose -f compose.yaml -f compose.enrollment.yaml up -d --no-build
~~~

平台看到 pending -> claimed -> enrolled 且节点在线后，移除 enrollment overlay：

~~~bash
docker compose -f compose.yaml -f compose.enrollment.yaml down
docker compose -f compose.yaml up -d --no-build
~~~

## 4. RK3588 转换/推理节点

板端宿主机示例为 172.30.82.12。编辑 [deploy/nodes/rk3588/compose.yaml](../deploy/nodes/rk3588/compose.yaml)：

- 转换服务使用平台转换 Endpoint ID，映射 172.30.82.12:10081。
- 推理服务使用平台推理 Endpoint ID，映射 172.30.82.12:10082。
- 两者的 RKNODE_PLATFORM_URL 都指向中央平台。
- 推理服务需要 RKNN/NPU 设备，转换服务只保留转换所需设备。

将两次注册码写入固定文件并启动：

~~~bash
cd deploy/nodes/rk3588
mkdir -p secrets output
umask 077
printf '%s' '<转换注册码>' > secrets/converter-enrollment-token
printf '%s' '<推理注册码>' > secrets/inference-enrollment-token
chmod 600 secrets/*-enrollment-token
docker compose -p rknode-rk3588 -f compose.yaml -f compose.enrollment.yaml up -d --no-build
~~~

确认两个 Endpoint 都是 enrolled 后，仅启动基础 Compose。检查板端：

~~~bash
docker compose -p rknode-rk3588 -f compose.yaml ps
curl -fsS http://172.30.82.12:10081/health
curl -fsS http://172.30.82.12:10082/health
~~~

推理镜像支持 MPP 硬解码、主 RKNN、ByteTrack、二级 YOLO、区域/越线分析、事件抓拍/录像、JSONL/HTTP、Kafka 和 ZLM SEI。任务创建时按 `media` 和 `analytics` 选择分支；不会固定串联全部算子。ZLM SEI 和事件录像要求 RTSP + `decoder=rkmpp`，区域/越线要求 YOLO + ByteTrack，Kafka 与 ZLM 是独立输出分支。详细节点契约见 [runtime-adapter README](../deploy/rk3588/runtime-adapter/README.md)。

## 5. 故障排查

~~~bash
docker compose -f compose.yaml config --quiet
docker compose -f compose.yaml logs --tail=200 <服务>
ss -lntp | grep -E '10081|10082'
curl -v http://<中央平台IP>:5173/api/v1/ready
~~~

RKNODE_PLATFORM_URL 错误会导致节点离线；节点服务地址错误会导致平台探活失败。优先使用同一内网或 VPN，服务端口不得暴露到公网。没有证书时使用 IP:端口；可以用 HTTPS 反向代理保护跨网访问。SSH 隧道只能临时使用，不得保存 SSH 密码。

## 旧版静态 Token 迁移

旧 Compose 若仍有静态 Token，先在平台创建一次性注册码，写入 secret 文件，使用 compose.enrollment.yaml 启动并等待 enrolled。停止 overlay 后清空旧 Token 并重建，保留 RKNODE_NODE_TOKEN_FILE 生成的节点状态卷。

## 6. 安全检查

节点宿主机 IP / 域名应只对中央平台和运维网开放。注册码文件权限必须是 0600，不得提交 Git。生产部署需要 VPN 或 HTTPS；临时 SSH 隧道必须使用密钥、限制来源、自动过期，并不得保存 SSH 密码。
