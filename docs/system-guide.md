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

端口只开放给内网或 VPN，服务不得暴露到公网。无证书部署使用受控 IP:端口；有条件时使用 HTTPS 反向代理。SSH 隧道仅作为临时故障排查手段，隧道示例地址为 172.29.0.1:11081 和 172.29.0.1:11082，跳板机应使用密钥、BatchMode=yes 和自动过期，不得保存 SSH 密码。

## 旧版静态 Token 迁移

旧节点仍使用静态 Token 时，在平台为同一 Endpoint 重新生成一次性注册码，写入对应 secret 文件并使用 compose.enrollment.yaml 启动。确认 enrolled 后停止 overlay，清空 Compose 中旧 Token 字段并重建服务。旧 Token 不得写入日志、镜像、Git 或公网主机。

## 7. 备份和升级

备份 Compose YAML、平台数据卷、节点数据卷和权限为 0600 的 secret 文件。升级时先执行 docker compose ... config --quiet，再使用 up -d --no-build --force-recreate；不要删除持久化卷。升级后检查平台 ready、节点 enrolled + online、Media RTSP/WS-FLV 和推理设备挂载。
