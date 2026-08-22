# RKNode 模型训练、转换与推理平台

RKNode 通过一个中央 Web/API 平台管理数据集、CPU/CUDA 训练、RK3588 模型转换、RK3588 推理部署和实时视频结果。当前版本统一采用“先在平台注册节点，再向节点下发一次性注册码”的接入方式；训练、转换和推理节点都通过同一套身份与健康探测流程上线。

## 文档入口

| 文档 | 适用人员 | 内容 |
| --- | --- | --- |
| [完整使用与部署手册](docs/system-guide.md) | 首次部署运维、平台管理员 | 架构、网络、平台部署、业务使用、升级、备份和故障处理 |
| [节点部署手册](docs/simple-node-deployment.md) | 节点运维 | CPU/CUDA 训练节点、RK3588 转换/推理节点、注册码下发和稳态切换 |
| [离线部署手册](docs/offline-deployment.md) | 隔离网运维 | 离线包制作、传输、校验、加载、部署和回滚 |
| [第三方源码清单](third_party/SOURCES.md) | 构建与审计人员 | 固定版本的训练、RKNN、媒体依赖来源 |

首次部署请从[完整使用与部署手册](docs/system-guide.md)开始，不要直接复制历史命令或使用无标签镜像。

## 当前发布版本

| 角色 | 最终镜像 | 架构 |
| --- | --- | --- |
| 平台 API | `rknode-platform-api:2026.08.20` | amd64 |
| 平台 Web | `rknode-platform-web:2026.08.20` | amd64 |
| 平台 Media | `rknode-platform-media:2026.08.20` | amd64 |
| Torch CPU 训练 | `rknode-trainer-torch-cpu:2026.08.20` | amd64 |
| Paddle CPU 训练 | `rknode-trainer-paddle-cpu:2026.08.20` | amd64 |
| Torch CUDA 12.4 训练 | `rknode-trainer-torch-cuda12.4:2026.08.20` | amd64 |
| Paddle CUDA 12.6 训练 | `rknode-trainer-paddle-cuda12.6:2026.08.20` | amd64 |
| RK3588 转换/推理 | `rknode-rk3588-node:2026.08.20-business` | arm64 |

RK3588 使用一个统一镜像启动 converter 和 inference 两个独立容器。当前最终镜像约 2.23 GB；构建期间产生的 RKNN 工具链中间镜像不是交付镜像。

生产部署禁止使用 `latest`、`local` 和 `<none>` 镜像。镜像的 `org.opencontainers.image.version`、`org.opencontainers.image.revision` 与架构必须在部署前核对。

## 当前已验证拓扑

```text
浏览器
  -> http://172.16.66.249:5173
       -> Web/Nginx -> API:8000（Compose 内部）
       -> Media WS-FLV:8081

训练节点
  平台 -> 172.16.66.249:10081

RK3588 板端
  平台 -> 172.29.0.1:11081 -> SSH 隧道 -> 板端 127.0.0.1:10081（转换）
  平台 -> 172.29.0.1:11082 -> SSH 隧道 -> 板端 127.0.0.1:10082（推理）
  节点 -> http://172.16.66.249:5173（中央平台）
```

`172.30.82.12:124` 当前是板端 SSH 管理入口，不等同于节点服务监听地址。中央服务器不能直接访问 `172.30.82.12:10081/10082` 时，平台必须继续使用 `172.29.0.1:11081/11082`。修通路由并完成真实业务验收后，才能改为板端宿主机 IP 加端口并停止隧道。

## 快速检查

```bash
# 中央平台
curl -fsS http://127.0.0.1:5173/api/v1/ready
docker compose --env-file deploy/.env -f deploy/compose.yaml ps

# 本机训练节点
docker compose -p rknode-direct-cpu \
  --env-file deploy/nodes/trainer/.env \
  -f deploy/nodes/trainer/compose.yaml ps

# RK3588 节点（在板端部署目录执行）
docker compose -p rknode-rk3588 --env-file .env -f compose.yaml ps
```

平台浏览器地址为 `http://<中央服务器IP>:5173`。首次打开时输入 `deploy/.env` 中的管理员令牌；令牌只保存在当前浏览器会话的 `sessionStorage` 中。

## 统一节点接入原则

1. 平台“系统设置”中的服务地址，是中央平台访问节点的地址。
2. 节点 `.env` 中的 `RKNODE_PLATFORM_URL`，是节点访问中央平台的地址，不附加 `/api/v1`。
3. 平台生成一次性注册码；运维把它保存为权限 `0600` 的 secret 文件。
4. 首次启动叠加 `compose.enrollment.yaml`，节点领取并持久化长期 Token。
5. 平台状态达到 `enrolled + online` 后，使用基础 Compose 重建容器，解除一次性 secret 挂载。
6. 确认重启仍在线后删除一次性注册码；长期 Token 保存在节点数据卷 `/data/state/node-token`。

统一变量和地址索引：平台表单使用“节点宿主机 IP / 域名”及服务端口；节点使用 `RKNODE_ENDPOINT_ID`、`RKNODE_PLATFORM_URL`、`RKNODE_ENROLLMENT_TOKEN_FILE` 和 `RKNODE_NODE_TOKEN_FILE`。状态按 `pending -> claimed -> enrolled` 变化。当前训练节点为 `172.16.66.249:10081`，RK3588 转换和推理通过 `172.29.0.1:11081/11082` 接入；直连验收地址为 `172.30.82.12:10081` 和 `172.30.82.12:10082`。

注册码不是长期运行凭据，不要把管理员令牌、注册码、长期节点 Token、SSH 密码或私钥写入文档、URL、镜像、日志和工单。

## 开发验证

```bash
uv sync --all-groups
npm ci
.venv/bin/ruff check backend workers tests scripts
.venv/bin/pyright backend workers
.venv/bin/pytest -q
npm run build
npm run test:unit
npm run test:ui
```

实际生产部署、节点矩阵和离线交付步骤见上方三份运维手册。

## 旧版静态 Token 迁移

旧节点可在维护窗口使用静态 Token 过渡，但新节点必须使用平台注册、一次性注册码和长期 Token 文件的统一 enrollment 流程。迁移步骤见[节点部署手册](docs/simple-node-deployment.md)。

安全边界：节点控制端口不得暴露到公网；跨网段优先使用 VPN，必要时使用 HTTPS 或 SSH 隧道。平台和文档不得保存 SSH 密码。
