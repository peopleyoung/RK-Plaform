# 离线部署手册

本文面向隔离网或无公网环境，说明如何制作、传输、校验和部署当前发布版本。离线只表示部署阶段不访问公网；中央平台和节点之间仍必须通过同一内网、VPN 或受控 SSH 隧道互通。

首次部署建议先阅读[完整使用与部署手册](system-guide.md)，节点环境变量和注册码流程见[节点部署手册](simple-node-deployment.md)。

## 旧版静态 Token 迁移

旧离线包可在回滚窗口临时使用静态 Token，但新离线包必须使用平台生成的一次性注册码。领取成功后清空静态变量，设置 `RKNODE_ENROLLMENT_COMPLETE=true`，确认 enrollment secret 已卸载，再删除一次性文件并重启验收。

安全边界：节点控制端口不得暴露到公网；跨网段优先使用 VPN，必要时使用 HTTPS 或 SSH 隧道。平台和文档不得保存 SSH 密码。

## 1. 当前交付矩阵

当前最终镜像版本如下：

| 角色 | 最终镜像 | 架构 | 是否纳入标准离线交付 |
| --- | --- | --- | --- |
| 平台 API | `rknode-platform-api:2026.08.20` | amd64 | 是 |
| 平台 Web | `rknode-platform-web:2026.08.20` | amd64 | 是 |
| 平台 Media | `rknode-platform-media:2026.08.20` | amd64 | 是 |
| Torch/Paddle CPU/CUDA 训练 | `rknode-trainer-*:2026.08.20` | amd64 | 否，单独受控交付 |
| RK3588 转换/推理 | `rknode-rk3588-node:2026.08.20-business` | arm64 | 是 |

标准发布生成两个外层归档：

| 外层归档 | 内容 |
| --- | --- |
| `rknode-platform-amd64-2026.08.20.tar` | 三个中央平台镜像、平台 Compose 和脚本 |
| `rknode-rk3588-node-arm64-2026.08.20-business.tar` | 一个 RK3588 统一镜像、converter/inference Compose 和脚本 |

外层归档内的镜像文件是 `images/*.tar.gz`。因此“镜像数量”和“外层 tar 数量”不是一一对应关系：平台包含 3 个镜像，RK3588 包含 1 个镜像，标准交付仍是 2 个外层 tar。转换和推理拆分包可以按需生成，但不应与统一 RK3588 包重复部署。

构建过程中的无标签镜像、旧版本镜像和 RKNN 工具链中间层都不属于交付物。生产部署只使用带完整版本标签且通过 manifest 校验的最终镜像。

## 2. 离线边界与网络拓扑

离线包不包含管理员令牌、一次性注册码、长期节点 Token、SSH 私钥、数据库、模型、业务数据或历史缓存。上述内容必须在目标环境内单独生成或下发。

当前已验证拓扑：

~~~text
浏览器 -> 中央平台 http://<中央IP>:5173
平台 -> 训练节点 <训练IP>:10081
平台 -> 172.29.0.1:11081 -> SSH 隧道 -> 板端 127.0.0.1:10081（转换）
平台 -> 172.29.0.1:11082 -> SSH 隧道 -> 板端 127.0.0.1:10082（推理）
板端节点 -> http://<中央IP>:5173
~~~

当前 `172.30.82.12:124` 只是板端 SSH 管理入口，不是节点服务端口。只有中央平台能直接访问板端服务并完成真实转换、推理和重启验收后，平台 endpoint 才能改成 `172.30.82.12:10081` 和 `172.30.82.12:10082`。离线部署脚本不会自动修改平台 endpoint 或建立/停止隧道。

统一接入索引：平台登记字段是“节点宿主机 IP / 域名”和服务端口；节点使用 `RKNODE_ENDPOINT_ID`、`RKNODE_PLATFORM_URL`、`RKNODE_ENROLLMENT_TOKEN_FILE` 和 `RKNODE_NODE_TOKEN_FILE`。状态按 `pending -> claimed -> enrolled` 变化。训练节点示例地址为 `172.16.66.249:10081`；RK3588 当前隧道地址为 `172.29.0.1:11081`、`172.29.0.1:11082`，板端直连地址为 `172.30.82.12:10081`、`172.30.82.12:10082`。

## 3. 构建机要求

构建机需要：

- Docker、BuildKit 和 `docker compose`；
- 完整项目源码、足够磁盘和临时空间；
- 平台包在 amd64 主机制作；
- RK3588 镜像在原生 arm64 板端或受控 arm64 builder 制作；
- 构建完成后能读取镜像的 OCI 版本、revision、架构和 `offline-ready` 标签。

先检查构建机和源码：

~~~bash
uname -m
docker version
docker compose version
git rev-parse --short HEAD
~~~

平台镜像在本机构建：

~~~bash
RKNODE_RELEASE_VERSION=2026.08.20 \
RKNODE_SOURCE_REVISION=rtsp-sei-2026.08.20 \
scripts/build_offline_images.sh platform
~~~

Torch CPU 训练镜像如需单独离线交付，也在 amd64 构建机制作；它不由本章的标准打包命令导出：

~~~bash
RKNODE_RELEASE_VERSION=2026.08.20 \
RKNODE_SOURCE_REVISION=rtsp-sei-trainer-2026.08.20 \
scripts/build_offline_images.sh trainer-torch-cpu
~~~

Paddle CPU、Torch CUDA 12.4 和 Paddle CUDA 12.6 使用相同的 `2026.08.20` 版本规则，分别按项目脚本和训练节点模板构建。CUDA 构建机还必须先通过 `nvidia-smi` 和 Container Toolkit 验证。

RK3588 推理/转换镜像在板端或原生 arm64 builder 构建：

~~~bash
RKNODE_RELEASE_VERSION=2026.08.20-business \
RKNODE_SOURCE_REVISION=rtsp-sei-business-2026.08.20-device-tree-fallback \
scripts/build_offline_images.sh rk3588
~~~

RK3588 构建脚本会将构建环境和运行时分层，随后通过 compact 流程导出为最终精简镜像。当前最终镜像约 2.23 GB；构建期间约 4.6 GB 的工具链中间镜像不应打包、打标签或部署。

## 4. 构建后镜像门禁

在导出之前逐个检查镜像：

~~~bash
docker image inspect rknode-platform-api:2026.08.20 \
  --format '{{.Architecture}} {{.Os}} {{index .Config.Labels "org.opencontainers.image.version"}} {{index .Config.Labels "org.opencontainers.image.revision"}} {{index .Config.Labels "io.rknode.offline-ready"}}'

docker image inspect rknode-rk3588-node:2026.08.20-business \
  --format '{{.Architecture}} {{.Os}} {{index .Config.Labels "org.opencontainers.image.version"}} {{index .Config.Labels "org.opencontainers.image.revision"}} {{index .Config.Labels "io.rknode.offline-ready"}}'
~~~

期望值分别为：

- 平台镜像：`amd64 linux`、版本 `2026.08.20`、`offline-ready=true`；
- RK3588 镜像：`arm64 linux`、版本 `2026.08.20-business`、`offline-ready=true`。

不接受 `latest`、`local`、`<none>`、旧日期标签或架构不匹配的镜像。不要把旧镜像重新打标签冒充本次构建。

## 5. 生成离线包

### 5.1 标准中央平台包

在 amd64 构建机执行：

~~~bash
python3 scripts/package_offline_bundle.py \
  platform-amd64 \
  --version 2026.08.20
~~~

### 5.2 标准 RK3588 统一包

在 arm64 板端或 arm64 builder 执行：

~~~bash
python3 scripts/package_offline_bundle.py \
  rk3588-node-arm64 \
  --version 2026.08.20-business
~~~

如果目标镜像已经确认是 arm64，但打包机架构不同，只能显式使用以下选项；这不会跨架构构建镜像：

~~~bash
python3 scripts/package_offline_bundle.py \
  rk3588-node-arm64 \
  --version 2026.08.20-business \
  --allow-cross-arch
~~~

转换和推理必须拆开交付时，可分别使用：

~~~bash
python3 scripts/package_offline_bundle.py converter-rk3588-arm64 --version 2026.08.20-business
python3 scripts/package_offline_bundle.py inference-rk3588-arm64 --version 2026.08.20-business
~~~

拆分包和统一包不要同时在同一板端、同一 Compose 项目启动。

脚本默认输出到 `release/offline/`。使用 `--directory-only` 可保留目录，便于审查：

~~~bash
python3 scripts/package_offline_bundle.py \
  platform-amd64 \
  --version 2026.08.20 \
  --directory-only
~~~

## 6. 包内容与完整性校验

每个离线包包含：

~~~text
images/*.tar.gz       镜像压缩档
compose*.yaml         no-build Compose
.env.example          不含秘密的环境模板
bundle.env            包规格、版本和 Compose 文件清单
manifest.json         镜像 ID、架构、revision 和档案摘要
SHA256SUMS            包内文件校验
load-images.sh        导入镜像并验证标签/架构
deploy.sh              --pull never --no-build 部署
verify.sh              健康、Token 和端口验收
stop.sh                停止容器但保留卷
README.md              包内速查
~~~

平台包另外包含 `configure-media-secrets.py`，用于在目标机生成 Media 的本地密钥。包内 `manifest.json` 的 `secretsIncluded` 和 `persistentDataIncluded` 必须均为 `false`。

构建机记录外层归档摘要：

~~~bash
cd release/offline
sha256sum rknode-platform-amd64-2026.08.20.tar \
  rknode-rk3588-node-arm64-2026.08.20-business.tar \
  > SHA256SUMS.outer
~~~

目标机收到文件后先校验，再解包：

~~~bash
sha256sum -c SHA256SUMS.outer
tar -xf rknode-platform-amd64-2026.08.20.tar
cd rknode-platform-amd64-2026.08.20
sha256sum -c SHA256SUMS
~~~

解包后必须看到 `manifest.json`、`bundle.env`、`.env.example` 和 `images/`。校验失败时停止部署，重新传输归档；不要通过删除校验项或重新压缩文件绕过门禁。

## 7. 传输与导入镜像

可使用受控介质、内网 `rsync` 或 `scp` 传输外层 tar 和 `SHA256SUMS.outer`。传输过程中不携带秘密文件：

~~~bash
rsync -av --progress release/offline/rknode-platform-amd64-2026.08.20.tar \
  release/offline/rknode-rk3588-node-arm64-2026.08.20-business.tar \
  release/offline/SHA256SUMS.outer <目标主机>:/srv/rknode-release/
~~~

目标目录内执行：

~~~bash
sha256sum -c SHA256SUMS.outer
tar -xf rknode-<bundle>-<version>.tar
cd rknode-<bundle>-<version>
sha256sum -c SHA256SUMS
./load-images.sh
docker image ls --format '{{.Repository}}:{{.Tag}}' | grep '^rknode-'
~~~

`load-images.sh` 会使用 `docker load` 导入包内压缩档，并检查镜像架构、版本和 `io.rknode.offline-ready=true`。导入和后续部署均不执行 pull 或 build。

## 8. 离线中央平台部署

目标中央服务器必须是 amd64，并已安装 Docker Compose。进入已解包的平台目录：

~~~bash
cp .env.example .env
chmod 600 .env
~~~

编辑 `.env`，至少填写：

~~~dotenv
RKNODE_ADMIN_TOKEN=<随机管理员令牌>
RKNODE_WORKER_TOKEN=<兼容旧 worker 的随机令牌>
RKNODE_CORS_ORIGINS=http://<中央服务器IP>:5173
RKNODE_PUBLIC_API_URL=http://<中央服务器IP>:5173/api/v1
RKNODE_MEDIA_PUBLISH_HOST=<中央服务器IP>
RKNODE_MEDIA_PLAYBACK_HOST=<中央服务器IP>
RKNODE_WEB_PORT=5173
RKNODE_MEDIA_RTSP_PORT=8554
RKNODE_MEDIA_WS_PORT=8081
~~~

通过脚本生成 Media 本地密钥。脚本只写入目标机 `.env`，不会从公网下载：

~~~bash
./configure-media-secrets.py --env-file .env
./deploy.sh
./verify.sh
~~~

验收：

~~~bash
curl -fsS http://127.0.0.1:5173/api/v1/ready
docker compose --env-file .env -f compose.yaml ps
nc -z 127.0.0.1 8554
nc -z 127.0.0.1 8081
~~~

浏览器访问 `http://<中央服务器IP>:5173`。前端管理员令牌只保存在当前浏览器会话；不要把它写入前端镜像或 URL。

## 9. 离线 RK3588 节点部署

目标板端必须是 arm64，并能访问中央平台。进入已解包的 RK3588 统一包目录：

~~~bash
uname -m
cp .env.example .env
chmod 600 .env
install -d -m 700 ./secrets ./output
~~~

平台注册两个 endpoint 后，将返回的一次性注册码分别保存：

~~~bash
printf '%s\n' '<转换一次性注册码>' > ./secrets/converter-enrollment-token
printf '%s\n' '<推理一次性注册码>' > ./secrets/inference-enrollment-token
chmod 600 ./secrets/converter-enrollment-token ./secrets/inference-enrollment-token
~~~

编辑 `.env`：

~~~dotenv
RKNODE_PLATFORM_URL=http://<中央服务器IP>:5173
RKNODE_CONVERTER_ENDPOINT_ID=<转换Endpoint-ID>
RKNODE_CONVERTER_ENROLLMENT_TOKEN_PATH=./secrets/converter-enrollment-token
RKNODE_INFERENCE_ENDPOINT_ID=<推理Endpoint-ID>
RKNODE_INFERENCE_ENROLLMENT_TOKEN_PATH=./secrets/inference-enrollment-token
RKNODE_CONVERTER_HOST_PORT=10081
RKNODE_INFERENCE_HOST_PORT=10082
RKNODE_ENROLLMENT_COMPLETE=false
RKNODE_REQUIRE_NPU_DEVICE=true
~~~

首次部署和验收：

~~~bash
./load-images.sh
./deploy.sh
./verify.sh
docker compose -p rknode-rk3588 --env-file .env \
  -f compose.converter.yaml -f compose.inference.yaml -f compose.enrollment.yaml \
  logs --tail=200 converter inference
~~~

平台“系统设置”应显示两个 endpoint 依次完成 `pending -> claimed -> enrolled + online`。`verify.sh` 还会检查两个容器中的长期 Token 权限是否为 `600`，以及带 Bearer Token 的本地健康接口。

## 10. 切换稳态并清理一次性注册码

只有两个 endpoint 都为 `enrolled + online` 后才能切换稳态。先保留注册码文件，修改：

~~~bash
sed -i 's/^RKNODE_ENROLLMENT_COMPLETE=.*/RKNODE_ENROLLMENT_COMPLETE=true/' .env
sed -i '/^RKNODE_\(NODE\|WORKER\|CONVERTER\|INFERENCE\)_TOKEN=/d' .env
./deploy.sh
./verify.sh
~~~

此时 `deploy.sh`、`verify.sh` 和 `stop.sh` 自动跳过 `compose.enrollment.yaml`。确认容器不再挂载一次性 secret、`/data/state/node-token` 非空且权限为 `600` 后，再删除注册码并重启验收：

~~~bash
unlink ./secrets/converter-enrollment-token
unlink ./secrets/inference-enrollment-token
./stop.sh
./deploy.sh
./verify.sh
~~~

重启后平台仍在线，才说明长期 Token 已写入对应持久卷。不要执行 `docker compose down -v`，否则会删除 Token、模型和任务状态。

## 11. 训练节点的离线交付

标准离线发布不包含训练 tar。需要在隔离网部署训练节点时，先通过内部制品库或受控介质单独交付以下最终镜像：

~~~text
rknode-trainer-torch-cpu:2026.08.20
rknode-trainer-paddle-cpu:2026.08.20
rknode-trainer-torch-cuda12.4:2026.08.20
rknode-trainer-paddle-cuda12.6:2026.08.20
~~~

镜像导入后复用 `deploy/offline/trainer/` 的 Compose 和环境模板，再按[节点部署手册](simple-node-deployment.md)执行统一 enrollment。每个训练节点使用自己的 Endpoint ID、注册码文件、节点名和数据卷，不能复制其他节点的长期 Token。

## 12. VPN、SSH 隧道与防火墙

离线环境仍要保护控制面。推荐使用 VPN，将平台 endpoint 填写为节点 VPN 地址，`RKNODE_PLATFORM_URL` 填写中央平台 VPN 地址；防火墙只允许中央平台访问节点控制端口。

当前板端没有直连路由时，可暂时使用 SSH 隧道：

~~~text
中央 172.29.0.1:11081 -> 板端 127.0.0.1:10081
中央 172.29.0.1:11082 -> 板端 127.0.0.1:10082
~~~

SSH 管理入口 `172.30.82.12:124` 只用于建立管理连接，不能填写为 converter/inference 服务地址。隧道应使用密钥、`BatchMode=yes`、`ServerAliveInterval`，不要在平台或文档保存 SSH 密码。

## 13. 升级与回滚

1. 在构建机校验镜像标签、架构、revision、manifest 和外层 SHA256。
2. 目标机校验 `SHA256SUMS.outer` 与包内 `SHA256SUMS`。
3. 备份中央 `platform-data` 或节点 named volume；保留旧的最终镜像标签。
4. 导入新镜像，修改 `.env` 的完整版本标签，执行 `./deploy.sh` 和 `./verify.sh`。
5. 完成平台 ready、节点 online、长期 Token 重启复用和一条最小真实任务验收。
6. 失败时恢复旧版本标签并重新部署；不得删除卷、endpoint、注册码或长期 Token。

回滚只切换到已经通过历史验收的完整标签，不使用 `<none>` 镜像，不重新构建中间层替代正式版本。

## 14. 离线验收清单

- [ ] 外层归档和包内文件 SHA256 全部通过。
- [ ] 镜像架构与目标主机一致，版本为 `2026.08.20` 或 `2026.08.20-business`。
- [ ] `manifest.json` 中 `requiresNetworkDuringDeploy=false`、`secretsIncluded=false`。
- [ ] `docker compose ... config --quiet` 成功。
- [ ] `load-images.sh`、`deploy.sh` 未执行 pull/build。
- [ ] 中央平台 ready，5173/8554/8081 端口符合防火墙策略。
- [ ] 节点状态完成 `pending -> claimed -> enrolled + online`。
- [ ] 稳态部署不再挂载 enrollment overlay，一次性注册码已安全删除。
- [ ] 节点重启后长期 Token 仍为权限 `600`，平台仍显示 online。
- [ ] RK3588 inference 的设备自检通过，且容器未使用 `privileged`。
- [ ] 至少完成一条与节点 capability 匹配的真实任务。
