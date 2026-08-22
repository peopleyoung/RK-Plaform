# RKNode 终版镜像收口与旧版本清理设计

## 目标

以本机 Docker 镜像元数据、运行中容器引用和离线包清单为准，固定当前项目的七个终版镜像；让在线部署、离线部署和首次运维说明统一使用显式终版标签；删除已确认属于本项目的旧镜像标签及旧离线包，同时不影响其他项目镜像、运行数据卷和业务缓存。

## 终版镜像矩阵

| 角色 | 终版镜像 | 镜像 ID | 架构 |
| --- | --- | --- | --- |
| 平台 API | `rknode-platform-api:2026.08.13` | `sha256:76539c54f88a5a86a4e242ee82b1e5c39fc47d913df521e36fe62e46948fae75` | amd64 |
| 平台 Web | `rknode-platform-web:2026.08.13` | `sha256:4f3cf18d5d322adf9bc366f6468f018c055df958e95ab2cf533aba996bafe3d9` | amd64 |
| Torch CPU 训练 | `rknode-trainer-torch-cpu:2026.08.13` | `sha256:b5173085fe5b523d54341f3561ca68ae9feb33990324822687fdb627cead61a3` | amd64 |
| Paddle CPU 训练 | `rknode-trainer-paddle-cpu:2026.08.13` | `sha256:a94c0ebcdbcf4f14a8c7ee703b0581081605e7798a1ff4e9ee937083b641a4da` | amd64 |
| Torch CUDA 训练 | `rknode-trainer-torch-cuda12.4:2026.08.13` | `sha256:0d894eb5daf0cbe15de0b75bcf4420318a70d246937ed2490a5532e47f02299e` | amd64 |
| Paddle CUDA 训练 | `rknode-trainer-paddle-cuda12.6:2026.08.13` | `sha256:c884f97e325cbefa80a381d43d87379384ecd0091f4d3ad6eda1f4913806901b` | amd64 |
| RK3588 转换/推理 | `rknode-rk3588-node:2026.08.13-business` | `sha256:c8c90ad4065eedb26b343ba1533ddf7d19f3569a0b1d13fde16fa87584f5585b` | arm64 |

镜像标签必须完整书写，不使用 `latest`、`local` 或缺省标签。RK3588 的 `2026.08.13-business` 是在 `2026.08.13` 基础上补齐区域、越线、ByteTrack、二级 YOLO、事件媒体、Kafka 和 ZLM SEI 的后续终版，旧标签不得继续出现在部署入口和操作说明中。

## 保留与删除边界

### 保留

- 上表七个终版镜像及其共享层。
- 六个终版离线包：平台包包含两个镜像，四个训练包各包含一个镜像，RK3588 business 包包含一个镜像。
- 当前平台数据卷、训练数据卷、容器日志和任务制品。
- 属于 `kbqa` 等其他 Compose 项目的无标签镜像。
- 终版镜像仍引用的 Docker 共享层。

### 删除

- Docker 标签和镜像 `rknode-rk3588-node:2026.08.13`。
- 兼容别名 `rknode-trainer:local`；它与 Torch CPU 终版镜像 ID 相同，只移除别名，不删除终版镜像层。
- `release/offline/` 下六个 `2026.08.12` 旧离线包。
- `release/offline/rknode-rk3588-node-arm64-2026.08.13.tar`。
- `release/offline/SHA256SUMS` 中旧 RK3588 包记录，并增加 business 终版包记录。

不执行 `docker system prune`、`docker image prune` 或全局 BuildKit 清理，因为当前 Docker 引擎由多个项目共享，无法可靠把全部无标签镜像和缓存归属到 RKNode。

## 配置和文档改动

在线节点 Compose 默认值、离线 Compose 默认值、实际节点环境文件及所有 `.env.example` 使用显式终版镜像标签。平台与训练示例使用 `2026.08.13`，RK3588 转换和推理统一使用 `2026.08.13-business`。

以下面向运维人员的说明文档建立同一终版矩阵并消除旧标签：

- `docs/system-guide.md`
- `docs/simple-node-deployment.md`
- `docs/offline-deployment.md`
- 根目录 `README.md` 中的部署入口说明（若缺少终版矩阵入口则补充链接）

构建脚本继续允许显式传入版本，不强行把所有角色改成同一个新版本号。平台/训练镜像和 RK3588 business 镜像来自两次已完成构建，保留真实标签比重新打一个无构建依据的统一标签更可审计。

## 运行中容器迁移

当前平台 API/Web 已直接引用终版标签，不重建。当前 CPU 训练容器引用 `rknode-trainer:local`，但实际镜像 ID 与 Torch CPU 终版完全相同。迁移步骤为：

1. 在 `deploy/nodes/trainer/.env` 写入 `RKNODE_TRAINER_IMAGE=rknode-trainer-torch-cpu:2026.08.13`。
2. 用原 Compose project 和环境文件执行 `up -d --no-build --force-recreate`。
3. 确认容器镜像引用、容器状态、健康接口和平台节点状态正常。
4. 仅在确认后删除 `rknode-trainer:local` 标签。

迁移不改变镜像 ID、节点名称、Token、端口或数据卷，因此回滚只需恢复同 ID 的别名或继续使用终版标签，不涉及数据恢复。

## 验证和错误处理

删除前验证终版镜像的 ID、架构和 OCI 版本标签，并校验每个终版离线包的 SHA256。任何终版包校验失败时停止删除对应旧包。

修改后执行：

- 搜索使用文档和部署配置，确保不存在 `2026.08.12`、旧 RK3588 `2026.08.13`、`rknode-trainer:local`、其他 RKNode `:local` 或 `:latest`。
- 展开平台、CPU/CUDA 训练、RK3588 在线/离线 Compose，确认最终 `image:` 字段与矩阵一致且日志/设备契约未变化。
- 检查运行中平台和 CPU 训练容器的镜像 ID、状态及健康响应。
- 再次列出 `rknode*` Docker 镜像和 `release/offline/` 文件，只保留终版矩阵与终版离线包。
- 运行文档命令静态审计，确保没有引入全局 prune、卷删除或硬编码真实 Token。

删除操作使用精确标签和精确文件路径，不使用通配符、目录递归删除或全局清理命令。

## 回滚

文档和配置在删除前完成并验证。CPU 训练容器迁移失败时保留原容器和 `local` 别名，不继续删除。旧 Docker 镜像删除后可从旧离线包恢复，但旧离线包本轮也会删除，因此删除前以终版包全量校验通过作为不可跳过的门禁。业务数据卷从始至终不在删除范围内。
