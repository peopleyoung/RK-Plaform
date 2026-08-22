# 节点部署与使用手册

本文只讲节点：CPU/CUDA 训练、RK3588 转换和 RK3588 推理。中央平台部署见 [完整使用与部署手册](system-guide.md)，无网络环境见 [离线部署手册](offline-deployment.md)。

## 1. 统一接入模型

三类节点使用同一个 direct enrollment 流程：

~~~text
平台系统设置登记 endpoint
  -> 平台生成一次性注册码
  -> 运维将注册码写入节点宿主机 0600 secret
  -> enrollment Compose 启动
  -> 节点 POST /node-enrollments/{endpoint}/claim
  -> 平台首次探测 /health
  -> enrolled + online
  -> 去掉 enrollment overlay，长期 Token 留在 /data/state/node-token
~~~

地址方向：

| 配置项 | 含义 | 示例 |
| --- | --- | --- |
| 平台 endpoint 地址 | 中央平台访问节点服务 | http://172.16.66.249:10081 |
| RKNODE_PLATFORM_URL | 节点访问中央平台 Web/API 根地址 | http://172.16.66.249:5173 |

RKNODE_PLATFORM_URL 不附加 /api/v1。节点服务地址不是中央平台地址。

统一变量名称为 `RKNODE_ENDPOINT_ID`、`RKNODE_PLATFORM_URL`、`RKNODE_ENROLLMENT_TOKEN_FILE` 和 `RKNODE_NODE_TOKEN_FILE`；实际 Compose 模板中的宿主机注册码路径变量会映射到容器内的 `RKNODE_ENROLLMENT_TOKEN_FILE`。平台登记字段名称是“节点宿主机 IP / 域名”。状态依次为 `pending -> claimed -> enrolled`。当前推理直连验收地址为 `172.30.82.12:10082`，转换直连验收地址为 `172.30.82.12:10081`。

## 2. 当前节点矩阵

| 节点 | 镜像 | 宿主机发布端口 | 加速器 | 当前平台注册地址 |
| --- | --- | --- | --- | --- |
| Torch CPU 训练 | rknode-trainer-torch-cpu:2026.08.20 | 10081 | cpu | 172.16.66.249:10081 |
| Paddle CPU 训练 | rknode-trainer-paddle-cpu:2026.08.20 | 10081 | cpu | 训练主机 IP:10081 |
| Torch CUDA 训练 | rknode-trainer-torch-cuda12.4:2026.08.20 | 10081 | cuda | CUDA 主机 IP:10081 |
| Paddle CUDA 训练 | rknode-trainer-paddle-cuda12.6:2026.08.20 | 10081 | cuda | CUDA 主机 IP:10081 |
| RK3588 转换 | rknode-rk3588-node:2026.08.20-business | 10081 | rk3588 | 当前 172.29.0.1:11081 |
| RK3588 推理 | rknode-rk3588-node:2026.08.20-business | 10082 | rk3588 | 当前 172.29.0.1:11082 |

当前 RK3588 平台地址是中央主机 SSH 隧道入口：

~~~text
172.29.0.1:11081 -> 板端 127.0.0.1:10081（converter）
172.29.0.1:11082 -> 板端 127.0.0.1:10082（inference）
~~~

板端 172.30.82.12:124 是 SSH 管理入口。修通中央到板端 10081/10082 的直连路由、完成转换和推理真实验收后，才可将平台地址改为 172.30.82.12:10081/10082。

## 3. 平台登记节点

1. 打开 http://<中央服务器IP>:5173，输入管理员令牌。
2. 进入“系统设置” -> “新增节点”。
3. 选择节点类型和“直连调度”。推理节点固定为 direct。
4. 填写节点宿主机地址和发布端口，不要填中央平台地址。
5. 填写与容器完全一致的名称、加速器和能力。
6. 保存，记录 Endpoint ID；注册码只在当前窗口显示，立即下载或复制。

能力填写规则：

| 节点 | capabilities |
| --- | --- |
| Torch 训练 | yolo-detect,deeplabv3plus |
| Paddle 训练 | ppocr-det,ppocr-rec |
| RK3588 转换 | yolo-detect,deeplabv3plus,ppocr-det,ppocr-rec |
| RK3588 推理 | yolo_dfl_split_v1,deeplab_logits_v1,ppocr_db_det_v1,ppocr_ctc_rec_v1 |

推理节点另外声明 RKNODE_MEDIA_FEATURES：

~~~text
rkmpp_decode,bytetrack,kafka,zlm_sei,analytics_area,analytics_line,event_snapshot,event_record,secondary_infer
~~~

Endpoint ID 不是秘密；一次性注册码和长期节点 Token 是秘密。平台只返回一次性注册码，不返回已保存的长期 Token。

## 4. 一次性注册码下发

在节点部署目录执行：

~~~bash
install -d -m 700 ./secrets
printf '%s\n' '<注册码>' > ./secrets/<角色>-enrollment-token
chmod 600 ./secrets/<角色>-enrollment-token
~~~

文件映射：

| 角色 | 环境变量 | 文件示例 |
| --- | --- | --- |
| 训练 | RKNODE_ENROLLMENT_TOKEN_PATH | ./secrets/trainer-enrollment-token |
| 转换 | RKNODE_CONVERTER_ENROLLMENT_TOKEN_PATH | ./secrets/converter-enrollment-token |
| 推理 | RKNODE_INFERENCE_ENROLLMENT_TOKEN_PATH | ./secrets/inference-enrollment-token |

不要把注册码写入 RKNODE_NODE_TOKEN、命令行历史、镜像、日志或聊天记录。

## 5. CPU Torch 训练节点

### 5.1 配置

当前仓库的在线节点配置目录由运维创建：

~~~bash
mkdir -p deploy/nodes/trainer/secrets
cp deploy/offline/trainer/torch-cpu.env.example deploy/nodes/trainer/.env
chmod 600 deploy/nodes/trainer/.env
~~~

至少修改：

~~~dotenv
RKNODE_TRAINER_IMAGE=rknode-trainer-torch-cpu:2026.08.20
RKNODE_PLATFORM_URL=http://<中央服务器IP>:5173
RKNODE_ENDPOINT_ID=<平台生成的Endpoint-ID>
RKNODE_ENROLLMENT_TOKEN_PATH=./secrets/trainer-enrollment-token
RKNODE_NODE_NAME=cpu-torch-trainer-01
RKNODE_NODE_HOST_PORT=10081
RKNODE_NODE_ACCELERATOR=cpu
RKNODE_NODE_CAPABILITIES=yolo-detect,deeplabv3plus
RKNODE_NODE_MAX_CONCURRENCY=1
~~~

首次启动：

~~~bash
docker compose -p rknode-trainer \
  --env-file deploy/nodes/trainer/.env \
  -f deploy/nodes/trainer/compose.yaml \
  -f deploy/nodes/trainer/compose.enrollment.yaml \
  config --quiet

docker compose -p rknode-trainer \
  --env-file deploy/nodes/trainer/.env \
  -f deploy/nodes/trainer/compose.yaml \
  -f deploy/nodes/trainer/compose.enrollment.yaml \
  up -d --no-build
~~~

### 5.2 Paddle CPU

复用同一个 compose.yaml，把配置改为：

~~~dotenv
RKNODE_TRAINER_IMAGE=rknode-trainer-paddle-cpu:2026.08.20
RKNODE_TRAINER_DOCKERFILE=deploy/Dockerfile.trainer-paddle
RKNODE_NODE_NAME=cpu-paddle-trainer-01
RKNODE_NODE_ACCELERATOR=cpu
RKNODE_NODE_CAPABILITIES=ppocr-det,ppocr-rec
~~~

每个节点必须使用自己的平台注册 endpoint、注册码文件和数据卷，不要复制已有节点的长期 Token。

## 6. CUDA 训练节点

### 6.1 主机检查

~~~bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
~~~

CUDA 12.4 Torch 使用 rknode-trainer-torch-cuda12.4:2026.08.20；CUDA 12.6 Paddle 使用 rknode-trainer-paddle-cuda12.6:2026.08.20。镜像 CUDA、NVIDIA 驱动和 Container Toolkit 必须兼容。

### 6.2 配置和启动

~~~dotenv
RKNODE_TRAINER_IMAGE=rknode-trainer-torch-cuda12.4:2026.08.20
RKNODE_PLATFORM_URL=http://<中央服务器IP>:5173
RKNODE_ENDPOINT_ID=<CUDA训练Endpoint-ID>
RKNODE_ENROLLMENT_TOKEN_PATH=./secrets/trainer-enrollment-token
RKNODE_NODE_NAME=cuda-torch-trainer-01
RKNODE_NODE_ACCELERATOR=cuda
RKNODE_NODE_CAPABILITIES=yolo-detect,deeplabv3plus
RKNODE_CUDA_DEVICE_COUNT=1
~~~

~~~bash
docker compose -p rknode-trainer-cuda \
  --env-file deploy/nodes/trainer/.env \
  -f deploy/nodes/trainer/compose.yaml \
  -f deploy/nodes/trainer/compose.cuda.yaml \
  -f deploy/nodes/trainer/compose.enrollment.yaml \
  up -d --no-build
~~~

Paddle CUDA 只需替换镜像、Dockerfile、节点名称和 capabilities：

~~~dotenv
RKNODE_TRAINER_IMAGE=rknode-trainer-paddle-cuda12.6:2026.08.20
RKNODE_TRAINER_DOCKERFILE=deploy/Dockerfile.trainer-paddle
RKNODE_NODE_ACCELERATOR=cuda
RKNODE_NODE_CAPABILITIES=ppocr-det,ppocr-rec
~~~

## 7. RK3588 转换与推理节点

### 7.1 板端检查

~~~bash
uname -m
ls -l /dev/dri/card0 /dev/dri/renderD128 /dev/mpp_service /dev/rga
ls -ld /dev/dma_heap /sys/firmware/devicetree/base
docker image inspect rknode-rk3588-node:2026.08.20-business
~~~

期望架构为 arm64。推理 Compose 映射 NPU/MPP/RGA/dma-heap，并挂载 /sys/firmware/devicetree/base；不使用 privileged。

### 7.2 配置文件

~~~bash
mkdir -p deploy/nodes/rk3588/secrets
cp deploy/offline/rk3588/node.env.example deploy/nodes/rk3588/.env
chmod 600 deploy/nodes/rk3588/.env
~~~

~~~dotenv
RKNODE_RK3588_IMAGE=rknode-rk3588-node:2026.08.20-business
RKNODE_PLATFORM_URL=http://<中央服务器IP>:5173

RKNODE_CONVERTER_NAME=rk3588-converter-01
RKNODE_CONVERTER_ENDPOINT_ID=<转换Endpoint-ID>
RKNODE_CONVERTER_ENROLLMENT_TOKEN_PATH=./secrets/converter-enrollment-token
RKNODE_CONVERTER_HOST_PORT=10081
RKNODE_CONVERTER_CAPABILITIES=yolo-detect,deeplabv3plus,ppocr-det,ppocr-rec

RKNODE_INFERENCE_NAME=rk3588-inference-01
RKNODE_INFERENCE_ENDPOINT_ID=<推理Endpoint-ID>
RKNODE_INFERENCE_ENROLLMENT_TOKEN_PATH=./secrets/inference-enrollment-token
RKNODE_INFERENCE_HOST_PORT=10082
RKNODE_INFERENCE_ADAPTERS=yolo_dfl_split_v1,deeplab_logits_v1,ppocr_db_det_v1,ppocr_ctc_rec_v1
RKNODE_MEDIA_FEATURES=rkmpp_decode,bytetrack,kafka,zlm_sei,analytics_area,analytics_line,event_snapshot,event_record,secondary_infer
RKNODE_MAX_MODEL_INSTANCES=4
RKNODE_RUNTIME_VERSION=rknn-runtime-2.3.2
RKNODE_PIPELINE_VERSION=rknode-cpp-runtime-2026.08.20-business
RKNODE_REQUIRE_NPU_DEVICE=true
~~~

### 7.3 首次启动

~~~bash
docker compose -p rknode-rk3588 \
  --env-file deploy/nodes/rk3588/.env \
  -f deploy/nodes/rk3588/compose.yaml \
  -f deploy/nodes/rk3588/compose.enrollment.yaml \
  config --quiet

docker compose -p rknode-rk3588 \
  --env-file deploy/nodes/rk3588/.env \
  -f deploy/nodes/rk3588/compose.yaml \
  -f deploy/nodes/rk3588/compose.enrollment.yaml \
  up -d --no-build
~~~

两个角色分别领取注册码，任何一个失败都可以独立排障。查看日志：

~~~bash
docker compose -p rknode-rk3588 --env-file deploy/nodes/rk3588/.env \
  -f deploy/nodes/rk3588/compose.yaml \
  -f deploy/nodes/rk3588/compose.enrollment.yaml logs --tail=200 converter inference
~~~

### 7.4 切换稳态和验收

平台确认两个 endpoint 都为 enrolled + online 后：

~~~bash
docker compose -p rknode-rk3588 \
  --env-file deploy/nodes/rk3588/.env \
  -f deploy/nodes/rk3588/compose.yaml \
  up -d --no-build --force-recreate

docker compose -p rknode-rk3588 --env-file deploy/nodes/rk3588/.env \
  -f deploy/nodes/rk3588/compose.yaml exec -T converter stat -c '%a:%s' /data/state/node-token
docker compose -p rknode-rk3588 --env-file deploy/nodes/rk3588/.env \
  -f deploy/nodes/rk3588/compose.yaml exec -T inference stat -c '%a:%s' /data/state/node-token
~~~

两个结果都应为 600:<非零大小>。删除一次性注册码后再次重启，平台仍应在线。

### 7.5 RK3588 设备验收

~~~bash
docker compose -p rknode-rk3588 --env-file deploy/nodes/rk3588/.env \
  -f deploy/nodes/rk3588/compose.yaml exec -T inference \
  /opt/rknode/runtime-adapter/self-test.sh
~~~

如果直接访问 /health 得到 401，说明端口已可达但缺少 Bearer Token；不要把 401 当成服务未启动。

## 8. 注册状态与平台探测

| 状态 | 解释 | 动作 |
| --- | --- | --- |
| pending | endpoint 已建，一次性注册码尚未领取 | 检查 overlay、Endpoint ID、注册码 |
| claimed | 节点已领取长期 Token，但首次探测失败 | 检查平台地址、端口和节点身份 |
| enrolled + online | 身份和健康探测通过 | 可以调度 |
| enrolled + offline/error | 已注册但当前不可用 | 检查服务、设备、网络和隧道 |

平台主动探测节点时使用节点长期 Token。普通无 Token 的 curl /health 返回 401 是预期鉴权行为。

## 9. 容器、卷和命令矩阵

| 角色 | Compose 服务 | 容器端口 | 宿主机默认端口 | 数据卷 |
| --- | --- | --- | --- | --- |
| Trainer | trainer | 10081 | 10081 | trainer-data |
| Converter | converter | 10081 | 10081 | converter-data |
| Inference | inference | 10081 | 10082 | inference-data + output |

稳态命令：

~~~bash
docker compose -p <项目名> --env-file .env -f compose.yaml ps
docker compose -p <项目名> --env-file .env -f compose.yaml logs -f --tail=200 <服务>
docker compose -p <项目名> --env-file .env -f compose.yaml restart <服务>
docker compose -p <项目名> --env-file .env -f compose.yaml down
~~~

down 不删除卷。禁止 down -v，除非已备份并明确要删除长期 Token、模型和任务缓存。

## 10. 直连、VPN 与 SSH 隧道

直连时平台填写节点宿主机的 VPN/LAN IP 和发布端口；节点 RKNODE_PLATFORM_URL 填中央平台通过同一网络可达的地址。从中央平台主机验证：

~~~bash
curl --connect-timeout 3 -i http://<节点宿主机IP>:10081/health
curl --connect-timeout 3 -i http://<节点宿主机IP>:10082/health
~~~

返回 401/200 说明端口可达。VPN 是标准跨网段方案；防火墙只允许中央平台 VPN 地址访问节点控制端口。

SSH 隧道只用于当前环境或紧急回滚，不把密码写入平台。模板位于 deploy/systemd/rknode-node-tunnel.service.example，使用密钥、BatchMode=yes 和 ServerAliveInterval：

~~~text
中央 172.29.0.1:11081 -> 板端 127.0.0.1:10081
中央 172.29.0.1:11082 -> 板端 127.0.0.1:10082
~~~

## 11. 旧静态 Token 节点迁移

旧节点可以在升级窗口暂时保留 RKNODE_NODE_TOKEN，但新节点不要使用它。迁移时：

1. 确认节点无活动任务。
2. 在系统设置对 legacy endpoint 执行“迁移统一接入/重新签发注册码”。
3. 将注册码写入同一节点 secret 文件。
4. 使用 enrollment overlay 启动，领取原有长期 Token。
5. 观察 enrolled + online。
6. 清空静态 Token 环境变量，使用基础 Compose 重建。
7. 确认不再挂载注册码后删除文件。

已经成功 enrollment 的 endpoint 不重复执行迁移。

## 旧版静态 Token 迁移

旧节点只在维护窗口保留静态 Token；迁移完成后清空静态变量、确认 enrollment overlay 已卸载，并使用 `/data/state/node-token` 中的长期 Token。

安全边界：节点控制端口不得暴露到公网；跨网段优先使用 VPN，必要时使用 HTTPS 或 SSH 隧道。平台和文档不得保存 SSH 密码。

## 12. 故障处理

| 故障 | 检查顺序 |
| --- | --- |
| pending | .env Endpoint ID、secret、注册码有效期、overlay 是否挂载 |
| claimed/offline | 平台 endpoint 地址、端口、防火墙、节点 /health、身份能力 |
| CUDA offline | nvidia-smi、Toolkit、GPU 映射、CUDA 版本 |
| converter offline | rknn.api、/dev/dri、/dev/dma_heap、平台路由 |
| inference offline | MPP/RGA、设备树挂载、runtime self-test、SSH 隧道 |
| Token 重启丢失 | 卷是否更换、Token 是否存在且权限 600 |
| 任务 queued | 节点 online、capability 匹配、context 容量 |

不要通过重复签发注册码掩盖卷丢失、地址错误或设备缺失；先保留日志和平台事件。

## 13. 节点验收清单

- [ ] 镜像版本和架构正确。
- [ ] Endpoint 类型、名称、加速器、能力与 .env 完全一致。
- [ ] 状态经历 pending -> claimed -> enrolled + online。
- [ ] 稳态 Compose 不再包含 compose.enrollment.yaml。
- [ ] 一次性 secret 已删除。
- [ ] /data/state/node-token 权限为 600。
- [ ] CPU/CUDA 框架和设备检查通过。
- [ ] RK3588 inference 无特权运行且 runtime self-test 通过。
- [ ] 平台可探测 /health 并显示 online。
- [ ] 至少完成一条匹配能力的真实任务。
