# 离线部署指南

离线包不使用 .env。包内的环境变量已经合并到 Compose YAML，运维人员只需修改 YAML 中的地址、Endpoint ID 和占位 secret，然后导入镜像并启动。

节点宿主机 IP / 域名必须能访问中央平台；首次注册由 compose.enrollment.yaml 或对应的拆分 enrollment overlay 挂载一次性注册码。

## 1. 生成和传输离线包

在可联网构建机执行：

~~~bash
python3 scripts/package_offline_bundle.py --help
~~~

按目标架构生成平台、训练、转换或推理包。每个包包含 manifest.json、固定版本镜像 tar、Compose 文件和 load-images.sh、deploy.sh、verify.sh、stop.sh。包不包含 .env、bundle.env、注册码或 SSH 凭证。

传输归档和校验文件到目标机，解包后先检查：

~~~bash
sha256sum -c SHA256SUMS
./load-images.sh
~~~

load-images.sh 会校验 OCI 架构、版本标签和 io.rknode.offline-ready=true，失败时不会启动服务。

## 2. 平台离线包

编辑 compose.yaml：

- 将 CENTRAL_SERVER_IP 或 replace-with-* 改为目标机实际值。
- admin-token 是前端登录令牌；无域名或证书时使用 http://<IP>:5173。
- worker-token、zlm-api-secret、zlm-hook-identity 必须直接写入 YAML，禁止提交真实值。

生成 Media secret 并启动：

~~~bash
./configure-media-secrets.py --compose-file compose.yaml
docker compose -f compose.yaml config --quiet
./deploy.sh
./verify.sh
~~~

## 3. 离线训练包

编辑 compose.yaml 中的 RKNODE_ENDPOINT_ID、RKNODE_PLATFORM_URL 和节点身份。平台地址示例为 http://172.16.66.249:5173，训练服务地址示例为 172.16.66.249:10081。

首次注册时写入固定 secret：

~~~bash
mkdir -p secrets
umask 077
printf '%s' '<训练注册码>' > secrets/trainer-enrollment-token
chmod 600 secrets/trainer-enrollment-token
./deploy.sh --enroll
./verify.sh --enroll
~~~

状态从 pending 到 claimed 再到 enrolled 后，删除或移走注册码文件，后续执行 ./deploy.sh 和 ./verify.sh。

Compose 中 RKNODE_ENROLLMENT_TOKEN_FILE 指向首次注册 secret，RKNODE_NODE_TOKEN_FILE 指向数据卷内注册成功后生成的节点 Token。

## 4. 离线 RK3588 包

转换和推理可以拆包，也可以使用统一包。编辑对应 Compose 的 Endpoint ID 和中央平台 URL。节点宿主机示例为 172.30.82.12：

- 转换服务地址为 172.30.82.12:10081。
- 推理服务地址为 172.30.82.12:10082。
- 临时 SSH 隧道地址 172.29.0.1:11081、172.29.0.1:11082 只用于应急，不能作为长期注册地址。

统一包首次注册：

~~~bash
mkdir -p secrets output
umask 077
printf '%s' '<转换注册码>' > secrets/converter-enrollment-token
printf '%s' '<推理注册码>' > secrets/inference-enrollment-token
chmod 600 secrets/*-enrollment-token
./deploy.sh --enroll
./verify.sh --enroll
~~~

拆分包使用各自的 compose.enrollment.converter.yaml 或 compose.enrollment.inference.yaml。两个 Endpoint 都 enrolled + online 后，执行 ./deploy.sh，再用 ./verify.sh 检查服务、节点 Token 权限、NPU 设备和中央平台连接。

## 5. 固定 Compose 操作

脚本从 manifest.json 读取 Compose 文件和项目名，不读取 shell 环境。人工排查也必须显式指定文件：

~~~bash
docker compose -f compose.yaml config --quiet
docker compose -f compose.yaml ps
docker compose -f compose.yaml logs --tail=200
docker compose -f compose.yaml down
~~~

离线部署禁止 pull 和 build；镜像必须已经由 load-images.sh 导入。不要用 docker compose --env-file，也不要手动创建 bundle.env。

## 6. 网络和安全

离线节点仍需通过内网或 VPN 访问中央平台。服务端口不得暴露到公网；没有域名或证书时使用受控 IP:端口，跨网场景使用 HTTPS 反向代理。SSH 隧道只作为临时排障，必须使用密钥和 BatchMode=yes，不得保存 SSH 密码。secret 文件权限为 0600，归档中不得包含注册码。

## 旧版静态 Token 迁移

旧离线包若包含静态 Token，先在中央平台生成一次性注册码，将它写入新的 Compose secret 文件并用 enrollment overlay 启动。确认 enrolled 后停止 overlay，清空 Compose 中旧 Token，保留节点数据卷并重新运行 ./deploy.sh。
