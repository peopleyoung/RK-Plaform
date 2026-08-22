# RKNode Deployment Operations Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `docs/system-guide.md` into an executable first-deployment and operations manual covering the complete online, prebuilt-image, offline, CPU/CUDA, Torch/Paddle, RK3588, VPN, and SSH-tunnel matrix.

**Architecture:** Keep one authoritative linear deployment runbook, followed by optional network/offline branches and operational reference sections. Derive commands from current Compose files, Dockerfiles, environment examples, scripts, and service health contracts; use placeholders for all infrastructure-specific values and explicitly separate destructive cleanup from routine upgrades.

**Tech Stack:** Markdown, Docker Engine, Docker Compose v2, Bash, NVIDIA Container Toolkit, RK3588 RKNN Toolkit2/Runtime, WireGuard, OpenSSH.

---

## File Map

- Create: `docs/superpowers/plans/2026-08-14-deployment-operations-guide.md` — execution checklist and verification record.
- Modify: `docs/system-guide.md` — sole complete deployment and operations manual.
- Reference only: `deploy/compose.yaml`, `deploy/.env.example` — central deployment contract.
- Reference only: `deploy/nodes/trainer/compose.yaml`, `deploy/nodes/trainer/compose.cuda.yaml`, `deploy/nodes/trainer/.env.example` — online trainer contract.
- Reference only: `deploy/nodes/rk3588/compose.yaml`, `deploy/nodes/rk3588/.env.example` — online RK3588 contract.
- Reference only: `deploy/offline/`, `scripts/build_offline_images.sh`, `scripts/package_offline_bundle.py` — offline bundle contract.
- Reference only: `backend/platform_api/routes.py`, `workers/node_service/app.py`, `deploy/rk3588/runtime-adapter/` — health and runtime validation contract.

### Task 1: Build The Authoritative Command Matrix

**Files:**
- Reference: `deploy/compose.yaml`
- Reference: `deploy/nodes/trainer/compose.yaml`
- Reference: `deploy/nodes/trainer/compose.cuda.yaml`
- Reference: `deploy/nodes/rk3588/compose.yaml`
- Reference: `deploy/offline/**/*.yaml`
- Reference: `deploy/offline/**/*.sh`

- [ ] **Step 1: Extract Compose services, images, ports, volumes, devices, and required variables**

Run:

```bash
docker compose --env-file deploy/.env.example -f deploy/compose.yaml config
docker compose --env-file deploy/nodes/trainer/.env.example -f deploy/nodes/trainer/compose.yaml config
docker compose --env-file deploy/nodes/rk3588/.env.example -f deploy/nodes/rk3588/compose.yaml config
```

Expected: all three configurations render without missing-variable or YAML errors.

- [ ] **Step 2: Extract supported offline bundle names and script arguments**

Run:

```bash
sed -n '1,240p' scripts/build_offline_images.sh
sed -n '1,280p' scripts/package_offline_bundle.py
sed -n '1,240p' deploy/offline/common/deploy.sh
sed -n '1,180p' deploy/offline/common/verify.sh
```

Expected: command syntax is available for platform, Torch/Paddle CPU/CUDA, RK3588 converter, RK3588 inference, and combined RK3588 bundles.

- [ ] **Step 3: Verify health paths and authentication requirements**

Run:

```bash
rg -n 'api/v1/ready|@app.get\("/health"|Authorization: Bearer|self-test.sh' backend workers deploy
```

Expected: central readiness uses `/api/v1/ready`; direct-node health uses authenticated `/health`; RK3588 exposes the runtime self-test script.

### Task 2: Rewrite The Foundation And Central Deployment Chapters

**Files:**
- Modify: `docs/system-guide.md`

- [ ] **Step 1: Replace the legacy introduction with the new runbook scope and architecture**

Write chapters covering audience, supported matrix, central/trainer/converter/inference responsibilities, control/data flow, host role labels, port table, and placeholder conventions.

- [ ] **Step 2: Add preflight and secret preparation commands**

Include exact checks for `uname -m`, `/etc/os-release`, Docker, Compose v2, disk, memory, ports, time synchronization, NVIDIA runtime, and RK3588 device nodes. Generate tokens with `openssl rand -hex 32`, copy `.env.example`, and apply `chmod 600`.

- [ ] **Step 3: Add scoped old-resource cleanup**

Include commands to list matching containers/images/build cache, stop the named Compose project, remove only explicit obsolete image tags, prune dangling images, and prune build cache. Put volume removal and broad `docker system prune` operations in a separate danger block.

- [ ] **Step 4: Add central configuration, build, startup, and validation**

Use:

```bash
cp deploy/.env.example deploy/.env
chmod 600 deploy/.env
docker compose --env-file deploy/.env -f deploy/compose.yaml config --quiet
docker compose --env-file deploy/.env -f deploy/compose.yaml build --pull
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d
curl -fsS http://127.0.0.1:8080/api/v1/ready
```

Expected: Compose configuration succeeds, API becomes healthy, and the Web entry is reachable on the configured port.

### Task 3: Rewrite The Complete Trainer Matrix

**Files:**
- Modify: `docs/system-guide.md`

- [ ] **Step 1: Document Torch CPU deployment**

Specify the Torch Dockerfile, CPU base image, CPU wheel index, explicit image tag, Compose project name, build/start commands, authenticated health check, node addition in System Settings, and a minimal training acceptance task.

- [ ] **Step 2: Document Torch CUDA deployment**

Specify driver and NVIDIA Container Toolkit checks, a compatible CUDA base image and Torch versions, `compose.cuda.yaml` overlay, `RKNODE_NODE_ACCELERATOR=cuda`, container `nvidia-smi`, and CUDA task acceptance.

- [ ] **Step 3: Document Paddle CPU deployment**

Specify `deploy/Dockerfile.trainer-paddle`, `paddlepaddle==3.2.2`, an explicit image tag, Paddle capabilities, Compose commands, import check, and minimal task acceptance.

- [ ] **Step 4: Document Paddle CUDA deployment**

Specify the Paddle GPU package, compatible CUDA/cuDNN base image, CUDA overlay, GPU import/device check, explicit image tag, and minimal task acceptance.

- [ ] **Step 5: Add trainer troubleshooting checks**

Include version mismatch, wheel/source download failure, node token mismatch, central URL with an accidental `/api/v1` suffix, occupied host port, insufficient shared memory, and GPU runtime failures.

### Task 4: Rewrite RK3588 Conversion And Inference Deployment

**Files:**
- Modify: `docs/system-guide.md`

- [ ] **Step 1: Add RK3588 host preflight**

Check `aarch64`, `/dev/dri`, `/dev/dma_heap`, `/dev/mpp_service`, `/dev/rga`, RKNPU driver information, Docker device access, storage, and the required Toolkit2 base image architecture.

- [ ] **Step 2: Add source-build and prebuilt-image flows**

Use `deploy/nodes/rk3588/.env.example`, an explicit `RKNODE_RK3588_IMAGE` tag, `deploy/nodes/rk3588/compose.yaml`, and separate commands for `--build` versus reusing an already loaded image without `--build`.

- [ ] **Step 3: Add converter and inference service validation**

Validate both authenticated `/health` endpoints, run `/opt/rknode/runtime-adapter/self-test.sh` in the inference container, inspect device mappings and logs, and confirm the output directory is writable.

- [ ] **Step 4: Add platform node configuration**

Document separate converter/inference names, ports, URLs, tokens, capabilities/adapters, probe success, enabled status, and the rule that every direct node token differs from admin/global worker/other node tokens.

- [ ] **Step 5: Add end-to-end RKNN acceptance**

Document training artifact selection, RKNN conversion, validation result, deployment-ready state, model release, node-group deployment, image/video/RTSP inference, preview/output inspection, and target-board-only validation markers.

### Task 5: Add Network And Offline Deployment Branches

**Files:**
- Modify: `docs/system-guide.md`

- [ ] **Step 1: Document direct-routing checks**

Provide node-to-central and central-to-node `curl`, `ss`, and firewall checks for ports `8080`, `10081`, and `10082`.

- [ ] **Step 2: Document WireGuard deployment**

Provide package installation, key generation without printing private keys, sample central/node configurations, forwarding/firewall requirements, `wg-quick` enablement, keepalive, and connectivity validation.

- [ ] **Step 3: Document persistent SSH tunnels**

Provide restricted key generation, authorized-key restrictions, `ExitOnForwardFailure`, `ServerAliveInterval`, `ServerAliveCountMax`, systemd service persistence, central/node port mapping, and reconnection validation.

- [ ] **Step 4: Document offline image construction and packaging**

Use the actual `scripts/build_offline_images.sh` and `scripts/package_offline_bundle.py` interfaces, explicit release tags, per-architecture build hosts, archive checksum verification, and the bundle matrix.

- [ ] **Step 5: Document offline load, deployment, and verification**

Use `deploy/offline/common/load-images.sh`, `deploy.sh`, `verify.sh`, and `stop.sh`; explain `.env` selection for all trainer variants and both RK3588 services, and prohibit `--build` on the offline target.

### Task 6: Add Operations, Recovery, And Reference Chapters

**Files:**
- Modify: `docs/system-guide.md`

- [ ] **Step 1: Add routine status, logs, restart, and disk inspection**

Provide service-specific Compose commands, `docker stats`, `docker system df`, volume inspection, log-tail, and bounded log rotation guidance.

- [ ] **Step 2: Add backup and restore procedures**

Back up central `platform-data`, node data volumes, RK3588 bind-mounted output, and encrypted `.env` files. Require stopping writers or using a consistent snapshot and include a restore verification sequence.

- [ ] **Step 3: Add image upgrade and rollback procedures**

Record the current image IDs/tags, prepare the new explicit tag, recreate only target services, run layered health/business checks, restore the previous tag on failure, and clean dangling resources only after success.

- [ ] **Step 4: Add symptom-based troubleshooting**

Cover build failure, wrong architecture, service crash loop, central readiness failure, node offline, authorization failure, CUDA unavailable, RK3588 device/runtime failure, conversion rejection, inference without preview, tunnel outage, and full disk.

- [ ] **Step 5: Add configuration, port, and command quick references**

List central, trainer, RK3588, offline, network, and log-retention variables; mark required values, defaults, secret values, and whether a restart is required.

### Task 7: Verify The Rewritten Manual

**Files:**
- Verify: `docs/system-guide.md`
- Verify: `docs/superpowers/specs/2026-08-14-deployment-operations-guide-design.md`

- [ ] **Step 1: Check required section coverage**

Run:

```bash
rg -n '^## ' docs/system-guide.md
rg -n 'Torch CPU|Torch CUDA|Paddle CPU|Paddle CUDA|RK3588|WireGuard|SSH 隧道|离线部署|升级|回滚|故障排查|速查' docs/system-guide.md
```

Expected: every required deployment and operations section appears at least once.

- [ ] **Step 2: Validate repository paths mentioned by the manual**

Run a shell loop over all backticked `deploy/` and `scripts/` paths extracted from the manual and confirm every repository-local path exists. Exclude documented output paths and wildcard patterns.

Expected: no missing repository source path.

- [ ] **Step 3: Validate all Compose examples**

Run:

```bash
docker compose --env-file deploy/.env.example -f deploy/compose.yaml config --quiet
docker compose --env-file deploy/nodes/trainer/.env.example -f deploy/nodes/trainer/compose.yaml config --quiet
docker compose --env-file deploy/nodes/trainer/.env.example -f deploy/nodes/trainer/compose.yaml -f deploy/nodes/trainer/compose.cuda.yaml config --quiet
docker compose --env-file deploy/nodes/rk3588/.env.example -f deploy/nodes/rk3588/compose.yaml config --quiet
```

Expected: all configurations exit with status `0`.

- [ ] **Step 4: Scan for unsafe or stale examples**

Run:

```bash
rg -n 'dev-admin-token|dev-worker-token|RKNODE_.*TOKEN=[^<]|:latest|rm -rf|docker system prune|docker volume rm' docs/system-guide.md
```

Expected: no real/default credential or implicit `latest`; every destructive match appears only in an explicitly marked danger section.

- [ ] **Step 5: Check internal links and Markdown hygiene**

Run the repository's available Markdown/link checker if present; otherwise inspect local Markdown targets and verify fenced-code delimiters are balanced.

Expected: all local links resolve and code fences are balanced.

- [ ] **Step 6: Record the Git limitation**

Run:

```bash
git status --short
```

Expected in this workspace: `fatal: not a git repository`; commit steps are skipped and must not be reported as completed.
