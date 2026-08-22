# RKNode Final Image Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every deployment entry point use the verified final RKNode image tags, migrate the running CPU trainer away from its ambiguous local alias, and delete only confirmed obsolete RKNode images and offline archives.

**Architecture:** Treat the verified seven-image matrix as the deployment contract. Update online/offline Compose defaults and operator documentation first, validate all expanded configurations, migrate the one running alias-backed container without changing its image ID or data volume, and perform exact-target deletion only after final bundles pass checksum verification.

**Tech Stack:** Docker Engine, Docker Compose v2, Bash, YAML, Markdown, SHA256 offline bundles.

---

### Task 1: Pin Online Node Deployment To Final Images

**Files:**
- Modify: `deploy/.env.example`
- Modify: `deploy/compose.yaml`
- Modify: `deploy/nodes/trainer/.env`
- Modify: `deploy/nodes/trainer/.env.example`
- Modify: `deploy/nodes/trainer/compose.yaml`
- Modify: `deploy/nodes/rk3588/.env`
- Modify: `deploy/nodes/rk3588/.env.example`
- Modify: `deploy/nodes/rk3588/compose.yaml`
- Modify: `deploy/rk3588/.env.example`
- Modify: `deploy/rk3588/compose.yaml`

- [x] **Step 1: Pin the trainer image**

Set the actual and example CPU trainer image to:

```dotenv
RKNODE_TRAINER_IMAGE=rknode-trainer-torch-cpu:2026.08.13
```

Set the Compose fallback to the same tag.

Also pin the central services to `rknode-platform-api:2026.08.13` and
`rknode-platform-web:2026.08.13` so an already-built deployment can use
`--no-build` without Compose-generated floating names.

- [x] **Step 2: Pin the RK3588 image and pipeline version**

Set every online RK3588 image value/fallback to:

```dotenv
RKNODE_RK3588_IMAGE=rknode-rk3588-node:2026.08.13-business
RKNODE_PIPELINE_VERSION=rknode-cpp-runtime-2026.08.13-business
```

Remove the obsolete `RKNODE_PIPELINE_IMAGE=nv-video-pipeline:rk3588-20260811-preview` value from the active direct-node environment because the current Compose does not consume it.

- [x] **Step 3: Expand online Compose files**

Run:

```bash
docker compose -p rknode-direct-cpu --env-file deploy/nodes/trainer/.env -f deploy/nodes/trainer/compose.yaml config
docker compose --env-file deploy/nodes/rk3588/.env -f deploy/nodes/rk3588/compose.yaml config
```

Expected: trainer resolves to `rknode-trainer-torch-cpu:2026.08.13`; converter and inference resolve to `rknode-rk3588-node:2026.08.13-business`; existing logging and device mappings remain present.

### Task 2: Pin Offline Deployment To The Business Image

**Files:**
- Modify: `deploy/offline/rk3588/node.env.example`
- Modify: `deploy/offline/rk3588/converter.env.example`
- Modify: `deploy/offline/rk3588/inference.env.example`
- Modify: `deploy/offline/rk3588/compose.converter.yaml`
- Modify: `deploy/offline/rk3588/compose.inference.yaml`
- Modify: `release/offline/SHA256SUMS`

- [x] **Step 1: Update offline RK3588 templates**

Use `rknode-rk3588-node:2026.08.13-business` for all image variables and fallbacks. Use `rknode-cpp-runtime-2026.08.13-business` for the pipeline version examples.

- [x] **Step 2: Replace the obsolete outer bundle checksum**

Remove the checksum entry for `rknode-rk3588-node-arm64-2026.08.13.tar` and add:

```text
e42beb93f4bbd374c50911a9b3db4698e7111fa41ec83fe393a88821d09764c7  release/offline/rknode-rk3588-node-arm64-2026.08.13-business.tar
```

- [x] **Step 3: Expand offline Compose files**

Run each converter and inference template with its matching example environment.

Expected: both services resolve to the business image and retain `pull_policy: never`, bounded logging, and explicit RK3588 devices.

- [x] **Step 4: Prevent duplicate semantic version suffixes during packaging**

Add a regression test for `2026.08.13` plus `2026.08.13-business`, then make template replacement skip versions that already carry a semantic suffix.

### Task 3: Rewrite Operator Documentation Around The Final Matrix

**Files:**
- Modify: `README.md`
- Modify: `docs/system-guide.md`
- Modify: `docs/simple-node-deployment.md`
- Modify: `docs/offline-deployment.md`

- [x] **Step 1: Add one authoritative seven-image matrix**

Add the platform, four trainer, and RK3588 business tags. State that tags are immutable deployment inputs and that `latest`/`local` are prohibited for production deployment.

- [x] **Step 2: Replace obsolete examples**

Replace trainer `local` examples with the role-specific `2026.08.13` tags and replace every deployment use of RK3588 `2026.08.13` or `local` with `2026.08.13-business`.

- [x] **Step 3: Separate deploy from rebuild commands**

Use `up -d --no-build` for already-built final images. Keep source rebuild instructions in an explicit development/rebuild subsection that requires a new tag rather than overwriting a final tag.

- [x] **Step 4: Update offline package inventory and verification**

List the business RK3588 archive, explain that the platform bundle carries two images, and make checksum/archive examples match the six retained final bundle files.

- [x] **Step 5: Audit documentation**

Search operator docs and deploy templates for `2026.08.12`, deployment uses of the old RK3588 tag, RKNode `:local`, and `:latest`.

Expected: no deployment instruction uses an obsolete or floating RKNode tag. Historical design/task records are excluded from this operator-doc audit.

### Task 4: Validate Final Bundles Before Destructive Work

**Files:**
- Inspect: `release/offline/SHA256SUMS`
- Inspect: retained final archives under `release/offline/`

- [x] **Step 1: Verify all retained outer archives**

Run:

```bash
sha256sum -c release/offline/SHA256SUMS
```

Expected: all six retained archives report `OK`.

- [x] **Step 2: Inspect the RK3588 business manifest**

Confirm the bundle references image ID `sha256:c8c90ad4065eedb26b343ba1533ddf7d19f3569a0b1d13fde16fa87584f5585b`, architecture `arm64`, version `2026.08.13-business`, and `requiresNetworkDuringDeploy=false`.

### Task 5: Migrate The Running CPU Trainer Alias

**Files:**
- Runtime state: Docker Compose project `rknode-direct-cpu`

- [x] **Step 1: Recreate with the pinned final tag**

Run:

```bash
docker compose -p rknode-direct-cpu --env-file deploy/nodes/trainer/.env -f deploy/nodes/trainer/compose.yaml up -d --no-build --force-recreate
```

- [x] **Step 2: Verify runtime identity and health**

Expected: container configuration references `rknode-trainer-torch-cpu:2026.08.13`, image ID remains `sha256:b5173085fe5b523d54341f3561ca68ae9feb33990324822687fdb627cead61a3`, container is running, and the platform reports the node online.

- [x] **Step 3: Stop if health verification fails**

Do not delete the local alias or any old archive until the trainer is healthy. Because both tags point to the same image ID, rollback consists of restoring the alias and recreating the same Compose service without changing the data volume.

### Task 6: Delete Only The Approved Old Artifacts

**Files:**
- Delete: `release/offline/rknode-platform-amd64-2026.08.12.tar`
- Delete: `release/offline/rknode-trainer-torch-cpu-amd64-2026.08.12.tar`
- Delete: `release/offline/rknode-trainer-paddle-cpu-amd64-2026.08.12.tar`
- Delete: `release/offline/rknode-trainer-torch-cuda-amd64-2026.08.12.tar`
- Delete: `release/offline/rknode-converter-rk3588-arm64-2026.08.12.tar`
- Delete: `release/offline/rknode-inference-rk3588-arm64-2026.08.12.tar`
- Delete: `release/offline/rknode-rk3588-node-arm64-2026.08.13.tar`

- [x] **Step 1: Remove the compatibility alias**

Run `docker image rm rknode-trainer:local`. Expected: only the tag is removed; the final Torch CPU image remains.

- [x] **Step 2: Remove the obsolete RK3588 image**

Run `docker image rm rknode-rk3588-node:2026.08.13`. Expected: the old image is removed while the business image remains.

- [x] **Step 3: Delete the seven exact old archive paths**

Use one explicit `rm --` command containing only the paths listed in this task. Do not use globs, recursive deletion, or a directory target.

- [x] **Step 4: Preserve shared-engine resources**

Do not run Docker prune commands and do not delete the unrelated dangling images labeled for the `kbqa` Compose project.

### Task 7: Final Verification And Record

**Files:**
- Modify: `.trellis/tasks/08-13-inference-business-integration/notes.md`

- [x] **Step 1: Verify Docker inventory and running containers**

Expected: exactly the seven approved RKNode final tags remain; platform API/Web and CPU trainer use final tags and remain running/healthy as applicable.

- [x] **Step 2: Verify release inventory and checksums**

Expected: exactly six final outer archives plus `SHA256SUMS` remain under `release/offline/`; all checksums pass.

- [x] **Step 3: Run Compose and documentation gates**

Validate all online/offline Compose variants and rerun the obsolete-tag audit. Run the frontend production build because the operator docs describe the currently deployed frontend artifact.

- [x] **Step 4: Record cleanup outcome**

Append the retained image matrix, removed artifact scope, reclaimed archive bytes, and external RK3588 hardware-acceptance boundary to the active Trellis task notes.

- [x] **Step 5: Record repository limitation**

The workspace root is not a Git repository, so do not create or claim a Git commit.
