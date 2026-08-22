# Unified Node Enrollment Design

Date: 2026-08-15
Status: Approved
Trellis task: `.trellis/tasks/08-15-unified-node-enrollment`

## 1. Purpose

Training, conversion, and inference nodes must use one direct-node onboarding model. An operator first registers the intended node identity and its node-host address in the platform. The node container is then deployed on that host, claims a short-lived enrollment credential, persists its long-lived bearer token, and becomes schedulable only after the platform completes an authenticated health probe.

The configured service address always means the node host that owns the container and published port. A central-host SSH forwarding address is an emergency transport workaround, not the normal endpoint identity.

## 2. Chosen Approach

The platform owns enrollment and long-lived credential generation, while the operator or existing deployment automation owns container deployment.

Alternatives considered:

1. Continue copying a long-lived Token between the node `.env` and the platform. This is simple but exposes durable credentials to shell history, deployment files, and the clipboard.
2. Have the platform SSH into each node and write the Token. This removes a copy step but requires storing high-privilege SSH credentials and expands the platform into an infrastructure orchestrator.
3. Use short-lived enrollment with node-initiated claim. This reuses an existing project pattern, requires only outbound node-to-platform access during bootstrap, and keeps SSH credentials out of the platform.

Option 3 is selected. The platform does not push containers or credentials over an unauthenticated inbound node endpoint.

## 3. Address Contract

`ServiceEndpointPayload.host` and `port` describe the node host as seen by the central API:

- a CPU trainer on the central server may use the central server's LAN IP because that server is also the trainer's node host;
- an RK3588 converter uses the RK3588 host IP and its published converter port;
- an RK3588 inference service uses the same RK3588 host IP and its distinct published inference port;
- a VPN deployment uses the node's VPN IP or DNS name;
- `127.0.0.1`, a platform Docker bridge gateway, or a central-host SSH forwarding port is not a production node identity unless the node container actually runs on that host.

For the current topology, the target records are:

```text
trainer    http://172.16.66.249:10081
converter  http://172.30.82.12:10081
inference  http://172.30.82.12:10082
```

The final two records cannot be activated until the central platform can connect directly to both RK3588 ports. The backend cannot reliably infer the correct IP on a multi-homed host, so the operator enters it explicitly and the platform validates it by probing.

## 4. Enrollment Lifecycle

Each direct `service_endpoints` record gains an independent enrollment lifecycle:

```text
pending -> claimed -> enrolled
    |         |          |
    +------ expired   online/offline
```

- `pending`: the endpoint exists, a registration code is valid, and no long-lived node Token is configured.
- `claimed`: the platform has issued the long-lived Token, but has not yet authenticated the running node at its configured address.
- `enrolled`: the first authenticated health probe passed and the one-time credential was destroyed.
- `expired`: represented as a pending endpoint whose expiration time is in the past; the admin can reissue a code.
- `online` and `offline` remain connectivity states in `probe_status`; they do not replace enrollment state.

`enabled` remains the operator's scheduling intent. The dispatcher probes enabled direct endpoints in `claimed` or `enrolled` state, but dispatches work only after the endpoint reaches `enrolled`. Pending and claimed endpoints are never schedulable.

## 5. Data Model

`ServiceEndpointRecord` adds:

```text
enrollment_status: pending | claimed | enrolled
enrollment_token_hash: nullable SHA-256 digest
enrollment_expires_at: nullable UTC datetime
enrollment_claimed_at: nullable UTC datetime
enrolled_at: nullable UTC datetime
```

The long-lived Token remains in `NodeSecretStore`, because the central platform must present the plaintext value when calling the node. It is never returned by normal endpoint list or detail APIs.

Existing rows migrate as follows:

- direct endpoint with a readable node secret: `enrolled`;
- direct endpoint without a node secret: `pending` with no active code until an admin reissues one;
- pull endpoint: unchanged and outside the new enrollment dispatcher filter.

Existing inference endpoints keep their linked `InferenceNodeRecord`. A newly created direct inference endpoint creates a linked pending inference record immediately so it is visible before deployment. The first successful health probe fills runtime metadata, marks it active, and provisions the internal Agent Token used for revision callbacks.

## 6. API Contracts

### 6.1 Create endpoint

```text
POST /api/v1/service-endpoints
Authorization: Bearer <admin>
```

For a direct endpoint, `token` is no longer required. The server does not probe during creation. It creates a pending endpoint and returns a one-time creation response:

```json
{
  "id": "service_...",
  "enrollmentStatus": "pending",
  "enrollmentToken": "one-time-secret",
  "enrollmentExpiresAt": "2026-08-15T12:00:00Z"
}
```

`enrollmentToken` appears only in the create or reissue response. List and update responses expose status and expiry but never either credential.

### 6.2 Reissue enrollment

```text
POST /api/v1/service-endpoints/{endpoint_id}/enrollment-token
Authorization: Bearer <admin>
```

This invalidates any prior one-time code. It is allowed for pending or claimed endpoints with no active jobs or inference deployment. Reissuing a claimed endpoint keeps the already generated long-lived Token so a lost claim response can be recovered without creating competing credentials. Re-enrolling an active node is a separate explicit action and must pass existing busy-node protections.

### 6.3 Claim enrollment

```text
POST /api/v1/node-enrollments/{endpoint_id}/claim
```

The request contains the one-time code and the node's immutable identity plus health metadata:

```json
{
  "enrollmentToken": "one-time-secret",
  "name": "rk3588-converter-01",
  "kind": "converter",
  "accelerator": "rk3588",
  "capabilities": ["yolo-detect"],
  "version": "0.1.0",
  "maxConcurrency": 1,
  "features": [],
  "diagnostics": {}
}
```

The platform validates the code hash, expiry, endpoint identity, accelerator, and complete configured capability set. It generates a cryptographically random long-lived node Token, stores it in `NodeSecretStore`, marks the endpoint claimed with `token_configured=true`, and returns the Token only through the claim response while that enrollment credential remains valid.

Claim is idempotent until activation: a retry with the same still-valid enrollment code returns the same stored long-lived Token. This covers a response loss after the server commits but before the node writes its state file. After the first authenticated health probe, the platform clears the enrollment hash and expiry; later claims return a conflict.

### 6.4 Probe and activation

The existing probe path uses the stored long-lived Token. A successful `/health` response must match name, kind, accelerator, and configured capabilities. Only then does the platform:

1. promote `enrollment_status` from `claimed` to `enrolled` (`token_configured` is already true after claim);
2. clear one-time enrollment material;
3. set `probe_status=online` and update metadata;
4. create or update the compatible Worker record for trainer/converter;
5. activate and update the linked inference record for inference;
6. allow the dispatcher to send work.

An unreachable or mismatched node remains `claimed` and unschedulable with a specific `last_error`.

## 7. Node Bootstrap

All node-service containers use the same credential resolution order:

1. read a non-empty persistent Token file;
2. accept legacy `RKNODE_NODE_TOKEN` for compatibility;
3. when neither exists, claim with the endpoint ID and enrollment secret file;
4. atomically write the returned Token with mode `0600` before starting the HTTP server or Worker runtime.

New variables:

```text
RKNODE_ENDPOINT_ID
RKNODE_PLATFORM_URL
RKNODE_ENROLLMENT_TOKEN_FILE=/run/secrets/rknode-enrollment-token
RKNODE_NODE_TOKEN_FILE=/data/state/node-token
```

The enrollment code is delivered as a short-lived file mounted through Compose secrets, not as a long-lived environment variable. The operator removes the bootstrap secret after the platform reports the endpoint enrolled. Reuse fails after activation even if the file remains temporarily present.

For trainer and converter, `WorkerConfig` receives the resolved node Token directly instead of requiring a duplicate `RKNODE_WORKER_TOKEN`. The name, kind, accelerator, and capabilities continue to be checked against the node-service identity. Inference continues to receive its separate internal Agent Token when the platform applies a desired revision.

## 8. Frontend Workflow

The System Settings page becomes the only new-node entry point for trainer, converter, and inference direct endpoints:

1. The operator selects type, enters the node name, node-host IP/DNS, published port, accelerator, and capabilities.
2. Saving creates a pending endpoint without testing a nonexistent service.
3. A result modal shows the endpoint ID, expiry, one-time enrollment code, and copyable bootstrap configuration. The secret is not placed in browser persistent storage.
4. The list displays `待部署`, `已领取/待探测`, `在线`, `离线`, `已停用`, or `注册错误` from enrollment plus probe state.
5. Pending/expired rows provide a reissue action. Claimed rows provide probe and guarded re-enrollment actions. Online rows retain probe/edit/delete operations.
6. The form label is `节点宿主机 IP / 域名`, with help text distinguishing the node address from `RKNODE_PLATFORM_URL`.

The separate legacy inference-node creation dialog remains available only for the old pull Agent compatibility path or is explicitly labeled legacy. New direct inference nodes are created through System Settings and appear automatically in the inference node view.

## 9. Security and Failure Handling

- Enrollment codes use at least 256 bits of entropy, expire after 15 minutes by default, are stored only as hashes, and are bound to one endpoint.
- Long-lived Tokens use independent random values and retain current conflict checks against admin, global Worker, and other node Tokens.
- Normal API responses, logs, events, and frontend state never include long-lived Tokens.
- Production enrollment and node control traffic must use HTTPS or a VPN when the network is not trusted. A firewall restricts node control ports to the central platform.
- Invalid, expired, already activated, or identity-mismatched claims return stable error codes without revealing which comparison failed beyond operator-safe diagnostics.
- A node that claims but never becomes reachable cannot receive work. The admin can correct its host/port, re-probe, reissue the short-lived code, or delete it.
- Platform restart is safe because enrollment metadata, server-side secrets, and node Token files are persistent.

## 10. Compatibility and Migration

The rollout is additive:

1. Add schema columns and API/node compatibility while continuing to accept static Token deployments.
2. Migrate existing token-configured direct endpoints to `enrolled` without changing their address or credentials.
3. Update Compose examples and the UI to use enrollment for new deployments.
4. Re-enroll existing nodes only during an operator-approved maintenance window; do not rotate all working Tokens automatically.
5. For the current RK3588, first expose and firewall `172.30.82.12:10081/10082`, verify both from the central API host, update the endpoint addresses, and only then stop the SSH tunnel.

Rollback keeps the legacy environment Token path and current tunnel service until direct-node verification succeeds. No database row, persistent volume, or working Token is deleted as part of address migration.

## 11. Test Strategy

Backend tests cover pending creation, one-time response redaction, expiry, wrong endpoint, identity mismatch, idempotent claim, activation, dispatch exclusion before activation, reissue, static-token migration, and all three node kinds.

Node-service tests use temporary state directories and a test enrollment server to verify first claim, atomic `0600` persistence, response-loss retry, restart without enrollment, legacy environment fallback, and refusal to start without any credential source.

Frontend tests cover the host-address wording, pending creation flow, one-time secret modal, absence from persistent storage, status rendering, reissue, and the legacy inference distinction.

Deployment tests expand every online/offline Compose variant and verify the enrollment secret mount, persistent Token path, distinct RK3588 ports/volumes, compatibility variables, and final pinned images. End-to-end verification registers one trainer, one converter, and one inference endpoint before starting their containers and observes each transition to online without a manually supplied long-lived Token.

## 12. Operational Boundary

The platform registers identities and schedules work. Operators or deployment automation load images and run Compose on each node host. The platform does not store SSH credentials, modify remote firewalls, choose host IPs, or create network tunnels.

The current environment cannot complete the RK3588 direct-address migration until the board exposes its published ports to the central host. The enrollment feature can be implemented and tested locally before that hardware/network acceptance step.
