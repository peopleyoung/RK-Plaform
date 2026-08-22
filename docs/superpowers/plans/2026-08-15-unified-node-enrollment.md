# Unified Node Enrollment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give trainer, converter, and inference direct nodes one secure platform-register-first enrollment flow, while retaining static-token compatibility and a non-disruptive migration path for running nodes.

**Architecture:** `service_endpoints` becomes the source of truth for both endpoint identity and enrollment state. The platform issues a short-lived hashed enrollment credential, the node exchanges it for a long-lived Token and persists that Token, and a shared probe lifecycle activates scheduling only after authenticated identity validation. The browser, online Compose files, offline bundles, and operator documentation all expose the same node-host address and bootstrap contract.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy, SQLite, pytest, Pyright, Ruff, React, TypeScript, Vite, Playwright, Docker Compose, Bash.

---

## Delivery Boundaries

The implementation stays in one project plan because the backend contract, node bootstrap, frontend workflow, and Compose changes form one atomic onboarding feature. Each task below leaves a testable checkpoint and does not require the RK3588 hardware until the final operational acceptance task.

The repository root is not a Git repository. Therefore, the usual per-task commit step is replaced with an explicit verification checkpoint. Do not initialize a new repository, fabricate commit hashes, or delete user-owned files. If the project is later placed under Git, create one commit per completed task using the task title as the commit subject.

## File Map

- `backend/platform_api/contracts.py`: public enrollment enums and request/response models.
- `backend/platform_api/db_models.py`: persisted endpoint enrollment fields.
- `backend/platform_api/database.py`: additive SQLite migration and legacy-row classification.
- `backend/platform_api/settings.py`: enrollment lifetime setting.
- `backend/platform_api/node_enrollment.py`: code generation, hashing, reissue, claim, and idempotency.
- `backend/platform_api/direct_node_lifecycle.py`: shared successful/failed probe recording and activation.
- `backend/platform_api/service.py`: endpoint create/update/probe integration and legacy static-token branch.
- `backend/platform_api/routes.py`: admin issue/reissue routes and unauthenticated claim route.
- `backend/platform_api/direct_dispatcher.py`: enrollment-aware probing and scheduling.
- `backend/platform_api/inference_service.py`: pending direct inference record and first-probe activation.
- `workers/node_service/enrollment.py`: node-side credential resolution, claim client, and atomic Token persistence.
- `workers/node_service/config.py`, `workers/node_service/main.py`, `workers/node_service/factory.py`, `workers/common/config.py`: bootstrap integration and one Token for HTTP plus Worker APIs.
- `src/types.ts`, `src/api/client.ts`, `src/pages/SettingsPage.tsx`, `src/styles.css`: platform-register-first UI.
- `scripts/smoke.mjs`: browser-level enrollment workflow coverage.
- `deploy/nodes/**`, `deploy/offline/**`, `deploy/*.example.yaml`: online/offline bootstrap configuration.
- `README.md`, `docs/simple-node-deployment.md`, `docs/system-guide.md`, `docs/offline-deployment.md`: first-deployment and migration instructions.
- `tests/test_database_migrations.py`, `tests/test_direct_nodes.py`, `tests/test_node_enrollment.py`, `tests/test_node_service.py`, `tests/test_self_contained_deploy.py`, `tests/test_offline_deploy.py`: regression coverage.

### Task 1: Persist and expose endpoint enrollment state

**Files:**
- Create: `tests/test_database_migrations.py`
- Modify: `backend/platform_api/contracts.py`
- Modify: `backend/platform_api/db_models.py`
- Modify: `backend/platform_api/database.py`
- Modify: `backend/platform_api/settings.py`
- Modify: `backend/platform_api/service.py`
- Test: `tests/test_database_migrations.py`
- Test: `tests/test_direct_nodes.py`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write failing contract and migration tests**

Add this legacy-schema helper and the exact migration cases below:

```python
def seed_legacy_service_endpoint(
    database_path: Path, *, token_configured: bool, mode: str
) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE service_endpoints (
                id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(120) NOT NULL UNIQUE,
                kind VARCHAR(30) NOT NULL,
                endpoint VARCHAR(500) NOT NULL,
                mode VARCHAR(20) NOT NULL DEFAULT 'pull',
                scheme VARCHAR(10) NOT NULL DEFAULT 'http',
                host VARCHAR(255) NOT NULL DEFAULT '',
                port INTEGER NOT NULL DEFAULT 10081,
                accelerator VARCHAR(30) NOT NULL,
                capabilities_json JSON NOT NULL,
                enabled BOOLEAN NOT NULL,
                token_configured BOOLEAN NOT NULL DEFAULT 0,
                probe_status VARCHAR(30) NOT NULL DEFAULT 'unprobed',
                last_probe_at DATETIME,
                last_error TEXT,
                remote_metadata_json JSON,
                inference_node_id VARCHAR(48),
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO service_endpoints (
                id, name, kind, endpoint, mode, scheme, host, port,
                accelerator, capabilities_json, enabled, token_configured,
                probe_status, remote_metadata_json, created_at, updated_at
            ) VALUES (
                'service_legacy', 'legacy-trainer', 'trainer',
                'http://192.0.2.10:10081', ?, 'http', '192.0.2.10', 10081,
                'cpu', '["yolo-detect"]', 1, ?, 'online', '{}',
                '2026-08-15 00:00:00', '2026-08-15 00:00:00'
            )
            """,
            (mode, int(token_configured)),
        )


def test_existing_direct_endpoint_with_token_is_migrated_as_enrolled(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    seed_legacy_service_endpoint(database_path, token_configured=True, mode="direct")
    database = Database(f"sqlite:///{database_path}")
    database.create_schema()
    with database.session() as session:
        record = session.get(ServiceEndpointRecord, "service_legacy")
        assert record is not None
        assert record.enrollment_status == "enrolled"
        assert record.enrollment_token_hash is None


def test_existing_direct_endpoint_without_token_is_migrated_as_pending(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    seed_legacy_service_endpoint(database_path, token_configured=False, mode="direct")
    database = Database(f"sqlite:///{database_path}")
    database.create_schema()
    with database.session() as session:
        record = session.get(ServiceEndpointRecord, "service_legacy")
        assert record is not None
        assert record.enrollment_status == "pending"


def test_endpoint_response_redacts_enrollment_secrets(client: TestClient) -> None:
    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        session.add(
            ServiceEndpointRecord(
                id="service_redaction",
                name="redaction-trainer",
                kind="trainer",
                endpoint="http://192.0.2.20:10081",
                mode="direct",
                scheme="http",
                host="192.0.2.20",
                port=10081,
                accelerator="cpu",
                capabilities_json=["yolo-detect"],
                enabled=True,
                token_configured=False,
                enrollment_status="pending",
                enrollment_token_hash="a" * 64,
                probe_status="unprobed",
            )
        )
    response = client.get("/api/v1/service-endpoints", headers=ADMIN_HEADERS)
    assert response.status_code == 200
    serialized = response.text
    assert "enrollmentTokenHash" not in serialized
    assert "enrollmentToken" not in serialized
```

In `tests/test_settings.py`, assert that the default enrollment lifetime is 900 seconds and that `RKNODE_NODE_ENROLLMENT_TTL_SECONDS=60` overrides it.

- [ ] **Step 2: Run the focused tests and confirm the red state**

Run:

```bash
.venv/bin/pytest -q tests/test_database_migrations.py tests/test_settings.py tests/test_direct_nodes.py -k 'migration or enrollment or redacts'
```

Expected: failures for missing model fields, response properties, and `node_enrollment_ttl_seconds`.

- [ ] **Step 3: Add the public types and settings contract**

Add the enum before the endpoint models, then place the create and reissue responses after `ServiceEndpointResponse` in `backend/platform_api/contracts.py`:

```python
class ServiceEndpointEnrollmentStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    ENROLLED = "enrolled"


```

Add these redacted fields to `ServiceEndpointResponse`:

```python
enrollment_status: ServiceEndpointEnrollmentStatus
enrollment_expires_at: datetime | None
enrollment_claimed_at: datetime | None
enrolled_at: datetime | None
```

Then add:

```python
class ServiceEndpointCreateResponse(ServiceEndpointResponse):
    enrollment_token: str | None = None


class ServiceEndpointEnrollmentResponse(ApiModel):
    endpoint_id: str
    enrollment_status: ServiceEndpointEnrollmentStatus
    enrollment_token: str
    enrollment_expires_at: datetime
```

Add this setting to `Settings` in `backend/platform_api/settings.py` using the existing environment parsing pattern:

```python
node_enrollment_ttl_seconds: int = Field(default=900, ge=60, le=86400)
```

Its environment name must be `RKNODE_NODE_ENROLLMENT_TTL_SECONDS`.

- [ ] **Step 4: Add additive columns and deterministic legacy classification**

Add the following SQLAlchemy fields to `ServiceEndpointRecord`:

```python
enrollment_status: Mapped[str] = mapped_column(String(20), default="enrolled", index=True)
enrollment_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
enrollment_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
enrollment_claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
enrolled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

In `Database.create_schema()`, add the five columns with `enrollment_status` defaulting to `enrolled`, then run this migration only after the columns exist:

```sql
UPDATE service_endpoints
SET enrollment_status = CASE
  WHEN mode = 'direct' AND token_configured = 0 THEN 'pending'
  ELSE 'enrolled'
END
WHERE enrollment_status IS NULL
   OR enrollment_status NOT IN ('pending', 'claimed', 'enrolled')
```

For SQLite databases where the newly added column already contains its default, also update direct rows with `token_configured = 0` and no enrollment hash to `pending`. Do not alter endpoint addresses, secret files, or pull-mode behavior.

- [ ] **Step 5: Serialize only redacted state**

Update `service_endpoint_response()` in `backend/platform_api/service.py` to map the four public timestamps and enum. Never add `enrollment_token_hash` to an API model.

```python
enrollment_status=ServiceEndpointEnrollmentStatus(record.enrollment_status),
enrollment_expires_at=record.enrollment_expires_at,
enrollment_claimed_at=record.enrollment_claimed_at,
enrolled_at=record.enrolled_at,
```

- [ ] **Step 6: Run the Task 1 checkpoint**

Run:

```bash
.venv/bin/pytest -q tests/test_database_migrations.py tests/test_settings.py tests/test_direct_nodes.py
.venv/bin/pyright backend/platform_api/contracts.py backend/platform_api/db_models.py backend/platform_api/database.py backend/platform_api/settings.py backend/platform_api/service.py
.venv/bin/ruff check backend/platform_api tests/test_database_migrations.py tests/test_settings.py tests/test_direct_nodes.py
```

Expected: all commands exit 0; endpoint JSON contains state and dates but no enrollment hash or secret.

### Task 2: Issue, reissue, and claim one-time credentials

**Files:**
- Create: `backend/platform_api/node_enrollment.py`
- Create: `tests/test_node_enrollment.py`
- Modify: `backend/platform_api/contracts.py`
- Modify: `backend/platform_api/routes.py`
- Modify: `backend/platform_api/service.py`
- Modify: `backend/platform_api/inference_service.py`
- Test: `tests/test_node_enrollment.py`
- Test: `tests/test_direct_nodes.py`

- [ ] **Step 1: Write failing API tests for the complete credential lifecycle**

Create reusable `trainer_payload()`, `converter_payload()`, and `inference_payload()` functions without a `token` property. Add these exact behavioral tests:

```python
@pytest.mark.parametrize("kind", ["trainer", "converter", "inference"])
def test_direct_endpoint_can_be_registered_before_node_starts(
    client: TestClient, kind: str
) -> None:
    created = client.post(
        "/api/v1/service-endpoints",
        headers=ADMIN_HEADERS,
        json=endpoint_payload(kind),
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["enrollmentStatus"] == "pending"
    assert body["tokenConfigured"] is False
    assert len(body["enrollmentToken"]) >= 43
    assert body["enrollmentExpiresAt"] is not None
    listed = client.get("/api/v1/service-endpoints", headers=ADMIN_HEADERS).json()
    assert "enrollmentToken" not in listed[0]


def test_claim_is_idempotent_until_first_authenticated_probe(client: TestClient) -> None:
    endpoint, code = create_pending_endpoint(client, "trainer")
    first = claim_endpoint(client, endpoint["id"], code, "trainer")
    second = claim_endpoint(client, endpoint["id"], code, "trainer")
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["nodeToken"] == second.json()["nodeToken"]


def test_reissue_invalidates_the_previous_code(client: TestClient) -> None:
    endpoint, old_code = create_pending_endpoint(client, "converter")
    issued = client.post(
        f"/api/v1/service-endpoints/{endpoint['id']}/enrollment-token",
        headers=ADMIN_HEADERS,
    )
    assert issued.status_code == 200, issued.text
    assert issued.json()["enrollmentToken"] != old_code
    assert claim_endpoint(client, endpoint["id"], old_code, "converter").status_code == 401
```

Also cover expired code, code bound to another endpoint, wrong name, wrong kind, wrong accelerator, incomplete capability set, already-enrolled conflict, and a claim response that never exposes admin/global Worker Tokens.

- [ ] **Step 2: Run the enrollment tests and confirm expected failures**

Run:

```bash
.venv/bin/pytest -q tests/test_node_enrollment.py
```

Expected: failures because create still requires a static Token and claim/reissue routes do not exist.

- [ ] **Step 3: Define claim request and response models**

Add the following models to `backend/platform_api/contracts.py`:

```python
class NodeEnrollmentClaim(ApiModel):
    enrollment_token: str = Field(min_length=16, max_length=512)
    name: str = Field(min_length=1, max_length=120)
    kind: ServiceEndpointKind
    accelerator: Literal["cpu", "cuda", "rk3588"]
    capabilities: list[str] = Field(min_length=1, max_length=32)
    version: str = Field(default="unknown", min_length=1, max_length=80)
    max_concurrency: int = Field(default=1, ge=1, le=1024)
    features: list[str] = Field(default_factory=list, max_length=64)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class NodeEnrollmentClaimResponse(ApiModel):
    endpoint_id: str
    node_token: str
    enrollment_status: ServiceEndpointEnrollmentStatus
```

Reuse the existing camel-case `ApiModel` serialization, so frontend and node clients receive `nodeToken`, `endpointId`, and `enrollmentStatus`.

- [ ] **Step 4: Implement focused enrollment state transitions**

Create `backend/platform_api/node_enrollment.py` with a module-level lock and this public surface:

```python
@dataclass(frozen=True)
class EnrollmentCredential:
    token: str
    expires_at: datetime


class NodeEnrollmentService:
    def __init__(self, context: AppContext) -> None:
        self.context = context

    def issue(self, endpoint_id: str) -> ServiceEndpointEnrollmentResponse:
        """Issue a new code for a pending or claimed direct endpoint."""

    def claim(
        self, endpoint_id: str, payload: NodeEnrollmentClaim
    ) -> NodeEnrollmentClaimResponse:
        """Validate endpoint identity and return one stable long-lived Token."""
```

Implement the following exact rules:

- Generate enrollment codes and long-lived Tokens with `secrets.token_urlsafe(48)`.
- Store enrollment codes only as `hashlib.sha256(token.encode("utf-8")).hexdigest()`.
- Compare hashes with `hmac.compare_digest`.
- Use `utc_now()` plus `settings.node_enrollment_ttl_seconds`.
- Serialize claim/issue inside one module-level `threading.Lock`, because the deployed API uses one process and the secret store is filesystem-backed.
- On first valid claim, reject collisions with admin Token, global Worker Token, and every existing node Token; write the long-lived Token to `NodeSecretStore`; set `token_configured=True`, `enrollment_status="claimed"`, and `enrollment_claimed_at=utc_now()`.
- On a retry with the same valid code, read and return the already stored Token.
- Reissuing a pending endpoint replaces only the one-time code. Reissuing a claimed endpoint keeps its existing long-lived Token and `claimed` state so a node that lost the first response receives the same credential.
- Leave the enrollment hash and expiry intact until activation.
- Return `node_enrollment_invalid` with HTTP 401 for invalid or cross-endpoint codes, `node_enrollment_expired` with HTTP 401 for expired codes, `node_enrollment_identity_mismatch` with HTTP 409 for identity mismatch, and `node_enrollment_already_active` with HTTP 409 after activation.

Use `AppError` directly for the two custom 401 codes because `AuthenticationError` always emits `unauthorized`.

- [ ] **Step 5: Change endpoint creation while preserving the legacy branch**

Change `PlatformService.create_service_endpoint()` to return `ServiceEndpointCreateResponse`.

- For a direct payload without `token`, create the endpoint as `pending`, do not probe it, issue a code, and return that code once.
- For a direct payload with `token`, keep the existing validation/probe behavior, mark it `enrolled`, set `enrolled_at`, and return `enrollmentToken: null`. This branch is API-only migration compatibility and is not exposed in the new browser form.
- For pull mode, preserve the existing behavior and never issue an enrollment code.
- For pending inference, create a linked `InferenceNodeRecord` through a new `InferenceService.create_pending_direct_node()` method. It must use `hardware_id=f"direct:{endpoint_id}"`, `lifecycle="pending_registration"`, `connectivity="offline"`, `health="unknown"`, the configured adapters, and no registration or access Token hash.

The new inference method has this exact signature:

```python
def create_pending_direct_node(
    self,
    session: Session,
    *,
    name: str,
    hardware_id: str,
    adapters: list[str],
    enabled: bool,
) -> InferenceNodeResponse:
```

- [ ] **Step 6: Add the two routes with explicit authentication boundaries**

Add to `backend/platform_api/routes.py`:

```python
@router.post(
    "/service-endpoints/{endpoint_id}/enrollment-token",
    response_model=ServiceEndpointEnrollmentResponse,
)
def reissue_service_endpoint_enrollment(
    endpoint_id: str, _: Admin, context: Context
) -> ServiceEndpointEnrollmentResponse:
    return NodeEnrollmentService(context).issue(endpoint_id)


@router.post(
    "/node-enrollments/{endpoint_id}/claim",
    response_model=NodeEnrollmentClaimResponse,
)
def claim_node_enrollment(
    endpoint_id: str, payload: NodeEnrollmentClaim, context: Context
) -> NodeEnrollmentClaimResponse:
    return NodeEnrollmentService(context).claim(endpoint_id, payload)
```

The claim route intentionally has no `Admin` or Worker dependency. The one-time code is its only bootstrap credential.

- [ ] **Step 7: Run the Task 2 checkpoint**

Run:

```bash
.venv/bin/pytest -q tests/test_node_enrollment.py tests/test_direct_nodes.py
.venv/bin/pyright backend/platform_api tests/test_node_enrollment.py tests/test_direct_nodes.py
.venv/bin/ruff check backend/platform_api tests/test_node_enrollment.py tests/test_direct_nodes.py
```

Expected: all enrollment cases pass; list/detail APIs expose no credentials; existing static-token direct-node tests remain green.

### Task 3: Activate claimed nodes only after a shared authenticated probe

**Files:**
- Create: `backend/platform_api/direct_node_lifecycle.py`
- Modify: `backend/platform_api/direct_dispatcher.py`
- Modify: `backend/platform_api/service.py`
- Modify: `backend/platform_api/inference_service.py`
- Modify: `tests/test_direct_nodes.py`
- Modify: `tests/test_node_enrollment.py`

- [ ] **Step 1: Write failing activation and scheduling tests**

Add tests which run the real `DirectNodeDispatcher` against `running_node()`:

```python
def test_claimed_endpoint_activates_on_probe_then_dispatches_next_iteration(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    endpoint, code = create_pending_endpoint_for_running_node(client, "trainer")
    token = claim_endpoint(client, endpoint["id"], code, "trainer").json()["nodeToken"]
    node_state.expected_token = token
    dispatcher = DirectNodeDispatcher(cast(AppContext, client.app.state.context))
    dispatcher.run_once()
    assert get_endpoint(client, endpoint["id"])["enrollmentStatus"] == "enrolled"
    assert node_state.dispatched_job_ids == []
    dispatcher.run_once()
    assert node_state.dispatched_job_ids == [queued_job_id]


def test_pending_endpoint_is_neither_probed_nor_scheduled(client: TestClient) -> None:
    endpoint, _ = create_pending_endpoint(client, "trainer")
    DirectNodeDispatcher(cast(AppContext, client.app.state.context)).run_once()
    body = get_endpoint(client, endpoint["id"])
    assert body["probeStatus"] == "unprobed"
    assert body["enrollmentStatus"] == "pending"


def test_activation_destroys_the_one_time_code(client: TestClient) -> None:
    endpoint, code, token = create_claimed_running_endpoint(client, "converter")
    DirectNodeDispatcher(cast(AppContext, client.app.state.context)).run_once()
    response = claim_endpoint(client, endpoint["id"], code, "converter")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "node_enrollment_already_active"
```

Add an inference case asserting that the linked node remains pending before probe, becomes active after probe, and has an internal Agent Token stored under `purpose="agent"` only after activation.

- [ ] **Step 2: Run the activation tests and confirm the red state**

Run:

```bash
.venv/bin/pytest -q tests/test_direct_nodes.py tests/test_node_enrollment.py -k 'activate or activation or pending_endpoint or claimed_endpoint'
```

Expected: claimed endpoints are currently either omitted from lifecycle handling or dispatched in the same iteration.

- [ ] **Step 3: Extract shared probe persistence**

Create `backend/platform_api/direct_node_lifecycle.py` with these functions:

```python
def record_probe_success(
    context: AppContext,
    endpoint_id: str,
    health: dict[str, Any],
) -> ServiceEndpointResponse:
    """Record validated health, promote claimed nodes, and sync linked records."""


def record_probe_failure(
    context: AppContext,
    endpoint_id: str,
    message: str,
) -> ServiceEndpointResponse | None:
    """Record connectivity failure without changing enrollment state."""
```

Move the existing Worker and inference metadata updates from `DirectNodeDispatcher._record_probe_success()` into this file. When a claimed endpoint succeeds:

```python
record.enrollment_status = ServiceEndpointEnrollmentStatus.ENROLLED.value
record.enrollment_token_hash = None
record.enrollment_expires_at = None
record.enrolled_at = now
```

The function must create or update trainer/converter `WorkerRecord` entries and call `InferenceService.activate_direct_node()` for inference. If that method returns a newly generated Agent Token, write it to `NodeSecretStore` with `purpose="agent"` before returning.

- [ ] **Step 4: Add first-probe inference activation**

Add this method to `InferenceService`:

```python
def activate_direct_node(
    self,
    session: Session,
    node_id: str,
    *,
    name: str,
    adapters: list[str],
    max_model_instances: int,
    metadata: dict[str, Any],
    enabled: bool,
) -> tuple[InferenceNodeResponse, str | None]:
```

It must update runtime/driver/pipeline/self-test metadata with the same rules as `create_direct_node()`. Generate and hash a new internal Agent Token only when `access_token_hash` is absent; return `None` on later probes so the stored Agent Token is not rotated.

- [ ] **Step 5: Make dispatcher state-aware and prevent same-iteration dispatch**

Add `enrollment_status: str` to `EndpointSnapshot`. `_endpoints()` must select only enabled direct endpoints whose status is `claimed` or `enrolled`.

In `run_once()`:

```python
record_probe_success(self.context, endpoint.id, health)
if endpoint.enrollment_status == ServiceEndpointEnrollmentStatus.CLAIMED.value:
    continue
self._process_cleanups(endpoint, client)
```

This guarantees that the activation probe cannot dispatch jobs, clean caches, or apply an inference revision in the same iteration. The next dispatcher iteration sees `enrolled` and may schedule work.

- [ ] **Step 6: Route manual probes through the same lifecycle**

Update `PlatformService.probe_service_endpoint()` so pending endpoints return a redacted response with `probe_status="unprobed"` and `last_error="node enrollment has not been claimed"`. Claimed/enrolled probes must validate remote identity and then call `record_probe_success()` or `record_probe_failure()`; do not keep a second partial activation implementation in `service.py`.

- [ ] **Step 7: Run the Task 3 checkpoint**

Run:

```bash
.venv/bin/pytest -q tests/test_direct_nodes.py tests/test_node_enrollment.py tests/test_inference_api.py
.venv/bin/pyright backend/platform_api
.venv/bin/ruff check backend/platform_api tests/test_direct_nodes.py tests/test_node_enrollment.py
```

Expected: three node kinds activate only after a matching health response; claimed nodes receive no work during activation; legacy inference Agent tests remain green.

### Task 4: Resolve and persist node credentials at startup

**Files:**
- Create: `workers/node_service/enrollment.py`
- Create: `tests/test_node_service_enrollment.py`
- Modify: `workers/node_service/config.py`
- Modify: `workers/node_service/main.py`
- Modify: `tests/test_node_service.py`

- [ ] **Step 1: Write failing node bootstrap tests**

Use a local `ThreadingHTTPServer` or `httpx.MockTransport` and temporary files. Add these tests:

```python
def test_claims_and_atomically_persists_node_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    enrollment_file = tmp_path / "enrollment-token"
    token_file = tmp_path / "state" / "node-token"
    enrollment_file.write_text("one-time-enrollment-token-with-32-characters\n")
    settings = enrollment_settings(monkeypatch, enrollment_file, token_file)
    token = resolve_node_token(settings)
    assert token == "long-lived-node-token-with-48-characters-value"
    assert token_file.read_text() == token + "\n"
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600


def test_persistent_token_file_wins_over_legacy_env_and_enrollment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token_file = tmp_path / "node-token"
    token_file.write_text("persistent-node-token-with-32-characters\n")
    token_file.chmod(0o600)
    monkeypatch.setenv("RKNODE_NODE_TOKEN", "legacy-node-token-with-32-characters")
    assert resolve_node_token(file_settings(token_file)) == token_file.read_text().strip()


def test_startup_fails_without_any_credential_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="node credential source"):
        resolve_node_token(empty_enrollment_settings(monkeypatch, tmp_path))
```

Also test response-loss retry, malformed claim JSON, non-200 error code propagation, too-short stored Token, and restart with the enrollment file removed.

- [ ] **Step 2: Run the bootstrap tests and confirm the red state**

Run:

```bash
.venv/bin/pytest -q tests/test_node_service_enrollment.py tests/test_node_service.py
```

Expected: import failures for `workers.node_service.enrollment` and settings fields.

- [ ] **Step 3: Separate identity settings from resolved credentials**

Change `NodeServiceSettings.token` to `str | None` and add:

```python
endpoint_id: str | None = None
platform_url: str | None = None
enrollment_token_file: Path = Path("/run/secrets/rknode-enrollment-token")
node_token_file: Path = Path("/data/state/node-token")
request_timeout_seconds: float = 30.0
```

`from_env()` must still require name, kind, accelerator, and capabilities. It must no longer require `RKNODE_NODE_TOKEN`, but if the legacy variable is present it must validate at least 16 characters. Parse:

```text
RKNODE_ENDPOINT_ID
RKNODE_PLATFORM_URL
RKNODE_ENROLLMENT_TOKEN_FILE
RKNODE_NODE_TOKEN_FILE
RKNODE_REQUEST_TIMEOUT_SECONDS
```

- [ ] **Step 4: Implement deterministic credential resolution**

Create this public surface in `workers/node_service/enrollment.py`:

```python
def resolve_node_token(settings: NodeServiceSettings) -> str:
    """Resolve file, legacy environment, or enrollment claim in that order."""


def claim_node_token(settings: NodeServiceSettings, enrollment_token: str) -> str:
    """POST immutable node identity to the platform claim endpoint."""


def persist_node_token(path: Path, token: str) -> None:
    """Write a newline-terminated Token atomically with mode 0600."""
```

Use only the standard library (`urllib.request`, `json`, `tempfile`, `os.replace`) so every existing node image already has the required client. Build the URL as:

```python
claim_url = (
    f"{settings.platform_url.rstrip('/')}/api/v1/"
    f"node-enrollments/{quote(settings.endpoint_id, safe='')}/claim"
)
```

Create the temp file in `path.parent`, call `os.fchmod(fd, 0o600)`, flush and `os.fsync()`, then `os.replace()` it. Refuse empty or shorter-than-16 Tokens from any source. Read the enrollment secret only when neither the persistent file nor legacy environment Token is usable.

- [ ] **Step 5: Resolve before constructing either server or Worker runtime**

In `workers/node_service/main.py`:

```python
settings = NodeServiceSettings.from_env()
settings = replace(settings, token=resolve_node_token(settings))
app = create_node_app(
    settings,
    runtime=build_runtime(settings),
    inference=build_inference(settings),
)
```

This ordering prevents the HTTP server from accepting traffic before its long-lived credential is durable.

- [ ] **Step 6: Run the Task 4 checkpoint**

Run:

```bash
.venv/bin/pytest -q tests/test_node_service_enrollment.py tests/test_node_service.py
.venv/bin/pyright workers/node_service tests/test_node_service_enrollment.py tests/test_node_service.py
.venv/bin/ruff check workers/node_service tests/test_node_service_enrollment.py tests/test_node_service.py
```

Expected: Token files are mode `0600`, restart uses the persistent file, and missing credentials fail before Uvicorn starts.

### Task 5: Use the resolved node Token for trainer and converter Worker APIs

**Files:**
- Modify: `workers/common/config.py`
- Modify: `workers/node_service/factory.py`
- Modify: `tests/test_node_service.py`
- Test: `tests/test_node_service_enrollment.py`

- [ ] **Step 1: Write failing Token-injection and compatibility tests**

Add:

```python
def test_worker_config_accepts_resolved_node_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_worker_identity(monkeypatch, tmp_path)
    monkeypatch.delenv("RKNODE_WORKER_TOKEN", raising=False)
    config = WorkerConfig.from_env(
        token_override="resolved-node-token-with-32-characters"
    )
    assert config.token == "resolved-node-token-with-32-characters"


def test_legacy_worker_token_still_loads_without_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_worker_identity(monkeypatch, tmp_path)
    monkeypatch.setenv("RKNODE_WORKER_TOKEN", "legacy-worker-token-with-32-characters")
    assert WorkerConfig.from_env().token == "legacy-worker-token-with-32-characters"
```

Also assert `build_runtime(settings)` passes `settings.token` and still rejects name, kind, accelerator, or capability mismatches.

- [ ] **Step 2: Run the focused tests and confirm the red state**

Run:

```bash
.venv/bin/pytest -q tests/test_node_service.py tests/test_node_service_enrollment.py -k 'worker_config or resolved_node_token or mismatch'
```

Expected: `WorkerConfig.from_env()` rejects the unknown keyword or requires `RKNODE_WORKER_TOKEN`.

- [ ] **Step 3: Add explicit Token injection**

Change the signature in `workers/common/config.py`:

```python
@classmethod
def from_env(cls, *, token_override: str | None = None) -> WorkerConfig:
    token = token_override or os.getenv("RKNODE_WORKER_TOKEN", "")
```

Keep every existing identity and capability validation. In `workers/node_service/factory.py`, require a resolved `settings.token` for trainer/converter and call:

```python
if not settings.token:
    raise ValueError("node Token must be resolved before building the Worker runtime")
worker_config = WorkerConfig.from_env(token_override=settings.token)
```

Inference does not construct `WorkerRuntime`, so its later internal Agent Token remains separate.

- [ ] **Step 4: Run the Task 5 checkpoint**

Run:

```bash
.venv/bin/pytest -q tests/test_node_service.py tests/test_node_service_enrollment.py tests/test_workspace_cleanup.py
.venv/bin/pyright workers/common workers/node_service
.venv/bin/ruff check workers/common workers/node_service tests/test_node_service.py tests/test_node_service_enrollment.py
```

Expected: enrolled trainer/converter containers need one long-lived node Token, while standalone legacy Workers can still use `RKNODE_WORKER_TOKEN`.

### Task 6: Make online and offline Compose use the same bootstrap contract

**Files:**
- Create: `deploy/nodes/trainer/compose.enrollment.yaml`
- Create: `deploy/nodes/rk3588/compose.enrollment.yaml`
- Create: `deploy/offline/trainer/compose.enrollment.yaml`
- Create: `deploy/offline/rk3588/compose.enrollment.yaml`
- Modify: `deploy/nodes/trainer/compose.yaml`
- Modify: `deploy/nodes/rk3588/compose.yaml`
- Modify: `deploy/offline/trainer/compose.yaml`
- Modify: `deploy/offline/rk3588/compose.converter.yaml`
- Modify: `deploy/offline/rk3588/compose.inference.yaml`
- Modify: `deploy/trainer.compose.example.yaml`
- Modify: `deploy/rk3588/compose.yaml`
- Modify: `deploy/offline/common/deploy.sh`
- Modify: `deploy/offline/common/verify.sh`
- Modify: `scripts/package_offline_bundle.py`
- Modify: all node `.env.example` files under `deploy/nodes`, `deploy/offline/trainer`, and `deploy/offline/rk3588`
- Modify: `tests/test_self_contained_deploy.py`
- Modify: `tests/test_offline_deploy.py`

- [ ] **Step 1: Write failing Compose contract tests**

Add assertions that every new-enrollment configuration resolves to:

```python
assert environment["RKNODE_ENDPOINT_ID"] == expected_endpoint_id
assert environment["RKNODE_PLATFORM_URL"] == "http://172.16.66.249:8000"
assert environment["RKNODE_ENROLLMENT_TOKEN_FILE"] == "/run/secrets/rknode-enrollment-token"
assert environment["RKNODE_NODE_TOKEN_FILE"] == "/data/state/node-token"
assert "RKNODE_NODE_TOKEN" not in environment
assert "RKNODE_WORKER_TOKEN" not in environment
```

For the combined RK3588 Compose, assert converter and inference have different endpoint IDs, enrollment secret sources, named data volumes, and published ports `10081` and `10082`. Retain tests proving the legacy base Compose accepts static node Token variables.

- [ ] **Step 2: Run deployment tests and confirm the red state**

Run:

```bash
.venv/bin/pytest -q tests/test_self_contained_deploy.py tests/test_offline_deploy.py
```

Expected: failures for missing enrollment overlays and required static Token interpolation.

- [ ] **Step 3: Keep base Compose backward compatible**

In existing base Compose files, change required static Token interpolation to optional empty values and add the persistent file setting:

```yaml
environment:
  RKNODE_NODE_TOKEN: ${RKNODE_NODE_TOKEN:-}
  RKNODE_NODE_TOKEN_FILE: ${RKNODE_NODE_TOKEN_FILE:-/data/state/node-token}
```

For RK3588 use the role-specific legacy values `${RKNODE_CONVERTER_TOKEN:-}` and `${RKNODE_INFERENCE_TOKEN:-}`. Remove duplicated `RKNODE_WORKER_TOKEN` from node-service containers because Task 5 injects the resolved node Token. Startup validation in `resolve_node_token()` remains the final guard against an uncredentialed service.

- [ ] **Step 4: Add enrollment overlays with role-specific secret files**

The trainer overlay must add:

```yaml
services:
  trainer:
    environment:
      RKNODE_ENDPOINT_ID: ${RKNODE_ENDPOINT_ID:?set endpoint ID from the platform}
      RKNODE_PLATFORM_URL: ${RKNODE_PLATFORM_URL:?set central platform URL}
      RKNODE_ENROLLMENT_TOKEN_FILE: /run/secrets/rknode-enrollment-token
      RKNODE_NODE_TOKEN_FILE: /data/state/node-token
    secrets:
      - rknode-enrollment-token

secrets:
  rknode-enrollment-token:
    file: ${RKNODE_ENROLLMENT_TOKEN_PATH:?set enrollment token file path}
```

The RK3588 overlay must define `rknode-converter-enrollment-token` and `rknode-inference-enrollment-token` separately and mount each as `/run/secrets/rknode-enrollment-token` in only its matching service. Use `RKNODE_CONVERTER_ENDPOINT_ID`, `RKNODE_INFERENCE_ENDPOINT_ID`, `RKNODE_CONVERTER_ENROLLMENT_TOKEN_PATH`, and `RKNODE_INFERENCE_ENROLLMENT_TOKEN_PATH`.

- [ ] **Step 5: Make offline scripts verify enrollment without reading the long-lived Token**

Update `deploy/offline/common/deploy.sh` to require the enrollment code files before `docker compose config` when an enrollment overlay is listed in `bundle.env`.

Update `scripts/package_offline_bundle.py` so every trainer and RK3588 bundle includes `compose.enrollment.yaml` after its base/CUDA files.

Update `deploy/offline/common/verify.sh` to verify local container health from inside the target container: read `/data/state/node-token` inside the container, verify its mode is `600`, and use it only in an in-container `urllib.request` call to `http://127.0.0.1:10081/health`. The script must not copy or print the long-lived Token on the host. For the combined RK3588 bundle, run this check independently in `converter` and `inference`. Finish by instructing the central operator to confirm `enrolled + online` in System Settings; do not put the central admin Token on a node host.

- [ ] **Step 6: Run expanded Compose validation**

Run:

```bash
.venv/bin/pytest -q tests/test_self_contained_deploy.py tests/test_offline_deploy.py
docker compose --env-file deploy/nodes/trainer/.env.example -f deploy/nodes/trainer/compose.yaml config --quiet
docker compose --env-file deploy/nodes/rk3588/.env.example -f deploy/nodes/rk3588/compose.yaml config --quiet
```

Create temporary secret files with `mktemp -d` for overlay expansion, populate them with a test code, set the required endpoint IDs, and run `docker compose config --quiet` for all four enrollment overlays. Delete only that exact temporary directory after validation.

Use this shell shape so cleanup targets only the directory created by the check:

```bash
rknode_enrollment_tmp="$(mktemp -d)"
trap 'rm -rf -- "${rknode_enrollment_tmp}"' EXIT
printf '%s\n' 'test-enrollment-code-with-32-characters' > "${rknode_enrollment_tmp}/trainer"
chmod 600 "${rknode_enrollment_tmp}/trainer"
RKNODE_ENDPOINT_ID=service_test \
RKNODE_ENROLLMENT_TOKEN_PATH="${rknode_enrollment_tmp}/trainer" \
docker compose --env-file deploy/nodes/trainer/.env.example \
  -f deploy/nodes/trainer/compose.yaml \
  -f deploy/nodes/trainer/compose.enrollment.yaml config --quiet
```

Repeat the same explicit expansion for offline trainer and for the two separate RK3588 secret paths.

Expected: base and enrollment configurations expand; no resolved enrollment configuration contains a required long-lived Token variable.

### Task 7: Replace manual Token entry with enrollment in System Settings

**Files:**
- Modify: `src/types.ts`
- Modify: `src/api/client.ts`
- Modify: `src/pages/SettingsPage.tsx`
- Modify: `src/styles.css`
- Modify: `scripts/smoke.mjs`

- [ ] **Step 1: Extend the smoke fixture and write failing browser assertions**

In `scripts/smoke.mjs`, mock a pending endpoint create response containing `enrollmentToken` only on the POST response. Add assertions that:

```javascript
await page.getByRole("button", { name: "新增节点" }).click();
await page.getByLabel("节点宿主机 IP / 域名").waitFor();
assert.equal(await page.getByLabel(/节点 Token/).count(), 0);
await page.getByRole("button", { name: "保存并生成注册码" }).click();
await page.getByRole("heading", { name: "节点部署凭据" }).waitFor();
await page.getByText("enroll-once-ui-secret").waitFor();
await page.getByRole("button", { name: "关闭部署凭据" }).click();
assert.equal(await page.getByText("enroll-once-ui-secret").count(), 0);
```

Also assert pending/claimed/enrolled labels, reissue POST behavior, and that editing an existing direct endpoint never renders the long-lived Token.

- [ ] **Step 2: Run UI build/smoke and confirm the red state**

Run:

```bash
npm run build
npm run test:ui
```

Expected: TypeScript or browser assertions fail because enrollment fields and modal do not exist.

- [ ] **Step 3: Add exact frontend types and client calls**

Add to `src/types.ts`:

```typescript
export type ServiceEndpointEnrollmentStatus = 'pending' | 'claimed' | 'enrolled'

export interface ServiceEndpointCreated extends ServiceEndpoint {
  enrollmentToken: string | null
}

export interface ServiceEndpointEnrollment {
  endpointId: string
  enrollmentStatus: ServiceEndpointEnrollmentStatus
  enrollmentToken: string
  enrollmentExpiresAt: string
}
```

Add `enrollmentStatus`, `enrollmentExpiresAt`, `enrollmentClaimedAt`, and `enrolledAt` to `ServiceEndpoint`. Keep `token?: string` in `ServiceEndpointInput` only for API compatibility, but do not bind it to the new UI.

Change/add client methods:

```typescript
createServiceEndpoint: (payload: ServiceEndpointInput) =>
  request<ServiceEndpointCreated>('/service-endpoints', {
    method: 'POST', body: JSON.stringify(payload),
  }),
reissueServiceEndpointEnrollment: (id: string) =>
  request<ServiceEndpointEnrollment>(
    `/service-endpoints/${encodeURIComponent(id)}/enrollment-token`,
    { method: 'POST' },
  ),
```

- [ ] **Step 4: Implement the register-first form and one-time modal**

In `SettingsPage.tsx`:

- Rename the address label to `节点宿主机 IP / 域名` and add concise help text that `RKNODE_PLATFORM_URL` points in the opposite direction to the central API.
- Remove the direct-node Token field and the create-time Token validation.
- Do not render `测试连接` for a new pending direct node; keep it for enrolled edits only.
- Save new direct endpoints with `token: undefined` and button text `保存并生成注册码`.
- Store `ServiceEndpointCreated | ServiceEndpointEnrollment` only in component state.
- Show a `节点部署凭据` modal containing endpoint ID, expiry, one-time code, platform URL variable names, copy buttons, and a download button that creates a text `Blob` in memory.
- Clear the secret-bearing state on close and component unmount. Never use `localStorage`, `sessionStorage`, URL parameters, console logging, or notification text for the secret.
- Provide reissue for `pending` or `claimed`; require a confirmation dialog before invalidating the prior code.

Use Lucide `Copy`, `Download`, and `RefreshCw` icons with accessible labels and existing tooltip/title conventions.

- [ ] **Step 5: Render combined enrollment and connectivity status**

Use this precedence:

```typescript
function endpointStatus(endpoint: ServiceEndpoint) {
  if (!endpoint.enabled) return { label: '已停用', tone: 'neutral' as const }
  if (endpoint.enrollmentStatus === 'pending') {
    const expired = endpoint.enrollmentExpiresAt
      ? Date.parse(endpoint.enrollmentExpiresAt) < Date.now()
      : true
    return { label: expired ? '注册错误' : '待部署', tone: 'warning' as const }
  }
  if (endpoint.enrollmentStatus === 'claimed') {
    return { label: '已领取/待探测', tone: 'warning' as const }
  }
  return { label: probeLabel(endpoint.probeStatus), tone: probeTone(endpoint.probeStatus) }
}
```

Count a node as schedulable only when enabled, enrolled, and online. Label pull endpoints as `旧版兼容`, not as newly enrolled direct nodes.

- [ ] **Step 6: Run the Task 7 checkpoint**

Run:

```bash
npm run build
npm run test:ui
```

Expected: both exit 0; the one-time code disappears when its modal closes and is absent from browser persistence.

### Task 8: Update first-deployment, offline, VPN, and tunnel instructions

**Files:**
- Modify: `README.md`
- Modify: `docs/simple-node-deployment.md`
- Modify: `docs/system-guide.md`
- Modify: `docs/offline-deployment.md`
- Modify: `deploy/systemd/rknode-node-tunnel.service.example`
- Modify: `.trellis/spec/backend/direct-node-contract.md`

- [ ] **Step 1: Add a documentation regression test before editing prose**

Add assertions to `tests/test_self_contained_deploy.py` that the four operator documents contain:

```python
required_terms = {
    "节点宿主机 IP / 域名",
    "RKNODE_ENDPOINT_ID",
    "RKNODE_PLATFORM_URL",
    "RKNODE_ENROLLMENT_TOKEN_FILE",
    "RKNODE_NODE_TOKEN_FILE",
    "pending",
    "claimed",
    "enrolled",
}
for path in operator_documents:
    text = path.read_text()
    assert required_terms <= set(term for term in required_terms if term in text)
```

Add a negative assertion that new-deployment command blocks do not instruct operators to generate or paste `RKNODE_NODE_TOKEN` or `RKNODE_WORKER_TOKEN`. Keep a clearly marked migration appendix that documents those legacy variables.

- [ ] **Step 2: Run the documentation contract test and confirm the red state**

Run:

```bash
.venv/bin/pytest -q tests/test_self_contained_deploy.py -k 'documentation or operator'
```

Expected: failures because current documents still make long-lived Token entry the primary flow.

- [ ] **Step 3: Rewrite the first-deployment sequence identically in all guides**

Document this concrete order for trainer, converter, and inference:

```text
1. Confirm central API reachability from the node host.
2. In System Settings register node name, kind, accelerator, capabilities,
   node-host IP/DNS, and published host port.
3. Save the endpoint ID and one-time enrollment code to a mode-0600 file.
4. Set RKNODE_ENDPOINT_ID and RKNODE_PLATFORM_URL in the node .env.
5. Start base Compose plus compose.enrollment.yaml.
6. Wait for claimed, then enrolled + online in System Settings.
7. Remove the one-time secret file after enrollment.
8. Restart once and verify the persistent node-token file is reused.
```

Include concrete target addresses:

```text
trainer    172.16.66.249:10081
converter  172.30.82.12:10081
inference  172.30.82.12:10082
```

State that the latter two remain on `172.29.0.1:11081/11082` in the current environment until direct probes succeed.

- [ ] **Step 4: Cover the full deployment matrix and security boundary**

For CPU Torch, CPU Paddle, CUDA Torch, CUDA Paddle, RK3588 conversion, and RK3588 inference, show exact `docker compose` commands with the enrollment overlay. For offline installation, show creation of the local secret file before `deploy.sh` and its removal after activation. For VPN, use the node's VPN IP/DNS in platform host/port. For an emergency SSH tunnel, label the central forwarding address temporary and preserve the current unit only as rollback documentation.

Every production section must state:

- allow the central platform IP to reach only the published node port;
- use VPN or HTTPS on untrusted networks;
- never expose node control ports to the public Internet;
- never store SSH passwords in platform settings;
- the node address is the node service, while `RKNODE_PLATFORM_URL` is the central platform API.

- [ ] **Step 5: Capture the contract in Trellis spec**

Update `.trellis/spec/backend/direct-node-contract.md` with the three-state lifecycle, code idempotency boundary, address semantics, Token resolution order, and scheduling prohibition before enrollment. Preserve existing protocol and error contracts that remain valid.

- [ ] **Step 6: Run the Task 8 checkpoint**

Run:

```bash
.venv/bin/pytest -q tests/test_self_contained_deploy.py tests/test_offline_deploy.py
rg -n 'RKNODE_(NODE|WORKER)_TOKEN=' README.md docs deploy --glob '*.md' --glob '*.example' --glob '*.yaml'
```

Expected: tests pass. Every remaining static Token match is inside a clearly labeled legacy/migration example or base compatibility Compose, not the new deployment path.

### Task 9: Run full quality gates and a local three-kind end-to-end enrollment

**Files:**
- Modify only files required by failures found in this task.
- Test: all Python and browser suites.

- [ ] **Step 1: Run all backend and node tests**

Run:

```bash
.venv/bin/pytest -q
```

Expected: all tests pass with no skipped enrollment tests.

- [ ] **Step 2: Run static analysis**

Run:

```bash
.venv/bin/pyright
.venv/bin/ruff check backend workers tests scripts
```

Expected: both commands exit 0.

- [ ] **Step 3: Run frontend verification**

Run:

```bash
npm run build
npm run test:ui
```

Expected: TypeScript compilation, Vite production build, and Playwright smoke all pass.

- [ ] **Step 4: Exercise one trainer, converter, and inference endpoint locally**

Start the test API on an unused loopback port with a temporary data directory, register three direct endpoints through HTTP, write each returned code to its own mode-0600 file, and start three node-service processes on separate ports. Poll the endpoint list until every record is `enrolled` and `online`.

Use this assertion script against the returned JSON:

```python
assert {item["kind"] for item in endpoints} == {"trainer", "converter", "inference"}
assert all(item["enrollmentStatus"] == "enrolled" for item in endpoints)
assert all(item["probeStatus"] == "online" for item in endpoints)
assert all(item["tokenConfigured"] is True for item in endpoints)
assert all("enrollmentToken" not in item for item in endpoints)
```

Stop only the temporary processes and remove only their explicit temporary directory after the test.

- [ ] **Step 5: Validate restart and expired-code behavior end to end**

Remove the three enrollment files, restart the node processes against the same state directories, and verify they return online without claiming again. Create one additional pending endpoint with a 60-second lifetime, let the code expire, and verify claim returns HTTP 401 with `node_enrollment_expired` while the endpoint remains unschedulable.

- [ ] **Step 6: Record the quality checkpoint in the Trellis task**

Update `.trellis/tasks/08-15-unified-node-enrollment/implement.md` checkboxes with the exact commands and results. Do not mark hardware migration complete from local emulation.

### Task 10: Build, pin, and deploy images without breaking the current tunnel rollback

**Files:**
- Modify: `deploy/offline/VERSION`
- Modify: online/offline Compose image defaults and `.env.example` files already touched in Task 6
- Modify: `README.md`
- Modify: `docs/simple-node-deployment.md`
- Modify: `docs/system-guide.md`
- Modify: `docs/offline-deployment.md`
- Generated outside source control: `release/offline/*.tar`

- [ ] **Step 1: Record current runtime and image rollback facts**

Before building or recreating containers, record:

```bash
docker ps --format '{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}'
docker image inspect \
  rknode-platform-api:2026.08.13 \
  rknode-platform-web:2026.08.13 \
  rknode-trainer-torch-cpu:2026.08.13 \
  --format '{{index .RepoTags 0}}|{{.Id}}|{{.Architecture}}'
systemctl --user is-active rknode-rk3588-direct-tunnel.service
```

Save the output in the Trellis task execution notes. Do not remove current image tags, containers, volumes, endpoint rows, or the tunnel during this task.

- [ ] **Step 2: Build the new seven-image matrix with explicit tags**

Set `deploy/offline/VERSION` to `2026.08.15`. Build on the matching architecture:

```text
rknode-platform-api:2026.08.15
rknode-platform-web:2026.08.15
rknode-trainer-torch-cpu:2026.08.15
rknode-trainer-paddle-cpu:2026.08.15
rknode-trainer-torch-cuda12.4:2026.08.15
rknode-trainer-paddle-cuda12.6:2026.08.15
rknode-rk3588-node:2026.08.15-business
```

Build the six amd64 images with:

```bash
RKNODE_RELEASE_VERSION=2026.08.15 RKNODE_SOURCE_REVISION=unified-enrollment-2026.08.15 scripts/build_offline_images.sh platform
RKNODE_RELEASE_VERSION=2026.08.15 RKNODE_SOURCE_REVISION=unified-enrollment-2026.08.15 scripts/build_offline_images.sh trainer-torch-cpu
RKNODE_RELEASE_VERSION=2026.08.15 RKNODE_SOURCE_REVISION=unified-enrollment-2026.08.15 scripts/build_offline_images.sh trainer-paddle-cpu
RKNODE_RELEASE_VERSION=2026.08.15 RKNODE_SOURCE_REVISION=unified-enrollment-2026.08.15 scripts/build_offline_images.sh trainer-torch-cuda
RKNODE_RELEASE_VERSION=2026.08.15 RKNODE_SOURCE_REVISION=unified-enrollment-2026.08.15 scripts/build_offline_images.sh trainer-paddle-cuda
```

Build the arm64 image on the board or an approved native arm64 builder with:

```bash
RKNODE_RELEASE_VERSION=2026.08.15-business RKNODE_SOURCE_REVISION=unified-enrollment-business-2026.08.15 scripts/build_offline_images.sh rk3588
```

Do not relabel the old `2026.08.13-business` image as the new build.

- [ ] **Step 3: Verify image labels, architecture, and embedded enrollment code**

Inspect all seven images for the exact OCI version/source labels and correct architecture. Run disposable containers to verify:

```bash
docker run --rm rknode-platform-api:2026.08.15 python -c \
  'from backend.platform_api.node_enrollment import NodeEnrollmentService; print(NodeEnrollmentService.__name__)'
docker run --rm --entrypoint python rknode-trainer-torch-cpu:2026.08.15 -c \
  'from workers.node_service.enrollment import resolve_node_token; print(resolve_node_token.__name__)'
```

Run the equivalent node-side import check for both Paddle images, both CUDA images, and the RK3588 image on compatible hosts.

- [ ] **Step 4: Package six archives for seven images**

Generate these six role bundles because the platform archive contains two images and the RK3588 archive serves both converter and inference containers:

```text
rknode-platform-amd64-2026.08.15.tar
rknode-trainer-torch-cpu-amd64-2026.08.15.tar
rknode-trainer-paddle-cpu-amd64-2026.08.15.tar
rknode-trainer-torch-cuda-amd64-2026.08.15.tar
rknode-trainer-paddle-cuda-amd64-2026.08.15.tar
rknode-rk3588-node-arm64-2026.08.15-business.tar
```

On amd64, run:

```bash
python3 scripts/package_offline_bundle.py platform-amd64 --version 2026.08.15
python3 scripts/package_offline_bundle.py trainer-torch-cpu-amd64 --version 2026.08.15
python3 scripts/package_offline_bundle.py trainer-paddle-cpu-amd64 --version 2026.08.15
python3 scripts/package_offline_bundle.py trainer-torch-cuda-amd64 --version 2026.08.15
python3 scripts/package_offline_bundle.py trainer-paddle-cuda-amd64 --version 2026.08.15
```

On arm64, run:

```bash
python3 scripts/package_offline_bundle.py rk3588-node-arm64 --version 2026.08.15-business
```

Verify every manifest and SHA256, and document explicitly that six tar files correctly contain the seven-image matrix.

- [ ] **Step 5: Deploy central API/Web first and preserve rollback**

Update central `.env` to the new API/Web tags, expand Compose, recreate only `api` and `web`, then verify `/api/v1/ready`, browser login on port `5173`, endpoint listing, and database migration. If any check fails, restore the recorded old tags and recreate only those two services.

- [ ] **Step 6: Migrate the local trainer through the enrollment workflow**

Wait for active training jobs to finish. Register or reissue the trainer endpoint, deploy `rknode-trainer-torch-cpu:2026.08.15` with the enrollment overlay and its existing persistent data volume, then verify `claimed -> enrolled -> online`, Worker registration, restart persistence, and one minimal training job. Keep the old image tag until this acceptance passes.

- [ ] **Step 7: Gate RK3588 address migration on direct connectivity**

From the central API host run:

```bash
curl --connect-timeout 3 http://172.30.82.12:10081/health
curl --connect-timeout 3 http://172.30.82.12:10082/health
```

An HTTP 401 response proves the port is reachable; connection refused or timeout means the gate is not met. Only after both ports are reachable may the operator deploy `rknode-rk3588-node:2026.08.15-business`, enroll the two independent endpoint IDs, change platform addresses to `172.30.82.12:10081/10082`, and verify both online. Stop `rknode-rk3588-direct-tunnel.service` only after conversion and inference acceptance both pass.

- [ ] **Step 8: Perform final non-destructive release verification**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/pyright
.venv/bin/ruff check backend workers tests scripts
npm run build
npm run test:ui
docker images --format '{{.Repository}}:{{.Tag}}|{{.ID}}' | sort
docker image inspect \
  rknode-platform-api:2026.08.15 \
  rknode-platform-web:2026.08.15 \
  rknode-trainer-torch-cpu:2026.08.15 \
  rknode-trainer-paddle-cpu:2026.08.15 \
  rknode-trainer-torch-cuda12.4:2026.08.15 \
  rknode-trainer-paddle-cuda12.6:2026.08.15 \
  --format '{{index .RepoTags 0}}|{{.Id}}|{{.Architecture}}'
```

Expected: all quality gates pass, exactly the intended new tags are documented as current, old tags remain available for rollback until an operator separately approves cleanup, and current endpoint connectivity is unchanged unless its migration gate passed.

## Completion Criteria

- Trainer, converter, and inference direct nodes share the same API, state machine, node bootstrap code, UI, Compose variables, and operator sequence.
- A node can be registered while offline, claim exactly one durable long-lived Token, restart without its code, and receive work only after an authenticated matching probe.
- Existing static-token nodes remain compatible, and no automatic migration changes their addresses or Tokens.
- The current RK3588 tunnel remains active until direct board ports are independently proven reachable and both business paths pass acceptance.
- The release uses seven new images in six offline tar bundles, with exact tags and checksums documented.
- All Python, type, lint, frontend, browser, Compose, local enrollment, and applicable hardware checks pass.
