from __future__ import annotations

from datetime import timedelta
from typing import cast

import pytest
from backend.platform_api.context import AppContext
from backend.platform_api.db_models import ServiceEndpointRecord, WorkerRecord
from backend.platform_api.state_machine import utc_now
from fastapi.testclient import TestClient

from .conftest import ADMIN_HEADERS


def endpoint_payload(kind: str, suffix: str = "test") -> dict[str, object]:
    if kind == "trainer":
        accelerator = "cpu"
        capabilities = ["yolo-detect"]
        port = 18081
    elif kind == "converter":
        accelerator = "rk3588"
        capabilities = ["yolo-detect"]
        port = 18082
    else:
        accelerator = "rk3588"
        capabilities = ["deeplab_logits_v1"]
        port = 18083
    return {
        "name": f"enrollment-{kind}-{suffix}",
        "kind": kind,
        "mode": "direct",
        "scheme": "http",
        "host": "127.0.0.1",
        "port": port,
        "accelerator": accelerator,
        "capabilities": capabilities,
        "enabled": True,
    }


def create_pending_endpoint(
    client: TestClient, kind: str, suffix: str = "test"
) -> tuple[dict[str, object], str]:
    response = client.post(
        "/api/v1/service-endpoints",
        headers=ADMIN_HEADERS,
        json=endpoint_payload(kind, suffix),
    )
    assert response.status_code == 201, response.text
    body = cast(dict[str, object], response.json())
    token = body.get("enrollmentToken")
    assert isinstance(token, str)
    return body, token


def claim_payload(kind: str, suffix: str = "test") -> dict[str, object]:
    configured = endpoint_payload(kind, suffix)
    return {
        "name": configured["name"],
        "kind": kind,
        "accelerator": configured["accelerator"],
        "capabilities": configured["capabilities"],
        "version": "test-1.0",
        "maxConcurrency": 1,
        "features": [],
        "diagnostics": {},
    }


def claim_endpoint(
    client: TestClient,
    endpoint_id: object,
    enrollment_token: str,
    kind: str,
    suffix: str = "test",
) -> object:
    return client.post(
        f"/api/v1/node-enrollments/{endpoint_id}/claim",
        json={
            "enrollmentToken": enrollment_token,
            **claim_payload(kind, suffix),
        },
    )


@pytest.mark.parametrize("kind", ["trainer", "converter", "inference"])
def test_direct_endpoint_can_be_registered_before_node_starts(
    client: TestClient, kind: str
) -> None:
    endpoint, code = create_pending_endpoint(client, kind)

    assert endpoint["enrollmentStatus"] == "pending"
    assert endpoint["tokenConfigured"] is False
    assert endpoint["probeStatus"] == "unprobed"
    assert len(code) >= 43
    assert endpoint["enrollmentExpiresAt"] is not None

    listed = client.get("/api/v1/service-endpoints", headers=ADMIN_HEADERS)
    assert listed.status_code == 200
    listed_endpoint = listed.json()[0]
    assert listed_endpoint["enrollmentStatus"] == "pending"
    assert "enrollmentToken" not in listed_endpoint


def test_claim_is_idempotent_until_activation(client: TestClient) -> None:
    endpoint, code = create_pending_endpoint(client, "trainer")

    first = claim_endpoint(client, endpoint["id"], code, "trainer")
    second = claim_endpoint(client, endpoint["id"], code, "trainer")

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_body = first.json()
    assert first_body["nodeToken"] == second.json()["nodeToken"]
    assert len(first_body["nodeToken"]) >= 43
    assert first_body["enrollmentStatus"] == "claimed"

    listed = client.get("/api/v1/service-endpoints", headers=ADMIN_HEADERS).json()[0]
    assert listed["tokenConfigured"] is True
    assert listed["enrollmentStatus"] == "claimed"
    assert "nodeToken" not in listed


def test_reissue_invalidates_previous_code(client: TestClient) -> None:
    endpoint, old_code = create_pending_endpoint(client, "converter")

    issued = client.post(
        f"/api/v1/service-endpoints/{endpoint['id']}/enrollment-token",
        headers=ADMIN_HEADERS,
    )

    assert issued.status_code == 200, issued.text
    new_code = issued.json()["enrollmentToken"]
    assert new_code != old_code
    rejected = claim_endpoint(client, endpoint["id"], old_code, "converter")
    assert rejected.status_code == 401
    assert rejected.json()["error"]["code"] == "node_enrollment_invalid"
    accepted = claim_endpoint(client, endpoint["id"], new_code, "converter")
    assert accepted.status_code == 200, accepted.text


def test_reissue_claimed_endpoint_returns_same_long_lived_token(
    client: TestClient,
) -> None:
    endpoint, code = create_pending_endpoint(client, "trainer")
    first = claim_endpoint(client, endpoint["id"], code, "trainer")
    assert first.status_code == 200, first.text

    issued = client.post(
        f"/api/v1/service-endpoints/{endpoint['id']}/enrollment-token",
        headers=ADMIN_HEADERS,
    )
    assert issued.status_code == 200, issued.text
    recovered = claim_endpoint(
        client,
        endpoint["id"],
        issued.json()["enrollmentToken"],
        "trainer",
    )

    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["nodeToken"] == first.json()["nodeToken"]


def test_legacy_enrolled_endpoint_can_migrate_without_rotating_long_lived_token(
    client: TestClient,
) -> None:
    endpoint, code = create_pending_endpoint(client, "trainer")
    first = claim_endpoint(client, endpoint["id"], code, "trainer")
    assert first.status_code == 200, first.text
    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        record = session.get(ServiceEndpointRecord, cast(str, endpoint["id"]))
        assert record is not None
        record.enrollment_status = "enrolled"
        record.enrollment_token_hash = None
        record.enrollment_expires_at = None
        record.enrollment_claimed_at = None
        record.enrolled_at = utc_now()
        record.probe_status = "online"

    issued = client.post(
        f"/api/v1/service-endpoints/{endpoint['id']}/enrollment-token",
        headers=ADMIN_HEADERS,
    )

    assert issued.status_code == 200, issued.text
    assert issued.json()["enrollmentStatus"] == "pending"
    listed = client.get("/api/v1/service-endpoints", headers=ADMIN_HEADERS).json()[0]
    assert listed["enrollmentStatus"] == "pending"
    assert listed["probeStatus"] == "unprobed"
    assert listed["tokenConfigured"] is False
    recovered = claim_endpoint(
        client,
        endpoint["id"],
        issued.json()["enrollmentToken"],
        "trainer",
    )
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["nodeToken"] == first.json()["nodeToken"]


def test_enrollment_managed_endpoint_cannot_be_reset_to_pending(client: TestClient) -> None:
    endpoint, code = create_pending_endpoint(client, "trainer")
    claimed = claim_endpoint(client, endpoint["id"], code, "trainer")
    assert claimed.status_code == 200, claimed.text
    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        record = session.get(ServiceEndpointRecord, cast(str, endpoint["id"]))
        assert record is not None
        record.enrollment_status = "enrolled"
        record.enrolled_at = utc_now()

    response = client.post(
        f"/api/v1/service-endpoints/{endpoint['id']}/enrollment-token",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "node_enrollment_already_active"


def test_legacy_endpoint_with_active_jobs_cannot_start_migration(
    client: TestClient,
) -> None:
    endpoint, _ = create_pending_endpoint(client, "trainer")
    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        record = session.get(ServiceEndpointRecord, cast(str, endpoint["id"]))
        assert record is not None
        record.enrollment_status = "enrolled"
        record.enrollment_token_hash = None
        record.enrollment_expires_at = None
        record.enrollment_claimed_at = None
        record.enrolled_at = utc_now()
        session.add(
            WorkerRecord(
                id="worker_active_migration",
                name=record.name,
                kind="trainer",
                status="busy",
                capabilities_json=["yolo-detect"],
                accelerator="cpu",
                max_concurrency=1,
                active_jobs=1,
                version="legacy",
                metadata_json={},
            )
        )

    response = client.post(
        f"/api/v1/service-endpoints/{endpoint['id']}/enrollment-token",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "node_enrollment_active_jobs"


def test_expired_enrollment_code_is_rejected(client: TestClient) -> None:
    endpoint, code = create_pending_endpoint(client, "trainer")
    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        record = session.get(ServiceEndpointRecord, cast(str, endpoint["id"]))
        assert record is not None
        record.enrollment_expires_at = utc_now() - timedelta(seconds=1)

    response = claim_endpoint(client, endpoint["id"], code, "trainer")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "node_enrollment_expired"


def test_enrollment_code_is_bound_to_endpoint(client: TestClient) -> None:
    _, first_code = create_pending_endpoint(client, "trainer", "first")
    second, _ = create_pending_endpoint(client, "trainer", "second")

    response = claim_endpoint(
        client, second["id"], first_code, "trainer", "second"
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "node_enrollment_invalid"


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("name", "different-name"),
        ("kind", "converter"),
        ("accelerator", "cuda"),
        ("capabilities", ["deeplabv3plus"]),
    ],
)
def test_claim_rejects_endpoint_identity_mismatch(
    client: TestClient, changed_field: str, changed_value: object
) -> None:
    endpoint, code = create_pending_endpoint(client, "trainer")
    payload = claim_payload("trainer")
    payload[changed_field] = changed_value

    response = client.post(
        f"/api/v1/node-enrollments/{endpoint['id']}/claim",
        json={"enrollmentToken": code, **payload},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "node_enrollment_identity_mismatch"


def test_activated_endpoint_rejects_later_claim(client: TestClient) -> None:
    endpoint, code = create_pending_endpoint(client, "trainer")
    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        record = session.get(ServiceEndpointRecord, cast(str, endpoint["id"]))
        assert record is not None
        record.enrollment_status = "enrolled"
        record.enrollment_token_hash = None
        record.enrollment_expires_at = None

    response = claim_endpoint(client, endpoint["id"], code, "trainer")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "node_enrollment_already_active"
