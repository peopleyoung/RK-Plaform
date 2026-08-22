from __future__ import annotations

from datetime import timedelta
from typing import cast

from backend.platform_api.context import AppContext
from backend.platform_api.db_models import MediaCredentialRecord, MediaGatewayRecord
from backend.platform_api.media_contracts import MediaGatewayStatus
from backend.platform_api.media_service import MediaService
from backend.platform_api.state_machine import utc_now
from fastapi.testclient import TestClient
from sqlalchemy import select

from .conftest import ADMIN_HEADERS
from .test_media_gateway_api import gateway_payload


def online_gateway(client: TestClient) -> tuple[AppContext, dict[str, object]]:
    gateway = client.post(
        "/api/v1/media-gateways", headers=ADMIN_HEADERS, json=gateway_payload()
    ).json()
    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        record = session.get(MediaGatewayRecord, cast(str, gateway["id"]))
        assert record is not None
        record.status = MediaGatewayStatus.ONLINE.value
        record.last_hook_at = utc_now()
    return context, gateway


def hook_payload(token_name: str, token: str) -> dict[str, object]:
    return {
        "mediaServerId": "hook-identity-value",
        "app": "live",
        "stream": "camera_01",
        "schema": "rtsp" if token_name == "publishToken" else "http",
        "params": f"{token_name}={token}",
        "vhost": "__defaultVhost__",
        "ip": "192.0.2.50",
        "port": 54321,
        "id": "zlm-session-id",
    }


def test_publish_and_play_tokens_are_role_bound_and_hash_only(client: TestClient) -> None:
    context, gateway = online_gateway(client)
    service = MediaService(context)
    publish_token = service.issue_publish_credential(
        gateway_id=cast(str, gateway["id"]),
        task_id="task_01",
        revision=7,
        app="live",
        stream_name="camera_01",
        principal="node_01",
    )
    play_token, _ = service.issue_play_credential(
        gateway_id=cast(str, gateway["id"]),
        task_id="task_01",
        revision=7,
        app="live",
        stream_name="camera_01",
        principal="admin",
    )

    publish = client.post(
        f"/api/v1/media-hooks/zlm/{gateway['id']}/on-publish",
        json=hook_payload("publishToken", publish_token),
    )
    assert publish.json() == {"code": 0, "msg": "success"}

    publisher_cannot_play = client.post(
        f"/api/v1/media-hooks/zlm/{gateway['id']}/on-play",
        json=hook_payload("playToken", publish_token),
    )
    player_cannot_publish = client.post(
        f"/api/v1/media-hooks/zlm/{gateway['id']}/on-publish",
        json=hook_payload("publishToken", play_token),
    )
    assert publisher_cannot_play.json()["code"] != 0
    assert player_cannot_publish.json()["code"] != 0

    with context.database.session() as session:
        credentials = session.scalars(select(MediaCredentialRecord)).all()
        assert len(credentials) == 2
        assert all(record.token_hash not in {publish_token, play_token} for record in credentials)
        assert all(not hasattr(record, "token") for record in credentials)


def test_play_token_is_consumed_once_and_expires(client: TestClient) -> None:
    context, gateway = online_gateway(client)
    service = MediaService(context)
    play_token, _ = service.issue_play_credential(
        gateway_id=cast(str, gateway["id"]),
        task_id="task_01",
        revision=7,
        app="live",
        stream_name="camera_01",
        principal="admin",
    )
    path = f"/api/v1/media-hooks/zlm/{gateway['id']}/on-play"

    accepted = client.post(path, json=hook_payload("playToken", play_token))
    replayed = client.post(path, json=hook_payload("playToken", play_token))
    assert accepted.json() == {"code": 0, "msg": "success"}
    assert replayed.json() == {"code": -1, "msg": "denied"}

    expired_token, _ = service.issue_play_credential(
        gateway_id=cast(str, gateway["id"]),
        task_id="task_01",
        revision=7,
        app="live",
        stream_name="camera_01",
        principal="admin",
    )
    with context.database.session() as session:
        record = session.scalar(
            select(MediaCredentialRecord).where(MediaCredentialRecord.used_at.is_(None))
        )
        assert record is not None
        record.expires_at = utc_now() - timedelta(seconds=1)
    expired = client.post(path, json=hook_payload("playToken", expired_token))
    assert expired.json() == {"code": -1, "msg": "denied"}


def test_hook_identity_failure_denies_without_revealing_reason(client: TestClient) -> None:
    context, gateway = online_gateway(client)
    token = MediaService(context).issue_publish_credential(
        gateway_id=cast(str, gateway["id"]),
        task_id="task_01",
        revision=7,
        app="live",
        stream_name="camera_01",
        principal="node_01",
    )
    payload = hook_payload("publishToken", token)
    payload["mediaServerId"] = "wrong-hook-identity"

    response = client.post(
        f"/api/v1/media-hooks/zlm/{gateway['id']}/on-publish", json=payload
    )
    assert response.status_code == 200
    assert response.json() == {"code": -1, "msg": "denied"}
