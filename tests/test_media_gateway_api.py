from __future__ import annotations

from typing import cast

from backend.platform_api.context import AppContext
from backend.platform_api.db_models import MediaGatewayRecord
from backend.platform_api.media_contracts import MediaGatewayStatus
from fastapi.testclient import TestClient

from .conftest import ADMIN_HEADERS


def gateway_payload(name: str = "External media") -> dict[str, object]:
    return {
        "name": name,
        "enabled": True,
        "publishHost": "192.0.2.10",
        "rtspPort": 8554,
        "playbackHost": "192.0.2.11",
        "wsPort": 8081,
        "apiHost": "192.0.2.12",
        "apiPort": 80,
        "app": "live",
        "apiSecret": "api-secret-value",
        "hookIdentity": "hook-identity-value",
    }


def test_admin_can_create_list_update_and_delete_media_gateway(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/v1/media-gateways", headers=ADMIN_HEADERS, json=gateway_payload()
    )
    assert created.status_code == 201, created.text
    gateway = created.json()
    assert gateway["status"] == "error"
    assert gateway["apiSecretConfigured"] is True
    assert gateway["hookIdentityConfigured"] is True
    assert "apiSecret" not in gateway
    assert "hookIdentity" not in gateway

    listed = client.get("/api/v1/media-gateways", headers=ADMIN_HEADERS)
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [gateway["id"]]

    updated_payload = gateway_payload("Updated media")
    updated_payload["playbackHost"] = "media.example.lan"
    updated_payload.pop("apiSecret")
    updated_payload.pop("hookIdentity")
    updated = client.put(
        f"/api/v1/media-gateways/{gateway['id']}",
        headers=ADMIN_HEADERS,
        json=updated_payload,
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "Updated media"
    assert updated.json()["playbackHost"] == "media.example.lan"
    assert updated.json()["apiSecretConfigured"] is True

    removed = client.delete(
        f"/api/v1/media-gateways/{gateway['id']}", headers=ADMIN_HEADERS
    )
    assert removed.status_code == 204, removed.text
    assert client.get("/api/v1/media-gateways", headers=ADMIN_HEADERS).json() == []
    context = cast(AppContext, client.app.state.context)
    assert context.media_secrets.configured(gateway["id"]) == (False, False)


def test_gateway_management_requires_admin(client: TestClient) -> None:
    response = client.post("/api/v1/media-gateways", json=gateway_payload())
    assert response.status_code == 401


def test_authenticated_keepalive_is_required_before_gateway_is_online(
    client: TestClient, monkeypatch: object
) -> None:
    created = client.post(
        "/api/v1/media-gateways", headers=ADMIN_HEADERS, json=gateway_payload()
    ).json()

    def server_config(_: object) -> dict[str, object]:
        return {
            "hook.on_publish": (
                f"http://api:8000/api/v1/media-hooks/zlm/{created['id']}/on-publish"
            ),
            "hook.on_play": (
                f"http://api:8000/api/v1/media-hooks/zlm/{created['id']}/on-play"
            ),
        }

    monkeypatch.setattr(
        "backend.platform_api.media_service.ZlmClient.get_server_config",
        server_config,
    )
    first_probe = client.post(
        f"/api/v1/media-gateways/{created['id']}/probe", headers=ADMIN_HEADERS
    )
    assert first_probe.status_code == 200
    assert first_probe.json()["status"] == "error"

    keepalive = client.post(
        f"/api/v1/media-hooks/zlm/{created['id']}/on-server-keepalive",
        json={
            "mediaServerId": "hook-identity-value",
            "hook_index": 7,
            "data": {"Buffer": 512, "MediaSource": 0},
        },
    )
    assert keepalive.status_code == 200
    assert keepalive.json() == {"code": 0, "msg": "success"}

    second_probe = client.post(
        f"/api/v1/media-gateways/{created['id']}/probe", headers=ADMIN_HEADERS
    )
    assert second_probe.status_code == 200
    assert second_probe.json()["status"] == "online"

    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        record = session.get(MediaGatewayRecord, created["id"])
        assert record is not None
        assert record.status == MediaGatewayStatus.ONLINE.value
        assert record.last_hook_at is not None
