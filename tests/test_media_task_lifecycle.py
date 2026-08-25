from __future__ import annotations

from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from backend.platform_api.context import AppContext
from backend.platform_api.db_models import MediaCredentialRecord, MediaGatewayRecord
from backend.platform_api.media_contracts import MediaGatewayStatus
from backend.platform_api.state_machine import utc_now
from fastapi.testclient import TestClient
from sqlalchemy import select

from .conftest import ADMIN_HEADERS
from .test_inference_api import _published_release, _register_active_node
from .test_media_gateway_api import gateway_payload


def create_online_gateway(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/api/v1/media-gateways", headers=ADMIN_HEADERS, json=gateway_payload()
    )
    assert response.status_code == 201, response.text
    gateway = cast(dict[str, Any], response.json())
    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        record = session.get(MediaGatewayRecord, gateway["id"])
        assert record is not None
        record.status = MediaGatewayStatus.ONLINE.value
        record.last_hook_at = utc_now()
    return gateway


def task_payload(
    release_id: str, node_id: str, gateway_id: str
) -> dict[str, object]:
    return {
        "name": "line-a-media",
        "nodeId": node_id,
        "inputUri": "rtsp://camera/line-a",
        "graph": {
            "schemaVersion": 1,
            "catalogVersion": "2026.08.25",
            "nodes": [
                {"id": "capture", "operator": "capture.rkmpp", "config": {}},
                {
                    "id": "primary",
                    "operator": "inference.primary",
                    "config": {"releaseId": release_id},
                },
                {"id": "json", "operator": "output.json", "config": {}},
                {
                    "id": "zlm",
                    "operator": "output.zlm_sei",
                    "config": {
                        "gatewayId": gateway_id,
                        "streamName": "camera_01",
                        "reconnectMs": 1000,
                    },
                },
            ],
            "edges": [
                {"source": "capture", "target": "primary"},
                {"source": "primary", "target": "json"},
                {"source": "primary", "target": "zlm"},
            ],
        },
        "layout": {"positions": {}},
    }


def test_task_gateway_binding_generates_separate_node_and_browser_credentials(
    client: TestClient,
) -> None:
    release, _ = _published_release(client)
    node_id, access_token = _register_active_node(
        client,
        "deeplab_logits_v1",
        features=["rkmpp_decode", "zlm_sei"],
    )
    gateway = create_online_gateway(client)
    payload = task_payload(release["id"], node_id, gateway["id"])

    created = client.post(
        "/api/v1/inference-tasks", headers=ADMIN_HEADERS, json=payload
    )
    assert created.status_code == 201, created.text
    task = created.json()
    zlm_node = next(node for node in task["graph"]["nodes"] if node["operator"] == "output.zlm_sei")
    assert zlm_node["config"]["gatewayId"] == gateway["id"]
    assert task["previewCapability"] == {"state": "available", "reason": None}

    deployment = client.post(
        "/api/v1/deployments",
        headers=ADMIN_HEADERS,
        json={
            "name": "line-a-media-rollout",
            "taskIds": [task["id"]],
            "strategy": "all_at_once",
        },
    )
    assert deployment.status_code == 201, deployment.text

    desired = client.get(
        f"/api/v1/inference-agent/nodes/{node_id}/desired",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert desired.status_code == 200, desired.text
    desired_media = desired.json()["tasks"][0]["runtimeBindings"]["media"]
    zlm = desired_media["zlmSei"]
    assert set(zlm) == {"enabled", "publishUri", "reconnectMs"}
    publish_uri = urlparse(zlm["publishUri"])
    assert publish_uri.scheme == "rtsp"
    assert publish_uri.hostname == "192.0.2.10"
    assert publish_uri.port == 8554
    publish_token = parse_qs(publish_uri.query)["publishToken"][0]

    playback = client.post(
        f"/api/v1/inference-tasks/{task['id']}/playback-session",
        headers=ADMIN_HEADERS,
    )
    assert playback.status_code == 200, playback.text
    descriptor = playback.json()
    play_uri = urlparse(descriptor["streamUrl"])
    assert play_uri.scheme == "ws"
    assert play_uri.hostname == "192.0.2.11"
    assert play_uri.port == 8081
    play_token = parse_qs(play_uri.query)["playToken"][0]
    assert play_token != publish_token
    assert descriptor["taskId"] == task["id"]
    assert descriptor["revision"] == desired.json()["revision"]

    publish_hook = client.post(
        f"/api/v1/media-hooks/zlm/{gateway['id']}/on-publish",
        json={
            "mediaServerId": "hook-identity-value",
            "app": "live",
            "stream": "camera_01",
            "schema": "rtsp",
            "params": f"publishToken={publish_token}",
        },
    )
    play_hook = client.post(
        f"/api/v1/media-hooks/zlm/{gateway['id']}/on-play",
        json={
            "mediaServerId": "hook-identity-value",
            "app": "live",
            "stream": "camera_01",
            "schema": "http",
            "params": f"playToken={play_token}",
        },
    )
    assert publish_hook.json()["code"] == 0
    assert play_hook.json()["code"] == 0

    stopped = client.post(
        f"/api/v1/inference-tasks/{task['id']}/stop", headers=ADMIN_HEADERS
    )
    assert stopped.status_code == 200, stopped.text
    denied = client.post(
        f"/api/v1/media-hooks/zlm/{gateway['id']}/on-publish",
        json={
            "mediaServerId": "hook-identity-value",
            "app": "live",
            "stream": "camera_01",
            "schema": "rtsp",
            "params": f"publishToken={publish_token}",
        },
    )
    assert denied.json() == {"code": -1, "msg": "denied"}

    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        publisher = session.scalar(
            select(MediaCredentialRecord).where(
                MediaCredentialRecord.task_id == task["id"],
                MediaCredentialRecord.role == "publish",
            )
        )
        assert publisher is not None
        assert publisher.revoked_at is not None


def test_task_rejects_legacy_output_uri_and_offline_gateway(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, _ = _register_active_node(
        client,
        "deeplab_logits_v1",
        features=["rkmpp_decode", "zlm_sei"],
    )
    gateway = create_online_gateway(client)
    base = task_payload(release["id"], node_id, gateway["id"])
    graph = cast(dict[str, Any], base["graph"])
    nodes = cast(list[dict[str, Any]], graph["nodes"])
    legacy = next(node["config"] for node in nodes if node["operator"] == "output.zlm_sei")
    legacy.pop("gatewayId")
    legacy.pop("streamName")
    legacy["outputUri"] = "rtsp://legacy/live/result?token=must-not-leak"

    rejected = client.post(
        "/api/v1/inference-tasks", headers=ADMIN_HEADERS, json=base
    )
    assert rejected.status_code == 409
    assert rejected.json()["error"]["code"] == "inference_graph_invalid"
    assert "must-not-leak" not in rejected.text

    context = cast(AppContext, client.app.state.context)
    with context.database.session() as session:
        record = session.get(MediaGatewayRecord, gateway["id"])
        assert record is not None
        record.status = MediaGatewayStatus.ERROR.value
    unavailable = client.post(
        "/api/v1/inference-tasks",
        headers=ADMIN_HEADERS,
        json=task_payload(release["id"], node_id, gateway["id"]),
    )
    assert unavailable.status_code == 409
    assert unavailable.json()["error"]["code"] == "media_gateway_offline"
