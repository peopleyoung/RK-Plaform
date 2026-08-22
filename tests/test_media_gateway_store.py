from __future__ import annotations

import stat
from pathlib import Path

import pytest
from backend.platform_api.database import Database
from backend.platform_api.db_models import (
    InferenceMediaBindingRecord,
    MediaCredentialRecord,
    MediaGatewayRecord,
)
from backend.platform_api.media_contracts import (
    MediaCredentialRole,
    MediaGatewayCreate,
    MediaGatewayStatus,
    media_gateway_response,
)
from backend.platform_api.media_secrets import MediaSecretStore


def gateway_payload() -> dict[str, object]:
    return {
        "name": "Built-in media",
        "enabled": True,
        "publishHost": "192.168.1.10",
        "rtspPort": 8554,
        "playbackHost": "media.lan",
        "wsPort": 8081,
        "apiHost": "media",
        "apiPort": 80,
        "app": "live",
        "apiSecret": "api-secret-value",
        "hookIdentity": "hook-identity-value",
    }


def test_gateway_contract_separates_origins_and_redacts_secrets() -> None:
    payload = MediaGatewayCreate.model_validate(gateway_payload())
    record = MediaGatewayRecord(
        id="gateway_builtin",
        name=payload.name,
        builtin=True,
        enabled=payload.enabled,
        publish_host=payload.publish_host,
        rtsp_port=payload.rtsp_port,
        playback_host=payload.playback_host,
        ws_port=payload.ws_port,
        api_host=payload.api_host,
        api_port=payload.api_port,
        app=payload.app,
        status=MediaGatewayStatus.DISABLED.value,
    )

    response = media_gateway_response(
        record,
        api_secret_configured=True,
        hook_identity_configured=True,
    ).model_dump(mode="json", by_alias=True)

    assert response["publishHost"] == "192.168.1.10"
    assert response["playbackHost"] == "media.lan"
    assert response["apiHost"] == "media"
    assert response["apiSecretConfigured"] is True
    assert response["hookIdentityConfigured"] is True
    assert "apiSecret" not in response
    assert "hookIdentity" not in response


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("publishHost", "http://192.168.1.10"),
        ("playbackHost", "media/path"),
        ("apiHost", "user:password@media"),
        ("rtspPort", 0),
        ("wsPort", 65536),
        ("app", "live/channel"),
    ],
)
def test_gateway_contract_rejects_unmanaged_origins(field: str, value: object) -> None:
    payload = gateway_payload()
    payload[field] = value

    with pytest.raises(ValueError):
        MediaGatewayCreate.model_validate(payload)


def test_media_secret_store_uses_restricted_files_and_separate_purposes(
    tmp_path: Path,
) -> None:
    store = MediaSecretStore(tmp_path / "media-secrets")

    store.write_api_secret("gateway_builtin", "api-secret")
    store.write_hook_identity("gateway_builtin", "hook-identity")
    store.write_publication_token("credential_01", "publication-token")

    assert store.api_secret("gateway_builtin") == "api-secret"
    assert store.hook_identity("gateway_builtin") == "hook-identity"
    assert store.publication_token("credential_01") == "publication-token"
    assert store.configured("gateway_builtin") == (True, True)
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert {
        stat.S_IMODE(path.stat().st_mode) for path in store.root.iterdir()
    } == {0o600}

    store.delete_publication_token("credential_01")
    store.delete("gateway_builtin")
    assert store.publication_token("credential_01") is None
    assert store.configured("gateway_builtin") == (False, False)


def test_media_records_persist_hashes_and_stream_projection(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'media.db'}")
    database.create_schema()

    with database.session() as session:
        gateway = MediaGatewayRecord(
                id="gateway_01",
                name="Media 01",
                builtin=False,
                enabled=True,
                publish_host="192.0.2.10",
                rtsp_port=8554,
                playback_host="192.0.2.11",
                ws_port=8081,
                api_host="192.0.2.12",
                api_port=80,
                app="live",
                status=MediaGatewayStatus.PROBING.value,
            )
        session.add(gateway)
        session.flush()
        session.add(
            MediaCredentialRecord(
                id="credential_01",
                token_hash="a" * 64,
                role=MediaCredentialRole.PUBLISH.value,
                gateway_id="gateway_01",
                task_id="task_01",
                revision=3,
                app="live",
                stream_name="camera_01",
                principal="node_01",
            )
        )
        session.add(
            InferenceMediaBindingRecord(
                task_id="task_01",
                gateway_id="gateway_01",
                app="live",
                stream_name="camera_01",
            )
        )

    with database.session() as session:
        credential = session.get(MediaCredentialRecord, "credential_01")
        binding = session.get(InferenceMediaBindingRecord, "task_01")
        assert credential is not None
        assert credential.token_hash == "a" * 64
        assert not hasattr(credential, "token")
        assert binding is not None
        assert binding.stream_name == "camera_01"
