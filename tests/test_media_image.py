from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml
from backend.platform_api.app import create_app
from backend.platform_api.settings import Settings
from fastapi.testclient import TestClient
from pydantic import SecretStr

from tests.conftest import ADMIN_HEADERS

ROOT = Path(__file__).resolve().parents[1]
MEDIA_ROOT = ROOT / "deploy" / "media"


def test_zlm_base_image_is_an_immutable_verified_digest() -> None:
    lock = (MEDIA_ROOT / "zlm-base-image.lock").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"zlmediakit/zlmediakit@sha256:[a-f0-9]{64}", lock)
    verification = (MEDIA_ROOT / "zlm-candidate-verification.json").read_text(
        encoding="utf-8"
    )
    assert lock in verification
    assert '"passed": true' in verification

    dockerfile = (MEDIA_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG ZLM_BASE_IMAGE" in dockerfile
    assert "FROM ${ZLM_BASE_IMAGE}" in dockerfile
    assert ":master" not in dockerfile
    assert "ARG RKNODE_RELEASE_VERSION=dev" in dockerfile
    assert 'org.opencontainers.image.version="${RKNODE_RELEASE_VERSION}"' in dockerfile
    assert 'io.rknode.component="platform-media"' in dockerfile
    assert 'io.rknode.offline-ready="true"' in dockerfile
    assert "./MediaServer -s default.pem -c ../conf/config.ini -l 0" in dockerfile


def test_media_template_has_only_required_hooks_and_low_latency_flv() -> None:
    template = (MEDIA_ROOT / "config.ini.template").read_text(encoding="utf-8")
    required = (
        "secret=${RKNODE_ZLM_API_SECRET}",
        "mediaServerId=${RKNODE_ZLM_HOOK_IDENTITY}",
        "on_publish=http://api:8000/api/v1/media-hooks/zlm/gateway_builtin/on-publish",
        "on_play=http://api:8000/api/v1/media-hooks/zlm/gateway_builtin/on-play",
        "on_server_keepalive=http://api:8000/api/v1/media-hooks/zlm/gateway_builtin/on-server-keepalive",
        "enable_hls=0",
        "enable_hls_fmp4=0",
        "enable_mp4=0",
        "enable_rtmp=1",
        "enable_ts=0",
        "enable_fmp4=0",
        "port=554",
        "port=80",
        "allow_cross_domains=1",
    )
    for value in required:
        assert value in template
    assert "admin_params=" not in template
    for hook in (
        "on_flow_report",
        "on_http_access",
        "on_record_mp4",
        "on_record_ts",
        "on_rtsp_auth",
        "on_rtsp_realm",
        "on_shell_login",
        "on_stream_changed",
        "on_stream_none_reader",
        "on_stream_not_found",
    ):
        assert re.search(rf"^{hook}=\s*$", template, re.MULTILINE)


def test_media_renderer_validates_and_never_echoes_secrets(tmp_path: Path) -> None:
    renderer = MEDIA_ROOT / "render_config.py"
    text = renderer.read_text(encoding="utf-8")
    assert "os.replace" in text
    assert "0o600" in text
    assert "print(" not in text

    output = tmp_path / "config.ini"
    environment = {
        **os.environ,
        "RKNODE_ZLM_API_SECRET": "a" * 64,
        "RKNODE_ZLM_HOOK_IDENTITY": "b" * 64,
        "RKNODE_ZLM_CONFIG_TEMPLATE": str(MEDIA_ROOT / "config.ini.template"),
        "RKNODE_ZLM_CONFIG_OUTPUT": str(output),
        "RKNODE_ZLM_RENDER_ONLY": "1",
    }
    subprocess.run(["python3", str(renderer)], check=True, env=environment)
    rendered = output.read_text(encoding="utf-8")
    assert "a" * 64 in rendered
    assert "b" * 64 in rendered
    assert output.stat().st_mode & 0o777 == 0o600


def test_online_compose_owns_media_ports_health_and_secret_boundaries() -> None:
    compose = yaml.safe_load((ROOT / "deploy" / "compose.yaml").read_text())
    services = compose["services"]
    assert set(services) == {"api", "frontend", "media"}
    media = services["media"]
    assert media["image"] == "rknode-platform-media:2026.08.20"
    assert media["ports"] == ["8554:554", "8081:80"]
    assert media["restart"] == "unless-stopped"
    assert "healthcheck" in media
    assert "logging" in media
    assert services["frontend"]["ports"] == ["5173:80"]
    for variable in ("RKNODE_ZLM_API_SECRET", "RKNODE_ZLM_HOOK_IDENTITY"):
        assert variable in services["api"]["environment"]
        assert variable in media["environment"]
        assert variable not in services["frontend"].get("environment", {})


def test_media_secret_configurator_is_idempotent_and_quiet(tmp_path: Path) -> None:
    target = tmp_path / "compose.yaml"
    target.write_text(
        "x-rknode-platform-config:\n"
        "  zlm-api-secret: &zlm-api-secret replace-with-zlm-api-secret\n"
        "  zlm-hook-identity: &zlm-hook-identity replace-with-zlm-hook-identity\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts" / "configure_media_secrets.py"),
            "--compose-file",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""
    assert result.stderr == ""
    content = target.read_text(encoding="utf-8")
    secret = re.search(r"zlm-api-secret: &zlm-api-secret ([a-f0-9]{64})$", content, re.M)
    identity = re.search(r"zlm-hook-identity: &zlm-hook-identity ([a-f0-9]{64})$", content, re.M)
    assert secret is not None
    assert identity is not None
    assert target.stat().st_mode & 0o777 == 0o600
    first = content
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts" / "configure_media_secrets.py"),
            "--compose-file",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert target.read_text(encoding="utf-8") == first


def test_api_bootstraps_builtin_gateway_without_overwriting_operator_identity(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'platform.db'}",
        model_profiles_path=ROOT / "config" / "model_profiles.json",
        admin_token="test-admin-token",
        worker_token="test-worker-token",
        direct_dispatch_enabled=False,
        media_builtin_enabled=True,
        media_publish_host="192.0.2.10",
        media_playback_host="192.0.2.11",
        media_rtsp_port=8554,
        media_ws_port=8081,
        media_api_host="media",
        media_api_port=80,
        zlm_api_secret=SecretStr("a" * 64),
        zlm_hook_identity=SecretStr("b" * 64),
    )
    with TestClient(create_app(settings)) as client:
        gateway = client.get("/api/v1/media-gateways", headers=ADMIN_HEADERS).json()[0]
        assert gateway["id"] == "gateway_builtin"
        assert gateway["builtin"] is True
        changed = {
            "name": "Operator media name",
            "enabled": False,
            "publishHost": "198.51.100.10",
            "rtspPort": 9554,
            "playbackHost": "198.51.100.11",
            "wsPort": 9081,
            "apiHost": "external-api",
            "apiPort": 8080,
            "app": "operator-app",
        }
        assert client.put(
            "/api/v1/media-gateways/gateway_builtin",
            headers=ADMIN_HEADERS,
            json=changed,
        ).status_code == 200

    with TestClient(create_app(settings)) as client:
        gateway = client.get("/api/v1/media-gateways", headers=ADMIN_HEADERS).json()[0]
        assert gateway["name"] == "Operator media name"
        assert gateway["app"] == "operator-app"
        assert gateway["enabled"] is True
        assert gateway["publishHost"] == "192.0.2.10"
        assert gateway["playbackHost"] == "192.0.2.11"
        assert gateway["apiHost"] == "media"
        assert gateway["apiSecretConfigured"] is True
        assert gateway["hookIdentityConfigured"] is True
