from __future__ import annotations

import json
import socket
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
from workers.node_service.config import NodeServiceSettings
from workers.node_service.enrollment import resolve_node_token

LONG_LIVED_TOKEN = "long-lived-node-token-with-48-characters-value"
ENROLLMENT_TOKEN = "one-time-enrollment-token-with-32-characters"


class ClaimServer(ThreadingHTTPServer):
    actions: list[tuple[int, dict[str, Any] | bytes] | None]
    requests: list[dict[str, Any]]


class ClaimHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server = cast(ClaimServer, self.server)
        content_length = int(self.headers.get("Content-Length", "0"))
        server.requests.append(json.loads(self.rfile.read(content_length)))
        action = server.actions.pop(0)
        if action is None:
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return

        status_code, payload = action
        content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def claim_server(
    *actions: tuple[int, dict[str, Any] | bytes] | None,
) -> Iterator[ClaimServer]:
    server = ClaimServer(("127.0.0.1", 0), ClaimHandler)
    server.actions = list(actions)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def enrollment_settings(
    tmp_path: Path,
    server: ClaimServer,
    *,
    enrollment_file: Path | None = None,
    token_file: Path | None = None,
    token: str | None = None,
) -> NodeServiceSettings:
    return NodeServiceSettings(
        token=token,
        name="edge-converter-01",
        kind="converter",
        accelerator="rk3588",
        capabilities=("yolo-detect", "ppocr-rec"),
        work_dir=tmp_path / "jobs",
        endpoint_id="service_converter_01",
        platform_url=f"http://127.0.0.1:{server.server_address[1]}",
        enrollment_token_file=enrollment_file or tmp_path / "enrollment-token",
        node_token_file=token_file or tmp_path / "state" / "node-token",
        request_timeout_seconds=2,
        version="2.0.0",
        max_concurrency=2,
        features=("rknn",),
        require_accelerator_device=False,
    )


def successful_claim() -> tuple[int, dict[str, Any]]:
    return (
        200,
        {
            "endpointId": "service_converter_01",
            "nodeToken": LONG_LIVED_TOKEN,
            "enrollmentStatus": "claimed",
        },
    )


def test_claims_and_atomically_persists_node_token(tmp_path: Path) -> None:
    enrollment_file = tmp_path / "enrollment-token"
    token_file = tmp_path / "state" / "node-token"
    enrollment_file.write_text(f"{ENROLLMENT_TOKEN}\n", encoding="utf-8")

    with claim_server(successful_claim()) as server:
        settings = enrollment_settings(
            tmp_path,
            server,
            enrollment_file=enrollment_file,
            token_file=token_file,
        )
        token = resolve_node_token(settings)

    assert token == LONG_LIVED_TOKEN
    assert token_file.read_text(encoding="utf-8") == f"{LONG_LIVED_TOKEN}\n"
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert server.requests == [
        {
            "enrollmentToken": ENROLLMENT_TOKEN,
            "name": "edge-converter-01",
            "kind": "converter",
            "accelerator": "rk3588",
            "capabilities": ["yolo-detect", "ppocr-rec"],
            "version": "2.0.0",
            "maxConcurrency": 2,
            "features": ["rknn"],
            "diagnostics": {},
        }
    ]


def test_persistent_token_file_wins_over_legacy_env_and_enrollment(tmp_path: Path) -> None:
    token_file = tmp_path / "node-token"
    token_file.write_text("persistent-node-token-with-32-characters\n", encoding="utf-8")
    token_file.chmod(0o600)

    with claim_server(successful_claim()) as server:
        settings = enrollment_settings(
            tmp_path,
            server,
            token_file=token_file,
            token="legacy-node-token-with-32-characters",
        )
        token = resolve_node_token(settings)

    assert token == "persistent-node-token-with-32-characters"
    assert server.requests == []


def test_legacy_node_token_is_used_without_enrollment(tmp_path: Path) -> None:
    with claim_server(successful_claim()) as server:
        settings = enrollment_settings(
            tmp_path,
            server,
            token="legacy-node-token-with-32-characters",
        )
        token = resolve_node_token(settings)

    assert token == "legacy-node-token-with-32-characters"
    assert server.requests == []


def test_restart_uses_persistent_token_after_enrollment_secret_is_removed(
    tmp_path: Path,
) -> None:
    enrollment_file = tmp_path / "enrollment-token"
    enrollment_file.write_text(ENROLLMENT_TOKEN, encoding="utf-8")

    with claim_server(successful_claim()) as server:
        settings = enrollment_settings(tmp_path, server, enrollment_file=enrollment_file)
        assert resolve_node_token(settings) == LONG_LIVED_TOKEN
        enrollment_file.unlink()
        assert resolve_node_token(settings) == LONG_LIVED_TOKEN

    assert len(server.requests) == 1


def test_startup_fails_without_any_credential_source(tmp_path: Path) -> None:
    with claim_server(successful_claim()) as server:
        settings = enrollment_settings(tmp_path, server)
        with pytest.raises(ValueError, match="node credential source"):
            resolve_node_token(settings)

    assert server.requests == []


def test_short_persistent_token_is_rejected(tmp_path: Path) -> None:
    token_file = tmp_path / "node-token"
    token_file.write_text("too-short\n", encoding="utf-8")

    with claim_server(successful_claim()) as server:
        settings = enrollment_settings(tmp_path, server, token_file=token_file)
        with pytest.raises(ValueError, match="stored node Token"):
            resolve_node_token(settings)

    assert server.requests == []


def test_response_loss_can_be_retried_with_the_same_enrollment_token(tmp_path: Path) -> None:
    enrollment_file = tmp_path / "enrollment-token"
    enrollment_file.write_text(ENROLLMENT_TOKEN, encoding="utf-8")

    with claim_server(None, successful_claim()) as server:
        settings = enrollment_settings(tmp_path, server, enrollment_file=enrollment_file)
        with pytest.raises(ValueError, match="node enrollment request failed"):
            resolve_node_token(settings)
        assert resolve_node_token(settings) == LONG_LIVED_TOKEN

    assert len(server.requests) == 2
    assert server.requests[0] == server.requests[1]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ((200, b"{"), "malformed JSON"),
        ((200, {"endpointId": "service_converter_01"}), "nodeToken"),
    ],
)
def test_malformed_claim_response_is_rejected(
    tmp_path: Path,
    response: tuple[int, dict[str, Any] | bytes],
    message: str,
) -> None:
    enrollment_file = tmp_path / "enrollment-token"
    enrollment_file.write_text(ENROLLMENT_TOKEN, encoding="utf-8")

    with claim_server(response) as server:
        settings = enrollment_settings(tmp_path, server, enrollment_file=enrollment_file)
        with pytest.raises(ValueError, match=message):
            resolve_node_token(settings)


def test_claim_error_propagates_safe_code_without_enrollment_secret(tmp_path: Path) -> None:
    enrollment_file = tmp_path / "enrollment-token"
    enrollment_file.write_text(ENROLLMENT_TOKEN, encoding="utf-8")
    response = (
        401,
        {
            "detail": {
                "code": "node_enrollment_expired",
                "message": "Enrollment credential expired",
            }
        },
    )

    with claim_server(response) as server:
        settings = enrollment_settings(tmp_path, server, enrollment_file=enrollment_file)
        with pytest.raises(ValueError) as captured:
            resolve_node_token(settings)

    assert "node_enrollment_expired" in str(captured.value)
    assert ENROLLMENT_TOKEN not in str(captured.value)


def test_settings_do_not_require_legacy_node_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "RKNODE_NODE_TOKEN",
        "RKNODE_ENDPOINT_ID",
        "RKNODE_PLATFORM_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RKNODE_NODE_NAME", "trainer-01")
    monkeypatch.setenv("RKNODE_NODE_KIND", "trainer")
    monkeypatch.setenv("RKNODE_NODE_ACCELERATOR", "cpu")
    monkeypatch.setenv("RKNODE_NODE_CAPABILITIES", "yolo-detect")
    monkeypatch.setenv("RKNODE_NODE_WORK_DIR", str(tmp_path / "jobs"))

    settings = NodeServiceSettings.from_env()

    assert settings.token is None
