#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import uvicorn

from workers.node_service.app import create_node_app
from workers.node_service.config import NodeServiceSettings
from workers.node_service.enrollment import resolve_node_token

ROOT = Path(__file__).resolve().parents[1]
ADMIN_TOKEN = "enrollment-e2e-admin-token-2026"
WORKER_TOKEN = "enrollment-e2e-worker-token-2026"


class FakeRuntime:
    def run_job(self, _job_id: str) -> bool:
        return False


class FakeInferenceController:
    def __init__(self) -> None:
        self.actual_revision = 0

    def preflight(self) -> bool:
        return True

    def apply(self, revision: int, _payload: dict[str, Any]) -> dict[str, Any]:
        self.actual_revision = revision
        return {"accepted": True, **self.status()}

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.actual_revision > 0,
            "actualRevision": self.actual_revision,
            "applyingRevision": None,
            "lastError": None,
        }


def _install_fake_rknn_toolkit() -> None:
    package = types.ModuleType("rknn")
    package.__path__ = []  # type: ignore[attr-defined]
    api = types.ModuleType("rknn.api")
    package.api = api  # type: ignore[attr-defined]
    sys.modules["rknn"] = package
    sys.modules["rknn.api"] = api


def serve_node() -> None:
    sys.path.insert(0, str(ROOT))
    from workers.node_service.app import create_node_app
    from workers.node_service.config import NodeServiceSettings
    from workers.node_service.enrollment import resolve_node_token

    settings = NodeServiceSettings.from_env()
    if settings.kind == "converter":
        _install_fake_rknn_toolkit()
    settings = replace(settings, token=resolve_node_token(settings))
    app = create_node_app(
        settings,
        runtime=FakeRuntime() if settings.kind != "inference" else None,  # type: ignore[arg-type]
        inference=FakeInferenceController() if settings.kind == "inference" else None,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def api_call(
    method: str,
    url: str,
    *,
    payload: dict[str, object] | None = None,
    token: str | None = None,
) -> tuple[int, dict[str, Any] | list[dict[str, Any]] | None]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=5) as response:
            content = response.read()
            status_code = response.status
    except HTTPError as error:
        content = error.read()
        status_code = error.code
    decoded = json.loads(content) if content else None
    return status_code, decoded


def require_json(
    method: str,
    url: str,
    *,
    payload: dict[str, object] | None = None,
    token: str | None = None,
    expected: int = 200,
) -> dict[str, Any] | list[dict[str, Any]]:
    status_code, body = api_call(method, url, payload=payload, token=token)
    if status_code != expected or body is None:
        raise RuntimeError(f"{method} {url} returned {status_code}: {body}")
    return body


def wait_for_api(base_url: str, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status_code, _ = api_call(
                "GET", f"{base_url}/api/v1/service-endpoints", token=ADMIN_TOKEN
            )
            if status_code == 200:
                return
        except (OSError, URLError, TimeoutError) as error:
            last_error = error
        time.sleep(0.2)
    raise RuntimeError(f"platform API did not start: {last_error}")


def wait_for_enrolled_nodes(base_url: str, endpoint_ids: set[str]) -> None:
    deadline = time.monotonic() + 30
    last_state: dict[str, tuple[object, object]] = {}
    while time.monotonic() < deadline:
        response = require_json(
            "GET", f"{base_url}/api/v1/service-endpoints", token=ADMIN_TOKEN
        )
        assert isinstance(response, list)
        selected = {
            str(item["id"]): item for item in response if str(item["id"]) in endpoint_ids
        }
        last_state = {
            endpoint_id: (item.get("enrollmentStatus"), item.get("probeStatus"))
            for endpoint_id, item in selected.items()
        }
        if len(selected) == len(endpoint_ids) and all(
            item.get("enrollmentStatus") == "enrolled"
            and item.get("probeStatus") == "online"
            and item.get("tokenConfigured") is True
            and "enrollmentToken" not in item
            for item in selected.values()
        ):
            return
        time.sleep(0.5)
    raise RuntimeError(f"nodes did not become enrolled and online: {last_state}")


def endpoint_payload(kind: str, name: str, port: int) -> dict[str, object]:
    return {
        "name": name,
        "kind": kind,
        "mode": "direct",
        "scheme": "http",
        "host": "127.0.0.1",
        "port": port,
        "accelerator": "cpu" if kind == "trainer" else "rk3588",
        "capabilities": [
            "deeplab_logits_v1" if kind == "inference" else "yolo-detect"
        ],
        "enabled": True,
    }


def start_process(
    command: list[str], env: dict[str, str], log_path: Path
) -> tuple[subprocess.Popen[bytes], Any]:
    log_handle = log_path.open("wb")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_handle


def stop_processes(processes: list[tuple[subprocess.Popen[bytes], Any]]) -> None:
    for process, _ in processes:
        if process.poll() is None:
            process.terminate()
    for process, log_handle in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log_handle.close()


def assert_processes_running(processes: list[tuple[subprocess.Popen[bytes], Any]]) -> None:
    failed = [process.pid for process, _ in processes if process.poll() is not None]
    if failed:
        raise RuntimeError(f"node processes exited early: {failed}")


def run_smoke() -> None:
    python = str(ROOT / ".venv" / "bin" / "python")
    if not Path(python).exists():
        raise RuntimeError(".venv/bin/python is required")

    with tempfile.TemporaryDirectory(prefix="rknode-enrollment-e2e-") as temporary:
        temporary_root = Path(temporary)
        api_port = unused_port()
        base_url = f"http://127.0.0.1:{api_port}"
        shared_env = {**os.environ, "PYTHONPATH": str(ROOT)}
        api_env = {
            **shared_env,
            "RKNODE_DATA_DIR": str(temporary_root / "platform"),
            "RKNODE_ADMIN_TOKEN": ADMIN_TOKEN,
            "RKNODE_WORKER_TOKEN": WORKER_TOKEN,
            "RKNODE_DIRECT_DISPATCH_INTERVAL_SECONDS": "1",
            "RKNODE_DIRECT_NODE_TIMEOUT_SECONDS": "2",
            "RKNODE_NODE_ENROLLMENT_TTL_SECONDS": "60",
            "RKNODE_PUBLIC_API_URL": f"{base_url}/api/v1",
            "RKNODE_MODEL_PROFILES_PATH": str(ROOT / "config" / "model_profiles.json"),
            "RKNODE_ENVIRONMENT": "development",
        }
        api_process = start_process(
            [
                python,
                "-m",
                "uvicorn",
                "backend.platform_api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
                "--log-level",
                "warning",
            ],
            api_env,
            temporary_root / "platform.log",
        )
        node_processes: list[tuple[subprocess.Popen[bytes], Any]] = []
        try:
            wait_for_api(base_url)
            node_specs: list[tuple[str, str, int]] = [
                ("trainer", "e2e-trainer", unused_port()),
                ("converter", "e2e-converter", unused_port()),
                ("inference", "e2e-inference", unused_port()),
            ]
            endpoints: list[dict[str, Any]] = []
            node_environments: list[dict[str, str]] = []
            for kind, name, port in node_specs:
                created = require_json(
                    "POST",
                    f"{base_url}/api/v1/service-endpoints",
                    payload=endpoint_payload(kind, name, port),
                    token=ADMIN_TOKEN,
                    expected=201,
                )
                assert isinstance(created, dict)
                enrollment_token = created.get("enrollmentToken")
                endpoint_id = created.get("id")
                if not isinstance(enrollment_token, str) or not isinstance(endpoint_id, str):
                    raise RuntimeError("platform did not return enrollment credentials")
                node_root = temporary_root / kind
                enrollment_file = node_root / "secrets" / "enrollment-token"
                enrollment_file.parent.mkdir(parents=True)
                enrollment_file.write_text(f"{enrollment_token}\n", encoding="utf-8")
                enrollment_file.chmod(0o600)
                node_environments.append(
                    {
                        **shared_env,
                        "RKNODE_ENDPOINT_ID": endpoint_id,
                        "RKNODE_PLATFORM_URL": base_url,
                        "RKNODE_ENROLLMENT_TOKEN_FILE": str(enrollment_file),
                        "RKNODE_NODE_TOKEN_FILE": str(node_root / "state" / "node-token"),
                        "RKNODE_NODE_NAME": name,
                        "RKNODE_NODE_KIND": kind,
                        "RKNODE_NODE_ACCELERATOR": (
                            "cpu" if kind == "trainer" else "rk3588"
                        ),
                        "RKNODE_NODE_CAPABILITIES": (
                            "deeplab_logits_v1" if kind == "inference" else "yolo-detect"
                        ),
                        "RKNODE_NODE_WORK_DIR": str(node_root / "jobs"),
                        "RKNODE_NODE_HOST": "127.0.0.1",
                        "RKNODE_NODE_PORT": str(port),
                        "RKNODE_NODE_VERSION": "e2e-2026.08.15",
                        "RKNODE_REQUIRE_NPU_DEVICE": "false",
                    }
                )
                endpoints.append(created)

            for environment in node_environments:
                kind = environment["RKNODE_NODE_KIND"]
                node_processes.append(
                    start_process(
                        [python, str(Path(__file__).resolve()), "--node"],
                        environment,
                        temporary_root / f"{kind}.log",
                    )
                )
            endpoint_ids = {str(endpoint["id"]) for endpoint in endpoints}
            wait_for_enrolled_nodes(base_url, endpoint_ids)
            assert_processes_running(node_processes)

            for environment in node_environments:
                token_file = Path(environment["RKNODE_NODE_TOKEN_FILE"])
                if stat.S_IMODE(token_file.stat().st_mode) != 0o600:
                    raise RuntimeError(f"persistent node token mode is not 0600: {token_file}")

            stop_processes(node_processes)
            node_processes = []
            for environment in node_environments:
                Path(environment["RKNODE_ENROLLMENT_TOKEN_FILE"]).unlink()
                kind = environment["RKNODE_NODE_KIND"]
                node_processes.append(
                    start_process(
                        [python, str(Path(__file__).resolve()), "--node"],
                        environment,
                        temporary_root / f"{kind}-restart.log",
                    )
                )
            wait_for_enrolled_nodes(base_url, endpoint_ids)
            assert_processes_running(node_processes)

            expired_port = unused_port()
            expired_name = "e2e-expired"
            expired = require_json(
                "POST",
                f"{base_url}/api/v1/service-endpoints",
                payload=endpoint_payload("trainer", expired_name, expired_port),
                token=ADMIN_TOKEN,
                expected=201,
            )
            assert isinstance(expired, dict)
            database_path = temporary_root / "platform" / "platform.db"
            with sqlite3.connect(database_path) as database:
                database.execute(
                    "UPDATE service_endpoints SET enrollment_expires_at = ? WHERE id = ?",
                    ("2000-01-01 00:00:00.000000", expired["id"]),
                )
                database.commit()
            status_code, error = api_call(
                "POST",
                f"{base_url}/api/v1/node-enrollments/{expired['id']}/claim",
                payload={
                    "enrollmentToken": expired["enrollmentToken"],
                    "name": expired_name,
                    "kind": "trainer",
                    "accelerator": "cpu",
                    "capabilities": ["yolo-detect"],
                    "version": "e2e-2026.08.15",
                    "maxConcurrency": 1,
                    "features": [],
                    "diagnostics": {},
                },
            )
            error_code = (
                error.get("error", {}).get("code") if isinstance(error, dict) else None
            )
            if status_code != 401 or error_code != "node_enrollment_expired":
                raise RuntimeError(
                    f"expired code was not rejected correctly: {status_code} {error}"
                )
            listed = require_json(
                "GET", f"{base_url}/api/v1/service-endpoints", token=ADMIN_TOKEN
            )
            assert isinstance(listed, list)
            expired_state = next(item for item in listed if item["id"] == expired["id"])
            if (
                expired_state["enrollmentStatus"] != "pending"
                or expired_state["probeStatus"] != "unprobed"
                or expired_state["tokenConfigured"] is not False
            ):
                raise RuntimeError(f"expired endpoint state changed unexpectedly: {expired_state}")

            workers = require_json(
                "GET", f"{base_url}/api/v1/workers", token=ADMIN_TOKEN
            )
            assert isinstance(workers, list)
            direct_workers = {
                item["name"]: item["status"]
                for item in workers
                if item["name"] in {"e2e-trainer", "e2e-converter"}
            }
            if direct_workers != {"e2e-trainer": "online", "e2e-converter": "online"}:
                raise RuntimeError(f"direct workers were not activated: {direct_workers}")

            print("PASS: trainer, converter, and inference enrolled and became online")
            print("PASS: persisted 0600 node tokens were reused after restart")
            print("PASS: expired enrollment credential was rejected without probing")
        except Exception:
            for log_path in sorted(temporary_root.glob("*.log")):
                if log_path.exists():
                    print(f"--- {log_path.name} ---", file=sys.stderr)
                    print(log_path.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
            raise
        finally:
            stop_processes(node_processes)
            stop_processes([api_process])


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test unified node enrollment")
    parser.add_argument("--node", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.node:
        serve_node()
    else:
        run_smoke()


if __name__ == "__main__":
    main()
