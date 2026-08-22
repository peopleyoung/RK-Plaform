from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from workers.common.config import WorkerConfig
from workers.common.runtime import WorkerRuntime
from workers.node_service.app import InferenceController, create_node_app
from workers.node_service.config import NodeServiceSettings
from workers.node_service.factory import build_runtime


class FakeRuntime:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def run_job(self, job_id: str) -> bool:
        self.job_ids.append(job_id)
        return True


class FakeInferenceController:
    def __init__(self) -> None:
        self.revisions: list[int] = []

    def preflight(self) -> bool:
        return True

    def apply(self, revision: int, payload: dict[str, Any]) -> dict[str, Any]:
        self.revisions.append(revision)
        return {"accepted": True, "actualRevision": revision}

    def status(self) -> dict[str, Any]:
        return {"configured": True, "actualRevision": self.revisions[-1] if self.revisions else 0}


def settings(
    tmp_path: Path,
    *,
    kind: str = "trainer",
    features: tuple[str, ...] = (),
    accelerator: str | None = None,
    capabilities: tuple[str, ...] | None = None,
) -> NodeServiceSettings:
    return NodeServiceSettings(
        token="test-node-token-with-32-characters",
        name=f"test-{kind}",
        kind=kind,
        accelerator=accelerator or ("rk3588" if kind != "trainer" else "cpu"),
        capabilities=capabilities
        or (("deeplab_logits_v1",) if kind == "inference" else ("yolo-detect",)),
        work_dir=tmp_path / "jobs",
        require_accelerator_device=False,
        features=features,
    )


def headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-node-token-with-32-characters"}


def configure_worker_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RKNODE_WORKER_NAME", "test-converter")
    monkeypatch.setenv("RKNODE_WORKER_KIND", "converter")
    monkeypatch.setenv("RKNODE_WORKER_ACCELERATOR", "rk3588")
    monkeypatch.setenv("RKNODE_WORKER_CAPABILITIES", "yolo-detect")
    monkeypatch.setenv("RKNODE_WORK_DIR", str(tmp_path / "jobs"))


def test_node_service_requires_token_and_dispatches_idempotently(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    app = create_node_app(settings(tmp_path), runtime=cast(WorkerRuntime, runtime))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 401
        health = client.get("/health", headers=headers())
        assert health.status_code == 200
        assert health.json()["kind"] == "trainer"

        first = client.post("/api/v1/jobs/train_123/dispatch", headers=headers())
        assert first.status_code == 202
        for _ in range(50):
            state = client.get("/api/v1/jobs/train_123", headers=headers()).json()
            if state["state"] == "succeeded":
                break
            time.sleep(0.01)
        assert state["state"] == "succeeded"
        second = client.post("/api/v1/jobs/train_123/dispatch", headers=headers())
        assert second.status_code == 202
        assert runtime.job_ids == ["train_123"]


def test_worker_config_accepts_resolved_node_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_worker_identity(monkeypatch, tmp_path)
    monkeypatch.delenv("RKNODE_WORKER_TOKEN", raising=False)

    config = WorkerConfig.from_env(token_override="resolved-node-token-with-32-characters")

    assert config.token == "resolved-node-token-with-32-characters"


def test_legacy_worker_token_still_loads_without_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_worker_identity(monkeypatch, tmp_path)
    monkeypatch.setenv("RKNODE_WORKER_TOKEN", "legacy-worker-token-with-32-characters")

    assert WorkerConfig.from_env().token == "legacy-worker-token-with-32-characters"


def test_build_runtime_injects_resolved_node_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_worker_identity(monkeypatch, tmp_path)
    monkeypatch.delenv("RKNODE_WORKER_TOKEN", raising=False)
    monkeypatch.setattr("workers.converter.executor.ConversionExecutor", lambda: object())
    node_settings = settings(tmp_path, kind="converter")

    runtime = build_runtime(node_settings)

    assert runtime is not None
    assert runtime.config.token == node_settings.token


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "other-converter"),
        ("kind", "trainer"),
        ("accelerator", "cpu"),
        ("capabilities", ("other-profile",)),
    ],
)
def test_build_runtime_rejects_worker_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    configure_worker_identity(monkeypatch, tmp_path)
    base_config = WorkerConfig.from_env(
        token_override="resolved-node-token-with-32-characters"
    )
    mismatched_config = replace(base_config, **{field: value})
    monkeypatch.setattr(
        WorkerConfig,
        "from_env",
        classmethod(lambda cls, **kwargs: mismatched_config),
    )

    with pytest.raises(ValueError, match=r"RKNODE_WORKER_\* settings must match"):
        build_runtime(settings(tmp_path, kind="converter"))


def test_paddle_cuda_health_uses_paddle_runtime(monkeypatch: Any, tmp_path: Path) -> None:
    paddle = SimpleNamespace(
        device=SimpleNamespace(
            is_compiled_with_cuda=lambda: True,
            cuda=SimpleNamespace(device_count=lambda: 2),
        )
    )

    def import_module(name: str) -> object:
        assert name == "paddle"
        return paddle

    monkeypatch.setattr("workers.node_service.app.importlib.import_module", import_module)
    app = create_node_app(
        settings(
            tmp_path,
            accelerator="cuda",
            capabilities=("ppocr-det", "ppocr-rec"),
        ),
        runtime=cast(WorkerRuntime, FakeRuntime()),
    )

    with TestClient(app) as client:
        response = client.get("/health", headers=headers())

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["diagnostics"] == {
        "workDir": str(tmp_path / "jobs"),
        "cudaFramework": "paddle",
        "cudaAvailable": True,
        "cudaDeviceCount": 2,
    }


def test_node_service_cleans_terminal_job_workspace(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    node_settings = settings(tmp_path)
    workspace = node_settings.work_dir / "train_456"
    workspace.mkdir(parents=True)
    (workspace / "checkpoint.pt").write_bytes(b"model")
    app = create_node_app(node_settings, runtime=cast(WorkerRuntime, runtime))
    with TestClient(app) as client:
        response = client.delete("/api/v1/jobs/train_456/cache", headers=headers())
    assert response.status_code == 204
    assert not workspace.exists()


def test_inference_node_service_accepts_revision(tmp_path: Path) -> None:
    controller = FakeInferenceController()
    app = create_node_app(
        settings(tmp_path, kind="inference"),
        inference=cast(InferenceController, controller),
    )
    payload = {
        "nodeId": "inode_test",
        "centralApiUrl": "http://platform.test/api/v1",
        "accessToken": "agent-access-token-with-32-characters",
        "desired": {"revision": 4, "releases": [], "tasks": []},
    }
    with TestClient(app) as client:
        mismatch = client.put(
            "/api/v1/inference/revisions/3", headers=headers(), json=payload
        )
        assert mismatch.status_code == 422
        accepted = client.put(
            "/api/v1/inference/revisions/4", headers=headers(), json=payload
        )
    assert accepted.status_code == 200
    assert controller.revisions == [4]


def test_inference_node_service_advertises_media_features(tmp_path: Path) -> None:
    controller = FakeInferenceController()
    app = create_node_app(
        settings(
            tmp_path,
            kind="inference",
            features=("rkmpp_decode", "bytetrack", "kafka", "zlm_sei"),
        ),
        inference=cast(InferenceController, controller),
    )

    with TestClient(app) as client:
        response = client.get("/health", headers=headers())

    assert response.status_code == 200
    assert response.json()["features"] == [
        "rkmpp_decode",
        "bytetrack",
        "kafka",
        "zlm_sei",
    ]
