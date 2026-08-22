from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from workers.inference_agent.agent import AgentSettings, InferenceAgent
from workers.inference_agent.client import InferenceAgentClient


class FakeClient(InferenceAgentClient):
    def __init__(self, source: bytes) -> None:
        super().__init__("http://platform.test/api/v1", access_token="token")
        self.source = source
        self.states: list[str] = []
        self.desired_payload: dict[str, object] = {}
        self.heartbeats: list[dict[str, object]] = []
        self.download_count = 0

    def desired(self, node_id: str) -> dict[str, object]:
        return self.desired_payload

    def heartbeat(self, node_id: str, payload: dict[str, object]) -> dict[str, object]:
        self.heartbeats.append(payload)
        return {}

    def download_artifact(self, node_id: str, artifact_id: str, target: Path) -> str:
        self.download_count += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.source)
        return hashlib.sha256(self.source).hexdigest()

    def report_target(self, node_id: str, target_id: str, payload: dict[str, object]) -> dict[str, object]:
        self.states.append(str(payload["state"]))
        return {}

def test_agent_downloads_verifies_and_reports_all_deployment_stages(tmp_path: Path) -> None:
    content = b"valid-rknn"
    client = FakeClient(content)
    settings = AgentSettings(
        api_url="http://platform.test/api/v1",
        node_id="inode_test",
        registration_token="",
        hardware_id="board",
        runtime_version="runtime",
        driver_version="driver",
        pipeline_version="pipeline",
        adapters=("deeplab_logits_v1",),
        model_dir=tmp_path,
        state_dir=tmp_path / "state",
        poll_seconds=1,
        command="",
    )
    agent = InferenceAgent(settings, client)
    client.desired_payload = {
        "revision": 3,
        "releases": [
            {
                "id": "release_test",
                "adapter": "deeplab_logits_v1",
                "artifact": {
                    "id": "artifact_test",
                    "filename": "model.rknn",
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
                "manifest": {"outputContract": "semantic_logits_nchw_v1"},
            }
        ],
        "tasks": [
            {
                "id": "task_test",
                "deploymentTargetId": "target_test",
                "releaseId": "release_test",
                "inputUri": "rtsp://camera",
            }
        ],
    }
    agent.reconcile_once()
    assert client.states == [
        "downloading",
        "verifying",
        "staged",
        "draining",
        "activating",
        "warming",
        "healthy",
    ]
    assert (tmp_path / "release_test" / "model.rknn").read_bytes() == content
    assert (tmp_path / "release_test" / "manifest.json").is_file()
    assert client.heartbeats[0]["actualRevision"] == 3
    assert (tmp_path / "state" / "actual-revision").read_text() == "3"

    client.desired_payload["revision"] = 4
    agent.reconcile_once()

    assert client.download_count == 1


def test_agent_keeps_actual_revision_when_activation_fails(tmp_path: Path) -> None:
    content = b"invalid-rknn"
    client = FakeClient(content)
    settings = AgentSettings(
        api_url="http://platform.test/api/v1",
        node_id="inode_test",
        registration_token="",
        hardware_id="board",
        runtime_version="runtime",
        driver_version="driver",
        pipeline_version="pipeline",
        adapters=("yolo_dfl_split_v1",),
        model_dir=tmp_path / "models",
        state_dir=tmp_path / "state",
        poll_seconds=1,
        command="",
    )
    agent = InferenceAgent(settings, client)
    client.desired_payload = {
        "revision": 4,
        "releases": [],
        "tasks": [
            {
                "id": "task_test",
                "deploymentTargetId": "target_test",
                "releaseId": "release_missing",
            }
        ],
    }

    agent.reconcile_once()

    assert client.states == ["failed"]
    assert client.heartbeats[0]["actualRevision"] == 0
    assert client.heartbeats[0]["health"] == "degraded"
    assert not (tmp_path / "state" / "actual-revision").exists()
    assert (tmp_path / "state" / "failed-revision").read_text() == "4"

    agent.reconcile_once()

    assert client.states == ["failed"]
    assert client.heartbeats[-1]["actualRevision"] == 0
    assert client.heartbeats[-1]["health"] == "degraded"


def test_agent_activates_one_runtime_context_for_shared_release(
    tmp_path: Path, monkeypatch
) -> None:
    content = b"shared-rknn"
    client = FakeClient(content)
    settings = AgentSettings(
        api_url="http://platform.test/api/v1",
        node_id="inode_test",
        registration_token="",
        hardware_id="board",
        runtime_version="runtime",
        driver_version="driver",
        pipeline_version="pipeline",
        adapters=("yolo_dfl_split_v1",),
        model_dir=tmp_path / "models",
        state_dir=tmp_path / "state",
        poll_seconds=1,
        command="runtime-adapter",
        staging_only=False,
        self_test_command="runtime-self-test",
        probe_command="model-probe",
        health_command="runtime-health",
    )
    agent = InferenceAgent(settings, client)
    client.desired_payload = {
        "revision": 8,
        "releases": [
            {
                "id": "release_shared",
                "adapter": "yolo_dfl_split_v1",
                "artifact": {
                    "id": "artifact_shared",
                    "filename": "model.rknn",
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
                "manifest": {"outputContract": "rknn_yolo_dfl_split_heads_v1"},
            }
        ],
        "tasks": [
            {
                "id": "task_a",
                "deploymentTargetId": "target_a",
                "releaseId": "release_shared",
                "inputUri": "rtsp://camera/a",
            },
            {
                "id": "task_b",
                "deploymentTargetId": "target_b",
                "releaseId": "release_shared",
                "inputUri": "rtsp://camera/b",
            },
        ],
    }
    calls: list[dict[str, str]] = []

    def run(command, *, env, check, timeout):
        calls.append(
            {
                "command": command[0],
                "taskConfigs": env.get("RKNODE_TASK_CONFIGS", ""),
                "releaseConfigs": env.get("RKNODE_RELEASE_CONFIGS", ""),
            }
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("workers.inference_agent.agent.subprocess.run", run)
    agent.reconcile_once()

    assert client.download_count == 1
    assert [item["command"] for item in calls] == [
        "runtime-self-test",
        "model-probe",
        "runtime-adapter",
        "runtime-health",
    ]
    assert [item["id"] for item in json.loads(calls[2]["taskConfigs"])] == [
        "task_a",
        "task_b",
    ]
    release_configs = json.loads(calls[2]["releaseConfigs"])
    assert [item["releaseId"] for item in release_configs] == ["release_shared"]
    assert [item["id"] for item in release_configs[0]["tasks"]] == ["task_a", "task_b"]
    assert client.states.count("healthy") == 2
    assert client.heartbeats[0]["actualRevision"] == 8


def test_agent_activates_multiple_releases_as_one_desired_revision(
    tmp_path: Path, monkeypatch
) -> None:
    content = b"multi-release-rknn"
    checksum = hashlib.sha256(content).hexdigest()
    client = FakeClient(content)
    settings = AgentSettings(
        api_url="http://platform.test/api/v1",
        node_id="inode_test",
        registration_token="",
        hardware_id="board",
        runtime_version="runtime",
        driver_version="driver",
        pipeline_version="pipeline",
        adapters=("deeplab_logits_v1", "ppocr_db_det_v1"),
        model_dir=tmp_path / "models",
        state_dir=tmp_path / "state",
        poll_seconds=1,
        command="runtime-adapter",
        staging_only=False,
        self_test_command="runtime-self-test",
        probe_command="model-probe",
        health_command="runtime-health",
    )
    agent = InferenceAgent(settings, client)
    client.desired_payload = {
        "revision": 9,
        "releases": [
            {
                "id": "release_deeplab",
                "adapter": "deeplab_logits_v1",
                "artifact": {
                    "id": "artifact_deeplab",
                    "filename": "deeplab.rknn",
                    "sha256": checksum,
                },
                "manifest": {"outputContract": "semantic_logits_nchw_v1"},
            },
            {
                "id": "release_ppocr",
                "adapter": "ppocr_db_det_v1",
                "artifact": {
                    "id": "artifact_ppocr",
                    "filename": "ppocr.rknn",
                    "sha256": checksum,
                },
                "manifest": {"outputContract": "ppocr_db_probability_map_v1"},
            },
        ],
        "tasks": [
            {
                "id": "task_deeplab",
                "deploymentTargetId": "target_deeplab",
                "releaseId": "release_deeplab",
                "inputUri": "rtsp://camera/deeplab",
            },
            {
                "id": "task_ppocr",
                "deploymentTargetId": "target_ppocr",
                "releaseId": "release_ppocr",
                "inputUri": "rtsp://camera/ppocr",
            },
        ],
    }
    calls: list[tuple[str, str]] = []

    def run(command, *, env, check, timeout):
        calls.append((command[0], env.get("RKNODE_RELEASE_CONFIGS", "")))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("workers.inference_agent.agent.subprocess.run", run)
    agent.reconcile_once()

    assert [command for command, _ in calls] == [
        "runtime-self-test",
        "model-probe",
        "model-probe",
        "runtime-adapter",
        "runtime-health",
    ]
    release_configs = json.loads(calls[3][1])
    assert [item["releaseId"] for item in release_configs] == [
        "release_deeplab",
        "release_ppocr",
    ]
    assert client.heartbeats[0]["actualRevision"] == 9


def test_agent_applies_empty_revision_to_stop_local_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(b"unused")
    settings = AgentSettings(
        api_url="http://platform.test/api/v1",
        node_id="inode_test",
        registration_token="",
        hardware_id="board",
        runtime_version="runtime",
        driver_version="driver",
        pipeline_version="pipeline",
        adapters=("deeplab_logits_v1",),
        model_dir=tmp_path / "models",
        state_dir=tmp_path / "state",
        poll_seconds=1,
        command="runtime-adapter",
        staging_only=False,
        self_test_command="runtime-self-test",
        probe_command="model-probe",
        health_command="runtime-health",
    )
    agent = InferenceAgent(settings, client)
    client.desired_payload = {"revision": 10, "releases": [], "tasks": []}
    calls: list[tuple[str, str, str]] = []

    def run(command, *, env, check, timeout):
        calls.append(
            (
                command[0],
                env.get("RKNODE_DESIRED_REVISION", ""),
                env.get("RKNODE_RELEASE_CONFIGS", ""),
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("workers.inference_agent.agent.subprocess.run", run)
    agent.reconcile_once()

    assert calls == [
        ("runtime-self-test", "", ""),
        ("runtime-adapter", "10", "[]"),
        ("runtime-health", "10", "[]"),
    ]
    assert client.heartbeats[0]["actualRevision"] == 10


def test_agent_checks_runtime_health_when_revision_is_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    client = FakeClient(b"unused")
    settings = AgentSettings(
        api_url="http://platform.test/api/v1",
        node_id="inode_test",
        registration_token="",
        hardware_id="board",
        runtime_version="runtime",
        driver_version="driver",
        pipeline_version="pipeline",
        adapters=("deeplab_logits_v1",),
        model_dir=tmp_path / "models",
        state_dir=tmp_path / "state",
        poll_seconds=1,
        command="runtime-adapter",
        staging_only=False,
        self_test_command="runtime-self-test",
        probe_command="model-probe",
        health_command="runtime-health",
    )
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "actual-revision").write_text("4", encoding="utf-8")
    agent = InferenceAgent(settings, client)
    client.desired_payload = {"revision": 4, "releases": [], "tasks": []}
    calls: list[str] = []

    def run(command, *, env, check, timeout):
        calls.append(command[0])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("workers.inference_agent.agent.subprocess.run", run)
    agent.reconcile_once()

    assert calls == ["runtime-self-test", "runtime-health"]
    assert client.heartbeats[0]["health"] == "healthy"


def test_production_agent_refuses_to_deploy_without_runtime_self_test(tmp_path: Path) -> None:
    client = FakeClient(b"unused")
    settings = AgentSettings(
        api_url="http://platform.test/api/v1",
        node_id="inode_test",
        registration_token="",
        hardware_id="board",
        runtime_version="runtime",
        driver_version="driver",
        pipeline_version="pipeline",
        adapters=("yolo_dfl_split_v1",),
        model_dir=tmp_path / "models",
        state_dir=tmp_path / "state",
        poll_seconds=1,
        command="",
        staging_only=False,
    )
    agent = InferenceAgent(settings, client)
    client.desired_payload = {"revision": 1, "releases": [], "tasks": []}

    agent.reconcile_once()

    assert client.states == []
    assert client.heartbeats == [
        {
            "actualRevision": 0,
            "health": "degraded",
            "selfTestPassed": False,
            "runtimeVersion": "runtime",
            "driverVersion": "driver",
            "pipelineVersion": "pipeline",
            "adapters": ["yolo_dfl_split_v1"],
            "metrics": {"desiredRevision": 1},
        }
    ]


def test_agent_reports_media_features_in_heartbeat(tmp_path: Path) -> None:
    client = FakeClient(b"unused")
    settings = AgentSettings(
        api_url="http://platform.test/api/v1",
        node_id="inode_test",
        registration_token="",
        hardware_id="board",
        runtime_version="runtime",
        driver_version="driver",
        pipeline_version="pipeline",
        adapters=("yolo_dfl_split_v1",),
        model_dir=tmp_path / "models",
        state_dir=tmp_path / "state",
        poll_seconds=1,
        command="",
        features=("rkmpp_decode", "bytetrack", "kafka", "zlm_sei"),
        staging_only=True,
    )
    agent = InferenceAgent(settings, client)
    agent.self_test_passed = True

    agent._heartbeat(2, health="healthy")

    assert client.heartbeats[0]["metadata"] == {
        "features": ["rkmpp_decode", "bytetrack", "kafka", "zlm_sei"]
    }
