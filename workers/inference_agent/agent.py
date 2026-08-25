from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .client import InferenceAgentClient

LOGGER = logging.getLogger("rknode.inference-agent")
StagedRelease = tuple[dict[str, Any], Path, Path, list[dict[str, Any]]]
TARGET_STAGES = (
    ("downloading", 10),
    ("verifying", 35),
    ("staged", 55),
    ("draining", 68),
    ("activating", 82),
    ("warming", 94),
    ("healthy", 100),
)


@dataclass(frozen=True)
class AgentSettings:
    api_url: str
    node_id: str
    registration_token: str
    hardware_id: str
    runtime_version: str
    driver_version: str
    pipeline_version: str
    adapters: tuple[str, ...]
    model_dir: Path
    state_dir: Path
    poll_seconds: float
    command: str
    features: tuple[str, ...] = ()
    staging_only: bool = True
    self_test_command: str = ""
    probe_command: str = ""
    health_command: str = ""
    command_timeout_seconds: float = 120
    runtime_state_dir: Path = Path("/data/runtime")

    @classmethod
    def from_env(
        cls,
        *,
        api_url_override: str | None = None,
        node_id_override: str | None = None,
    ) -> AgentSettings:
        api_url = (api_url_override or os.environ.get("RKNODE_API_URL", "")).strip()
        node_id = (node_id_override or os.environ.get("RKNODE_NODE_ID", "")).strip()
        if not api_url or not node_id:
            raise ValueError("RKNODE_API_URL and RKNODE_NODE_ID are required")
        return cls(
            api_url=api_url,
            node_id=node_id,
            registration_token=os.environ.get("RKNODE_REGISTRATION_TOKEN", "").strip(),
            hardware_id=os.environ.get("RKNODE_HARDWARE_ID", node_id).strip(),
            runtime_version=os.environ.get("RKNODE_RUNTIME_VERSION", "unknown").strip(),
            driver_version=os.environ.get("RKNODE_DRIVER_VERSION", "unknown").strip(),
            pipeline_version=os.environ.get("RKNODE_PIPELINE_VERSION", "unknown").strip(),
            adapters=tuple(
                item.strip()
                for item in os.environ.get("RKNODE_ADAPTERS", "").split(",")
                if item.strip()
            ),
            model_dir=Path(os.environ.get("RKNODE_MODEL_DIR", "/var/lib/rknode/models")),
            state_dir=Path(os.environ.get("RKNODE_STATE_DIR", "/var/lib/rknode/state")),
            poll_seconds=max(1.0, float(os.environ.get("RKNODE_POLL_SECONDS", "3"))),
            command=os.environ.get("RKNODE_RUNTIME_COMMAND", "").strip(),
            features=tuple(
                item.strip()
                for item in os.environ.get("RKNODE_MEDIA_FEATURES", "").split(",")
                if item.strip()
            ),
            staging_only=os.environ.get("RKNODE_STAGING_ONLY", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            self_test_command=os.environ.get("RKNODE_SELF_TEST_COMMAND", "").strip(),
            probe_command=os.environ.get("RKNODE_MODEL_PROBE_COMMAND", "").strip(),
            health_command=os.environ.get("RKNODE_RUNTIME_HEALTH_COMMAND", "").strip(),
            command_timeout_seconds=max(
                1.0, float(os.environ.get("RKNODE_RUNTIME_TIMEOUT_SECONDS", "120"))
            ),
            runtime_state_dir=Path(
                os.environ.get("RKNODE_RUNTIME_STATE_DIR", "/data/runtime")
            ),
        )


class InferenceAgent:
    def __init__(self, settings: AgentSettings, client: InferenceAgentClient) -> None:
        self.settings = settings
        self.client = client
        self.settings.model_dir.mkdir(parents=True, exist_ok=True)
        self.settings.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.settings.state_dir.chmod(0o700)
        self.last_revision = self._load_actual_revision()
        self.failed_revision: int | None = self._load_failed_revision()
        self.self_test_passed: bool | None = None

    def run(self) -> None:
        self._ensure_registered()
        while True:
            try:
                self.reconcile_once()
            except Exception:
                LOGGER.exception("reconciliation failed; retrying")
            time.sleep(self.settings.poll_seconds)

    def apply_desired(self, desired: dict[str, Any]) -> bool:
        """Apply one desired revision pushed by the central control plane."""
        revision = int(desired.get("revision", 0))
        if self.self_test_passed is None:
            self.self_test_passed = self._run_self_test()
        if not self.self_test_passed:
            self._heartbeat(revision, health="degraded")
            return False
        if revision == self.last_revision:
            self._heartbeat(revision, health="healthy")
            return True
        if revision == self.failed_revision:
            self._heartbeat(revision, health="degraded")
            return False
        if not self._apply_revision(desired):
            self.failed_revision = revision
            self._persist_failed_revision(revision)
            self._heartbeat(revision, health="degraded")
            return False
        self.last_revision = revision
        self._persist_actual_revision(revision)
        self.failed_revision = None
        (self.settings.state_dir / "failed-revision").unlink(missing_ok=True)
        self._heartbeat(revision, health="healthy")
        return True

    def run_self_test(self) -> bool:
        return self._run_self_test()

    def status(self) -> dict[str, Any]:
        return {
            "configured": True,
            "nodeId": self.settings.node_id,
            "actualRevision": self.last_revision,
            "failedRevision": self.failed_revision,
            "selfTestPassed": self.self_test_passed,
            "runtimeVersion": self.settings.runtime_version,
            "driverVersion": self.settings.driver_version,
            "pipelineVersion": self.settings.pipeline_version,
            "adapters": list(self.settings.adapters),
            "features": list(self.settings.features),
        }

    def reconcile_once(self) -> None:
        desired = self.client.desired(self.settings.node_id)
        revision = int(desired.get("revision", 0))
        if self.self_test_passed is None:
            self.self_test_passed = self._run_self_test()
        if not self.self_test_passed:
            self._heartbeat(revision, health="degraded")
            return
        if revision != self.last_revision:
            if revision == self.failed_revision:
                self._heartbeat(revision, health="degraded")
                return
            LOGGER.info("desired revision changed from %s to %s", self.last_revision, revision)
            if not self._apply_revision(desired):
                self.failed_revision = revision
                self._persist_failed_revision(revision)
                self._heartbeat(revision, health="degraded")
                return
            self.last_revision = revision
            self._persist_actual_revision(revision)
            self.failed_revision = None
            (self.settings.state_dir / "failed-revision").unlink(missing_ok=True)
        elif self.last_revision > 0 and self.settings.health_command:
            try:
                self._run_command(
                    self.settings.health_command,
                    os.environ.copy(),
                    "periodic runtime health check",
                )
            except Exception:
                LOGGER.exception("periodic runtime health check failed")
                self._heartbeat(revision, health="degraded")
                return
        self._heartbeat(revision, health="healthy")

    def _heartbeat(self, desired_revision: int, *, health: str) -> None:
        if self.self_test_passed is None:
            self.self_test_passed = self._run_self_test()
        reported_health = health if self.self_test_passed else "degraded"
        payload: dict[str, object] = {
            "actualRevision": self.last_revision,
            "health": reported_health,
            "selfTestPassed": self.self_test_passed,
            "runtimeVersion": self.settings.runtime_version,
            "driverVersion": self.settings.driver_version,
            "pipelineVersion": self.settings.pipeline_version,
            "adapters": list(self.settings.adapters),
            "metrics": {"desiredRevision": desired_revision},
        }
        if self.failed_revision is not None:
            payload["failedRevision"] = self.failed_revision
        if self.settings.features:
            payload["metadata"] = {"features": list(self.settings.features)}
        self.client.heartbeat(
            self.settings.node_id,
            payload,
        )

    def _ensure_registered(self) -> None:
        if self.client.access_token:
            return
        if not self.settings.registration_token:
            raise ValueError("RKNODE_REGISTRATION_TOKEN or a persisted access token is required")
        metadata: dict[str, object] = {"agent": "rknode-inference-agent"}
        if self.settings.features:
            metadata["features"] = list(self.settings.features)
        response = self.client.register(
            {
                "nodeId": self.settings.node_id,
                "registrationToken": self.settings.registration_token,
                "hardwareId": self.settings.hardware_id,
                "runtimeVersion": self.settings.runtime_version,
                "driverVersion": self.settings.driver_version,
                "pipelineVersion": self.settings.pipeline_version,
                "adapters": list(self.settings.adapters),
                "metadata": metadata,
            }
        )
        self.client.access_token = str(response["accessToken"])
        self._write_state_file("access-token", self.client.access_token, mode=0o600)
        LOGGER.info("node registered; access token stored in the protected state directory")

    def _apply_revision(self, desired: dict[str, Any]) -> bool:
        failed = False
        releases: dict[str, dict[str, Any]] = {}
        raw_releases_value = desired.get("releases", [])
        raw_releases = (
            cast(list[Any], raw_releases_value) if isinstance(raw_releases_value, list) else []
        )
        for item in raw_releases:
            if not isinstance(item, dict):
                continue
            typed_item = cast(dict[str, Any], item)
            if typed_item.get("id"):
                releases[str(typed_item["id"])] = typed_item
        raw_tasks_value = desired.get("tasks", [])
        raw_tasks = cast(list[Any], raw_tasks_value) if isinstance(raw_tasks_value, list) else []
        tasks_by_release: dict[
            str, list[tuple[str | None, dict[str, Any], dict[str, Any]]]
        ] = {}
        reference_tasks_by_release: dict[str, list[tuple[str | None, dict[str, Any]]]] = {}
        for raw_task in raw_tasks:
            if not isinstance(raw_task, dict):
                continue
            raw_task = cast(dict[str, Any], raw_task)
            target_id = raw_task.get("deploymentTargetId")
            release_id = raw_task.get("releaseId")
            if target_id is not None and not isinstance(target_id, str):
                failed = True
                continue
            if not isinstance(release_id, str):
                failed = True
                continue
            release = releases.get(release_id)
            if release is None:
                if target_id is not None:
                    self._report_failure(
                        target_id,
                        revision=int(desired["revision"]),
                        code="release_missing",
                        message=release_id,
                    )
                failed = True
                continue
            tasks_by_release.setdefault(release_id, []).append((target_id, raw_task, release))
            reference_tasks_by_release.setdefault(release_id, []).append((target_id, raw_task))
            analytics_value = raw_task.get("analytics", {})
            analytics = (
                cast(dict[str, Any], analytics_value)
                if isinstance(analytics_value, dict)
                else {}
            )
            secondary_value = analytics.get("secondaryModels", [])
            secondary_models = (
                cast(list[Any], secondary_value) if isinstance(secondary_value, list) else []
            )
            for secondary in secondary_models:
                if not isinstance(secondary, dict):
                    failed = True
                    continue
                secondary_config = cast(dict[str, object], secondary)
                secondary_release_id = secondary_config.get("releaseId")
                if not isinstance(secondary_release_id, str):
                    failed = True
                    continue
                if secondary_release_id not in releases:
                    if target_id is not None:
                        self._report_failure(
                            target_id,
                            revision=int(desired["revision"]),
                            code="secondary_release_missing",
                            message=secondary_release_id,
                        )
                    failed = True
                    continue
                reference_tasks_by_release.setdefault(secondary_release_id, []).append(
                    (target_id, raw_task)
                )

        revision = int(desired["revision"])
        staged_releases: list[StagedRelease] = []
        staged_targets: set[str] = set()
        for release_id, release_references in reference_tasks_by_release.items():
            release = releases[release_id]
            target_ids = sorted(
                target_id for target_id, _ in release_references if target_id is not None
            )
            tasks = [task for _, task, _ in tasks_by_release.get(release_id, [])]
            probe_task = release_references[0][1]
            try:
                model_path, manifest_path = self._stage_release(revision, target_ids, release)
                self._run_model_probe(probe_task, release, model_path, manifest_path, tasks)
            except Exception as error:
                LOGGER.exception("release %s failed preflight", release_id)
                for target_id in target_ids:
                    self._report_failure(
                        target_id,
                        revision=revision,
                        code="preflight_failed",
                        message=str(error),
                    )
                failed = True
                continue
            staged_releases.append((release, model_path, manifest_path, tasks))
            staged_targets.update(target_ids)

        if failed:
            for target_id in sorted(staged_targets):
                self._report_failure(
                    target_id,
                    revision=revision,
                    code="revision_preflight_failed",
                    message="Another release in this desired revision failed preflight",
                )
            return False

        for target_id in sorted(staged_targets):
            self._report(target_id, revision, "draining", 68)
        try:
            self._run_revision_command(revision, staged_releases)
        except Exception as error:
            LOGGER.exception("desired revision %s failed to activate", revision)
            for target_id in sorted(staged_targets):
                self._report_failure(
                    target_id,
                    revision=revision,
                    code="activation_failed",
                    message=str(error),
                )
            return False
        for target_id in sorted(staged_targets):
            self._report(target_id, revision, "activating", 82)
            self._report(target_id, revision, "warming", 94)
            self._report(target_id, revision, "healthy", 100)
        return True

    def _load_actual_revision(self) -> int:
        value = self._read_state_file("actual-revision")
        if value is None:
            return 0
        try:
            return max(0, int(value))
        except ValueError:
            LOGGER.warning("ignoring invalid persisted actual revision: %s", value)
            return 0

    def _persist_actual_revision(self, revision: int) -> None:
        self._write_state_file("actual-revision", str(revision), mode=0o600)

    def _load_failed_revision(self) -> int | None:
        value = self._read_state_file("failed-revision")
        if value is None:
            return None
        try:
            return max(0, int(value))
        except ValueError:
            LOGGER.warning("ignoring invalid persisted failed revision: %s", value)
            return None

    def _persist_failed_revision(self, revision: int) -> None:
        self._write_state_file("failed-revision", str(revision), mode=0o600)

    def _read_state_file(self, name: str) -> str | None:
        path = self.settings.state_dir / name
        try:
            value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return value or None

    def _write_state_file(self, name: str, value: str, *, mode: int) -> None:
        path = self.settings.state_dir / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.chmod(mode)
        temporary.replace(path)

    def _stage_release(
        self, revision: int, target_ids: list[str], release: dict[str, Any]
    ) -> tuple[Path, Path]:
        artifact = release.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError("release artifact descriptor is missing")
        artifact = cast(dict[str, Any], artifact)
        artifact_id = str(artifact["id"])
        filename = Path(str(artifact.get("filename", f"{release['id']}.rknn"))).name
        target_dir = self.settings.model_dir / str(release["id"])
        model_path = target_dir / filename
        for target_id in target_ids:
            self._report(target_id, revision, "downloading", 10)
        expected = str(artifact.get("sha256", ""))
        digest = self._sha256(model_path) if model_path.is_file() else ""
        if not expected or digest != expected:
            digest = self.client.download_artifact(self.settings.node_id, artifact_id, model_path)
        for target_id in target_ids:
            self._report(target_id, revision, "verifying", 35)
        if expected and digest != expected:
            raise ValueError(f"artifact checksum mismatch: expected {expected}, got {digest}")
        manifest_path = target_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(release.get("manifest", {}), indent=2), encoding="utf-8"
        )
        for target_id in target_ids:
            self._report(target_id, revision, "staged", 55)
        return model_path, manifest_path

    def _runtime_environment(
        self,
        task: dict[str, Any],
        release: dict[str, Any],
        model_path: Path,
        manifest_path: Path,
        tasks: list[dict[str, Any]],
    ) -> dict[str, str]:
        return {
            **os.environ,
            "RKNODE_TASK_ID": str(task["id"]),
            "RKNODE_RELEASE_ID": str(release["id"]),
            "RKNODE_MODEL_PATH": str(model_path),
            "RKNODE_MANIFEST_PATH": str(manifest_path),
            "RKNODE_ADAPTER": str(release.get("adapter", "")),
            "RKNODE_INPUT_URI": str(task.get("inputUri", "")),
            "RKNODE_TASK_CONFIG": json.dumps(task, separators=(",", ":")),
            "RKNODE_TASK_CONFIGS": json.dumps(tasks, separators=(",", ":")),
        }

    def _run_self_test(self) -> bool:
        if not self.settings.self_test_command:
            if self.settings.staging_only:
                LOGGER.warning("staging-only mode enabled; runtime self-test is bypassed")
                return True
            LOGGER.error("RKNODE_SELF_TEST_COMMAND is required outside staging-only mode")
            return False
        try:
            self._run_command(
                self.settings.self_test_command,
                os.environ.copy(),
                "runtime self-test",
            )
        except Exception:
            LOGGER.exception("runtime self-test failed")
            return False
        return True

    def _run_model_probe(
        self,
        task: dict[str, Any],
        release: dict[str, Any],
        model_path: Path,
        manifest_path: Path,
        tasks: list[dict[str, Any]],
    ) -> None:
        if not self.settings.probe_command:
            if self.settings.staging_only:
                LOGGER.warning("staging-only mode enabled; model probe is bypassed")
                return
            raise RuntimeError("RKNODE_MODEL_PROBE_COMMAND is required outside staging-only mode")
        self._run_command(
            self.settings.probe_command,
            self._runtime_environment(task, release, model_path, manifest_path, tasks),
            "model probe",
        )

    def _run_revision_command(self, revision: int, staged_releases: list[StagedRelease]) -> None:
        if not self.settings.command:
            if self.settings.staging_only:
                LOGGER.warning("staging-only mode enabled; runtime activation is bypassed")
                return
            raise RuntimeError("RKNODE_RUNTIME_COMMAND is required outside staging-only mode")
        if staged_releases:
            release, model_path, manifest_path, tasks = staged_releases[0]
            environment = self._runtime_environment(
                tasks[0], release, model_path, manifest_path, tasks
            )
        else:
            environment = {
                **os.environ,
                "RKNODE_TASK_ID": "",
                "RKNODE_RELEASE_ID": "",
                "RKNODE_MODEL_PATH": "",
                "RKNODE_MANIFEST_PATH": "",
                "RKNODE_ADAPTER": "",
                "RKNODE_INPUT_URI": "",
                "RKNODE_TASK_CONFIG": "{}",
                "RKNODE_TASK_CONFIGS": "[]",
            }
        environment["RKNODE_DESIRED_REVISION"] = str(revision)
        environment["RKNODE_RELEASE_CONFIGS"] = json.dumps(
            [
                {
                    "releaseId": release["id"],
                    "name": release.get("name", ""),
                    "version": release.get("version", ""),
                    "adapter": release.get("adapter", ""),
                    "modelPath": str(model_path),
                    "manifestPath": str(manifest_path),
                    "tasks": tasks,
                }
                for release, model_path, manifest_path, tasks in staged_releases
            ],
            separators=(",", ":"),
        )
        self._run_command(self.settings.command, environment, "runtime activation")
        if self.settings.health_command:
            self._run_command(self.settings.health_command, environment, "runtime health check")
        elif not self.settings.staging_only:
            raise RuntimeError(
                "RKNODE_RUNTIME_HEALTH_COMMAND is required outside staging-only mode"
            )

    def _run_command(self, raw_command: str, environment: dict[str, str], purpose: str) -> None:
        command = shlex.split(raw_command)
        if not command:
            raise RuntimeError(f"{purpose} command is empty")
        LOGGER.info("running %s with %s", purpose, command[0])
        completed = subprocess.run(
            command,
            env=environment,
            check=False,
            timeout=self.settings.command_timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{purpose} exited with {completed.returncode}")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _report(self, target_id: str, revision: int, state: str, progress: int) -> None:
        self.client.report_target(
            self.settings.node_id,
            target_id,
            {"revision": revision, "state": state, "progress": progress, "stage": state},
        )

    def _report_failure(self, target_id: str, revision: int, code: str, message: str) -> None:
        try:
            self.client.report_target(
                self.settings.node_id,
                target_id,
                {
                    "revision": revision,
                    "state": "failed",
                    "progress": 0,
                    "stage": "failed",
                    "errorCode": code,
                    "errorMessage": message[:4000],
                    "message": message[:4000],
                },
            )
        except Exception:
            LOGGER.exception("failed to report deployment target failure")


def build_agent_from_env() -> InferenceAgent:
    settings = AgentSettings.from_env()
    access_token_path = settings.state_dir / "access-token"
    try:
        access_token = access_token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        access_token = ""
    client = InferenceAgentClient(settings.api_url, access_token=access_token)
    return InferenceAgent(settings, client)
