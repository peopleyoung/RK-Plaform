from __future__ import annotations

from pathlib import Path
from typing import Any

from workers.common.client import PlatformClient, WorkerApiError
from workers.common.config import WorkerConfig
from workers.common.runtime import WorkerRuntime
from workers.common.workspace import WorkspaceCleanup, prune_orphan_workspaces

RETAINED_JOB_ID = "train_" + "1" * 32
ORPHAN_JOB_ID = "train_" + "2" * 32
CONVERSION_ORPHAN_ID = "convert_" + "3" * 32


class _NoJobClient(PlatformClient):
    def __init__(self, retained_job_ids: set[str]) -> None:
        super().__init__("http://platform.test/api/v1", "worker-token")
        self._retained_job_ids = retained_job_ids

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "worker-1"}

    def retained_job_ids(self) -> set[str]:
        return set(self._retained_job_ids)

    def claim(self, worker_id: str) -> dict[str, Any] | None:
        return None


class _UnusedExecutor:
    def execute(
        self,
        claim: dict[str, Any],
        client: PlatformClient,
        workspace: Path,
    ) -> dict[str, Any]:
        raise AssertionError("No job should be executed")


class _StaleWorkerClient(_NoJobClient):
    def __init__(self) -> None:
        super().__init__(set())
        self.register_count = 0
        self.claim_count = 0

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.register_count += 1
        return {"id": f"worker-{self.register_count}"}

    def claim(self, worker_id: str) -> dict[str, Any] | None:
        self.claim_count += 1
        if self.claim_count == 1:
            raise WorkerApiError(404, "not_found", "worker is gone")
        return None


def test_prune_removes_only_safe_orphan_job_directories(tmp_path: Path) -> None:
    work_dir = tmp_path / "jobs"
    work_dir.mkdir()
    retained = work_dir / RETAINED_JOB_ID
    retained.mkdir()
    (retained / "model.bin").write_bytes(b"retained")
    orphan = work_dir / ORPHAN_JOB_ID
    orphan.mkdir()
    (orphan / "dataset.zip").write_bytes(b"orphan-data")
    unrelated = work_dir / "operator-notes"
    unrelated.mkdir()
    valid_name_file = work_dir / CONVERSION_ORPHAN_ID
    valid_name_file.write_text("not a directory")
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = work_dir / ("convert_" + "4" * 32)
    symlink.symlink_to(outside, target_is_directory=True)

    removed = prune_orphan_workspaces(work_dir, {RETAINED_JOB_ID})

    assert removed == [WorkspaceCleanup(ORPHAN_JOB_ID, len(b"orphan-data"))]
    assert not orphan.exists()
    assert retained.is_dir()
    assert unrelated.is_dir()
    assert valid_name_file.is_file()
    assert symlink.is_symlink()
    assert outside.is_dir()


def test_worker_reconciles_deleted_workspaces_before_claim(tmp_path: Path) -> None:
    work_dir = tmp_path / "jobs"
    orphan = work_dir / CONVERSION_ORPHAN_ID
    orphan.mkdir(parents=True)
    (orphan / "model.rknn").write_bytes(b"cached")
    config = WorkerConfig(
        api_url="http://platform.test/api/v1",
        token="worker-token",
        name="test-worker",
        kind="converter",
        accelerator="rk3588",
        capabilities=("deeplabv3plus",),
        work_dir=work_dir,
    )
    runtime = WorkerRuntime(
        config,
        _UnusedExecutor(),
        client=_NoJobClient(retained_job_ids=set()),
    )

    assert runtime.run_once() is False
    assert not orphan.exists()


def test_worker_reconciliation_preserves_locally_active_workspace(tmp_path: Path) -> None:
    work_dir = tmp_path / "jobs"
    active = work_dir / ORPHAN_JOB_ID
    active.mkdir(parents=True)
    (active / "partial-output.bin").write_bytes(b"in-use")
    config = WorkerConfig(
        api_url="http://platform.test/api/v1",
        token="worker-token",
        name="test-worker",
        kind="trainer",
        accelerator="cpu",
        capabilities=("deeplabv3plus",),
        work_dir=work_dir,
    )
    runtime = WorkerRuntime(
        config,
        _UnusedExecutor(),
        client=_NoJobClient(retained_job_ids=set()),
    )
    runtime._active_job_ids.add(ORPHAN_JOB_ID)

    runtime._reconcile_workspaces(force=True)

    assert active.is_dir()


def test_worker_reregisters_when_api_loses_worker_record(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config = WorkerConfig(
        api_url="http://platform.test/api/v1",
        token="worker-token",
        name="test-worker",
        kind="trainer",
        accelerator="cpu",
        capabilities=("deeplabv3plus",),
        work_dir=tmp_path / "jobs",
        poll_seconds=0,
    )
    client = _StaleWorkerClient()
    runtime = WorkerRuntime(config, _UnusedExecutor(), client=client)

    def stop_after_retry(_: float) -> None:
        runtime._stopping = True

    monkeypatch.setattr("workers.common.runtime.time.sleep", stop_after_retry)
    runtime.run_forever()

    assert client.register_count == 2
    assert client.claim_count == 1
    assert runtime.worker_id == "worker-2"
