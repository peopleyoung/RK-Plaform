from __future__ import annotations

import logging
import multiprocessing
import queue
import signal
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Protocol, cast

from .client import PlatformClient, WorkerApiError
from .config import WorkerConfig
from .workspace import prune_orphan_workspaces

LOGGER = logging.getLogger("rknode.worker")


class JobExecutor(Protocol):
    def execute(
        self,
        claim: dict[str, Any],
        client: PlatformClient,
        workspace: Path,
    ) -> dict[str, Any]: ...


class ResultQueue(Protocol):
    def put(self, value: object) -> None: ...


def _execute_job_process(
    executor: JobExecutor,
    claim: dict[str, Any],
    api_url: str,
    token: str,
    request_timeout_seconds: float,
    workspace: Path,
    result_queue: ResultQueue,
) -> None:
    client = PlatformClient(api_url, token, timeout=request_timeout_seconds)
    try:
        result_queue.put({"ok": True, "result": executor.execute(claim, client, workspace)})
    except Exception as error:
        result_queue.put(
            {
                "ok": False,
                "code": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )


class JobExecutionError(RuntimeError):
    def __init__(self, code: str, message: str, remote_traceback: str) -> None:
        super().__init__(message)
        self.code = code
        self.remote_traceback = remote_traceback


class WorkerRuntime:
    def __init__(
        self,
        config: WorkerConfig,
        executor: JobExecutor,
        *,
        client: PlatformClient | None = None,
    ) -> None:
        self.config = config
        self.executor = executor
        self.client = client or PlatformClient(
            config.api_url, config.token, timeout=config.request_timeout_seconds
        )
        self._stopping = False
        self.worker_id: str | None = None
        self._last_workspace_reconcile_at: float | None = None
        self._identity_lock = threading.Lock()
        self._workspace_lock = threading.Lock()
        self._active_job_ids: set[str] = set()

    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._register_worker()
        self._reconcile_workspaces(force=True)
        while not self._stopping:
            try:
                self._reconcile_workspaces()
                worker_id = self.worker_id
                if worker_id is None:
                    self._register_worker()
                    worker_id = self.worker_id
                assert worker_id is not None
                claim = self.client.claim(worker_id)
                if claim is None:
                    self.client.heartbeat(worker_id)
                    time.sleep(self.config.poll_seconds)
                    continue
                self._execute_claim(claim)
            except WorkerApiError as error:
                if error.status == 404 and error.code == "not_found":
                    # The API may have lost this worker record after a database
                    # restore or node cleanup. Re-register instead of retrying
                    # a permanently invalid worker ID.
                    LOGGER.warning("worker identity is no longer registered; re-registering")
                    self.worker_id = None
                    try:
                        self._register_worker()
                    except WorkerApiError as register_error:
                        LOGGER.warning("worker re-registration failed: %s", register_error)
                LOGGER.warning("worker API error: %s", error)
                time.sleep(self.config.poll_seconds)

    def run_once(self) -> bool:
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_registered()
        self._reconcile_workspaces(force=True)
        worker_id = self.worker_id
        assert worker_id is not None
        claim = self.client.claim(worker_id)
        if claim is None:
            return False
        self._execute_claim(claim)
        return True

    def run_job(self, job_id: str) -> bool:
        """Claim and execute one explicitly dispatched central job."""
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_registered()
        self._reconcile_workspaces(force=True)
        worker_id = self.worker_id
        assert worker_id is not None
        claim = self.client.claim(worker_id, job_id)
        if claim is None:
            return False
        self._execute_claim(claim)
        return True

    def _ensure_registered(self) -> None:
        if self.worker_id is not None:
            return
        with self._identity_lock:
            if self.worker_id is None:
                self._register_worker()

    def _register_worker(self) -> None:
        response = self.client.register(self._registration())
        self.worker_id = str(response["id"])
        LOGGER.info("registered worker %s as %s", self.config.name, self.worker_id)

    def _execute_claim(self, claim: dict[str, Any]) -> None:
        raw_job = claim.get("job")
        lease_token = claim.get("leaseToken")
        if not isinstance(raw_job, dict) or not isinstance(lease_token, str):
            raise ValueError("Invalid claim response")
        job = cast(dict[str, Any], raw_job)
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("Claim response has no job ID")
        workspace = self.config.work_dir / job_id
        process_context = multiprocessing.get_context("spawn")
        result_queue = process_context.Queue(maxsize=1)
        execution = process_context.Process(
            target=_execute_job_process,
            args=(
                self.executor,
                claim,
                self.config.api_url,
                self.config.token,
                self.config.request_timeout_seconds,
                workspace,
                result_queue,
            ),
            name=f"job-{job_id}",
        )
        with self._workspace_lock:
            workspace.mkdir(parents=True, exist_ok=True)
            self._active_job_ids.add(job_id)
        started = False
        try:
            execution.start()
            started = True
            while execution.is_alive():
                execution.join(timeout=self.config.lease_renew_seconds)
                if execution.is_alive():
                    self.client.renew(job_id, lease_token)
            try:
                raw_result = result_queue.get(timeout=5)
            except queue.Empty as error:
                raise RuntimeError(
                    f"Job process exited with code {execution.exitcode} without a result"
                ) from error
            if not isinstance(raw_result, dict):
                raise RuntimeError("Job process returned an invalid result")
            payload = cast(dict[str, Any], raw_result)
            if not payload.get("ok"):
                raise JobExecutionError(
                    str(payload.get("code", "JobExecutionError")),
                    str(payload.get("message", "Job execution failed")),
                    str(payload.get("traceback", "")),
                )
            raw_payload_result = payload.get("result")
            if not isinstance(raw_payload_result, dict):
                raise RuntimeError("Job process returned a non-object result")
            result = cast(dict[str, Any], raw_payload_result)
            self.client.complete(job_id, lease_token, result)
        except Exception as error:
            if isinstance(error, JobExecutionError):
                LOGGER.error("job %s failed in child process\n%s", job_id, error.remote_traceback)
                error_code = error.code
            else:
                LOGGER.exception("job %s failed", job_id)
                error_code = type(error).__name__
            try:
                self.client.fail(job_id, lease_token, error_code, str(error))
            except WorkerApiError:
                LOGGER.exception("failed to report job %s failure", job_id)
        finally:
            if started and execution.is_alive():
                execution.terminate()
                execution.join(timeout=5)
            result_queue.close()
            result_queue.join_thread()
            with self._workspace_lock:
                self._active_job_ids.discard(job_id)

    def _reconcile_workspaces(self, *, force: bool = False) -> None:
        with self._workspace_lock:
            now = time.monotonic()
            if (
                not force
                and self._last_workspace_reconcile_at is not None
                and now - self._last_workspace_reconcile_at
                < self.config.workspace_reconcile_seconds
            ):
                return
            self._last_workspace_reconcile_at = now
            try:
                retained_job_ids = self.client.retained_job_ids() | self._active_job_ids
                cleanups = prune_orphan_workspaces(self.config.work_dir, retained_job_ids)
            except (OSError, WorkerApiError) as error:
                LOGGER.warning("workspace reconciliation failed: %s", error)
                return
        for cleanup in cleanups:
            LOGGER.info(
                "removed deleted job workspace %s (%d bytes)",
                cleanup.job_id,
                cleanup.bytes_removed,
            )

    def _registration(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "kind": self.config.kind,
            "capabilities": list(self.config.capabilities),
            "accelerator": self.config.accelerator,
            "maxConcurrency": self.config.max_concurrency,
            "version": self.config.version,
            "metadata": {},
        }

    def _stop(self, _signum: int, _frame: object) -> None:
        self._stopping = True
