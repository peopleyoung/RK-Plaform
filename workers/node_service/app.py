from __future__ import annotations

import hmac
import importlib
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field

from workers.common.rk3588_devices import RK3588_INFERENCE_PATHS
from workers.common.runtime import WorkerRuntime

from .config import NodeServiceSettings

LOGGER = logging.getLogger("rknode.node-service")
PROTOCOL_VERSION = "1.0"


class InferenceController(Protocol):
    def preflight(self) -> bool: ...

    def apply(self, revision: int, payload: dict[str, Any]) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...


class CudaRuntime(Protocol):
    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...


class TorchModule(Protocol):
    cuda: CudaRuntime


class PaddleDevice(Protocol):
    cuda: CudaRuntime

    def is_compiled_with_cuda(self) -> bool: ...


class PaddleModule(Protocol):
    device: PaddleDevice


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=lambda value: _camel(value), populate_by_name=True)


def _camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class DispatchResponse(ApiModel):
    job_id: str
    state: str
    accepted: bool
    message: str = ""


class JobState(ApiModel):
    job_id: str
    state: str
    message: str = ""
    updated_at: datetime


class InferenceRevisionRequest(ApiModel):
    node_id: str = Field(min_length=1)
    central_api_url: str = Field(min_length=1)
    access_token: str = Field(min_length=16)
    desired: dict[str, Any]


class JobRegistry:
    def __init__(self, runtime: WorkerRuntime, max_concurrency: int) -> None:
        self.runtime = runtime
        self.max_concurrency = max_concurrency
        self._states: dict[str, JobState] = {}
        self._active: set[str] = set()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="direct-job",
        )

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def dispatch(self, job_id: str) -> DispatchResponse:
        with self._lock:
            current = self._states.get(job_id)
            if job_id in self._active:
                return DispatchResponse(job_id=job_id, state="running", accepted=True)
            if current is not None and current.state == "succeeded":
                return DispatchResponse(job_id=job_id, state=current.state, accepted=True)
            if len(self._active) >= self.max_concurrency:
                raise HTTPException(status_code=409, detail="node capacity is full")
            self._active.add(job_id)
            self._states[job_id] = self._state(job_id, "accepted")
        self._executor.submit(self._run, job_id)
        return DispatchResponse(job_id=job_id, state="accepted", accepted=True)

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._states.get(job_id)

    def cleanup(self, job_id: str, work_dir: Path) -> None:
        with self._lock:
            if job_id in self._active:
                raise HTTPException(status_code=409, detail="cannot clean a running job")
        root = work_dir.resolve()
        target = (root / job_id).resolve()
        if target.parent != root:
            raise HTTPException(status_code=400, detail="invalid job ID")
        shutil.rmtree(target, ignore_errors=True)
        with self._lock:
            self._states.pop(job_id, None)

    def _run(self, job_id: str) -> None:
        self._set(job_id, "running")
        try:
            claimed = self.runtime.run_job(job_id)
            self._set(
                job_id,
                "succeeded" if claimed else "not_claimed",
                "" if claimed else "central job was no longer queued",
            )
        except Exception as error:
            LOGGER.exception("direct job %s failed", job_id)
            self._set(job_id, "failed", str(error))
        finally:
            with self._lock:
                self._active.discard(job_id)

    def _set(self, job_id: str, state: str, message: str = "") -> None:
        with self._lock:
            self._states[job_id] = self._state(job_id, state, message)

    @staticmethod
    def _state(job_id: str, state: str, message: str = "") -> JobState:
        return JobState(
            job_id=job_id,
            state=state,
            message=message[:1000],
            updated_at=datetime.now(UTC),
        )


def create_node_app(
    settings: NodeServiceSettings,
    *,
    runtime: WorkerRuntime | None = None,
    inference: InferenceController | None = None,
) -> FastAPI:
    if not settings.token:
        raise ValueError("node Token must be resolved before creating the node service")
    node_token = settings.token
    if settings.kind in {"trainer", "converter"} and runtime is None:
        raise ValueError("training and conversion node services require a worker runtime")
    if settings.kind == "inference" and inference is None:
        raise ValueError("inference node services require an inference controller")
    jobs = JobRegistry(runtime, settings.max_concurrency) if runtime is not None else None
    inference_preflight = inference.preflight() if inference is not None else True

    app = FastAPI(title=f"RKNode {settings.kind} service", version=settings.version)

    def authorize(authorization: str | None = Header(default=None)) -> None:
        prefix = "Bearer "
        supplied = (
            authorization[len(prefix) :]
            if authorization and authorization.startswith(prefix)
            else ""
        )
        if not supplied or not hmac.compare_digest(supplied, node_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid node token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def health_payload() -> dict[str, Any]:
        diagnostics = _diagnostics(settings)
        if inference is not None:
            diagnostics["inference"] = inference.status()
            diagnostics["inferenceSelfTestPassed"] = inference_preflight
        healthy = bool(diagnostics.pop("healthy", True)) and inference_preflight
        return {
            "status": "healthy" if healthy else "unhealthy",
            "protocolVersion": PROTOCOL_VERSION,
            "name": settings.name,
            "kind": settings.kind,
            "accelerator": settings.accelerator,
            "capabilities": list(settings.capabilities),
            "features": list(settings.features),
            "version": settings.version,
            "maxConcurrency": settings.max_concurrency,
            "activeJobs": jobs.active_count if jobs is not None else 0,
            "diagnostics": diagnostics,
        }

    @app.get("/health", dependencies=[Depends(authorize)])
    def health() -> dict[str, Any]:
        return health_payload()

    @app.get("/api/v1/capabilities", dependencies=[Depends(authorize)])
    def capabilities() -> dict[str, Any]:
        return health_payload()

    @app.post(
        "/api/v1/jobs/{job_id}/dispatch",
        response_model=DispatchResponse,
        response_model_by_alias=True,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authorize)],
    )
    def dispatch(job_id: str) -> DispatchResponse:
        if jobs is None:
            raise HTTPException(status_code=409, detail="this node does not execute jobs")
        return jobs.dispatch(job_id)

    @app.get(
        "/api/v1/jobs/{job_id}",
        response_model=JobState,
        response_model_by_alias=True,
        dependencies=[Depends(authorize)],
    )
    def job_status(job_id: str) -> JobState:
        if jobs is None:
            raise HTTPException(status_code=409, detail="this node does not execute jobs")
        state_record = jobs.get(job_id)
        if state_record is None:
            raise HTTPException(status_code=404, detail="job is not known by this node")
        return state_record

    @app.delete(
        "/api/v1/jobs/{job_id}/cache",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[Depends(authorize)],
    )
    def clean_job_cache(job_id: str) -> Response:
        if jobs is None:
            raise HTTPException(status_code=409, detail="this node does not execute jobs")
        jobs.cleanup(job_id, settings.work_dir)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.put(
        "/api/v1/inference/revisions/{revision}",
        dependencies=[Depends(authorize)],
    )
    def apply_revision(revision: int, payload: InferenceRevisionRequest) -> dict[str, Any]:
        if inference is None:
            raise HTTPException(status_code=409, detail="this node is not an inference service")
        desired_revision = int(payload.desired.get("revision", -1))
        if revision < 0 or desired_revision != revision:
            raise HTTPException(status_code=422, detail="revision path and payload must match")
        return inference.apply(revision, payload.model_dump(by_alias=True))

    @app.get("/api/v1/inference/status", dependencies=[Depends(authorize)])
    def inference_status() -> dict[str, Any]:
        if inference is None:
            raise HTTPException(status_code=409, detail="this node is not an inference service")
        return inference.status()

    _ = (
        health,
        capabilities,
        dispatch,
        job_status,
        clean_job_cache,
        apply_revision,
        inference_status,
    )
    return app


def _diagnostics(settings: NodeServiceSettings) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"workDir": str(settings.work_dir)}
    if settings.accelerator == "cuda":
        try:
            paddle_only = all(
                capability.startswith("ppocr-") for capability in settings.capabilities
            )
            if paddle_only:
                paddle = cast(PaddleModule, importlib.import_module("paddle"))
                device_count = int(paddle.device.cuda.device_count())
                cuda_available = (
                    bool(paddle.device.is_compiled_with_cuda()) and device_count > 0
                )
                framework = "paddle"
            else:
                torch = cast(TorchModule, importlib.import_module("torch"))
                device_count = int(torch.cuda.device_count())
                cuda_available = bool(torch.cuda.is_available()) and device_count > 0
                framework = "torch"
            diagnostics.update(
                {
                    "cudaFramework": framework,
                    "cudaAvailable": cuda_available,
                    "cudaDeviceCount": device_count,
                    "healthy": cuda_available,
                }
            )
        except Exception as error:
            diagnostics.update(
                {"cudaAvailable": False, "cudaError": str(error), "healthy": False}
            )
    elif settings.kind in {"converter", "inference"}:
        if settings.kind == "converter":
            try:
                importlib.import_module("rknn.api")
                diagnostics["rknnToolkitAvailable"] = True
            except Exception as error:
                diagnostics.update(
                    {
                        "rknnToolkitAvailable": False,
                        "rknnToolkitError": str(error),
                        "healthy": False,
                    }
                )
            device_candidates = (Path("/dev/dri"), Path("/dev/dma_heap"))
        else:
            device_candidates = RK3588_INFERENCE_PATHS
        available = [str(path) for path in device_candidates if path.exists()]
        runtime_healthy = bool(diagnostics.get("healthy", True))
        devices_healthy = (
            len(available) == len(device_candidates)
            or not settings.require_accelerator_device
        )
        diagnostics.update(
            {
                "devices": available,
                "healthy": runtime_healthy and devices_healthy,
                "deviceCheckRequired": settings.require_accelerator_device,
            }
        )
    return diagnostics
