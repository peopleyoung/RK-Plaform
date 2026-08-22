from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from pydantic import ValidationError
from sqlalchemy import text

from .auth import get_context, require_admin, require_admin_or_worker, require_worker
from .context import AppContext
from .contracts import (
    ArtifactResponse,
    ConversionJobCreate,
    DatasetClassesUpdate,
    DatasetMetadata,
    DatasetResponse,
    JobClaim,
    JobClaimRequest,
    JobCompletion,
    JobEventResponse,
    JobFailure,
    JobLeaseRenewal,
    JobProgressUpdate,
    JobResponse,
    JobTelemetryAccepted,
    JobTelemetryUpdate,
    JobType,
    ModelProfileDocument,
    NodeEnrollmentClaim,
    NodeEnrollmentClaimResponse,
    ServiceEndpointCreateResponse,
    ServiceEndpointEnrollmentResponse,
    ServiceEndpointPayload,
    ServiceEndpointResponse,
    ServiceEndpointTestResponse,
    TrainingJobCreate,
    WorkerHeartbeat,
    WorkerRegistration,
    WorkerResponse,
    WorkspaceRetentionResponse,
)
from .node_enrollment import NodeEnrollmentService
from .service import PlatformService, artifact_response, dataset_response
from .state_machine import job_response

router = APIRouter()
Admin = Annotated[None, Depends(require_admin)]
Worker = Annotated[str | None, Depends(require_worker)]
AdminOrWorker = Annotated[str | None, Depends(require_admin_or_worker)]
Context = Annotated[AppContext, Depends(get_context)]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(context: Context) -> dict[str, str]:
    with context.database.session() as session:
        session.execute(text("SELECT 1"))
    return {"status": "ready"}


@router.get("/model-profiles", response_model=ModelProfileDocument)
def model_profiles(_: Admin, context: Context) -> ModelProfileDocument:
    return context.profiles.document


@router.post(
    "/datasets",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_dataset(
    _: Admin,
    context: Context,
    metadata: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> DatasetResponse:
    try:
        parsed = DatasetMetadata.model_validate_json(metadata)
    except ValidationError as error:
        raise RequestValidationError(error.errors()) from error
    return await PlatformService(context).create_dataset(parsed, file)


@router.get("/datasets", response_model=list[DatasetResponse])
def list_datasets(_: Admin, context: Context) -> list[DatasetResponse]:
    return PlatformService(context).list_datasets()


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, _: Admin, context: Context) -> DatasetResponse:
    return dataset_response(PlatformService(context).get_dataset(dataset_id))


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: str, _: Admin, context: Context) -> None:
    PlatformService(context).delete_dataset(dataset_id)


@router.get("/worker/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str, endpoint_id: Worker, context: Context) -> FileResponse:
    service = PlatformService(context)
    service.require_worker_dataset_access(dataset_id, endpoint_id)
    record = service.get_dataset(dataset_id)
    path = context.storage.require(record.storage_key)
    return FileResponse(path, filename=record.filename, media_type="application/octet-stream")


@router.put("/worker/datasets/{dataset_id}/classes", response_model=DatasetResponse)
def update_dataset_classes(
    dataset_id: str,
    payload: DatasetClassesUpdate,
    endpoint_id: Worker,
    context: Context,
) -> DatasetResponse:
    service = PlatformService(context)
    service.require_worker_dataset_access(dataset_id, endpoint_id)
    return service.update_dataset_classes(dataset_id, payload.classes)


@router.post(
    "/training-jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_training_job(payload: TrainingJobCreate, _: Admin, context: Context) -> JobResponse:
    return PlatformService(context).create_training_job(payload)


@router.post(
    "/conversion-jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversion_job(payload: ConversionJobCreate, _: Admin, context: Context) -> JobResponse:
    return PlatformService(context).create_conversion_job(payload)


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(
    _: Admin,
    context: Context,
    job_type: Annotated[JobType | None, Query(alias="type")] = None,
) -> list[JobResponse]:
    return PlatformService(context).list_jobs(job_type)


@router.post(
    "/jobs/{job_id}/retry",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
)
def retry_job(job_id: str, _: Admin, context: Context) -> JobResponse:
    return PlatformService(context).retry_job(job_id)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, _: Admin, context: Context) -> JobResponse:
    return PlatformService(context).get_job(job_id)


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str, _: Admin, context: Context) -> None:
    PlatformService(context).delete_job(job_id)


@router.get("/jobs/{job_id}/events", response_model=list[JobEventResponse])
def get_job_events(
    job_id: str,
    _: Admin,
    context: Context,
    after_id: Annotated[int, Query(alias="afterId", ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> list[JobEventResponse]:
    return PlatformService(context).job_events(job_id, after_id=after_id, limit=limit)


@router.post(
    "/workers/register",
    response_model=WorkerResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_worker(
    payload: WorkerRegistration, endpoint_id: Worker, context: Context
) -> WorkerResponse:
    return PlatformService(context).register_worker(payload, endpoint_id)


@router.post("/workers/{worker_id}/heartbeat", response_model=WorkerResponse)
def worker_heartbeat(
    worker_id: str,
    payload: WorkerHeartbeat,
    endpoint_id: Worker,
    context: Context,
) -> WorkerResponse:
    with context.database.session() as session:
        PlatformService.require_worker_identity(session, worker_id, endpoint_id)
    return PlatformService(context).heartbeat(worker_id, payload)


@router.get("/workers", response_model=list[WorkerResponse])
def list_workers(_: Admin, context: Context) -> list[WorkerResponse]:
    return PlatformService(context).list_workers()


@router.delete("/workers/{worker_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_worker(worker_id: str, _: Admin, context: Context) -> None:
    PlatformService(context).delete_worker(worker_id)


@router.get("/service-endpoints", response_model=list[ServiceEndpointResponse])
def list_service_endpoints(_: Admin, context: Context) -> list[ServiceEndpointResponse]:
    return PlatformService(context).list_service_endpoints()


@router.post("/service-endpoints/test", response_model=ServiceEndpointTestResponse)
def test_service_endpoint(
    payload: ServiceEndpointPayload, _: Admin, context: Context
) -> ServiceEndpointTestResponse:
    return PlatformService(context).test_service_endpoint(payload)


@router.post(
    "/service-endpoints/{endpoint_id}/test", response_model=ServiceEndpointTestResponse
)
def test_service_endpoint_update(
    endpoint_id: str,
    payload: ServiceEndpointPayload,
    _: Admin,
    context: Context,
) -> ServiceEndpointTestResponse:
    return PlatformService(context).test_service_endpoint_update(endpoint_id, payload)


@router.post(
    "/service-endpoints",
    response_model=ServiceEndpointCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_service_endpoint(
    payload: ServiceEndpointPayload, _: Admin, context: Context
) -> ServiceEndpointCreateResponse:
    return PlatformService(context).create_service_endpoint(payload)


@router.post(
    "/service-endpoints/{endpoint_id}/enrollment-token",
    response_model=ServiceEndpointEnrollmentResponse,
)
def reissue_service_endpoint_enrollment(
    endpoint_id: str, _: Admin, context: Context
) -> ServiceEndpointEnrollmentResponse:
    return NodeEnrollmentService(context).issue(endpoint_id)


@router.post(
    "/node-enrollments/{endpoint_id}/claim",
    response_model=NodeEnrollmentClaimResponse,
)
def claim_node_enrollment(
    endpoint_id: str, payload: NodeEnrollmentClaim, context: Context
) -> NodeEnrollmentClaimResponse:
    return NodeEnrollmentService(context).claim(endpoint_id, payload)


@router.put("/service-endpoints/{endpoint_id}", response_model=ServiceEndpointResponse)
def update_service_endpoint(
    endpoint_id: str, payload: ServiceEndpointPayload, _: Admin, context: Context
) -> ServiceEndpointResponse:
    return PlatformService(context).update_service_endpoint(endpoint_id, payload)


@router.post(
    "/service-endpoints/{endpoint_id}/probe", response_model=ServiceEndpointResponse
)
def probe_service_endpoint(
    endpoint_id: str, _: Admin, context: Context
) -> ServiceEndpointResponse:
    return PlatformService(context).probe_service_endpoint(endpoint_id)


@router.delete("/service-endpoints/{endpoint_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_service_endpoint(endpoint_id: str, _: Admin, context: Context) -> None:
    PlatformService(context).delete_service_endpoint(endpoint_id)


@router.post("/worker/jobs/claim", response_model=JobClaim | None)
def claim_job(
    payload: JobClaimRequest, endpoint_id: Worker, context: Context
) -> JobClaim | None:
    with context.database.session() as session:
        PlatformService.require_worker_identity(session, payload.worker_id, endpoint_id)
        return context.jobs.claim(session, payload.worker_id, payload.job_id)


@router.get("/worker/jobs/retained", response_model=WorkspaceRetentionResponse)
def retained_job_ids(endpoint_id: Worker, context: Context) -> WorkspaceRetentionResponse:
    return WorkspaceRetentionResponse(
        job_ids=PlatformService(context).retained_job_ids(endpoint_id)
    )


@router.post("/worker/jobs/{job_id}/progress", response_model=JobResponse)
def update_job_progress(
    job_id: str,
    payload: JobProgressUpdate,
    _: Worker,
    context: Context,
) -> JobResponse:
    with context.database.session() as session:
        return job_response(context.jobs.progress(session, job_id, payload))


@router.post("/worker/jobs/{job_id}/events", response_model=JobTelemetryAccepted)
def append_job_events(
    job_id: str,
    payload: JobTelemetryUpdate,
    _: Worker,
    context: Context,
) -> JobTelemetryAccepted:
    with context.database.session() as session:
        accepted = context.jobs.telemetry(session, job_id, payload)
        return JobTelemetryAccepted(accepted=accepted)


@router.post("/worker/jobs/{job_id}/renew", response_model=JobResponse)
def renew_job_lease(
    job_id: str,
    payload: JobLeaseRenewal,
    _: Worker,
    context: Context,
) -> JobResponse:
    with context.database.session() as session:
        return job_response(context.jobs.renew(session, job_id, payload))


@router.post("/worker/jobs/{job_id}/complete", response_model=JobResponse)
def complete_job(
    job_id: str,
    payload: JobCompletion,
    _: Worker,
    context: Context,
) -> JobResponse:
    with context.database.session() as session:
        return job_response(context.jobs.complete(session, job_id, payload))


@router.post("/worker/jobs/{job_id}/fail", response_model=JobResponse)
def fail_job(
    job_id: str,
    payload: JobFailure,
    _: Worker,
    context: Context,
) -> JobResponse:
    with context.database.session() as session:
        return job_response(context.jobs.fail(session, job_id, payload))


@router.post(
    "/worker/jobs/{job_id}/artifacts",
    response_model=ArtifactResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_artifact(
    job_id: str,
    _: Worker,
    context: Context,
    lease_token: Annotated[str, Form()],
    kind: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    manifest: Annotated[str | None, Form()] = None,
) -> ArtifactResponse:
    return await PlatformService(context).create_artifact(job_id, lease_token, kind, manifest, file)


@router.get("/artifacts", response_model=list[ArtifactResponse])
def list_artifacts(
    _: Admin,
    context: Context,
    job_id: Annotated[str | None, Query(alias="jobId")] = None,
    kind: str | None = None,
) -> list[ArtifactResponse]:
    return PlatformService(context).list_artifacts(job_id=job_id, kind=kind)


@router.get("/artifacts/{artifact_id}", response_model=ArtifactResponse)
def get_artifact(
    artifact_id: str, endpoint_id: AdminOrWorker, context: Context
) -> ArtifactResponse:
    service = PlatformService(context)
    service.require_worker_artifact_access(artifact_id, endpoint_id)
    return artifact_response(service.get_artifact(artifact_id))


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(
    artifact_id: str, endpoint_id: AdminOrWorker, context: Context
) -> FileResponse:
    service = PlatformService(context)
    service.require_worker_artifact_access(artifact_id, endpoint_id)
    record = service.get_artifact(artifact_id)
    path = context.storage.require(record.storage_key)
    return FileResponse(path, filename=record.filename, media_type=record.media_type)
