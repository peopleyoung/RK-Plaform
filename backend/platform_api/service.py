from __future__ import annotations

import hmac
import tarfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import UploadFile
from pydantic import ValidationError
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .context import AppContext
from .contracts import (
    ArtifactResponse,
    ConversionJobCreate,
    DatasetFormat,
    DatasetMetadata,
    DatasetResponse,
    DatasetStatus,
    DeploymentManifest,
    JobEventResponse,
    JobResponse,
    JobStatus,
    JobType,
    ServiceEndpointCreateResponse,
    ServiceEndpointEnrollmentStatus,
    ServiceEndpointKind,
    ServiceEndpointMode,
    ServiceEndpointPayload,
    ServiceEndpointResponse,
    ServiceEndpointTestResponse,
    TaskType,
    TrainingJobCreate,
    WorkerHeartbeat,
    WorkerKind,
    WorkerRegistration,
    WorkerResponse,
    WorkerStatus,
)
from .db_models import (
    ArtifactRecord,
    DatasetRecord,
    JobEventRecord,
    JobRecord,
    ModelReleaseRecord,
    NodeCleanupRecord,
    ServiceEndpointRecord,
    WorkerRecord,
)
from .errors import AppError, AuthenticationError, ConflictError, NotFoundError
from .node_client import DirectNodeClient, NodeClientError
from .state_machine import as_utc, job_response, utc_now


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def dataset_response(record: DatasetRecord) -> DatasetResponse:
    return DatasetResponse(
        id=record.id,
        name=record.name,
        description=record.description,
        version=record.version,
        task_type=TaskType(record.task_type),
        dataset_format=DatasetFormat(record.dataset_format),
        classes=record.classes_json,
        status=DatasetStatus(record.status),
        filename=record.filename,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
        created_at=record.created_at,
        updated_at=record.updated_at,
        error_message=record.error_message,
    )


def worker_response(record: WorkerRecord) -> WorkerResponse:
    return WorkerResponse(
        id=record.id,
        name=record.name,
        kind=WorkerKind(record.kind),
        status=WorkerStatus(record.status),
        capabilities=record.capabilities_json,
        accelerator=cast(Literal["cpu", "cuda", "rk3588"], record.accelerator),
        max_concurrency=record.max_concurrency,
        active_jobs=record.active_jobs,
        version=record.version,
        metadata=record.metadata_json,
        last_seen_at=record.last_seen_at,
        created_at=record.created_at,
    )


def artifact_response(record: ArtifactRecord) -> ArtifactResponse:
    return ArtifactResponse(
        id=record.id,
        job_id=record.job_id,
        kind=record.kind,
        filename=record.filename,
        media_type=record.media_type,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
        manifest=record.manifest_json,
        created_at=record.created_at,
    )


def service_endpoint_response(record: ServiceEndpointRecord) -> ServiceEndpointResponse:
    return ServiceEndpointResponse(
        id=record.id,
        name=record.name,
        kind=ServiceEndpointKind(record.kind),
        mode=ServiceEndpointMode(record.mode),
        endpoint=record.endpoint,
        scheme=cast(Literal["http", "https"], record.scheme),
        host=record.host,
        port=record.port,
        accelerator=cast(Literal["cpu", "cuda", "rk3588"], record.accelerator),
        capabilities=record.capabilities_json,
        enabled=record.enabled,
        token_configured=record.token_configured,
        enrollment_status=ServiceEndpointEnrollmentStatus(record.enrollment_status),
        enrollment_expires_at=record.enrollment_expires_at,
        enrollment_claimed_at=record.enrollment_claimed_at,
        enrolled_at=record.enrolled_at,
        probe_status=record.probe_status,
        last_probe_at=record.last_probe_at,
        last_error=record.last_error,
        remote_metadata=record.remote_metadata_json or {},
        inference_node_id=record.inference_node_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _positive_int(value: object, default: int = 1) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(1, value)
    if isinstance(value, str) and value.isdigit():
        return max(1, int(value))
    return default


class PlatformService:
    def __init__(self, context: AppContext) -> None:
        self.context = context

    async def create_dataset(
        self,
        metadata: DatasetMetadata,
        upload: UploadFile,
    ) -> DatasetResponse:
        filename = self.context.storage.safe_filename(upload.filename)
        if not self._supported_archive_name(filename):
            raise AppError(
                "unsupported_dataset_archive",
                "Dataset must be a .zip, .tar.gz, or .tgz archive",
            )
        stored = await self.context.storage.write_upload(upload, "datasets")
        archive_path = self.context.storage.require(stored.storage_key)
        if not self._valid_archive(archive_path):
            archive_path.unlink(missing_ok=True)
            raise AppError("invalid_dataset_archive", "Uploaded file is not a valid archive")
        now = utc_now()
        record = DatasetRecord(
            id=new_id("ds"),
            name=metadata.name,
            description=metadata.description,
            version=metadata.version,
            task_type=metadata.task_type.value,
            dataset_format=metadata.dataset_format.value,
            classes_json=metadata.classes,
            status=DatasetStatus.READY.value,
            filename=stored.filename,
            storage_key=stored.storage_key,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            created_at=now,
            updated_at=now,
        )
        with self.context.database.session() as session:
            session.add(record)
            session.flush()
            return dataset_response(record)

    def list_datasets(self) -> list[DatasetResponse]:
        with self.context.database.session() as session:
            records = session.scalars(
                select(DatasetRecord).order_by(DatasetRecord.created_at.desc())
            ).all()
            return [dataset_response(record) for record in records]

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        with self.context.database.session() as session:
            record = session.get(DatasetRecord, dataset_id)
            if record is None:
                raise NotFoundError("dataset", dataset_id)
            session.expunge(record)
            return record

    def update_dataset_classes(self, dataset_id: str, classes: list[str]) -> DatasetResponse:
        with self.context.database.session() as session:
            record = session.get(DatasetRecord, dataset_id)
            if record is None:
                raise NotFoundError("dataset", dataset_id)
            if record.task_type == TaskType.SEMANTIC_SEGMENTATION.value and not (
                2 <= len(classes) <= 256
            ):
                raise AppError(
                    "invalid_dataset_classes",
                    "Segmentation datasets require between 2 and 256 classes",
                    status_code=422,
                )
            if record.classes_json:
                if record.classes_json != classes:
                    raise ConflictError(
                        "dataset_classes_mismatch",
                        "Discovered classes differ from the classes already stored "
                        "for this dataset",
                        storedClasses=record.classes_json,
                        discoveredClasses=classes,
                    )
                return dataset_response(record)
            record.classes_json = classes
            record.updated_at = utc_now()
            session.flush()
            return dataset_response(record)

    def delete_dataset(self, dataset_id: str) -> None:
        storage_key: str
        with self.context.database.session() as session:
            record = session.get(DatasetRecord, dataset_id)
            if record is None:
                raise NotFoundError("dataset", dataset_id)
            dependent_jobs = session.scalars(
                select(JobRecord).where(JobRecord.dataset_id == dataset_id)
            ).all()
            active_jobs = [
                job.id
                for job in dependent_jobs
                if JobStatus(job.status) in {JobStatus.QUEUED, JobStatus.CLAIMED, JobStatus.RUNNING}
            ]
            if active_jobs:
                raise ConflictError(
                    "dataset_in_use",
                    "Dataset is referenced by active jobs and cannot be deleted",
                    jobIds=active_jobs,
                )
            for job in dependent_jobs:
                job.dataset_id = None
            storage_key = record.storage_key
            session.delete(record)
        self.context.storage.remove(storage_key)

    def create_training_job(
        self,
        payload: TrainingJobCreate,
        *,
        retry_of_job_id: str | None = None,
    ) -> JobResponse:
        profile = self.context.profiles.get(payload.profile_id)
        self.context.profiles.validate_resolution(payload.profile_id, payload.resolution)
        if payload.variant not in profile.variants:
            raise AppError("unsupported_variant", f"Unsupported variant '{payload.variant}'")
        with self.context.database.session() as session:
            dataset = session.get(DatasetRecord, payload.dataset_id)
            if dataset is None:
                raise NotFoundError("dataset", payload.dataset_id)
            if dataset.status != DatasetStatus.READY.value:
                raise ConflictError("dataset_not_ready", "Dataset is not ready for training")
            if dataset.task_type != profile.task_type.value:
                raise AppError(
                    "dataset_task_mismatch",
                    "Dataset task type does not match model profile",
                    details={
                        "datasetTask": dataset.task_type,
                        "profileTask": profile.task_type.value,
                    },
                )
            now = utc_now()
            spec = payload.model_dump(mode="json", by_alias=True)
            if retry_of_job_id is not None:
                spec["retryOfJobId"] = retry_of_job_id
            spec["dataset"] = {
                "id": dataset.id,
                "name": dataset.name,
                "version": dataset.version,
                "filename": dataset.filename,
                "sha256": dataset.sha256,
                "classes": dataset.classes_json,
                "taskType": dataset.task_type,
                "datasetFormat": dataset.dataset_format,
            }
            job = JobRecord(
                id=new_id("train"),
                type=JobType.TRAINING.value,
                name=payload.name,
                status=JobStatus.QUEUED.value,
                profile_id=payload.profile_id,
                dataset_id=payload.dataset_id,
                progress=0,
                stage="queued",
                spec_json=spec,
                retry_count=0,
                max_retries=self.context.settings.worker_max_retries,
                created_at=now,
                updated_at=now,
            )
            session.add(job)
            session.add(
                JobEventRecord(
                    job_id=job.id,
                    type="queued",
                    level="info",
                    message="Training job queued",
                    data_json={
                        "profileId": payload.profile_id,
                        **(
                            {"retryOfJobId": retry_of_job_id} if retry_of_job_id is not None else {}
                        ),
                    },
                )
            )
            session.flush()
            return job_response(job)

    def create_conversion_job(
        self,
        payload: ConversionJobCreate,
        *,
        retry_of_job_id: str | None = None,
    ) -> JobResponse:
        with self.context.database.session() as session:
            source = session.get(ArtifactRecord, payload.source_artifact_id)
            if source is None:
                raise NotFoundError("artifact", payload.source_artifact_id)
            if source.kind != "onnx" or not source.manifest_json:
                raise AppError(
                    "artifact_not_convertible",
                    "Source artifact must be an ONNX model with a deployment manifest",
                )
            manifest = DeploymentManifest.model_validate(source.manifest_json)
            profile = self.context.profiles.validate_manifest(manifest)
            self.context.profiles.validate_precision(
                profile.id, payload.precision, payload.calibration_dataset_id
            )
            calibration: DatasetRecord | None = None
            if payload.calibration_dataset_id:
                calibration = session.get(DatasetRecord, payload.calibration_dataset_id)
                if calibration is None:
                    raise NotFoundError("dataset", payload.calibration_dataset_id)
                if calibration.status != DatasetStatus.READY.value:
                    raise ConflictError(
                        "calibration_dataset_not_ready", "Calibration dataset is not ready"
                    )
                if calibration.task_type != profile.task_type.value:
                    raise AppError(
                        "calibration_task_mismatch",
                        "Calibration dataset task type does not match model profile",
                    )
            now = utc_now()
            spec = payload.model_dump(mode="json", by_alias=True)
            if retry_of_job_id is not None:
                spec["retryOfJobId"] = retry_of_job_id
            spec["profileId"] = profile.id
            spec["resolution"] = manifest.resolution.model_dump(mode="json", by_alias=True)
            spec["manifest"] = manifest.model_dump(mode="json", by_alias=True)
            spec["sourceArtifact"] = {
                "id": source.id,
                "filename": source.filename,
                "sha256": source.sha256,
            }
            if payload.calibration_dataset_id and calibration is not None:
                spec["calibrationDataset"] = {
                    "id": calibration.id,
                    "filename": calibration.filename,
                    "sha256": calibration.sha256,
                }
            job = JobRecord(
                id=new_id("convert"),
                type=JobType.CONVERSION.value,
                name=payload.name,
                status=JobStatus.QUEUED.value,
                profile_id=profile.id,
                dataset_id=payload.calibration_dataset_id,
                progress=0,
                stage="queued",
                spec_json=spec,
                retry_count=0,
                max_retries=self.context.settings.worker_max_retries,
                created_at=now,
                updated_at=now,
            )
            session.add(job)
            session.add(
                JobEventRecord(
                    job_id=job.id,
                    type="queued",
                    level="info",
                    message="Conversion job queued",
                    data_json={
                        "precision": payload.precision.value,
                        **(
                            {"retryOfJobId": retry_of_job_id} if retry_of_job_id is not None else {}
                        ),
                    },
                )
            )
            session.flush()
            return job_response(job)

    def retry_job(self, job_id: str) -> JobResponse:
        with self.context.database.session() as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                raise NotFoundError("job", job_id)
            if JobStatus(record.status) != JobStatus.FAILED:
                raise ConflictError(
                    "job_not_retryable",
                    "Only failed jobs can be retried",
                    status=record.status,
                )
            job_type = JobType(record.type)
            spec = dict(record.spec_json)

        try:
            if job_type == JobType.TRAINING:
                payload = TrainingJobCreate.model_validate(
                    {
                        key: spec[key]
                        for key in (
                            "name",
                            "datasetId",
                            "profileId",
                            "variant",
                            "resolution",
                            "hyperparameters",
                            "accelerator",
                        )
                        if key in spec
                    }
                )
                return self.create_training_job(payload, retry_of_job_id=job_id)

            payload = ConversionJobCreate.model_validate(
                {
                    key: spec[key]
                    for key in (
                        "name",
                        "sourceArtifactId",
                        "precision",
                        "calibrationDatasetId",
                    )
                    if key in spec
                }
            )
            return self.create_conversion_job(payload, retry_of_job_id=job_id)
        except ValidationError as error:
            raise ConflictError(
                "job_retry_invalid_spec",
                "Stored job specification can no longer be retried",
                validationErrorCount=error.error_count(),
            ) from error

    def list_jobs(self, job_type: JobType | None = None) -> list[JobResponse]:
        with self.context.database.session() as session:
            statement = select(JobRecord)
            if job_type:
                statement = statement.where(JobRecord.type == job_type.value)
            records = session.scalars(statement.order_by(JobRecord.created_at.desc())).all()
            return [job_response(record) for record in records]

    def get_job(self, job_id: str) -> JobResponse:
        with self.context.database.session() as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                raise NotFoundError("job", job_id)
            return job_response(record)

    def retained_job_ids(self, authenticated_endpoint_id: str | None = None) -> list[str]:
        with self.context.database.session() as session:
            statement = select(JobRecord.id)
            if authenticated_endpoint_id is not None:
                worker = self._worker_for_endpoint(session, authenticated_endpoint_id)
                statement = statement.where(JobRecord.worker_id == worker.id)
            return list(session.scalars(statement.order_by(JobRecord.id)).all())

    def delete_job(self, job_id: str) -> None:
        storage_keys: list[str]
        cleanup_id: int | None = None
        with self.context.database.session() as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                raise NotFoundError("job", job_id)
            if JobStatus(record.status) not in {
                JobStatus.QUEUED,
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                raise ConflictError(
                    "job_not_deletable",
                    "Claimed or running jobs cannot be deleted",
                    status=record.status,
                )
            artifacts = session.scalars(
                select(ArtifactRecord).where(ArtifactRecord.job_id == job_id)
            ).all()
            artifact_ids = {artifact.id for artifact in artifacts}
            release_references = session.scalars(
                select(ModelReleaseRecord).where(
                    or_(
                        ModelReleaseRecord.source_training_job_id == job_id,
                        ModelReleaseRecord.source_conversion_job_id == job_id,
                        ModelReleaseRecord.rknn_artifact_id.in_(artifact_ids),
                        ModelReleaseRecord.validation_artifact_id.in_(artifact_ids),
                    )
                )
            ).all()
            if release_references:
                raise ConflictError(
                    "job_artifacts_published",
                    "Job and artifacts are retained by model releases",
                    releaseIds=[item.id for item in release_references],
                )
            active_dependents = [
                candidate.id
                for candidate in session.scalars(select(JobRecord)).all()
                if candidate.id != job_id
                and JobStatus(candidate.status)
                in {JobStatus.QUEUED, JobStatus.CLAIMED, JobStatus.RUNNING}
                and isinstance(candidate.spec_json.get("sourceArtifact"), dict)
                and candidate.spec_json["sourceArtifact"].get("id") in artifact_ids
            ]
            if active_dependents:
                raise ConflictError(
                    "job_artifacts_in_use",
                    "Job artifacts are required by active conversion jobs",
                    jobIds=active_dependents,
                )
            storage_keys = [artifact.storage_key for artifact in artifacts]
            worker = session.get(WorkerRecord, record.worker_id) if record.worker_id else None
            endpoint = (
                session.scalar(
                    select(ServiceEndpointRecord).where(
                        ServiceEndpointRecord.name == worker.name,
                        ServiceEndpointRecord.mode == ServiceEndpointMode.DIRECT.value,
                    )
                )
                if worker is not None
                else None
            )
            if endpoint is not None:
                cleanup = NodeCleanupRecord(endpoint_id=endpoint.id, job_id=job_id)
                session.add(cleanup)
                session.flush()
                cleanup_id = cleanup.id
            for artifact in artifacts:
                session.delete(artifact)
            for event in session.scalars(
                select(JobEventRecord).where(JobEventRecord.job_id == job_id)
            ).all():
                session.delete(event)
            session.delete(record)
        for storage_key in storage_keys:
            self.context.storage.remove(storage_key)
        if cleanup_id is not None:
            self._attempt_node_cleanup(cleanup_id)

    def _attempt_node_cleanup(self, cleanup_id: int) -> None:
        with self.context.database.session() as session:
            cleanup = session.get(NodeCleanupRecord, cleanup_id)
            if cleanup is None:
                return
            endpoint = session.get(ServiceEndpointRecord, cleanup.endpoint_id)
            if endpoint is None:
                session.delete(cleanup)
                return
            endpoint_url = endpoint.endpoint
            endpoint_id = endpoint.id
            job_id = cleanup.job_id
        token = self.context.node_secrets.read(endpoint_id)
        if not token:
            return
        try:
            DirectNodeClient(endpoint_url, token).clean_job_cache(job_id)
        except NodeClientError as error:
            with self.context.database.session() as session:
                cleanup = session.get(NodeCleanupRecord, cleanup_id)
                if cleanup is not None:
                    cleanup.attempts += 1
                    cleanup.last_error = str(error)[:1000]
            return
        with self.context.database.session() as session:
            cleanup = session.get(NodeCleanupRecord, cleanup_id)
            if cleanup is not None:
                session.delete(cleanup)

    def job_events(
        self, job_id: str, *, after_id: int = 0, limit: int = 500
    ) -> list[JobEventResponse]:
        with self.context.database.session() as session:
            if session.get(JobRecord, job_id) is None:
                raise NotFoundError("job", job_id)
            records = session.scalars(
                select(JobEventRecord)
                .where(JobEventRecord.job_id == job_id, JobEventRecord.id > after_id)
                .order_by(JobEventRecord.id.asc())
                .limit(limit)
            ).all()
            return [
                JobEventResponse(
                    id=record.id,
                    type=record.type,
                    level=record.level,
                    message=record.message,
                    data=record.data_json,
                    created_at=record.created_at,
                )
                for record in records
            ]

    def register_worker(
        self,
        payload: WorkerRegistration,
        authenticated_endpoint_id: str | None = None,
    ) -> WorkerResponse:
        with self.context.database.session() as session:
            endpoint = session.scalar(
                select(ServiceEndpointRecord).where(ServiceEndpointRecord.name == payload.name)
            )
            if authenticated_endpoint_id is not None and (
                endpoint is None or endpoint.id != authenticated_endpoint_id
            ):
                raise AuthenticationError(
                    "This node token cannot register another service identity"
                )
            if endpoint is not None:
                self._validate_worker_registration(endpoint, payload)
            record = session.scalar(select(WorkerRecord).where(WorkerRecord.name == payload.name))
            now = utc_now()
            if record is None:
                record = WorkerRecord(
                    id=new_id("worker"),
                    name=payload.name,
                    created_at=now,
                )
                session.add(record)
            record.kind = payload.kind.value
            record.status = WorkerStatus.ONLINE.value
            record.capabilities_json = payload.capabilities
            record.accelerator = payload.accelerator
            record.max_concurrency = payload.max_concurrency
            record.version = payload.version
            record.metadata_json = payload.metadata
            record.last_seen_at = now
            session.flush()
            return worker_response(record)

    @staticmethod
    def require_worker_identity(
        session: Session, worker_id: str, authenticated_endpoint_id: str | None
    ) -> None:
        if authenticated_endpoint_id is None:
            return
        worker = session.get(WorkerRecord, worker_id)
        endpoint = session.get(ServiceEndpointRecord, authenticated_endpoint_id)
        if worker is None or endpoint is None or worker.name != endpoint.name:
            raise AuthenticationError(
                "This node token cannot operate another worker identity"
            )

    @staticmethod
    def _worker_for_endpoint(session: Session, endpoint_id: str) -> WorkerRecord:
        endpoint = session.get(ServiceEndpointRecord, endpoint_id)
        if endpoint is None:
            raise AuthenticationError("The configured node identity no longer exists")
        worker = session.scalar(
            select(WorkerRecord).where(WorkerRecord.name == endpoint.name)
        )
        if worker is None:
            raise AuthenticationError("The configured node has not registered its worker yet")
        return worker

    def require_worker_dataset_access(
        self, dataset_id: str, authenticated_endpoint_id: str | None
    ) -> None:
        if authenticated_endpoint_id is None:
            return
        with self.context.database.session() as session:
            worker = self._worker_for_endpoint(session, authenticated_endpoint_id)
            assigned = session.scalar(
                select(JobRecord.id).where(
                    JobRecord.worker_id == worker.id,
                    JobRecord.dataset_id == dataset_id,
                    JobRecord.status.in_(
                        {JobStatus.CLAIMED.value, JobStatus.RUNNING.value}
                    ),
                )
            )
            if assigned is None:
                raise AuthenticationError(
                    "This node token cannot access a dataset outside its active jobs"
                )

    def require_worker_artifact_access(
        self, artifact_id: str, authenticated_endpoint_id: str | None
    ) -> None:
        if authenticated_endpoint_id is None:
            return
        with self.context.database.session() as session:
            worker = self._worker_for_endpoint(session, authenticated_endpoint_id)
            jobs = session.scalars(
                select(JobRecord).where(
                    JobRecord.worker_id == worker.id,
                    JobRecord.status.in_(
                        {JobStatus.CLAIMED.value, JobStatus.RUNNING.value}
                    ),
                )
            ).all()
            for job in jobs:
                source = job.spec_json.get("sourceArtifact")
                if isinstance(source, dict) and cast(dict[str, object], source).get(
                    "id"
                ) == artifact_id:
                    return
            raise AuthenticationError(
                "This node token cannot access a model outside its active jobs"
            )

    def list_service_endpoints(self) -> list[ServiceEndpointResponse]:
        with self.context.database.session() as session:
            records = session.scalars(
                select(ServiceEndpointRecord).order_by(ServiceEndpointRecord.created_at.desc())
            ).all()
            return [service_endpoint_response(record) for record in records]

    def create_service_endpoint(
        self, payload: ServiceEndpointPayload
    ) -> ServiceEndpointCreateResponse:
        self._validate_service_endpoint(payload)
        remote: dict[str, object] = {}
        if payload.token:
            self._validate_node_token_available(payload.token)
        if (
            payload.mode == ServiceEndpointMode.DIRECT
            and payload.enabled
            and payload.token
        ):
            remote = self._probe_payload(payload)
        with self.context.database.session() as session:
            existing = session.scalar(
                select(ServiceEndpointRecord).where(ServiceEndpointRecord.name == payload.name)
            )
            if existing is not None:
                raise ConflictError(
                    "service_endpoint_name_exists",
                    "A service endpoint with this node name already exists",
                )
            endpoint_id = new_id("service")
            record = ServiceEndpointRecord(
                id=endpoint_id,
                name=payload.name,
                kind=payload.kind.value,
                endpoint=payload.endpoint,
                mode=payload.mode.value,
                scheme=payload.scheme,
                host=payload.host,
                port=payload.port,
                accelerator=payload.accelerator,
                capabilities_json=payload.capabilities,
                enabled=payload.enabled,
                token_configured=bool(payload.token),
                enrollment_status=(
                    ServiceEndpointEnrollmentStatus.PENDING.value
                    if payload.mode == ServiceEndpointMode.DIRECT and not payload.token
                    else ServiceEndpointEnrollmentStatus.ENROLLED.value
                ),
                enrolled_at=(
                    utc_now()
                    if payload.mode == ServiceEndpointMode.DIRECT and payload.token
                    else None
                ),
                probe_status=(
                    "online"
                    if payload.mode == ServiceEndpointMode.DIRECT
                    and payload.enabled
                    and payload.token
                    else "unprobed"
                ),
                last_probe_at=(
                    utc_now()
                    if payload.mode == ServiceEndpointMode.DIRECT
                    and payload.enabled
                    and payload.token
                    else None
                ),
                remote_metadata_json=remote,
            )
            session.add(record)
            session.flush()
            if payload.kind == ServiceEndpointKind.INFERENCE:
                from .inference_service import InferenceService

                inference_service = InferenceService(self.context)
                if payload.token:
                    node, agent_token = inference_service.create_direct_node(
                        session,
                        name=payload.name,
                        hardware_id=f"direct:{endpoint_id}",
                        adapters=payload.capabilities,
                        max_model_instances=_positive_int(remote.get("maxConcurrency")),
                        metadata=cast(dict[str, Any], remote),
                        enabled=payload.enabled,
                    )
                    self.context.node_secrets.write(
                        endpoint_id, agent_token, purpose="agent"
                    )
                else:
                    node = inference_service.create_pending_direct_node(
                        session,
                        name=payload.name,
                        hardware_id=f"direct:{endpoint_id}",
                        adapters=payload.capabilities,
                        enabled=payload.enabled,
                    )
                record.inference_node_id = node.id
            if payload.token:
                self.context.node_secrets.write(endpoint_id, payload.token)
            response = service_endpoint_response(record)
            enrollment_token: str | None = None
            if (
                payload.mode == ServiceEndpointMode.DIRECT
                and record.enrollment_status
                == ServiceEndpointEnrollmentStatus.PENDING.value
            ):
                from .node_enrollment import NodeEnrollmentService

                issued = NodeEnrollmentService(self.context).issue_record(
                    record, session=session
                )
                enrollment_token = issued.enrollment_token
                response = service_endpoint_response(record)
            return ServiceEndpointCreateResponse(
                **response.model_dump(), enrollment_token=enrollment_token
            )

    def update_service_endpoint(
        self, endpoint_id: str, payload: ServiceEndpointPayload
    ) -> ServiceEndpointResponse:
        self._validate_service_endpoint(payload)
        with self.context.database.session() as session:
            record = session.get(ServiceEndpointRecord, endpoint_id)
            if record is None:
                raise NotFoundError("service endpoint", endpoint_id)
            duplicate = session.scalar(
                select(ServiceEndpointRecord).where(
                    ServiceEndpointRecord.name == payload.name,
                    ServiceEndpointRecord.id != endpoint_id,
                )
            )
            if duplicate is not None:
                raise ConflictError(
                    "service_endpoint_name_exists",
                    "A service endpoint with this node name already exists",
                )
            if payload.kind.value != record.kind:
                raise ConflictError(
                    "service_kind_change_forbidden",
                    "Node type is immutable; create another node to change its type",
                )
            if payload.name != record.name:
                raise ConflictError(
                    "service_name_change_forbidden",
                    "Node name is immutable; create another node to change its name",
                )
            worker = session.scalar(
                select(WorkerRecord).where(WorkerRecord.name == record.name)
            )
            if worker is not None and worker.active_jobs > 0 and (
                payload.name != record.name
                or payload.kind.value != record.kind
                or payload.mode.value != record.mode
                or payload.accelerator != record.accelerator
                or payload.capabilities != record.capabilities_json
                or payload.enabled != record.enabled
                or payload.token is not None
            ):
                raise ConflictError(
                    "service_endpoint_busy",
                    "A busy node cannot change its execution configuration",
                )
            if payload.token:
                self._validate_node_token_available(payload.token, endpoint_id=endpoint_id)
            token = payload.token or self.context.node_secrets.read(endpoint_id)
            remote: dict[str, object] = {}
            if payload.mode == ServiceEndpointMode.DIRECT and payload.enabled:
                if not token:
                    raise ConflictError(
                        "node_token_required",
                        "A token is required when configuring a direct node",
                    )
                remote = self._probe_payload(payload, token=token)
            record.name = payload.name
            record.kind = payload.kind.value
            record.endpoint = payload.endpoint
            record.mode = payload.mode.value
            record.scheme = payload.scheme
            record.host = payload.host
            record.port = payload.port
            record.accelerator = payload.accelerator
            record.capabilities_json = payload.capabilities
            record.enabled = payload.enabled
            record.token_configured = bool(token)
            record.probe_status = (
                "online"
                if payload.mode == ServiceEndpointMode.DIRECT and payload.enabled
                else "unprobed"
            )
            record.last_probe_at = (
                utc_now()
                if payload.mode == ServiceEndpointMode.DIRECT and payload.enabled
                else record.last_probe_at
            )
            record.last_error = None
            if remote:
                record.remote_metadata_json = remote
            if payload.kind == ServiceEndpointKind.INFERENCE:
                from .inference_service import InferenceService

                if record.inference_node_id is None:
                    node, agent_token = InferenceService(self.context).create_direct_node(
                        session,
                        name=payload.name,
                        hardware_id=f"direct:{endpoint_id}",
                        adapters=payload.capabilities,
                        max_model_instances=_positive_int(remote.get("maxConcurrency")),
                        metadata=cast(
                            dict[str, Any], remote or record.remote_metadata_json or {}
                        ),
                        enabled=payload.enabled,
                    )
                    record.inference_node_id = node.id
                    self.context.node_secrets.write(
                        endpoint_id, agent_token, purpose="agent"
                    )
                else:
                    InferenceService(self.context).update_direct_node(
                        session,
                        record.inference_node_id,
                        name=payload.name,
                        adapters=payload.capabilities,
                        max_model_instances=_positive_int(remote.get("maxConcurrency")),
                        metadata=cast(dict[str, Any], remote),
                        enabled=payload.enabled,
                    )
            elif record.inference_node_id is not None:
                raise ConflictError(
                    "service_kind_change_forbidden",
                    "An inference service with history cannot be changed to another kind",
                )
            session.flush()
            if payload.token:
                self.context.node_secrets.write(endpoint_id, payload.token)
            return service_endpoint_response(record)

    def delete_service_endpoint(self, endpoint_id: str) -> None:
        with self.context.database.session() as session:
            record = session.get(ServiceEndpointRecord, endpoint_id)
            if record is None:
                raise NotFoundError("service endpoint", endpoint_id)
            worker = session.scalar(
                select(WorkerRecord).where(WorkerRecord.name == record.name)
            )
            if worker is not None and worker.active_jobs > 0:
                raise ConflictError(
                    "service_endpoint_busy", "A node with active jobs cannot be deleted"
                )
            if record.inference_node_id is not None:
                from .inference_service import InferenceService

                InferenceService(self.context).retire_direct_node(
                    session, record.inference_node_id
                )
            for cleanup in session.scalars(
                select(NodeCleanupRecord).where(NodeCleanupRecord.endpoint_id == endpoint_id)
            ).all():
                session.delete(cleanup)
            session.delete(record)
        self.context.node_secrets.delete(endpoint_id)
        self.context.node_secrets.delete(endpoint_id, purpose="agent")

    def test_service_endpoint(
        self, payload: ServiceEndpointPayload
    ) -> ServiceEndpointTestResponse:
        self._validate_service_endpoint(payload)
        if payload.mode != ServiceEndpointMode.DIRECT:
            return ServiceEndpointTestResponse(
                ok=True,
                endpoint=payload.endpoint,
                message="pull endpoints are validated when their Worker registers",
            )
        token = payload.token
        if not token:
            raise ConflictError("node_token_required", "A token is required to test a direct node")
        remote = self._probe_payload(payload, token=token)
        return ServiceEndpointTestResponse(
            ok=True,
            endpoint=payload.endpoint,
            message="node is healthy and compatible",
            remote=remote,
        )

    def test_service_endpoint_update(
        self, endpoint_id: str, payload: ServiceEndpointPayload
    ) -> ServiceEndpointTestResponse:
        self._validate_service_endpoint(payload)
        with self.context.database.session() as session:
            record = session.get(ServiceEndpointRecord, endpoint_id)
            if record is None:
                raise NotFoundError("service endpoint", endpoint_id)
            if payload.kind.value != record.kind:
                raise ConflictError(
                    "service_kind_change_forbidden",
                    "Node type is immutable; create another node to change its type",
                )
            if payload.name != record.name:
                raise ConflictError(
                    "service_name_change_forbidden",
                    "Node name is immutable; create another node to change its name",
                )
        if payload.mode != ServiceEndpointMode.DIRECT:
            return ServiceEndpointTestResponse(
                ok=True,
                endpoint=payload.endpoint,
                message="pull endpoints are validated when their Worker registers",
            )
        token = payload.token or self.context.node_secrets.read(endpoint_id)
        if not token:
            raise ConflictError("node_token_required", "A token is required to test this node")
        remote = self._probe_payload(payload, token=token)
        return ServiceEndpointTestResponse(
            ok=True,
            endpoint=payload.endpoint,
            message="node is healthy and compatible",
            remote=remote,
        )

    def probe_service_endpoint(self, endpoint_id: str) -> ServiceEndpointResponse:
        with self.context.database.session() as session:
            record = session.get(ServiceEndpointRecord, endpoint_id)
            if record is None:
                raise NotFoundError("service endpoint", endpoint_id)
            if record.mode != ServiceEndpointMode.DIRECT.value:
                return service_endpoint_response(record)
            if (
                record.enrollment_status
                == ServiceEndpointEnrollmentStatus.PENDING.value
            ):
                return service_endpoint_response(record)
            token = self.context.node_secrets.read(endpoint_id)
            if not token:
                record.probe_status = "error"
                record.last_error = "node token is not configured"
                record.last_probe_at = utc_now()
                session.flush()
                return service_endpoint_response(record)
            try:
                remote = self._probe_record(record, token)
            except NodeClientError as error:
                from .direct_node_lifecycle import record_probe_failure

                record_probe_failure(self.context, endpoint_id, str(error))
            except ConflictError as error:
                from .direct_node_lifecycle import record_probe_failure

                record_probe_failure(self.context, endpoint_id, error.message)
                raise
            else:
                from .direct_node_lifecycle import record_probe_success

                record_probe_success(self.context, endpoint_id, remote)
        with self.context.database.session() as session:
            refreshed = session.get(ServiceEndpointRecord, endpoint_id)
            if refreshed is None:
                raise NotFoundError("service endpoint", endpoint_id)
            return service_endpoint_response(refreshed)

    def _validate_node_token_available(
        self, token: str, *, endpoint_id: str | None = None
    ) -> None:
        protected_tokens = (
            self.context.settings.admin_token.get_secret_value(),
            self.context.settings.worker_token.get_secret_value(),
        )
        if any(hmac.compare_digest(token, protected) for protected in protected_tokens):
            raise ConflictError(
                "node_token_conflict",
                "A node token must differ from the admin and global worker tokens",
            )
        matched_endpoint_id = self.context.node_secrets.matching_endpoint_id(token)
        if matched_endpoint_id is not None and matched_endpoint_id != endpoint_id:
            raise ConflictError(
                "node_token_conflict", "Each node must use an independent token"
            )

    def _validate_service_endpoint(self, payload: ServiceEndpointPayload) -> None:
        if payload.kind in {ServiceEndpointKind.TRAINER, ServiceEndpointKind.CONVERTER}:
            for profile_id in payload.capabilities:
                self.context.profiles.get(profile_id)

    def _probe_payload(
        self, payload: ServiceEndpointPayload, *, token: str | None = None
    ) -> dict[str, object]:
        try:
            health = DirectNodeClient(
                payload.endpoint, token or payload.token or ""
            ).health()
        except NodeClientError as error:
            raise AppError(
                "node_unreachable",
                f"Cannot reach direct node at {payload.endpoint}",
                status_code=502,
            ) from error
        return self._validate_remote_health(health, payload)

    def _probe_record(self, record: ServiceEndpointRecord, token: str) -> dict[str, object]:
        remote = DirectNodeClient(record.endpoint, token).health()
        expected = ServiceEndpointPayload(
            name=record.name,
            kind=ServiceEndpointKind(record.kind),
            mode=ServiceEndpointMode(record.mode),
            endpoint=record.endpoint,
            scheme=cast(Literal["http", "https"], record.scheme),
            host=record.host,
            port=record.port,
            accelerator=record.accelerator,  # type: ignore[arg-type]
            capabilities=record.capabilities_json,
            enabled=record.enabled,
        )
        return self._validate_remote_health(remote, expected)

    @staticmethod
    def _validate_remote_health(
        remote: dict[str, Any], payload: ServiceEndpointPayload
    ) -> dict[str, object]:
        if remote.get("status") != "healthy":
            raise ConflictError("node_unhealthy", "Remote node health check did not pass")
        if remote.get("name") != payload.name:
            raise ConflictError(
                "node_name_mismatch", "Remote node name does not match configuration"
            )
        if remote.get("kind") != payload.kind.value:
            raise ConflictError(
                "node_kind_mismatch", "Remote node kind does not match configuration"
            )
        if remote.get("accelerator") != payload.accelerator:
            raise ConflictError(
                "node_accelerator_mismatch",
                "Remote node accelerator does not match configuration",
            )
        raw_capabilities = remote.get("capabilities")
        remote_capabilities = (
            cast(list[object], raw_capabilities) if isinstance(raw_capabilities, list) else []
        )
        if not set(payload.capabilities).issubset({str(item) for item in remote_capabilities}):
            raise ConflictError(
                "node_capability_mismatch",
                "Remote node does not provide all capabilities",
            )
        return cast(dict[str, object], remote)

    @staticmethod
    def _validate_worker_registration(
        endpoint: ServiceEndpointRecord, payload: WorkerRegistration
    ) -> None:
        if endpoint.kind != payload.kind.value or endpoint.accelerator != payload.accelerator:
            raise ConflictError(
                "worker_service_mismatch",
                "Worker kind or accelerator differs from its system service configuration",
            )
        if set(endpoint.capabilities_json) != set(payload.capabilities):
            raise ConflictError(
                "worker_service_mismatch",
                "Worker capabilities differ from its system service configuration",
            )

    def heartbeat(self, worker_id: str, payload: WorkerHeartbeat) -> WorkerResponse:
        with self.context.database.session() as session:
            worker = session.get(WorkerRecord, worker_id)
            if worker is None:
                raise NotFoundError("worker", worker_id)
            worker.active_jobs = payload.active_jobs
            worker.metadata_json = {**worker.metadata_json, **payload.metadata}
            worker.status = (
                WorkerStatus.BUSY.value if payload.active_jobs else WorkerStatus.ONLINE.value
            )
            worker.last_seen_at = utc_now()
            session.flush()
            return worker_response(worker)

    def list_workers(self) -> list[WorkerResponse]:
        with self.context.database.session() as session:
            records = session.scalars(
                select(WorkerRecord).order_by(WorkerRecord.created_at.asc())
            ).all()
            now = utc_now()
            responses: list[WorkerResponse] = []
            for record in records:
                response = worker_response(record)
                age = (now - as_utc(record.last_seen_at)).total_seconds()
                if age > self.context.settings.worker_offline_seconds:
                    response = response.model_copy(update={"status": WorkerStatus.OFFLINE})
                responses.append(response)
            return responses

    def delete_worker(self, worker_id: str) -> None:
        with self.context.database.session() as session:
            worker = session.get(WorkerRecord, worker_id)
            if worker is None:
                raise NotFoundError("worker", worker_id)
            now = utc_now()
            age = (now - as_utc(worker.last_seen_at)).total_seconds()
            active_jobs = session.scalars(
                select(JobRecord).where(
                    JobRecord.worker_id == worker_id,
                    JobRecord.status.in_([JobStatus.CLAIMED.value, JobStatus.RUNNING.value]),
                )
            ).all()
            if active_jobs:
                raise ConflictError(
                    "worker_has_active_jobs",
                    "Worker has claimed or running jobs and cannot be deleted",
                    jobIds=[job.id for job in active_jobs],
                )
            if age <= self.context.settings.worker_offline_seconds:
                raise ConflictError(
                    "worker_not_offline",
                    "Only offline workers can be deleted",
                    lastSeenAt=worker.last_seen_at.isoformat(),
                )
            historical_jobs = session.scalars(
                select(JobRecord).where(JobRecord.worker_id == worker_id)
            ).all()
            for job in historical_jobs:
                job.worker_id = None
            session.delete(worker)

    async def create_artifact(
        self,
        job_id: str,
        lease_token: str,
        kind: str,
        manifest_json: str | None,
        upload: UploadFile,
    ) -> ArtifactResponse:
        manifest: DeploymentManifest | None = None
        if manifest_json:
            try:
                manifest = DeploymentManifest.model_validate_json(manifest_json)
            except ValueError as error:
                raise AppError("invalid_deployment_manifest", str(error)) from error
            self.context.profiles.validate_manifest(manifest)
        if kind == "onnx" and manifest is None:
            raise AppError("deployment_manifest_required", "ONNX artifacts require a manifest")
        with self.context.database.session() as session:
            self.context.jobs.require_lease(session, job_id, lease_token)
        stored = await self.context.storage.write_upload(upload, "artifacts")
        if manifest and manifest.onnx_sha256 != stored.sha256:
            self.context.storage.resolve(stored.storage_key).unlink(missing_ok=True)
            raise AppError(
                "artifact_checksum_mismatch",
                "Uploaded artifact checksum does not match deployment manifest",
            )
        with self.context.database.session() as session:
            self.context.jobs.require_lease(session, job_id, lease_token)
            record = ArtifactRecord(
                id=new_id("artifact"),
                job_id=job_id,
                kind=kind,
                filename=stored.filename,
                storage_key=stored.storage_key,
                media_type=upload.content_type or "application/octet-stream",
                size_bytes=stored.size_bytes,
                sha256=stored.sha256,
                manifest_json=(
                    manifest.model_dump(mode="json", by_alias=True) if manifest else None
                ),
            )
            session.add(record)
            session.flush()
            return artifact_response(record)

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        with self.context.database.session() as session:
            record = session.get(ArtifactRecord, artifact_id)
            if record is None:
                raise NotFoundError("artifact", artifact_id)
            session.expunge(record)
            return record

    def list_artifacts(
        self,
        *,
        job_id: str | None = None,
        kind: str | None = None,
    ) -> list[ArtifactResponse]:
        with self.context.database.session() as session:
            statement = select(ArtifactRecord)
            if job_id:
                statement = statement.where(ArtifactRecord.job_id == job_id)
            if kind:
                statement = statement.where(ArtifactRecord.kind == kind)
            records = session.scalars(statement.order_by(ArtifactRecord.created_at.desc())).all()
            return [artifact_response(record) for record in records]

    @staticmethod
    def _supported_archive_name(filename: str) -> bool:
        lowered = filename.lower()
        return lowered.endswith((".zip", ".tar.gz", ".tgz"))

    @staticmethod
    def _valid_archive(path: Path) -> bool:
        return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)
