from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import Final

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from .contracts import (
    JobClaim,
    JobCompletion,
    JobFailure,
    JobLeaseRenewal,
    JobProgressUpdate,
    JobResponse,
    JobStatus,
    JobTelemetryUpdate,
    JobType,
    WorkerKind,
    WorkerStatus,
)
from .db_models import JobEventRecord, JobRecord, ServiceEndpointRecord, WorkerRecord
from .errors import AppError, ConflictError, NotFoundError

TERMINAL_STATUSES: Final = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
ALLOWED_TRANSITIONS: Final[dict[JobStatus, set[JobStatus]]] = {
    JobStatus.QUEUED: {JobStatus.CLAIMED, JobStatus.CANCELLED},
    JobStatus.CLAIMED: {JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.RUNNING: {
        JobStatus.QUEUED,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.SUCCEEDED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def job_response(record: JobRecord) -> JobResponse:
    return JobResponse(
        id=record.id,
        type=JobType(record.type),
        name=record.name,
        status=JobStatus(record.status),
        profile_id=record.profile_id,
        dataset_id=record.dataset_id,
        worker_id=record.worker_id,
        progress=record.progress,
        stage=record.stage,
        spec=record.spec_json,
        result=record.result_json,
        retry_count=record.retry_count,
        max_retries=record.max_retries,
        error_code=record.error_code,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
    )


class JobStateMachine:
    def __init__(self, *, lease_seconds: int) -> None:
        self.lease_seconds = lease_seconds

    def transition(self, job: JobRecord, target: JobStatus) -> None:
        current = JobStatus(job.status)
        if target not in ALLOWED_TRANSITIONS[current]:
            raise ConflictError(
                "invalid_job_transition",
                f"Cannot transition job from {current.value} to {target.value}",
                current=current.value,
                target=target.value,
            )
        job.status = target.value
        job.row_version += 1
        job.updated_at = utc_now()

    def requeue_expired(self, session: Session) -> int:
        now = utc_now()
        statement = select(JobRecord).where(
            JobRecord.status.in_([JobStatus.CLAIMED.value, JobStatus.RUNNING.value]),
            JobRecord.lease_expires_at.is_not(None),
            JobRecord.lease_expires_at < now,
        )
        records = session.scalars(statement).all()
        for job in records:
            job.retry_count += 1
            self._release_worker(session, job)
            job.worker_id = None
            if job.retry_count > job.max_retries:
                self.transition(job, JobStatus.FAILED)
                job.error_code = "worker_lease_exhausted"
                job.error_message = "Worker lease expired too many times"
                job.completed_at = now
            else:
                self.transition(job, JobStatus.QUEUED)
                job.stage = "requeued"
                job.progress = 0
                job.error_code = None
                job.error_message = None
            self._event(
                session,
                job,
                "lease_expired",
                "warning",
                "Worker lease expired",
                {"retryCount": job.retry_count},
            )
        return len(records)

    def claim(
        self,
        session: Session,
        worker_id: str,
        job_id: str | None = None,
    ) -> JobClaim | None:
        self.requeue_expired(session)
        worker = session.get(WorkerRecord, worker_id)
        if worker is None:
            raise NotFoundError("worker", worker_id)
        endpoint = session.scalar(
            select(ServiceEndpointRecord).where(ServiceEndpointRecord.name == worker.name)
        )
        if endpoint is not None and not endpoint.enabled:
            return None
        if worker.active_jobs >= worker.max_concurrency:
            return None
        job_type = (
            JobType.TRAINING if worker.kind == WorkerKind.TRAINER.value else JobType.CONVERSION
        )
        statement: Select[tuple[JobRecord]] = (
            select(JobRecord)
            .where(JobRecord.type == job_type.value, JobRecord.status == JobStatus.QUEUED.value)
            .order_by(JobRecord.created_at.asc())
        )
        if job_id is not None:
            statement = statement.where(JobRecord.id == job_id)
        candidates = session.scalars(statement).all()
        job = next((item for item in candidates if self._compatible(item, worker)), None)
        if job is None:
            return None
        lease_token = secrets.token_urlsafe(32)
        expires_at = utc_now() + timedelta(seconds=self.lease_seconds)
        self.transition(job, JobStatus.CLAIMED)
        job.worker_id = worker.id
        job.lease_token_hash = hash_token(lease_token)
        job.lease_expires_at = expires_at
        job.stage = "claimed"
        worker.active_jobs += 1
        worker.status = WorkerStatus.BUSY.value
        worker.last_seen_at = utc_now()
        self._event(session, job, "claimed", "info", f"Claimed by worker {worker.name}")
        session.flush()
        return JobClaim(job=job_response(job), lease_token=lease_token, lease_expires_at=expires_at)

    def progress(self, session: Session, job_id: str, update: JobProgressUpdate) -> JobRecord:
        job = self._require_leased_job(session, job_id, update.lease_token)
        if JobStatus(job.status) == JobStatus.CLAIMED:
            self.transition(job, JobStatus.RUNNING)
            job.started_at = job.started_at or utc_now()
        if update.progress < job.progress:
            raise ConflictError(
                "progress_regression",
                "Job progress cannot move backwards",
                current=job.progress,
                requested=update.progress,
            )
        job.progress = update.progress
        job.stage = update.stage
        job.lease_expires_at = utc_now() + timedelta(seconds=self.lease_seconds)
        self._event(
            session,
            job,
            "progress",
            "info",
            update.message,
            {"progress": update.progress, "stage": update.stage, "metrics": update.metrics},
        )
        return job

    def require_lease(self, session: Session, job_id: str, lease_token: str) -> JobRecord:
        return self._require_leased_job(session, job_id, lease_token)

    def telemetry(self, session: Session, job_id: str, update: JobTelemetryUpdate) -> int:
        job = self._require_leased_job(session, job_id, update.lease_token)
        job.lease_expires_at = utc_now() + timedelta(seconds=self.lease_seconds)
        job.updated_at = utc_now()
        for entry in update.entries:
            data: dict[str, object] = {
                "stage": entry.stage,
                "metrics": entry.metrics,
            }
            if entry.step is not None:
                data["step"] = entry.step
            if entry.epoch is not None:
                data["epoch"] = entry.epoch
            if entry.total_epochs is not None:
                data["totalEpochs"] = entry.total_epochs
            self._event(
                session,
                job,
                entry.type,
                entry.level,
                entry.message,
                data,
            )
        return len(update.entries)

    def renew(self, session: Session, job_id: str, payload: JobLeaseRenewal) -> JobRecord:
        job = self._require_leased_job(session, job_id, payload.lease_token)
        job.lease_expires_at = utc_now() + timedelta(seconds=self.lease_seconds)
        job.updated_at = utc_now()
        worker = session.get(WorkerRecord, job.worker_id) if job.worker_id else None
        if worker:
            worker.last_seen_at = utc_now()
            worker.status = WorkerStatus.BUSY.value
        return job

    def complete(self, session: Session, job_id: str, payload: JobCompletion) -> JobRecord:
        job = self._require_leased_job(session, job_id, payload.lease_token)
        if JobStatus(job.status) == JobStatus.CLAIMED:
            self.transition(job, JobStatus.RUNNING)
        self.transition(job, JobStatus.SUCCEEDED)
        job.progress = 100
        job.stage = "completed"
        job.result_json = payload.result
        job.completed_at = utc_now()
        self._release_worker(session, job)
        self._event(session, job, "completed", "info", "Job completed")
        return job

    def fail(self, session: Session, job_id: str, payload: JobFailure) -> JobRecord:
        job = self._require_leased_job(session, job_id, payload.lease_token)
        job.error_code = payload.code
        job.error_message = payload.message
        self._release_worker(session, job)
        if payload.retryable and job.retry_count < job.max_retries:
            job.retry_count += 1
            self.transition(job, JobStatus.QUEUED)
            job.stage = "retry_queued"
            job.progress = 0
            job.worker_id = None
            job.error_code = None
            job.error_message = None
        else:
            self.transition(job, JobStatus.FAILED)
            job.stage = "failed"
            job.completed_at = utc_now()
        self._event(
            session,
            job,
            "failed",
            "error",
            payload.message,
            {"code": payload.code, "retryable": payload.retryable},
        )
        return job

    @staticmethod
    def _compatible(job: JobRecord, worker: WorkerRecord) -> bool:
        if job.profile_id not in worker.capabilities_json:
            return False
        if job.type == JobType.TRAINING.value:
            return job.spec_json.get("accelerator") == worker.accelerator
        return worker.accelerator == "rk3588"

    @staticmethod
    def _event(
        session: Session,
        job: JobRecord,
        event_type: str,
        level: str,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        session.add(
            JobEventRecord(
                job_id=job.id,
                type=event_type,
                level=level,
                message=message,
                data_json=data or {},
            )
        )

    @staticmethod
    def _release_worker(session: Session, job: JobRecord) -> None:
        worker = session.get(WorkerRecord, job.worker_id) if job.worker_id else None
        if worker:
            worker.active_jobs = max(0, worker.active_jobs - 1)
            worker.status = WorkerStatus.ONLINE.value
            worker.last_seen_at = utc_now()
        job.lease_token_hash = None
        job.lease_expires_at = None

    @staticmethod
    def _require_leased_job(session: Session, job_id: str, lease_token: str) -> JobRecord:
        job = session.get(JobRecord, job_id)
        if job is None:
            raise NotFoundError("job", job_id)
        if JobStatus(job.status) not in {JobStatus.CLAIMED, JobStatus.RUNNING}:
            raise ConflictError("job_not_leased", "Job is not currently leased")
        if not job.lease_token_hash or not hmac.compare_digest(
            job.lease_token_hash, hash_token(lease_token)
        ):
            raise AppError("invalid_lease_token", "Invalid job lease token", status_code=403)
        if job.lease_expires_at is None or as_utc(job.lease_expires_at) < utc_now():
            raise ConflictError("lease_expired", "Job lease has expired")
        return job
