from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import select

from .context import AppContext
from .contracts import (
    JobStatus,
    JobType,
    ServiceEndpointEnrollmentStatus,
    ServiceEndpointKind,
)
from .db_models import (
    JobRecord,
    NodeCleanupRecord,
    ServiceEndpointRecord,
)
from .direct_node_lifecycle import record_probe_failure, record_probe_success
from .inference_service import InferenceService
from .node_client import DirectNodeClient, NodeClientError

LOGGER = logging.getLogger("rknode.direct-dispatcher")


@dataclass(frozen=True)
class EndpointSnapshot:
    id: str
    name: str
    kind: str
    endpoint: str
    accelerator: str
    capabilities: tuple[str, ...]
    inference_node_id: str | None
    enrollment_status: str


class DirectNodeDispatcher:
    def __init__(self, context: AppContext) -> None:
        self.context = context
        self._inflight_jobs: dict[str, float] = {}

    async def run(self, stop: asyncio.Event) -> None:
        interval = max(1.0, self.context.settings.direct_dispatch_interval_seconds)
        while not stop.is_set():
            try:
                await asyncio.to_thread(self.run_once)
            except Exception:
                LOGGER.exception("direct dispatcher iteration failed")
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval)

    def run_once(self) -> None:
        self._reconcile_inflight_jobs()
        assigned_jobs: set[str] = set()
        for endpoint in self._endpoints():
            token = self.context.node_secrets.read(endpoint.id)
            if not token:
                self._record_probe_failure(endpoint, "node token is not configured")
                continue
            client = DirectNodeClient(
                endpoint.endpoint,
                token,
                timeout=self.context.settings.direct_node_timeout_seconds,
            )
            try:
                health = client.health()
                self._validate_health(endpoint, health)
            except (NodeClientError, ValueError) as error:
                self._record_probe_failure(endpoint, str(error))
                continue
            record_probe_success(self.context, endpoint.id, health)
            if (
                endpoint.enrollment_status
                == ServiceEndpointEnrollmentStatus.CLAIMED.value
            ):
                continue
            self._process_cleanups(endpoint, client)
            try:
                if endpoint.kind == ServiceEndpointKind.INFERENCE.value:
                    self._dispatch_inference(endpoint, client, health)
                else:
                    self._dispatch_jobs(endpoint, client, health, assigned_jobs)
            except NodeClientError as error:
                self._record_probe_failure(endpoint, str(error))

    def _endpoints(self) -> list[EndpointSnapshot]:
        with self.context.database.session() as session:
            records = session.scalars(
                select(ServiceEndpointRecord)
                .where(
                    ServiceEndpointRecord.mode == "direct",
                    ServiceEndpointRecord.enabled.is_(True),
                    ServiceEndpointRecord.enrollment_status.in_(
                        {
                            ServiceEndpointEnrollmentStatus.CLAIMED.value,
                            ServiceEndpointEnrollmentStatus.ENROLLED.value,
                        }
                    ),
                )
                .order_by(ServiceEndpointRecord.created_at)
            ).all()
            return [
                EndpointSnapshot(
                    id=record.id,
                    name=record.name,
                    kind=record.kind,
                    endpoint=record.endpoint,
                    accelerator=record.accelerator,
                    capabilities=tuple(record.capabilities_json),
                    inference_node_id=record.inference_node_id,
                    enrollment_status=record.enrollment_status,
                )
                for record in records
            ]

    def _dispatch_jobs(
        self,
        endpoint: EndpointSnapshot,
        client: DirectNodeClient,
        health: dict[str, Any],
        assigned_jobs: set[str],
    ) -> None:
        max_concurrency = self._integer(health.get("maxConcurrency"), 1)
        active_jobs = self._integer(health.get("activeJobs"), 0)
        capacity = max(0, max_concurrency - active_jobs)
        if capacity == 0:
            return
        for job in self._queued_jobs(endpoint, capacity):
            if job.id in assigned_jobs or job.id in self._inflight_jobs:
                continue
            try:
                client.dispatch(job.id)
            except NodeClientError as error:
                self._record_probe_failure(endpoint, str(error))
                return
            assigned_jobs.add(job.id)
            self._inflight_jobs[job.id] = time.monotonic()

    def _process_cleanups(
        self, endpoint: EndpointSnapshot, client: DirectNodeClient
    ) -> None:
        with self.context.database.session() as session:
            cleanups = session.scalars(
                select(NodeCleanupRecord)
                .where(NodeCleanupRecord.endpoint_id == endpoint.id)
                .order_by(NodeCleanupRecord.created_at)
                .limit(20)
            ).all()
            cleanup_items = [(item.id, item.job_id) for item in cleanups]
        for cleanup_id, job_id in cleanup_items:
            try:
                client.clean_job_cache(job_id)
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

    def _queued_jobs(self, endpoint: EndpointSnapshot, limit: int) -> list[JobRecord]:
        job_type = (
            JobType.TRAINING.value
            if endpoint.kind == ServiceEndpointKind.TRAINER.value
            else JobType.CONVERSION.value
        )
        with self.context.database.session() as session:
            candidates = session.scalars(
                select(JobRecord)
                .where(
                    JobRecord.type == job_type,
                    JobRecord.status == JobStatus.QUEUED.value,
                    JobRecord.profile_id.in_(endpoint.capabilities),
                )
                .order_by(JobRecord.created_at)
                .limit(max(limit * 4, limit))
            ).all()
            result = [
                candidate
                for candidate in candidates
                if job_type != JobType.TRAINING.value
                or candidate.spec_json.get("accelerator") == endpoint.accelerator
            ]
            for record in result:
                session.expunge(record)
            return result[:limit]

    def _dispatch_inference(
        self,
        endpoint: EndpointSnapshot,
        client: DirectNodeClient,
        health: dict[str, Any],
    ) -> None:
        node_id = endpoint.inference_node_id
        if node_id is None:
            return
        diagnostics = health.get("diagnostics")
        inference = (
            cast(dict[str, Any], diagnostics).get("inference")
            if isinstance(diagnostics, dict)
            else None
        )
        actual_revision = 0
        failed_revision: int | None = None
        if isinstance(inference, dict):
            actual_revision = self._integer(
                cast(dict[str, Any], inference).get("actualRevision"), 0
            )
            failed_revision = self._optional_integer(
                cast(dict[str, Any], inference).get("failedRevision")
            )
        desired_revision = InferenceService(self.context).reconcile_direct_revision(
            node_id,
            actual_revision=actual_revision,
            failed_revision=failed_revision,
        )
        if desired_revision == actual_revision:
            return
        if failed_revision == desired_revision:
            return
        agent_token = self.context.node_secrets.read(endpoint.id, purpose="agent")
        if not agent_token:
            self._record_probe_failure(endpoint, "inference access token is not configured")
            return
        desired = InferenceService(self.context).direct_desired_state(node_id)
        client.apply_inference_revision(
            desired.revision,
            node_id=node_id,
            central_api_url=self.context.settings.public_api_url,
            access_token=agent_token,
            desired=desired.model_dump(mode="json", by_alias=True),
        )

    def _record_probe_success(
        self, endpoint: EndpointSnapshot, health: dict[str, Any]
    ) -> None:
        record_probe_success(self.context, endpoint.id, health)

    def _record_probe_failure(self, endpoint: EndpointSnapshot, message: str) -> None:
        record_probe_failure(self.context, endpoint.id, message)

    def _reconcile_inflight_jobs(self) -> None:
        if not self._inflight_jobs:
            return
        with self.context.database.session() as session:
            queued_ids = set(
                session.scalars(
                    select(JobRecord.id).where(
                        JobRecord.id.in_(self._inflight_jobs.keys()),
                        JobRecord.status == JobStatus.QUEUED.value,
                    )
                ).all()
            )
        now = time.monotonic()
        self._inflight_jobs = {
            job_id: dispatched_at
            for job_id, dispatched_at in self._inflight_jobs.items()
            if job_id in queued_ids and now - dispatched_at < 15
        }

    @staticmethod
    def _validate_health(endpoint: EndpointSnapshot, health: dict[str, Any]) -> None:
        if health.get("status") != "healthy":
            raise ValueError("node health check did not pass")
        if health.get("name") != endpoint.name:
            raise ValueError("node name does not match configuration")
        if health.get("kind") != endpoint.kind:
            raise ValueError("node kind does not match configuration")
        if health.get("accelerator") != endpoint.accelerator:
            raise ValueError("node accelerator does not match configuration")
        raw_capabilities = health.get("capabilities")
        remote_capabilities: set[str] = set()
        if isinstance(raw_capabilities, list):
            remote_capabilities = {
                str(item) for item in cast(list[object], raw_capabilities)
            }
        if not set(endpoint.capabilities).issubset(remote_capabilities):
            raise ValueError("node capabilities do not match configuration")

    @staticmethod
    def _integer(value: object, default: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
        if isinstance(value, str) and value.isdigit():
            return max(0, int(value))
        return default

    @staticmethod
    def _optional_integer(value: object) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
        if isinstance(value, str) and value.isdigit():
            return max(0, int(value))
        return None
