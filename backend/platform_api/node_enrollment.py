from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .context import AppContext
from .contracts import (
    InferenceNodeConnectivity,
    InferenceNodeHealth,
    NodeEnrollmentClaim,
    NodeEnrollmentClaimResponse,
    ServiceEndpointEnrollmentResponse,
    ServiceEndpointEnrollmentStatus,
    ServiceEndpointMode,
)
from .db_models import InferenceNodeRecord, ServiceEndpointRecord, WorkerRecord
from .errors import AppError, ConflictError, NotFoundError
from .state_machine import as_utc, utc_now

_CLAIM_LOCK = threading.Lock()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class NodeEnrollmentService:
    def __init__(self, context: AppContext) -> None:
        self.context = context

    def issue(self, endpoint_id: str) -> ServiceEndpointEnrollmentResponse:
        with _CLAIM_LOCK, self.context.database.session() as session:
            record = session.get(ServiceEndpointRecord, endpoint_id)
            if record is None:
                raise NotFoundError("service endpoint", endpoint_id)
            return self.issue_record(record, session=session)

    def issue_record(
        self, record: ServiceEndpointRecord, *, session: Session | None = None
    ) -> ServiceEndpointEnrollmentResponse:
        if record.mode != ServiceEndpointMode.DIRECT.value:
            raise ConflictError(
                "node_enrollment_not_supported",
                "Only direct service endpoints use node enrollment",
            )
        if record.enrollment_status == ServiceEndpointEnrollmentStatus.ENROLLED.value:
            if record.enrollment_claimed_at is not None:
                raise ConflictError(
                    "node_enrollment_already_active",
                    "This node endpoint is already enrolled",
                )
            if session is None:
                raise RuntimeError("legacy node migration requires a database session")
            self._prepare_legacy_migration(record, session)
        token = secrets.token_urlsafe(48)
        expires_at = utc_now() + timedelta(
            seconds=self.context.settings.node_enrollment_ttl_seconds
        )
        record.enrollment_token_hash = _token_hash(token)
        record.enrollment_expires_at = expires_at
        return ServiceEndpointEnrollmentResponse(
            endpoint_id=record.id,
            enrollment_status=ServiceEndpointEnrollmentStatus(
                record.enrollment_status
            ),
            enrollment_token=token,
            enrollment_expires_at=expires_at,
        )

    @staticmethod
    def _prepare_legacy_migration(
        record: ServiceEndpointRecord, session: Session
    ) -> None:
        worker = session.scalar(
            select(WorkerRecord).where(WorkerRecord.name == record.name)
        )
        if worker is not None and worker.active_jobs > 0:
            raise ConflictError(
                "node_enrollment_active_jobs",
                "The node cannot migrate while jobs are active",
            )
        if worker is not None:
            worker.status = "offline"
        if record.inference_node_id is not None:
            inference_node = session.get(InferenceNodeRecord, record.inference_node_id)
            if inference_node is not None:
                inference_node.connectivity = InferenceNodeConnectivity.OFFLINE.value
                inference_node.health = InferenceNodeHealth.UNKNOWN.value

        record.enrollment_status = ServiceEndpointEnrollmentStatus.PENDING.value
        record.token_configured = False
        record.enrollment_claimed_at = None
        record.enrolled_at = None
        record.probe_status = "unprobed"
        record.last_probe_at = None
        record.last_error = None
        record.remote_metadata_json = {}

    def claim(
        self, endpoint_id: str, payload: NodeEnrollmentClaim
    ) -> NodeEnrollmentClaimResponse:
        with _CLAIM_LOCK, self.context.database.session() as session:
            record = session.get(ServiceEndpointRecord, endpoint_id)
            if record is None:
                raise AppError(
                    "node_enrollment_invalid",
                    "Invalid node enrollment credential",
                    status_code=401,
                )
            if record.enrollment_status == ServiceEndpointEnrollmentStatus.ENROLLED.value:
                raise ConflictError(
                    "node_enrollment_already_active",
                    "This node endpoint is already enrolled",
                )
            expected_hash = record.enrollment_token_hash
            if expected_hash is None or not hmac.compare_digest(
                expected_hash, _token_hash(payload.enrollment_token)
            ):
                raise AppError(
                    "node_enrollment_invalid",
                    "Invalid node enrollment credential",
                    status_code=401,
                )
            if (
                record.enrollment_expires_at is None
                or as_utc(record.enrollment_expires_at) <= utc_now()
            ):
                raise AppError(
                    "node_enrollment_expired",
                    "Node enrollment credential has expired",
                    status_code=401,
                )
            if (
                payload.name != record.name
                or payload.kind.value != record.kind
                or payload.accelerator != record.accelerator
                or set(payload.capabilities) != set(record.capabilities_json)
            ):
                raise ConflictError(
                    "node_enrollment_identity_mismatch",
                    "Node identity does not match the registered endpoint",
                )

            node_token = self.context.node_secrets.read(record.id)
            if node_token is None:
                node_token = self._new_node_token()
                self.context.node_secrets.write(record.id, node_token)
            record.token_configured = True
            record.enrollment_status = ServiceEndpointEnrollmentStatus.CLAIMED.value
            if record.enrollment_claimed_at is None:
                record.enrollment_claimed_at = utc_now()
            return NodeEnrollmentClaimResponse(
                endpoint_id=record.id,
                node_token=node_token,
                enrollment_status=ServiceEndpointEnrollmentStatus.CLAIMED,
            )

    def _new_node_token(self) -> str:
        protected = {
            self.context.settings.admin_token.get_secret_value(),
            self.context.settings.worker_token.get_secret_value(),
        }
        while True:
            token = secrets.token_urlsafe(48)
            if token in protected:
                continue
            if self.context.node_secrets.matching_endpoint_id(token) is None:
                return token
