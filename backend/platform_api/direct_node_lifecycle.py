from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from .context import AppContext
from .contracts import (
    InferenceNodeConnectivity,
    InferenceNodeHealth,
    ServiceEndpointEnrollmentStatus,
    ServiceEndpointKind,
    WorkerStatus,
)
from .db_models import InferenceNodeRecord, ServiceEndpointRecord, WorkerRecord
from .inference_service import InferenceService
from .state_machine import utc_now


def _integer(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    if isinstance(value, str) and value.isdigit():
        return max(0, int(value))
    return default


def record_probe_success(
    context: AppContext, endpoint_id: str, health: dict[str, Any]
) -> None:
    now = utc_now()
    with context.database.session() as session:
        record = session.get(ServiceEndpointRecord, endpoint_id)
        if record is None:
            return
        was_claimed = (
            record.enrollment_status
            == ServiceEndpointEnrollmentStatus.CLAIMED.value
        )
        record.probe_status = "online"
        record.last_probe_at = now
        record.last_error = None
        record.remote_metadata_json = health
        if was_claimed:
            record.enrollment_status = ServiceEndpointEnrollmentStatus.ENROLLED.value
            record.enrollment_token_hash = None
            record.enrollment_expires_at = None
            record.enrolled_at = now

        if record.kind in {
            ServiceEndpointKind.TRAINER.value,
            ServiceEndpointKind.CONVERTER.value,
        }:
            worker = session.scalar(
                select(WorkerRecord).where(WorkerRecord.name == record.name)
            )
            if worker is None:
                worker = WorkerRecord(
                    id=f"worker_{uuid.uuid4().hex}", name=record.name
                )
                session.add(worker)
            worker.kind = record.kind
            worker.status = (
                WorkerStatus.BUSY.value
                if worker.active_jobs
                else WorkerStatus.ONLINE.value
            )
            worker.capabilities_json = list(record.capabilities_json)
            worker.accelerator = record.accelerator
            worker.max_concurrency = _integer(health.get("maxConcurrency"), 1)
            worker.version = str(health.get("version", "unknown"))
            worker.metadata_json = {"mode": "direct", "endpointId": record.id}
            worker.last_seen_at = now
            return

        if record.inference_node_id is None:
            return
        _, agent_token = InferenceService(context).activate_direct_node(
            session,
            record.inference_node_id,
            name=record.name,
            adapters=record.capabilities_json,
            max_model_instances=_integer(health.get("maxConcurrency"), 1),
            metadata=health,
            enabled=record.enabled,
        )
        if agent_token is not None:
            context.node_secrets.write(record.id, agent_token, purpose="agent")


def record_probe_failure(context: AppContext, endpoint_id: str, message: str) -> None:
    now = utc_now()
    with context.database.session() as session:
        record = session.get(ServiceEndpointRecord, endpoint_id)
        if record is None:
            return
        record.probe_status = "offline"
        record.last_probe_at = now
        record.last_error = message[:1000]
        if record.inference_node_id is None:
            return
        node = session.get(InferenceNodeRecord, record.inference_node_id)
        if node is not None:
            node.connectivity = InferenceNodeConnectivity.OFFLINE.value
            node.health = InferenceNodeHealth.UNKNOWN.value
