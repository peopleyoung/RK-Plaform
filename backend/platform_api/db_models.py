from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    version: Mapped[str] = mapped_column(String(40))
    task_type: Mapped[str] = mapped_column(String(40), index=True)
    dataset_format: Mapped[str] = mapped_column(String(40), default="auto", index=True)
    classes_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(30), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class WorkerRecord(Base):
    __tablename__ = "workers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(30), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    capabilities_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    accelerator: Mapped[str] = mapped_column(String(30), index=True)
    max_concurrency: Mapped[int] = mapped_column(Integer, default=1)
    active_jobs: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[str] = mapped_column(String(80), default="unknown")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    jobs: Mapped[list[JobRecord]] = relationship(back_populates="worker")


class ServiceEndpointRecord(Base):
    __tablename__ = "service_endpoints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(30), index=True)
    endpoint: Mapped[str] = mapped_column(String(500))
    mode: Mapped[str] = mapped_column(String(20), default="pull", index=True)
    scheme: Mapped[str] = mapped_column(String(10), default="http")
    host: Mapped[str] = mapped_column(String(255), default="")
    port: Mapped[int] = mapped_column(Integer, default=10081)
    accelerator: Mapped[str] = mapped_column(String(30), index=True)
    capabilities_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    token_configured: Mapped[bool] = mapped_column(Boolean, default=False)
    enrollment_status: Mapped[str] = mapped_column(
        String(20), default="enrolled", index=True
    )
    enrollment_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enrollment_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    enrollment_claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    enrolled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    probe_status: Mapped[str] = mapped_column(String(30), default="unprobed", index=True)
    last_probe_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    remote_metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    inference_node_id: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class NodeCleanupRecord(Base):
    __tablename__ = "node_cleanups"
    __table_args__ = (
        UniqueConstraint("endpoint_id", "job_id", name="uq_node_cleanup_job"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    endpoint_id: Mapped[str] = mapped_column(String(36), index=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class JobRecord(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_queue", "type", "status", "created_at"),
        Index("ix_jobs_lease", "status", "lease_expires_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    type: Mapped[str] = mapped_column(String(30), index=True)
    name: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(30), index=True)
    profile_id: Mapped[str] = mapped_column(String(80), index=True)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(ForeignKey("workers.id"), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(String(120), default="queued")
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=2)
    lease_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    worker: Mapped[WorkerRecord | None] = relationship(back_populates="jobs")
    events: Mapped[list[JobEventRecord]] = relationship(back_populates="job")
    artifacts: Mapped[list[ArtifactRecord]] = relationship(back_populates="job")


class JobEventRecord(Base):
    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    type: Mapped[str] = mapped_column(String(80))
    level: Mapped[str] = mapped_column(String(20), default="info")
    message: Mapped[str] = mapped_column(Text, default="")
    data_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    job: Mapped[JobRecord] = relationship(back_populates="events")


class ArtifactRecord(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(80), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    media_type: Mapped[str] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    manifest_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    job: Mapped[JobRecord | None] = relationship(back_populates="artifacts")


class ModelReleaseRecord(Base):
    __tablename__ = "model_releases"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_release_name_version"),)

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    version: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), index=True)
    profile_id: Mapped[str] = mapped_column(String(80), index=True)
    variant: Mapped[str] = mapped_column(String(80), index=True)
    task_type: Mapped[str] = mapped_column(String(40), index=True)
    precision: Mapped[str] = mapped_column(String(20))
    adapter: Mapped[str] = mapped_column(String(120), index=True)
    rknn_artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id"), index=True)
    validation_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True
    )
    source_training_job_id: Mapped[str | None] = mapped_column(String(48), nullable=True)
    source_conversion_job_id: Mapped[str] = mapped_column(String(48), index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(48), nullable=True)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class NodeGroupRecord(Base):
    __tablename__ = "node_groups"

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    labels_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class InferenceNodeRecord(Base):
    __tablename__ = "inference_nodes"

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    group_id: Mapped[str | None] = mapped_column(ForeignKey("node_groups.id"), nullable=True)
    labels_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    lifecycle: Mapped[str] = mapped_column(String(40), index=True)
    connectivity: Mapped[str] = mapped_column(String(30), index=True)
    health: Mapped[str] = mapped_column(String(30), index=True)
    deployment_status: Mapped[str] = mapped_column(String(30), default="idle")
    max_model_instances: Mapped[int] = mapped_column(Integer, default=1)
    hardware_id: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    runtime_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    driver_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    adapters_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    registration_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    registration_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    access_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    desired_revision: Mapped[int] = mapped_column(Integer, default=0)
    actual_revision: Mapped[int] = mapped_column(Integer, default=0)
    self_test_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class InferenceTaskRecord(Base):
    __tablename__ = "inference_tasks"

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.id"), index=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("inference_nodes.id"), index=True)
    group_id: Mapped[str | None] = mapped_column(ForeignKey("node_groups.id"), nullable=True)
    input_uri: Mapped[str] = mapped_column(String(2000))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    thresholds_json: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    media_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    analytics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    npu_core_mask: Mapped[str] = mapped_column(String(30), default="auto")
    npu_core_policy: Mapped[str] = mapped_column(String(30), default="shared")
    context_count: Mapped[int] = mapped_column(Integer, default=1)
    worker_count: Mapped[int] = mapped_column(Integer, default=1)
    media_migration_required: Mapped[bool] = mapped_column(Boolean, default=False)
    config_revision: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class MediaGatewayRecord(Base):
    __tablename__ = "media_gateways"

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    publish_host: Mapped[str] = mapped_column(String(255))
    rtsp_port: Mapped[int] = mapped_column(Integer)
    playback_host: Mapped[str] = mapped_column(String(255))
    ws_port: Mapped[int] = mapped_column(Integer)
    api_host: Mapped[str] = mapped_column(String(255))
    api_port: Mapped[int] = mapped_column(Integer)
    app: Mapped[str] = mapped_column(String(64), default="live")
    status: Mapped[str] = mapped_column(String(20), default="disabled", index=True)
    last_probe_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_hook_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class MediaCredentialRecord(Base):
    __tablename__ = "media_credentials"
    __table_args__ = (
        Index("ix_media_credentials_gateway_role", "gateway_id", "role", "revoked_at"),
    )

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(20), index=True)
    gateway_id: Mapped[str] = mapped_column(ForeignKey("media_gateways.id"), index=True)
    task_id: Mapped[str] = mapped_column(String(48), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    app: Mapped[str] = mapped_column(String(64))
    stream_name: Mapped[str] = mapped_column(String(64))
    principal: Mapped[str] = mapped_column(String(120))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class InferenceMediaBindingRecord(Base):
    __tablename__ = "inference_media_bindings"
    __table_args__ = (
        Index("ix_inference_media_binding_stream", "gateway_id", "app", "stream_name"),
    )

    task_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    gateway_id: Mapped[str] = mapped_column(ForeignKey("media_gateways.id"), index=True)
    app: Mapped[str] = mapped_column(String(64))
    stream_name: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class DeploymentRecord(Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.id"), index=True)
    strategy: Mapped[str] = mapped_column(String(30))
    batch_size: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DeploymentTargetRecord(Base):
    __tablename__ = "deployment_targets"
    __table_args__ = (
        UniqueConstraint("deployment_id", "task_id", name="uq_deployment_target_task"),
        Index("ix_deployment_targets_node_revision", "node_id", "desired_revision"),
    )

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    deployment_id: Mapped[str] = mapped_column(ForeignKey("deployments.id"), index=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("inference_nodes.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("inference_tasks.id"), index=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.id"))
    previous_release_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_releases.id"), nullable=True
    )
    previous_task_status: Mapped[str] = mapped_column(String(30))
    sequence: Mapped[int] = mapped_column(Integer)
    desired_revision: Mapped[int] = mapped_column(Integer, default=0)
    state: Mapped[str] = mapped_column(String(30), index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(String(120), default="queued")
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class DeploymentEventRecord(Base):
    __tablename__ = "deployment_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deployment_id: Mapped[str] = mapped_column(ForeignKey("deployments.id"), index=True)
    target_id: Mapped[str | None] = mapped_column(
        ForeignKey("deployment_targets.id"), nullable=True
    )
    node_id: Mapped[str | None] = mapped_column(ForeignKey("inference_nodes.id"), nullable=True)
    type: Mapped[str] = mapped_column(String(80))
    level: Mapped[str] = mapped_column(String(20), default="info")
    message: Mapped[str] = mapped_column(Text, default="")
    data_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
