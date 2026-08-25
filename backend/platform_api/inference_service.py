from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import timedelta
from typing import Any, cast

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .context import AppContext
from .contracts import (
    AgentArtifactDescriptor,
    AgentDesiredState,
    AgentReleaseDescriptor,
    AgentTaskDescriptor,
    DeploymentCreate,
    DeploymentEventResponse,
    DeploymentListResponse,
    DeploymentResponse,
    DeploymentStatus,
    DeploymentTargetReport,
    DeploymentTargetResponse,
    DeploymentTargetState,
    InferenceNodeConnectivity,
    InferenceNodeCreate,
    InferenceNodeCreated,
    InferenceNodeHealth,
    InferenceNodeHeartbeat,
    InferenceNodeLifecycle,
    InferenceNodeListResponse,
    InferenceNodeRegistration,
    InferenceNodeRegistrationResponse,
    InferenceNodeResponse,
    InferenceSummaryResponse,
    InferenceTaskCreate,
    InferenceTaskListResponse,
    InferenceTaskResponse,
    InferenceTaskStatus,
    InferenceTaskUpdate,
    ModelReleaseCreate,
    ModelReleaseListResponse,
    ModelReleaseResponse,
    ModelReleaseStatus,
    NodeGroupCreate,
    NodeGroupResponse,
    NodeGroupUpdate,
    NpuCoreMask,
    NpuCorePolicy,
    Precision,
    TaskType,
)
from .db_models import (
    ArtifactRecord,
    DeploymentEventRecord,
    DeploymentRecord,
    DeploymentTargetRecord,
    InferenceMediaBindingRecord,
    InferenceNodeRecord,
    InferenceTaskRecord,
    JobRecord,
    ModelReleaseRecord,
    NodeCleanupRecord,
    NodeGroupRecord,
    ServiceEndpointRecord,
)
from .errors import AuthenticationError, ConflictError, NotFoundError
from .media_contracts import PreviewCapability
from .media_service import MediaService
from .service import new_id
from .state_machine import as_utc, utc_now

ADAPTER_BY_OUTPUT_CONTRACT = {
    "rknn_yolov5_anchored_heads_v1": "yolo_anchored_v1",
    "rknn_yolov6_split_heads_v1": "yolo_v6_split_v1",
    "rknn_yolov7_anchored_heads_v1": "yolo_anchored_v1",
    "rknn_yolo_dfl_split_heads_v1": "yolo_dfl_split_v1",
    "rknn_yolov10_split_heads_v1": "yolo_v10_split_v1",
    "semantic_logits_nchw_v1": "deeplab_logits_v1",
    "ppocr_db_probability_map_v1": "ppocr_db_det_v1",
    "ppocr_ctc_logits_v1": "ppocr_ctc_rec_v1",
}

ACTIVE_TASK_STATUSES = {
    InferenceTaskStatus.DEPLOYING.value,
    InferenceTaskStatus.RUNNING.value,
    InferenceTaskStatus.DEGRADED.value,
}

TARGET_ORDER = {
    DeploymentTargetState.PENDING: 0,
    DeploymentTargetState.DOWNLOADING: 1,
    DeploymentTargetState.VERIFYING: 2,
    DeploymentTargetState.STAGED: 3,
    DeploymentTargetState.DRAINING: 4,
    DeploymentTargetState.ACTIVATING: 5,
    DeploymentTargetState.WARMING: 6,
    DeploymentTargetState.HEALTHY: 7,
}

NPU_CORE_BITS = {
    NpuCoreMask.AUTO.value: 0b111,
    NpuCoreMask.CORE_0.value: 0b001,
    NpuCoreMask.CORE_1.value: 0b010,
    NpuCoreMask.CORE_2.value: 0b100,
    NpuCoreMask.CORE_0_1.value: 0b011,
    NpuCoreMask.CORE_0_1_2.value: 0b111,
}


def _token_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ConflictError("invalid_release_manifest", f"Manifest has no valid {key}")
    return value


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ConflictError("invalid_release_manifest", f"Manifest has no valid {key}")
    return cast(dict[str, Any], value)


def _direct_runtime_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    diagnostics_value = metadata.get("diagnostics")
    diagnostics = (
        cast(dict[str, Any], diagnostics_value)
        if isinstance(diagnostics_value, dict)
        else {}
    )
    inference_value = diagnostics.get("inference")
    return (
        cast(dict[str, Any], inference_value)
        if isinstance(inference_value, dict)
        else {}
    )


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            return None
        result.append(item)
    return result


def _manifest_labels(manifest: dict[str, Any]) -> list[str]:
    """Read labels from both current and legacy release manifest layouts."""
    labels = _string_list(manifest.get("labels"))
    if labels is not None:
        return labels
    deployment = manifest.get("deployment")
    if isinstance(deployment, dict):
        nested = _string_list(cast(dict[str, Any], deployment).get("labels"))
        if nested is not None:
            return nested
    return []


def _agent_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize the persisted bundle contract consumed by the inference agent."""
    normalized = dict(manifest)
    if "labels" not in normalized:
        normalized["labels"] = _manifest_labels(manifest)
    return normalized


def model_release_response(record: ModelReleaseRecord) -> ModelReleaseResponse:
    return ModelReleaseResponse(
        id=record.id,
        name=record.name,
        version=record.version,
        description=record.description,
        status=ModelReleaseStatus(record.status),
        profile_id=record.profile_id,
        variant=record.variant,
        task_type=TaskType(record.task_type),
        precision=Precision(record.precision),
        adapter=record.adapter,
        rknn_artifact_id=record.rknn_artifact_id,
        validation_artifact_id=record.validation_artifact_id,
        source_training_job_id=record.source_training_job_id,
        source_conversion_job_id=record.source_conversion_job_id,
        dataset_id=record.dataset_id,
        manifest=record.manifest_json,
        created_at=record.created_at,
        published_at=record.published_at,
    )


def node_group_response(record: NodeGroupRecord) -> NodeGroupResponse:
    return NodeGroupResponse(
        id=record.id,
        name=record.name,
        description=record.description,
        labels=record.labels_json,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def inference_task_response(
    record: InferenceTaskRecord, preview_capability: PreviewCapability
) -> InferenceTaskResponse:
    return InferenceTaskResponse(
        id=record.id,
        name=record.name,
        status=InferenceTaskStatus(record.status),
        release_id=record.release_id,
        node_id=record.node_id,
        group_id=record.group_id,
        input_uri=record.input_uri,
        interval=record.interval,
        thresholds=record.thresholds_json,
        output=record.output_json,
        media=record.media_json,
        analytics=record.analytics_json,
        npu_core_mask=NpuCoreMask(record.npu_core_mask),
        npu_core_policy=NpuCorePolicy(record.npu_core_policy),
        context_count=record.context_count,
        worker_count=record.worker_count,
        preview_capability=preview_capability,
        config_revision=record.config_revision,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def deployment_target_response(record: DeploymentTargetRecord) -> DeploymentTargetResponse:
    return DeploymentTargetResponse(
        id=record.id,
        deployment_id=record.deployment_id,
        node_id=record.node_id,
        task_id=record.task_id,
        release_id=record.release_id,
        previous_release_id=record.previous_release_id,
        sequence=record.sequence,
        desired_revision=record.desired_revision,
        state=DeploymentTargetState(record.state),
        progress=record.progress,
        stage=record.stage,
        error_code=record.error_code,
        error_message=record.error_message,
        started_at=record.started_at,
        completed_at=record.completed_at,
        updated_at=record.updated_at,
    )


class InferenceService:
    def __init__(self, context: AppContext) -> None:
        self.context = context

    @staticmethod
    def _validate_task_media(media: dict[str, Any], adapter: str) -> None:
        tracking_value = media.get("tracking", {})
        tracking = cast(dict[str, Any], tracking_value) if isinstance(tracking_value, dict) else {}
        if (
            tracking.get("enabled") is True
            and not adapter.startswith("yolo_")
        ):
            raise ConflictError(
                "tracking_adapter_mismatch",
                "ByteTrack can only be enabled for detection model releases",
                adapter=adapter,
            )

    @staticmethod
    def _required_media_features(media: dict[str, Any]) -> set[str]:
        required: set[str] = set()
        if media.get("decoder") == "rkmpp":
            required.add("rkmpp_decode")
        for field, feature in (
            ("tracking", "bytetrack"),
            ("kafka", "kafka"),
            ("zlmSei", "zlm_sei"),
        ):
            value = media.get(field)
            config = cast(dict[str, Any], value) if isinstance(value, dict) else {}
            if config.get("enabled") is True:
                required.add(feature)
        return required

    @staticmethod
    def _required_analytics_features(analytics: dict[str, Any]) -> set[str]:
        required: set[str] = set()
        if analytics.get("areas"):
            required.add("analytics_area")
        if analytics.get("lines"):
            required.add("analytics_line")
        events_value = analytics.get("events", {})
        events = cast(dict[str, Any], events_value) if isinstance(events_value, dict) else {}
        if events.get("enabled") is True and events.get("snapshot", True) is True:
            required.add("event_snapshot")
        if events.get("enabled") is True and events.get("record") is True:
            required.add("event_record")
        if analytics.get("secondaryModels"):
            required.add("secondary_infer")
        return required

    @staticmethod
    def _secondary_release_ids(analytics: dict[str, Any]) -> list[str]:
        raw = analytics.get("secondaryModels", [])
        if not isinstance(raw, list):
            return []
        release_ids: list[str] = []
        for item in cast(list[object], raw):
            if not isinstance(item, dict):
                continue
            secondary = cast(dict[str, object], item)
            release_id = secondary.get("releaseId")
            if isinstance(release_id, str):
                release_ids.append(release_id)
        return release_ids

    def _validate_task_analytics(
        self,
        session: Session,
        analytics: dict[str, Any],
        release: ModelReleaseRecord,
        media: dict[str, Any],
    ) -> list[ModelReleaseRecord]:
        uses_detection_business = bool(
            analytics.get("areas")
            or analytics.get("lines")
            or analytics.get("secondaryModels")
        )
        if uses_detection_business and not release.adapter.startswith("yolo_"):
            raise ConflictError(
                "analytics_adapter_mismatch",
                "Area, line and secondary inference require a detection model release",
                adapter=release.adapter,
            )
        if (analytics.get("areas") or analytics.get("lines")):
            tracking_value = media.get("tracking", {})
            tracking = (
                cast(dict[str, Any], tracking_value)
                if isinstance(tracking_value, dict)
                else {}
            )
            if tracking.get("enabled") is not True:
                raise ConflictError(
                    "analytics_tracking_required",
                    "Area and line analytics require ByteTrack to be enabled",
                )
        secondary_releases: list[ModelReleaseRecord] = []
        for release_id in self._secondary_release_ids(analytics):
            secondary = self._published_release(session, release_id)
            if not secondary.adapter.startswith("yolo_"):
                raise ConflictError(
                    "secondary_adapter_mismatch",
                    "Secondary inference currently supports detection model releases only",
                    releaseId=secondary.id,
                    adapter=secondary.adapter,
                )
            secondary_releases.append(secondary)
        return secondary_releases

    @staticmethod
    def _node_media_features(node: InferenceNodeRecord) -> set[str]:
        raw = node.metadata_json.get("features", [])
        if not isinstance(raw, list):
            return set()
        return {str(item) for item in cast(list[object], raw) if isinstance(item, str)}

    def _validate_node_media(
        self, node: InferenceNodeRecord, media: dict[str, Any]
    ) -> None:
        required = self._required_media_features(media)
        missing = sorted(required - self._node_media_features(node))
        if missing:
            raise ConflictError(
                "inference_media_feature_missing",
                "Inference node does not provide required media runtime features",
                nodeId=node.id,
                missingFeatures=missing,
            )

    def _validate_node_analytics(
        self,
        node: InferenceNodeRecord,
        analytics: dict[str, Any],
        secondary_releases: list[ModelReleaseRecord],
    ) -> None:
        required = self._required_analytics_features(analytics)
        missing = sorted(required - self._node_media_features(node))
        if missing:
            raise ConflictError(
                "inference_analytics_feature_missing",
                "Inference node does not provide required analytics runtime features",
                nodeId=node.id,
                missingFeatures=missing,
            )
        missing_adapters = sorted(
            {release.adapter for release in secondary_releases} - set(node.adapters_json)
        )
        if missing_adapters:
            raise ConflictError(
                "inference_adapter_missing",
                "Inference node does not support a secondary model output contract",
                nodeId=node.id,
                missingAdapters=missing_adapters,
            )

    def create_release(self, payload: ModelReleaseCreate) -> ModelReleaseResponse:
        with self.context.database.session() as session:
            duplicate = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.name == payload.name,
                    ModelReleaseRecord.version == payload.version,
                )
            )
            if duplicate is not None:
                raise ConflictError(
                    "model_release_exists",
                    "A model release with this name and version already exists",
                    releaseId=duplicate.id,
                )
            job = session.get(JobRecord, payload.conversion_job_id)
            if job is None:
                raise NotFoundError("conversion job", payload.conversion_job_id)
            if job.type != "conversion" or job.status != "succeeded":
                raise ConflictError(
                    "conversion_not_publishable",
                    "Only succeeded conversion jobs can create a model release",
                    status=job.status,
                )
            result = job.result_json or {}
            if result.get("deploymentReady") is not True:
                raise ConflictError(
                    "conversion_not_deployment_ready",
                    "Conversion validation did not mark this model deployment-ready",
                )
            manifest = _mapping(job.spec_json, "manifest")
            output_contract = _string(manifest, "outputContract")
            adapter = ADAPTER_BY_OUTPUT_CONTRACT.get(output_contract)
            if adapter is None:
                raise ConflictError(
                    "unsupported_output_contract",
                    "No trusted inference adapter is registered for this output contract",
                    outputContract=output_contract,
                )
            artifacts = session.scalars(
                select(ArtifactRecord).where(ArtifactRecord.job_id == job.id)
            ).all()
            rknn = next((item for item in artifacts if item.kind == "rknn"), None)
            if rknn is None:
                raise ConflictError("rknn_artifact_missing", "Conversion job has no RKNN artifact")
            validation = next(
                (item for item in artifacts if item.kind == "validation_report"), None
            )
            training_job_id = manifest.get("trainingJobId")
            training_job = (
                session.get(JobRecord, training_job_id)
                if isinstance(training_job_id, str)
                else None
            )
            source = _mapping(job.spec_json, "sourceArtifact")
            bundle_manifest = {
                "schemaVersion": 1,
                "targetPlatform": "rk3588",
                "name": payload.name,
                "version": payload.version,
                "adapter": adapter,
                "outputContract": output_contract,
                "labels": _manifest_labels(manifest),
                "deployment": manifest,
                "validation": result.get("validation", {}),
                "artifact": {
                    "id": rknn.id,
                    "filename": rknn.filename,
                    "sha256": rknn.sha256,
                    "sizeBytes": rknn.size_bytes,
                },
                "sourceOnnxArtifactId": source.get("id"),
            }
            record = ModelReleaseRecord(
                id=new_id("release"),
                name=payload.name,
                version=payload.version,
                description=payload.description,
                status=ModelReleaseStatus.QUALIFIED.value,
                profile_id=_string(manifest, "profileId"),
                variant=_string(manifest, "variant"),
                task_type=_string(manifest, "taskType"),
                precision=_string(job.spec_json, "precision"),
                adapter=adapter,
                rknn_artifact_id=rknn.id,
                validation_artifact_id=validation.id if validation else None,
                source_training_job_id=(
                    training_job_id if isinstance(training_job_id, str) else None
                ),
                source_conversion_job_id=job.id,
                dataset_id=training_job.dataset_id if training_job else None,
                manifest_json=bundle_manifest,
            )
            session.add(record)
            session.flush()
            return model_release_response(record)

    def list_releases(self, page: int, page_size: int) -> ModelReleaseListResponse:
        with self.context.database.session() as session:
            total = session.scalar(select(func.count()).select_from(ModelReleaseRecord)) or 0
            records = session.scalars(
                select(ModelReleaseRecord)
                .order_by(ModelReleaseRecord.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return ModelReleaseListResponse(
                items=[model_release_response(item) for item in records],
                page=page,
                page_size=page_size,
                total=total,
            )

    def get_release(self, release_id: str) -> ModelReleaseRecord:
        with self.context.database.session() as session:
            record = session.get(ModelReleaseRecord, release_id)
            if record is None:
                raise NotFoundError("model release", release_id)
            session.expunge(record)
            return record

    def publish_release(self, release_id: str) -> ModelReleaseResponse:
        with self.context.database.session() as session:
            record = self._release(session, release_id)
            if record.status == ModelReleaseStatus.PUBLISHED.value:
                return model_release_response(record)
            if record.status != ModelReleaseStatus.QUALIFIED.value:
                raise ConflictError(
                    "model_release_not_publishable",
                    "Only qualified releases can be published",
                    status=record.status,
                )
            record.status = ModelReleaseStatus.PUBLISHED.value
            record.published_at = utc_now()
            return model_release_response(record)

    def deprecate_release(self, release_id: str) -> ModelReleaseResponse:
        with self.context.database.session() as session:
            record = self._release(session, release_id)
            if record.status not in {
                ModelReleaseStatus.PUBLISHED.value,
                ModelReleaseStatus.DEPRECATED.value,
            }:
                raise ConflictError(
                    "model_release_not_deprecatable",
                    "Only published releases can be deprecated",
                    status=record.status,
                )
            record.status = ModelReleaseStatus.DEPRECATED.value
            return model_release_response(record)

    def delete_release(self, release_id: str) -> None:
        with self.context.database.session() as session:
            record = self._release(session, release_id)
            if record.status != ModelReleaseStatus.DEPRECATED.value:
                raise ConflictError(
                    "model_release_not_deletable",
                    "Only deprecated model releases can be deleted",
                    status=record.status,
                )
            task_ids = list(
                session.scalars(
                    select(InferenceTaskRecord.id).where(
                        InferenceTaskRecord.release_id == release_id
                    )
                ).all()
            )
            secondary_task_ids = [
                task.id
                for task in session.scalars(select(InferenceTaskRecord)).all()
                if release_id in self._secondary_release_ids(task.analytics_json)
            ]
            deployment_count = (
                session.scalar(
                    select(func.count())
                    .select_from(DeploymentRecord)
                    .where(DeploymentRecord.release_id == release_id)
                )
                or 0
            )
            target_count = (
                session.scalar(
                    select(func.count())
                    .select_from(DeploymentTargetRecord)
                    .where(
                        (DeploymentTargetRecord.release_id == release_id)
                        | (DeploymentTargetRecord.previous_release_id == release_id)
                    )
                )
                or 0
            )
            referenced_task_ids = sorted(set(task_ids + secondary_task_ids))
            if referenced_task_ids or deployment_count or target_count:
                raise ConflictError(
                    "model_release_in_use",
                    "Delete referencing inference tasks and deployment history first",
                    taskIds=referenced_task_ids,
                    deploymentCount=deployment_count,
                    deploymentTargetCount=target_count,
                )
            session.delete(record)

    def create_node_group(self, payload: NodeGroupCreate) -> NodeGroupResponse:
        with self.context.database.session() as session:
            duplicate = session.scalar(
                select(NodeGroupRecord).where(NodeGroupRecord.name == payload.name)
            )
            if duplicate is not None:
                raise ConflictError("node_group_exists", "Node group name already exists")
            record = NodeGroupRecord(
                id=new_id("nodegroup"),
                name=payload.name.strip(),
                description=payload.description,
                labels_json=payload.labels,
            )
            session.add(record)
            session.flush()
            return node_group_response(record)

    def list_node_groups(self) -> list[NodeGroupResponse]:
        with self.context.database.session() as session:
            records = session.scalars(select(NodeGroupRecord).order_by(NodeGroupRecord.name)).all()
            return [node_group_response(item) for item in records]

    def update_node_group(
        self, group_id: str, payload: NodeGroupUpdate
    ) -> NodeGroupResponse:
        with self.context.database.session() as session:
            record = session.get(NodeGroupRecord, group_id)
            if record is None:
                raise NotFoundError("node group", group_id)
            duplicate = session.scalar(
                select(NodeGroupRecord).where(
                    NodeGroupRecord.name == payload.name,
                    NodeGroupRecord.id != group_id,
                )
            )
            if duplicate is not None:
                raise ConflictError("node_group_exists", "Node group name already exists")
            record.name = payload.name.strip()
            record.description = payload.description
            record.labels_json = payload.labels
            session.flush()
            return node_group_response(record)

    def delete_node_group(self, group_id: str) -> NodeGroupResponse:
        with self.context.database.session() as session:
            record = session.get(NodeGroupRecord, group_id)
            if record is None:
                raise NotFoundError("node group", group_id)
            node_count = session.scalar(
                select(func.count())
                .select_from(InferenceNodeRecord)
                .where(InferenceNodeRecord.group_id == group_id)
            )
            task_count = session.scalar(
                select(func.count())
                .select_from(InferenceTaskRecord)
                .where(
                    InferenceTaskRecord.group_id == group_id,
                    InferenceTaskRecord.status.in_(ACTIVE_TASK_STATUSES),
                )
            )
            if node_count or task_count:
                raise ConflictError(
                    "node_group_not_empty",
                    "Cannot delete a node group that still has nodes or active tasks",
                    nodeCount=node_count or 0,
                    activeTaskCount=task_count or 0,
                )
            session.delete(record)
            return node_group_response(record)

    def create_node(self, payload: InferenceNodeCreate) -> InferenceNodeCreated:
        token = secrets.token_urlsafe(32)
        now = utc_now()
        expires_at = now + timedelta(seconds=payload.registration_ttl_seconds)
        with self.context.database.session() as session:
            if session.scalar(
                select(InferenceNodeRecord).where(InferenceNodeRecord.name == payload.name)
            ):
                raise ConflictError("inference_node_exists", "Inference node name already exists")
            if (
                payload.group_id is not None
                and session.get(NodeGroupRecord, payload.group_id) is None
            ):
                raise NotFoundError("node group", payload.group_id)
            record = InferenceNodeRecord(
                id=new_id("inode"),
                name=payload.name.strip(),
                group_id=payload.group_id,
                labels_json=payload.labels,
                lifecycle=InferenceNodeLifecycle.PENDING_REGISTRATION.value,
                connectivity=InferenceNodeConnectivity.OFFLINE.value,
                health=InferenceNodeHealth.UNKNOWN.value,
                deployment_status="idle",
                max_model_instances=payload.max_model_instances,
                registration_token_hash=_token_hash(token),
                registration_expires_at=expires_at,
            )
            session.add(record)
            session.flush()
            response = self._node_response(record)
            return InferenceNodeCreated(
                **response.model_dump(),
                registration_token=token,
                registration_expires_at=expires_at,
            )

    def create_direct_node(
        self,
        session: Session,
        *,
        name: str,
        hardware_id: str,
        adapters: list[str],
        max_model_instances: int,
        metadata: dict[str, Any],
        enabled: bool = True,
    ) -> tuple[InferenceNodeResponse, str]:
        if session.scalar(select(InferenceNodeRecord).where(InferenceNodeRecord.name == name)):
            raise ConflictError("inference_node_exists", "Inference node name already exists")
        access_token = secrets.token_urlsafe(48)
        now = utc_now()
        runtime = _direct_runtime_metadata(metadata)
        diagnostics_value = metadata.get("diagnostics")
        diagnostics = (
            cast(dict[str, Any], diagnostics_value)
            if isinstance(diagnostics_value, dict)
            else {}
        )
        record = InferenceNodeRecord(
            id=new_id("inode"),
            name=name.strip(),
            labels_json=["direct"],
            lifecycle=(
                InferenceNodeLifecycle.ACTIVE.value
                if enabled
                else InferenceNodeLifecycle.MAINTENANCE.value
            ),
            connectivity=(
                InferenceNodeConnectivity.ONLINE.value
                if enabled
                else InferenceNodeConnectivity.OFFLINE.value
            ),
            health=(
                InferenceNodeHealth.HEALTHY.value
                if enabled
                else InferenceNodeHealth.UNKNOWN.value
            ),
            deployment_status="idle",
            max_model_instances=max_model_instances,
            hardware_id=hardware_id,
            runtime_version=str(runtime.get("runtimeVersion", metadata.get("version", "unknown"))),
            driver_version=str(runtime.get("driverVersion", "unknown")),
            pipeline_version=str(runtime.get("pipelineVersion", "unknown")),
            adapters_json=sorted(set(adapters)),
            metadata_json={**metadata, "mode": "direct"},
            access_token_hash=_token_hash(access_token),
            self_test_passed=diagnostics.get("inferenceSelfTestPassed") is True,
            last_seen_at=now,
        )
        session.add(record)
        session.flush()
        return self._node_response(record), access_token

    def create_pending_direct_node(
        self,
        session: Session,
        *,
        name: str,
        hardware_id: str,
        adapters: list[str],
        enabled: bool,
    ) -> InferenceNodeResponse:
        if session.scalar(select(InferenceNodeRecord).where(InferenceNodeRecord.name == name)):
            raise ConflictError("inference_node_exists", "Inference node name already exists")
        record = InferenceNodeRecord(
            id=new_id("inode"),
            name=name.strip(),
            labels_json=["direct"],
            lifecycle=InferenceNodeLifecycle.PENDING_REGISTRATION.value,
            connectivity=InferenceNodeConnectivity.OFFLINE.value,
            health=InferenceNodeHealth.UNKNOWN.value,
            deployment_status="idle",
            max_model_instances=1,
            hardware_id=hardware_id,
            adapters_json=sorted(set(adapters)),
            metadata_json={"mode": "direct", "enabled": enabled},
        )
        session.add(record)
        session.flush()
        return self._node_response(record)

    def activate_direct_node(
        self,
        session: Session,
        node_id: str,
        *,
        name: str,
        adapters: list[str],
        max_model_instances: int,
        metadata: dict[str, Any],
        enabled: bool,
    ) -> tuple[InferenceNodeResponse, str | None]:
        self.update_direct_node(
            session,
            node_id,
            name=name,
            adapters=adapters,
            max_model_instances=max_model_instances,
            metadata=metadata,
            enabled=enabled,
        )
        record = self._node(session, node_id)
        diagnostics_value = metadata.get("diagnostics")
        diagnostics = (
            cast(dict[str, Any], diagnostics_value)
            if isinstance(diagnostics_value, dict)
            else {}
        )
        inference_value = diagnostics.get("inference")
        inference = (
            cast(dict[str, Any], inference_value)
            if isinstance(inference_value, dict)
            else {}
        )
        if enabled and (
            diagnostics.get("inferenceSelfTestPassed") is not True
            or inference.get("lastError") is not None
        ):
            record.health = InferenceNodeHealth.DEGRADED.value
        access_token: str | None = None
        if record.access_token_hash is None:
            access_token = secrets.token_urlsafe(48)
            record.access_token_hash = _token_hash(access_token)
        session.flush()
        return self._node_response(record), access_token

    def update_direct_node(
        self,
        session: Session,
        node_id: str,
        *,
        name: str,
        adapters: list[str],
        max_model_instances: int,
        metadata: dict[str, Any],
        enabled: bool,
    ) -> InferenceNodeResponse:
        record = self._node(session, node_id)
        runtime = _direct_runtime_metadata(metadata)
        diagnostics_value = metadata.get("diagnostics")
        diagnostics = (
            cast(dict[str, Any], diagnostics_value)
            if isinstance(diagnostics_value, dict)
            else {}
        )
        duplicate = session.scalar(
            select(InferenceNodeRecord).where(
                InferenceNodeRecord.name == name,
                InferenceNodeRecord.id != node_id,
            )
        )
        if duplicate is not None:
            raise ConflictError("inference_node_exists", "Inference node name already exists")
        record.name = name.strip()
        record.adapters_json = sorted(set(adapters))
        record.max_model_instances = max_model_instances
        record.metadata_json = {**metadata, "mode": "direct"}
        record.runtime_version = str(
            runtime.get(
                "runtimeVersion",
                metadata.get("version", record.runtime_version or "unknown"),
            )
        )
        record.driver_version = str(
            runtime.get("driverVersion", record.driver_version or "unknown")
        )
        record.pipeline_version = str(
            runtime.get("pipelineVersion", record.pipeline_version or "unknown")
        )
        record.connectivity = (
            InferenceNodeConnectivity.ONLINE.value
            if enabled
            else InferenceNodeConnectivity.OFFLINE.value
        )
        record.health = (
            InferenceNodeHealth.HEALTHY.value
            if enabled
            else InferenceNodeHealth.UNKNOWN.value
        )
        record.lifecycle = (
            InferenceNodeLifecycle.ACTIVE.value
            if enabled
            else InferenceNodeLifecycle.MAINTENANCE.value
        )
        record.self_test_passed = diagnostics.get("inferenceSelfTestPassed") is True
        record.last_seen_at = utc_now()
        session.flush()
        return self._node_response(record)

    def retire_direct_node(self, session: Session, node_id: str) -> None:
        record = self._node(session, node_id)
        active_tasks = session.scalar(
            select(func.count())
            .select_from(InferenceTaskRecord)
            .where(
                InferenceTaskRecord.node_id == node_id,
                InferenceTaskRecord.status.in_(ACTIVE_TASK_STATUSES),
            )
        )
        if active_tasks:
            raise ConflictError(
                "inference_node_has_active_tasks",
                "Stop or retire inference tasks before deleting this node service",
            )
        record.lifecycle = InferenceNodeLifecycle.RETIRED.value
        record.connectivity = InferenceNodeConnectivity.OFFLINE.value
        record.health = InferenceNodeHealth.UNKNOWN.value
        record.access_token_hash = None
        record.last_seen_at = utc_now()

    def list_nodes(self, page: int, page_size: int) -> InferenceNodeListResponse:
        with self.context.database.session() as session:
            total = session.scalar(select(func.count()).select_from(InferenceNodeRecord)) or 0
            records = session.scalars(
                select(InferenceNodeRecord)
                .order_by(InferenceNodeRecord.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return InferenceNodeListResponse(
                items=[self._node_response(item) for item in records],
                page=page,
                page_size=page_size,
                total=total,
            )

    def summary(self) -> InferenceSummaryResponse:
        with self.context.database.session() as session:
            nodes = session.scalars(select(InferenceNodeRecord)).all()
            published_releases = (
                session.scalar(
                    select(func.count())
                    .select_from(ModelReleaseRecord)
                    .where(ModelReleaseRecord.status == ModelReleaseStatus.PUBLISHED.value)
                )
                or 0
            )
            running_tasks = (
                session.scalar(
                    select(func.count())
                    .select_from(InferenceTaskRecord)
                    .where(InferenceTaskRecord.status == InferenceTaskStatus.RUNNING.value)
                )
                or 0
            )
            active_deployments = (
                session.scalar(
                    select(func.count())
                    .select_from(DeploymentRecord)
                    .where(
                        DeploymentRecord.status.in_(
                            {
                                DeploymentStatus.QUEUED.value,
                                DeploymentStatus.ROLLING.value,
                                DeploymentStatus.ROLLING_BACK.value,
                            }
                        )
                    )
                )
                or 0
            )
            return InferenceSummaryResponse(
                online_nodes=sum(
                    self._node_connectivity(node) == InferenceNodeConnectivity.ONLINE
                    for node in nodes
                ),
                total_nodes=len(nodes),
                published_releases=published_releases,
                running_tasks=running_tasks,
                active_deployments=active_deployments,
            )

    def register_node(
        self, payload: InferenceNodeRegistration
    ) -> InferenceNodeRegistrationResponse:
        access_token = secrets.token_urlsafe(48)
        with self.context.database.session() as session:
            record = self._node(session, payload.node_id)
            if record.lifecycle != InferenceNodeLifecycle.PENDING_REGISTRATION.value:
                raise ConflictError(
                    "inference_node_already_registered",
                    "This node registration token has already been used",
                )
            if record.registration_token_hash is None or not hmac.compare_digest(
                record.registration_token_hash, _token_hash(payload.registration_token)
            ):
                raise AuthenticationError("Invalid inference-node registration token")
            if (
                record.registration_expires_at is None
                or as_utc(record.registration_expires_at) < utc_now()
            ):
                raise AuthenticationError("Inference-node registration token has expired")
            duplicate_hardware = session.scalar(
                select(InferenceNodeRecord).where(
                    InferenceNodeRecord.hardware_id == payload.hardware_id,
                    InferenceNodeRecord.id != record.id,
                )
            )
            if duplicate_hardware is not None:
                raise ConflictError(
                    "inference_hardware_registered",
                    "This RK3588 hardware identity is already registered",
                    nodeId=duplicate_hardware.id,
                )
            record.hardware_id = payload.hardware_id
            record.runtime_version = payload.runtime_version
            record.driver_version = payload.driver_version
            record.pipeline_version = payload.pipeline_version
            record.adapters_json = sorted(set(payload.adapters))
            record.metadata_json = payload.metadata
            record.registration_token_hash = None
            record.registration_expires_at = None
            record.access_token_hash = _token_hash(access_token)
            record.lifecycle = InferenceNodeLifecycle.AWAITING_APPROVAL.value
            record.connectivity = InferenceNodeConnectivity.ONLINE.value
            record.health = InferenceNodeHealth.VALIDATING.value
            record.last_seen_at = utc_now()
            return InferenceNodeRegistrationResponse(
                node=self._node_response(record), access_token=access_token
            )

    def heartbeat_node(
        self, node_id: str, token: str, payload: InferenceNodeHeartbeat
    ) -> InferenceNodeResponse:
        with self.context.database.session() as session:
            record = self._authorized_node(session, node_id, token)
            if payload.actual_revision > record.desired_revision:
                raise ConflictError(
                    "invalid_actual_revision",
                    "Node actual revision cannot exceed desired revision",
                    desiredRevision=record.desired_revision,
                )
            record.connectivity = InferenceNodeConnectivity.ONLINE.value
            record.last_seen_at = utc_now()
            record.actual_revision = payload.actual_revision
            record.health = payload.health.value
            record.self_test_passed = record.self_test_passed or payload.self_test_passed
            if payload.runtime_version is not None:
                record.runtime_version = payload.runtime_version
            if payload.driver_version is not None:
                record.driver_version = payload.driver_version
            if payload.pipeline_version is not None:
                record.pipeline_version = payload.pipeline_version
            if payload.adapters:
                record.adapters_json = sorted(set(payload.adapters))
            record.metadata_json = {
                **record.metadata_json,
                **payload.metadata,
                "metrics": payload.metrics,
            }
            self._reconcile_task_revisions(session, record, payload)
            self._finish_stopped_rollbacks(session, record)
            return self._node_response(record)

    def _reconcile_task_revisions(
        self,
        session: Session,
        node: InferenceNodeRecord,
        payload: InferenceNodeHeartbeat,
    ) -> None:
        deploying = session.scalars(
            select(InferenceTaskRecord).where(
                InferenceTaskRecord.node_id == node.id,
                InferenceTaskRecord.status == InferenceTaskStatus.DEPLOYING.value,
            )
        ).all()
        for task in deploying:
            tracked = session.scalar(
                select(DeploymentTargetRecord.id).where(
                    DeploymentTargetRecord.node_id == node.id,
                    DeploymentTargetRecord.task_id == task.id,
                    DeploymentTargetRecord.desired_revision == task.config_revision,
                )
            )
            if tracked is not None:
                continue
            if payload.failed_revision == task.config_revision:
                task.status = InferenceTaskStatus.FAILED.value
                task.error_message = f"Revision {task.config_revision} failed on the inference node"
                node.deployment_status = "failed"
            elif payload.actual_revision >= task.config_revision:
                task.status = InferenceTaskStatus.RUNNING.value
                task.error_message = None
                node.deployment_status = "idle"

    def approve_node(self, node_id: str) -> InferenceNodeResponse:
        with self.context.database.session() as session:
            record = self._node(session, node_id)
            if record.lifecycle == InferenceNodeLifecycle.ACTIVE.value:
                return self._node_response(record)
            if record.lifecycle != InferenceNodeLifecycle.AWAITING_APPROVAL.value:
                raise ConflictError(
                    "inference_node_not_approvable",
                    "Node must register before approval",
                    lifecycle=record.lifecycle,
                )
            if not record.self_test_passed:
                raise ConflictError(
                    "inference_node_self_test_required",
                    "Node must pass its runtime self-test before approval",
                )
            record.lifecycle = InferenceNodeLifecycle.ACTIVE.value
            record.health = InferenceNodeHealth.HEALTHY.value
            return self._node_response(record)

    def retire_node(self, node_id: str) -> InferenceNodeResponse:
        with self.context.database.session() as session:
            record = self._node(session, node_id)
            active_tasks = session.scalar(
                select(func.count())
                .select_from(InferenceTaskRecord)
                .where(
                    InferenceTaskRecord.node_id == node_id,
                    InferenceTaskRecord.status.in_(ACTIVE_TASK_STATUSES),
                )
            )
            if active_tasks:
                raise ConflictError(
                    "inference_node_has_active_tasks",
                    "Stop or migrate active inference tasks before retiring the node",
                    activeTaskCount=active_tasks,
                )
            record.lifecycle = InferenceNodeLifecycle.RETIRED.value
            record.access_token_hash = None
            record.connectivity = InferenceNodeConnectivity.OFFLINE.value
            return self._node_response(record)

    def delete_retired_node(self, node_id: str) -> None:
        endpoint_ids: list[str] = []
        with self.context.database.session() as session:
            record = self._node(session, node_id)
            if record.lifecycle != InferenceNodeLifecycle.RETIRED.value:
                raise ConflictError(
                    "inference_node_not_retired",
                    "Retire the inference node before permanently deleting it",
                    lifecycle=record.lifecycle,
                )
            endpoints = session.scalars(
                select(ServiceEndpointRecord).where(
                    ServiceEndpointRecord.inference_node_id == node_id
                )
            ).all()
            tasks = session.scalars(
                select(InferenceTaskRecord).where(InferenceTaskRecord.node_id == node_id)
            ).all()
            non_retired_tasks = [
                task
                for task in tasks
                if task.status != InferenceTaskStatus.RETIRED.value
            ]
            target_count = (
                session.scalar(
                    select(func.count())
                    .select_from(DeploymentTargetRecord)
                    .where(DeploymentTargetRecord.node_id == node_id)
                )
                or 0
            )
            event_count = (
                session.scalar(
                    select(func.count())
                    .select_from(DeploymentEventRecord)
                    .where(DeploymentEventRecord.node_id == node_id)
                )
                or 0
            )
            if non_retired_tasks:
                raise ConflictError(
                    "inference_node_has_tasks",
                    "Retire all inference tasks before permanently deleting this node",
                    nonRetiredTaskCount=len(non_retired_tasks),
                    taskIds=[task.id for task in non_retired_tasks],
                )
            if target_count or event_count:
                raise ConflictError(
                    "inference_node_has_history",
                    "Delete deployment history before permanently deleting this node",
                    serviceCount=len(endpoints),
                    retiredTaskCount=len(tasks),
                    deploymentTargetCount=target_count,
                    deploymentEventCount=event_count,
                )

            endpoint_ids = [endpoint.id for endpoint in endpoints]
            for endpoint in endpoints:
                cleanups = session.scalars(
                    select(NodeCleanupRecord).where(
                        NodeCleanupRecord.endpoint_id == endpoint.id
                    )
                ).all()
                for cleanup in cleanups:
                    session.delete(cleanup)
                session.delete(endpoint)
            for task in tasks:
                session.execute(
                    delete(InferenceMediaBindingRecord).where(
                        InferenceMediaBindingRecord.task_id == task.id
                    )
                )
                session.delete(task)
            session.flush()
            session.delete(record)

        for endpoint_id in endpoint_ids:
            self.context.node_secrets.delete(endpoint_id)
            self.context.node_secrets.delete(endpoint_id, purpose="agent")

    def create_task(self, payload: InferenceTaskCreate) -> InferenceTaskResponse:
        with self.context.database.session() as session:
            release = self._published_release(session, payload.release_id)
            self._validate_task_media(payload.media, release.adapter)
            MediaService(self.context).validate_task_media(session, payload.media)
            secondary_releases = self._validate_task_analytics(
                session, payload.analytics, release, payload.media
            )
            node = self._select_node(
                session,
                payload.node_id,
                payload.group_id,
                release.adapter,
                payload.media,
                payload.analytics,
                secondary_releases,
            )
            record = InferenceTaskRecord(
                id=new_id("itask"),
                name=payload.name.strip(),
                status=InferenceTaskStatus.STOPPED.value,
                release_id=release.id,
                node_id=node.id,
                group_id=payload.group_id,
                input_uri=payload.input_uri.strip(),
                interval=payload.interval,
                thresholds_json=payload.thresholds,
                output_json=payload.output,
                media_json=payload.media,
                analytics_json=payload.analytics,
                npu_core_mask=payload.npu_core_mask.value,
                npu_core_policy=payload.npu_core_policy.value,
                context_count=payload.context_count,
                worker_count=payload.worker_count,
            )
            session.add(record)
            session.flush()
            MediaService(self.context).bind_task(session, record)
            return self._task_response(session, record)

    def update_task(self, task_id: str, payload: InferenceTaskUpdate) -> InferenceTaskResponse:
        with self.context.database.session() as session:
            record = self._task(session, task_id)
            if record.status not in {
                InferenceTaskStatus.STOPPED.value,
                InferenceTaskStatus.DRAFT.value,
                InferenceTaskStatus.FAILED.value,
            }:
                raise ConflictError(
                    "inference_task_not_editable",
                    "Stop the inference task before editing it",
                    status=record.status,
                )
            release = self._published_release(session, payload.release_id)
            self._validate_task_media(payload.media, release.adapter)
            MediaService(self.context).validate_task_media(
                session, payload.media, task_id=record.id
            )
            secondary_releases = self._validate_task_analytics(
                session, payload.analytics, release, payload.media
            )
            node = self._select_node(
                session,
                payload.node_id,
                payload.group_id,
                release.adapter,
                payload.media,
                payload.analytics,
                secondary_releases,
            )
            record.name = payload.name.strip()
            record.release_id = release.id
            record.node_id = node.id
            record.group_id = payload.group_id
            record.input_uri = payload.input_uri.strip()
            record.interval = payload.interval
            record.thresholds_json = payload.thresholds
            record.output_json = payload.output
            record.media_json = payload.media
            record.analytics_json = payload.analytics
            record.npu_core_mask = payload.npu_core_mask.value
            record.npu_core_policy = payload.npu_core_policy.value
            record.context_count = payload.context_count
            record.worker_count = payload.worker_count
            record.media_migration_required = False
            record.error_message = None
            MediaService(self.context).bind_task(session, record)
            return self._task_response(session, record)

    def list_tasks(self, page: int, page_size: int) -> InferenceTaskListResponse:
        with self.context.database.session() as session:
            visible = InferenceTaskRecord.status != InferenceTaskStatus.RETIRED.value
            total = (
                session.scalar(select(func.count()).select_from(InferenceTaskRecord).where(visible))
                or 0
            )
            records = session.scalars(
                select(InferenceTaskRecord)
                .where(visible)
                .order_by(InferenceTaskRecord.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return InferenceTaskListResponse(
                items=[self._task_response(session, item) for item in records],
                page=page,
                page_size=page_size,
                total=total,
            )

    def _task_response(
        self, session: Session, record: InferenceTaskRecord
    ) -> InferenceTaskResponse:
        capability = MediaService(self.context).preview_capability(session, record)
        return inference_task_response(record, capability)

    def stop_task(self, task_id: str) -> InferenceTaskResponse:
        with self.context.database.session() as session:
            task = self._task(session, task_id)
            if task.status == InferenceTaskStatus.RETIRED.value:
                raise ConflictError("inference_task_retired", "Retired tasks cannot be stopped")
            if task.status != InferenceTaskStatus.STOPPED.value:
                node = self._node(session, task.node_id)
                node.desired_revision += 1
                node.deployment_status = "deploying"
                task.status = InferenceTaskStatus.STOPPED.value
                task.config_revision = node.desired_revision
            MediaService(self.context).revoke_task_publication(session, task.id)
            response = self._task_response(session, task)
        MediaService(self.context).close_task_stream(task_id)
        return response

    def restart_task(self, task_id: str) -> InferenceTaskResponse:
        with self.context.database.session() as session:
            task = self._task(session, task_id)
            if task.status not in {
                InferenceTaskStatus.STOPPED.value,
                InferenceTaskStatus.FAILED.value,
            }:
                raise ConflictError(
                    "inference_task_not_restartable",
                    "Only stopped or failed inference tasks can be restarted",
                    taskId=task.id,
                    status=task.status,
                )
            release = self._published_release(session, task.release_id)
            node = self._validate_task_for_deployment(session, task, release)
            self._validate_node_runtime_plan(session, node, [task], release.id)
            previous_status = task.status
            reserved = (
                session.query(InferenceTaskRecord)
                .filter(
                    InferenceTaskRecord.id == task.id,
                    InferenceTaskRecord.status == previous_status,
                )
                .update(
                    {InferenceTaskRecord.status: InferenceTaskStatus.DEPLOYING.value},
                    synchronize_session=False,
                )
            )
            if reserved != 1:
                raise ConflictError(
                    "inference_task_not_restartable",
                    "Inference task state changed before restart could be reserved",
                    taskId=task.id,
                    status=previous_status,
                )
            node.desired_revision += 1
            node.deployment_status = "deploying"
            task.status = InferenceTaskStatus.DEPLOYING.value
            task.config_revision = node.desired_revision
            task.error_message = None
            MediaService(self.context).ensure_publish_credential(session, task)
            session.flush()
            return self._task_response(session, task)

    def retire_task(self, task_id: str) -> InferenceTaskResponse:
        with self.context.database.session() as session:
            task = self._task(session, task_id)
            if task.status not in {
                InferenceTaskStatus.STOPPED.value,
                InferenceTaskStatus.DRAFT.value,
                InferenceTaskStatus.FAILED.value,
            }:
                raise ConflictError(
                    "inference_task_not_deletable",
                    "Stop the inference task before deleting it",
                    status=task.status,
                )
            task.status = InferenceTaskStatus.RETIRED.value
            MediaService(self.context).revoke_task_publication(session, task.id)
            response = self._task_response(session, task)
        MediaService(self.context).close_task_stream(task_id)
        return response

    def create_deployment(self, payload: DeploymentCreate) -> DeploymentResponse:
        with self.context.database.session() as session:
            return self._create_deployment(session, payload)

    def list_deployments(self, page: int, page_size: int) -> DeploymentListResponse:
        with self.context.database.session() as session:
            total = session.scalar(select(func.count()).select_from(DeploymentRecord)) or 0
            records = session.scalars(
                select(DeploymentRecord)
                .order_by(DeploymentRecord.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return DeploymentListResponse(
                items=[self._deployment_response(session, item) for item in records],
                page=page,
                page_size=page_size,
                total=total,
            )

    def get_deployment(self, deployment_id: str) -> DeploymentResponse:
        with self.context.database.session() as session:
            deployment = self._deployment(session, deployment_id)
            return self._deployment_response(session, deployment)

    def delete_deployment(self, deployment_id: str) -> None:
        with self.context.database.session() as session:
            deployment = self._deployment(session, deployment_id)
            if deployment.status in {
                DeploymentStatus.QUEUED.value,
                DeploymentStatus.ROLLING.value,
                DeploymentStatus.ROLLING_BACK.value,
            }:
                raise ConflictError(
                    "deployment_active",
                    "An active deployment cannot be deleted",
                    status=deployment.status,
                )
            events = session.scalars(
                select(DeploymentEventRecord).where(
                    DeploymentEventRecord.deployment_id == deployment_id
                )
            ).all()
            for event in events:
                session.delete(event)
            # Events reference deployment targets, so flush their deletes before
            # removing the targets and finally the deployment row.
            session.flush()
            targets = self._targets(session, deployment_id)
            for target in targets:
                session.delete(target)
            session.flush()
            session.delete(deployment)

    def deployment_events(
        self, deployment_id: str, after_id: int, limit: int
    ) -> list[DeploymentEventResponse]:
        with self.context.database.session() as session:
            self._deployment(session, deployment_id)
            records = session.scalars(
                select(DeploymentEventRecord)
                .where(
                    DeploymentEventRecord.deployment_id == deployment_id,
                    DeploymentEventRecord.id > after_id,
                )
                .order_by(DeploymentEventRecord.id)
                .limit(limit)
            ).all()
            return [
                DeploymentEventResponse(
                    id=item.id,
                    deployment_id=item.deployment_id,
                    target_id=item.target_id,
                    node_id=item.node_id,
                    type=item.type,
                    level=item.level,
                    message=item.message,
                    data=item.data_json,
                    created_at=item.created_at,
                )
                for item in records
            ]

    def rollback_deployment(self, deployment_id: str) -> DeploymentResponse:
        with self.context.database.session() as session:
            deployment = self._deployment(session, deployment_id)
            if deployment.status not in {
                DeploymentStatus.SUCCEEDED.value,
                DeploymentStatus.FAILED.value,
                DeploymentStatus.PAUSED.value,
            }:
                raise ConflictError(
                    "deployment_not_rollbackable",
                    "Only completed, failed, or paused deployments can be rolled back",
                    status=deployment.status,
                )
            deployment.status = DeploymentStatus.ROLLING_BACK.value
            targets = self._targets(session, deployment.id)
            for target in targets:
                task = self._task(session, target.task_id)
                if (
                    target.previous_task_status
                    in {
                        InferenceTaskStatus.RUNNING.value,
                        InferenceTaskStatus.DEGRADED.value,
                    }
                    and target.previous_release_id
                ):
                    target.release_id = target.previous_release_id
                    target.state = DeploymentTargetState.PENDING.value
                    target.desired_revision = 0
                    target.progress = 0
                    target.stage = "waiting_for_rollback"
                else:
                    node = self._node(session, target.node_id)
                    node.desired_revision += 1
                    node.deployment_status = "rollback"
                    task.status = InferenceTaskStatus.STOPPED.value
                    task.config_revision = node.desired_revision
                    target.state = DeploymentTargetState.PENDING.value
                    target.desired_revision = node.desired_revision
                    target.progress = 0
                    target.stage = "stopping_for_rollback"
                    target.error_code = None
                    target.error_message = None
            self._event(session, deployment, "rollback_started", "Rollback requested")
            self._activate_next_batch(session, deployment)
            return self._deployment_response(session, deployment)

    def retry_deployment(self, deployment_id: str) -> DeploymentResponse:
        with self.context.database.session() as session:
            deployment = self._deployment(session, deployment_id)
            if deployment.status not in {
                DeploymentStatus.FAILED.value,
                DeploymentStatus.PAUSED.value,
            }:
                raise ConflictError(
                    "deployment_not_retryable",
                    "Only failed or paused deployments can be retried",
                    status=deployment.status,
                )
            failed_targets = [
                target
                for target in self._targets(session, deployment.id)
                if target.state == DeploymentTargetState.FAILED.value
            ]
            if not failed_targets:
                raise ConflictError(
                    "deployment_has_no_failed_targets",
                    "Deployment has no failed targets to retry",
                )
            for target in failed_targets:
                target.state = DeploymentTargetState.PENDING.value
                target.desired_revision = 0
                target.progress = 0
                target.stage = "waiting_for_retry"
                target.error_code = None
                target.error_message = None
            deployment.status = DeploymentStatus.QUEUED.value
            deployment.completed_at = None
            self._event(
                session,
                deployment,
                "retry_started",
                f"Retrying {len(failed_targets)} failed deployment target(s)",
            )
            self._activate_next_batch(session, deployment)
            return self._deployment_response(session, deployment)

    def desired_state(self, node_id: str, token: str) -> AgentDesiredState:
        with self.context.database.session() as session:
            node = self._authorized_node(session, node_id, token)
            return self._desired_state(session, node)

    def direct_desired_state(self, node_id: str) -> AgentDesiredState:
        with self.context.database.session() as session:
            node = self._node(session, node_id)
            if node.lifecycle != InferenceNodeLifecycle.ACTIVE.value:
                raise ConflictError(
                    "inference_node_not_active",
                    "Direct inference node is not active",
                )
            return self._desired_state(session, node)

    def _desired_state(
        self, session: Session, node: InferenceNodeRecord
    ) -> AgentDesiredState:
            tasks = session.scalars(
                select(InferenceTaskRecord)
                .where(
                    InferenceTaskRecord.node_id == node.id,
                    InferenceTaskRecord.status.in_(ACTIVE_TASK_STATUSES),
                )
                .order_by(InferenceTaskRecord.id)
            ).all()
            releases_by_id: dict[str, ModelReleaseRecord] = {}
            task_descriptors: list[AgentTaskDescriptor] = []
            for task in tasks:
                release = self._release(session, task.release_id)
                releases_by_id[release.id] = release
                for secondary_release_id in self._secondary_release_ids(task.analytics_json):
                    secondary_release = self._release(session, secondary_release_id)
                    releases_by_id[secondary_release.id] = secondary_release
                target = session.scalar(
                    select(DeploymentTargetRecord)
                    .where(
                        DeploymentTargetRecord.node_id == node.id,
                        DeploymentTargetRecord.task_id == task.id,
                        DeploymentTargetRecord.desired_revision == task.config_revision,
                    )
                    .order_by(DeploymentTargetRecord.updated_at.desc())
                )
                task_descriptors.append(
                    AgentTaskDescriptor(
                        id=task.id,
                        name=task.name,
                        release_id=task.release_id,
                        deployment_target_id=target.id if target else None,
                        input_uri=task.input_uri,
                        interval=task.interval,
                        thresholds=task.thresholds_json,
                        output=task.output_json,
                        media=MediaService(self.context).node_media(session, task),
                        analytics=task.analytics_json,
                        npu_core_mask=NpuCoreMask(task.npu_core_mask),
                        npu_core_policy=NpuCorePolicy(task.npu_core_policy),
                        context_count=task.context_count,
                        worker_count=task.worker_count,
                    )
                )
            release_descriptors: list[AgentReleaseDescriptor] = []
            for release in sorted(releases_by_id.values(), key=lambda item: item.id):
                artifact = session.get(ArtifactRecord, release.rknn_artifact_id)
                if artifact is None:
                    raise ConflictError(
                        "release_artifact_missing",
                        "Published release artifact is missing",
                        releaseId=release.id,
                    )
                release_descriptors.append(
                    AgentReleaseDescriptor(
                        id=release.id,
                        name=release.name,
                        version=release.version,
                        adapter=release.adapter,
                        artifact=AgentArtifactDescriptor(
                            id=artifact.id,
                            filename=artifact.filename,
                            sha256=artifact.sha256,
                            size_bytes=artifact.size_bytes,
                            media_type=artifact.media_type,
                        ),
                        manifest=_agent_manifest(release.manifest_json),
                    )
                )
            content = {
                "nodeId": node.id,
                "revision": node.desired_revision,
                "releases": [
                    item.model_dump(mode="json", by_alias=True) for item in release_descriptors
                ],
                "tasks": [item.model_dump(mode="json", by_alias=True) for item in task_descriptors],
            }
            config_hash = hashlib.sha256(
                json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            return AgentDesiredState(
                node_id=node.id,
                revision=node.desired_revision,
                config_hash=config_hash,
                releases=release_descriptors,
                tasks=task_descriptors,
            )

    def report_target(
        self, node_id: str, target_id: str, token: str, payload: DeploymentTargetReport
    ) -> DeploymentTargetResponse:
        with self.context.database.session() as session:
            node = self._authorized_node(session, node_id, token)
            target = session.get(DeploymentTargetRecord, target_id)
            if target is None or target.node_id != node.id:
                raise NotFoundError("deployment target", target_id)
            if target.desired_revision != payload.revision:
                raise ConflictError(
                    "stale_deployment_revision",
                    "Deployment report revision does not match the active target",
                    expectedRevision=target.desired_revision,
                )
            requested_state = payload.state
            current_state = DeploymentTargetState(target.state)
            if (
                requested_state
                not in {
                    DeploymentTargetState.FAILED,
                    DeploymentTargetState.ROLLED_BACK,
                }
                and current_state
                not in {
                    DeploymentTargetState.FAILED,
                    DeploymentTargetState.ROLLED_BACK,
                }
                and TARGET_ORDER.get(requested_state, -1) < TARGET_ORDER.get(current_state, -1)
            ):
                raise ConflictError(
                    "deployment_target_state_regression",
                    "Deployment target state cannot move backwards",
                    currentState=current_state.value,
                    requestedState=requested_state.value,
                )
            if target.started_at is None:
                target.started_at = utc_now()
            target.state = requested_state.value
            target.progress = payload.progress
            target.stage = payload.stage
            target.error_code = payload.error_code
            target.error_message = payload.error_message
            deployment = self._deployment(session, target.deployment_id)
            task = self._task(session, target.task_id)
            event_level = "error" if requested_state == DeploymentTargetState.FAILED else "info"
            self._event(
                session,
                deployment,
                "target_progress",
                payload.message or payload.stage,
                level=event_level,
                target=target,
                data={"progress": payload.progress, "state": requested_state.value},
            )
            if requested_state == DeploymentTargetState.FAILED:
                target.completed_at = utc_now()
                if (
                    target.previous_task_status in ACTIVE_TASK_STATUSES
                    and target.previous_release_id
                ):
                    task.release_id = target.previous_release_id
                    task.status = InferenceTaskStatus.DEGRADED.value
                else:
                    task.status = InferenceTaskStatus.FAILED.value
                task.error_message = payload.error_message or payload.message
                deployment.status = (
                    DeploymentStatus.FAILED.value
                    if deployment.strategy == "all_at_once"
                    else DeploymentStatus.PAUSED.value
                )
                node.deployment_status = "failed"
            elif requested_state in {
                DeploymentTargetState.HEALTHY,
                DeploymentTargetState.ROLLED_BACK,
            }:
                target.progress = 100
                target.completed_at = utc_now()
                task.status = InferenceTaskStatus.RUNNING.value
                task.error_message = None
                if deployment.status == DeploymentStatus.ROLLING_BACK.value:
                    target.state = DeploymentTargetState.ROLLED_BACK.value
                self._advance_deployment(session, deployment, node, target.desired_revision)
            return deployment_target_response(target)

    def artifact_for_node(self, node_id: str, artifact_id: str, token: str) -> ArtifactRecord:
        with self.context.database.session() as session:
            node = self._authorized_node(session, node_id, token)
            tasks = session.scalars(
                select(InferenceTaskRecord).where(
                    InferenceTaskRecord.node_id == node.id,
                    InferenceTaskRecord.status.in_(ACTIVE_TASK_STATUSES),
                )
            ).all()
            allowed_release_ids = {task.release_id for task in tasks}
            for task in tasks:
                allowed_release_ids.update(self._secondary_release_ids(task.analytics_json))
            allowed = session.scalar(
                select(ModelReleaseRecord).where(
                    ModelReleaseRecord.id.in_(allowed_release_ids),
                    ModelReleaseRecord.rknn_artifact_id == artifact_id,
                )
            ) if allowed_release_ids else None
            if allowed is None:
                raise AuthenticationError("Artifact is not assigned to this inference node")
            artifact = session.get(ArtifactRecord, artifact_id)
            if artifact is None:
                raise NotFoundError("artifact", artifact_id)
            session.expunge(artifact)
            return artifact

    def _activate_next_batch(self, session: Session, deployment: DeploymentRecord) -> None:
        waiting = [
            item
            for item in self._targets(session, deployment.id)
            if item.state == DeploymentTargetState.PENDING.value and item.desired_revision == 0
        ]
        if not waiting:
            return
        if deployment.strategy == "all_at_once":
            selected = waiting
        else:
            completed = session.scalar(
                select(func.count())
                .select_from(DeploymentTargetRecord)
                .where(
                    DeploymentTargetRecord.deployment_id == deployment.id,
                    DeploymentTargetRecord.state.in_(
                        [
                            DeploymentTargetState.HEALTHY.value,
                            DeploymentTargetState.ROLLED_BACK.value,
                        ]
                    ),
                )
            )
            size = 1 if deployment.strategy == "canary" and not completed else deployment.batch_size
            selected = waiting[:size]
        nodes: dict[str, InferenceNodeRecord] = {}
        seen_nodes: set[str] = set()
        for target in selected:
            node = nodes.setdefault(target.node_id, self._node(session, target.node_id))
            if target.node_id not in seen_nodes:
                node.desired_revision += 1
                node.deployment_status = (
                    "rollback"
                    if deployment.status == DeploymentStatus.ROLLING_BACK.value
                    else "deploying"
                )
                seen_nodes.add(target.node_id)
        for target in selected:
            node = nodes[target.node_id]
            task = self._task(session, target.task_id)
            target.desired_revision = node.desired_revision
            target.stage = "queued"
            target.progress = 0
            target.error_code = None
            target.error_message = None
            task.release_id = target.release_id
            task.status = InferenceTaskStatus.DEPLOYING.value
            task.config_revision = node.desired_revision
            task.error_message = None
            MediaService(self.context).ensure_publish_credential(session, task)
            self._event(
                session,
                deployment,
                "target_activated",
                f"Activated rollout target for node {node.name}",
                target=target,
                data={"revision": node.desired_revision},
            )
        if deployment.status != DeploymentStatus.ROLLING_BACK.value:
            deployment.status = DeploymentStatus.ROLLING.value

    def _advance_deployment(
        self,
        session: Session,
        deployment: DeploymentRecord,
        node: InferenceNodeRecord,
        revision: int,
    ) -> None:
        node_targets = session.scalars(
            select(DeploymentTargetRecord).where(
                DeploymentTargetRecord.node_id == node.id,
                DeploymentTargetRecord.desired_revision == revision,
            )
        ).all()
        if node_targets and all(
            item.state
            in {
                DeploymentTargetState.HEALTHY.value,
                DeploymentTargetState.ROLLED_BACK.value,
            }
            for item in node_targets
        ):
            node.actual_revision = max(node.actual_revision, revision)
            node.deployment_status = "idle"
        targets = self._targets(session, deployment.id)
        assigned = [item for item in targets if item.desired_revision > 0]
        active = [
            item
            for item in assigned
            if item.state
            not in {
                DeploymentTargetState.HEALTHY.value,
                DeploymentTargetState.ROLLED_BACK.value,
            }
        ]
        if active:
            return
        waiting = [item for item in targets if item.desired_revision == 0]
        if waiting:
            self._activate_next_batch(session, deployment)
            return
        expected = (
            DeploymentTargetState.ROLLED_BACK.value
            if deployment.status == DeploymentStatus.ROLLING_BACK.value
            else DeploymentTargetState.HEALTHY.value
        )
        if all(item.state == expected for item in targets):
            deployment.status = (
                DeploymentStatus.ROLLED_BACK.value
                if expected == DeploymentTargetState.ROLLED_BACK.value
                else DeploymentStatus.SUCCEEDED.value
            )
            deployment.completed_at = utc_now()
            self._event(session, deployment, "completed", f"Deployment {deployment.status}")

    def _finish_stopped_rollbacks(self, session: Session, node: InferenceNodeRecord) -> None:
        targets = session.scalars(
            select(DeploymentTargetRecord).where(
                DeploymentTargetRecord.node_id == node.id,
                DeploymentTargetRecord.state == DeploymentTargetState.PENDING.value,
                DeploymentTargetRecord.desired_revision > 0,
                DeploymentTargetRecord.desired_revision <= node.actual_revision,
                DeploymentTargetRecord.stage == "stopping_for_rollback",
            )
        ).all()
        for target in targets:
            target.state = DeploymentTargetState.ROLLED_BACK.value
            target.progress = 100
            target.completed_at = utc_now()
            deployment = self._deployment(session, target.deployment_id)
            self._advance_deployment(session, deployment, node, target.desired_revision)

    def _validate_node_release(self, node: InferenceNodeRecord, adapter: str) -> None:
        if node.lifecycle != InferenceNodeLifecycle.ACTIVE.value or not node.self_test_passed:
            raise ConflictError(
                "inference_node_not_active",
                "Inference node is not approved and healthy",
                nodeId=node.id,
                lifecycle=node.lifecycle,
            )
        if adapter not in node.adapters_json:
            raise ConflictError(
                "inference_adapter_missing",
                "Inference node does not support this model output contract",
                nodeId=node.id,
                adapter=adapter,
            )

    @staticmethod
    def _runtime_instance_key(
        task: InferenceTaskRecord, release_id: str
    ) -> tuple[str, str, str, str, str, int, int]:
        thresholds = json.dumps(task.thresholds_json, sort_keys=True, separators=(",", ":"))
        return (
            "primary",
            release_id,
            task.npu_core_mask,
            task.npu_core_policy,
            thresholds,
            task.context_count,
            task.worker_count,
        )

    @staticmethod
    def _secondary_runtime_instance_keys(
        task: InferenceTaskRecord,
    ) -> list[tuple[str, str, str, str, str, int, int]]:
        raw = task.analytics_json.get("secondaryModels", [])
        if not isinstance(raw, list):
            return []
        keys: list[tuple[str, str, str, str, str, int, int]] = []
        for item in cast(list[object], raw):
            if not isinstance(item, dict):
                continue
            secondary = cast(dict[str, object], item)
            release_id = secondary.get("releaseId")
            if not isinstance(release_id, str) or not release_id:
                continue
            context_count = secondary.get("contextCount", 1)
            worker_count = secondary.get("workerCount", 1)
            if (
                not isinstance(context_count, int)
                or isinstance(context_count, bool)
                or context_count < 1
                or not isinstance(worker_count, int)
                or isinstance(worker_count, bool)
                or worker_count < 1
                or worker_count > context_count
            ):
                continue
            thresholds = json.dumps(
                {"confidence": secondary.get("confidenceThreshold", 0.25)},
                sort_keys=True,
                separators=(",", ":"),
            )
            keys.append(
                (
                    "secondary",
                    release_id,
                    task.npu_core_mask,
                    task.npu_core_policy,
                    thresholds,
                    context_count,
                    worker_count,
                )
            )
        return keys

    def _validate_node_runtime_plan(
        self,
        session: Session,
        node: InferenceNodeRecord,
        candidates: list[InferenceTaskRecord],
        release_id: str,
    ) -> None:
        active_tasks = session.scalars(
            select(InferenceTaskRecord).where(
                InferenceTaskRecord.node_id == node.id,
                InferenceTaskRecord.status.in_(ACTIVE_TASK_STATUSES),
            )
        ).all()
        planned: dict[str, tuple[InferenceTaskRecord, str]] = {
            task.id: (task, task.release_id) for task in active_tasks
        }
        planned.update({task.id: (task, release_id) for task in candidates})
        instances: dict[
            tuple[str, str, str, str, str, int, int], list[InferenceTaskRecord]
        ] = {}
        for task, planned_release_id in planned.values():
            key = self._runtime_instance_key(task, planned_release_id)
            instances.setdefault(key, []).append(task)
            for secondary_key in self._secondary_runtime_instance_keys(task):
                instances.setdefault(secondary_key, []).append(task)

        required_contexts = sum(key[5] for key in instances)
        if required_contexts > node.max_model_instances:
            raise ConflictError(
                "inference_node_capacity_exceeded",
                "Deployment would exceed the node RKNN context limit",
                nodeId=node.id,
                requiredContexts=required_contexts,
                maxContexts=node.max_model_instances,
            )

        instance_groups = list(instances.items())
        for index, (left_key, left_tasks) in enumerate(instance_groups):
            for right_key, right_tasks in instance_groups[index + 1 :]:
                left_policy = left_key[3]
                right_policy = right_key[3]
                if NpuCorePolicy.EXCLUSIVE.value not in {left_policy, right_policy}:
                    continue
                left_mask = left_key[2]
                right_mask = right_key[2]
                if NPU_CORE_BITS[left_mask] & NPU_CORE_BITS[right_mask] == 0:
                    continue
                raise ConflictError(
                    "inference_npu_core_conflict",
                    "Exclusive NPU core assignment overlaps another runtime instance",
                    nodeId=node.id,
                    leftTaskIds=[task.id for task in left_tasks],
                    rightTaskIds=[task.id for task in right_tasks],
                    leftCoreMask=left_mask,
                    rightCoreMask=right_mask,
                )

    def _create_deployment(
        self,
        session: Session,
        payload: DeploymentCreate,
        *,
        previous_task_statuses: dict[str, str] | None = None,
    ) -> DeploymentResponse:
        release = self._published_release(session, payload.release_id)
        tasks = [self._task(session, item) for item in payload.task_ids]
        tasks_by_node: dict[str, list[InferenceTaskRecord]] = {}
        for task in tasks:
            reserved_restart = (
                previous_task_statuses is not None and task.id in previous_task_statuses
            )
            if task.status == InferenceTaskStatus.RETIRED.value or (
                task.status == InferenceTaskStatus.DEPLOYING.value and not reserved_restart
            ):
                raise ConflictError(
                    "inference_task_not_deployable",
                    "Task cannot enter a new deployment in its current state",
                    taskId=task.id,
                    status=task.status,
                )
            node = self._validate_task_for_deployment(session, task, release)
            tasks_by_node.setdefault(node.id, []).append(task)
        for node_id, node_tasks in tasks_by_node.items():
            self._validate_node_runtime_plan(
                session, self._node(session, node_id), node_tasks, release.id
            )
        deployment = DeploymentRecord(
            id=new_id("deployment"),
            name=payload.name.strip(),
            status=DeploymentStatus.QUEUED.value,
            release_id=release.id,
            strategy=payload.strategy,
            batch_size=payload.batch_size,
        )
        session.add(deployment)
        session.flush()
        for sequence, task in enumerate(tasks):
            session.add(
                DeploymentTargetRecord(
                    id=new_id("dtarget"),
                    deployment_id=deployment.id,
                    node_id=task.node_id,
                    task_id=task.id,
                    release_id=release.id,
                    previous_release_id=task.release_id,
                    previous_task_status=(previous_task_statuses or {}).get(
                        task.id, task.status
                    ),
                    sequence=sequence,
                    desired_revision=0,
                    state=DeploymentTargetState.PENDING.value,
                    progress=0,
                    stage="waiting_for_batch",
                )
            )
        self._event(
            session,
            deployment,
            "created",
            "Deployment created and awaiting the first rollout batch",
            data={"strategy": payload.strategy, "taskIds": payload.task_ids},
        )
        session.flush()
        self._activate_next_batch(session, deployment)
        return self._deployment_response(session, deployment)

    def _validate_task_for_deployment(
        self,
        session: Session,
        task: InferenceTaskRecord,
        release: ModelReleaseRecord,
    ) -> InferenceNodeRecord:
        node = self._node(session, task.node_id)
        if task.media_migration_required:
            raise ConflictError(
                "media_migration_required",
                "Select a managed media gateway and stream before deployment",
                taskId=task.id,
            )
        self._validate_node_release(node, release.adapter)
        self._validate_node_media(node, task.media_json)
        MediaService(self.context).validate_task_media(
            session, task.media_json, task_id=task.id
        )
        secondary_releases = self._validate_task_analytics(
            session, task.analytics_json, release, task.media_json
        )
        self._validate_node_analytics(node, task.analytics_json, secondary_releases)
        return node

    def _select_node(
        self,
        session: Session,
        node_id: str | None,
        group_id: str | None,
        adapter: str,
        media: dict[str, Any],
        analytics: dict[str, Any],
        secondary_releases: list[ModelReleaseRecord],
    ) -> InferenceNodeRecord:
        if node_id:
            node = self._node(session, node_id)
            self._validate_node_release(node, adapter)
            self._validate_node_media(node, media)
            self._validate_node_analytics(node, analytics, secondary_releases)
            return node
        if group_id is None:
            raise ConflictError("inference_placement_required", "Node or node group is required")
        if session.get(NodeGroupRecord, group_id) is None:
            raise NotFoundError("node group", group_id)
        candidates = session.scalars(
            select(InferenceNodeRecord).where(
                InferenceNodeRecord.group_id == group_id,
                InferenceNodeRecord.lifecycle == InferenceNodeLifecycle.ACTIVE.value,
                InferenceNodeRecord.self_test_passed.is_(True),
            )
        ).all()
        candidates = [
            item
            for item in candidates
            if adapter in item.adapters_json
            and self._node_connectivity(item) == InferenceNodeConnectivity.ONLINE
            and not (
                self._required_media_features(media) - self._node_media_features(item)
            )
            and not (
                self._required_analytics_features(analytics) - self._node_media_features(item)
            )
            and all(release.adapter in item.adapters_json for release in secondary_releases)
        ]
        if not candidates:
            raise ConflictError(
                "no_compatible_inference_node",
                "Node group has no online node supporting the required adapter",
                groupId=group_id,
                adapter=adapter,
            )
        load_by_node = {
            item.id: session.scalar(
                select(func.count())
                .select_from(InferenceTaskRecord)
                .where(
                    InferenceTaskRecord.node_id == item.id,
                    InferenceTaskRecord.status.in_(ACTIVE_TASK_STATUSES),
                )
            )
            or 0
            for item in candidates
        }
        return min(candidates, key=lambda item: (load_by_node[item.id], item.name))

    def _deployment_response(
        self, session: Session, record: DeploymentRecord
    ) -> DeploymentResponse:
        return DeploymentResponse(
            id=record.id,
            name=record.name,
            status=DeploymentStatus(record.status),
            release_id=record.release_id,
            strategy=record.strategy,
            batch_size=record.batch_size,
            targets=[
                deployment_target_response(item) for item in self._targets(session, record.id)
            ],
            created_at=record.created_at,
            updated_at=record.updated_at,
            completed_at=record.completed_at,
        )

    def _event(
        self,
        session: Session,
        deployment: DeploymentRecord,
        event_type: str,
        message: str,
        *,
        level: str = "info",
        target: DeploymentTargetRecord | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            DeploymentEventRecord(
                deployment_id=deployment.id,
                target_id=target.id if target else None,
                node_id=target.node_id if target else None,
                type=event_type,
                level=level,
                message=message,
                data_json=data or {},
            )
        )

    def _node_response(self, record: InferenceNodeRecord) -> InferenceNodeResponse:
        return InferenceNodeResponse(
            id=record.id,
            name=record.name,
            group_id=record.group_id,
            labels=record.labels_json,
            lifecycle=InferenceNodeLifecycle(record.lifecycle),
            connectivity=self._node_connectivity(record),
            health=InferenceNodeHealth(record.health),
            deployment_status=record.deployment_status,
            max_model_instances=record.max_model_instances,
            hardware_id=record.hardware_id,
            runtime_version=record.runtime_version,
            driver_version=record.driver_version,
            pipeline_version=record.pipeline_version,
            adapters=record.adapters_json,
            metadata=record.metadata_json,
            desired_revision=record.desired_revision,
            actual_revision=record.actual_revision,
            self_test_passed=record.self_test_passed,
            last_seen_at=record.last_seen_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def _node_connectivity(self, record: InferenceNodeRecord) -> InferenceNodeConnectivity:
        if record.last_seen_at is None:
            return InferenceNodeConnectivity.OFFLINE
        age = (utc_now() - as_utc(record.last_seen_at)).total_seconds()
        if age > self.context.settings.inference_node_offline_seconds:
            return InferenceNodeConnectivity.OFFLINE
        return InferenceNodeConnectivity(record.connectivity)

    @staticmethod
    def _release(session: Session, release_id: str) -> ModelReleaseRecord:
        record = session.get(ModelReleaseRecord, release_id)
        if record is None:
            raise NotFoundError("model release", release_id)
        return record

    def _published_release(self, session: Session, release_id: str) -> ModelReleaseRecord:
        record = self._release(session, release_id)
        if record.status != ModelReleaseStatus.PUBLISHED.value:
            raise ConflictError(
                "model_release_not_published",
                "Inference tasks require a published model release",
                status=record.status,
            )
        return record

    @staticmethod
    def _node(session: Session, node_id: str) -> InferenceNodeRecord:
        record = session.get(InferenceNodeRecord, node_id)
        if record is None:
            raise NotFoundError("inference node", node_id)
        return record

    def _authorized_node(self, session: Session, node_id: str, token: str) -> InferenceNodeRecord:
        record = self._node(session, node_id)
        if record.access_token_hash is None or not hmac.compare_digest(
            record.access_token_hash, _token_hash(token)
        ):
            raise AuthenticationError("Invalid inference-node access token")
        if record.lifecycle == InferenceNodeLifecycle.RETIRED.value:
            raise AuthenticationError("Inference node has been retired")
        return record

    @staticmethod
    def _task(session: Session, task_id: str) -> InferenceTaskRecord:
        record = session.get(InferenceTaskRecord, task_id)
        if record is None:
            raise NotFoundError("inference task", task_id)
        return record

    @staticmethod
    def _deployment(session: Session, deployment_id: str) -> DeploymentRecord:
        record = session.get(DeploymentRecord, deployment_id)
        if record is None:
            raise NotFoundError("deployment", deployment_id)
        return record

    @staticmethod
    def _targets(session: Session, deployment_id: str) -> list[DeploymentTargetRecord]:
        return list(
            session.scalars(
                select(DeploymentTargetRecord)
                .where(DeploymentTargetRecord.deployment_id == deployment_id)
                .order_by(DeploymentTargetRecord.sequence)
            ).all()
        )
