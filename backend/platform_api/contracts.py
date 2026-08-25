from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, cast
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator

from .api_models import ApiModel
from .media_contracts import PreviewCapability, TaskMediaConfig


class TaskType(StrEnum):
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    OCR_DETECTION = "ocr_detection"
    OCR_RECOGNITION = "ocr_recognition"


class DatasetFormat(StrEnum):
    AUTO = "auto"
    YOLO = "yolo"
    COCO_DETECTION = "coco_detection"
    VOC_DETECTION = "voc_detection"
    MASK_PAIRS = "mask_pairs"
    COCO_SEGMENTATION = "coco_segmentation"
    VOC_SEGMENTATION = "voc_segmentation"
    PPOCR_DETECTION = "ppocr_detection"
    PPOCR_RECOGNITION = "ppocr_recognition"


DATASET_FORMAT_TASKS: dict[DatasetFormat, TaskType] = {
    DatasetFormat.YOLO: TaskType.OBJECT_DETECTION,
    DatasetFormat.COCO_DETECTION: TaskType.OBJECT_DETECTION,
    DatasetFormat.VOC_DETECTION: TaskType.OBJECT_DETECTION,
    DatasetFormat.MASK_PAIRS: TaskType.SEMANTIC_SEGMENTATION,
    DatasetFormat.COCO_SEGMENTATION: TaskType.SEMANTIC_SEGMENTATION,
    DatasetFormat.VOC_SEGMENTATION: TaskType.SEMANTIC_SEGMENTATION,
    DatasetFormat.PPOCR_DETECTION: TaskType.OCR_DETECTION,
    DatasetFormat.PPOCR_RECOGNITION: TaskType.OCR_RECOGNITION,
}


class Precision(StrEnum):
    INT8 = "int8"
    FP16 = "fp16"


class DatasetStatus(StrEnum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"


class JobType(StrEnum):
    TRAINING = "training"
    CONVERSION = "conversion"


class JobStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerKind(StrEnum):
    TRAINER = "trainer"
    CONVERTER = "converter"


class ServiceEndpointKind(StrEnum):
    TRAINER = "trainer"
    CONVERTER = "converter"
    INFERENCE = "inference"


class ServiceEndpointMode(StrEnum):
    PULL = "pull"
    DIRECT = "direct"


class ServiceEndpointEnrollmentStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    ENROLLED = "enrolled"


class WorkerStatus(StrEnum):
    ONLINE = "online"
    BUSY = "busy"
    OFFLINE = "offline"


class ModelReleaseStatus(StrEnum):
    QUALIFIED = "qualified"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"


class InferenceNodeLifecycle(StrEnum):
    PENDING_REGISTRATION = "pending_registration"
    AWAITING_APPROVAL = "awaiting_approval"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    RETIRED = "retired"


class InferenceNodeConnectivity(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"


class InferenceNodeHealth(StrEnum):
    UNKNOWN = "unknown"
    VALIDATING = "validating"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class InferenceTaskStatus(StrEnum):
    DRAFT = "draft"
    STOPPED = "stopped"
    DEPLOYING = "deploying"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    RETIRED = "retired"


class NpuCoreMask(StrEnum):
    AUTO = "auto"
    CORE_0 = "core0"
    CORE_1 = "core1"
    CORE_2 = "core2"
    CORE_0_1 = "core0_1"
    CORE_0_1_2 = "core0_1_2"


class NpuCorePolicy(StrEnum):
    SHARED = "shared"
    EXCLUSIVE = "exclusive"


class DeploymentStatus(StrEnum):
    QUEUED = "queued"
    ROLLING = "rolling"
    SUCCEEDED = "succeeded"
    PAUSED = "paused"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class DeploymentTargetState(StrEnum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    STAGED = "staged"
    DRAINING = "draining"
    ACTIVATING = "activating"
    WARMING = "warming"
    HEALTHY = "healthy"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class Resolution(ApiModel):
    width: int
    height: int


class ResolutionRule(ApiModel):
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    width_multiple: int
    height_multiple: int


class InputProfile(ApiModel):
    name: str
    layout: Literal["NCHW", "NHWC"]
    channels: int
    dtype: str
    color_space: Literal["RGB", "BGR", "GRAY"]
    resize_policy: str


class PreprocessingProfile(ApiModel):
    mean: list[float]
    std: list[float]

    @model_validator(mode="after")
    def validate_channels(self) -> PreprocessingProfile:
        if len(self.mean) != len(self.std):
            msg = "mean and std must have the same number of channels"
            raise ValueError(msg)
        if any(value == 0 for value in self.std):
            msg = "std values must be non-zero"
            raise ValueError(msg)
        return self


class ExportProfile(ApiModel):
    opset: int
    batch: int
    dynamic: bool


class RknnProfile(ApiModel):
    target_platform: Literal["rk3588"]
    quantized_algorithm: str
    optimization_level: int
    requires_calibration_for: list[Precision]


class VariantContract(ApiModel):
    exporter: str
    opset: int = Field(ge=9, le=21)
    output_contract: str


class ModelProfile(ApiModel):
    id: str
    family: str
    label: str
    task_type: TaskType
    framework: str
    variants: list[str]
    source_formats: list[str]
    precisions: list[Precision]
    default_resolution: Resolution
    resolution_rule: ResolutionRule
    input: InputProfile
    preprocessing: PreprocessingProfile
    output_contract: str
    export: ExportProfile
    rknn: RknnProfile
    variant_contracts: dict[str, VariantContract] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_variant_contracts(self) -> ModelProfile:
        unknown = set(self.variant_contracts) - set(self.variants)
        if unknown:
            msg = f"variant contracts reference unknown variants: {sorted(unknown)}"
            raise ValueError(msg)
        return self


class ModelProfileDocument(ApiModel):
    schema_version: int
    profiles: list[ModelProfile]


class TensorContract(ApiModel):
    name: str
    layout: Literal["NCHW", "NHWC"]
    shape: list[int]
    dtype: str
    color_space: Literal["RGB", "BGR", "GRAY"]

    @field_validator("shape")
    @classmethod
    def validate_static_shape(cls, value: list[int]) -> list[int]:
        if len(value) != 4 or any(dimension <= 0 for dimension in value):
            msg = "tensor shape must contain four positive static dimensions"
            raise ValueError(msg)
        return value


class OutputTensor(ApiModel):
    name: str
    semantic: str


class DeploymentManifest(ApiModel):
    schema_version: Literal[1] = 1
    model_family: str
    profile_id: str
    variant: str
    task_type: TaskType
    training_job_id: str
    onnx_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    opset: int
    resolution: Resolution
    input: TensorContract
    preprocessing: PreprocessingProfile
    resize_policy: str
    output_contract: str
    outputs: list[OutputTensor]
    labels: list[str] = Field(default_factory=list)
    supported_precisions: list[Precision]
    rknn: RknnProfile


class DatasetMetadata(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    version: str = Field(default="v1", min_length=1, max_length=40)
    task_type: TaskType
    dataset_format: DatasetFormat = DatasetFormat.AUTO
    classes: list[str] = Field(default_factory=list, max_length=10000)

    @field_validator("classes")
    @classmethod
    def clean_classes(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if len(cleaned) != len(set(cleaned)):
            msg = "dataset classes must be unique"
            raise ValueError(msg)
        return cleaned

    @model_validator(mode="after")
    def validate_dataset_format(self) -> DatasetMetadata:
        expected_task = DATASET_FORMAT_TASKS.get(self.dataset_format)
        if expected_task is not None and expected_task != self.task_type:
            raise ValueError(
                f"dataset format '{self.dataset_format}' is not valid for task '{self.task_type}'"
            )
        if (
            self.task_type == TaskType.SEMANTIC_SEGMENTATION
            and self.classes
            and len(self.classes) < 2
        ):
            raise ValueError("segmentation datasets require at least two classes")
        if self.task_type == TaskType.SEMANTIC_SEGMENTATION and len(self.classes) > 256:
            raise ValueError("segmentation datasets support at most 256 classes")
        return self


class DatasetClassesUpdate(ApiModel):
    classes: list[str] = Field(min_length=1, max_length=10000)

    @field_validator("classes")
    @classmethod
    def clean_classes(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("dataset classes must not be empty")
        if len(cleaned) != len(value):
            raise ValueError("dataset classes must not contain empty names")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("dataset classes must be unique")
        return cleaned


class DatasetResponse(ApiModel):
    id: str
    name: str
    description: str
    version: str
    task_type: TaskType
    dataset_format: DatasetFormat
    classes: list[str]
    status: DatasetStatus
    filename: str
    size_bytes: int
    sha256: str
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None


class TrainingHyperparameters(ApiModel):
    epochs: int = Field(default=100, ge=1, le=10000)
    batch_size: int = Field(default=16, ge=1, le=1024)
    learning_rate: float | None = Field(default=None, gt=0)
    optimizer: Literal["auto", "AdamW", "SGD"] = "auto"
    pretrained: bool = True
    seed: int = Field(default=42, ge=0)


class TrainingJobCreate(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    dataset_id: str
    profile_id: str
    variant: str
    resolution: Resolution
    hyperparameters: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    accelerator: Literal["cpu", "cuda"]

    @model_validator(mode="after")
    def validate_training_controls(self) -> TrainingJobCreate:
        if (
            self.hyperparameters.learning_rate is not None
            and self.hyperparameters.optimizer == "auto"
        ):
            raise ValueError("A custom learning rate requires an explicit optimizer")
        if (
            self.profile_id == "yolo-detect"
            and self.variant.startswith(("yolov6", "yolov7"))
            and self.hyperparameters.optimizer == "AdamW"
        ):
            raise ValueError(f"{self.variant} does not support the AdamW optimizer")
        return self


class ConversionJobCreate(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    source_artifact_id: str
    precision: Precision
    calibration_dataset_id: str | None = None


class JobResponse(ApiModel):
    id: str
    type: JobType
    name: str
    status: JobStatus
    profile_id: str
    dataset_id: str | None
    worker_id: str | None
    progress: int
    stage: str
    spec: dict[str, Any]
    result: dict[str, Any] | None
    retry_count: int
    max_retries: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class WorkerRegistration(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    kind: WorkerKind
    capabilities: list[str]
    accelerator: Literal["cpu", "cuda", "rk3588"]
    max_concurrency: int = Field(default=1, ge=1, le=64)
    version: str = Field(default="unknown", max_length=80)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerResponse(ApiModel):
    id: str
    name: str
    kind: WorkerKind
    status: WorkerStatus
    capabilities: list[str]
    accelerator: str
    max_concurrency: int
    active_jobs: int
    version: str
    metadata: dict[str, Any]
    last_seen_at: datetime
    created_at: datetime


class WorkerHeartbeat(ApiModel):
    active_jobs: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ServiceEndpointPayload(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    kind: ServiceEndpointKind
    mode: ServiceEndpointMode = ServiceEndpointMode.PULL
    endpoint: str = Field(default="", max_length=500)
    scheme: Literal["http", "https"] = "http"
    host: str = Field(default="", max_length=255)
    port: int = Field(default=10081, ge=1, le=65535)
    accelerator: Literal["cpu", "cuda", "rk3588"]
    capabilities: list[str] = Field(min_length=1, max_length=32)
    enabled: bool = True
    token: str | None = Field(default=None, min_length=16, max_length=512)

    @field_validator("capabilities")
    @classmethod
    def clean_capabilities(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("capabilities must be unique")
        if not cleaned:
            raise ValueError("at least one capability is required")
        return cleaned

    @model_validator(mode="after")
    def validate_kind_and_accelerator(self) -> ServiceEndpointPayload:
        if (
            self.kind in {ServiceEndpointKind.CONVERTER, ServiceEndpointKind.INFERENCE}
            and self.accelerator != "rk3588"
        ):
            raise ValueError("converter and inference services must use the rk3588 accelerator")
        if self.kind == ServiceEndpointKind.TRAINER and self.accelerator not in {"cpu", "cuda"}:
            raise ValueError("trainer services must use cpu or cuda")
        if self.kind == ServiceEndpointKind.INFERENCE and self.mode != ServiceEndpointMode.DIRECT:
            raise ValueError("inference services must use direct mode")
        if self.mode == ServiceEndpointMode.DIRECT:
            if not self.host.strip():
                if not self.endpoint:
                    raise ValueError("direct endpoints require a host")
                parsed_endpoint = urlparse(self.endpoint.rstrip("/"))
                if parsed_endpoint.hostname is None:
                    raise ValueError("endpoint must contain a host")
                self.host = parsed_endpoint.hostname
                self.scheme = parsed_endpoint.scheme  # type: ignore[assignment]
                self.port = parsed_endpoint.port or (443 if self.scheme == "https" else 80)
            if self.endpoint:
                parsed = urlparse(self.endpoint.rstrip("/"))
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    raise ValueError("endpoint must be an absolute HTTP(S) URL")
                if parsed.username is not None or parsed.password is not None:
                    raise ValueError("endpoint must not contain credentials")
            self.endpoint = f"{self.scheme}://{self.host}:{self.port}"
        else:
            normalized = self.endpoint.strip().rstrip("/")
            parsed = urlparse(normalized)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("endpoint must be an absolute HTTP(S) URL")
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("endpoint must not contain credentials")
            self.endpoint = normalized
            self.scheme = parsed.scheme  # type: ignore[assignment]
            self.host = parsed.hostname or ""
            self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return self


class ServiceEndpointResponse(ApiModel):
    id: str
    name: str
    kind: ServiceEndpointKind
    mode: ServiceEndpointMode
    endpoint: str
    scheme: Literal["http", "https"]
    host: str
    port: int
    accelerator: Literal["cpu", "cuda", "rk3588"]
    capabilities: list[str]
    enabled: bool
    token_configured: bool
    enrollment_status: ServiceEndpointEnrollmentStatus
    enrollment_expires_at: datetime | None
    enrollment_claimed_at: datetime | None
    enrolled_at: datetime | None
    probe_status: str
    last_probe_at: datetime | None
    last_error: str | None
    remote_metadata: dict[str, Any]
    inference_node_id: str | None
    created_at: datetime
    updated_at: datetime


class ServiceEndpointCreateResponse(ServiceEndpointResponse):
    enrollment_token: str | None = None


class ServiceEndpointEnrollmentResponse(ApiModel):
    endpoint_id: str
    enrollment_status: ServiceEndpointEnrollmentStatus
    enrollment_token: str
    enrollment_expires_at: datetime


class NodeEnrollmentClaim(ApiModel):
    enrollment_token: str = Field(min_length=16, max_length=512)
    name: str = Field(min_length=1, max_length=120)
    kind: ServiceEndpointKind
    accelerator: Literal["cpu", "cuda", "rk3588"]
    capabilities: list[str] = Field(min_length=1, max_length=32)
    version: str = Field(default="unknown", min_length=1, max_length=80)
    max_concurrency: int = Field(default=1, ge=1, le=1024)
    features: list[str] = Field(default_factory=list, max_length=64)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class NodeEnrollmentClaimResponse(ApiModel):
    endpoint_id: str
    node_token: str
    enrollment_status: ServiceEndpointEnrollmentStatus


class ServiceEndpointTestResponse(ApiModel):
    ok: bool
    endpoint: str
    message: str = ""
    remote: dict[str, Any] = Field(default_factory=dict)


class JobClaimRequest(ApiModel):
    worker_id: str
    job_id: str | None = None


class JobClaim(ApiModel):
    job: JobResponse
    lease_token: str
    lease_expires_at: datetime


class WorkspaceRetentionResponse(ApiModel):
    job_ids: list[str]


class JobProgressUpdate(ApiModel):
    lease_token: str
    progress: int = Field(ge=0, le=100)
    stage: str = Field(min_length=1, max_length=120)
    message: str = Field(default="", max_length=2000)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)


class JobTelemetryEntry(ApiModel):
    type: Literal["log", "metric"]
    level: Literal["debug", "info", "warning", "error"] = "info"
    message: str = Field(default="", max_length=16000)
    stage: str = Field(min_length=1, max_length=120)
    metrics: dict[str, float] = Field(default_factory=dict)
    step: int | None = Field(default=None, ge=0)
    epoch: int | None = Field(default=None, ge=1)
    total_epochs: int | None = Field(default=None, ge=1)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if len(value) > 64:
            raise ValueError("telemetry entries support at most 64 metrics")
        for name, metric in value.items():
            if not name or len(name) > 80:
                raise ValueError("metric names must contain between 1 and 80 characters")
            if not math.isfinite(metric):
                raise ValueError(f"metric '{name}' must be finite")
        return value


class JobTelemetryUpdate(ApiModel):
    lease_token: str
    entries: list[JobTelemetryEntry] = Field(min_length=1, max_length=100)


class JobTelemetryAccepted(ApiModel):
    accepted: int


class JobEventResponse(ApiModel):
    id: int
    type: str
    level: str
    message: str
    data: dict[str, Any]
    created_at: datetime


class JobLeaseRenewal(ApiModel):
    lease_token: str


class JobFailure(ApiModel):
    lease_token: str
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    retryable: bool = False


class JobCompletion(ApiModel):
    lease_token: str
    result: dict[str, Any] = Field(default_factory=dict)


class ArtifactResponse(ApiModel):
    id: str
    job_id: str | None
    kind: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    manifest: dict[str, Any] | None
    created_at: datetime


class PageInfo(ApiModel):
    page: int
    page_size: int
    total: int


class ModelReleaseCreate(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=80)
    conversion_job_id: str
    description: str = Field(default="", max_length=1000)

    @field_validator("name", "version")
    @classmethod
    def strip_release_identity(cls, value: str) -> str:
        return value.strip()


class ModelReleaseResponse(ApiModel):
    id: str
    name: str
    version: str
    description: str
    status: ModelReleaseStatus
    profile_id: str
    variant: str
    task_type: TaskType
    precision: Precision
    adapter: str
    rknn_artifact_id: str
    validation_artifact_id: str | None
    source_training_job_id: str | None
    source_conversion_job_id: str
    dataset_id: str | None
    manifest: dict[str, Any]
    created_at: datetime
    published_at: datetime | None


class ModelReleaseListResponse(PageInfo):
    items: list[ModelReleaseResponse]


class NodeGroupCreate(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    labels: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("labels")
    @classmethod
    def clean_group_labels(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("labels must be unique")
        return cleaned


class NodeGroupUpdate(NodeGroupCreate):
    pass


class NodeGroupResponse(NodeGroupCreate):
    id: str
    created_at: datetime
    updated_at: datetime


class InferenceNodeCreate(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    group_id: str | None = None
    labels: list[str] = Field(default_factory=list, max_length=64)
    max_model_instances: int = Field(default=1, ge=1, le=16)
    registration_ttl_seconds: int = Field(default=900, ge=60, le=86400)

    @field_validator("labels")
    @classmethod
    def clean_node_labels(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("labels must be unique")
        return cleaned


class InferenceNodeResponse(ApiModel):
    id: str
    name: str
    group_id: str | None
    labels: list[str]
    lifecycle: InferenceNodeLifecycle
    connectivity: InferenceNodeConnectivity
    health: InferenceNodeHealth
    deployment_status: str
    max_model_instances: int
    hardware_id: str | None
    runtime_version: str | None
    driver_version: str | None
    pipeline_version: str | None
    adapters: list[str]
    metadata: dict[str, Any]
    desired_revision: int
    actual_revision: int
    self_test_passed: bool
    last_seen_at: datetime | None
    created_at: datetime
    updated_at: datetime


class InferenceNodeCreated(InferenceNodeResponse):
    registration_token: str
    registration_expires_at: datetime


class InferenceNodeListResponse(PageInfo):
    items: list[InferenceNodeResponse]


class InferenceSummaryResponse(ApiModel):
    online_nodes: int
    total_nodes: int
    published_releases: int
    running_tasks: int
    active_deployments: int


class InferenceNodeRegistration(ApiModel):
    node_id: str
    registration_token: str = Field(min_length=16, max_length=512)
    hardware_id: str = Field(min_length=1, max_length=255)
    runtime_version: str = Field(min_length=1, max_length=80)
    driver_version: str = Field(min_length=1, max_length=80)
    pipeline_version: str = Field(min_length=1, max_length=80)
    adapters: list[str] = Field(default_factory=list, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceNodeRegistrationResponse(ApiModel):
    node: InferenceNodeResponse
    access_token: str


class InferenceNodeHeartbeat(ApiModel):
    actual_revision: int = Field(default=0, ge=0)
    failed_revision: int | None = Field(default=None, ge=0)
    health: InferenceNodeHealth = InferenceNodeHealth.HEALTHY
    self_test_passed: bool = False
    runtime_version: str | None = Field(default=None, min_length=1, max_length=80)
    driver_version: str | None = Field(default=None, min_length=1, max_length=80)
    pipeline_version: str | None = Field(default=None, min_length=1, max_length=80)
    adapters: list[str] = Field(default_factory=list, max_length=64)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceTaskCreate(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    release_id: str
    node_id: str | None = None
    group_id: str | None = None
    input_uri: str = Field(min_length=1, max_length=2000)
    interval: int = Field(default=1, ge=1, le=10000)
    thresholds: dict[str, float] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=lambda: {"type": "jsonl"})
    media: dict[str, Any] = Field(default_factory=dict)
    analytics: dict[str, Any] = Field(default_factory=dict)
    npu_core_mask: NpuCoreMask = NpuCoreMask.AUTO
    npu_core_policy: NpuCorePolicy = NpuCorePolicy.SHARED
    context_count: int = Field(default=1, ge=1)
    worker_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_task_placement(self) -> InferenceTaskCreate:
        if bool(self.node_id) == bool(self.group_id):
            raise ValueError("exactly one of nodeId or groupId is required")
        if self.worker_count > self.context_count:
            raise ValueError("workerCount must not exceed contextCount")
        if (
            self.npu_core_policy == NpuCorePolicy.EXCLUSIVE
            and self.npu_core_mask == NpuCoreMask.AUTO
        ):
            raise ValueError("exclusive NPU core policy requires an explicit core mask")
        output_type = self.output.get("type", "jsonl")
        if output_type == "jsonl":
            self._validate_media()
            self._validate_analytics()
            return self
        if output_type != "http":
            raise ValueError("output.type must be jsonl or http")
        url = self.output.get("url")
        parsed = urlparse(url) if isinstance(url, str) else None
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("HTTP output requires an HTTP(S) URL without embedded credentials")
        connect_timeout = self.output.get("connectTimeoutMs", 1000)
        request_timeout = self.output.get("requestTimeoutMs", 3000)
        if (
            not isinstance(connect_timeout, int)
            or isinstance(connect_timeout, bool)
            or not isinstance(request_timeout, int)
            or isinstance(request_timeout, bool)
            or not 100 <= connect_timeout <= request_timeout <= 60000
        ):
            raise ValueError(
                "output timeouts must satisfy "
                "100 <= connectTimeoutMs <= requestTimeoutMs <= 60000"
            )
        authorization_env = self.output.get("authorizationEnv", "")
        if not isinstance(authorization_env, str) or (
            authorization_env
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", authorization_env) is None
        ):
            raise ValueError("output.authorizationEnv must be a valid environment name")
        self._validate_media()
        self._validate_analytics()
        return self

    def _validate_media(self) -> None:
        TaskMediaConfig.model_validate(self.media)
        allowed = {"decoder", "tracking", "kafka", "zlmSei"}
        unknown = set(self.media) - allowed
        if unknown:
            raise ValueError(f"media contains unsupported fields: {sorted(unknown)}")
        decoder = self.media.get("decoder", "opencv")
        if decoder not in {"opencv", "rkmpp"}:
            raise ValueError("media.decoder must be opencv or rkmpp")
        if decoder == "rkmpp" and not self.input_uri.startswith("rtsp://"):
            raise ValueError("RKMPP decoder requires an RTSP input")
        tracking = self.media.get("tracking", {})
        kafka = self.media.get("kafka", {})
        zlm = self.media.get("zlmSei", {})
        if not all(isinstance(item, dict) for item in (tracking, kafka, zlm)):
            raise ValueError("media tracking, kafka and zlmSei settings must be objects")
        track_buffer = tracking.get("trackBuffer", 30)
        queue_messages = kafka.get("queueMessages", 10000)
        message_timeout = kafka.get("messageTimeoutMs", 3000)
        reconnect = zlm.get("reconnectMs", 1000)
        if (
            not isinstance(track_buffer, int)
            or isinstance(track_buffer, bool)
            or not 1 <= track_buffer <= 10000
        ):
            raise ValueError("media.tracking.trackBuffer must be between 1 and 10000")
        if kafka.get("enabled") is True:
            if not isinstance(kafka.get("brokers"), str) or not kafka.get("brokers", "").strip():
                raise ValueError("enabled Kafka requires media.kafka.brokers")
            topic = kafka.get("topic", "sei_msg")
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError("enabled Kafka requires media.kafka.topic")
        if (
            not isinstance(queue_messages, int)
            or isinstance(queue_messages, bool)
            or not 1 <= queue_messages <= 1000000
        ):
            raise ValueError("media.kafka.queueMessages must be between 1 and 1000000")
        if (
            not isinstance(message_timeout, int)
            or isinstance(message_timeout, bool)
            or not 100 <= message_timeout <= 60000
        ):
            raise ValueError("media.kafka.messageTimeoutMs must be between 100 and 60000")
        if zlm.get("enabled") is True and decoder != "rkmpp":
            raise ValueError("ZLM SEI requires the RKMPP decoder")
        if (
            not isinstance(reconnect, int)
            or isinstance(reconnect, bool)
            or not 1000 <= reconnect <= 4000
        ):
            raise ValueError("media.zlmSei.reconnectMs must be between 1000 and 4000")

    @staticmethod
    def _validate_normalized_point(value: object, path: str) -> tuple[float, float]:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain only x and y")
        point = cast(dict[str, object], value)
        if set(point) != {"x", "y"}:
            raise ValueError(f"{path} must contain only x and y")
        x = point.get("x")
        y = point.get("y")
        if (
            not isinstance(x, (int, float))
            or isinstance(x, bool)
            or not isinstance(y, (int, float))
            or isinstance(y, bool)
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
            or not 0 <= float(x) <= 1
            or not 0 <= float(y) <= 1
        ):
            raise ValueError(f"{path} coordinates must be finite numbers between 0 and 1")
        return float(x), float(y)

    @staticmethod
    def _validate_class_ids(value: object, path: str) -> None:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array with at most 256 entries")
        class_ids = cast(list[object], value)
        if len(class_ids) > 256:
            raise ValueError(f"{path} must be an array with at most 256 entries")
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in class_ids
        ):
            raise ValueError(f"{path} entries must be non-negative integers")
        if len(class_ids) != len(set(class_ids)):
            raise ValueError(f"{path} must not contain duplicates")

    @staticmethod
    def _validate_bool_fields(value: dict[str, Any], allowed: set[str], path: str) -> None:
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"{path} contains unsupported fields: {sorted(unknown)}")
        for key, item in value.items():
            if not isinstance(item, bool):
                raise ValueError(f"{path}.{key} must be a boolean")

    def _validate_analytics(self) -> None:
        allowed = {"areas", "lines", "osd", "events", "secondaryModels"}
        unknown = set(self.analytics) - allowed
        if unknown:
            raise ValueError(f"analytics contains unsupported fields: {sorted(unknown)}")

        areas = self.analytics.get("areas", [])
        lines = self.analytics.get("lines", [])
        secondary_models = self.analytics.get("secondaryModels", [])
        osd = self.analytics.get("osd", {})
        events = self.analytics.get("events", {})
        if not isinstance(areas, list):
            raise ValueError("analytics.areas must be an array with at most 32 entries")
        area_items = cast(list[object], areas)
        if len(area_items) > 32:
            raise ValueError("analytics.areas must be an array with at most 32 entries")
        if not isinstance(lines, list):
            raise ValueError("analytics.lines must be an array with at most 32 entries")
        line_items = cast(list[object], lines)
        if len(line_items) > 32:
            raise ValueError("analytics.lines must be an array with at most 32 entries")
        if not isinstance(secondary_models, list):
            raise ValueError("analytics.secondaryModels must be an array with at most 4 entries")
        secondary_items = cast(list[object], secondary_models)
        if len(secondary_items) > 4:
            raise ValueError("analytics.secondaryModels must be an array with at most 4 entries")
        if not isinstance(osd, dict) or not isinstance(events, dict):
            raise ValueError("analytics osd and events settings must be objects")
        osd_config = cast(dict[str, object], osd)
        event_config = cast(dict[str, object], events)

        ids: set[str] = set()
        for index, raw_area in enumerate(area_items):
            path = f"analytics.areas[{index}]"
            if not isinstance(raw_area, dict):
                raise ValueError(f"{path} must be an object")
            area = cast(dict[str, object], raw_area)
            unknown_area = set(area) - {
                "id", "name", "polygon", "classIds", "minCount", "holdFrames"
            }
            if unknown_area:
                raise ValueError(f"{path} contains unsupported fields: {sorted(unknown_area)}")
            area_id = area.get("id")
            if not isinstance(area_id, str) or not 1 <= len(area_id.strip()) <= 80:
                raise ValueError(f"{path}.id must contain between 1 and 80 characters")
            if area_id in ids:
                raise ValueError("analytics area and line ids must be unique")
            ids.add(area_id)
            name = area.get("name", area_id)
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
                raise ValueError(f"{path}.name must contain between 1 and 120 characters")
            polygon = area.get("polygon")
            if not isinstance(polygon, list):
                raise ValueError(f"{path}.polygon must contain between 3 and 32 points")
            polygon_points = cast(list[object], polygon)
            if not 3 <= len(polygon_points) <= 32:
                raise ValueError(f"{path}.polygon must contain between 3 and 32 points")
            for point_index, point in enumerate(polygon_points):
                self._validate_normalized_point(point, f"{path}.polygon[{point_index}]")
            self._validate_class_ids(area.get("classIds", []), f"{path}.classIds")
            for field, maximum in (("minCount", 100000), ("holdFrames", 10000)):
                value = area.get(field, 1)
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 1 <= value <= maximum
                ):
                    raise ValueError(f"{path}.{field} must be between 1 and {maximum}")

        for index, raw_line in enumerate(line_items):
            path = f"analytics.lines[{index}]"
            if not isinstance(raw_line, dict):
                raise ValueError(f"{path} must be an object")
            line = cast(dict[str, object], raw_line)
            unknown_line = set(line) - {"id", "name", "start", "end", "direction", "classIds"}
            if unknown_line:
                raise ValueError(f"{path} contains unsupported fields: {sorted(unknown_line)}")
            line_id = line.get("id")
            if not isinstance(line_id, str) or not 1 <= len(line_id.strip()) <= 80:
                raise ValueError(f"{path}.id must contain between 1 and 80 characters")
            if line_id in ids:
                raise ValueError("analytics area and line ids must be unique")
            ids.add(line_id)
            name = line.get("name", line_id)
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
                raise ValueError(f"{path}.name must contain between 1 and 120 characters")
            start = self._validate_normalized_point(line.get("start"), f"{path}.start")
            end = self._validate_normalized_point(line.get("end"), f"{path}.end")
            if start == end:
                raise ValueError(f"{path} start and end must differ")
            if line.get("direction", "both") not in {"both", "a_to_b", "b_to_a"}:
                raise ValueError(f"{path}.direction must be both, a_to_b or b_to_a")
            self._validate_class_ids(line.get("classIds", []), f"{path}.classIds")

        self._validate_bool_fields(
            osd_config,
            {
                "enabled", "showLabels", "showConfidence", "showTrackId",
                "showAreas", "showLines",
            },
            "analytics.osd",
        )
        unknown_events = set(event_config) - {
            "enabled", "snapshot", "record", "preSeconds", "postSeconds", "retentionDays"
        }
        if unknown_events:
            raise ValueError(
                f"analytics.events contains unsupported fields: {sorted(unknown_events)}"
            )
        for field in ("enabled", "snapshot", "record"):
            if field in event_config and not isinstance(event_config[field], bool):
                raise ValueError(f"analytics.events.{field} must be a boolean")
        for field, minimum, maximum, default in (
            ("preSeconds", 0, 60, 3),
            ("postSeconds", 0, 300, 5),
            ("retentionDays", 1, 3650, 30),
        ):
            value = event_config.get(field, default)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"analytics.events.{field} must be between {minimum} and {maximum}"
                )
        if event_config.get("enabled") is True:
            if not area_items and not line_items:
                raise ValueError("enabled analytics events require at least one area or line")
            if (
                event_config.get("snapshot", True) is not True
                and event_config.get("record", False) is not True
            ):
                raise ValueError("enabled analytics events require snapshot or record output")
        if (
            event_config.get("record") is True
            and self.media.get("decoder", "opencv") != "rkmpp"
        ):
            raise ValueError("analytics event recording requires the RKMPP decoder")

        secondary_release_ids: set[str] = set()
        for index, raw_secondary in enumerate(secondary_items):
            path = f"analytics.secondaryModels[{index}]"
            if not isinstance(raw_secondary, dict):
                raise ValueError(f"{path} must be an object")
            secondary = cast(dict[str, object], raw_secondary)
            unknown_secondary = set(secondary) - {
                "releaseId",
                "sourceClassIds",
                "confidenceThreshold",
                "contextCount",
                "workerCount",
            }
            if unknown_secondary:
                raise ValueError(
                    f"{path} contains unsupported fields: {sorted(unknown_secondary)}"
                )
            release_id = secondary.get("releaseId")
            if not isinstance(release_id, str) or not release_id.strip():
                raise ValueError(f"{path}.releaseId is required")
            if release_id in secondary_release_ids:
                raise ValueError("analytics.secondaryModels releaseId values must be unique")
            secondary_release_ids.add(release_id)
            self._validate_class_ids(
                secondary.get("sourceClassIds", []), f"{path}.sourceClassIds"
            )
            threshold = secondary.get("confidenceThreshold", 0.25)
            if (
                not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)
                or not math.isfinite(float(threshold))
                or not 0 <= float(threshold) <= 1
            ):
                raise ValueError(f"{path}.confidenceThreshold must be between 0 and 1")
            context_count = secondary.get("contextCount", 1)
            worker_count = secondary.get("workerCount", 1)
            if (
                not isinstance(context_count, int)
                or isinstance(context_count, bool)
                or context_count < 1
            ):
                raise ValueError(f"{path}.contextCount must be a positive integer")
            if (
                not isinstance(worker_count, int)
                or isinstance(worker_count, bool)
                or worker_count < 1
            ):
                raise ValueError(f"{path}.workerCount must be a positive integer")
            if worker_count > context_count:
                raise ValueError(f"{path}.workerCount must not exceed contextCount")
            secondary["contextCount"] = context_count
            secondary["workerCount"] = worker_count


class InferenceTaskUpdate(InferenceTaskCreate):
    pass


class InferenceTaskResponse(ApiModel):
    id: str
    name: str
    status: InferenceTaskStatus
    release_id: str
    node_id: str
    group_id: str | None
    input_uri: str
    interval: int
    thresholds: dict[str, float]
    output: dict[str, Any]
    media: dict[str, Any]
    analytics: dict[str, Any]
    npu_core_mask: NpuCoreMask
    npu_core_policy: NpuCorePolicy
    context_count: int
    worker_count: int
    preview_capability: PreviewCapability
    config_revision: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class InferenceTaskListResponse(PageInfo):
    items: list[InferenceTaskResponse]


class InferencePlaybackSessionResponse(ApiModel):
    stream_url: str
    expires_at: datetime
    task_id: str
    revision: int
    gateway_id: str
    app: str
    stream_name: str
    codec: Literal["h264", "h265", "unknown"]
    reconnect_ms: int


class DeploymentCreate(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    release_id: str
    task_ids: list[str] = Field(min_length=1, max_length=1000)
    strategy: Literal["canary", "rolling", "all_at_once"] = "rolling"
    batch_size: int = Field(default=1, ge=1, le=100)

    @field_validator("task_ids")
    @classmethod
    def unique_task_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("taskIds must be unique")
        return value


class DeploymentTargetResponse(ApiModel):
    id: str
    deployment_id: str
    node_id: str
    task_id: str
    release_id: str
    previous_release_id: str | None
    sequence: int
    desired_revision: int
    state: DeploymentTargetState
    progress: int
    stage: str
    error_code: str | None
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime


class DeploymentResponse(ApiModel):
    id: str
    name: str
    status: DeploymentStatus
    release_id: str
    strategy: str
    batch_size: int
    targets: list[DeploymentTargetResponse]
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class DeploymentListResponse(PageInfo):
    items: list[DeploymentResponse]


class DeploymentEventResponse(ApiModel):
    id: int
    deployment_id: str
    target_id: str | None
    node_id: str | None
    type: str
    level: str
    message: str
    data: dict[str, Any]
    created_at: datetime


class AgentArtifactDescriptor(ApiModel):
    id: str
    filename: str
    sha256: str
    size_bytes: int
    media_type: str


class AgentReleaseDescriptor(ApiModel):
    id: str
    name: str
    version: str
    adapter: str
    artifact: AgentArtifactDescriptor
    manifest: dict[str, Any]


class AgentTaskDescriptor(ApiModel):
    id: str
    name: str
    release_id: str
    deployment_target_id: str | None
    input_uri: str
    interval: int
    thresholds: dict[str, float]
    output: dict[str, Any]
    media: dict[str, Any]
    analytics: dict[str, Any]
    npu_core_mask: NpuCoreMask
    npu_core_policy: NpuCorePolicy
    context_count: int = 1
    worker_count: int = 1


class AgentDesiredState(ApiModel):
    node_id: str
    revision: int
    config_hash: str
    releases: list[AgentReleaseDescriptor]
    tasks: list[AgentTaskDescriptor]


class DeploymentTargetReport(ApiModel):
    revision: int = Field(ge=1)
    state: DeploymentTargetState
    progress: int = Field(ge=0, le=100)
    stage: str = Field(min_length=1, max_length=120)
    message: str = Field(default="", max_length=4000)
    error_code: str | None = Field(default=None, max_length=100)
    error_message: str | None = Field(default=None, max_length=4000)


class ApiErrorBody(ApiModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class ApiErrorResponse(ApiModel):
    error: ApiErrorBody
