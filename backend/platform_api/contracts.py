from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator

from .api_models import ApiModel
from .inference_graph import GraphLayout, InferenceGraph
from .media_contracts import PreviewCapability


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


class InferenceGraphTaskCreate(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    node_id: str | None = None
    group_id: str | None = None
    input_uri: str = Field(min_length=1, max_length=2000)
    graph: InferenceGraph
    layout: GraphLayout = Field(default_factory=GraphLayout)
    npu_core_mask: NpuCoreMask = NpuCoreMask.AUTO
    npu_core_policy: NpuCorePolicy = NpuCorePolicy.SHARED

    @model_validator(mode="after")
    def validate_graph_task(self) -> InferenceGraphTaskCreate:
        if bool(self.node_id) == bool(self.group_id):
            raise ValueError("exactly one of nodeId or groupId is required")
        if (
            self.npu_core_policy == NpuCorePolicy.EXCLUSIVE
            and self.npu_core_mask == NpuCoreMask.AUTO
        ):
            raise ValueError("exclusive NPU core policy requires an explicit core mask")
        unknown_positions = set(self.layout.positions) - {node.id for node in self.graph.nodes}
        if unknown_positions:
            raise ValueError(
                f"layout positions reference unknown graph nodes: {sorted(unknown_positions)}"
            )
        return self


class InferenceTaskUpdate(InferenceGraphTaskCreate):
    base_revision_id: str = Field(min_length=1, max_length=48)


class InferenceTaskResponse(ApiModel):
    id: str
    name: str
    status: InferenceTaskStatus
    node_id: str
    group_id: str | None
    input_uri: str
    graph: InferenceGraph
    layout: GraphLayout
    graph_revision_id: str
    graph_hash: str
    npu_core_mask: NpuCoreMask
    npu_core_policy: NpuCorePolicy
    preview_capability: PreviewCapability
    config_revision: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class InferenceTaskListResponse(PageInfo):
    items: list[InferenceTaskResponse]


class InferenceGraphRevisionResponse(ApiModel):
    id: str
    task_id: str
    revision: int
    graph: InferenceGraph
    graph_hash: str
    created_at: datetime


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
    graph_revision_id: str
    graph_hash: str
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
    deployment_target_id: str | None
    input_uri: str
    graph: InferenceGraph
    graph_revision_id: str
    graph_hash: str
    runtime_bindings: dict[str, Any] = Field(default_factory=dict)
    npu_core_mask: NpuCoreMask
    npu_core_policy: NpuCorePolicy
    config_revision: int = Field(default=0, ge=0)


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
