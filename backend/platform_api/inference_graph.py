from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, cast
from urllib.parse import urlparse

from pydantic import Field, model_validator

from .api_models import ApiModel
from .media_contracts import MEDIA_IDENTIFIER_PATTERN

GRAPH_SCHEMA_VERSION = 1
GRAPH_CATALOG_VERSION = "2026.08.25"
YOLO_ADAPTER = "yolo_dfl_split_v1"
SUPPORTED_ADAPTERS = (
    YOLO_ADAPTER,
    "deeplab_logits_v1",
    "ppocr_db_det_v1",
    "ppocr_ctc_rec_v1",
)


class GraphNode(ApiModel):
    id: str = Field(min_length=1, max_length=80)
    operator: str = Field(min_length=1, max_length=120)
    config: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(ApiModel):
    source: str = Field(min_length=1, max_length=80)
    source_port: str = Field(default="frame", min_length=1, max_length=40)
    target: str = Field(min_length=1, max_length=80)
    target_port: str = Field(default="frame", min_length=1, max_length=40)


class InferenceGraph(ApiModel):
    schema_version: int = Field(default=GRAPH_SCHEMA_VERSION, ge=1)
    catalog_version: str = Field(default=GRAPH_CATALOG_VERSION, min_length=1, max_length=40)
    nodes: list[GraphNode] = Field(min_length=1, max_length=64)
    edges: list[GraphEdge] = Field(default_factory=lambda: list[GraphEdge](), max_length=128)

    @model_validator(mode="after")
    def validate_ids(self) -> InferenceGraph:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node ids must be unique")
        known_ids = set(node_ids)
        for edge in self.edges:
            if edge.source not in known_ids or edge.target not in known_ids:
                raise ValueError("graph edges must reference existing nodes")
            if edge.source == edge.target:
                raise ValueError("graph edges cannot point to the same node")
        return self


class GraphPosition(ApiModel):
    x: float
    y: float


class GraphLayout(ApiModel):
    positions: dict[str, GraphPosition] = Field(default_factory=dict)


class GraphValidationRequest(ApiModel):
    graph: InferenceGraph
    node_id: str | None = None
    task_id: str | None = None
    input_uri: str | None = Field(default=None, min_length=1, max_length=2000)


class GraphValidationIssue(ApiModel):
    code: str
    message: str
    path: str = "graph"
    severity: str = "error"
    details: dict[str, Any] = Field(default_factory=dict)


class GraphValidationResponse(ApiModel):
    valid: bool
    normalized_graph: InferenceGraph | None = None
    graph_hash: str | None = None
    issues: list[GraphValidationIssue] = Field(default_factory=lambda: list[GraphValidationIssue]())
    release_ids: list[str] = Field(default_factory=list)
    required_features: list[str] = Field(default_factory=list)
    required_adapters: list[str] = Field(default_factory=list)
    required_contexts: int = 0
    compatible_node_ids: list[str] = Field(default_factory=list)


class OperatorCatalogEntry(ApiModel):
    operator_id: str
    runtime_node: str
    category: str
    title: str
    description: str
    input_ports: list[str] = Field(default_factory=list)
    output_ports: list[str] = Field(default_factory=list)
    min_instances: int = Field(default=0, ge=0)
    max_instances: int = Field(default=1, ge=1)
    defaults: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    supported_adapters: list[str] = Field(default_factory=list)
    required_features: list[str] = Field(default_factory=list)
    configurable_fields: list[str] = Field(default_factory=list)
    read_only_fields: list[str] = Field(default_factory=list)


class OperatorCatalogResponse(ApiModel):
    schema_version: int
    catalog_version: str
    operators: list[OperatorCatalogEntry]


class GraphRuntimeProjection(ApiModel):
    primary_release_id: str
    interval: int
    thresholds: dict[str, float]
    output: dict[str, Any]
    media: dict[str, Any]
    analytics: dict[str, Any]
    context_count: int
    worker_count: int


def _entry(
    operator_id: str,
    runtime_node: str,
    category: str,
    title: str,
    description: str,
    *,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = ("frame",),
    min_instances: int = 0,
    max_instances: int = 1,
    defaults: dict[str, Any] | None = None,
    dependencies: tuple[str, ...] = (),
    adapters: tuple[str, ...] = (),
    features: tuple[str, ...] = (),
    configurable: tuple[str, ...] = (),
    read_only: tuple[str, ...] = (),
) -> OperatorCatalogEntry:
    return OperatorCatalogEntry(
        operator_id=operator_id,
        runtime_node=runtime_node,
        category=category,
        title=title,
        description=description,
        input_ports=list(inputs),
        output_ports=list(outputs),
        min_instances=min_instances,
        max_instances=max_instances,
        defaults=defaults or {},
        dependencies=list(dependencies),
        supported_adapters=list(adapters),
        required_features=list(features),
        configurable_fields=list(configurable),
        read_only_fields=list(read_only),
    )


OPERATOR_CATALOG: tuple[OperatorCatalogEntry, ...] = (
    _entry(
        "capture.opencv",
        "VideoCaptureNode",
        "capture",
        "通用解码",
        "OpenCV/FFmpeg input for files, directories and RTSP streams.",
        min_instances=1,
        defaults={"loop": True, "reconnectMs": 1000},
        configurable=("loop", "reconnectMs"),
        read_only=("runtimeNode",),
    ),
    _entry(
        "capture.rkmpp",
        "RkMppCaptureNode",
        "capture",
        "MPP 硬解码",
        "RK3588 MPP H.264/H.265 RTSP 输入并保留编码包。",
        min_instances=1,
        defaults={"loop": True, "reconnectMs": 1000},
        dependencies=("input.rtsp",),
        features=("rkmpp_decode",),
        configurable=("loop", "reconnectMs"),
        read_only=("runtimeNode",),
    ),
    _entry(
        "inference.primary",
        "InferNode",
        "inference",
        "主推理",
        "使用已发布 RKNN 模型对每帧执行主推理。",
        inputs=("frame",),
        min_instances=1,
        defaults={
            "releaseId": "",
            "interval": 1,
            "confidence": 0.4,
            "nms": 0.5,
            "contextCount": 1,
            "workerCount": 1,
        },
        adapters=SUPPORTED_ADAPTERS,
        configurable=(
            "releaseId",
            "interval",
            "confidence",
            "nms",
            "contextCount",
            "workerCount",
        ),
        read_only=("adapter", "modelType", "modelPath", "manifestPath"),
    ),
    _entry(
        "processing.bytetrack",
        "ByteTrackNode",
        "processing",
        "ByteTrack",
        "为 YOLO 检测目标生成稳定跟踪 ID。",
        inputs=("frame",),
        dependencies=("inference.primary",),
        adapters=(YOLO_ADAPTER,),
        features=("bytetrack",),
        defaults={"trackBuffer": 30},
        configurable=("trackBuffer",),
    ),
    _entry(
        "inference.secondary",
        "SecondaryInferNode",
        "inference",
        "二级推理",
        "对主推理检测框裁剪后执行已发布 YOLO 模型。",
        inputs=("frame",),
        max_instances=4,
        dependencies=("inference.primary", "processing.bytetrack"),
        adapters=(YOLO_ADAPTER,),
        features=("secondary_infer",),
        defaults={
            "releaseId": "",
            "confidence": 0.25,
            "sourceClassIds": [],
            "contextCount": 1,
            "workerCount": 1,
        },
        configurable=(
            "releaseId",
            "confidence",
            "sourceClassIds",
            "contextCount",
            "workerCount",
        ),
        read_only=("adapter", "modelType", "modelPath", "manifestPath"),
    ),
    _entry(
        "processing.analytics",
        "AnalyticsNode",
        "processing",
        "区域/越线分析",
        "基于跟踪目标计算区域进入离开和越线事件。",
        inputs=("frame",),
        dependencies=("processing.bytetrack",),
        adapters=(YOLO_ADAPTER,),
        features=("analytics_area", "analytics_line"),
        defaults={
            "areas": [],
            "lines": [],
            "osd": {
                "enabled": True,
                "showLabels": True,
                "showConfidence": True,
                "showTrackId": True,
                "showAreas": True,
                "showLines": True,
            },
        },
        configurable=("areas", "lines", "osd"),
    ),
    _entry(
        "processing.events",
        "EventOutputNode",
        "processing",
        "事件抓拍/录像",
        "按区域或越线事件输出 JPEG 抓拍和原码流事件录像。",
        inputs=("frame",),
        dependencies=("processing.analytics",),
        features=("event_snapshot", "event_record"),
        defaults={
            "enabled": True,
            "snapshot": True,
            "record": False,
            "preSeconds": 3,
            "postSeconds": 5,
            "retentionDays": 30,
        },
        configurable=(
            "enabled",
            "snapshot",
            "record",
            "preSeconds",
            "postSeconds",
            "retentionDays",
        ),
    ),
    _entry(
        "output.json",
        "JsonOutputNode",
        "output",
        "JSONL/HTTP 输出",
        "将结构化结果写入板端 JSONL 或发送到 HTTP(S) 业务接口。",
        inputs=("frame",),
        outputs=(),
        defaults={
            "type": "jsonl",
            "url": "",
            "authorizationEnv": "",
            "connectTimeoutMs": 1000,
            "requestTimeoutMs": 3000,
        },
        configurable=(
            "type",
            "url",
            "authorizationEnv",
            "connectTimeoutMs",
            "requestTimeoutMs",
        ),
        read_only=("outputPath",),
    ),
    _entry(
        "output.kafka",
        "KafkaOutputNode",
        "output",
        "Kafka 输出",
        "异步发送结构化结果到 Kafka。",
        inputs=("frame",),
        outputs=(),
        features=("kafka",),
        defaults={
            "brokers": "",
            "topic": "sei_msg",
            "key": "",
            "queueMessages": 10000,
            "messageTimeoutMs": 3000,
        },
        configurable=(
            "brokers",
            "topic",
            "key",
            "queueMessages",
            "messageTimeoutMs",
        ),
    ),
    _entry(
        "output.zlm_sei",
        "ZlmSeiOutputNode",
        "output",
        "ZLM SEI 输出",
        "向媒体网关发布带检测 SEI 的原始 RTSP 码流。",
        inputs=("frame",),
        outputs=(),
        dependencies=("capture.rkmpp",),
        features=("zlm_sei",),
        defaults={"gatewayId": "", "streamName": "", "reconnectMs": 1000},
        configurable=("gatewayId", "streamName", "reconnectMs"),
    ),
)

CATALOG_BY_ID = {entry.operator_id: entry for entry in OPERATOR_CATALOG}
CAPTURE_OPERATORS = frozenset({"capture.opencv", "capture.rkmpp"})
OUTPUT_OPERATORS = frozenset({"output.json", "output.kafka", "output.zlm_sei"})
YOLO_ONLY_OPERATORS = frozenset(
    {
        "processing.bytetrack",
        "inference.secondary",
        "processing.analytics",
        "processing.events",
    }
)
ALLOWED_PREDECESSORS: dict[str, frozenset[str]] = {
    "inference.primary": CAPTURE_OPERATORS,
    "processing.bytetrack": frozenset({"inference.primary"}),
    "inference.secondary": frozenset({"processing.bytetrack", "inference.secondary"}),
    "processing.analytics": frozenset({"processing.bytetrack", "inference.secondary"}),
    "processing.events": frozenset({"processing.analytics"}),
    "output.json": frozenset(
        {
            "inference.primary",
            "processing.bytetrack",
            "inference.secondary",
            "processing.analytics",
            "processing.events",
        }
    ),
    "output.kafka": frozenset(
        {
            "inference.primary",
            "processing.bytetrack",
            "inference.secondary",
            "processing.analytics",
            "processing.events",
        }
    ),
    "output.zlm_sei": frozenset(
        {
            "inference.primary",
            "processing.bytetrack",
            "inference.secondary",
            "processing.analytics",
            "processing.events",
        }
    ),
}


def _issue(code: str, message: str, path: str, **details: Any) -> GraphValidationIssue:
    return GraphValidationIssue(code=code, message=message, path=path, details=details)


def _deep_defaults(defaults: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(defaults)
    for key, value in supplied.items():
        default_value = result.get(key)
        if isinstance(value, dict) and isinstance(default_value, dict):
            result[key] = _deep_defaults(
                cast(dict[str, Any], default_value), cast(dict[str, Any], value)
            )
        else:
            result[key] = cast(Any, value)
    return result


def _operator_counts(graph: InferenceGraph) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for node in graph.nodes:
        counts[node.operator] += 1
    return counts


def _config_issue(
    code: str,
    message: str,
    node: GraphNode,
    field: str = "",
) -> GraphValidationIssue:
    suffix = f".{field}" if field else ""
    return _issue(code, message, f"nodes.{node.id}.config{suffix}")


def _valid_integer(value: object, minimum: int, maximum: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= maximum
    )


def _valid_probability(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


def _validate_class_ids(
    value: object,
    node: GraphNode,
    field: str,
) -> list[GraphValidationIssue]:
    if not isinstance(value, list):
        return [
            _config_issue(
                "class_ids_invalid",
                f"{field} must contain at most 256 unique non-negative integers",
                node,
                field,
            )
        ]
    items = cast(list[object], value)
    if (
        len(items) > 256
        or any(not _valid_integer(item, 0, 2**31 - 1) for item in items)
        or len(items) != len(set(items))
    ):
        return [
            _config_issue(
                "class_ids_invalid",
                f"{field} must contain at most 256 unique non-negative integers",
                node,
                field,
            )
        ]
    return []


def _validate_point(
    value: object,
    node: GraphNode,
    field: str,
) -> list[GraphValidationIssue]:
    if not isinstance(value, dict):
        return [
            _config_issue(
                "analytics_point_invalid",
                f"{field} must contain only normalized x and y coordinates",
                node,
                field,
            )
        ]
    point = cast(dict[str, object], value)
    if set(point) != {"x", "y"}:
        return [
            _config_issue(
                "analytics_point_invalid",
                f"{field} must contain only normalized x and y coordinates",
                node,
                field,
            )
        ]
    if not _valid_probability(point.get("x")) or not _valid_probability(point.get("y")):
        return [
            _config_issue(
                "analytics_point_invalid",
                f"{field} coordinates must be finite numbers between 0 and 1",
                node,
                field,
            )
        ]
    return []


def _validate_analytics_config(node: GraphNode) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    areas_value: object = node.config.get("areas", [])
    lines_value: object = node.config.get("lines", [])
    osd_value: object = node.config.get("osd", {})
    if not isinstance(areas_value, list) or len(cast(list[object], areas_value)) > 32:
        issues.append(
            _config_issue(
                "analytics_areas_invalid",
                "areas must be an array with at most 32 entries",
                node,
                "areas",
            )
        )
        areas: list[object] = []
    else:
        areas = cast(list[object], areas_value)
    if not isinstance(lines_value, list) or len(cast(list[object], lines_value)) > 32:
        issues.append(
            _config_issue(
                "analytics_lines_invalid",
                "lines must be an array with at most 32 entries",
                node,
                "lines",
            )
        )
        lines: list[object] = []
    else:
        lines = cast(list[object], lines_value)
    if not isinstance(osd_value, dict):
        issues.append(
            _config_issue("analytics_osd_invalid", "osd must be an object", node, "osd")
        )
        osd: dict[str, object] = {}
    else:
        osd = cast(dict[str, object], osd_value)

    rule_ids: set[str] = set()
    for index, raw_area in enumerate(areas):
        field = f"areas[{index}]"
        if not isinstance(raw_area, dict):
            issues.append(
                _config_issue(
                    "analytics_area_invalid", "area rules must be objects", node, field
                )
            )
            continue
        area = cast(dict[str, object], raw_area)
        unknown = set(area) - {
            "id",
            "name",
            "polygon",
            "classIds",
            "minCount",
            "holdFrames",
        }
        if unknown:
            issues.append(
                _config_issue(
                    "analytics_area_field_unknown",
                    f"area rule contains unsupported fields: {sorted(unknown)}",
                    node,
                    field,
                )
            )
        rule_id = area.get("id")
        if not isinstance(rule_id, str) or not 1 <= len(rule_id.strip()) <= 80:
            issues.append(
                _config_issue(
                    "analytics_rule_id_invalid",
                    "area id must contain between 1 and 80 characters",
                    node,
                    f"{field}.id",
                )
            )
        elif rule_id in rule_ids:
            issues.append(
                _config_issue(
                    "analytics_rule_id_duplicate",
                    "area and line ids must be unique",
                    node,
                    f"{field}.id",
                )
            )
        else:
            rule_ids.add(rule_id)
        name = area.get("name", rule_id)
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
            issues.append(
                _config_issue(
                    "analytics_rule_name_invalid",
                    "area name must contain between 1 and 120 characters",
                    node,
                    f"{field}.name",
                )
            )
        polygon = area.get("polygon")
        if not isinstance(polygon, list) or not 3 <= len(cast(list[object], polygon)) <= 32:
            issues.append(
                _config_issue(
                    "analytics_polygon_invalid",
                    "area polygon must contain between 3 and 32 points",
                    node,
                    f"{field}.polygon",
                )
            )
        else:
            for point_index, point in enumerate(cast(list[object], polygon)):
                issues.extend(_validate_point(point, node, f"{field}.polygon[{point_index}]"))
        issues.extend(_validate_class_ids(area.get("classIds", []), node, f"{field}.classIds"))
        for item, maximum in (("minCount", 100000), ("holdFrames", 10000)):
            if not _valid_integer(area.get(item, 1), 1, maximum):
                issues.append(
                    _config_issue(
                        "analytics_rule_range_invalid",
                        f"{item} must be between 1 and {maximum}",
                        node,
                        f"{field}.{item}",
                    )
                )

    for index, raw_line in enumerate(lines):
        field = f"lines[{index}]"
        if not isinstance(raw_line, dict):
            issues.append(
                _config_issue(
                    "analytics_line_invalid", "line rules must be objects", node, field
                )
            )
            continue
        line = cast(dict[str, object], raw_line)
        unknown = set(line) - {"id", "name", "start", "end", "direction", "classIds"}
        if unknown:
            issues.append(
                _config_issue(
                    "analytics_line_field_unknown",
                    f"line rule contains unsupported fields: {sorted(unknown)}",
                    node,
                    field,
                )
            )
        rule_id = line.get("id")
        if not isinstance(rule_id, str) or not 1 <= len(rule_id.strip()) <= 80:
            issues.append(
                _config_issue(
                    "analytics_rule_id_invalid",
                    "line id must contain between 1 and 80 characters",
                    node,
                    f"{field}.id",
                )
            )
        elif rule_id in rule_ids:
            issues.append(
                _config_issue(
                    "analytics_rule_id_duplicate",
                    "area and line ids must be unique",
                    node,
                    f"{field}.id",
                )
            )
        else:
            rule_ids.add(rule_id)
        name = line.get("name", rule_id)
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
            issues.append(
                _config_issue(
                    "analytics_rule_name_invalid",
                    "line name must contain between 1 and 120 characters",
                    node,
                    f"{field}.name",
                )
            )
        start_issues = _validate_point(line.get("start"), node, f"{field}.start")
        end_issues = _validate_point(line.get("end"), node, f"{field}.end")
        issues.extend(start_issues)
        issues.extend(end_issues)
        if not start_issues and not end_issues and line.get("start") == line.get("end"):
            issues.append(
                _config_issue(
                    "analytics_line_zero_length",
                    "line start and end must differ",
                    node,
                    field,
                )
            )
        if line.get("direction", "both") not in {"both", "a_to_b", "b_to_a"}:
            issues.append(
                _config_issue(
                    "analytics_line_direction_invalid",
                    "direction must be both, a_to_b or b_to_a",
                    node,
                    f"{field}.direction",
                )
            )
        issues.extend(_validate_class_ids(line.get("classIds", []), node, f"{field}.classIds"))

    osd_config = osd
    allowed_osd = {
        "enabled",
        "showLabels",
        "showConfidence",
        "showTrackId",
        "showAreas",
        "showLines",
    }
    if set(osd_config) - allowed_osd or any(
        not isinstance(value, bool) for value in osd_config.values()
    ):
        issues.append(
            _config_issue(
                "analytics_osd_invalid",
                "osd supports only the documented boolean fields",
                node,
                "osd",
            )
        )
    return issues


def _validate_config(node: GraphNode, entry: OperatorCatalogEntry) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    allowed = set(entry.configurable_fields)
    for field_name in node.config:
        if field_name not in allowed:
            issues.append(
                _issue(
                    "operator_config_unknown",
                    f"{field_name} is not configurable for {entry.title}",
                    f"nodes.{node.id}.config.{field_name}",
                )
            )
    if node.operator in CAPTURE_OPERATORS:
        if not isinstance(node.config.get("loop", True), bool):
            issues.append(
                _config_issue(
                    "capture_loop_invalid", "loop must be a boolean", node, "loop"
                )
            )
        reconnect_minimum = 100 if node.operator == "capture.rkmpp" else 0
        if not _valid_integer(
            node.config.get("reconnectMs", 1000), reconnect_minimum, 60000
        ):
            issues.append(
                _config_issue(
                    "capture_reconnect_invalid",
                    f"reconnectMs must be between {reconnect_minimum} and 60000",
                    node,
                    "reconnectMs",
                )
            )
    if node.operator in {"inference.primary", "inference.secondary"}:
        release_id = node.config.get("releaseId")
        if not isinstance(release_id, str) or not release_id.strip():
            issues.append(
                _issue(
                    "release_required",
                    "Inference operators must reference a published release",
                    f"nodes.{node.id}.config.releaseId",
                )
            )
        confidence = node.config.get("confidence", 0.4)
        if not _valid_probability(confidence):
            issues.append(
                _config_issue(
                    "confidence_invalid",
                    "confidence must be a finite number between 0 and 1",
                    node,
                    "confidence",
                )
            )
        context_count = node.config.get("contextCount", 1)
        worker_count = node.config.get("workerCount", 1)
        if (
            not isinstance(context_count, int)
            or isinstance(context_count, bool)
            or context_count < 1
        ):
            issues.append(
                _issue(
                    "context_count_invalid",
                    "contextCount must be a positive integer",
                    f"nodes.{node.id}.config.contextCount",
                )
            )
        if (
            not isinstance(worker_count, int)
            or isinstance(worker_count, bool)
            or worker_count < 1
            or (
                isinstance(context_count, int)
                and not isinstance(context_count, bool)
                and worker_count > context_count
            )
        ):
            issues.append(
                _issue(
                    "worker_count_invalid",
                    "workerCount must be between 1 and contextCount",
                    f"nodes.{node.id}.config.workerCount",
                )
            )
    if node.operator == "inference.primary":
        if not _valid_integer(node.config.get("interval", 1), 1, 10000):
            issues.append(
                _config_issue(
                    "interval_invalid",
                    "interval must be an integer between 1 and 10000",
                    node,
                    "interval",
                )
            )
        if not _valid_probability(node.config.get("nms", 0.5)):
            issues.append(
                _config_issue(
                    "nms_invalid",
                    "nms must be a finite number between 0 and 1",
                    node,
                    "nms",
                )
            )
    elif node.operator == "inference.secondary":
        issues.extend(
            _validate_class_ids(
                node.config.get("sourceClassIds", []), node, "sourceClassIds"
            )
        )
    elif node.operator == "processing.bytetrack":
        if not _valid_integer(node.config.get("trackBuffer", 30), 1, 10000):
            issues.append(
                _config_issue(
                    "track_buffer_invalid",
                    "trackBuffer must be an integer between 1 and 10000",
                    node,
                    "trackBuffer",
                )
            )
    elif node.operator == "processing.analytics":
        issues.extend(_validate_analytics_config(node))
    elif node.operator == "processing.events":
        for field_name in ("enabled", "snapshot", "record"):
            if not isinstance(node.config.get(field_name), bool):
                issues.append(
                    _config_issue(
                        "event_boolean_invalid",
                        f"{field_name} must be a boolean",
                        node,
                        field_name,
                    )
                )
        for field_name, minimum, maximum in (
            ("preSeconds", 0, 60),
            ("postSeconds", 0, 300),
            ("retentionDays", 1, 3650),
        ):
            if not _valid_integer(node.config.get(field_name), minimum, maximum):
                issues.append(
                    _config_issue(
                        "event_range_invalid",
                        f"{field_name} must be between {minimum} and {maximum}",
                        node,
                        field_name,
                    )
                )
    if node.operator == "output.json":
        output_type = node.config.get("type", "jsonl")
        if output_type not in {"jsonl", "http"}:
            issues.append(
                _issue(
                    "json_output_type_invalid",
                    "JSON output type must be jsonl or http",
                    f"nodes.{node.id}.config.type",
                )
            )
        if output_type == "http":
            url = node.config.get("url")
            parsed = urlparse(url) if isinstance(url, str) else None
            if (
                parsed is None
                or parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
            ):
                issues.append(
                    _issue(
                        "http_output_url_invalid",
                        "HTTP output requires an HTTP(S) URL without embedded credentials",
                        f"nodes.{node.id}.config.url",
                    )
                )
            connect_timeout = node.config.get("connectTimeoutMs", 1000)
            request_timeout = node.config.get("requestTimeoutMs", 3000)
            if (
                not isinstance(connect_timeout, int)
                or isinstance(connect_timeout, bool)
                or not isinstance(request_timeout, int)
                or isinstance(request_timeout, bool)
                or not 100 <= connect_timeout <= request_timeout <= 60000
            ):
                issues.append(
                    _issue(
                        "http_output_timeout_invalid",
                        "HTTP output timeouts must satisfy 100 <= connect <= request <= 60000",
                        f"nodes.{node.id}.config",
                    )
                )
            authorization_env = node.config.get("authorizationEnv", "")
            if not isinstance(authorization_env, str) or (
                authorization_env
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", authorization_env) is None
            ):
                issues.append(
                    _issue(
                        "http_output_authorization_env_invalid",
                        "authorizationEnv must be a valid environment variable name",
                        f"nodes.{node.id}.config.authorizationEnv",
                    )
                )
    elif node.operator == "output.kafka":
        for field_name in ("brokers", "topic", "key"):
            value = node.config.get(field_name)
            if not isinstance(value, str) or (
                field_name in {"brokers", "topic"} and not value.strip()
            ):
                issues.append(
                    _config_issue(
                        "kafka_destination_invalid",
                        f"{field_name} must be a non-empty string"
                        if field_name != "key"
                        else "key must be a string",
                        node,
                        field_name,
                    )
                )
        if not _valid_integer(node.config.get("queueMessages"), 1, 1000000):
            issues.append(
                _config_issue(
                    "kafka_queue_invalid",
                    "queueMessages must be between 1 and 1000000",
                    node,
                    "queueMessages",
                )
            )
        if not _valid_integer(node.config.get("messageTimeoutMs"), 100, 60000):
            issues.append(
                _config_issue(
                    "kafka_timeout_invalid",
                    "messageTimeoutMs must be between 100 and 60000",
                    node,
                    "messageTimeoutMs",
                )
            )
    elif node.operator == "output.zlm_sei":
        gateway_id = node.config.get("gatewayId")
        if not isinstance(gateway_id, str) or not 1 <= len(gateway_id.strip()) <= 48:
            issues.append(
                _config_issue(
                    "zlm_gateway_invalid",
                    "gatewayId must contain between 1 and 48 characters",
                    node,
                    "gatewayId",
                )
            )
        stream_name = node.config.get("streamName")
        if (
            not isinstance(stream_name, str)
            or MEDIA_IDENTIFIER_PATTERN.fullmatch(stream_name) is None
        ):
            issues.append(
                _config_issue(
                    "zlm_stream_invalid",
                    "streamName must be a valid media identifier",
                    node,
                    "streamName",
                )
            )
        if not _valid_integer(node.config.get("reconnectMs"), 1000, 4000):
            issues.append(
                _config_issue(
                    "zlm_reconnect_invalid",
                    "reconnectMs must be between 1000 and 4000",
                    node,
                    "reconnectMs",
                )
            )
    return issues


def graph_issues(
    graph: InferenceGraph,
    *,
    release_adapters: dict[str, str] | None = None,
) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    if graph.schema_version != GRAPH_SCHEMA_VERSION:
        issues.append(
            _issue(
                "graph_schema_unsupported",
                f"Only graph schema version {GRAPH_SCHEMA_VERSION} is supported",
                "graph.schemaVersion",
                expected=GRAPH_SCHEMA_VERSION,
            )
        )
    if graph.catalog_version != GRAPH_CATALOG_VERSION:
        issues.append(
            _issue(
                "graph_catalog_unsupported",
                f"Catalog version {graph.catalog_version} must be explicitly "
                f"upgraded to {GRAPH_CATALOG_VERSION}",
                "graph.catalogVersion",
                expected=GRAPH_CATALOG_VERSION,
            )
        )

    by_id = {node.id: node for node in graph.nodes}
    counts = _operator_counts(graph)
    for node in graph.nodes:
        entry = CATALOG_BY_ID.get(node.operator)
        if entry is None:
            issues.append(
                _issue(
                    "operator_unsupported",
                    f"Unknown operator {node.operator}",
                    f"nodes.{node.id}.operator",
                )
            )
            continue
        if counts[node.operator] > entry.max_instances:
            issues.append(
                _issue(
                    "operator_limit_exceeded",
                    f"{entry.title} allows at most {entry.max_instances} instance(s)",
                    f"nodes.{node.id}",
                )
            )
        issues.extend(_validate_config(node, entry))

    captures = [node for node in graph.nodes if node.operator in CAPTURE_OPERATORS]
    primaries = [node for node in graph.nodes if node.operator == "inference.primary"]
    outputs = [node for node in graph.nodes if node.operator in OUTPUT_OPERATORS]
    if len(captures) != 1:
        issues.append(
            _issue(
                "graph_capture_count",
                "A graph must contain exactly one capture operator",
                "nodes",
            )
        )
    if len(primaries) != 1:
        issues.append(
            _issue(
                "graph_primary_count",
                "A graph must contain exactly one primary inference operator",
                "nodes",
            )
        )
    if not outputs:
        issues.append(
            _issue(
                "graph_output_required",
                "A graph must contain at least one output operator",
                "nodes",
            )
        )

    incoming: dict[str, list[GraphEdge]] = defaultdict(list)
    adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
    seen_edges: set[tuple[str, str, str, str]] = set()
    for edge in graph.edges:
        edge_key = (edge.source, edge.source_port, edge.target, edge.target_port)
        if edge_key in seen_edges:
            issues.append(
                _issue(
                    "graph_edge_duplicate",
                    "Duplicate graph edge",
                    f"edges.{edge.source}.{edge.target}",
                )
            )
        seen_edges.add(edge_key)
        incoming[edge.target].append(edge)
        adjacency[edge.source].append(edge)
        source_entry = CATALOG_BY_ID.get(by_id[edge.source].operator)
        target_entry = CATALOG_BY_ID.get(by_id[edge.target].operator)
        if source_entry and edge.source_port not in source_entry.output_ports:
            issues.append(
                _issue(
                    "graph_port_invalid",
                    f"Unknown source port {edge.source_port}",
                    f"edges.{edge.source}.{edge.target}.sourcePort",
                )
            )
        if target_entry and edge.target_port not in target_entry.input_ports:
            issues.append(
                _issue(
                    "graph_port_invalid",
                    f"Unknown target port {edge.target_port}",
                    f"edges.{edge.source}.{edge.target}.targetPort",
                )
            )
        if len(incoming[edge.target]) > 1:
            issues.append(
                _issue(
                    "graph_multiple_inputs",
                    "A runtime node may have only one upstream edge",
                    f"nodes.{edge.target}",
                )
            )

    roots = [node for node in graph.nodes if not incoming[node.id]]
    if len(roots) != 1 or roots[0].operator not in CAPTURE_OPERATORS:
        issues.append(
            _issue(
                "graph_root_invalid",
                "The capture operator must be the graph's only root",
                "edges",
            )
        )
    queue = deque(node.id for node in roots)
    visited: set[str] = set()
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        queue.extend(edge.target for edge in adjacency[current])
    if len(visited) != len(graph.nodes):
        issues.append(
            _issue(
                "graph_cycle_or_disconnected",
                "Graph contains a cycle or disconnected nodes",
                "edges",
            )
        )

    for node in graph.nodes:
        entry = CATALOG_BY_ID.get(node.operator)
        if entry is None:
            continue
        if node.operator in CAPTURE_OPERATORS:
            if incoming[node.id]:
                issues.append(
                    _issue(
                        "capture_must_be_root",
                        "Capture operators cannot have an upstream edge",
                        f"nodes.{node.id}",
                    )
                )
        else:
            predecessors = incoming[node.id]
            if len(predecessors) != 1:
                issues.append(
                    _issue(
                        "operator_input_required",
                        f"{entry.title} requires exactly one upstream edge",
                        f"nodes.{node.id}",
                    )
                )
            elif node.operator in ALLOWED_PREDECESSORS:
                predecessor = by_id[predecessors[0].source].operator
                if predecessor not in ALLOWED_PREDECESSORS[node.operator]:
                    predecessor_title = (
                        CATALOG_BY_ID[predecessor].title
                        if predecessor in CATALOG_BY_ID
                        else predecessor
                    )
                    issues.append(
                        _issue(
                            "operator_order_invalid",
                            f"{entry.title} cannot follow {predecessor_title}",
                            f"nodes.{node.id}",
                            predecessor=predecessor,
                        )
                    )
        if node.operator in OUTPUT_OPERATORS and adjacency[node.id]:
            issues.append(
                _issue(
                    "output_must_be_terminal",
                    "Output operators cannot have downstream edges",
                    f"nodes.{node.id}",
                )
            )

    for operator_id, entry in CATALOG_BY_ID.items():
        if (
            entry.min_instances
            and counts[operator_id] < entry.min_instances
            and operator_id not in CAPTURE_OPERATORS
        ):
            issues.append(
                _issue(
                    "operator_required",
                    f"{entry.title} is required",
                    "nodes",
                )
            )

    secondary_children: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        if by_id[edge.target].operator == "inference.secondary":
            secondary_children[edge.source] += 1
    for source_id, child_count in secondary_children.items():
        if child_count > 1:
            issues.append(
                _issue(
                    "secondary_chain_branching",
                    "Secondary inference operators must form one linear chain",
                    f"nodes.{source_id}",
                )
            )

    rkmpp = len(captures) == 1 and captures[0].operator == "capture.rkmpp"
    if any(node.operator == "output.zlm_sei" for node in outputs) and not rkmpp:
        issues.append(
            _issue(
                "zlm_requires_rkmpp",
                "ZLM SEI output requires MPP capture",
                "nodes",
            )
        )
    if (
        any(
            node.operator == "processing.events" and node.config.get("record") is True
            for node in graph.nodes
        )
        and not rkmpp
    ):
        issues.append(
            _issue(
                "event_record_requires_rkmpp",
                "Event recording requires MPP capture",
                "nodes",
            )
        )
    analytics_nodes = [
        node for node in graph.nodes if node.operator == "processing.analytics"
    ]
    event_nodes = [node for node in graph.nodes if node.operator == "processing.events"]
    if event_nodes and event_nodes[0].config.get("enabled") is True:
        analytics_config = analytics_nodes[0].config if analytics_nodes else {}
        if not analytics_config.get("areas") and not analytics_config.get("lines"):
            issues.append(
                _issue(
                    "event_rule_required",
                    "Enabled event output requires at least one area or line rule",
                    f"nodes.{event_nodes[0].id}.config.enabled",
                )
            )
        if (
            event_nodes[0].config.get("snapshot") is not True
            and event_nodes[0].config.get("record") is not True
        ):
            issues.append(
                _issue(
                    "event_destination_required",
                    "Enabled event output requires snapshot or record output",
                    f"nodes.{event_nodes[0].id}.config",
                )
            )

    if release_adapters is not None:
        primary_adapter: str | None = None
        for node in graph.nodes:
            if node.operator not in {"inference.primary", "inference.secondary"}:
                continue
            release_id = node.config.get("releaseId")
            adapter = release_adapters.get(release_id) if isinstance(release_id, str) else None
            if adapter is None:
                issues.append(
                    _issue(
                        "release_unavailable",
                        "The selected model release is not published or does not exist",
                        f"nodes.{node.id}.config.releaseId",
                    )
                )
                continue
            if adapter not in SUPPORTED_ADAPTERS:
                issues.append(
                    _issue(
                        "release_adapter_unsupported",
                        f"Adapter {adapter} is not supported by graph schema "
                        f"{GRAPH_SCHEMA_VERSION}",
                        f"nodes.{node.id}.config.releaseId",
                    )
                )
            if node.operator == "inference.primary":
                primary_adapter = adapter
            elif adapter != YOLO_ADAPTER:
                issues.append(
                    _issue(
                        "secondary_adapter_mismatch",
                        "Secondary inference requires a YOLO DFL split release",
                        f"nodes.{node.id}.config.releaseId",
                    )
                )
        if (
            primary_adapter is not None
            and primary_adapter != YOLO_ADAPTER
            and any(node.operator in YOLO_ONLY_OPERATORS for node in graph.nodes)
        ):
            issues.append(
                _issue(
                    "operator_adapter_mismatch",
                    "Tracking, analytics, events and secondary inference require "
                    "a YOLO primary release",
                    "nodes",
                )
            )
    return issues


def normalize_graph(graph: InferenceGraph) -> InferenceGraph:
    normalized = graph.model_copy(deep=True)
    for node in normalized.nodes:
        entry = CATALOG_BY_ID.get(node.operator)
        if entry is not None:
            node.config = _deep_defaults(entry.defaults, node.config)
    normalized.nodes.sort(key=lambda node: node.id)
    normalized.edges.sort(
        key=lambda edge: (
            edge.source,
            edge.target,
            edge.source_port,
            edge.target_port,
        )
    )
    return normalized


def _semantic_payload(graph: InferenceGraph) -> dict[str, Any]:
    normalized = normalize_graph(graph)
    by_id = {node.id: node for node in normalized.nodes}
    outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
    incoming_count: dict[str, int] = {node.id: 0 for node in normalized.nodes}
    for edge in normalized.edges:
        outgoing[edge.source].append(edge)
        incoming_count[edge.target] += 1

    signatures: dict[str, str] = {}

    def signature(node_id: str, active: set[str]) -> str:
        if node_id in signatures:
            return signatures[node_id]
        if node_id in active:
            return "cycle"
        node = by_id[node_id]
        child_values = sorted(
            (
                edge.source_port,
                edge.target_port,
                signature(edge.target, active | {node_id}),
            )
            for edge in outgoing[node_id]
        )
        value = json.dumps(
            {
                "operator": node.operator,
                "config": node.config,
                "children": child_values,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        signatures[node_id] = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return signatures[node_id]

    roots = sorted(
        (node.id for node in normalized.nodes if incoming_count[node.id] == 0),
        key=lambda node_id: signature(node_id, set()),
    )
    ordered_ids: list[str] = []
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        visited.add(node_id)
        ordered_ids.append(node_id)
        children = sorted(
            outgoing[node_id],
            key=lambda edge: (
                edge.source_port,
                edge.target_port,
                signature(edge.target, set()),
            ),
        )
        for edge in children:
            visit(edge.target)

    for root_id in roots:
        visit(root_id)
    for node_id in sorted(set(by_id) - visited, key=lambda value: signature(value, set())):
        visit(node_id)

    canonical_ids = {original_id: f"node-{index}" for index, original_id in enumerate(ordered_ids)}
    nodes = [
        {
            "id": canonical_ids[node_id],
            "operator": by_id[node_id].operator,
            "config": by_id[node_id].config,
        }
        for node_id in ordered_ids
    ]
    edges = sorted(
        (
            {
                "source": canonical_ids[edge.source],
                "sourcePort": edge.source_port,
                "target": canonical_ids[edge.target],
                "targetPort": edge.target_port,
            }
            for edge in normalized.edges
        ),
        key=lambda edge: (
            edge["source"],
            edge["target"],
            edge["sourcePort"],
            edge["targetPort"],
        ),
    )
    return {
        "schemaVersion": normalized.schema_version,
        "catalogVersion": normalized.catalog_version,
        "nodes": nodes,
        "edges": edges,
    }


def graph_hash(graph: InferenceGraph) -> str:
    payload = _semantic_payload(graph)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def graph_release_ids(graph: InferenceGraph) -> list[str]:
    return sorted(
        {
            release_id
            for node in graph.nodes
            if node.operator in {"inference.primary", "inference.secondary"}
            and isinstance((release_id := node.config.get("releaseId")), str)
            and release_id
        }
    )


def project_graph(graph: InferenceGraph) -> GraphRuntimeProjection:
    normalized = normalize_graph(graph)
    by_operator: dict[str, list[GraphNode]] = defaultdict(list)
    for node in normalized.nodes:
        by_operator[node.operator].append(node)

    primary = by_operator["inference.primary"][0]
    capture = next(node for operator in CAPTURE_OPERATORS for node in by_operator.get(operator, []))
    json_nodes = by_operator.get("output.json", [])
    output = dict(json_nodes[0].config) if json_nodes else {"type": "jsonl"}
    media: dict[str, Any] = {
        "decoder": "rkmpp" if capture.operator == "capture.rkmpp" else "opencv"
    }
    tracking_nodes = by_operator.get("processing.bytetrack", [])
    media["tracking"] = {
        "enabled": bool(tracking_nodes),
        **(dict(tracking_nodes[0].config) if tracking_nodes else {}),
    }
    kafka_nodes = by_operator.get("output.kafka", [])
    media["kafka"] = {
        "enabled": bool(kafka_nodes),
        **(dict(kafka_nodes[0].config) if kafka_nodes else {}),
    }
    zlm_nodes = by_operator.get("output.zlm_sei", [])
    media["zlmSei"] = {
        "enabled": bool(zlm_nodes),
        **(dict(zlm_nodes[0].config) if zlm_nodes else {}),
    }

    analytics_nodes = by_operator.get("processing.analytics", [])
    analytics: dict[str, Any] = dict(analytics_nodes[0].config) if analytics_nodes else {}
    event_nodes = by_operator.get("processing.events", [])
    if event_nodes:
        analytics["events"] = dict(event_nodes[0].config)
    secondary_models: list[dict[str, Any]] = []
    for node in by_operator.get("inference.secondary", []):
        config = dict(node.config)
        config["confidenceThreshold"] = config.pop("confidence")
        secondary_models.append(config)
    if secondary_models:
        analytics["secondaryModels"] = secondary_models

    return GraphRuntimeProjection(
        primary_release_id=str(primary.config["releaseId"]),
        interval=int(primary.config["interval"]),
        thresholds={
            "confidence": float(primary.config["confidence"]),
            "nms": float(primary.config["nms"]),
        },
        output=output,
        media=media,
        analytics=analytics,
        context_count=int(primary.config["contextCount"]),
        worker_count=int(primary.config["workerCount"]),
    )


def catalog_response() -> OperatorCatalogResponse:
    return OperatorCatalogResponse(
        schema_version=GRAPH_SCHEMA_VERSION,
        catalog_version=GRAPH_CATALOG_VERSION,
        operators=[entry.model_copy(deep=True) for entry in OPERATOR_CATALOG],
    )


def graph_validation_response(
    graph: InferenceGraph,
    *,
    release_adapters: dict[str, str] | None = None,
    compatible_node_ids: list[str] | None = None,
) -> GraphValidationResponse:
    normalized = normalize_graph(graph)
    issues = graph_issues(normalized, release_adapters=release_adapters)
    release_ids = graph_release_ids(normalized)
    required_adapters = sorted(
        {
            release_adapters[release_id]
            for release_id in release_ids
            if release_adapters is not None and release_id in release_adapters
        }
    )
    required_features: set[str] = set()
    for node in normalized.nodes:
        if node.operator == "capture.rkmpp":
            required_features.add("rkmpp_decode")
        elif node.operator == "processing.bytetrack":
            required_features.add("bytetrack")
        elif node.operator == "inference.secondary":
            required_features.add("secondary_infer")
        elif node.operator == "processing.analytics":
            if node.config.get("areas"):
                required_features.add("analytics_area")
            if node.config.get("lines"):
                required_features.add("analytics_line")
        elif node.operator == "processing.events" and node.config.get("enabled"):
            if node.config.get("snapshot"):
                required_features.add("event_snapshot")
            if node.config.get("record"):
                required_features.add("event_record")
        elif node.operator == "output.kafka":
            required_features.add("kafka")
        elif node.operator == "output.zlm_sei":
            required_features.add("zlm_sei")
    required_contexts = sum(
        int(node.config.get("contextCount", 1))
        for node in normalized.nodes
        if node.operator in {"inference.primary", "inference.secondary"}
        and isinstance(node.config.get("contextCount", 1), int)
        and not isinstance(node.config.get("contextCount", 1), bool)
    )
    valid = not any(issue.severity == "error" for issue in issues)
    return GraphValidationResponse(
        valid=valid,
        normalized_graph=normalized if valid else None,
        graph_hash=graph_hash(normalized) if valid else None,
        issues=issues,
        release_ids=release_ids,
        required_features=sorted(required_features),
        required_adapters=required_adapters,
        required_contexts=required_contexts,
        compatible_node_ids=sorted(compatible_node_ids or []),
    )
