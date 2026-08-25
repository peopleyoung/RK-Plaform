from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from workers.common.rk3588_devices import missing_rk3588_inference_paths

ADAPTERS: dict[str, tuple[str, str, str]] = {
    "yolo_dfl_split_v1": (
        "rknn_yolo",
        "YOLO_DFL_SPLIT",
        "rknn_yolo_dfl_split_heads_v1",
    ),
    "deeplab_logits_v1": (
        "rknn_structured",
        "DEEPLAB_LOGITS",
        "semantic_logits_nchw_v1",
    ),
    "ppocr_db_det_v1": (
        "rknn_structured",
        "PPOCR_DB",
        "ppocr_db_probability_map_v1",
    ),
    "ppocr_ctc_rec_v1": (
        "rknn_structured",
        "PPOCR_CTC",
        "ppocr_ctc_logits_v1",
    ),
}

NPU_CORE_BITS: dict[str, int] = {
    "auto": 0b111,
    "core0": 0b001,
    "core1": 0b010,
    "core2": 0b100,
    "core0_1": 0b011,
    "core0_1_2": 0b111,
}
NPU_CORE_POLICIES = {"shared", "exclusive"}


class RuntimeAdapterError(RuntimeError):
    pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _state_root() -> Path:
    return Path(os.environ.get("RKNODE_RUNTIME_STATE_DIR", "/data/runtime"))


def _output_root() -> Path:
    return Path(os.environ.get("RKNODE_INFERENCE_OUTPUT_DIR", "/data/output"))


def _pipeline_binary() -> Path:
    return Path(os.environ.get("RKNODE_PIPELINE_BINARY", "/usr/local/bin/rknn_pipeline"))


def _probe_binary() -> Path:
    return Path(
        os.environ.get("RKNODE_INSTANCE_PROBE_BINARY", "/usr/local/bin/rknn_instance_probe")
    )


def _protocol_probe_binary() -> Path:
    return Path(
        os.environ.get("RKNODE_PROTOCOL_PROBE_BINARY", "/usr/local/bin/rknn_protocol_probe")
    )


def _safe_name(value: str, fallback: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in value)
    cleaned = cleaned.strip("-")
    return (cleaned or fallback)[:80]


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeAdapterError(f"Cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeAdapterError(f"Expected JSON object in {path}")
    return cast(dict[str, Any], value)


def _load_release_configs() -> list[dict[str, Any]]:
    raw = os.environ.get("RKNODE_RELEASE_CONFIGS", "")
    if not raw:
        release_id = os.environ.get("RKNODE_RELEASE_ID", "")
        model_path = os.environ.get("RKNODE_MODEL_PATH", "")
        manifest_path = os.environ.get("RKNODE_MANIFEST_PATH", "")
        adapter = os.environ.get("RKNODE_ADAPTER", "")
        tasks_raw = os.environ.get("RKNODE_TASK_CONFIGS", "[]")
        if not release_id and not model_path and not manifest_path:
            return []
        try:
            tasks_value: object = json.loads(tasks_raw)
        except ValueError as error:
            raise RuntimeAdapterError(f"RKNODE_TASK_CONFIGS is invalid JSON: {error}") from error
        raw = json.dumps(
            [
                {
                    "releaseId": release_id,
                    "adapter": adapter,
                    "modelPath": model_path,
                    "manifestPath": manifest_path,
                    "tasks": tasks_value,
                }
            ]
        )
    try:
        value: object = json.loads(raw)
    except ValueError as error:
        raise RuntimeAdapterError(f"RKNODE_RELEASE_CONFIGS is invalid JSON: {error}") from error
    if not isinstance(value, list):
        raise RuntimeAdapterError("RKNODE_RELEASE_CONFIGS must be a JSON array")
    configs: list[dict[str, Any]] = []
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            raise RuntimeAdapterError("Every release config must be a JSON object")
        configs.append(cast(dict[str, Any], item))
    return configs


def _manifest_for(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(str(config.get("manifestPath", "")))
    if not manifest_path.is_file():
        raise RuntimeAdapterError(f"Manifest does not exist: {manifest_path}")
    return manifest_path, _load_object(manifest_path)


def _tasks_for(config: dict[str, Any]) -> list[dict[str, Any]]:
    value = config.get("tasks", [])
    if not isinstance(value, list):
        raise RuntimeAdapterError("Release tasks must be a JSON array")
    tasks: list[dict[str, Any]] = []
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            raise RuntimeAdapterError("Every task descriptor must be a JSON object")
        tasks.append(cast(dict[str, Any], item))
    return tasks


def _task_output(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("output", {"type": "jsonl"})
    if not isinstance(value, dict):
        raise RuntimeAdapterError(f"Task {task.get('id', '<missing>')} output must be an object")
    output = cast(dict[str, Any], value)
    output_type = str(output.get("type", "jsonl"))
    if output_type == "jsonl":
        return {"type": "jsonl"}
    if output_type != "http":
        raise RuntimeAdapterError(
            f"Task {task.get('id', '<missing>')} output type must be jsonl or http"
        )
    url = str(output.get("url", "")).strip()
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeAdapterError(
            f"Task {task.get('id', '<missing>')} requires an HTTP(S) output URL without credentials"
        )
    connect_timeout = int(output.get("connectTimeoutMs", 1000))
    request_timeout = int(output.get("requestTimeoutMs", 3000))
    if not 100 <= connect_timeout <= request_timeout <= 60000:
        raise RuntimeAdapterError(
            f"Task {task.get('id', '<missing>')} output timeouts must satisfy "
            "100 <= connectTimeoutMs <= requestTimeoutMs <= 60000"
        )
    authorization_env = str(output.get("authorizationEnv", "")).strip()
    if authorization_env and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", authorization_env) is None:
        raise RuntimeAdapterError(
            f"Task {task.get('id', '<missing>')} authorizationEnv is not a valid environment name"
        )
    return {
        "type": "http",
        "url": url,
        "connect_timeout_ms": connect_timeout,
        "request_timeout_ms": request_timeout,
        **({"authorization_env": authorization_env} if authorization_env else {}),
    }


def _task_media(task: dict[str, Any], adapter: str) -> dict[str, Any]:
    task_id = str(task.get("id", "<missing>"))
    raw = task.get("media", {})
    if not isinstance(raw, dict):
        raise RuntimeAdapterError(f"Task {task_id} media must be an object")
    media = cast(dict[str, Any], raw)
    decoder = str(media.get("decoder", "opencv")).strip().lower()
    if decoder not in {"opencv", "rkmpp"}:
        raise RuntimeAdapterError(f"Task {task_id} decoder must be opencv or rkmpp")
    input_uri = str(task.get("inputUri", ""))
    if decoder == "rkmpp" and not input_uri.startswith("rtsp://"):
        raise RuntimeAdapterError(f"Task {task_id} RKMPP decoder requires an RTSP input")

    tracking_raw = media.get("tracking", {})
    if not isinstance(tracking_raw, dict):
        raise RuntimeAdapterError(f"Task {task_id} tracking must be an object")
    tracking = cast(dict[str, Any], tracking_raw)
    tracking_enabled = tracking.get("enabled", False) is True
    if tracking_enabled and not adapter.startswith("yolo_"):
        raise RuntimeAdapterError(f"Task {task_id} tracking requires a detection adapter")
    track_buffer = int(tracking.get("trackBuffer", 30))
    if not 1 <= track_buffer <= 10000:
        raise RuntimeAdapterError(f"Task {task_id} trackBuffer must be between 1 and 10000")

    kafka_raw = media.get("kafka", {})
    if not isinstance(kafka_raw, dict):
        raise RuntimeAdapterError(f"Task {task_id} kafka must be an object")
    kafka = cast(dict[str, Any], kafka_raw)
    kafka_enabled = kafka.get("enabled", False) is True
    brokers = str(kafka.get("brokers", "")).strip()
    topic = str(kafka.get("topic", "sei_msg")).strip()
    key = str(kafka.get("key", "")).strip()
    queue_messages = int(kafka.get("queueMessages", 10000))
    message_timeout_ms = int(kafka.get("messageTimeoutMs", 3000))
    if kafka_enabled and (not brokers or not topic):
        raise RuntimeAdapterError(f"Task {task_id} enabled Kafka requires brokers and topic")
    if not 1 <= queue_messages <= 1000000 or not 100 <= message_timeout_ms <= 60000:
        raise RuntimeAdapterError(f"Task {task_id} Kafka queue or timeout is out of range")

    zlm_raw = media.get("zlmSei", {})
    if not isinstance(zlm_raw, dict):
        raise RuntimeAdapterError(f"Task {task_id} zlmSei must be an object")
    zlm = cast(dict[str, Any], zlm_raw)
    zlm_enabled = zlm.get("enabled", False) is True
    zlm_url = str(zlm.get("publishUri", "")).strip()
    reconnect_ms = int(zlm.get("reconnectMs", 1000))
    if zlm_enabled and decoder != "rkmpp":
        raise RuntimeAdapterError(f"Task {task_id} ZLM SEI requires the RKMPP decoder")
    parsed_zlm_url = urlparse(zlm_url)
    if zlm_enabled and (
        parsed_zlm_url.scheme != "rtsp"
        or not parsed_zlm_url.hostname
        or parsed_zlm_url.username is not None
        or parsed_zlm_url.password is not None
    ):
        raise RuntimeAdapterError(
            f"Task {task_id} ZLM SEI publishUri must be an RTSP URL without userinfo"
        )
    if not 1000 <= reconnect_ms <= 4000:
        raise RuntimeAdapterError(
            f"Task {task_id} ZLM reconnectMs must be between 1000 and 4000"
        )
    return {
        "decoder": decoder,
        "tracking": {"enabled": tracking_enabled, "track_buffer": track_buffer},
        "kafka": {
            "enabled": kafka_enabled,
            "brokers": brokers,
            "topic": topic,
            "key": key,
            "queue_messages": queue_messages,
            "message_timeout_ms": message_timeout_ms,
        },
        "zlm_sei": {
            "enabled": zlm_enabled,
            "output": zlm_url,
            "reconnect_ms": reconnect_ms,
        },
    }


def _task_analytics(
    task: dict[str, Any],
    adapter: str,
    releases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_id = str(task.get("id", "<missing>"))
    raw = task.get("analytics", {})
    if not isinstance(raw, dict):
        raise RuntimeAdapterError(f"Task {task_id} analytics must be an object")
    analytics = cast(dict[str, Any], raw)
    areas_value = analytics.get("areas", [])
    lines_value = analytics.get("lines", [])
    secondary_value = analytics.get("secondaryModels", [])
    osd_value = analytics.get("osd", {})
    events_value = analytics.get("events", {})
    if not isinstance(areas_value, list) or not isinstance(lines_value, list):
        raise RuntimeAdapterError(f"Task {task_id} analytics areas and lines must be arrays")
    if not isinstance(secondary_value, list):
        raise RuntimeAdapterError(f"Task {task_id} secondaryModels must be an array")
    if not isinstance(osd_value, dict) or not isinstance(events_value, dict):
        raise RuntimeAdapterError(f"Task {task_id} analytics osd and events must be objects")
    areas = cast(list[Any], areas_value)
    lines = cast(list[Any], lines_value)
    osd = cast(dict[str, Any], osd_value)
    events = cast(dict[str, Any], events_value)
    if (areas or lines or secondary_value) and not adapter.startswith("yolo_"):
        raise RuntimeAdapterError(
            f"Task {task_id} analytics requires a detection adapter"
        )
    media = _task_media(task, adapter)
    if (areas or lines) and not media["tracking"]["enabled"]:
        raise RuntimeAdapterError(f"Task {task_id} area/line analytics requires tracking")
    if events.get("record") is True and media["decoder"] != "rkmpp":
        raise RuntimeAdapterError(f"Task {task_id} event recording requires RKMPP")

    secondary_models: list[dict[str, Any]] = []
    for index, item in enumerate(cast(list[Any], secondary_value)):
        if not isinstance(item, dict):
            raise RuntimeAdapterError(f"Task {task_id} secondary model {index} must be an object")
        secondary = cast(dict[str, Any], item)
        release_id = str(secondary.get("releaseId", "")).strip()
        release_config = releases.get(release_id)
        if release_config is None:
            raise RuntimeAdapterError(
                f"Task {task_id} references missing secondary release {release_id}"
            )
        secondary_adapter = str(release_config.get("adapter", ""))
        if not secondary_adapter.startswith("yolo_"):
            raise RuntimeAdapterError(
                f"Task {task_id} secondary release {release_id} is not a detection adapter"
            )
        source_class_ids = secondary.get("sourceClassIds", [])
        if not isinstance(source_class_ids, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in cast(list[Any], source_class_ids)
        ):
            raise RuntimeAdapterError(
                f"Task {task_id} secondary sourceClassIds must be non-negative integers"
            )
        confidence_threshold = float(secondary.get("confidenceThreshold", 0.25))
        if not 0 <= confidence_threshold <= 1:
            raise RuntimeAdapterError(
                f"Task {task_id} secondary confidenceThreshold is out of range"
            )
        context_count, worker_count = _pool_config(
            secondary, f"Task {task_id} secondary model {index}"
        )
        secondary_models.append(
            {
                "release_id": release_id,
                "source_class_ids": source_class_ids,
                "confidence_threshold": confidence_threshold,
                "context_count": context_count,
                "worker_count": worker_count,
            }
        )
    return {
        "areas": areas,
        "lines": lines,
        "osd": {
            "enabled": osd.get("enabled", True) is True,
            "show_labels": osd.get("showLabels", True) is True,
            "show_confidence": osd.get("showConfidence", True) is True,
            "show_track_id": osd.get("showTrackId", True) is True,
            "show_areas": osd.get("showAreas", True) is True,
            "show_lines": osd.get("showLines", True) is True,
        },
        "events": {
            "enabled": events.get("enabled", False) is True,
            "snapshot": events.get("snapshot", True) is True,
            "record": events.get("record", False) is True,
            "pre_seconds": int(events.get("preSeconds", 3)),
            "post_seconds": int(events.get("postSeconds", 5)),
            "retention_days": int(events.get("retentionDays", 30)),
        },
        "secondary_models": secondary_models,
    }


def _task_npu_config(task: dict[str, Any]) -> tuple[str, str]:
    mask = str(task.get("npuCoreMask", task.get("npu_core_mask", "auto"))).strip().lower()
    policy = str(
        task.get("npuCorePolicy", task.get("npu_core_policy", "shared"))
    ).strip().lower()
    if mask not in NPU_CORE_BITS:
        raise RuntimeAdapterError(
            f"Task {task.get('id', '<missing>')} has unsupported NPU core mask {mask}"
        )
    if policy not in NPU_CORE_POLICIES:
        raise RuntimeAdapterError(
            f"Task {task.get('id', '<missing>')} has unsupported NPU core policy {policy}"
        )
    if policy == "exclusive" and mask == "auto":
        raise RuntimeAdapterError(
            f"Task {task.get('id', '<missing>')} exclusive NPU policy requires "
            "an explicit core mask"
        )
    return mask, policy


def _pool_config(config: dict[str, Any], owner: str) -> tuple[int, int]:
    context_count = config.get("contextCount", config.get("context_count", 1))
    worker_count = config.get("workerCount", config.get("worker_count", 1))
    if (
        not isinstance(context_count, int)
        or isinstance(context_count, bool)
        or context_count < 1
    ):
        raise RuntimeAdapterError(f"{owner} contextCount must be a positive integer")
    if (
        not isinstance(worker_count, int)
        or isinstance(worker_count, bool)
        or worker_count < 1
    ):
        raise RuntimeAdapterError(f"{owner} workerCount must be a positive integer")
    if worker_count > context_count:
        raise RuntimeAdapterError(f"{owner} workerCount must not exceed contextCount")
    return context_count, worker_count


def _task_runtime_key(task: dict[str, Any], release_id: str, adapter: str) -> str:
    mask, policy = _task_npu_config(task)
    context_count, worker_count = _pool_config(
        task, f"Task {task.get('id', '<missing>')}"
    )
    return json.dumps(
        {
            "releaseId": release_id,
            "adapter": adapter,
            "thresholds": json.loads(_canonical_thresholds(task)),
            "coreMask": mask,
            "corePolicy": policy,
            "contextCount": context_count,
            "workerCount": worker_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_release(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    release_id = str(config.get("releaseId", "")).strip()
    adapter = str(config.get("adapter", "")).strip()
    model_path = Path(str(config.get("modelPath", "")))
    if not release_id or adapter not in ADAPTERS:
        raise RuntimeAdapterError(
            f"Release {release_id or '<missing>'} has unsupported adapter {adapter}"
        )
    if not model_path.is_file() or model_path.suffix.lower() != ".rknn":
        raise RuntimeAdapterError(f"RKNN model does not exist: {model_path}")
    _, manifest = _manifest_for(config)
    expected_contract = ADAPTERS[adapter][2]
    if manifest.get("outputContract") != expected_contract:
        raise RuntimeAdapterError(
            f"Release {release_id} contract {manifest.get('outputContract')} "
            f"does not match {adapter}"
        )
    labels = manifest.get("labels", [])
    if not isinstance(labels, list) or any(
        not isinstance(item, str) for item in cast(list[object], labels)
    ):
        raise RuntimeAdapterError(f"Release {release_id} labels must be a string array")
    if adapter != "ppocr_db_det_v1" and not labels:
        raise RuntimeAdapterError(f"Release {release_id} requires non-empty labels")
    tasks = _tasks_for(config)
    for task in tasks:
        task_id = str(task.get("id", "")).strip()
        input_uri = str(task.get("inputUri", "")).strip()
        if not task_id or not input_uri:
            raise RuntimeAdapterError(f"Release {release_id} contains a task without id/inputUri")
        interval = int(task.get("interval", 1))
        if interval < 1:
            raise RuntimeAdapterError(f"Task {task_id} interval must be at least 1")
        if adapter != "yolo_dfl_split_v1" and interval != 1:
            raise RuntimeAdapterError(f"Structured task {task_id} requires interval=1")
        _task_output(task)
        _task_media(task, adapter)
        _task_npu_config(task)
        _pool_config(task, f"Task {task_id}")
    return manifest, tasks


def validate_release_configs(configs: list[dict[str, Any]]) -> None:
    release_ids: set[str] = set()
    task_ids: set[str] = set()
    runtime_groups: dict[
        tuple[str, str, str, str, str, str, int, int], list[str]
    ] = {}
    releases = {str(config.get("releaseId", "")): config for config in configs}
    for config in configs:
        release_id = str(config.get("releaseId", ""))
        if release_id in release_ids:
            raise RuntimeAdapterError(f"Duplicate release config: {release_id}")
        release_ids.add(release_id)
        _, tasks = _validate_release(config)
        for task in tasks:
            task_id = str(task["id"])
            if task_id in task_ids:
                raise RuntimeAdapterError(f"Duplicate task config: {task_id}")
            task_ids.add(task_id)
            mask, policy = _task_npu_config(task)
            context_count, worker_count = _pool_config(task, f"Task {task_id}")
            adapter = str(config.get("adapter", ""))
            runtime_groups.setdefault(
                (
                    "primary",
                    release_id,
                    mask,
                    policy,
                    _canonical_thresholds(task),
                    adapter,
                    context_count,
                    worker_count,
                ),
                [],
            ).append(task_id)
            analytics = _task_analytics(task, adapter, releases)
            for secondary_index, secondary in enumerate(analytics["secondary_models"]):
                secondary_release_id = str(secondary["release_id"])
                secondary_config = releases[secondary_release_id]
                secondary_adapter = str(secondary_config.get("adapter", ""))
                secondary_task = {
                    **task,
                    "thresholds": {
                        "confidence": secondary["confidence_threshold"],
                        "nms": 0.5,
                    },
                    "contextCount": secondary["context_count"],
                    "workerCount": secondary["worker_count"],
                }
                secondary_context_count, secondary_worker_count = _pool_config(
                    secondary_task, f"Task {task_id} secondary model {secondary_index}"
                )
                runtime_groups.setdefault(
                    (
                        "secondary",
                        secondary_release_id,
                        mask,
                        policy,
                        _canonical_thresholds(secondary_task),
                        secondary_adapter,
                        secondary_context_count,
                        secondary_worker_count,
                    ),
                    [],
                ).append(f"{task_id}:secondary:{secondary_index}")
    groups = list(runtime_groups.items())
    for index, (left_key, left_tasks) in enumerate(groups):
        for right_key, right_tasks in groups[index + 1 :]:
            if "exclusive" not in {left_key[3], right_key[3]}:
                continue
            if NPU_CORE_BITS[left_key[2]] & NPU_CORE_BITS[right_key[2]] == 0:
                continue
            raise RuntimeAdapterError(
                "Exclusive NPU core assignments overlap: "
                f"{left_key[2]} ({','.join(left_tasks)}) and "
                f"{right_key[2]} ({','.join(right_tasks)})"
            )


def _threshold(task: dict[str, Any], names: tuple[str, ...], default: float) -> float:
    value = task.get("thresholds", {})
    thresholds = cast(dict[str, Any], value) if isinstance(value, dict) else {}
    for name in names:
        if name in thresholds:
            return float(thresholds[name])
    return default


def _instance_config(
    adapter: str,
    model_path: str,
    label_path: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    factory, model_type, _ = ADAPTERS[adapter]
    result: dict[str, Any] = {
        "instance_name": factory,
        "model_path": model_path,
        "queue_capacity": 8,
        "type": model_type,
        "enable": 1,
    }
    if adapter != "ppocr_db_det_v1":
        result["label_path"] = label_path
    core_mask, core_policy = _task_npu_config(task)
    result["core_mask"] = core_mask
    result["core_policy"] = core_policy
    context_count, worker_count = _pool_config(
        task, f"Task {task.get('id', '<missing>')}"
    )
    result["context_count"] = context_count
    result["worker_count"] = worker_count
    result["queue_capacity"] = max(8, worker_count * 2)
    if adapter == "yolo_dfl_split_v1":
        result.update(
            {
                "confidence_threshold": _threshold(
                    task, ("confidence", "confidenceThreshold"), 0.4
                ),
                "nms_threshold": _threshold(task, ("nms", "nmsThreshold"), 0.5),
                "class_scores_logits": False,
                "max_detections": 1024,
            }
        )
    elif adapter == "ppocr_db_det_v1":
        result.update(
            {
                "binary_threshold": _threshold(task, ("binary", "binaryThreshold"), 0.3),
                "box_threshold": _threshold(task, ("box", "boxThreshold"), 0.6),
                "unclip_ratio": _threshold(task, ("unclip", "unclipRatio"), 1.5),
                "min_size": 3,
                "max_candidates": 1000,
                "max_regions": 100,
            }
        )
    elif adapter == "ppocr_ctc_rec_v1":
        result.update({"blank_index": 0, "ctc_scores_logits": True})
    return result


def _canonical_thresholds(task: dict[str, Any]) -> str:
    value = task.get("thresholds", {})
    thresholds = cast(dict[str, Any], value) if isinstance(value, dict) else {}
    return json.dumps(thresholds, sort_keys=True, separators=(",", ":"))


def _task_revision(task: dict[str, Any], node_revision: int) -> int:
    """Use the task revision for result envelopes, with legacy fallback."""
    value = task.get("configRevision", task.get("config_revision"))
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return node_revision


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def prepare_revision(
    configs: list[dict[str, Any]], revision: int, state_root: Path, output_root: Path
) -> tuple[Path, list[str]]:
    validate_release_configs(configs)
    canonical = json.dumps(configs, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    revisions_root = state_root / "revisions"
    destination = revisions_root / str(revision)
    if destination.exists():
        metadata = _load_object(destination / "revision.json")
        if metadata.get("configHash") != config_hash:
            raise RuntimeAdapterError(f"Revision {revision} already exists with different content")
        names_value = metadata.get("instances", [])
        if not isinstance(names_value, list) or any(
            not isinstance(item, str) for item in cast(list[object], names_value)
        ):
            raise RuntimeAdapterError(f"Revision {revision} instance metadata is invalid")
        return destination, [str(item) for item in cast(list[object], names_value)]

    temporary = revisions_root / f".{revision}.staging-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "pipelines").mkdir(parents=True)
    (temporary / "labels").mkdir()
    output_root.mkdir(parents=True, exist_ok=True)
    instances: dict[str, Any] = {}
    instance_names: list[str] = []
    secondary_instances: dict[str, str] = {}
    task_outputs: list[dict[str, str]] = []
    try:
        release_lookup = {str(config.get("releaseId", "")): config for config in configs}
        release_details: dict[str, tuple[dict[str, Any], Path]] = {}
        for config in configs:
            manifest, _ = _validate_release(config)
            release_id = str(config["releaseId"])
            label_path = destination / "labels" / f"{_safe_name(release_id, 'release')}.txt"
            labels = cast(list[str], manifest.get("labels", []))
            (temporary / "labels" / label_path.name).write_text(
                "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
            )
            release_details[release_id] = (manifest, label_path)
        for release_index, config in enumerate(configs):
            manifest, tasks = _validate_release(config)
            release_id = str(config["releaseId"])
            adapter = str(config["adapter"])
            final_label_path = release_details[release_id][1]
            grouped: dict[str, tuple[str, list[dict[str, Any]]]] = {}
            for task in tasks:
                key = _task_runtime_key(task, release_id, adapter)
                grouped.setdefault(key, (key, []))[1].append(task)
            for group_index, (_, group_tasks) in enumerate(grouped.values()):
                release_slug = _safe_name(release_id, "release")[:40]
                instance_name = _safe_name(
                    f"release-{release_index}-{release_slug}-{group_index}-"
                    f"{hashlib.sha256(f'{release_id}:{group_index}'.encode()).hexdigest()[:8]}",
                    f"release-{release_index}-{group_index}",
                )
                instances[instance_name] = _instance_config(
                    adapter,
                    str(config["modelPath"]),
                    str(final_label_path),
                    group_tasks[0],
                )
                instance_names.append(instance_name)
                for task in group_tasks:
                    task_id = str(task["id"])
                    task_revision = _task_revision(task, revision)
                    task_slug = _safe_name(task_id, "task")[:60]
                    pipeline_name = _safe_name(
                        f"{task_slug}-{hashlib.sha256(task_id.encode()).hexdigest()[:8]}",
                        f"task-{len(instance_names)}",
                    )
                    interval = int(task.get("interval", 1))
                    output = _task_output(task)
                    media = _task_media(task, adapter)
                    analytics = _task_analytics(task, adapter, release_lookup)
                    sink: str | dict[str, Any]
                    if output["type"] == "http":
                        sink = output
                    else:
                        sink = str(output_root / f"{pipeline_name}.jsonl")
                        task_outputs.append(
                            {"taskId": task_id, "type": "jsonl", "path": sink}
                        )
                    capture_node = (
                        "RkMppCaptureNode"
                        if media["decoder"] == "rkmpp"
                        else "VideoCaptureNode"
                    )
                    result_upstream = "infer"
                    pipeline: dict[str, Any] = {
                        "inputs": [str(task["inputUri"])],
                        "capture": {"node": capture_node, "loop": True, "reconnect_ms": 1000},
                        "infer": {
                            "node": "InferNode",
                            "instance": instance_name,
                            "interval": interval,
                            "link_to": ["capture"],
                        },
                    }
                    if media["tracking"]["enabled"]:
                        pipeline["tracking"] = {
                            "node": "ByteTrackNode",
                            "track_buffer": media["tracking"]["track_buffer"],
                            "link_to": ["infer"],
                        }
                        result_upstream = "tracking"
                    for secondary_index, secondary in enumerate(analytics["secondary_models"]):
                        secondary_release_id = str(secondary["release_id"])
                        secondary_config = release_lookup[secondary_release_id]
                        secondary_adapter = str(secondary_config["adapter"])
                        secondary_label_path = release_details[secondary_release_id][1]
                        secondary_task = {
                            **task,
                            "contextCount": secondary["context_count"],
                            "workerCount": secondary["worker_count"],
                            "thresholds": {
                                "confidence": secondary["confidence_threshold"],
                                "nms": 0.5,
                            },
                        }
                        secondary_key = _task_runtime_key(
                            secondary_task, secondary_release_id, secondary_adapter
                        )
                        secondary_instance = secondary_instances.get(secondary_key)
                        if secondary_instance is None:
                            secondary_slug = _safe_name(
                                secondary_release_id, "secondary-release"
                            )[:40]
                            secondary_instance = _safe_name(
                                f"secondary-{secondary_slug}-"
                                f"{hashlib.sha256(secondary_key.encode()).hexdigest()[:8]}",
                                f"secondary-{secondary_index}",
                            )
                            instances[secondary_instance] = _instance_config(
                                secondary_adapter,
                                str(secondary_config["modelPath"]),
                                str(secondary_label_path),
                                secondary_task,
                            )
                            instance_names.append(secondary_instance)
                            secondary_instances[secondary_key] = secondary_instance
                        node_name = f"secondary_{secondary_index}"
                        pipeline[node_name] = {
                            "node": "SecondaryInferNode",
                            "instance": secondary_instance,
                            "primary_instance": instance_name,
                            "source_class_ids": secondary["source_class_ids"],
                            "confidence_threshold": secondary["confidence_threshold"],
                            "link_to": [result_upstream],
                        }
                        result_upstream = node_name
                    if analytics["areas"] or analytics["lines"]:
                        pipeline["analytics"] = {
                            "node": "AnalyticsNode",
                            "task_id": task_id,
                            "primary_instance": instance_name,
                            "areas": analytics["areas"],
                            "lines": analytics["lines"],
                            "link_to": [result_upstream],
                        }
                        result_upstream = "analytics"
                    if analytics["events"]["enabled"]:
                        event_root = output_root / "events" / pipeline_name
                        task_outputs.append(
                            {"taskId": task_id, "type": "events", "path": str(event_root)}
                        )
                        pipeline["events"] = {
                            "node": "EventOutputNode",
                            "task_id": task_id,
                            "output": str(event_root),
                            **analytics["events"],
                            "link_to": [result_upstream],
                        }
                        result_upstream = "events"
                    pipeline["result"] = {
                        "node": "JsonOutputNode",
                        "instance": instance_name,
                        "task_id": task_id,
                        "revision": task_revision,
                        "output": sink,
                        "link_to": [result_upstream],
                    }
                    if media["kafka"]["enabled"]:
                        kafka_config = {
                            "node": "KafkaOutputNode",
                            "input": str(task["inputUri"]),
                            "instance": instance_name,
                            "task_id": task_id,
                            "revision": task_revision,
                            "brokers": media["kafka"]["brokers"],
                            "topic": media["kafka"]["topic"],
                            "queue_messages": media["kafka"]["queue_messages"],
                            "message_timeout_ms": media["kafka"]["message_timeout_ms"],
                            "link_to": [result_upstream],
                        }
                        if media["kafka"]["key"]:
                            kafka_config["key"] = media["kafka"]["key"]
                        pipeline["kafka"] = kafka_config
                    if media["zlm_sei"]["enabled"]:
                        pipeline["zlm_sei"] = {
                            "node": "ZlmSeiOutputNode",
                            "output": media["zlm_sei"]["output"],
                            "instance": instance_name,
                            "task_id": task_id,
                            "revision": task_revision,
                            "reconnect_ms": media["zlm_sei"]["reconnect_ms"],
                            "link_to": [result_upstream],
                        }
                    _write_json(temporary / "pipelines" / f"{pipeline_name}.yaml", pipeline)
        _write_json(temporary / "instances.yaml", instances)
        _write_json(
            temporary / "base.yaml",
            {
                "instances": "instances.yaml",
                "pipelines": "pipelines",
                "log_config": {"log_level": 2},
                "pipe_perf_interval": 5,
                "instance_perf_interval": 5,
            },
        )
        _write_json(
            temporary / "revision.json",
            {
                "revision": revision,
                "configHash": config_hash,
                "releaseCount": len(configs),
                "taskCount": sum(len(_tasks_for(config)) for config in configs),
                "instances": instance_names,
                "contextCount": sum(
                    int(instance.get("context_count", 1))
                    for instance in instances.values()
                ),
                "instancePools": [
                    {
                        "name": name,
                        "contextCount": int(config.get("context_count", 1)),
                        "workerCount": int(config.get("worker_count", 1)),
                        "queueCapacity": int(config.get("queue_capacity", 8)),
                    }
                    for name, config in instances.items()
                ],
                "taskOutputs": task_outputs,
            },
        )
        revisions_root.mkdir(parents=True, exist_ok=True)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination, instance_names


def _read_optional_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _load_object(path)
    except RuntimeAdapterError:
        return None


def _process_start_time(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except OSError:
        return None
    return fields[21] if len(fields) > 21 else None


def _active_process(state_root: Path) -> tuple[int, str] | None:
    active = _read_optional_object(state_root / "active.json")
    if active is None or active.get("empty") is True:
        return None
    try:
        pid = int(active["pid"])
        expected_start = str(active["processStart"])
    except (KeyError, TypeError, ValueError):
        return None
    if _process_start_time(pid) != expected_start:
        return None
    return pid, expected_start


def stop_active(state_root: Path) -> None:
    active = _active_process(state_root)
    if active is None:
        (state_root / "active.json").unlink(missing_ok=True)
        return
    pid, _ = active
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + float(os.environ.get("RKNODE_PIPELINE_STOP_TIMEOUT", "20"))
    while time.monotonic() < deadline and _process_start_time(pid) is not None:
        time.sleep(0.1)
    if _process_start_time(pid) is not None:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    (state_root / "active.json").unlink(missing_ok=True)


def _atomic_link(link: Path, target: Path | None) -> None:
    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    if target is None:
        link.unlink(missing_ok=True)
        return
    temporary.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
    temporary.replace(link)


def start_revision(state_root: Path, revision_dir: Path) -> None:
    metadata = _load_object(revision_dir / "revision.json")
    revision = int(metadata["revision"])
    if int(metadata.get("taskCount", 0)) == 0:
        _write_json(state_root / "active.json", {"revision": revision, "empty": True})
        return
    binary = _pipeline_binary()
    if not binary.is_file():
        raise RuntimeAdapterError(f"Pipeline binary does not exist: {binary}")
    ready_file = revision_dir / "ready"
    ready_file.unlink(missing_ok=True)
    environment = {**os.environ, "RKNODE_READY_FILE": str(ready_file)}
    # Keep pipeline output on the container streams so Docker's bounded log
    # rotation applies. A persistent append-only file can otherwise grow until
    # it consumes the model/state volume.
    process = subprocess.Popen(
        [str(binary), str(revision_dir / "base.yaml"), "0"],
        env=environment,
        start_new_session=True,
    )
    deadline = time.monotonic() + float(os.environ.get("RKNODE_PIPELINE_START_TIMEOUT", "30"))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeAdapterError(f"Pipeline exited during startup with {process.returncode}")
        if ready_file.is_file():
            process_start = _process_start_time(process.pid)
            if process_start is None:
                raise RuntimeAdapterError("Pipeline process disappeared after readiness")
            _write_json(
                state_root / "active.json",
                {
                    "revision": revision,
                    "pid": process.pid,
                    "processStart": process_start,
                    "empty": False,
                },
            )
            return
        time.sleep(0.1)
    process.terminate()
    raise RuntimeAdapterError("Pipeline readiness timed out")


def probe_instances(revision_dir: Path, instance_names: list[str]) -> None:
    if not instance_names:
        return
    binary = _probe_binary()
    if not binary.is_file():
        raise RuntimeAdapterError(f"Instance probe binary does not exist: {binary}")
    completed = subprocess.run(
        [str(binary), str(revision_dir / "instances.yaml"), *instance_names],
        check=False,
        timeout=float(os.environ.get("RKNODE_PROBE_TIMEOUT", "120")),
    )
    if completed.returncode != 0:
        raise RuntimeAdapterError(f"RKNN instance probe exited with {completed.returncode}")


def cleanup_unused_outputs(revision_dir: Path) -> None:
    metadata = _load_object(revision_dir / "revision.json")
    if "taskOutputs" not in metadata:
        return
    raw_outputs = metadata.get("taskOutputs", [])
    if not isinstance(raw_outputs, list):
        return
    output_root = _output_root().resolve()
    event_root = (output_root / "events").resolve()
    expected_jsonl: set[Path] = set()
    expected_events: set[Path] = set()
    for raw_output in cast(list[object], raw_outputs):
        if not isinstance(raw_output, dict):
            continue
        item = cast(dict[str, Any], raw_output)
        path_value = item.get("path")
        output_type = item.get("type")
        if not isinstance(path_value, str) or not isinstance(output_type, str):
            continue
        path = Path(path_value).resolve()
        if output_type == "jsonl" and path.parent == output_root:
            expected_jsonl.add(path)
        elif output_type == "events" and path.parent == event_root:
            expected_events.add(path)
    if output_root.is_dir():
        for path in output_root.iterdir():
            if path.is_file() and path.suffix.lower() == ".jsonl" and path not in expected_jsonl:
                path.unlink(missing_ok=True)
    if event_root.is_dir():
        for path in event_root.iterdir():
            if path.is_dir() and path not in expected_events:
                shutil.rmtree(path)


def _current_revision_dir(state_root: Path) -> Path | None:
    current = state_root / "current"
    if not current.is_symlink():
        return None
    try:
        target = current.resolve(strict=True)
    except OSError:
        return None
    revisions_root = (state_root / "revisions").resolve()
    if target.parent != revisions_root:
        raise RuntimeAdapterError("Current runtime revision points outside the revision root")
    return target


@contextmanager
def _runtime_lock(state_root: Path) -> Generator[None, None, None]:
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "runtime.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def command_self_test() -> None:
    for binary in (_pipeline_binary(), _probe_binary(), _protocol_probe_binary()):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeAdapterError(f"Required runtime binary is not executable: {binary}")
    protocol_probe = subprocess.run(
        [str(_protocol_probe_binary())], check=False, timeout=10, capture_output=True, text=True
    )
    if protocol_probe.returncode != 0:
        raise RuntimeAdapterError(
            f"Runtime protocol probe failed: {protocol_probe.stderr.strip()}"
        )
    if _env_bool("RKNODE_REQUIRE_NPU_DEVICE", True):
        missing_paths = missing_rk3588_inference_paths()
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise RuntimeAdapterError(
                f"RK3588 inference devices or device tree are unavailable: {missing}"
            )
    advertised = {
        item.strip() for item in os.environ.get("RKNODE_ADAPTERS", "").split(",") if item.strip()
    }
    unsupported = advertised - ADAPTERS.keys()
    if unsupported:
        raise RuntimeAdapterError(f"Unsupported advertised adapters: {sorted(unsupported)}")
    state_root = _state_root()
    with _runtime_lock(state_root):
        current = _current_revision_dir(state_root)
        if current is not None and not runtime_healthy(state_root):
            stop_active(state_root)
            start_revision(state_root, current)


def command_probe() -> None:
    configs = _load_release_configs()
    if len(configs) != 1:
        raise RuntimeAdapterError("Model probe requires exactly one release config")
    validate_release_configs(configs)


def command_activate() -> None:
    configs = _load_release_configs()
    revision_text = os.environ.get("RKNODE_DESIRED_REVISION", "").strip()
    if not revision_text:
        raise RuntimeAdapterError("RKNODE_DESIRED_REVISION is required")
    revision = int(revision_text)
    if revision < 0:
        raise RuntimeAdapterError("RKNODE_DESIRED_REVISION must be non-negative")
    state_root = _state_root()
    with _runtime_lock(state_root):
        revision_dir, instance_names = prepare_revision(
            configs, revision, state_root, _output_root()
        )
        previous = _current_revision_dir(state_root)
        stop_active(state_root)
        try:
            probe_instances(revision_dir, instance_names)
            _atomic_link(state_root / "current", revision_dir)
            start_revision(state_root, revision_dir)
            cleanup_unused_outputs(revision_dir)
        except Exception:
            _atomic_link(state_root / "current", previous)
            if previous is not None:
                start_revision(state_root, previous)
            raise
        if previous is not None and previous != revision_dir:
            _atomic_link(state_root / "previous", previous)


def runtime_healthy(state_root: Path) -> bool:
    current = _current_revision_dir(state_root)
    if current is None:
        return False
    metadata = _load_object(current / "revision.json")
    active = _read_optional_object(state_root / "active.json")
    if active is None:
        return False
    try:
        active_revision = int(active.get("revision", -1))
        expected_revision = int(metadata["revision"])
    except (KeyError, TypeError, ValueError):
        return False
    if active_revision != expected_revision:
        return False
    if int(metadata.get("taskCount", 0)) == 0:
        return active.get("empty") is True and _active_process(state_root) is None
    return _active_process(state_root) is not None and (current / "ready").is_file()


def command_health() -> None:
    state_root = _state_root()
    with _runtime_lock(state_root):
        if runtime_healthy(state_root):
            return
        if not _env_bool("RKNODE_AUTO_RECOVER", True):
            raise RuntimeAdapterError("Runtime is not healthy")
        current = _current_revision_dir(state_root)
        if current is None:
            raise RuntimeAdapterError("Runtime has no active revision")
        stop_active(state_root)
        start_revision(state_root, current)
        if not runtime_healthy(state_root):
            raise RuntimeAdapterError("Runtime recovery did not become healthy")


def main() -> None:
    parser = argparse.ArgumentParser(description="RK3588 nv_video_pipeline runtime adapter")
    parser.add_argument("command", choices=("self-test", "probe", "activate", "health"))
    args = parser.parse_args()
    commands = {
        "self-test": command_self_test,
        "probe": command_probe,
        "activate": command_activate,
        "health": command_health,
    }
    try:
        commands[args.command]()
    except (
        KeyError,
        OSError,
        ValueError,
        RuntimeAdapterError,
        subprocess.SubprocessError,
    ) as error:
        raise SystemExit(f"runtime adapter {args.command} failed: {error}") from error


if __name__ == "__main__":
    main()
