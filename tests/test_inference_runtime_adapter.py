from __future__ import annotations

import json
from pathlib import Path

import pytest
from workers.common.rk3588_devices import (
    RK3588_INFERENCE_PATHS,
    missing_rk3588_inference_paths,
)
from workers.inference_agent.runtime_adapter import (
    RuntimeAdapterError,
    cleanup_unused_outputs,
    command_activate,
    command_health,
    prepare_revision,
    runtime_healthy,
    stop_active,
    validate_release_configs,
)


def _release(tmp_path: Path, *, adapter: str = "deeplab_logits_v1") -> dict[str, object]:
    model = tmp_path / f"{adapter}.rknn"
    model.write_bytes(b"rknn")
    contracts = {
        "deeplab_logits_v1": "semantic_logits_nchw_v1",
        "ppocr_db_det_v1": "ppocr_db_probability_map_v1",
        "yolo_dfl_split_v1": "rknn_yolo_dfl_split_heads_v1",
    }
    manifest = tmp_path / f"{adapter}.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "outputContract": contracts[adapter],
                "labels": ["background", "defect"] if adapter != "ppocr_db_det_v1" else [],
            }
        ),
        encoding="utf-8",
    )
    return {
        "releaseId": f"release_{adapter}",
        "adapter": adapter,
        "modelPath": str(model),
        "manifestPath": str(manifest),
        "tasks": [
            {
                "id": "task_a",
                "inputUri": "rtsp://camera/a",
                "interval": 1,
                "thresholds": {},
                "output": {"type": "jsonl"},
            },
            {
                "id": "task_b",
                "inputUri": "rtsp://camera/b",
                "interval": 1,
                "thresholds": {},
                "output": {"type": "jsonl"},
            },
        ],
    }


def test_rk3588_device_contract_reports_only_missing_paths(tmp_path: Path) -> None:
    present = tmp_path / "present"
    present.touch()
    missing = tmp_path / "missing"

    assert missing_rk3588_inference_paths((present, missing)) == [missing]


def test_rk3588_device_tree_contract_uses_mounted_sysfs_path() -> None:
    assert Path("/sys/firmware/devicetree/base/compatible") in RK3588_INFERENCE_PATHS


def test_prepare_revision_shares_instance_and_generates_task_pipelines(tmp_path: Path) -> None:
    state_root = tmp_path / "runtime"
    output_root = tmp_path / "output"

    revision, instances = prepare_revision([_release(tmp_path)], 7, state_root, output_root)

    instance_config = json.loads((revision / "instances.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((revision / "revision.json").read_text(encoding="utf-8"))
    assert len(instances) == 1
    assert list(instance_config) == instances
    assert instance_config[instances[0]]["type"] == "DEEPLAB_LOGITS"
    assert metadata["releaseCount"] == 1
    assert metadata["taskCount"] == 2
    assert "previews" not in metadata
    assert len(list((revision / "pipelines").glob("*.yaml"))) == 2
    pipelines = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (revision / "pipelines").glob("*.yaml")
    ]
    assert all("preview" not in item for item in pipelines)


def test_prepare_revision_groups_tasks_by_context_worker_pool(tmp_path: Path) -> None:
    release = _release(tmp_path)
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    assert isinstance(tasks[1], dict)
    tasks[0].update({"contextCount": 3, "workerCount": 2})
    tasks[1].update({"contextCount": 3, "workerCount": 2})

    shared_revision, shared_instances = prepare_revision(
        [release], 20, tmp_path / "runtime-shared", tmp_path / "output-shared"
    )

    shared_config = json.loads(
        (shared_revision / "instances.yaml").read_text(encoding="utf-8")
    )
    shared_metadata = json.loads(
        (shared_revision / "revision.json").read_text(encoding="utf-8")
    )
    assert len(shared_instances) == 1
    assert shared_config[shared_instances[0]]["context_count"] == 3
    assert shared_config[shared_instances[0]]["worker_count"] == 2
    assert shared_config[shared_instances[0]]["queue_capacity"] == 8
    assert shared_metadata["contextCount"] == 3

    tasks[1].update({"contextCount": 2, "workerCount": 2})
    split_revision, split_instances = prepare_revision(
        [release], 21, tmp_path / "runtime-split", tmp_path / "output-split"
    )
    split_config = json.loads(
        (split_revision / "instances.yaml").read_text(encoding="utf-8")
    )
    split_metadata = json.loads(
        (split_revision / "revision.json").read_text(encoding="utf-8")
    )
    assert len(split_instances) == 2
    assert {item["context_count"] for item in split_config.values()} == {2, 3}
    assert split_metadata["contextCount"] == 5


def test_runtime_adapter_rejects_invalid_context_worker_pool(tmp_path: Path) -> None:
    release = _release(tmp_path)
    tasks = release["tasks"]
    assert isinstance(tasks, list) and isinstance(tasks[0], dict)
    tasks[0].update({"contextCount": 1, "workerCount": 2})

    with pytest.raises(RuntimeAdapterError, match="workerCount must not exceed contextCount"):
        validate_release_configs([release])


def test_cleanup_unused_outputs_removes_retired_task_results(
    tmp_path: Path, monkeypatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setenv("RKNODE_INFERENCE_OUTPUT_DIR", str(output_root))
    release = _release(tmp_path, adapter="yolo_dfl_split_v1")
    tasks = release["tasks"]
    assert isinstance(tasks, list) and isinstance(tasks[0], dict)
    tasks[0]["media"] = {"tracking": {"enabled": True}}
    tasks[0]["analytics"] = {
        "areas": [
            {
                "id": "zone-a",
                "polygon": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.9, "y": 0.1},
                    {"x": 0.9, "y": 0.9},
                ],
            }
        ],
        "events": {"enabled": True, "snapshot": True},
    }
    revision, _ = prepare_revision(
        [release], 11, tmp_path / "runtime", output_root
    )
    metadata = json.loads((revision / "revision.json").read_text(encoding="utf-8"))
    expected_jsonl = Path(
        next(item["path"] for item in metadata["taskOutputs"] if item["type"] == "jsonl")
    )
    expected_events = Path(
        next(item["path"] for item in metadata["taskOutputs"] if item["type"] == "events")
    )
    expected_jsonl.write_text("active\n", encoding="utf-8")
    expected_events.mkdir(parents=True)
    (expected_events / "events.jsonl").write_text("active\n", encoding="utf-8")
    stale_jsonl = output_root / "retired-task.jsonl"
    stale_jsonl.write_text("stale\n", encoding="utf-8")
    stale_events = output_root / "events" / "retired-task"
    stale_events.mkdir(parents=True)
    (stale_events / "events.jsonl").write_text("stale\n", encoding="utf-8")

    cleanup_unused_outputs(revision)

    assert expected_jsonl.is_file()
    assert expected_events.is_dir()
    assert not stale_jsonl.exists()
    assert not stale_events.exists()


def test_cleanup_unused_outputs_preserves_results_for_legacy_revision_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setenv("RKNODE_INFERENCE_OUTPUT_DIR", str(output_root))
    revision = tmp_path / "runtime" / "revisions" / "legacy"
    revision.mkdir(parents=True)
    (revision / "revision.json").write_text(
        json.dumps({"revision": 1}), encoding="utf-8"
    )
    output_root.mkdir()
    result = output_root / "legacy-task.jsonl"
    result.write_text("active\n", encoding="utf-8")
    events = output_root / "events" / "legacy-task"
    events.mkdir(parents=True)

    cleanup_unused_outputs(revision)

    assert result.is_file()
    assert events.is_dir()


def test_prepare_revision_splits_contexts_by_npu_core_mask(tmp_path: Path) -> None:
    release = _release(tmp_path)
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    assert isinstance(tasks[1], dict)
    tasks[0].update({"npuCoreMask": "core0", "npuCorePolicy": "exclusive"})
    tasks[1].update({"npuCoreMask": "core1", "npuCorePolicy": "exclusive"})

    revision, instances = prepare_revision(
        [release], 12, tmp_path / "runtime", tmp_path / "output"
    )

    instance_config = json.loads((revision / "instances.yaml").read_text(encoding="utf-8"))
    assert len(instances) == 2
    assert {item["core_mask"] for item in instance_config.values()} == {"core0", "core1"}
    assert {item["core_policy"] for item in instance_config.values()} == {"exclusive"}


def test_release_rejects_overlapping_exclusive_npu_cores(tmp_path: Path) -> None:
    release = _release(tmp_path)
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    assert isinstance(tasks[1], dict)
    tasks[0].update({"npuCoreMask": "core0_1", "npuCorePolicy": "exclusive"})
    tasks[1].update({"npuCoreMask": "core1", "npuCorePolicy": "shared"})

    with pytest.raises(RuntimeAdapterError, match="Exclusive NPU core assignments overlap"):
        validate_release_configs([release])


def test_release_rejects_primary_secondary_exclusive_overlap_when_configs_match(
    tmp_path: Path,
) -> None:
    release = _release(tmp_path, adapter="yolo_dfl_split_v1")
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    tasks[:] = [tasks[0]]
    tasks[0].update(
        {
            "thresholds": {"confidence": 0.25, "nms": 0.5},
            "npuCoreMask": "core0",
            "npuCorePolicy": "exclusive",
            "analytics": {
                "secondaryModels": [
                    {
                        "releaseId": release["releaseId"],
                        "sourceClassIds": [0],
                        "confidenceThreshold": 0.25,
                        "contextCount": 1,
                        "workerCount": 1,
                    }
                ]
            },
        }
    )

    with pytest.raises(RuntimeAdapterError, match="Exclusive NPU core assignments overlap"):
        validate_release_configs([release])


def test_exclusive_npu_policy_requires_explicit_mask(tmp_path: Path) -> None:
    release = _release(tmp_path)
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    tasks[0]["npuCorePolicy"] = "exclusive"

    with pytest.raises(RuntimeAdapterError, match="requires an explicit core mask"):
        validate_release_configs([release])


def test_structured_release_rejects_interval_reuse(tmp_path: Path) -> None:
    release = _release(tmp_path)
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    tasks[0]["interval"] = 2

    with pytest.raises(RuntimeAdapterError, match="requires interval=1"):
        validate_release_configs([release])


def test_prepare_revision_generates_bounded_http_sink(tmp_path: Path) -> None:
    release = _release(tmp_path)
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    tasks[0]["output"] = {
        "type": "http",
        "url": "https://consumer.example/results",
        "authorizationEnv": "RKNODE_RESULT_SINK_TOKEN",
        "connectTimeoutMs": 500,
        "requestTimeoutMs": 2000,
    }

    revision, _ = prepare_revision([release], 8, tmp_path / "runtime", tmp_path / "output")

    pipelines = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (revision / "pipelines").glob("*.yaml")
    ]
    http_pipeline = next(item for item in pipelines if item["inputs"][0] == "rtsp://camera/a")
    assert http_pipeline["result"]["output"] == {
        "type": "http",
        "url": "https://consumer.example/results",
        "authorization_env": "RKNODE_RESULT_SINK_TOKEN",
        "connect_timeout_ms": 500,
        "request_timeout_ms": 2000,
    }


def test_http_sink_rejects_embedded_credentials(tmp_path: Path) -> None:
    release = _release(tmp_path)
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    tasks[0]["output"] = {
        "type": "http",
        "url": "https://user:secret@consumer.example/results",
    }

    with pytest.raises(RuntimeAdapterError, match="without credentials"):
        validate_release_configs([release])


def test_prepare_revision_generates_rkmpp_tracking_kafka_and_zlm_graph(tmp_path: Path) -> None:
    release = _release(tmp_path, adapter="yolo_dfl_split_v1")
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    tasks[0]["media"] = {
        "decoder": "rkmpp",
        "tracking": {"enabled": True, "trackBuffer": 45},
        "kafka": {
            "enabled": True,
            "brokers": "kafka-a:9092,kafka-b:9092",
            "topic": "sei_msg",
        },
        "zlmSei": {
            "enabled": True,
            "publishUri": "rtsp://zlm/live/line-a-result?publishToken=opaque",
            "reconnectMs": 1500,
        },
    }

    revision, _ = prepare_revision([release], 13, tmp_path / "runtime", tmp_path / "output")
    pipelines = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (revision / "pipelines").glob("*.yaml")
    ]
    pipeline = next(item for item in pipelines if item["inputs"][0].endswith("/a"))
    assert pipeline["capture"]["node"] == "RkMppCaptureNode"
    assert pipeline["tracking"] == {
        "node": "ByteTrackNode",
        "track_buffer": 45,
        "link_to": ["infer"],
    }
    assert pipeline["result"]["link_to"] == ["tracking"]
    assert pipeline["kafka"]["link_to"] == ["tracking"]
    assert pipeline["kafka"]["brokers"] == "kafka-a:9092,kafka-b:9092"
    assert pipeline["kafka"]["input"] == "rtsp://camera/a"
    assert "key" not in pipeline["kafka"]
    assert pipeline["zlm_sei"]["link_to"] == ["tracking"]
    assert pipeline["zlm_sei"]["output"].startswith("rtsp://zlm/live/line-a-result?")


def test_prepare_revision_generates_secondary_analytics_and_event_graph(
    tmp_path: Path,
) -> None:
    primary = _release(tmp_path, adapter="yolo_dfl_split_v1")
    primary["releaseId"] = "release_primary"
    tasks = primary["tasks"]
    assert isinstance(tasks, list) and isinstance(tasks[0], dict)
    tasks[0]["media"] = {"tracking": {"enabled": True}}
    tasks[0]["analytics"] = {
        "areas": [
            {
                "id": "zone-a",
                "polygon": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.9, "y": 0.1},
                    {"x": 0.9, "y": 0.9},
                ],
            }
        ],
        "lines": [
            {
                "id": "line-a",
                "start": {"x": 0.1, "y": 0.5},
                "end": {"x": 0.9, "y": 0.5},
            }
        ],
        "events": {"enabled": True, "snapshot": True},
        "secondaryModels": [
            {
                "releaseId": "release_secondary",
                "sourceClassIds": [0],
                "confidenceThreshold": 0.3,
            }
        ],
    }
    secondary = _release(tmp_path, adapter="yolo_dfl_split_v1")
    secondary["releaseId"] = "release_secondary"
    secondary["tasks"] = []

    revision, instances = prepare_revision(
        [primary, secondary], 14, tmp_path / "runtime", tmp_path / "output"
    )

    pipelines = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (revision / "pipelines").glob("*.yaml")
    ]
    pipeline = next(item for item in pipelines if item["inputs"][0].endswith("/a"))
    assert pipeline["secondary_0"]["node"] == "SecondaryInferNode"
    assert pipeline["secondary_0"]["link_to"] == ["tracking"]
    assert pipeline["analytics"]["node"] == "AnalyticsNode"
    assert pipeline["analytics"]["link_to"] == ["secondary_0"]
    assert pipeline["events"]["node"] == "EventOutputNode"
    assert pipeline["events"]["link_to"] == ["analytics"]
    assert pipeline["result"]["link_to"] == ["events"]
    assert len(instances) == 2


def test_prepare_revision_shares_secondary_context_worker_pool(tmp_path: Path) -> None:
    primary = _release(tmp_path, adapter="yolo_dfl_split_v1")
    primary["releaseId"] = "release_primary_shared_secondary"
    tasks = primary["tasks"]
    assert isinstance(tasks, list)
    secondary_rule = {
        "releaseId": "release_secondary_shared",
        "sourceClassIds": [0],
        "confidenceThreshold": 0.3,
        "contextCount": 2,
        "workerCount": 1,
    }
    for task in tasks:
        assert isinstance(task, dict)
        task["analytics"] = {"secondaryModels": [secondary_rule]}
    secondary = _release(tmp_path, adapter="yolo_dfl_split_v1")
    secondary["releaseId"] = "release_secondary_shared"
    secondary["tasks"] = []

    revision, instances = prepare_revision(
        [primary, secondary], 22, tmp_path / "runtime", tmp_path / "output"
    )

    instance_config = json.loads((revision / "instances.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((revision / "revision.json").read_text(encoding="utf-8"))
    pipelines = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (revision / "pipelines").glob("*.yaml")
    ]
    secondary_names = {pipeline["secondary_0"]["instance"] for pipeline in pipelines}
    assert len(instances) == 2
    assert len(secondary_names) == 1
    secondary_instance = next(iter(secondary_names))
    assert instance_config[secondary_instance]["context_count"] == 2
    assert instance_config[secondary_instance]["worker_count"] == 1
    assert metadata["contextCount"] == 3


def test_media_contract_rejects_tracking_on_structured_model(tmp_path: Path) -> None:
    release = _release(tmp_path)
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    tasks[0]["media"] = {"tracking": {"enabled": True}}

    with pytest.raises(RuntimeAdapterError, match="tracking requires a detection adapter"):
        validate_release_configs([release])


def test_media_contract_requires_rkmpp_for_zlm_sei(tmp_path: Path) -> None:
    release = _release(tmp_path, adapter="yolo_dfl_split_v1")
    tasks = release["tasks"]
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], dict)
    tasks[0]["media"] = {
        "decoder": "opencv",
        "zlmSei": {"enabled": True, "publishUri": "rtsp://zlm/live/result"},
    }

    with pytest.raises(RuntimeAdapterError, match="ZLM SEI requires the RKMPP decoder"):
        validate_release_configs([release])


@pytest.mark.parametrize(
    "zlm, message",
    [
        (
            {"enabled": True, "publishUri": "rtsp://user:secret@zlm/live/result"},
            "without userinfo",
        ),
        (
            {
                "enabled": True,
                "publishUri": "rtsp://zlm/live/result",
                "reconnectMs": 999,
            },
            "between 1000 and 4000",
        ),
        (
            {
                "enabled": True,
                "publishUri": "rtsp://zlm/live/result",
                "reconnectMs": 4001,
            },
            "between 1000 and 4000",
        ),
    ],
)
def test_media_contract_rejects_invalid_node_publication_config(
    tmp_path: Path, zlm: dict[str, object], message: str
) -> None:
    release = _release(tmp_path, adapter="yolo_dfl_split_v1")
    tasks = release["tasks"]
    assert isinstance(tasks, list) and isinstance(tasks[0], dict)
    tasks[0]["media"] = {"decoder": "rkmpp", "zlmSei": zlm}

    with pytest.raises(RuntimeAdapterError, match=message):
        validate_release_configs([release])


def test_activate_health_and_empty_revision_stop(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "runtime"
    output_root = tmp_path / "output"
    pipeline = tmp_path / "fake-pipeline"
    pipeline.write_text(
        '#!/bin/sh\nprintf "ready\\n" > "$RKNODE_READY_FILE"\nexec sleep 30\n',
        encoding="utf-8",
    )
    pipeline.chmod(0o755)
    probe = tmp_path / "fake-probe"
    probe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    probe.chmod(0o755)
    monkeypatch.setenv("RKNODE_RUNTIME_STATE_DIR", str(state_root))
    monkeypatch.setenv("RKNODE_INFERENCE_OUTPUT_DIR", str(output_root))
    monkeypatch.setenv("RKNODE_PIPELINE_BINARY", str(pipeline))
    monkeypatch.setenv("RKNODE_INSTANCE_PROBE_BINARY", str(probe))
    monkeypatch.setenv("RKNODE_DESIRED_REVISION", "1")
    monkeypatch.setenv("RKNODE_RELEASE_CONFIGS", json.dumps([_release(tmp_path)]))

    try:
        command_activate()
        assert runtime_healthy(state_root)
        assert not (state_root / "logs" / "pipeline.log").exists()
        command_health()

        monkeypatch.setenv("RKNODE_DESIRED_REVISION", "2")
        monkeypatch.setenv("RKNODE_RELEASE_CONFIGS", "[]")
        command_activate()
        assert runtime_healthy(state_root)
        active = json.loads((state_root / "active.json").read_text(encoding="utf-8"))
        assert active == {"revision": 2, "empty": True}
    finally:
        stop_active(state_root)
