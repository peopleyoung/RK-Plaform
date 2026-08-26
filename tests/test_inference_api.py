from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from backend.platform_api.contracts import InferenceTaskStatus
from backend.platform_api.db_models import (
    ArtifactRecord,
    InferenceNodeRecord,
    InferenceTaskRecord,
    JobRecord,
    ModelReleaseRecord,
    NodeCleanupRecord,
    ServiceEndpointRecord,
)
from backend.platform_api.inference_graph import (
    GRAPH_CATALOG_VERSION,
    GRAPH_SCHEMA_VERSION,
    InferenceGraph,
    project_graph,
)
from backend.platform_api.service import new_id
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import select

from tests.conftest import ADMIN_HEADERS


def _graph_task_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "graph" in payload:
        return payload
    media = payload.get("media") if isinstance(payload.get("media"), dict) else {}
    analytics = (
        payload.get("analytics") if isinstance(payload.get("analytics"), dict) else {}
    )
    thresholds = (
        payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    )
    decoder = media.get("decoder", "opencv")
    nodes: list[dict[str, Any]] = [
        {
            "id": "capture",
            "operator": "capture.rkmpp" if decoder == "rkmpp" else "capture.opencv",
            "config": {},
        },
        {
            "id": "primary",
            "operator": "inference.primary",
            "config": {
                "releaseId": payload.get("releaseId", ""),
                "interval": payload.get("interval", 1),
                "confidence": thresholds.get("confidence", 0.4),
                "nms": thresholds.get("nms", 0.5),
                "contextCount": payload.get("contextCount", 1),
                "workerCount": payload.get("workerCount", 1),
            },
        },
    ]
    tracking = media.get("tracking") if isinstance(media.get("tracking"), dict) else {}
    if tracking.get("enabled") is True:
        nodes.append(
            {
                "id": "tracking",
                "operator": "processing.bytetrack",
                "config": {key: value for key, value in tracking.items() if key != "enabled"},
            }
        )
    secondary_models = (
        analytics.get("secondaryModels")
        if isinstance(analytics.get("secondaryModels"), list)
        else []
    )
    for index, value in enumerate(secondary_models):
        config = dict(value) if isinstance(value, dict) else {}
        if "confidenceThreshold" in config:
            config["confidence"] = config.pop("confidenceThreshold")
        nodes.append(
            {
                "id": f"secondary-{index + 1}",
                "operator": "inference.secondary",
                "config": config,
            }
        )
    analytics_config = {
        key: analytics[key] for key in ("areas", "lines", "osd") if key in analytics
    }
    if analytics_config or "events" in analytics:
        nodes.append(
            {
                "id": "analytics",
                "operator": "processing.analytics",
                "config": analytics_config,
            }
        )
    events = analytics.get("events")
    if isinstance(events, dict):
        nodes.append(
            {"id": "events", "operator": "processing.events", "config": dict(events)}
        )

    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    output_config = {"type": "jsonl", **output}
    outputs: list[dict[str, Any]] = [
        {"id": "json-output", "operator": "output.json", "config": output_config}
    ]
    kafka = media.get("kafka") if isinstance(media.get("kafka"), dict) else {}
    if kafka.get("enabled") is True:
        outputs.append(
            {
                "id": "kafka-output",
                "operator": "output.kafka",
                "config": {key: value for key, value in kafka.items() if key != "enabled"},
            }
        )
    zlm = media.get("zlmSei") if isinstance(media.get("zlmSei"), dict) else {}
    if zlm.get("enabled") is True:
        outputs.append(
            {
                "id": "zlm-output",
                "operator": "output.zlm_sei",
                "config": {
                    key: value
                    for key, value in zlm.items()
                    if key in {"gatewayId", "streamName", "reconnectMs"}
                },
            }
        )
    terminal = nodes[-1]["id"]
    edges = [
        {"source": nodes[index - 1]["id"], "target": node["id"]}
        for index, node in enumerate(nodes[1:], start=1)
    ]
    edges.extend({"source": terminal, "target": node["id"]} for node in outputs)
    graph_nodes = [*nodes, *outputs]
    result = {
        key: payload[key]
        for key in (
            "name",
            "nodeId",
            "groupId",
            "inputUri",
            "npuCoreMask",
            "npuCorePolicy",
        )
        if key in payload
    }
    result["graph"] = {
        "schemaVersion": GRAPH_SCHEMA_VERSION,
        "catalogVersion": GRAPH_CATALOG_VERSION,
        "nodes": graph_nodes,
        "edges": edges,
    }
    result["layout"] = {"positions": {}}
    return result


def _post_inference_task(
    client: TestClient,
    *,
    headers: dict[str, str],
    json: dict[str, Any],
) -> Response:
    return client.post(
        "/api/v1/inference-tasks",
        headers=headers,
        json=_graph_task_payload(json),
    )


def _post_deployment(
    client: TestClient,
    *,
    headers: dict[str, str],
    json: dict[str, Any],
) -> Response:
    payload = {key: value for key, value in json.items() if key != "releaseId"}
    return client.post("/api/v1/deployments", headers=headers, json=payload)


def _operator_config(payload: dict[str, Any], operator: str) -> dict[str, Any]:
    return next(
        node["config"] for node in payload["graph"]["nodes"] if node["operator"] == operator
    )


def _projection(payload: dict[str, Any]) -> dict[str, Any]:
    return project_graph(InferenceGraph.model_validate(payload["graph"])).model_dump(
        mode="json", by_alias=True
    )


def _manifest(training_job_id: str) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "modelFamily": "DeepLabV3+",
        "profileId": "deeplabv3plus",
        "variant": "mobilenet_v2_rknn",
        "taskType": "semantic_segmentation",
        "trainingJobId": training_job_id,
        "onnxSha256": "a" * 64,
        "opset": 12,
        "resolution": {"width": 512, "height": 512},
        "input": {
            "name": "images",
            "layout": "NCHW",
            "shape": [1, 3, 512, 512],
            "dtype": "float32",
            "colorSpace": "RGB",
        },
        "preprocessing": {
            "mean": [127.5, 127.5, 127.5],
            "std": [127.5, 127.5, 127.5],
        },
        "resizePolicy": "stretch",
        "outputContract": "semantic_logits_nchw_v1",
        "outputs": [{"name": "output", "semantic": "semantic_logits"}],
        "labels": ["background", "scratch"],
        "supportedPrecisions": ["int8", "fp16"],
        "rknn": {
            "targetPlatform": "rk3588",
            "quantizedAlgorithm": "normal",
            "optimizationLevel": 3,
            "requiresCalibrationFor": ["int8"],
        },
    }


def _seed_succeeded_conversion(client: TestClient) -> tuple[str, str]:
    context = client.app.state.context
    now = datetime.now(UTC)
    training_job_id = new_id("train")
    conversion_job_id = new_id("convert")
    source_id = new_id("artifact")
    rknn_id = new_id("artifact")
    validation_id = new_id("artifact")
    manifest = _manifest(training_job_id)
    onnx_bytes = b"onnx-model"
    rknn_bytes = b"rknn-model"
    validation_bytes = b'{"passed": true}'

    with context.database.session() as session:
        session.add(
            JobRecord(
                id=training_job_id,
                type="training",
                name="seed training",
                status="succeeded",
                profile_id="deeplabv3plus",
                dataset_id=None,
                worker_id=None,
                progress=100,
                stage="completed",
                spec_json={},
                result_json={},
                created_at=now,
                updated_at=now,
                completed_at=now,
            )
        )
        session.add(
            JobRecord(
                id=conversion_job_id,
                type="conversion",
                name="seed conversion",
                status="succeeded",
                profile_id="deeplabv3plus",
                dataset_id=None,
                worker_id=None,
                progress=100,
                stage="validated",
                spec_json={
                    "manifest": manifest,
                    "precision": "fp16",
                    "sourceArtifact": {"id": source_id},
                },
                result_json={
                    "deploymentReady": True,
                    "validation": {"passed": True, "inferenceSamples": 3},
                },
                created_at=now,
                updated_at=now,
                completed_at=now,
            )
        )
        for artifact_id, job_id, kind, filename, content, artifact_manifest in (
            (source_id, training_job_id, "onnx", "deeplab.onnx", onnx_bytes, manifest),
            (rknn_id, conversion_job_id, "rknn", "deeplab.rknn", rknn_bytes, None),
            (
                validation_id,
                conversion_job_id,
                "validation_report",
                "validation.json",
                validation_bytes,
                None,
            ),
        ):
            storage_key = f"artifacts/{artifact_id}/{filename}"
            path = context.storage.resolve(storage_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            session.add(
                ArtifactRecord(
                    id=artifact_id,
                    job_id=job_id,
                    kind=kind,
                    filename=filename,
                    storage_key=storage_key,
                    media_type="application/octet-stream",
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    manifest_json=artifact_manifest,
                    created_at=now,
                )
            )
    return conversion_job_id, rknn_id


def _register_active_node(
    client: TestClient,
    adapter: str,
    *,
    suffix: str = "01",
    features: list[str] | None = None,
) -> tuple[str, str]:
    created = client.post(
        "/api/v1/inference-nodes",
        headers=ADMIN_HEADERS,
        json={"name": f"rk3588-test-{suffix}", "maxModelInstances": 2},
    )
    assert created.status_code == 201, created.text
    node = created.json()
    registration = client.post(
        "/api/v1/inference-agent/register",
        json={
            "nodeId": node["id"],
            "registrationToken": node["registrationToken"],
            "hardwareId": f"rk3588-test-hw-{suffix}",
            "runtimeVersion": "rknn-runtime-2.3.2",
            "driverVersion": "rknpu2",
            "pipelineVersion": "test-pipeline",
            "adapters": [adapter],
            "metadata": {"features": features} if features is not None else {},
        },
    )
    assert registration.status_code == 201, registration.text
    access_token = registration.json()["accessToken"]
    heartbeat = client.post(
        f"/api/v1/inference-agent/nodes/{node['id']}/heartbeat",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"health": "healthy", "selfTestPassed": True, "actualRevision": 0},
    )
    assert heartbeat.status_code == 200, heartbeat.text
    approved = client.post(
        f"/api/v1/inference-nodes/{node['id']}/approve", headers=ADMIN_HEADERS
    )
    assert approved.status_code == 200, approved.text
    summary = client.get("/api/v1/inference-summary", headers=ADMIN_HEADERS)
    assert summary.status_code == 200, summary.text
    assert summary.json()["totalNodes"] >= 1
    assert summary.json()["onlineNodes"] == summary.json()["totalNodes"]
    return node["id"], access_token


def _published_release(client: TestClient) -> tuple[dict[str, Any], str]:
    conversion_job_id, artifact_id = _seed_succeeded_conversion(client)
    created = client.post(
        "/api/v1/model-releases",
        headers=ADMIN_HEADERS,
        json={
            "name": "deeplabv3plus-mobilenet-v2",
            "version": conversion_job_id,
            "conversionJobId": conversion_job_id,
        },
    )
    assert created.status_code == 201, created.text
    published = client.post(
        f"/api/v1/model-releases/{created.json()['id']}/publish",
        headers=ADMIN_HEADERS,
    )
    assert published.status_code == 200, published.text
    return published.json(), artifact_id


def _published_yolo_release(client: TestClient) -> tuple[dict[str, Any], str]:
    release, artifact_id = _published_release(client)
    context = client.app.state.context
    with context.database.session() as session:
        record = session.get(ModelReleaseRecord, release["id"])
        assert record is not None
        manifest = dict(record.manifest_json)
        manifest.update(
            {
                "profileId": "yolov8",
                "modelFamily": "YOLO",
                "taskType": "object_detection",
                "outputContract": "rknn_yolo_dfl_split_heads_v1",
                "labels": ["person", "helmet", "vehicle"],
            }
        )
        record.profile_id = "yolov8"
        record.task_type = "object_detection"
        record.adapter = "yolo_dfl_split_v1"
        record.manifest_json = manifest
    release.update(
        {
            "profileId": "yolov8",
            "taskType": "object_detection",
            "adapter": "yolo_dfl_split_v1",
            "manifest": manifest,
        }
    )
    return release, artifact_id


def test_release_node_task_deployment_and_agent_artifact_access(client: TestClient) -> None:
    release, artifact_id = _published_release(client)
    node_id, access_token = _register_active_node(client, "deeplab_logits_v1")

    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "line-a-segmentation",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/line-a",
            "npuCoreMask": "core1",
            "npuCorePolicy": "exclusive",
        },
    )
    assert task.status_code == 201, task.text
    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "line-a-rollout",
            "releaseId": release["id"],
            "taskIds": [task.json()["id"]],
            "strategy": "all_at_once",
        },
    )
    assert deployment.status_code == 201, deployment.text
    target = deployment.json()["targets"][0]
    assert target["desiredRevision"] == 1
    agent_headers = {"Authorization": f"Bearer {access_token}"}
    desired = client.get(
        f"/api/v1/inference-agent/nodes/{node_id}/desired", headers=agent_headers
    )
    assert desired.status_code == 200, desired.text
    assert desired.json()["releases"][0]["artifact"]["id"] == artifact_id
    assert desired.json()["releases"][0]["manifest"]["labels"] == ["background", "scratch"]
    assert desired.json()["tasks"][0]["npuCoreMask"] == "core1"
    assert desired.json()["tasks"][0]["npuCorePolicy"] == "exclusive"
    assert desired.json()["tasks"][0]["configRevision"] == target["desiredRevision"]

    states = ["downloading", "verifying", "staged", "draining", "activating", "warming", "healthy"]
    for state in states:
        report = client.post(
            f"/api/v1/inference-agent/nodes/{node_id}/targets/{target['id']}/status",
            headers=agent_headers,
            json={
                "revision": target["desiredRevision"],
                "state": state,
                "progress": 100 if state == "healthy" else 50,
                "stage": state,
            },
        )
        assert report.status_code == 200, report.text
    final_task = client.get(
        "/api/v1/inference-tasks?page=1&pageSize=20", headers=ADMIN_HEADERS
    )
    assert final_task.status_code == 200
    assert final_task.json()["items"][0]["status"] == "running"
    downloaded = client.get(
        f"/api/v1/inference-agent/nodes/{node_id}/artifacts/{artifact_id}/download",
        headers=agent_headers,
    )
    assert downloaded.status_code == 200
    assert downloaded.content == b"rknn-model"


def test_service_endpoint_reports_active_inference_task_load(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, _ = _register_active_node(client, "deeplab_logits_v1", suffix="load-count")
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "load-count-task",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/load-count",
        },
    )
    assert task.status_code == 201, task.text
    context = client.app.state.context
    with context.database.session() as session:
        endpoint_id = new_id("service")
        session.add(
            ServiceEndpointRecord(
                id=endpoint_id,
                name="load-count-inference",
                kind="inference",
                endpoint="http://127.0.0.1:11082",
                mode="direct",
                scheme="http",
                host="127.0.0.1",
                port=11082,
                accelerator="rk3588",
                capabilities_json=["deeplab_logits_v1"],
                enabled=True,
                token_configured=True,
                enrollment_status="enrolled",
                probe_status="online",
                remote_metadata_json={"maxConcurrency": 4, "activeJobs": 0},
                inference_node_id=node_id,
            )
        )
        record = session.get(InferenceTaskRecord, task.json()["id"])
        assert record is not None
        record.status = InferenceTaskStatus.RUNNING.value

    listed = client.get("/api/v1/service-endpoints", headers=ADMIN_HEADERS)
    assert listed.status_code == 200, listed.text
    endpoint = next(item for item in listed.json() if item["id"] == endpoint_id)
    assert endpoint["remoteMetadata"]["activeJobs"] == 1


def test_inference_task_context_worker_counts_round_trip_and_validate(
    client: TestClient,
) -> None:
    release, _ = _published_release(client)
    node_id, _ = _register_active_node(
        client, "deeplab_logits_v1", suffix="pool-contract"
    )

    default_task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "default-pool",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/default-pool",
        },
    )
    assert default_task.status_code == 201, default_task.text
    assert _operator_config(default_task.json(), "inference.primary")["contextCount"] == 1
    assert _operator_config(default_task.json(), "inference.primary")["workerCount"] == 1

    explicit_task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "parallel-pool",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/parallel-pool",
            "contextCount": 3,
            "workerCount": 2,
        },
    )
    assert explicit_task.status_code == 201, explicit_task.text
    assert _operator_config(explicit_task.json(), "inference.primary")["contextCount"] == 3
    assert _operator_config(explicit_task.json(), "inference.primary")["workerCount"] == 2

    for invalid_counts in (
        {"contextCount": 0, "workerCount": 1},
        {"contextCount": 1, "workerCount": 0},
        {"contextCount": 1, "workerCount": 2},
    ):
        invalid = _post_inference_task(
            client,
            headers=ADMIN_HEADERS,
            json={
                "name": "invalid-pool",
                "releaseId": release["id"],
                "nodeId": node_id,
                "inputUri": "rtsp://camera/invalid-pool",
                **invalid_counts,
            },
        )
        assert invalid.status_code == 409, invalid.text
        assert invalid.json()["error"]["code"] == "inference_graph_invalid"


def test_deployment_capacity_counts_primary_contexts(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, _ = _register_active_node(
        client, "deeplab_logits_v1", suffix="primary-context-capacity"
    )
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "oversized-primary-pool",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/oversized-primary-pool",
            "contextCount": 3,
            "workerCount": 2,
        },
    )
    assert task.status_code == 201, task.text

    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "oversized-primary-rollout",
            "releaseId": release["id"],
            "taskIds": [task.json()["id"]],
            "strategy": "all_at_once",
        },
    )

    assert deployment.status_code == 409, deployment.text
    assert deployment.json()["error"]["code"] == "inference_node_capacity_exceeded"
    assert deployment.json()["error"]["details"]["requiredContexts"] == 3
    assert deployment.json()["error"]["details"]["maxContexts"] == 2


def test_secondary_pool_counts_round_trip_and_consume_context_capacity(
    client: TestClient,
) -> None:
    primary, _ = _published_yolo_release(client)
    secondary, _ = _published_yolo_release(client)
    node_id, _ = _register_active_node(
        client,
        "yolo_dfl_split_v1",
        suffix="secondary-context-capacity",
        features=["bytetrack", "secondary_infer"],
    )
    analytics = {
        "secondaryModels": [
            {
                "releaseId": secondary["id"],
                "sourceClassIds": [0],
                "confidenceThreshold": 0.3,
                "contextCount": 2,
                "workerCount": 1,
            }
        ]
    }
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "secondary-context-pool",
            "releaseId": primary["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/secondary-context-pool",
            "media": {"tracking": {"enabled": True}},
            "analytics": analytics,
        },
    )
    assert task.status_code == 201, task.text
    secondary_config = _operator_config(task.json(), "inference.secondary")
    assert secondary_config["contextCount"] == 2
    assert secondary_config["workerCount"] == 1

    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "secondary-context-rollout",
            "releaseId": primary["id"],
            "taskIds": [task.json()["id"]],
            "strategy": "all_at_once",
        },
    )
    assert deployment.status_code == 409, deployment.text
    assert deployment.json()["error"]["code"] == "inference_node_capacity_exceeded"
    assert deployment.json()["error"]["details"]["requiredContexts"] == 3
    assert deployment.json()["error"]["details"]["maxContexts"] == 2

    invalid = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "invalid-secondary-context-pool",
            "releaseId": primary["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/invalid-secondary-context-pool",
            "media": {"tracking": {"enabled": True}},
            "analytics": {
                "secondaryModels": [
                    {
                        **analytics["secondaryModels"][0],
                        "contextCount": 1,
                        "workerCount": 2,
                    }
                ]
            },
        },
    )
    assert invalid.status_code == 409, invalid.text
    assert invalid.json()["error"]["code"] == "inference_graph_invalid"


def test_exclusive_npu_core_overlap_is_rejected_before_deployment(
    client: TestClient,
) -> None:
    release, _ = _published_release(client)
    node_id, _ = _register_active_node(client, "deeplab_logits_v1", suffix="core-conflict")
    context = client.app.state.context
    with context.database.session() as session:
        node = session.get(InferenceNodeRecord, node_id)
        assert node is not None
        node.max_model_instances = 3

    invalid = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "invalid-exclusive-auto",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/invalid",
            "npuCoreMask": "auto",
            "npuCorePolicy": "exclusive",
        },
    )
    assert invalid.status_code == 422, invalid.text

    first = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "exclusive-core-0-1",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/core-01",
            "npuCoreMask": "core0_1",
            "npuCorePolicy": "exclusive",
        },
    )
    assert first.status_code == 201, first.text
    first_deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "activate-exclusive-core-0-1",
            "releaseId": release["id"],
            "taskIds": [first.json()["id"]],
            "strategy": "all_at_once",
        },
    )
    assert first_deployment.status_code == 201, first_deployment.text

    second = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "shared-core-1",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/core-1",
            "npuCoreMask": "core1",
            "npuCorePolicy": "shared",
        },
    )
    assert second.status_code == 201, second.text
    conflict = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "overlapping-core-deployment",
            "releaseId": release["id"],
            "taskIds": [second.json()["id"]],
            "strategy": "all_at_once",
        },
    )
    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["error"]["code"] == "inference_npu_core_conflict"


def test_stopped_inference_task_restart_reuses_the_task_revision(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, access_token = _register_active_node(client, "deeplab_logits_v1")
    created = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "restartable-segmentation",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/restartable",
        },
    )
    assert created.status_code == 201, created.text
    task_id = created.json()["id"]
    assert created.json()["status"] == "draft"

    stopped_draft = client.post(
        f"/api/v1/inference-tasks/{task_id}/stop", headers=ADMIN_HEADERS
    )
    assert stopped_draft.status_code == 200, stopped_draft.text
    assert stopped_draft.json()["status"] == "stopped"
    assert stopped_draft.json()["configRevision"] == 1

    deployments_before = client.get(
        "/api/v1/deployments?page=1&pageSize=100", headers=ADMIN_HEADERS
    ).json()["total"]
    restarted = client.post(
        f"/api/v1/inference-tasks/{task_id}/restart", headers=ADMIN_HEADERS
    )
    assert restarted.status_code == 200, restarted.text
    assert restarted.json()["id"] == task_id
    assert restarted.json()["status"] == "deploying"
    assert restarted.json()["configRevision"] == 2
    deployments_after = client.get(
        "/api/v1/deployments?page=1&pageSize=100", headers=ADMIN_HEADERS
    ).json()["total"]
    assert deployments_after == deployments_before

    desired = client.get(
        f"/api/v1/inference-agent/nodes/{node_id}/desired",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert desired.status_code == 200, desired.text
    assert desired.json()["tasks"][0]["deploymentTargetId"] is None

    duplicate = client.post(
        f"/api/v1/inference-tasks/{task_id}/restart", headers=ADMIN_HEADERS
    )
    assert duplicate.status_code == 409, duplicate.text
    assert duplicate.json()["error"]["code"] == "inference_task_not_restartable"

    healthy = client.post(
        f"/api/v1/inference-agent/nodes/{node_id}/heartbeat",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "actualRevision": 2,
            "health": "healthy",
            "selfTestPassed": True,
        },
    )
    assert healthy.status_code == 200, healthy.text

    stopped = client.post(
        f"/api/v1/inference-tasks/{task_id}/stop", headers=ADMIN_HEADERS
    )
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["status"] == "stopped"
    assert stopped.json()["configRevision"] == 3
    desired = client.get(
        f"/api/v1/inference-agent/nodes/{node_id}/desired",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert desired.status_code == 200, desired.text
    assert desired.json()["tasks"] == []

    restarted_again = client.post(
        f"/api/v1/inference-tasks/{task_id}/restart", headers=ADMIN_HEADERS
    )
    assert restarted_again.status_code == 200, restarted_again.text
    assert restarted_again.json()["configRevision"] == 4


def test_failed_inference_task_restart_clears_previous_runtime_error(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, access_token = _register_active_node(client, "deeplab_logits_v1")
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "failed-restartable-task",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/failed-restartable",
        },
    ).json()
    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "failed-rollout",
            "releaseId": release["id"],
            "taskIds": [task["id"]],
            "strategy": "all_at_once",
        },
    ).json()
    target = deployment["targets"][0]
    failed = client.post(
        f"/api/v1/inference-agent/nodes/{node_id}/targets/{target['id']}/status",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "revision": target["desiredRevision"],
            "state": "failed",
            "progress": 40,
            "stage": "activating",
            "errorCode": "runtime_start_failed",
            "errorMessage": "runtime could not start",
        },
    )
    assert failed.status_code == 200, failed.text

    restarted = client.post(
        f"/api/v1/inference-tasks/{task['id']}/restart", headers=ADMIN_HEADERS
    )
    assert restarted.status_code == 200, restarted.text
    assert restarted.json()["id"] == task["id"]
    assert restarted.json()["configRevision"] == 2
    listed = client.get(
        "/api/v1/inference-tasks?page=1&pageSize=20", headers=ADMIN_HEADERS
    )
    restarted_task = next(
        item for item in listed.json()["items"] if item["id"] == task["id"]
    )
    assert restarted_task["status"] == "deploying"
    assert restarted_task["errorMessage"] is None

    heartbeat = client.post(
        f"/api/v1/inference-agent/nodes/{node_id}/heartbeat",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "actualRevision": 1,
            "failedRevision": 2,
            "health": "degraded",
            "selfTestPassed": True,
        },
    )
    assert heartbeat.status_code == 200, heartbeat.text
    listed = client.get(
        "/api/v1/inference-tasks?page=1&pageSize=20", headers=ADMIN_HEADERS
    )
    failed_task = next(
        item for item in listed.json()["items"] if item["id"] == task["id"]
    )
    assert failed_task["status"] == "failed"
    assert failed_task["errorMessage"] == "Revision 2 failed on the inference node"


def test_heartbeat_failure_closes_tracked_deployment_target(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, access_token = _register_active_node(client, "deeplab_logits_v1")
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "heartbeat-failed-rollout",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/heartbeat-failure",
        },
    )
    assert task.status_code == 201, task.text
    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "heartbeat-failed-deployment",
            "taskIds": [task.json()["id"]],
            "strategy": "all_at_once",
        },
    )
    assert deployment.status_code == 201, deployment.text
    target = deployment.json()["targets"][0]

    heartbeat = client.post(
        f"/api/v1/inference-agent/nodes/{node_id}/heartbeat",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "actualRevision": 0,
            "failedRevision": target["desiredRevision"],
            "health": "degraded",
            "selfTestPassed": True,
        },
    )
    assert heartbeat.status_code == 200, heartbeat.text

    listed = client.get(
        "/api/v1/inference-tasks?page=1&pageSize=20", headers=ADMIN_HEADERS
    )
    failed_task = next(
        item for item in listed.json()["items"] if item["id"] == task.json()["id"]
    )
    assert failed_task["status"] == "failed"
    assert failed_task["errorMessage"] == (
        f"Revision {target['desiredRevision']} failed on the inference node"
    )

    failed_deployment = client.get(
        f"/api/v1/deployments/{deployment.json()['id']}", headers=ADMIN_HEADERS
    )
    assert failed_deployment.status_code == 200, failed_deployment.text
    assert failed_deployment.json()["status"] == "failed"
    failed_target = failed_deployment.json()["targets"][0]
    assert failed_target["state"] == "failed"
    assert failed_target["errorCode"] == "revision_apply_failed"
    assert failed_target["stage"] == "failed"


def test_desired_state_promotes_labels_from_legacy_release_manifest(client: TestClient) -> None:
    release, _ = _published_release(client)
    context = client.app.state.context
    with context.database.session() as session:
        record = session.get(ModelReleaseRecord, release["id"])
        assert record is not None
        legacy_manifest = dict(record.manifest_json)
        legacy_manifest.pop("labels", None)
        record.manifest_json = legacy_manifest

    node_id, access_token = _register_active_node(client, "deeplab_logits_v1", suffix="legacy")
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "legacy-label-task",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/legacy",
        },
    )
    assert task.status_code == 201, task.text
    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "legacy-label-rollout",
            "releaseId": release["id"],
            "taskIds": [task.json()["id"]],
            "strategy": "all_at_once",
        },
    )
    assert deployment.status_code == 201, deployment.text
    desired = client.get(
        f"/api/v1/inference-agent/nodes/{node_id}/desired",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert desired.status_code == 200, desired.text
    assert desired.json()["releases"][0]["manifest"]["labels"] == ["background", "scratch"]


def test_model_release_prevents_source_job_deletion(client: TestClient) -> None:
    conversion_job_id, _ = _seed_succeeded_conversion(client)
    release = client.post(
        "/api/v1/model-releases",
        headers=ADMIN_HEADERS,
        json={"name": "retained-model", "version": "1", "conversionJobId": conversion_job_id},
    )
    assert release.status_code == 201, release.text
    blocked = client.delete(f"/api/v1/jobs/{conversion_job_id}", headers=ADMIN_HEADERS)
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["error"]["code"] == "job_artifacts_published"


def test_deprecated_model_release_can_be_deleted(client: TestClient) -> None:
    release, _ = _published_release(client)
    deprecated = client.post(
        f"/api/v1/model-releases/{release['id']}/deprecate", headers=ADMIN_HEADERS
    )
    assert deprecated.status_code == 200, deprecated.text

    deleted = client.delete(
        f"/api/v1/model-releases/{release['id']}", headers=ADMIN_HEADERS
    )
    assert deleted.status_code == 204, deleted.text
    listed = client.get(
        "/api/v1/model-releases?page=1&pageSize=100", headers=ADMIN_HEADERS
    )
    assert listed.status_code == 200, listed.text
    assert release["id"] not in {item["id"] for item in listed.json()["items"]}


def test_model_release_delete_requires_deprecated_unreferenced_version(
    client: TestClient,
) -> None:
    release, _ = _published_release(client)
    published = client.delete(
        f"/api/v1/model-releases/{release['id']}", headers=ADMIN_HEADERS
    )
    assert published.status_code == 409, published.text
    assert published.json()["error"]["code"] == "model_release_not_deletable"

    node_id, _ = _register_active_node(client, "deeplab_logits_v1", suffix="release-delete")
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "release-delete-reference",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/release-delete-reference",
        },
    )
    assert task.status_code == 201, task.text
    deprecated = client.post(
        f"/api/v1/model-releases/{release['id']}/deprecate", headers=ADMIN_HEADERS
    )
    assert deprecated.status_code == 200, deprecated.text
    referenced = client.delete(
        f"/api/v1/model-releases/{release['id']}", headers=ADMIN_HEADERS
    )
    assert referenced.status_code == 409, referenced.text
    assert referenced.json()["error"]["code"] == "model_release_in_use"


def test_inference_task_http_output_validation_and_update(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, _ = _register_active_node(client, "deeplab_logits_v1")
    base = {
        "name": "line-a-http",
        "releaseId": release["id"],
        "nodeId": node_id,
        "inputUri": "rtsp://camera/line-a",
    }

    invalid = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            **base,
            "output": {"type": "http", "url": "https://user:secret@consumer/results"},
        },
    )
    assert invalid.status_code == 409, invalid.text
    assert invalid.json()["error"]["code"] == "inference_graph_invalid"

    created = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            **base,
            "output": {
                "type": "http",
                "url": "https://consumer.example/results",
                "authorizationEnv": "RKNODE_RESULT_SINK_TOKEN",
                "connectTimeoutMs": 500,
                "requestTimeoutMs": 2000,
            },
        },
    )
    assert created.status_code == 201, created.text
    assert _operator_config(created.json(), "output.json")["type"] == "http"

    update_payload = _graph_task_payload(
        {**base, "name": "line-a-jsonl", "output": {"type": "jsonl"}}
    )
    update_payload["baseRevisionId"] = created.json()["graphRevisionId"]
    updated = client.put(
        f"/api/v1/inference-tasks/{created.json()['id']}",
        headers=ADMIN_HEADERS,
        json=update_payload,
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "line-a-jsonl"
    assert _operator_config(updated.json(), "output.json")["type"] == "jsonl"


def test_inference_task_media_validation_and_desired_state(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, access_token = _register_active_node(
        client,
        "deeplab_logits_v1",
        features=["rkmpp_decode", "bytetrack", "kafka", "zlm_sei"],
    )
    base = {
        "name": "line-a-media",
        "releaseId": release["id"],
        "nodeId": node_id,
        "inputUri": "rtsp://camera/line-a",
    }
    mismatch = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={**base, "media": {"tracking": {"enabled": True}}},
    )
    assert mismatch.status_code == 409, mismatch.text
    assert mismatch.json()["error"]["code"] == "inference_graph_invalid"
    assert {
        issue["code"] for issue in mismatch.json()["error"]["details"]["issues"]
    } == {"operator_adapter_mismatch"}

    media = {
        "decoder": "rkmpp",
        "kafka": {"enabled": True, "brokers": "kafka:9092", "topic": "sei_msg"},
    }
    created = _post_inference_task(
        client, headers=ADMIN_HEADERS, json={**base, "media": media}
    )
    assert created.status_code == 201, created.text
    assert next(
        node["operator"] for node in created.json()["graph"]["nodes"] if node["id"] == "capture"
    ) == "capture.rkmpp"
    assert _operator_config(created.json(), "output.kafka")["brokers"] == "kafka:9092"

    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "line-a-media-rollout",
            "releaseId": release["id"],
            "taskIds": [created.json()["id"]],
            "strategy": "all_at_once",
        },
    )
    assert deployment.status_code == 201, deployment.text
    desired = client.get(
        f"/api/v1/inference-agent/nodes/{node_id}/desired",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert desired.status_code == 200, desired.text
    assert desired.json()["tasks"][0]["graph"] == created.json()["graph"]


def test_inference_analytics_round_trip_and_secondary_release_delivery(
    client: TestClient,
) -> None:
    primary, _ = _published_yolo_release(client)
    secondary, secondary_artifact_id = _published_yolo_release(client)
    features = [
        "bytetrack",
        "analytics_area",
        "analytics_line",
        "event_snapshot",
        "secondary_infer",
    ]
    node_id, access_token = _register_active_node(
        client,
        "yolo_dfl_split_v1",
        suffix="analytics",
        features=features,
    )
    analytics = {
        "areas": [
            {
                "id": "loading-zone",
                "name": "Loading zone",
                "polygon": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.8, "y": 0.1},
                    {"x": 0.8, "y": 0.8},
                ],
                "classIds": [0],
                "minCount": 1,
                "holdFrames": 2,
            }
        ],
        "lines": [
            {
                "id": "gate-a",
                "name": "Gate A",
                "start": {"x": 0.2, "y": 0.5},
                "end": {"x": 0.8, "y": 0.5},
                "direction": "both",
                "classIds": [0],
            }
        ],
        "osd": {"enabled": True, "showTrackId": True},
        "events": {"enabled": True, "snapshot": True, "record": False},
        "secondaryModels": [
            {
                "releaseId": secondary["id"],
                "sourceClassIds": [0],
                "confidenceThreshold": 0.35,
            }
        ],
    }
    base = {
        "name": "analytics-task",
        "releaseId": primary["id"],
        "nodeId": node_id,
        "inputUri": "rtsp://camera/analytics",
        "media": {"tracking": {"enabled": True}},
        "analytics": analytics,
    }
    created = _post_inference_task(client, headers=ADMIN_HEADERS, json=base)
    assert created.status_code == 201, created.text
    created_projection = _projection(created.json())
    assert created_projection["analytics"]["secondaryModels"][0]["contextCount"] == 1
    assert created_projection["analytics"]["secondaryModels"][0]["workerCount"] == 1

    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "analytics-rollout",
            "releaseId": primary["id"],
            "taskIds": [created.json()["id"]],
            "strategy": "all_at_once",
        },
    )
    assert deployment.status_code == 201, deployment.text
    agent_headers = {"Authorization": f"Bearer {access_token}"}
    desired = client.get(
        f"/api/v1/inference-agent/nodes/{node_id}/desired", headers=agent_headers
    )
    assert desired.status_code == 200, desired.text
    assert {item["id"] for item in desired.json()["releases"]} == {
        primary["id"],
        secondary["id"],
    }
    assert desired.json()["tasks"][0]["graph"] == created.json()["graph"]
    secondary_download = client.get(
        f"/api/v1/inference-agent/nodes/{node_id}/artifacts/"
        f"{secondary_artifact_id}/download",
        headers=agent_headers,
    )
    assert secondary_download.status_code == 200, secondary_download.text


def test_inference_analytics_rejects_invalid_geometry_and_missing_runtime_requirements(
    client: TestClient,
) -> None:
    release, _ = _published_yolo_release(client)
    legacy_node_id, _ = _register_active_node(
        client, "yolo_dfl_split_v1", suffix="analytics-legacy"
    )
    invalid_geometry = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "invalid-area",
            "releaseId": release["id"],
            "nodeId": legacy_node_id,
            "inputUri": "rtsp://camera/invalid",
            "analytics": {
                "areas": [
                    {
                        "id": "area-a",
                        "polygon": [
                            {"x": -0.1, "y": 0.1},
                            {"x": 0.8, "y": 0.1},
                            {"x": 0.8, "y": 0.8},
                        ],
                    }
                ]
            },
        },
    )
    assert invalid_geometry.status_code == 409, invalid_geometry.text
    assert invalid_geometry.json()["error"]["code"] == "inference_graph_invalid"

    missing_tracking = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "missing-tracking",
            "releaseId": release["id"],
            "nodeId": legacy_node_id,
            "inputUri": "rtsp://camera/no-tracking",
            "analytics": {
                "lines": [
                    {
                        "id": "line-a",
                        "start": {"x": 0.1, "y": 0.5},
                        "end": {"x": 0.9, "y": 0.5},
                    }
                ]
            },
        },
    )
    assert missing_tracking.status_code == 409, missing_tracking.text
    assert missing_tracking.json()["error"]["code"] == "inference_graph_invalid"

    missing_feature = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "missing-feature",
            "releaseId": release["id"],
            "nodeId": legacy_node_id,
            "inputUri": "rtsp://camera/missing-feature",
            "media": {"tracking": {"enabled": True}},
            "analytics": {
                "lines": [
                    {
                        "id": "line-a",
                        "start": {"x": 0.1, "y": 0.5},
                        "end": {"x": 0.9, "y": 0.5},
                    }
                ]
            },
        },
    )
    assert missing_feature.status_code == 409, missing_feature.text
    assert missing_feature.json()["error"]["code"] == "inference_media_feature_missing"


def test_inference_task_rejects_media_feature_missing_on_legacy_node(
    client: TestClient,
) -> None:
    release, _ = _published_release(client)
    node_id, _ = _register_active_node(
        client, "deeplab_logits_v1", suffix="legacy-media"
    )

    response = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "legacy-node-rkmpp",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/legacy",
            "media": {"decoder": "rkmpp"},
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "inference_media_feature_missing"
    assert response.json()["error"]["details"]["missingFeatures"] == ["rkmpp_decode"]


def test_node_groups_support_update_and_guarded_delete(client: TestClient) -> None:
    created = client.post(
        "/api/v1/node-groups",
        headers=ADMIN_HEADERS,
        json={"name": "production-a", "description": "line a", "labels": ["plant-a"]},
    )
    assert created.status_code == 201, created.text
    group = created.json()

    updated = client.put(
        f"/api/v1/node-groups/{group['id']}",
        headers=ADMIN_HEADERS,
        json={"name": "production-main", "description": "main line", "labels": ["main"]},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "production-main"
    assert updated.json()["labels"] == ["main"]

    node = client.post(
        "/api/v1/inference-nodes",
        headers=ADMIN_HEADERS,
        json={"name": "grouped-board", "groupId": group["id"]},
    )
    assert node.status_code == 201, node.text
    blocked = client.delete(f"/api/v1/node-groups/{group['id']}", headers=ADMIN_HEADERS)
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["error"]["code"] == "node_group_not_empty"

    empty = client.post(
        "/api/v1/node-groups",
        headers=ADMIN_HEADERS,
        json={"name": "empty-group", "labels": []},
    ).json()
    deleted = client.delete(f"/api/v1/node-groups/{empty['id']}", headers=ADMIN_HEADERS)
    assert deleted.status_code == 200, deleted.text


def test_failed_deployment_can_be_retried_with_a_new_revision(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, access_token = _register_active_node(client, "deeplab_logits_v1")
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "retry-task",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/retry",
        },
    ).json()
    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "retry-rollout",
            "releaseId": release["id"],
            "taskIds": [task["id"]],
            "strategy": "all_at_once",
        },
    ).json()
    target = deployment["targets"][0]
    agent_headers = {"Authorization": f"Bearer {access_token}"}
    failed = client.post(
        f"/api/v1/inference-agent/nodes/{node_id}/targets/{target['id']}/status",
        headers=agent_headers,
        json={
            "revision": 1,
            "state": "failed",
            "progress": 0,
            "stage": "failed",
            "errorCode": "download_failed",
        },
    )
    assert failed.status_code == 200, failed.text

    retried = client.post(
        f"/api/v1/deployments/{deployment['id']}/retry", headers=ADMIN_HEADERS
    )
    assert retried.status_code == 200, retried.text
    retried_target = retried.json()["targets"][0]
    assert retried_target["desiredRevision"] == 2
    assert retried_target["state"] == "pending"

    stale = client.post(
        f"/api/v1/inference-agent/nodes/{node_id}/targets/{target['id']}/status",
        headers=agent_headers,
        json={"revision": 1, "state": "healthy", "progress": 100, "stage": "healthy"},
    )
    assert stale.status_code == 409, stale.text
    recovered = client.post(
        f"/api/v1/inference-agent/nodes/{node_id}/targets/{target['id']}/status",
        headers=agent_headers,
        json={"revision": 2, "state": "healthy", "progress": 100, "stage": "healthy"},
    )
    assert recovered.status_code == 200, recovered.text
    completed = client.get(
        f"/api/v1/deployments/{deployment['id']}", headers=ADMIN_HEADERS
    )
    assert completed.json()["status"] == "succeeded"


def test_canary_deployment_advances_one_node_at_a_time(client: TestClient) -> None:
    release, _ = _published_release(client)
    first_node, first_token = _register_active_node(
        client, "deeplab_logits_v1", suffix="canary"
    )
    second_node, second_token = _register_active_node(
        client, "deeplab_logits_v1", suffix="batch"
    )
    tasks: list[dict[str, Any]] = []
    for name, node_id in (("canary-task", first_node), ("batch-task", second_node)):
        response = _post_inference_task(
            client,
            headers=ADMIN_HEADERS,
            json={
                "name": name,
                "releaseId": release["id"],
                "nodeId": node_id,
                "inputUri": f"rtsp://camera/{name}",
            },
        )
        assert response.status_code == 201, response.text
        tasks.append(response.json())
    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "two-board-canary",
            "releaseId": release["id"],
            "taskIds": [item["id"] for item in tasks],
            "strategy": "canary",
            "batchSize": 1,
        },
    ).json()
    assert [item["desiredRevision"] for item in deployment["targets"]] == [1, 0]

    first_target = deployment["targets"][0]
    first_report = client.post(
        f"/api/v1/inference-agent/nodes/{first_node}/targets/{first_target['id']}/status",
        headers={"Authorization": f"Bearer {first_token}"},
        json={"revision": 1, "state": "healthy", "progress": 100, "stage": "healthy"},
    )
    assert first_report.status_code == 200, first_report.text
    advanced = client.get(
        f"/api/v1/deployments/{deployment['id']}", headers=ADMIN_HEADERS
    ).json()
    assert [item["desiredRevision"] for item in advanced["targets"]] == [1, 1]
    assert [item["state"] for item in advanced["targets"]] == ["healthy", "pending"]

    second_target = advanced["targets"][1]
    second_report = client.post(
        f"/api/v1/inference-agent/nodes/{second_node}/targets/{second_target['id']}/status",
        headers={"Authorization": f"Bearer {second_token}"},
        json={"revision": 1, "state": "healthy", "progress": 100, "stage": "healthy"},
    )
    assert second_report.status_code == 200, second_report.text
    completed = client.get(
        f"/api/v1/deployments/{deployment['id']}", headers=ADMIN_HEADERS
    )
    assert completed.json()["status"] == "succeeded"


def test_retired_inference_node_record_can_be_permanently_deleted(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/v1/inference-nodes",
        headers=ADMIN_HEADERS,
        json={"name": "retired-empty-board", "maxModelInstances": 1},
    )
    assert created.status_code == 201, created.text
    node_id = created.json()["id"]

    before_retirement = client.delete(
        f"/api/v1/inference-nodes/{node_id}/record", headers=ADMIN_HEADERS
    )
    assert before_retirement.status_code == 409, before_retirement.text
    assert before_retirement.json()["error"]["code"] == "inference_node_not_retired"

    retired = client.delete(
        f"/api/v1/inference-nodes/{node_id}", headers=ADMIN_HEADERS
    )
    assert retired.status_code == 200, retired.text
    assert retired.json()["lifecycle"] == "retired"

    deleted = client.delete(
        f"/api/v1/inference-nodes/{node_id}/record", headers=ADMIN_HEADERS
    )
    assert deleted.status_code == 204, deleted.text
    listed = client.get(
        "/api/v1/inference-nodes?page=1&pageSize=100", headers=ADMIN_HEADERS
    )
    assert listed.status_code == 200, listed.text
    assert all(item["id"] != node_id for item in listed.json()["items"])


def test_retired_node_delete_cleans_retired_tasks_service_and_secrets(
    client: TestClient,
) -> None:
    release, _ = _published_release(client)
    node_id, _ = _register_active_node(
        client, "deeplab_logits_v1", suffix="cascade-delete"
    )
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "retired-cascade-task",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/retired-cascade",
        },
    )
    assert task.status_code == 201, task.text
    task_id = task.json()["id"]
    retired_task = client.delete(
        f"/api/v1/inference-tasks/{task_id}", headers=ADMIN_HEADERS
    )
    assert retired_task.status_code == 200, retired_task.text

    endpoint_id = new_id("service")
    context = client.app.state.context
    with context.database.session() as session:
        session.add(
            ServiceEndpointRecord(
                id=endpoint_id,
                name="retired-cascade-service",
                kind="inference",
                endpoint="http://127.0.0.1:11082",
                mode="direct",
                scheme="http",
                host="127.0.0.1",
                port=11082,
                accelerator="rk3588",
                capabilities_json=["deeplab_logits_v1"],
                enabled=False,
                token_configured=True,
                inference_node_id=node_id,
            )
        )
        session.add(NodeCleanupRecord(endpoint_id=endpoint_id, job_id="cleanup-job"))
    context.node_secrets.write(endpoint_id, "node-secret")
    context.node_secrets.write(endpoint_id, "agent-secret", purpose="agent")

    retired_node = client.delete(
        f"/api/v1/inference-nodes/{node_id}", headers=ADMIN_HEADERS
    )
    assert retired_node.status_code == 200, retired_node.text
    deleted = client.delete(
        f"/api/v1/inference-nodes/{node_id}/record", headers=ADMIN_HEADERS
    )
    assert deleted.status_code == 204, deleted.text

    with context.database.session() as session:
        assert session.get(InferenceNodeRecord, node_id) is None
        assert session.get(InferenceTaskRecord, task_id) is None
        assert session.get(ServiceEndpointRecord, endpoint_id) is None
        assert session.scalar(
            select(NodeCleanupRecord).where(
                NodeCleanupRecord.endpoint_id == endpoint_id
            )
        ) is None
    assert context.node_secrets.read(endpoint_id) is None
    assert context.node_secrets.read(endpoint_id, purpose="agent") is None


def test_retired_node_delete_rejects_non_retired_tasks(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, _ = _register_active_node(
        client, "deeplab_logits_v1", suffix="delete-task-guard"
    )
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "still-configured-task",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/still-configured",
        },
    )
    assert task.status_code == 201, task.text
    retired_node = client.delete(
        f"/api/v1/inference-nodes/{node_id}", headers=ADMIN_HEADERS
    )
    assert retired_node.status_code == 200, retired_node.text

    blocked = client.delete(
        f"/api/v1/inference-nodes/{node_id}/record", headers=ADMIN_HEADERS
    )
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["error"]["code"] == "inference_node_has_tasks"
    assert blocked.json()["error"]["details"]["nonRetiredTaskCount"] == 1


def test_retired_node_delete_preserves_deployment_history(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, access_token = _register_active_node(
        client, "deeplab_logits_v1", suffix="delete-history-guard"
    )
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "deployment-history-task",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/history-guard",
        },
    )
    assert task.status_code == 201, task.text
    task_id = task.json()["id"]
    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "deployment-history-batch",
            "releaseId": release["id"],
            "taskIds": [task_id],
            "strategy": "all_at_once",
        },
    )
    assert deployment.status_code == 201, deployment.text
    deployment_id = deployment.json()["id"]
    target = deployment.json()["targets"][0]
    healthy = client.post(
        f"/api/v1/inference-agent/nodes/{node_id}/targets/{target['id']}/status",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "revision": target["desiredRevision"],
            "state": "healthy",
            "progress": 100,
            "stage": "healthy",
        },
    )
    assert healthy.status_code == 200, healthy.text
    assert client.post(
        f"/api/v1/inference-tasks/{task_id}/stop", headers=ADMIN_HEADERS
    ).status_code == 200
    assert client.delete(
        f"/api/v1/inference-tasks/{task_id}", headers=ADMIN_HEADERS
    ).status_code == 200
    assert client.delete(
        f"/api/v1/inference-nodes/{node_id}", headers=ADMIN_HEADERS
    ).status_code == 200

    blocked = client.delete(
        f"/api/v1/inference-nodes/{node_id}/record", headers=ADMIN_HEADERS
    )
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["error"]["code"] == "inference_node_has_history"
    assert blocked.json()["error"]["details"]["deploymentTargetCount"] == 1
    assert blocked.json()["error"]["details"]["deploymentEventCount"] > 0

    removed_deployment = client.delete(
        f"/api/v1/deployments/{deployment_id}", headers=ADMIN_HEADERS
    )
    assert removed_deployment.status_code == 204, removed_deployment.text
    deleted_node = client.delete(
        f"/api/v1/inference-nodes/{node_id}/record", headers=ADMIN_HEADERS
    )
    assert deleted_node.status_code == 204, deleted_node.text


def test_completed_deployment_can_be_deleted_with_targets_and_events(
    client: TestClient,
) -> None:
    release, _ = _published_release(client)
    node_id, access_token = _register_active_node(
        client, "deeplab_logits_v1", suffix="delete-deployment"
    )
    task = _post_inference_task(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "deployment-delete-task",
            "releaseId": release["id"],
            "nodeId": node_id,
            "inputUri": "rtsp://camera/deployment-delete",
        },
    )
    assert task.status_code == 201, task.text
    deployment = _post_deployment(
        client,
        headers=ADMIN_HEADERS,
        json={
            "name": "completed-delete-rollout",
            "releaseId": release["id"],
            "taskIds": [task.json()["id"]],
            "strategy": "all_at_once",
        },
    )
    assert deployment.status_code == 201, deployment.text
    deployment_id = deployment.json()["id"]
    target = deployment.json()["targets"][0]

    active_delete = client.delete(
        f"/api/v1/deployments/{deployment_id}", headers=ADMIN_HEADERS
    )
    assert active_delete.status_code == 409, active_delete.text
    assert active_delete.json()["error"]["code"] == "deployment_active"

    healthy = client.post(
        f"/api/v1/inference-agent/nodes/{node_id}/targets/{target['id']}/status",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "revision": target["desiredRevision"],
            "state": "healthy",
            "progress": 100,
            "stage": "healthy",
        },
    )
    assert healthy.status_code == 200, healthy.text
    events = client.get(
        f"/api/v1/deployments/{deployment_id}/events", headers=ADMIN_HEADERS
    )
    assert events.status_code == 200, events.text
    assert events.json()

    deleted = client.delete(
        f"/api/v1/deployments/{deployment_id}", headers=ADMIN_HEADERS
    )
    assert deleted.status_code == 204, deleted.text
    missing = client.get(
        f"/api/v1/deployments/{deployment_id}", headers=ADMIN_HEADERS
    )
    assert missing.status_code == 404, missing.text
    missing_events = client.get(
        f"/api/v1/deployments/{deployment_id}/events", headers=ADMIN_HEADERS
    )
    assert missing_events.status_code == 404, missing_events.text
