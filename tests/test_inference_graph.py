from __future__ import annotations

from backend.platform_api.inference_graph import (
    GRAPH_CATALOG_VERSION,
    GRAPH_SCHEMA_VERSION,
    InferenceGraph,
    catalog_response,
    graph_hash,
    graph_validation_response,
)
from fastapi.testclient import TestClient

from tests.conftest import ADMIN_HEADERS
from tests.test_inference_api import _published_release, _register_active_node


def _minimal_graph(
    *,
    capture_id: str = "capture",
    primary_id: str = "primary",
    output_id: str = "output",
) -> InferenceGraph:
    return InferenceGraph.model_validate(
        {
            "schemaVersion": GRAPH_SCHEMA_VERSION,
            "catalogVersion": GRAPH_CATALOG_VERSION,
            "nodes": [
                {"id": capture_id, "operator": "capture.opencv", "config": {}},
                {
                    "id": primary_id,
                    "operator": "inference.primary",
                    "config": {"releaseId": "release-1"},
                },
                {"id": output_id, "operator": "output.json", "config": {}},
            ],
            "edges": [
                {"source": capture_id, "target": primary_id},
                {"source": primary_id, "target": output_id},
            ],
        }
    )


def test_catalog_matches_the_registered_runtime_operator_set() -> None:
    response = catalog_response()

    assert response.schema_version == GRAPH_SCHEMA_VERSION
    assert response.catalog_version == GRAPH_CATALOG_VERSION
    assert {operator.runtime_node for operator in response.operators} == {
        "VideoCaptureNode",
        "RkMppCaptureNode",
        "InferNode",
        "ByteTrackNode",
        "SecondaryInferNode",
        "AnalyticsNode",
        "EventOutputNode",
        "JsonOutputNode",
        "KafkaOutputNode",
        "ZlmSeiOutputNode",
    }
    adapters = {
        adapter for operator in response.operators for adapter in operator.supported_adapters
    }
    assert adapters == {
        "yolo_dfl_split_v1",
        "deeplab_logits_v1",
        "ppocr_db_det_v1",
        "ppocr_ctc_rec_v1",
    }
    assert "V5" not in adapters
    assert "ByteTrack" not in adapters


def test_validation_fills_a_complete_default_snapshot() -> None:
    response = graph_validation_response(
        _minimal_graph(),
        release_adapters={"release-1": "deeplab_logits_v1"},
    )

    assert response.valid is True
    assert response.normalized_graph is not None
    primary = next(
        node for node in response.normalized_graph.nodes if node.operator == "inference.primary"
    )
    output = next(
        node for node in response.normalized_graph.nodes if node.operator == "output.json"
    )
    assert primary.config == {
        "releaseId": "release-1",
        "interval": 1,
        "confidence": 0.4,
        "nms": 0.5,
        "contextCount": 1,
        "workerCount": 1,
    }
    assert output.config["type"] == "jsonl"
    assert response.required_adapters == ["deeplab_logits_v1"]
    assert response.required_contexts == 1


def test_semantic_hash_ignores_client_node_ids_and_input_order() -> None:
    first = _minimal_graph()
    second = _minimal_graph(
        capture_id="source-renamed",
        primary_id="model-renamed",
        output_id="sink-renamed",
    )
    second.nodes.reverse()
    second.edges.reverse()

    assert graph_hash(first) == graph_hash(second)


def test_structured_primary_rejects_yolo_only_operators() -> None:
    graph = InferenceGraph.model_validate(
        {
            "nodes": [
                {"id": "capture", "operator": "capture.opencv"},
                {
                    "id": "primary",
                    "operator": "inference.primary",
                    "config": {"releaseId": "release-1"},
                },
                {"id": "tracker", "operator": "processing.bytetrack"},
                {"id": "output", "operator": "output.json"},
            ],
            "edges": [
                {"source": "capture", "target": "primary"},
                {"source": "primary", "target": "tracker"},
                {"source": "tracker", "target": "output"},
            ],
        }
    )

    response = graph_validation_response(graph, release_adapters={"release-1": "ppocr_db_det_v1"})

    assert response.valid is False
    assert "operator_adapter_mismatch" in {issue.code for issue in response.issues}


def test_validation_rejects_invalid_operator_parameters_before_projection() -> None:
    graph = _minimal_graph()
    primary = next(node for node in graph.nodes if node.operator == "inference.primary")
    primary.config.update({"interval": "every-frame", "confidence": 1.5, "nms": False})

    response = graph_validation_response(
        graph,
        release_adapters={"release-1": "deeplab_logits_v1"},
    )

    codes = {issue.code for issue in response.issues}
    assert response.valid is False
    assert {"interval_invalid", "confidence_invalid", "nms_invalid"}.issubset(codes)


def test_validation_requires_complete_kafka_and_zlm_destinations() -> None:
    graph = _minimal_graph()
    graph.nodes[0].operator = "capture.rkmpp"
    graph.nodes[-1].operator = "output.kafka"
    graph.nodes.append(
        type(graph.nodes[0]).model_validate(
            {"id": "zlm", "operator": "output.zlm_sei", "config": {}}
        )
    )
    graph.edges.append(
        type(graph.edges[0]).model_validate({"source": "primary", "target": "zlm"})
    )

    response = graph_validation_response(
        graph,
        release_adapters={"release-1": "deeplab_logits_v1"},
    )

    codes = {issue.code for issue in response.issues}
    assert response.valid is False
    assert "kafka_destination_invalid" in codes
    assert "zlm_gateway_invalid" in codes
    assert "zlm_stream_invalid" in codes


def test_graph_rejects_multiple_inputs_and_invalid_operator_order() -> None:
    graph = InferenceGraph.model_validate(
        {
            "nodes": [
                {"id": "capture", "operator": "capture.rkmpp"},
                {
                    "id": "primary",
                    "operator": "inference.primary",
                    "config": {"releaseId": "release-1"},
                },
                {"id": "analytics", "operator": "processing.analytics"},
                {"id": "output", "operator": "output.kafka"},
            ],
            "edges": [
                {"source": "capture", "target": "primary"},
                {"source": "primary", "target": "analytics"},
                {"source": "primary", "target": "output"},
                {"source": "analytics", "target": "output"},
            ],
        }
    )

    response = graph_validation_response(graph, release_adapters={"release-1": "yolo_dfl_split_v1"})

    codes = {issue.code for issue in response.issues}
    assert response.valid is False
    assert "graph_multiple_inputs" in codes
    assert "operator_order_invalid" in codes


def test_catalog_and_validation_endpoints_use_camel_case_contract(
    client: TestClient,
) -> None:
    catalog = client.get("/api/v1/inference-operator-catalog", headers=ADMIN_HEADERS)

    assert catalog.status_code == 200, catalog.text
    assert catalog.json()["schemaVersion"] == GRAPH_SCHEMA_VERSION
    assert len(catalog.json()["operators"]) == 10
    assert catalog.json()["operators"][0]["operatorId"] == "capture.opencv"

    validation = client.post(
        "/api/v1/inference-graphs/validate",
        headers=ADMIN_HEADERS,
        json={"graph": _minimal_graph().model_dump(mode="json", by_alias=True)},
    )

    assert validation.status_code == 200, validation.text
    body = validation.json()
    assert body["valid"] is False
    assert body["releaseIds"] == ["release-1"]
    assert {issue["code"] for issue in body["issues"]} == {"release_unavailable"}


def test_validation_endpoint_checks_task_input_and_returns_parameter_issues(
    client: TestClient,
) -> None:
    release, _ = _published_release(client)
    graph = _minimal_graph()
    graph.nodes[0].operator = "capture.rkmpp"
    primary = next(node for node in graph.nodes if node.operator == "inference.primary")
    primary.config.update({"releaseId": release["id"], "interval": "invalid"})

    validation = client.post(
        "/api/v1/inference-graphs/validate",
        headers=ADMIN_HEADERS,
        json={
            "graph": graph.model_dump(mode="json", by_alias=True),
            "inputUri": "/data/video.mp4",
        },
    )

    assert validation.status_code == 200, validation.text
    assert validation.json()["valid"] is False
    assert "interval_invalid" in {
        issue["code"] for issue in validation.json()["issues"]
    }

    create = client.post(
        "/api/v1/inference-tasks",
        headers=ADMIN_HEADERS,
        json={
            "name": "invalid-graph",
            "nodeId": "inode_missing",
            "inputUri": "/data/video.mp4",
            "graph": graph.model_dump(mode="json", by_alias=True),
        },
    )
    assert create.status_code == 409, create.text
    assert create.json()["error"]["code"] == "inference_graph_invalid"


def test_validation_endpoint_rejects_non_rtsp_mpp_input(client: TestClient) -> None:
    release, _ = _published_release(client)
    graph = _minimal_graph()
    graph.nodes[0].operator = "capture.rkmpp"
    primary = next(node for node in graph.nodes if node.operator == "inference.primary")
    primary.config["releaseId"] = release["id"]

    validation = client.post(
        "/api/v1/inference-graphs/validate",
        headers=ADMIN_HEADERS,
        json={
            "graph": graph.model_dump(mode="json", by_alias=True),
            "inputUri": "/data/video.mp4",
        },
    )

    assert validation.status_code == 200, validation.text
    assert validation.json()["valid"] is False
    assert "rkmpp_input_invalid" in {
        issue["code"] for issue in validation.json()["issues"]
    }


def test_graph_task_revision_and_deployment_snapshot_flow(client: TestClient) -> None:
    release, _ = _published_release(client)
    node_id, access_token = _register_active_node(
        client, "deeplab_logits_v1", suffix="graph-contract"
    )
    graph = _minimal_graph().model_copy(deep=True)
    primary = next(node for node in graph.nodes if node.operator == "inference.primary")
    primary.config["releaseId"] = release["id"]

    created = client.post(
        "/api/v1/inference-tasks",
        headers=ADMIN_HEADERS,
        json={
            "name": "graph-task",
            "nodeId": node_id,
            "inputUri": "rtsp://camera/graph-task",
            "graph": graph.model_dump(mode="json", by_alias=True),
            "layout": {
                "positions": {
                    "capture": {"x": 40, "y": 80},
                    "primary": {"x": 280, "y": 80},
                    "output": {"x": 520, "y": 80},
                }
            },
        },
    )

    assert created.status_code == 201, created.text
    task = created.json()
    assert task["status"] == "draft"
    assert "releaseId" not in task
    task_primary = next(
        node
        for node in task["graph"]["nodes"]
        if node["operator"] == "inference.primary"
    )
    assert task_primary["config"]["contextCount"] == 1

    revisions = client.get(
        f"/api/v1/inference-tasks/{task['id']}/graph-revisions",
        headers=ADMIN_HEADERS,
    )
    assert revisions.status_code == 200, revisions.text
    assert [item["revision"] for item in revisions.json()] == [1]

    primary.config["interval"] = 2
    updated = client.put(
        f"/api/v1/inference-tasks/{task['id']}",
        headers=ADMIN_HEADERS,
        json={
            "name": "graph-task",
            "nodeId": node_id,
            "inputUri": "rtsp://camera/graph-task",
            "graph": graph.model_dump(mode="json", by_alias=True),
            "layout": task["layout"],
            "baseRevisionId": task["graphRevisionId"],
        },
    )
    assert updated.status_code == 200, updated.text
    changed = updated.json()
    assert changed["graphRevisionId"] != task["graphRevisionId"]

    stale = client.put(
        f"/api/v1/inference-tasks/{task['id']}",
        headers=ADMIN_HEADERS,
        json={
            "name": "stale-edit",
            "nodeId": node_id,
            "inputUri": "rtsp://camera/graph-task",
            "graph": graph.model_dump(mode="json", by_alias=True),
            "baseRevisionId": task["graphRevisionId"],
        },
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()["error"]["code"] == "graph_revision_conflict"

    deployment = client.post(
        "/api/v1/deployments",
        headers=ADMIN_HEADERS,
        json={
            "name": "graph-rollout",
            "taskIds": [task["id"]],
            "strategy": "all_at_once",
        },
    )
    assert deployment.status_code == 201, deployment.text
    target = deployment.json()["targets"][0]
    assert target["graphRevisionId"] == changed["graphRevisionId"]
    assert target["graphHash"] == changed["graphHash"]

    desired = client.get(
        f"/api/v1/inference-agent/nodes/{node_id}/desired",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert desired.status_code == 200, desired.text
    descriptor = desired.json()["tasks"][0]
    assert descriptor["graphRevisionId"] == target["graphRevisionId"]
    descriptor_primary = next(
        node
        for node in descriptor["graph"]["nodes"]
        if node["operator"] == "inference.primary"
    )
    assert descriptor_primary["config"]["interval"] == 2
