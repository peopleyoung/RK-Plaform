from __future__ import annotations

import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from backend.platform_api.context import AppContext
from backend.platform_api.direct_dispatcher import DirectNodeDispatcher
from fastapi.testclient import TestClient

from .conftest import ADMIN_HEADERS, zip_dataset_bytes

NODE_TOKEN = "direct-node-test-token-with-32-characters"


@dataclass
class NodeState:
    kind: str
    accelerator: str
    capabilities: list[str]
    node_token: str = NODE_TOKEN
    max_concurrency: int = 1
    dispatched_job_ids: list[str] = field(default_factory=list)
    inference_status: dict[str, object] = field(
        default_factory=lambda: {"configured": False, "actualRevision": 0}
    )


def handler_for(state: NodeState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if not self._authorized():
                return
            if self.path != "/health":
                self.send_error(404)
                return
            self._json(
                200,
                {
                    "status": "healthy",
                    "protocolVersion": "1.0",
                    "name": f"direct-{state.kind}-test",
                    "kind": state.kind,
                    "accelerator": state.accelerator,
                    "capabilities": state.capabilities,
                    "version": "test-1.0",
                    "maxConcurrency": state.max_concurrency,
                    "activeJobs": 0,
                    "diagnostics": {
                        "inference": state.inference_status,
                        "inferenceSelfTestPassed": True,
                    },
                },
            )

        def do_POST(self) -> None:
            if not self._authorized():
                return
            prefix = "/api/v1/jobs/"
            suffix = "/dispatch"
            if not self.path.startswith(prefix) or not self.path.endswith(suffix):
                self.send_error(404)
                return
            job_id = self.path[len(prefix) : -len(suffix)]
            state.dispatched_job_ids.append(job_id)
            self._json(202, {"jobId": job_id, "state": "accepted", "accepted": True})

        def log_message(self, format: str, *args: object) -> None:
            _ = format, args

        def _authorized(self) -> bool:
            if self.headers.get("Authorization") == f"Bearer {state.node_token}":
                return True
            self._json(401, {"detail": "invalid node token"})
            return False

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


@contextmanager
def running_node(state: NodeState) -> Generator[tuple[str, int]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def inference_payload(host: str, port: int, token: str = NODE_TOKEN) -> dict[str, object]:
    return {
        "name": "direct-inference-test",
        "kind": "inference",
        "mode": "direct",
        "scheme": "http",
        "host": host,
        "port": port,
        "accelerator": "rk3588",
        "capabilities": ["deeplab_logits_v1"],
        "enabled": True,
        "token": token,
    }


def test_direct_inference_endpoint_is_probed_and_creates_internal_node(
    client: TestClient,
) -> None:
    state = NodeState(
        kind="inference",
        accelerator="rk3588",
        capabilities=["deeplab_logits_v1"],
        max_concurrency=3,
        inference_status={
            "configured": True,
            "actualRevision": 0,
            "runtimeVersion": "rknn-runtime-test",
            "driverVersion": "rknpu-test",
            "pipelineVersion": "pipeline-test",
        },
    )
    with running_node(state) as (host, port):
        tested = client.post(
            "/api/v1/service-endpoints/test",
            headers=ADMIN_HEADERS,
            json=inference_payload(host, port),
        )
        assert tested.status_code == 200, tested.text
        assert tested.json()["remote"]["maxConcurrency"] == 3

        created = client.post(
            "/api/v1/service-endpoints",
            headers=ADMIN_HEADERS,
            json=inference_payload(host, port),
        )
        assert created.status_code == 201, created.text
        endpoint = created.json()
        assert endpoint["endpoint"] == f"http://{host}:{port}"
        assert endpoint["probeStatus"] == "online"
        assert endpoint["tokenConfigured"] is True
        assert endpoint["inferenceNodeId"].startswith("inode_")
        assert "token" not in endpoint

        nodes = client.get("/api/v1/inference-nodes", headers=ADMIN_HEADERS).json()
        direct_node = next(
            item for item in nodes["items"] if item["id"] == endpoint["inferenceNodeId"]
        )
        assert direct_node["lifecycle"] == "active"
        assert direct_node["connectivity"] == "online"
        assert direct_node["adapters"] == ["deeplab_logits_v1"]
        assert direct_node["runtimeVersion"] == "rknn-runtime-test"
        assert direct_node["driverVersion"] == "rknpu-test"
        assert direct_node["pipelineVersion"] == "pipeline-test"
        assert direct_node["selfTestPassed"] is True


def test_pending_direct_endpoint_is_not_probed(client: TestClient) -> None:
    response = client.post(
        "/api/v1/service-endpoints",
        headers=ADMIN_HEADERS,
        json={
            "name": "direct-trainer-pending",
            "kind": "trainer",
            "mode": "direct",
            "scheme": "http",
            "host": "127.0.0.1",
            "port": 18991,
            "accelerator": "cpu",
            "capabilities": ["yolo-detect"],
            "enabled": True,
        },
    )
    assert response.status_code == 201, response.text

    DirectNodeDispatcher(cast(AppContext, client.app.state.context)).run_once()

    endpoint = client.get("/api/v1/service-endpoints", headers=ADMIN_HEADERS).json()[0]
    assert endpoint["enrollmentStatus"] == "pending"
    assert endpoint["probeStatus"] == "unprobed"
    assert endpoint["lastError"] is None

    manual = client.post(
        f"/api/v1/service-endpoints/{endpoint['id']}/probe",
        headers=ADMIN_HEADERS,
    )
    assert manual.status_code == 200, manual.text
    assert manual.json()["probeStatus"] == "unprobed"
    assert manual.json()["enrollmentStatus"] == "pending"


def test_claimed_trainer_activates_before_dispatching_next_iteration(
    client: TestClient,
    detection_dataset: dict[str, object],
) -> None:
    state = NodeState(
        kind="trainer",
        accelerator="cpu",
        capabilities=["yolo-detect"],
        node_token="not-claimed-yet-token",
    )
    with running_node(state) as (host, port):
        created = client.post(
            "/api/v1/service-endpoints",
            headers=ADMIN_HEADERS,
            json={
                "name": "direct-trainer-test",
                "kind": "trainer",
                "mode": "direct",
                "scheme": "http",
                "host": host,
                "port": port,
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
                "enabled": True,
            },
        )
        assert created.status_code == 201, created.text
        endpoint = created.json()
        claimed = client.post(
            f"/api/v1/node-enrollments/{endpoint['id']}/claim",
            json={
                "enrollmentToken": endpoint["enrollmentToken"],
                "name": "direct-trainer-test",
                "kind": "trainer",
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
                "version": "test-1.0",
                "maxConcurrency": 1,
            },
        )
        assert claimed.status_code == 200, claimed.text
        state.node_token = claimed.json()["nodeToken"]
        node_headers = {"Authorization": f"Bearer {state.node_token}"}
        premature_registration = client.post(
            "/api/v1/workers/register",
            headers=node_headers,
            json={
                "name": "direct-trainer-test",
                "kind": "trainer",
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
                "version": "test-1.0",
                "maxConcurrency": 1,
            },
        )
        assert premature_registration.status_code == 401
        job = client.post(
            "/api/v1/training-jobs",
            headers=ADMIN_HEADERS,
            json={
                "name": "Activation-gated dispatch",
                "datasetId": detection_dataset["id"],
                "profileId": "yolo-detect",
                "variant": "yolov8n",
                "resolution": {"width": 640, "height": 640},
                "hyperparameters": {"epochs": 1, "batchSize": 1},
                "accelerator": "cpu",
            },
        )
        assert job.status_code == 201, job.text
        dispatcher = DirectNodeDispatcher(cast(AppContext, client.app.state.context))

        dispatcher.run_once()

        activated = client.get(
            "/api/v1/service-endpoints", headers=ADMIN_HEADERS
        ).json()[0]
        assert activated["enrollmentStatus"] == "enrolled"
        assert activated["probeStatus"] == "online"
        assert state.dispatched_job_ids == []
        rejected_claim = client.post(
            f"/api/v1/node-enrollments/{endpoint['id']}/claim",
            json={
                "enrollmentToken": endpoint["enrollmentToken"],
                "name": "direct-trainer-test",
                "kind": "trainer",
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
            },
        )
        assert rejected_claim.status_code == 409

        dispatcher.run_once()

        assert state.dispatched_job_ids == [job.json()["id"]]


def test_claimed_inference_endpoint_activates_linked_node_on_probe(
    client: TestClient,
) -> None:
    state = NodeState(
        kind="inference",
        accelerator="rk3588",
        capabilities=["deeplab_logits_v1"],
        node_token="not-claimed-yet-token",
    )
    with running_node(state) as (host, port):
        created = client.post(
            "/api/v1/service-endpoints",
            headers=ADMIN_HEADERS,
            json={
                "name": "direct-inference-test",
                "kind": "inference",
                "mode": "direct",
                "scheme": "http",
                "host": host,
                "port": port,
                "accelerator": "rk3588",
                "capabilities": ["deeplab_logits_v1"],
                "enabled": True,
            },
        )
        assert created.status_code == 201, created.text
        endpoint = created.json()
        before = client.get("/api/v1/inference-nodes", headers=ADMIN_HEADERS).json()
        linked_before = next(
            item for item in before["items"] if item["id"] == endpoint["inferenceNodeId"]
        )
        assert linked_before["lifecycle"] == "pending_registration"
        context = cast(AppContext, client.app.state.context)
        assert context.node_secrets.read(endpoint["id"], purpose="agent") is None
        claimed = client.post(
            f"/api/v1/node-enrollments/{endpoint['id']}/claim",
            json={
                "enrollmentToken": endpoint["enrollmentToken"],
                "name": "direct-inference-test",
                "kind": "inference",
                "accelerator": "rk3588",
                "capabilities": ["deeplab_logits_v1"],
                "version": "test-1.0",
                "maxConcurrency": 1,
            },
        )
        assert claimed.status_code == 200, claimed.text
        state.node_token = claimed.json()["nodeToken"]

        DirectNodeDispatcher(context).run_once()

        after = client.get("/api/v1/inference-nodes", headers=ADMIN_HEADERS).json()
        linked_after = next(
            item for item in after["items"] if item["id"] == endpoint["inferenceNodeId"]
        )
        assert linked_after["lifecycle"] == "active"
        assert linked_after["connectivity"] == "online"
        assert context.node_secrets.read(endpoint["id"], purpose="agent") is not None


def test_manual_probe_uses_the_same_claimed_activation_lifecycle(
    client: TestClient,
) -> None:
    state = NodeState(
        kind="trainer",
        accelerator="cpu",
        capabilities=["yolo-detect"],
        node_token="not-claimed-yet-token",
    )
    with running_node(state) as (host, port):
        created = client.post(
            "/api/v1/service-endpoints",
            headers=ADMIN_HEADERS,
            json={
                "name": "direct-trainer-test",
                "kind": "trainer",
                "mode": "direct",
                "scheme": "http",
                "host": host,
                "port": port,
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
                "enabled": True,
            },
        )
        endpoint = created.json()
        claimed = client.post(
            f"/api/v1/node-enrollments/{endpoint['id']}/claim",
            json={
                "enrollmentToken": endpoint["enrollmentToken"],
                "name": "direct-trainer-test",
                "kind": "trainer",
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
            },
        )
        assert claimed.status_code == 200, claimed.text
        state.node_token = claimed.json()["nodeToken"]

        probed = client.post(
            f"/api/v1/service-endpoints/{endpoint['id']}/probe",
            headers=ADMIN_HEADERS,
        )

    assert probed.status_code == 200, probed.text
    assert probed.json()["enrollmentStatus"] == "enrolled"
    assert probed.json()["probeStatus"] == "online"
    workers = client.get("/api/v1/workers", headers=ADMIN_HEADERS).json()
    assert any(worker["name"] == "direct-trainer-test" for worker in workers)


def test_direct_node_rejects_wrong_token(client: TestClient) -> None:
    state = NodeState(
        kind="inference",
        accelerator="rk3588",
        capabilities=["deeplab_logits_v1"],
    )
    with running_node(state) as (host, port):
        response = client.post(
            "/api/v1/service-endpoints",
            headers=ADMIN_HEADERS,
            json=inference_payload(host, port, "wrong-token-value-with-32-characters"),
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "node_unreachable"


def test_dispatcher_pushes_queued_training_job_to_direct_node(
    client: TestClient,
    detection_dataset: dict[str, object],
) -> None:
    state = NodeState(
        kind="trainer",
        accelerator="cpu",
        capabilities=["yolo-detect"],
    )
    with running_node(state) as (host, port):
        endpoint = client.post(
            "/api/v1/service-endpoints",
            headers=ADMIN_HEADERS,
            json={
                "name": "direct-trainer-test",
                "kind": "trainer",
                "mode": "direct",
                "scheme": "http",
                "host": host,
                "port": port,
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
                "enabled": True,
                "token": NODE_TOKEN,
            },
        )
        assert endpoint.status_code == 201, endpoint.text
        job = client.post(
            "/api/v1/training-jobs",
            headers=ADMIN_HEADERS,
            json={
                "name": "Direct dispatch test",
                "datasetId": detection_dataset["id"],
                "profileId": "yolo-detect",
                "variant": "yolov8n",
                "resolution": {"width": 640, "height": 640},
                "hyperparameters": {"epochs": 1, "batchSize": 1},
                "accelerator": "cpu",
            },
        )
        assert job.status_code == 201, job.text
        dispatcher = DirectNodeDispatcher(cast(AppContext, client.app.state.context))
        dispatcher.run_once()
        assert state.dispatched_job_ids == [job.json()["id"]]
        dispatcher.run_once()
        assert state.dispatched_job_ids == [job.json()["id"]]


def test_direct_node_token_cannot_register_another_worker_identity(
    client: TestClient,
) -> None:
    state = NodeState(
        kind="trainer",
        accelerator="cpu",
        capabilities=["yolo-detect"],
    )
    with running_node(state) as (host, port):
        created = client.post(
            "/api/v1/service-endpoints",
            headers=ADMIN_HEADERS,
            json={
                "name": "direct-trainer-test",
                "kind": "trainer",
                "mode": "direct",
                "scheme": "http",
                "host": host,
                "port": port,
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
                "enabled": True,
                "token": NODE_TOKEN,
            },
        )
        assert created.status_code == 201, created.text
        response = client.post(
            "/api/v1/workers/register",
            headers={"Authorization": f"Bearer {NODE_TOKEN}"},
            json={
                "name": "another-trainer",
                "kind": "trainer",
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
                "version": "test",
                "maxConcurrency": 1,
            },
        )
    assert response.status_code == 401


def test_direct_node_token_only_reads_resources_for_its_claimed_jobs(
    client: TestClient,
    detection_dataset: dict[str, object],
) -> None:
    state = NodeState(
        kind="trainer",
        accelerator="cpu",
        capabilities=["yolo-detect"],
    )
    node_headers = {"Authorization": f"Bearer {NODE_TOKEN}"}
    with running_node(state) as (host, port):
        created = client.post(
            "/api/v1/service-endpoints",
            headers=ADMIN_HEADERS,
            json={
                "name": "direct-trainer-test",
                "kind": "trainer",
                "mode": "direct",
                "scheme": "http",
                "host": host,
                "port": port,
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
                "enabled": True,
                "token": NODE_TOKEN,
            },
        )
        assert created.status_code == 201, created.text

        worker = client.post(
            "/api/v1/workers/register",
            headers=node_headers,
            json={
                "name": "direct-trainer-test",
                "kind": "trainer",
                "accelerator": "cpu",
                "capabilities": ["yolo-detect"],
                "version": "test",
                "maxConcurrency": 1,
            },
        )
        assert worker.status_code == 201, worker.text

        other_dataset = client.post(
            "/api/v1/datasets",
            headers=ADMIN_HEADERS,
            data={
                "metadata": (
                    '{"name":"Other defects","version":"v1",'
                    '"taskType":"object_detection","classes":["dent"]}'
                )
            },
            files={"file": ("other.zip", zip_dataset_bytes(), "application/zip")},
        )
        assert other_dataset.status_code == 201, other_dataset.text

        dataset_id = str(detection_dataset["id"])
        denied_before_claim = client.get(
            f"/api/v1/worker/datasets/{dataset_id}/download", headers=node_headers
        )
        assert denied_before_claim.status_code == 401
        assert client.get(
            "/api/v1/worker/jobs/retained", headers=node_headers
        ).json() == {"jobIds": []}

        job = client.post(
            "/api/v1/training-jobs",
            headers=ADMIN_HEADERS,
            json={
                "name": "Scoped resource test",
                "datasetId": dataset_id,
                "profileId": "yolo-detect",
                "variant": "yolov8n",
                "resolution": {"width": 640, "height": 640},
                "hyperparameters": {"epochs": 1, "batchSize": 1},
                "accelerator": "cpu",
            },
        )
        assert job.status_code == 201, job.text
        claim = client.post(
            "/api/v1/worker/jobs/claim",
            headers=node_headers,
            json={"workerId": worker.json()["id"], "jobId": job.json()["id"]},
        )
        assert claim.status_code == 200, claim.text

        allowed = client.get(
            f"/api/v1/worker/datasets/{dataset_id}/download", headers=node_headers
        )
        assert allowed.status_code == 200, allowed.text
        denied_other = client.get(
            f"/api/v1/worker/datasets/{other_dataset.json()['id']}/download",
            headers=node_headers,
        )
        assert denied_other.status_code == 401
        assert client.get(
            "/api/v1/worker/jobs/retained", headers=node_headers
        ).json() == {"jobIds": [job.json()["id"]]}


def test_direct_node_token_is_unique_and_disabled_endpoint_rejects_it(
    client: TestClient,
) -> None:
    state = NodeState(
        kind="trainer",
        accelerator="cpu",
        capabilities=["yolo-detect"],
    )
    payload: dict[str, object]
    with running_node(state) as (host, port):
        payload = {
            "name": "direct-trainer-test",
            "kind": "trainer",
            "mode": "direct",
            "scheme": "http",
            "host": host,
            "port": port,
            "accelerator": "cpu",
            "capabilities": ["yolo-detect"],
            "enabled": True,
            "token": NODE_TOKEN,
        }
        created = client.post(
            "/api/v1/service-endpoints", headers=ADMIN_HEADERS, json=payload
        )
        assert created.status_code == 201, created.text

        duplicate_token = client.post(
            "/api/v1/service-endpoints",
            headers=ADMIN_HEADERS,
            json={
                **payload,
                "name": "direct-trainer-disabled",
                "port": port + 1,
                "enabled": False,
            },
        )
        assert duplicate_token.status_code == 409
        assert duplicate_token.json()["error"]["code"] == "node_token_conflict"

        global_token = client.post(
            "/api/v1/service-endpoints",
            headers=ADMIN_HEADERS,
            json={
                **payload,
                "name": "direct-trainer-global-token",
                "port": port + 2,
                "enabled": False,
                "token": "test-worker-token",
            },
        )
        assert global_token.status_code == 409
        assert global_token.json()["error"]["code"] == "node_token_conflict"

        disabled_payload = {**payload, "enabled": False, "token": None}
        disabled = client.put(
            f"/api/v1/service-endpoints/{created.json()['id']}",
            headers=ADMIN_HEADERS,
            json=disabled_payload,
        )
        assert disabled.status_code == 200, disabled.text

    rejected = client.get(
        "/api/v1/worker/jobs/retained",
        headers={"Authorization": f"Bearer {NODE_TOKEN}"},
    )
    assert rejected.status_code == 401
