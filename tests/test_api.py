from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

from backend.platform_api.app import create_app
from backend.platform_api.database import Database
from backend.platform_api.db_models import ArtifactRecord, JobRecord, WorkerRecord, utc_now
from backend.platform_api.settings import Settings
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text

from tests.conftest import ADMIN_HEADERS, WORKER_HEADERS


def create_training_job(client: TestClient, dataset_id: str) -> dict[str, object]:
    response = client.post(
        "/api/v1/training-jobs",
        headers=ADMIN_HEADERS,
        json={
            "name": "YOLO custom resolution",
            "datasetId": dataset_id,
            "profileId": "yolo-detect",
            "variant": "yolov8n",
            "resolution": {"width": 640, "height": 384},
            "hyperparameters": {"epochs": 2, "batchSize": 1},
            "accelerator": "cpu",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def register_cpu_worker(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/api/v1/workers/register",
        headers=WORKER_HEADERS,
        json={
            "name": "cpu-test-worker",
            "kind": "trainer",
            "capabilities": ["yolo-detect"],
            "accelerator": "cpu",
            "maxConcurrency": 1,
            "version": "test",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_health_authentication_and_profiles(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/api/v1/model-profiles").status_code == 401
    response = client.get("/api/v1/model-profiles", headers=ADMIN_HEADERS)
    assert response.status_code == 200
    assert {item["id"] for item in response.json()["profiles"]} == {
        "yolo-detect",
        "deeplabv3plus",
        "ppocr-det",
        "ppocr-rec",
    }
    assert response.headers["X-Request-ID"]


def test_dataset_format_is_persisted_and_bound_to_task(client: TestClient) -> None:
    payload = {
        "name": "VOC defects",
        "version": "v1",
        "taskType": "object_detection",
        "datasetFormat": "voc_detection",
    }
    response = client.post(
        "/api/v1/datasets",
        headers=ADMIN_HEADERS,
        data={"metadata": json.dumps(payload)},
        files={"file": ("voc.zip", b"PK\x05\x06" + b"\0" * 18, "application/zip")},
    )
    assert response.status_code == 201, response.text
    assert response.json()["datasetFormat"] == "voc_detection"
    assert response.json()["classes"] == []

    dataset_id = response.json()["id"]
    discovered = client.put(
        f"/api/v1/worker/datasets/{dataset_id}/classes",
        headers=WORKER_HEADERS,
        json={"classes": ["scratch"]},
    )
    assert discovered.status_code == 200, discovered.text
    assert discovered.json()["classes"] == ["scratch"]

    repeated = client.put(
        f"/api/v1/worker/datasets/{dataset_id}/classes",
        headers=WORKER_HEADERS,
        json={"classes": ["scratch"]},
    )
    assert repeated.status_code == 200

    conflict = client.put(
        f"/api/v1/worker/datasets/{dataset_id}/classes",
        headers=WORKER_HEADERS,
        json={"classes": ["dent"]},
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "dataset_classes_mismatch"

    job = create_training_job(client, dataset_id)
    assert job["spec"]["dataset"]["name"] == "VOC defects"
    assert job["spec"]["dataset"]["version"] == "v1"
    assert job["spec"]["dataset"]["datasetFormat"] == "voc_detection"
    assert job["spec"]["dataset"]["classes"] == ["scratch"]

    payload["taskType"] = "semantic_segmentation"
    mismatch = client.post(
        "/api/v1/datasets",
        headers=ADMIN_HEADERS,
        data={"metadata": json.dumps(payload)},
        files={"file": ("voc.zip", b"PK\x05\x06" + b"\0" * 18, "application/zip")},
    )
    assert mismatch.status_code == 422


def test_existing_database_gets_dataset_format_column(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    database = Database(f"sqlite:///{database_path}")
    with database.engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE datasets (
                    id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(120) NOT NULL,
                    description TEXT NOT NULL,
                    version VARCHAR(40) NOT NULL,
                    task_type VARCHAR(40) NOT NULL,
                    classes_json JSON NOT NULL,
                    status VARCHAR(30) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    storage_key VARCHAR(255) NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    sha256 VARCHAR(64) NOT NULL,
                    error_message TEXT,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )
    database.create_schema()
    columns = {column["name"] for column in inspect(database.engine).get_columns("datasets")}
    assert "dataset_format" in columns
    with database.engine.connect() as connection:
        default_value = connection.execute(
            text(
                "SELECT dflt_value FROM pragma_table_info('datasets') "
                "WHERE name = 'dataset_format'"
            )
        ).scalar_one()
    assert default_value == "'auto'"


def test_existing_database_gets_inference_npu_core_columns(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy-inference.db"
    database = Database(f"sqlite:///{database_path}")
    with database.engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE inference_tasks (id VARCHAR(48) PRIMARY KEY)")
        )

    database.create_schema()
    database.create_schema()

    columns = {
        column["name"]: column
        for column in inspect(database.engine).get_columns("inference_tasks")
    }
    assert columns["npu_core_mask"]["default"] == "'auto'"
    assert columns["npu_core_policy"]["default"] == "'shared'"


def test_dataset_training_worker_lease_and_progress(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    job = create_training_job(client, str(detection_dataset["id"]))
    worker = register_cpu_worker(client)
    claim_response = client.post(
        "/api/v1/worker/jobs/claim",
        headers=WORKER_HEADERS,
        json={"workerId": worker["id"]},
    )
    assert claim_response.status_code == 200, claim_response.text
    claim = claim_response.json()
    assert claim["job"]["id"] == job["id"]
    assert claim["job"]["status"] == "claimed"

    progress_response = client.post(
        f"/api/v1/worker/jobs/{job['id']}/progress",
        headers=WORKER_HEADERS,
        json={
            "leaseToken": claim["leaseToken"],
            "progress": 25,
            "stage": "training",
            "message": "epoch 1/2",
            "metrics": {"loss": 0.5},
        },
    )
    assert progress_response.status_code == 200, progress_response.text
    assert progress_response.json()["status"] == "running"

    existing_events = client.get(
        f"/api/v1/jobs/{job['id']}/events", headers=ADMIN_HEADERS
    ).json()
    telemetry_response = client.post(
        f"/api/v1/worker/jobs/{job['id']}/events",
        headers=WORKER_HEADERS,
        json={
            "leaseToken": claim["leaseToken"],
            "entries": [
                {
                    "type": "log",
                    "stage": "training",
                    "message": "loading batch 1\nloading batch 2",
                },
                {
                    "type": "metric",
                    "stage": "training",
                    "message": "epoch=1/2 loss=0.5 accuracy=0.75",
                    "metrics": {"loss": 0.5, "accuracy": 0.75},
                    "step": 4,
                    "epoch": 1,
                    "totalEpochs": 2,
                },
            ],
        },
    )
    assert telemetry_response.status_code == 200, telemetry_response.text
    assert telemetry_response.json() == {"accepted": 2}
    new_events_response = client.get(
        f"/api/v1/jobs/{job['id']}/events",
        headers=ADMIN_HEADERS,
        params={"afterId": existing_events[-1]["id"], "limit": 10},
    )
    assert new_events_response.status_code == 200
    new_events = new_events_response.json()
    assert [event["type"] for event in new_events] == ["log", "metric"]
    assert new_events[1]["data"] == {
        "stage": "training",
        "metrics": {"loss": 0.5, "accuracy": 0.75},
        "step": 4,
        "epoch": 1,
        "totalEpochs": 2,
    }

    renewed = client.post(
        f"/api/v1/worker/jobs/{job['id']}/renew",
        headers=WORKER_HEADERS,
        json={"leaseToken": claim["leaseToken"]},
    )
    assert renewed.status_code == 200
    assert renewed.json()["status"] == "running"

    regression = client.post(
        f"/api/v1/worker/jobs/{job['id']}/progress",
        headers=WORKER_HEADERS,
        json={"leaseToken": claim["leaseToken"], "progress": 20, "stage": "training"},
    )
    assert regression.status_code == 409
    assert regression.json()["error"]["code"] == "progress_regression"

    completion = client.post(
        f"/api/v1/worker/jobs/{job['id']}/complete",
        headers=WORKER_HEADERS,
        json={"leaseToken": claim["leaseToken"], "result": {"metric": 0.9}},
    )
    assert completion.status_code == 200
    assert completion.json()["status"] == "succeeded"
    assert completion.json()["progress"] == 100


def test_worker_can_claim_a_specific_compatible_job(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    first = create_training_job(client, str(detection_dataset["id"]))
    second = create_training_job(client, str(detection_dataset["id"]))
    worker = register_cpu_worker(client)

    response = client.post(
        "/api/v1/worker/jobs/claim",
        headers=WORKER_HEADERS,
        json={"workerId": worker["id"], "jobId": second["id"]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["job"]["id"] == second["id"]
    queued = client.get(f"/api/v1/jobs/{first['id']}", headers=ADMIN_HEADERS)
    assert queued.json()["status"] == "queued"


def test_failed_training_and_conversion_jobs_can_be_retried(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    failed_training = create_training_job(client, str(detection_dataset["id"]))
    worker = register_cpu_worker(client)
    claim = client.post(
        "/api/v1/worker/jobs/claim",
        headers=WORKER_HEADERS,
        json={"workerId": worker["id"]},
    ).json()
    failure = client.post(
        f"/api/v1/worker/jobs/{failed_training['id']}/fail",
        headers=WORKER_HEADERS,
        json={
            "leaseToken": claim["leaseToken"],
            "code": "training_failed",
            "message": "Training process exited",
            "retryable": False,
        },
    )
    assert failure.status_code == 200
    assert failure.json()["status"] == "failed"

    retried_training_response = client.post(
        f"/api/v1/jobs/{failed_training['id']}/retry", headers=ADMIN_HEADERS
    )
    assert retried_training_response.status_code == 201, retried_training_response.text
    retried_training = retried_training_response.json()
    assert retried_training["id"] != failed_training["id"]
    assert retried_training["status"] == "queued"
    assert retried_training["datasetId"] == failed_training["datasetId"]
    assert retried_training["spec"]["retryOfJobId"] == failed_training["id"]
    original = client.get(
        f"/api/v1/jobs/{failed_training['id']}", headers=ADMIN_HEADERS
    ).json()
    assert original["status"] == "failed"
    assert original["errorCode"] == "training_failed"

    manifest = {
        "schemaVersion": 1,
        "modelFamily": "YOLO",
        "profileId": "yolo-detect",
        "variant": "yolov8n",
        "taskType": "object_detection",
        "trainingJobId": failed_training["id"],
        "onnxSha256": "1" * 64,
        "opset": 12,
        "resolution": {"width": 640, "height": 384},
        "input": {
            "name": "images",
            "layout": "NCHW",
            "shape": [1, 3, 384, 640],
            "dtype": "float32",
            "colorSpace": "RGB",
        },
        "preprocessing": {"mean": [0, 0, 0], "std": [255, 255, 255]},
        "resizePolicy": "letterbox",
        "outputContract": "rknn_yolo_dfl_split_heads_v1",
        "outputs": [{"name": "output0", "semantic": "detections"}],
        "labels": ["scratch"],
        "supportedPrecisions": ["int8", "fp16"],
        "rknn": {
            "targetPlatform": "rk3588",
            "quantizedAlgorithm": "normal",
            "optimizationLevel": 3,
            "requiresCalibrationFor": ["int8"],
        },
    }
    with client.app.state.context.database.session() as session:
        session.add(
            ArtifactRecord(
                id="artifact_retry_source",
                job_id=None,
                kind="onnx",
                filename="retry-source.onnx",
                storage_key="artifacts/retry-source.onnx",
                media_type="application/onnx",
                size_bytes=10,
                sha256="1" * 64,
                manifest_json=manifest,
            )
        )

    conversion_response = client.post(
        "/api/v1/conversion-jobs",
        headers=ADMIN_HEADERS,
        json={
            "name": "Retry RKNN conversion",
            "sourceArtifactId": "artifact_retry_source",
            "precision": "fp16",
        },
    )
    assert conversion_response.status_code == 201, conversion_response.text
    failed_conversion = conversion_response.json()
    with client.app.state.context.database.session() as session:
        conversion_record = session.get(JobRecord, failed_conversion["id"])
        assert conversion_record is not None
        conversion_record.status = "failed"
        conversion_record.stage = "failed"
        conversion_record.error_code = "rknn_build_failed"
        conversion_record.error_message = "RKNN build failed"
        conversion_record.completed_at = utc_now()

    retried_conversion_response = client.post(
        f"/api/v1/jobs/{failed_conversion['id']}/retry", headers=ADMIN_HEADERS
    )
    assert retried_conversion_response.status_code == 201, retried_conversion_response.text
    retried_conversion = retried_conversion_response.json()
    assert retried_conversion["id"] != failed_conversion["id"]
    assert retried_conversion["status"] == "queued"
    assert retried_conversion["spec"]["sourceArtifactId"] == "artifact_retry_source"
    assert retried_conversion["spec"]["retryOfJobId"] == failed_conversion["id"]

    not_failed = client.post(
        f"/api/v1/jobs/{retried_conversion['id']}/retry", headers=ADMIN_HEADERS
    )
    assert not_failed.status_code == 409
    assert not_failed.json()["error"]["code"] == "job_not_retryable"


def test_expired_lease_releases_worker_and_resets_retry_progress(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    job = create_training_job(client, str(detection_dataset["id"]))
    worker = register_cpu_worker(client)
    claim = client.post(
        "/api/v1/worker/jobs/claim",
        headers=WORKER_HEADERS,
        json={"workerId": worker["id"]},
    ).json()
    progress = client.post(
        f"/api/v1/worker/jobs/{job['id']}/progress",
        headers=WORKER_HEADERS,
        json={"leaseToken": claim["leaseToken"], "progress": 40, "stage": "training"},
    )
    assert progress.status_code == 200

    context = client.app.state.context
    with context.database.session() as session:
        record = session.get(JobRecord, job["id"])
        assert record is not None
        record.lease_expires_at = utc_now() - timedelta(seconds=1)

    retried = client.post(
        "/api/v1/worker/jobs/claim",
        headers=WORKER_HEADERS,
        json={"workerId": worker["id"]},
    )
    assert retried.status_code == 200, retried.text
    retried_job = retried.json()["job"]
    assert retried_job["id"] == job["id"]
    assert retried_job["progress"] == 0
    assert retried_job["retryCount"] == 1

    with context.database.session() as session:
        worker_record = session.get(WorkerRecord, worker["id"])
        assert worker_record is not None
        assert worker_record.active_jobs == 1


def test_invalid_resolution_is_rejected(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    response = client.post(
        "/api/v1/training-jobs",
        headers=ADMIN_HEADERS,
        json={
            "name": "invalid shape",
            "datasetId": detection_dataset["id"],
            "profileId": "yolo-detect",
            "variant": "yolov8n",
            "resolution": {"width": 650, "height": 640},
            "accelerator": "cpu",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_resolution"


def test_worker_uploads_checksum_bound_manifest(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    job = create_training_job(client, str(detection_dataset["id"]))
    worker = register_cpu_worker(client)
    claim = client.post(
        "/api/v1/worker/jobs/claim",
        headers=WORKER_HEADERS,
        json={"workerId": worker["id"]},
    ).json()
    model_bytes = b"synthetic-onnx-placeholder"
    digest = hashlib.sha256(model_bytes).hexdigest()
    manifest = {
        "schemaVersion": 1,
        "modelFamily": "YOLO",
        "profileId": "yolo-detect",
        "variant": "yolov8n",
        "taskType": "object_detection",
        "trainingJobId": job["id"],
        "onnxSha256": digest,
        "opset": 12,
        "resolution": {"width": 640, "height": 384},
        "input": {
            "name": "images",
            "layout": "NCHW",
            "shape": [1, 3, 384, 640],
            "dtype": "float32",
            "colorSpace": "RGB",
        },
        "preprocessing": {"mean": [0, 0, 0], "std": [255, 255, 255]},
        "resizePolicy": "letterbox",
        "outputContract": "rknn_yolo_dfl_split_heads_v1",
        "outputs": [{"name": "output0", "semantic": "detections"}],
        "labels": ["scratch"],
        "supportedPrecisions": ["int8", "fp16"],
        "rknn": {
            "targetPlatform": "rk3588",
            "quantizedAlgorithm": "normal",
            "optimizationLevel": 3,
            "requiresCalibrationFor": ["int8"],
        },
    }
    response = client.post(
        f"/api/v1/worker/jobs/{job['id']}/artifacts",
        headers=WORKER_HEADERS,
        data={"lease_token": claim["leaseToken"], "kind": "onnx", "manifest": json.dumps(manifest)},
        files={"file": ("model.onnx", model_bytes, "application/onnx")},
    )
    assert response.status_code == 201, response.text
    artifact = response.json()
    assert artifact["sha256"] == digest
    assert artifact["manifest"]["input"]["shape"] == [1, 3, 384, 640]

    listed = client.get("/api/v1/artifacts?kind=onnx", headers=ADMIN_HEADERS)
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == artifact["id"]

    conversion = client.post(
        "/api/v1/conversion-jobs",
        headers=ADMIN_HEADERS,
        json={
            "name": "RKNN INT8",
            "sourceArtifactId": artifact["id"],
            "precision": "int8",
        },
    )
    assert conversion.status_code == 400
    assert conversion.json()["error"]["code"] == "calibration_dataset_required"


def test_database_persists_across_app_restart(
    settings: Settings, detection_dataset: dict[str, object]
) -> None:
    with TestClient(create_app(settings)) as restarted:
        response = restarted.get("/api/v1/datasets", headers=ADMIN_HEADERS)
        assert response.status_code == 200
        assert response.json()[0]["id"] == detection_dataset["id"]


def test_storage_path_traversal_is_rejected(client: TestClient) -> None:
    context = client.app.state.context
    try:
        context.storage.resolve("../../etc/passwd")
    except Exception as error:
        assert getattr(error, "code", None) == "invalid_storage_key"
    else:
        raise AssertionError("path traversal should be rejected")


def test_queued_or_terminal_jobs_and_unused_datasets_can_be_deleted(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    queued_job = create_training_job(client, str(detection_dataset["id"]))
    blocked_dataset = client.delete(
        f"/api/v1/datasets/{detection_dataset['id']}", headers=ADMIN_HEADERS
    )
    assert blocked_dataset.status_code == 409
    assert blocked_dataset.json()["error"]["code"] == "dataset_in_use"
    assert (
        client.delete(
            f"/api/v1/jobs/{queued_job['id']}", headers=ADMIN_HEADERS
        ).status_code
        == 204
    )

    claimed_job = create_training_job(client, str(detection_dataset["id"]))
    worker = register_cpu_worker(client)
    claim = client.post(
        "/api/v1/worker/jobs/claim",
        headers=WORKER_HEADERS,
        json={"workerId": worker["id"]},
    ).json()
    claimed_delete = client.delete(
        f"/api/v1/jobs/{claimed_job['id']}", headers=ADMIN_HEADERS
    )
    assert claimed_delete.status_code == 409
    assert claimed_delete.json()["error"]["code"] == "job_not_deletable"
    completed = client.post(
        f"/api/v1/worker/jobs/{claimed_job['id']}/complete",
        headers=WORKER_HEADERS,
        json={"leaseToken": claim["leaseToken"], "result": {}},
    )
    assert completed.status_code == 200
    assert (
        client.delete(
            f"/api/v1/jobs/{claimed_job['id']}", headers=ADMIN_HEADERS
        ).status_code
        == 204
    )
    deleted = client.delete(
        f"/api/v1/datasets/{detection_dataset['id']}", headers=ADMIN_HEADERS
    )
    assert deleted.status_code == 204


def test_worker_retention_list_removes_deleted_job(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    job = create_training_job(client, str(detection_dataset["id"]))

    retained = client.get("/api/v1/worker/jobs/retained", headers=WORKER_HEADERS)
    assert retained.status_code == 200
    assert retained.json() == {"jobIds": [job["id"]]}
    assert client.get("/api/v1/worker/jobs/retained", headers=ADMIN_HEADERS).status_code == 401

    deleted = client.delete(f"/api/v1/jobs/{job['id']}", headers=ADMIN_HEADERS)
    assert deleted.status_code == 204
    assert client.get(
        "/api/v1/worker/jobs/retained", headers=WORKER_HEADERS
    ).json() == {"jobIds": []}


def test_service_endpoint_config_controls_matching_worker(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    payload = {
        "name": "configured-cpu-worker",
        "kind": "trainer",
        "endpoint": "http://trainer.internal:9000",
        "accelerator": "cpu",
        "capabilities": ["yolo-detect"],
        "enabled": False,
    }
    created = client.post("/api/v1/service-endpoints", headers=ADMIN_HEADERS, json=payload)
    assert created.status_code == 201, created.text
    endpoint = created.json()
    assert endpoint["endpoint"] == payload["endpoint"]

    mismatched = client.post(
        "/api/v1/workers/register",
        headers=WORKER_HEADERS,
        json={
            "name": payload["name"],
            "kind": payload["kind"],
            "accelerator": "cuda",
            "capabilities": payload["capabilities"],
            "version": "test",
            "maxConcurrency": 1,
        },
    )
    assert mismatched.status_code == 409

    worker = client.post(
        "/api/v1/workers/register",
        headers=WORKER_HEADERS,
        json={
            "name": payload["name"],
            "kind": payload["kind"],
            "accelerator": payload["accelerator"],
            "capabilities": payload["capabilities"],
            "version": "test",
            "maxConcurrency": 1,
        },
    )
    assert worker.status_code == 201, worker.text
    job = create_training_job(client, str(detection_dataset["id"]))
    disabled_claim = client.post(
        "/api/v1/worker/jobs/claim",
        headers=WORKER_HEADERS,
        json={"workerId": worker.json()["id"]},
    )
    assert disabled_claim.status_code == 200
    assert disabled_claim.json() is None
    assert client.put(
        f"/api/v1/service-endpoints/{endpoint['id']}",
        headers=ADMIN_HEADERS,
        json={**payload, "enabled": True},
    ).status_code == 200
    enabled_claim = client.post(
        "/api/v1/worker/jobs/claim",
        headers=WORKER_HEADERS,
        json={"workerId": worker.json()["id"]},
    )
    assert enabled_claim.status_code == 200
    assert enabled_claim.json()["job"]["id"] == job["id"]


def test_only_offline_workers_without_active_jobs_can_be_deleted(
    client: TestClient, detection_dataset: dict[str, object]
) -> None:
    worker = register_cpu_worker(client)
    online_delete = client.delete(f"/api/v1/workers/{worker['id']}", headers=ADMIN_HEADERS)
    assert online_delete.status_code == 409
    assert online_delete.json()["error"]["code"] == "worker_not_offline"

    job = create_training_job(client, str(detection_dataset["id"]))
    claim = client.post(
        "/api/v1/worker/jobs/claim",
        headers=WORKER_HEADERS,
        json={"workerId": worker["id"]},
    ).json()
    with client.app.state.context.database.session() as session:
        record = session.get(WorkerRecord, worker["id"])
        assert record is not None
        record.last_seen_at = utc_now() - timedelta(seconds=120)
    active_delete = client.delete(f"/api/v1/workers/{worker['id']}", headers=ADMIN_HEADERS)
    assert active_delete.status_code == 409
    assert active_delete.json()["error"]["code"] == "worker_has_active_jobs"

    completed = client.post(
        f"/api/v1/worker/jobs/{job['id']}/complete",
        headers=WORKER_HEADERS,
        json={"leaseToken": claim["leaseToken"], "result": {}},
    )
    assert completed.status_code == 200
    with client.app.state.context.database.session() as session:
        record = session.get(WorkerRecord, worker["id"])
        assert record is not None
        record.last_seen_at = utc_now() - timedelta(seconds=120)
    deleted = client.delete(f"/api/v1/workers/{worker['id']}", headers=ADMIN_HEADERS)
    assert deleted.status_code == 204
    assert client.get("/api/v1/workers", headers=ADMIN_HEADERS).json() == []
    assert client.get(f"/api/v1/jobs/{job['id']}", headers=ADMIN_HEADERS).json()["workerId"] is None
