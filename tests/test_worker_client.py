from __future__ import annotations

import json
from typing import Any

import pytest
from workers.common.client import PlatformClient, WorkerApiError


class _NullResponse:
    def __enter__(self) -> _NullResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b"null"


class _JsonResponse(_NullResponse):
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_claim_accepts_json_null(monkeypatch: Any) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _NullResponse())

    client = PlatformClient("http://platform.test/api/v1", "worker-token")

    assert client.claim("worker-1") is None


def test_worker_updates_discovered_dataset_classes(monkeypatch: Any) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: Any, **_kwargs: object) -> _JsonResponse:
        captured["method"] = request.get_method()
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return _JsonResponse({"id": "dataset-1", "classes": ["scratch"]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = PlatformClient("http://platform.test/api/v1", "worker-token")

    response = client.update_dataset_classes("dataset-1", ["scratch"])

    assert response["classes"] == ["scratch"]
    assert captured == {
        "method": "PUT",
        "url": "http://platform.test/api/v1/worker/datasets/dataset-1/classes",
        "payload": {"classes": ["scratch"]},
    }


def test_worker_reads_retained_job_ids(monkeypatch: Any) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: Any, **_kwargs: object) -> _JsonResponse:
        captured["method"] = request.get_method()
        captured["url"] = request.full_url
        return _JsonResponse({"jobIds": ["train_1", "convert_2"]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = PlatformClient("http://platform.test/api/v1", "worker-token")

    assert client.retained_job_ids() == {"train_1", "convert_2"}
    assert captured == {
        "method": "GET",
        "url": "http://platform.test/api/v1/worker/jobs/retained",
    }


def test_worker_rejects_invalid_retained_job_ids(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _JsonResponse({"jobIds": ["train_1", 2]}),
    )
    client = PlatformClient("http://platform.test/api/v1", "worker-token")

    with pytest.raises(WorkerApiError, match="invalid_response"):
        client.retained_job_ids()


def test_worker_batches_training_telemetry(monkeypatch: Any) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: Any, **_kwargs: object) -> _JsonResponse:
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return _JsonResponse({"accepted": 1})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = PlatformClient("http://platform.test/api/v1", "worker-token")
    entries = [
        {
            "type": "metric",
            "stage": "train",
            "message": "epoch=1/2 loss=0.5",
            "metrics": {"loss": 0.5},
            "epoch": 1,
            "totalEpochs": 2,
        }
    ]

    response = client.telemetry("train-1", "lease-1", entries)

    assert response == {"accepted": 1}
    assert captured == {
        "url": "http://platform.test/api/v1/worker/jobs/train-1/events",
        "payload": {"leaseToken": "lease-1", "entries": entries},
    }
