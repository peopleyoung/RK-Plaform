from __future__ import annotations

import io
import zipfile
from collections.abc import Generator
from pathlib import Path

import pytest
from backend.platform_api.app import create_app
from backend.platform_api.settings import Settings
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parents[1]
ADMIN_HEADERS = {"Authorization": "Bearer test-admin-token"}
WORKER_HEADERS = {"Authorization": "Bearer test-worker-token"}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'platform.db'}",
        model_profiles_path=PROJECT_ROOT / "config/model_profiles.json",
        admin_token="test-admin-token",
        worker_token="test-worker-token",
        upload_limit_bytes=10 * 1024 * 1024,
        worker_lease_seconds=60,
        direct_dispatch_enabled=False,
    )


@pytest.fixture
def client(settings: Settings) -> Generator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def zip_dataset_bytes() -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("images/train/example.jpg", b"not-a-real-image")
        archive.writestr("labels/train/example.txt", "0 0.5 0.5 0.25 0.25\n")
    return target.getvalue()


@pytest.fixture
def detection_dataset(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/api/v1/datasets",
        headers=ADMIN_HEADERS,
        data={
            "metadata": (
                '{"name":"Defects","version":"v1","taskType":"object_detection",'
                '"classes":["scratch"]}'
            )
        },
        files={"file": ("dataset.zip", zip_dataset_bytes(), "application/zip")},
    )
    assert response.status_code == 201, response.text
    return response.json()
