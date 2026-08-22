from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from backend.platform_api.profiles import ModelProfileRegistry
from workers.trainer.adapters import AdapterEnvironment
from workers.trainer.executor import TrainingExecutor


class FakePlatformClient:
    def __init__(self, archive: Path) -> None:
        self.archive = archive
        self.progress_stages: list[str] = []
        self.uploads: list[tuple[str, Path]] = []
        self.updated_classes: list[tuple[str, list[str]]] = []
        self.telemetry_entries: list[dict[str, object]] = []

    def progress(
        self,
        _job_id: str,
        _lease_token: str,
        _progress: int,
        stage: str,
        _message: str,
    ) -> dict[str, Any]:
        self.progress_stages.append(stage)
        return {}

    def download_dataset(self, _dataset_id: str, target: Path) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.archive, target)
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def telemetry(
        self,
        _job_id: str,
        _lease_token: str,
        entries: list[dict[str, object]],
    ) -> dict[str, Any]:
        self.telemetry_entries.extend(entries)
        return {"accepted": len(entries)}

    def update_dataset_classes(
        self, dataset_id: str, classes: list[str]
    ) -> dict[str, Any]:
        self.updated_classes.append((dataset_id, classes))
        return {"id": dataset_id, "classes": classes}

    def upload_artifact(
        self,
        _job_id: str,
        _lease_token: str,
        kind: str,
        path: Path,
        *,
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert path.is_file()
        if kind == "onnx":
            assert manifest is not None
        self.uploads.append((kind, path))
        return {"id": f"artifact_{kind}"}


def _write_fake_ultralytics(root: Path) -> None:
    torch_package = root / "torch"
    torch_package.mkdir(parents=True)
    (torch_package / "__init__.py").write_text(
        """
from types import SimpleNamespace

onnx = SimpleNamespace(export=lambda *args, **kwargs: None)
""".lstrip(),
        encoding="utf-8",
    )
    package = root / "ultralytics"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """
from pathlib import Path
from types import SimpleNamespace

class YOLO:
    def __init__(self, source):
        self.source = str(source)

    def train(self, **kwargs):
        best = Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"checkpoint")
        self.trainer = SimpleNamespace(best=str(best))

    def export(self, **kwargs):
        import onnx
        from onnx import TensorProto, helper
        height, width = kwargs["imgsz"]
        shape = [1, 3, height, width]
        source = helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)
        target = helper.make_tensor_value_info("output0", TensorProto.FLOAT, shape)
        graph = helper.make_graph(
            [helper.make_node("Identity", ["images"], ["output0"])],
            "fake_training_export",
            [source],
            [target],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", kwargs["opset"])]
        )
        model.ir_version = 10
        output = Path(self.source).with_suffix(".onnx")
        onnx.save(model, output)
        return str(output)
""".lstrip(),
        encoding="utf-8",
    )


def test_training_executor_runs_download_validation_export_and_upload(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(
            "data.yaml",
            "path: .\ntrain: images/train\nval: images/val\nnames: [target]\n",
        )
        target.writestr("images/train/sample.jpg", b"fixture")
        target.writestr("images/val/sample.jpg", b"fixture")
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    framework_root = tmp_path / "fake-yolov8"
    _write_fake_ultralytics(framework_root)
    project_root = Path(__file__).parents[1]
    executor = TrainingExecutor(
        ModelProfileRegistry(project_root / "config/model_profiles.json"),
        AdapterEnvironment(
            python=sys.executable,
            project_root=project_root,
            framework_roots={"yolov8": framework_root},
        ),
    )
    client = FakePlatformClient(archive)
    claim = {
        "leaseToken": "lease_test",
        "job": {
            "id": "train_executor",
            "spec": {
                "name": "executor contract",
                "datasetId": "dataset_test",
                "profileId": "yolo-detect",
                "variant": "yolov8n",
                "resolution": {"width": 640, "height": 384},
                "hyperparameters": {
                    "epochs": 1,
                    "batchSize": 1,
                    "optimizer": "SGD",
                    "pretrained": False,
                    "seed": 9,
                },
                "accelerator": "cpu",
                "retryOfJobId": "train_failed_source",
                "dataset": {
                    "id": "dataset_test",
                    "filename": "dataset.zip",
                    "sha256": checksum,
                    "classes": [],
                },
            },
        },
    }

    result = executor.execute(claim, client, tmp_path / "workspace")  # type: ignore[arg-type]

    assert result["onnxArtifactId"] == "artifact_onnx"
    assert "validate_dataset" in client.progress_stages
    assert client.updated_classes == [("dataset_test", ["target"])]
    assert any(entry["type"] == "log" for entry in client.telemetry_entries)
    assert [kind for kind, _ in client.uploads] == [
        "onnx",
        "manifest",
        "training_log",
        "training_checkpoint",
    ]
    upload_paths = {kind: path.name for kind, path in client.uploads}
    assert upload_paths["onnx"] == "yolov8n-640x384.onnx"
    assert upload_paths["training_checkpoint"] == "yolov8n-640x384.pt"
    manifest = result["manifest"]
    assert manifest["input"]["shape"] == [1, 3, 384, 640]
    assert manifest["labels"] == ["target"]
