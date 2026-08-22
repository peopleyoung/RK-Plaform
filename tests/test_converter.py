from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from workers.converter.calibration import create_calibration_list
from workers.converter.engine import ConversionError, RknnConverter, _parse_performance_report
from workers.converter.executor import ConversionExecutor
from workers.converter.graph_optimizer import optimize_conversion_graph


class FakeBackend:
    latest: FakeBackend | None = None
    instances: ClassVar[list[FakeBackend]] = []

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.released = False
        FakeBackend.latest = self
        FakeBackend.instances.append(self)

    @property
    def toolkit_version(self) -> str:
        return "2.3.2-test"

    def config(self, **values: Any) -> int:
        self.calls.append(("config", values))
        return 0

    def load_onnx(self, model: str) -> int:
        self.calls.append(("load", model))
        return 0

    def load_rknn(self, path: str) -> int:
        self.calls.append(("load_rknn", path))
        return 0

    def build(self, *, do_quantization: bool, dataset: str | None) -> int:
        self.calls.append(("build", {"quantized": do_quantization, "dataset": dataset}))
        return 0

    def export_rknn(self, path: str) -> int:
        Path(path).write_bytes(b"rknn-test")
        self.calls.append(("export", path))
        return 0

    def init_runtime(self, *, target: str, perf_debug: bool = False) -> int:
        self.calls.append(("runtime", {"target": target, "perfDebug": perf_debug}))
        return 0

    def inference(self, inputs: list[Any], *, data_format: str) -> list[Any]:
        self.calls.append(("inference", {"shape": inputs[0].shape, "format": data_format}))
        return [np.zeros((1, 10, 6), dtype=np.float32)]

    def eval_perf(self, *, is_print: bool, fix_freq: bool) -> str:
        self.calls.append(("eval_perf", {"isPrint": is_print, "fixFreq": fix_freq}))
        return """Total Operator Elapsed Per Frame Time(us): 1200
OpType             CallNumber   CPUTime(us)  GPUTime(us)  NPUTime(us)  TotalTime(us)  TimeRatio(%)
InputOperator      1            5            0            0            5              0.42%
Conv               2            0            0            1188         1188           99.00%
OutputOperator     1            7            0            0            7              0.58%
"""

    def release(self) -> None:
        self.released = True


class RuntimeFailureBackend(FakeBackend):
    def init_runtime(self, *, target: str, perf_debug: bool = False) -> int:
        return -1


class FakeConversionClient:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.progress_events: list[tuple[int, str]] = []
        self.uploads: list[tuple[str, str]] = []

    def progress(
        self,
        _job_id: str,
        _lease_token: str,
        value: int,
        stage: str,
        _message: str,
    ) -> dict[str, Any]:
        self.progress_events.append((value, stage))
        return {}

    def download_artifact(self, _artifact_id: str, target: Path) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.source, target)
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def upload_artifact(
        self,
        _job_id: str,
        _lease_token: str,
        kind: str,
        path: Path,
    ) -> dict[str, Any]:
        assert path.is_file()
        self.uploads.append((kind, path.name))
        return {"id": f"artifact_{kind}"}


def write_model(path: Path) -> None:
    source = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 384, 640])
    target = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, 384, 640])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "converter",
        [source],
        [target],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    onnx.save(model, path)


def write_legacy_deeplab_model(path: Path, *, decoder_scale: int = 4) -> None:
    source = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 512, 512])
    target = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3, 512, 512])
    decoder_scales = helper.make_tensor(
        "decoder_scales",
        TensorProto.FLOAT,
        [4],
        [1.0, 1.0, float(decoder_scale), float(decoder_scale)],
    )
    empty_roi = helper.make_tensor("empty_roi", TensorProto.FLOAT, [0], [])
    output_scales = helper.make_tensor(
        "output_scales", TensorProto.FLOAT, [4], [1.0, 1.0, 4.0, 4.0]
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["unused_constant"],
                value=helper.make_tensor("unused", TensorProto.FLOAT, [1], [1.0]),
            ),
            helper.make_node(
                "AveragePool",
                ["images"],
                ["decoder_input"],
                kernel_shape=[4 * decoder_scale, 4 * decoder_scale],
                strides=[4 * decoder_scale, 4 * decoder_scale],
            ),
            helper.make_node(
                "Resize",
                ["decoder_input", "empty_roi", "decoder_scales"],
                ["decoder_output"],
                name="decoder_resize",
                mode="linear",
                coordinate_transformation_mode="align_corners",
            ),
            helper.make_node("Identity", ["decoder_output"], ["deployment_logits"]),
            helper.make_node(
                "Resize",
                ["deployment_logits", "", "output_scales"],
                ["logits"],
                name="terminal_resize",
                mode="linear",
                coordinate_transformation_mode="half_pixel",
            ),
        ],
        "legacy-deeplab",
        [source],
        [target],
        [empty_roi, decoder_scales, output_scales],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    onnx.save(model, path)


def manifest(path: Path) -> dict[str, Any]:
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "variant": "yolov8n",
        "resolution": {"width": 640, "height": 384},
        "onnxSha256": checksum,
        "opset": 12,
        "input": {
            "name": "images",
            "layout": "NCHW",
            "shape": [1, 3, 384, 640],
        },
        "outputs": [{"name": "output0", "semantic": "detections"}],
        "preprocessing": {"mean": [0, 0, 0], "std": [255, 255, 255]},
        "rknn": {
            "targetPlatform": "rk3588",
            "quantizedAlgorithm": "normal",
            "optimizationLevel": 3,
        },
    }


def deeplab_manifest(path: Path) -> dict[str, Any]:
    value = manifest(path)
    value.update(
        {
            "profileId": "deeplabv3plus",
            "variant": "mobilenet_v2",
            "resolution": {"width": 512, "height": 512},
            "labels": ["background", "ng", "scratch"],
            "input": {
                "name": "images",
                "layout": "NCHW",
                "shape": [1, 3, 512, 512],
            },
            "outputs": [{"name": "logits", "semantic": "semantic_logits"}],
        }
    )
    return value


def test_legacy_deeplab_terminal_resize_is_removed(tmp_path: Path) -> None:
    source = tmp_path / "legacy.onnx"
    optimized = tmp_path / "optimized.onnx"
    write_legacy_deeplab_model(source)

    result = optimize_conversion_graph(source, deeplab_manifest(source), optimized)
    model = onnx.load(optimized)

    assert result.temporary is True
    assert result.details == {
        "applied": True,
        "name": "deeplab_promote_pre_resize_logits",
        "removedNode": "terminal_resize",
        "sourceOutputShape": [1, 3, 512, 512],
        "optimizedOutputShape": [1, 3, 128, 128],
        "scaleFactor": 4,
        "rewrittenResizeNodes": ["decoder_resize"],
    }
    output_shape = [
        dimension.dim_value for dimension in model.graph.output[0].type.tensor_type.shape.dim
    ]
    assert output_shape == [
        1,
        3,
        128,
        128,
    ]
    assert model.graph.output[0].name == "logits"
    resize_nodes = [node for node in model.graph.node if node.op_type == "Resize"]
    assert [node.name for node in resize_nodes] == ["decoder_resize"]
    coordinate_mode = next(
        item for item in resize_nodes[0].attribute if item.name == "coordinate_transformation_mode"
    )
    assert helper.get_attribute_value(coordinate_mode) == b"half_pixel"


def test_non_fourfold_decoder_resize_is_not_rewritten(tmp_path: Path) -> None:
    source = tmp_path / "legacy.onnx"
    optimized = tmp_path / "optimized.onnx"
    write_legacy_deeplab_model(source, decoder_scale=2)

    result = optimize_conversion_graph(source, deeplab_manifest(source), optimized)
    model = onnx.load(optimized)
    resize_node = next(node for node in model.graph.node if node.op_type == "Resize")
    coordinate_mode = next(
        item for item in resize_node.attribute if item.name == "coordinate_transformation_mode"
    )

    assert result.details["rewrittenResizeNodes"] == []
    assert helper.get_attribute_value(coordinate_mode) == b"align_corners"


def test_fp16_conversion_requires_runtime_inference_before_ready(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    target = tmp_path / "model.rknn"
    write_model(source)
    progress: list[tuple[int, str]] = []

    result = RknnConverter(FakeBackend).convert(
        onnx_path=source,
        manifest=manifest(source),
        precision="fp16",
        output_path=target,
        calibration_list=None,
        progress=lambda value, stage, _message: progress.append((value, stage)),
    )

    assert result.report["deploymentReady"] is True
    assert result.report["runtimeInitialized"] is True
    assert result.report["outputShapes"] == [[1, 10, 6]]
    assert result.report["benchmark"]["warmupRuns"] == 5
    assert result.report["benchmark"]["measuredRuns"] == 20
    assert result.report["performanceReady"] is True
    assert result.report["performance"]["cpuFallbackDetected"] is False
    assert target.read_bytes() == b"rknn-test"
    assert FakeBackend.latest is not None and FakeBackend.latest.released
    assert progress == [
        (20, "validate_onnx"),
        (24, "verify_integrity"),
        (26, "optimize_graph"),
        (28, "initialize_toolkit"),
        (32, "configure_rknn"),
        (40, "load_onnx"),
        (50, "build_rknn"),
        (75, "export_rknn"),
        (83, "initialize_runtime"),
        (86, "validate_inference"),
        (88, "benchmark_runtime"),
        (90, "audit_performance"),
    ]


def test_int8_build_receives_calibration_list(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    target = tmp_path / "model.rknn"
    dataset = tmp_path / "dataset.txt"
    dataset.write_text("image.jpg\n", encoding="utf-8")
    write_model(source)

    RknnConverter(FakeBackend).convert(
        onnx_path=source,
        manifest=manifest(source),
        precision="int8",
        output_path=target,
        calibration_list=dataset,
    )

    build = next(
        value
        for backend in reversed(FakeBackend.instances)
        for name, value in backend.calls
        if name == "build"
    )
    assert build == {"quantized": True, "dataset": str(dataset)}


def test_conversion_executor_reports_monotonic_weighted_progress(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    write_model(source)
    client = FakeConversionClient(source)
    claim = {
        "leaseToken": "lease_test",
        "job": {
            "id": "convert_progress",
            "spec": {
                "precision": "fp16",
                "manifest": manifest(source),
                "sourceArtifact": {
                    "id": "artifact_source",
                    "filename": "model.onnx",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
            },
        },
    }

    result = ConversionExecutor(RknnConverter(FakeBackend)).execute(
        claim,
        client,  # type: ignore[arg-type]
        tmp_path / "workspace",
    )

    assert [value for value, _stage in client.progress_events] == [
        3,
        8,
        20,
        24,
        26,
        28,
        32,
        40,
        50,
        75,
        83,
        86,
        88,
        90,
        92,
        95,
        98,
    ]
    assert client.uploads == [
        ("rknn", "yolov8n-640x384.rknn"),
        ("validation_report", "validation-report.json"),
        ("conversion_log", "conversion.log"),
    ]
    assert result["deploymentReady"] is True
    assert result["performanceReady"] is True


def test_performance_report_detects_non_io_cpu_fallback() -> None:
    report = _parse_performance_report(
        """Total Operator Elapsed Per Frame Time(us): 108690
OpType             CallNumber   CPUTime(us)  GPUTime(us)  NPUTime(us)  TotalTime(us)  TimeRatio(%)
Resize             3            60554        0            254          60808          55.95%
Conv               15           0            0            10970        10970          10.09%
InputOperator      1            10           0            0            10             0.01%
OutputOperator     1            12           0            0            12             0.01%
"""
    )

    assert report["available"] is True
    assert report["cpuFallbackDetected"] is True
    assert report["cpuFallbackTimeUs"] == 60554
    assert report["cpuFallbackOperators"] == [
        {
            "opType": "Resize",
            "calls": 3,
            "cpuTimeUs": 60554,
            "gpuTimeUs": 0,
            "npuTimeUs": 254,
            "totalTimeUs": 60808,
            "timeRatio": 0.5595,
        }
    ]


def test_runtime_failure_is_not_deployment_ready(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    write_model(source)

    with pytest.raises(ConversionError, match="runtime_init"):
        RknnConverter(RuntimeFailureBackend).convert(
            onnx_path=source,
            manifest=manifest(source),
            precision="fp16",
            output_path=tmp_path / "model.rknn",
            calibration_list=None,
        )


def test_calibration_archive_builds_bounded_image_list(tmp_path: Path) -> None:
    archive = tmp_path / "calibration.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("images/b.jpg", io.BytesIO(b"b").getvalue())
        target.writestr("images/a.png", io.BytesIO(b"a").getvalue())
        target.writestr("labels.txt", json.dumps({"unused": True}))

    list_path = create_calibration_list(archive, tmp_path / "work", max_samples=1)

    lines = list_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("a.png")
