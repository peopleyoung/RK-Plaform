from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from workers.common.onnx_contract import validate_onnx_manifest
from workers.converter.graph_optimizer import GraphOptimizationError, optimize_conversion_graph

BENCHMARK_WARMUP_RUNS = 5
BENCHMARK_MEASURED_RUNS = 20
EXPECTED_CPU_OPERATORS = frozenset({"InputOperator", "OutputOperator"})


class ConversionError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"RKNN {stage} failed: {message}")
        self.stage = stage


class RknnBackend(Protocol):
    @property
    def toolkit_version(self) -> str: ...

    def config(self, **values: Any) -> int: ...

    def load_onnx(self, model: str) -> int: ...

    def load_rknn(self, path: str) -> int: ...

    def build(self, *, do_quantization: bool, dataset: str | None) -> int: ...

    def export_rknn(self, path: str) -> int: ...

    def init_runtime(self, *, target: str, perf_debug: bool = False) -> int: ...

    def inference(self, inputs: list[Any], *, data_format: str) -> list[Any] | None: ...

    def eval_perf(self, *, is_print: bool, fix_freq: bool) -> str: ...

    def release(self) -> None: ...


class Toolkit2Backend:
    def __init__(self) -> None:
        try:
            module = cast(Any, importlib.import_module("rknn.api"))
        except ImportError as error:
            raise ConversionError(
                "initialize", "rknn-toolkit2 is not installed in this worker image"
            ) from error
        self._rknn: Any = module.RKNN(verbose=True)

    @property
    def toolkit_version(self) -> str:
        for distribution in ("rknn-toolkit2", "rknn_toolkit2"):
            try:
                return importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
        return "unknown"

    def config(self, **values: Any) -> int:
        return int(self._rknn.config(**values))

    def load_onnx(self, model: str) -> int:
        return int(self._rknn.load_onnx(model=model))

    def load_rknn(self, path: str) -> int:
        return int(self._rknn.load_rknn(path))

    def build(self, *, do_quantization: bool, dataset: str | None) -> int:
        return int(self._rknn.build(do_quantization=do_quantization, dataset=dataset))

    def export_rknn(self, path: str) -> int:
        return int(self._rknn.export_rknn(path))

    def init_runtime(self, *, target: str, perf_debug: bool = False) -> int:
        return int(self._rknn.init_runtime(target=target, perf_debug=perf_debug))

    def inference(self, inputs: list[Any], *, data_format: str) -> list[Any] | None:
        value: object = self._rknn.inference(inputs=inputs, data_format=data_format)
        if value is None:
            return None
        if not isinstance(value, list):
            raise ConversionError("inference", "Toolkit returned a non-list result")
        return cast(list[Any], value)

    def eval_perf(self, *, is_print: bool, fix_freq: bool) -> str:
        value: object = self._rknn.eval_perf(is_print=is_print, fix_freq=fix_freq)
        if not isinstance(value, str) or not value.strip():
            raise ConversionError("performance_audit", "Toolkit returned no performance report")
        return value

    def release(self) -> None:
        self._rknn.release()


@dataclass(frozen=True)
class ConversionResult:
    report: dict[str, Any]
    log: str


class RknnConverter:
    def __init__(self, backend_factory: type[RknnBackend] | None = None) -> None:
        self.backend_factory = backend_factory or Toolkit2Backend

    def convert(
        self,
        *,
        onnx_path: Path,
        manifest: dict[str, Any],
        precision: str,
        output_path: Path,
        calibration_list: Path | None,
        progress: Callable[[int, str, str], None] | None = None,
    ) -> ConversionResult:
        _report_progress(progress, 20, "validate_onnx", "Validating ONNX graph contract")
        graph = validate_onnx_manifest(onnx_path, manifest)
        _report_progress(progress, 24, "verify_integrity", "Verifying ONNX integrity")
        expected_checksum = manifest.get("onnxSha256")
        actual_checksum = _sha256(onnx_path)
        if actual_checksum != expected_checksum:
            raise ConversionError(
                "integrity",
                f"ONNX checksum mismatch: expected {expected_checksum}, got {actual_checksum}",
            )
        if precision not in {"int8", "fp16"}:
            raise ConversionError("configuration", f"Unsupported precision '{precision}'")
        if precision == "int8" and calibration_list is None:
            raise ConversionError("configuration", "INT8 conversion requires a calibration list")

        _report_progress(
            progress,
            26,
            "optimize_graph",
            "Optimizing deployment graph for RK3588",
        )
        try:
            graph_optimization = optimize_conversion_graph(
                onnx_path,
                manifest,
                output_path.with_suffix(".optimized.onnx"),
            )
        except GraphOptimizationError as error:
            raise ConversionError("graph_optimization", str(error)) from error

        rknn_profile = _mapping(manifest, "rknn")
        preprocessing = _mapping(manifest, "preprocessing")
        target = _string(rknn_profile, "targetPlatform")
        _report_progress(progress, 28, "initialize_toolkit", "Initializing RKNN Toolkit2")
        backend = self.backend_factory()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_capture = io.StringIO()
        runtime_initialized = False
        inference_succeeded = False
        output_shapes: list[list[int]] = []
        benchmark: dict[str, int | float] = {}
        performance: dict[str, Any] = {
            "available": False,
            "cpuFallbackDetected": False,
            "cpuFallbackOperators": [],
        }
        toolkit_version = backend.toolkit_version
        try:
            with (
                contextlib.redirect_stdout(output_capture),
                contextlib.redirect_stderr(output_capture),
            ):
                _report_progress(progress, 32, "configure_rknn", "Configuring RKNN conversion")
                self._check(
                    "config",
                    backend.config(
                        mean_values=[_number_list(preprocessing, "mean")],
                        std_values=[_number_list(preprocessing, "std")],
                        target_platform=target,
                        quantized_algorithm=_string(rknn_profile, "quantizedAlgorithm"),
                        optimization_level=_integer(rknn_profile, "optimizationLevel"),
                    ),
                )
                _report_progress(progress, 40, "load_onnx", "Loading ONNX model")
                self._check("load", backend.load_onnx(str(graph_optimization.model_path)))
                _report_progress(progress, 50, "build_rknn", "Building RKNN model")
                self._check(
                    "build",
                    backend.build(
                        do_quantization=precision == "int8",
                        dataset=str(calibration_list) if calibration_list else None,
                    ),
                )
                _report_progress(progress, 75, "export_rknn", "Exporting RKNN model")
                self._check("export", backend.export_rknn(str(output_path)))
                _report_progress(
                    progress,
                    83,
                    "initialize_runtime",
                    "Initializing RK3588 runtime",
                )
                self._check("runtime_init", backend.init_runtime(target=target))
                runtime_initialized = True
                _report_progress(
                    progress,
                    86,
                    "validate_inference",
                    "Running RK3588 validation inference",
                )
                sample = _deterministic_sample(manifest)
                outputs = _require_outputs(
                    backend.inference([sample], data_format="nhwc")
                )
                output_shapes = [_shape(item) for item in outputs]
                inference_succeeded = True
                _report_progress(
                    progress,
                    88,
                    "benchmark_runtime",
                    "Benchmarking warmed RK3588 inference",
                )
                benchmark = _benchmark_inference(backend, sample)
        finally:
            backend.release()
            if graph_optimization.temporary:
                graph_optimization.model_path.unlink(missing_ok=True)

        _report_progress(
            progress,
            90,
            "audit_performance",
            "Auditing RKNN CPU and NPU operator placement",
        )
        audit_backend = self.backend_factory()
        try:
            with (
                contextlib.redirect_stdout(output_capture),
                contextlib.redirect_stderr(output_capture),
            ):
                self._check("audit_load", audit_backend.load_rknn(str(output_path)))
                self._check(
                    "audit_runtime_init",
                    audit_backend.init_runtime(target=target, perf_debug=True),
                )
                audit_outputs = audit_backend.inference([sample], data_format="nhwc")
                _require_outputs(audit_outputs)
                raw_performance = audit_backend.eval_perf(is_print=False, fix_freq=False)
                print(raw_performance)
                performance = _parse_performance_report(raw_performance)
        except Exception as error:
            performance = {
                "available": False,
                "cpuFallbackDetected": False,
                "cpuFallbackOperators": [],
                "error": str(error),
            }
            print(f"RKNN performance audit unavailable: {error}", file=output_capture)
        finally:
            audit_backend.release()

        performance_ready = bool(
            performance.get("available") and not performance.get("cpuFallbackDetected")
        )

        report = {
            "schemaVersion": 2,
            "toolkitVersion": toolkit_version,
            "sourceOnnxSha256": actual_checksum,
            "targetPlatform": target,
            "precision": precision,
            "inputShape": manifest["input"]["shape"],
            "onnxGraph": graph.to_dict(),
            "graphOptimization": graph_optimization.details,
            "converted": output_path.is_file(),
            "runtimeInitialized": runtime_initialized,
            "inferenceSucceeded": inference_succeeded,
            "outputShapes": output_shapes,
            "benchmark": benchmark,
            "performance": performance,
            "performanceReady": performance_ready,
            "deploymentReady": output_path.is_file()
            and runtime_initialized
            and inference_succeeded,
        }
        if not report["deploymentReady"]:
            raise ConversionError("validation", "RKNN artifact did not pass runtime validation")
        return ConversionResult(report=report, log=output_capture.getvalue())

    @staticmethod
    def _check(stage: str, return_code: int) -> None:
        if return_code != 0:
            raise ConversionError(stage, f"toolkit returned {return_code}")


def _report_progress(
    reporter: Callable[[int, str, str], None] | None,
    value: int,
    stage: str,
    message: str,
) -> None:
    if reporter is not None:
        reporter(value, stage, message)


def _deterministic_sample(manifest: dict[str, Any]) -> Any:
    try:
        numpy = cast(Any, importlib.import_module("numpy"))
    except ImportError as error:
        raise ConversionError("inference", "numpy is required for runtime validation") from error
    input_contract = _mapping(manifest, "input")
    raw_shape = input_contract.get("shape")
    if not isinstance(raw_shape, list):
        raise ConversionError("inference", "Manifest input shape is invalid")
    shape = cast(list[object], raw_shape)
    if len(shape) != 4:
        raise ConversionError("inference", "Manifest input shape is invalid")
    if not all(isinstance(item, int) and item > 0 for item in shape):
        raise ConversionError("inference", "Manifest input shape must be static and positive")
    layout = _string(input_contract, "layout")
    typed_shape = cast(list[int], shape)
    if layout == "NCHW":
        batch, channels, height, width = typed_shape
        runtime_shape = (batch, height, width, channels)
    elif layout == "NHWC":
        runtime_shape = tuple(typed_shape)
    else:
        raise ConversionError("inference", f"Unsupported input layout '{layout}'")
    return numpy.zeros(runtime_shape, dtype=numpy.uint8)


def _require_outputs(outputs: list[Any] | None) -> list[Any]:
    if outputs is None or not outputs:
        raise ConversionError("inference", "Runtime returned no outputs")
    return outputs


def _benchmark_inference(backend: RknnBackend, sample: Any) -> dict[str, int | float]:
    for _ in range(BENCHMARK_WARMUP_RUNS):
        _require_outputs(backend.inference([sample], data_format="nhwc"))

    samples_ms: list[float] = []
    for _ in range(BENCHMARK_MEASURED_RUNS):
        started = time.perf_counter()
        _require_outputs(backend.inference([sample], data_format="nhwc"))
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(samples_ms)
    average_ms = statistics.fmean(samples_ms)
    p95_index = max(0, min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1))
    return {
        "warmupRuns": BENCHMARK_WARMUP_RUNS,
        "measuredRuns": BENCHMARK_MEASURED_RUNS,
        "averageMs": round(average_ms, 3),
        "p50Ms": round(statistics.median(samples_ms), 3),
        "p95Ms": round(ordered[p95_index], 3),
        "minMs": round(ordered[0], 3),
        "maxMs": round(ordered[-1], 3),
        "fps": round(1000.0 / average_ms, 3) if average_ms > 0 else 0.0,
    }


def _parse_performance_report(raw: str) -> dict[str, Any]:
    total_match = re.search(r"Total Operator Elapsed Per Frame Time\(us\):\s*(\d+)", raw)
    ranking_header = "OpType             CallNumber"
    ranking = raw.split(ranking_header, 1)[1] if ranking_header in raw else ""
    row_pattern = re.compile(
        r"^(?P<op>\S+)\s+(?P<calls>\d+)\s+(?P<cpu>\d+)\s+(?P<gpu>\d+)\s+"
        r"(?P<npu>\d+)\s+(?P<total>\d+)\s+(?P<ratio>\d+(?:\.\d+)?)%\s*$"
    )
    rows: list[dict[str, int | float | str]] = []
    for line in ranking.splitlines():
        match = row_pattern.match(line.strip())
        if match is None:
            continue
        rows.append(
            {
                "opType": match.group("op"),
                "calls": int(match.group("calls")),
                "cpuTimeUs": int(match.group("cpu")),
                "gpuTimeUs": int(match.group("gpu")),
                "npuTimeUs": int(match.group("npu")),
                "totalTimeUs": int(match.group("total")),
                "timeRatio": float(match.group("ratio")) / 100.0,
            }
        )

    if total_match is None or not rows:
        return {
            "available": False,
            "cpuFallbackDetected": False,
            "cpuFallbackOperators": [],
            "error": "RKNN performance report format was not recognized",
        }

    total_time_us = int(total_match.group(1))
    cpu_time_us = sum(int(row["cpuTimeUs"]) for row in rows)
    gpu_time_us = sum(int(row["gpuTimeUs"]) for row in rows)
    npu_time_us = sum(int(row["npuTimeUs"]) for row in rows)
    fallbacks = [
        row
        for row in rows
        if int(row["cpuTimeUs"]) > 0 and row["opType"] not in EXPECTED_CPU_OPERATORS
    ]
    fallback_time_us = sum(int(row["cpuTimeUs"]) for row in fallbacks)
    top_operators = sorted(rows, key=lambda row: int(row["totalTimeUs"]), reverse=True)[:8]
    return {
        "available": True,
        "instrumented": True,
        "totalOperatorTimeUs": total_time_us,
        "cpuTimeUs": cpu_time_us,
        "gpuTimeUs": gpu_time_us,
        "npuTimeUs": npu_time_us,
        "cpuFallbackDetected": bool(fallbacks),
        "cpuFallbackTimeUs": fallback_time_us,
        "cpuFallbackRatio": round(fallback_time_us / total_time_us, 6),
        "cpuFallbackOperators": fallbacks,
        "topOperators": top_operators,
    }


def _shape(value: Any) -> list[int]:
    raw_shape: object = getattr(value, "shape", None)
    if not isinstance(raw_shape, list | tuple):
        raise ConversionError("inference", "Runtime output has no shape")
    dimensions = list(cast(list[object] | tuple[object, ...], raw_shape))
    if not all(isinstance(item, int) for item in dimensions):
        raise ConversionError("inference", "Runtime output shape is not integral")
    return cast(list[int], dimensions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ConversionError("manifest", f"{key} must be an object")
    return cast(dict[str, Any], item)


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ConversionError("manifest", f"{key} must be a non-empty string")
    return item


def _integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int):
        raise ConversionError("manifest", f"{key} must be an integer")
    return item


def _number_list(value: dict[str, Any], key: str) -> list[float]:
    item = value.get(key)
    if not isinstance(item, list) or not all(
        isinstance(number, int | float) for number in cast(list[object], item)
    ):
        raise ConversionError("manifest", f"{key} must be a numeric list")
    return [float(number) for number in cast(list[int | float], item)]
