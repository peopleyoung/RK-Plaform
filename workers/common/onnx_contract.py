from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


class OnnxContractError(ValueError):
    pass


RKNN_TOOLKIT2_MAX_IR_VERSION = 10


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: list[int | str | None]
    dtype: str


@dataclass(frozen=True)
class OnnxGraphInfo:
    ir_version: int
    opsets: dict[str, int]
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_onnx(path: Path) -> OnnxGraphInfo:
    try:
        onnx_module = cast(Any, importlib.import_module("onnx"))
    except ImportError as error:
        raise OnnxContractError("The onnx package is required to inspect model graphs") from error

    try:
        model: Any = onnx_module.load(path, load_external_data=False)
        _check_model(onnx_module, model)
    except Exception as error:
        raise OnnxContractError(f"Invalid ONNX model: {error}") from error

    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [
        _tensor_info(onnx_module, item)
        for item in model.graph.input
        if item.name not in initializer_names
    ]
    outputs = [_tensor_info(onnx_module, item) for item in model.graph.output]
    opsets = {(item.domain or "ai.onnx"): item.version for item in model.opset_import}
    return OnnxGraphInfo(
        ir_version=int(model.ir_version),
        opsets=opsets,
        inputs=inputs,
        outputs=outputs,
    )


def _check_model(onnx_module: Any, model: Any) -> None:
    try:
        onnx_module.checker.check_model(model, full_check=False)
        return
    except Exception as error:
        missing_output_shape = "Field 'shape' of 'type' is required but missing" in str(error)
        outputs_without_shape = [
            output for output in model.graph.output if not output.type.tensor_type.HasField("shape")
        ]
        if not missing_output_shape or not outputs_without_shape:
            raise

    compatible_model = onnx_module.ModelProto()
    compatible_model.CopyFrom(model)
    for output in compatible_model.graph.output:
        if not output.type.tensor_type.HasField("shape"):
            output.type.tensor_type.shape.dim.add().dim_param = "__unknown_output_shape__"
    onnx_module.checker.check_model(compatible_model, full_check=False)


def validate_onnx_manifest(path: Path, manifest: dict[str, Any]) -> OnnxGraphInfo:
    graph = inspect_onnx(path)
    if graph.ir_version > RKNN_TOOLKIT2_MAX_IR_VERSION:
        raise OnnxContractError(
            "ONNX IR version is incompatible with RKNN Toolkit2 2.3.2: "
            f"expected <= {RKNN_TOOLKIT2_MAX_IR_VERSION}, got {graph.ir_version}"
        )
    expected_input = _require_mapping(manifest, "input")
    expected_name = _require_string(expected_input, "name")
    raw_expected_shape = expected_input.get("shape")
    if not isinstance(raw_expected_shape, list) or not all(
        isinstance(item, int) for item in cast(list[object], raw_expected_shape)
    ):
        raise OnnxContractError("Manifest input.shape must contain static integer dimensions")
    expected_shape = cast(list[int], raw_expected_shape)

    if len(graph.inputs) != 1:
        raise OnnxContractError(f"Expected exactly one ONNX graph input, found {len(graph.inputs)}")
    actual_input = graph.inputs[0]
    if actual_input.name != expected_name:
        raise OnnxContractError(
            f"ONNX input name mismatch: expected '{expected_name}', got '{actual_input.name}'"
        )
    if actual_input.shape != expected_shape:
        raise OnnxContractError(
            f"ONNX input shape mismatch: expected {expected_shape}, got {actual_input.shape}"
        )
    if any(not isinstance(item, int) or item <= 0 for item in actual_input.shape):
        raise OnnxContractError(f"ONNX input must be fully static, got {actual_input.shape}")

    expected_opset = manifest.get("opset")
    actual_opset = graph.opsets.get("ai.onnx")
    if not isinstance(expected_opset, int) or actual_opset != expected_opset:
        raise OnnxContractError(
            f"ONNX opset mismatch: expected {expected_opset}, got {actual_opset}"
        )

    raw_output_contracts = manifest.get("outputs")
    if not isinstance(raw_output_contracts, list) or not raw_output_contracts:
        raise OnnxContractError("Manifest must declare at least one output")
    expected_outputs: set[str] = set()
    for raw_contract in cast(list[object], raw_output_contracts):
        if not isinstance(raw_contract, dict):
            raise OnnxContractError("Manifest outputs must contain objects")
        contract = cast(dict[str, object], raw_contract)
        name = contract.get("name")
        if not isinstance(name, str) or not name:
            raise OnnxContractError("Manifest output name must be a non-empty string")
        expected_outputs.add(name)
    actual_outputs = {item.name for item in graph.outputs}
    if expected_outputs != actual_outputs:
        raise OnnxContractError(
            "ONNX output names mismatch: "
            f"expected {sorted(expected_outputs)}, got {sorted(actual_outputs)}"
        )
    return graph


def _tensor_info(onnx_module: Any, value: Any) -> TensorInfo:
    tensor_type = value.type.tensor_type
    shape: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(str(dimension.dim_param))
        else:
            shape.append(None)
    dtype = str(onnx_module.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type))
    return TensorInfo(name=value.name, shape=shape, dtype=dtype)


def _require_mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise OnnxContractError(f"Manifest {key} must be an object")
    return cast(dict[str, Any], item)


def _require_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise OnnxContractError(f"Manifest {key} must be a non-empty string")
    return item
