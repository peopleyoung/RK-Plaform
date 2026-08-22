from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper
from workers.common.onnx_contract import (
    OnnxContractError,
    inspect_onnx,
    validate_onnx_manifest,
)


def write_identity_model(path: Path, shape: list[int | str], *, opset: int = 12) -> None:
    source = helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)
    target = helper.make_tensor_value_info("logits", TensorProto.FLOAT, shape)
    node = helper.make_node("Identity", ["images"], ["logits"])
    graph = helper.make_graph([node], "identity", [source], [target])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.save(model, path)


def manifest(shape: list[int]) -> dict[str, object]:
    return {
        "opset": 12,
        "input": {"name": "images", "shape": shape},
        "outputs": [{"name": "logits", "semantic": "semantic_logits"}],
    }


def test_inspect_and_validate_static_custom_resolution(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    write_identity_model(path, [1, 3, 384, 640])

    graph = validate_onnx_manifest(path, manifest([1, 3, 384, 640]))

    assert graph.inputs[0].shape == [1, 3, 384, 640]
    assert graph.opsets["ai.onnx"] == 12
    assert graph.ir_version == 10


def test_rejects_ir_version_newer_than_toolkit_support(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    write_identity_model(path, [1, 3, 384, 640])
    model = onnx.load(path)
    model.ir_version = 11
    onnx.save(model, path)

    with pytest.raises(OnnxContractError, match="expected <= 10, got 11"):
        validate_onnx_manifest(path, manifest([1, 3, 384, 640]))


def test_rejects_dynamic_or_mismatched_input_shape(tmp_path: Path) -> None:
    path = tmp_path / "dynamic.onnx"
    write_identity_model(path, [1, 3, "height", "width"])

    with pytest.raises(OnnxContractError, match="input shape mismatch"):
        validate_onnx_manifest(path, manifest([1, 3, 384, 640]))


def test_rejects_output_and_opset_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    write_identity_model(path, [1, 3, 384, 640], opset=13)
    bad = manifest([1, 3, 384, 640])
    bad["outputs"] = [{"name": "other", "semantic": "wrong"}]

    with pytest.raises(OnnxContractError, match="opset mismatch"):
        validate_onnx_manifest(path, bad)

    assert inspect_onnx(path).outputs[0].name == "logits"


def test_accepts_official_graph_with_missing_output_shape(tmp_path: Path) -> None:
    path = tmp_path / "missing-output-shape.onnx"
    write_identity_model(path, [1, 3, 48, 320])
    model = onnx.load(path)
    model.graph.output[0].type.tensor_type.ClearField("shape")
    onnx.save(model, path)

    graph = validate_onnx_manifest(path, manifest([1, 3, 48, 320]))

    assert graph.outputs[0].name == "logits"
    assert graph.outputs[0].shape == []
