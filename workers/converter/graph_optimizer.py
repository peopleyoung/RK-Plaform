from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from workers.common.onnx_contract import inspect_onnx


class GraphOptimizationError(ValueError):
    pass


@dataclass(frozen=True)
class GraphOptimization:
    model_path: Path
    details: dict[str, Any]
    temporary: bool = False


def optimize_conversion_graph(
    source_path: Path,
    manifest: dict[str, Any],
    destination_path: Path,
) -> GraphOptimization:
    if manifest.get("profileId") != "deeplabv3plus":
        return GraphOptimization(
            source_path,
            {"applied": False, "reason": "profile_not_applicable"},
        )

    try:
        onnx = cast(Any, importlib.import_module("onnx"))
    except ImportError as error:
        raise GraphOptimizationError(
            "The onnx package is required for graph optimization"
        ) from error

    try:
        model = onnx.load(source_path, load_external_data=True)
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception as error:
        raise GraphOptimizationError(f"Cannot inspect DeepLab ONNX graph: {error}") from error

    if len(inferred.graph.output) != 1:
        raise GraphOptimizationError(
            f"DeepLab graph optimization requires one output, found {len(inferred.graph.output)}"
        )

    output = inferred.graph.output[0]
    output_shape = _tensor_shape(output)
    input_shape = _manifest_input_shape(manifest)
    if output_shape[-2:] != input_shape[-2:]:
        return GraphOptimization(
            source_path,
            {
                "applied": False,
                "reason": "output_is_already_deployment_logits",
                "outputShape": output_shape,
            },
        )

    producers = {
        tensor_name: node for node in inferred.graph.node for tensor_name in node.output
    }
    final_resize = producers.get(output.name)
    if final_resize is None or final_resize.op_type != "Resize":
        raise GraphOptimizationError(
            "Full-resolution DeepLab output is not produced by one terminal Resize"
        )
    if not final_resize.input or not final_resize.input[0]:
        raise GraphOptimizationError("Terminal DeepLab Resize has no data input")

    deployment_tensor = final_resize.input[0]
    shapes = _tensor_shapes(inferred)
    deployment_shape = shapes.get(deployment_tensor)
    if deployment_shape is None:
        raise GraphOptimizationError(
            f"Cannot resolve terminal Resize input shape for '{deployment_tensor}'"
        )
    _validate_deployment_shape(deployment_shape, output_shape, manifest)

    consumers = [
        node for node in inferred.graph.node if deployment_tensor in set(node.input)
    ]
    if len(consumers) != 1 or consumers[0].output[0] != output.name:
        raise GraphOptimizationError(
            "Terminal Resize input is shared and cannot be safely promoted to graph output"
        )

    optimized = onnx.ModelProto()
    optimized.CopyFrom(model)
    optimized_output = optimized.graph.output[0]
    final_node = next(
        (
            node
            for node in optimized.graph.node
            if output.name in node.output and node.op_type == "Resize"
        ),
        None,
    )
    if final_node is None:
        raise GraphOptimizationError("Terminal Resize disappeared while copying the ONNX graph")
    optimized.graph.node.remove(final_node)

    renamed = False
    for node in optimized.graph.node:
        for index, tensor_name in enumerate(node.output):
            if tensor_name == deployment_tensor:
                node.output[index] = optimized_output.name
                renamed = True
    if not renamed:
        raise GraphOptimizationError(
            f"Cannot find producer for terminal Resize input '{deployment_tensor}'"
        )

    rewritten_resize_nodes = _rewrite_rknn_compatible_decoder_resizes(optimized, shapes)
    shape = optimized_output.type.tensor_type.shape
    shape.ClearField("dim")
    for value in deployment_shape:
        shape.dim.add().dim_value = value
    _prune_unreachable_graph(optimized)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        onnx.checker.check_model(optimized, full_check=False)
        onnx.save_model(optimized, destination_path)
        optimized_graph = inspect_onnx(destination_path)
    except Exception as error:
        destination_path.unlink(missing_ok=True)
        raise GraphOptimizationError(f"Optimized DeepLab ONNX is invalid: {error}") from error

    actual_shape = optimized_graph.outputs[0].shape
    if optimized_graph.outputs[0].name != output.name or actual_shape != deployment_shape:
        destination_path.unlink(missing_ok=True)
        raise GraphOptimizationError(
            "Optimized DeepLab output contract does not match the promoted logits tensor"
        )

    return GraphOptimization(
        destination_path,
        {
            "applied": True,
            "name": "deeplab_promote_pre_resize_logits",
            "removedNode": final_resize.name or final_resize.op_type,
            "sourceOutputShape": output_shape,
            "optimizedOutputShape": deployment_shape,
            "scaleFactor": output_shape[-1] // deployment_shape[-1],
            "rewrittenResizeNodes": rewritten_resize_nodes,
        },
        temporary=True,
    )


def _manifest_input_shape(manifest: dict[str, Any]) -> list[int]:
    raw_input = manifest.get("input")
    if not isinstance(raw_input, dict):
        raise GraphOptimizationError("DeepLab graph optimization requires an NCHW input")
    typed_input = cast(dict[str, object], raw_input)
    if typed_input.get("layout") != "NCHW":
        raise GraphOptimizationError("DeepLab graph optimization requires an NCHW input")
    raw_shape = typed_input.get("shape")
    if not isinstance(raw_shape, list):
        raise GraphOptimizationError("DeepLab manifest input shape must be four positive integers")
    typed_shape = cast(list[object], raw_shape)
    if len(typed_shape) != 4 or not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in typed_shape
    ):
        raise GraphOptimizationError("DeepLab manifest input shape must be four positive integers")
    return cast(list[int], typed_shape)


def _tensor_shapes(model: Any) -> dict[str, list[int]]:
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    shapes: dict[str, list[int]] = {}
    for value in values:
        dimensions = value.type.tensor_type.shape.dim
        if len(dimensions) != 4 or any(not item.HasField("dim_value") for item in dimensions):
            continue
        shape = [int(item.dim_value) for item in dimensions]
        if all(item > 0 for item in shape):
            shapes[value.name] = shape
    return shapes


def _tensor_shape(value: Any) -> list[int]:
    dimensions = value.type.tensor_type.shape.dim
    if len(dimensions) != 4 or any(not item.HasField("dim_value") for item in dimensions):
        raise GraphOptimizationError(
            f"Tensor '{value.name}' must have a static four-dimensional shape"
        )
    shape = [int(item.dim_value) for item in dimensions]
    if any(item <= 0 for item in shape):
        raise GraphOptimizationError(f"Tensor '{value.name}' has an invalid shape {shape}")
    return shape


def _validate_deployment_shape(
    deployment_shape: list[int],
    output_shape: list[int],
    manifest: dict[str, Any],
) -> None:
    if deployment_shape[:2] != output_shape[:2]:
        raise GraphOptimizationError(
            "Terminal Resize changes the DeepLab batch or class-channel dimension"
        )
    if output_shape[-2] != deployment_shape[-2] * 4 or output_shape[-1] != deployment_shape[-1] * 4:
        raise GraphOptimizationError(
            "Terminal DeepLab Resize must promote 1/4-resolution logits to full resolution"
        )
    labels = manifest.get("labels")
    typed_labels = cast(list[object], labels) if isinstance(labels, list) else []
    if typed_labels and deployment_shape[1] != len(typed_labels):
        raise GraphOptimizationError(
            "DeepLab logits channel count does not match the manifest label count"
        )


def _rewrite_rknn_compatible_decoder_resizes(
    model: Any,
    shapes: dict[str, list[int]],
) -> list[str]:
    rewritten: list[str] = []
    for node in model.graph.node:
        if node.op_type != "Resize" or not node.input or not node.output:
            continue
        attributes = {item.name: item for item in node.attribute}
        mode = attributes.get("mode")
        coordinate_mode = attributes.get("coordinate_transformation_mode")
        if (
            mode is None
            or mode.s != b"linear"
            or coordinate_mode is None
            or coordinate_mode.s != b"align_corners"
        ):
            continue

        input_shape = shapes.get(node.input[0])
        output_shape = shapes.get(node.output[0])
        if input_shape is None or output_shape is None:
            continue
        if input_shape[:2] != output_shape[:2]:
            continue
        if (
            output_shape[-2] != input_shape[-2] * 4
            or output_shape[-1] != input_shape[-1] * 4
        ):
            continue

        coordinate_mode.s = b"half_pixel"
        rewritten.append(node.name or node.output[0])
    return rewritten


def _prune_unreachable_graph(model: Any) -> None:
    required = {output.name for output in model.graph.output}
    kept_nodes: list[Any] = []
    for node in reversed(model.graph.node):
        if any(output in required for output in node.output):
            kept_nodes.append(node)
            required.update(item for item in node.input if item)
    kept_nodes.reverse()
    model.graph.ClearField("node")
    model.graph.node.extend(kept_nodes)

    kept_initializers = [item for item in model.graph.initializer if item.name in required]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept_initializers)

    valid_tensors = {
        *required,
        *(item.name for item in model.graph.input),
        *(item.name for item in model.graph.output),
        *(name for node in kept_nodes for name in node.output),
    }
    kept_value_info = [item for item in model.graph.value_info if item.name in valid_tensors]
    model.graph.ClearField("value_info")
    model.graph.value_info.extend(kept_value_info)
