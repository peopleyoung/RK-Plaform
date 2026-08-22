from __future__ import annotations

import hashlib
from pathlib import Path

from backend.platform_api.contracts import (
    DeploymentManifest,
    OutputTensor,
    Resolution,
    TensorContract,
)
from backend.platform_api.profiles import ModelProfileRegistry

from workers.common.onnx_contract import inspect_onnx, validate_onnx_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_deployment_manifest(
    profiles: ModelProfileRegistry,
    *,
    job_id: str,
    profile_id: str,
    variant: str,
    resolution: Resolution,
    onnx_path: Path,
    labels: list[str],
) -> DeploymentManifest:
    profile = profiles.get(profile_id)
    variant_contract = profiles.variant_contract(profile_id, variant)
    graph = inspect_onnx(onnx_path)
    if len(graph.inputs) != 1:
        raise ValueError(f"Expected one deployment input, found {len(graph.inputs)}")
    graph_input = graph.inputs[0]
    outputs = [
        OutputTensor(name=item.name, semantic=variant_contract.output_contract)
        for item in graph.outputs
    ]
    manifest = DeploymentManifest(
        model_family=profile.family,
        profile_id=profile.id,
        variant=variant,
        task_type=profile.task_type,
        training_job_id=job_id,
        onnx_sha256=sha256_file(onnx_path),
        opset=variant_contract.opset,
        resolution=resolution,
        input=TensorContract(
            name=graph_input.name,
            layout=profile.input.layout,
            shape=[
                dimension if isinstance(dimension, int) else -1 for dimension in graph_input.shape
            ],
            dtype=graph_input.dtype,
            color_space=profile.input.color_space,
        ),
        preprocessing=profile.preprocessing,
        resize_policy=profile.input.resize_policy,
        output_contract=variant_contract.output_contract,
        outputs=outputs,
        labels=labels,
        supported_precisions=profile.precisions,
        rknn=profile.rknn,
    )
    profiles.validate_manifest(manifest)
    validate_onnx_manifest(onnx_path, manifest.model_dump(mode="json", by_alias=True))
    return manifest
