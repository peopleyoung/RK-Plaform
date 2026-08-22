from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import httpx
import onnx
from onnx import TensorProto, helper


def require(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    value: object = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected an object from {response.request.url}")
    return cast(dict[str, Any], value)


def create_dataset_archive(path: Path) -> None:
    sample = path.parent / "README.txt"
    image = path.parent / "calibration.png"
    sample.write_text("RKNode conversion end-to-end fixture\n", encoding="utf-8")
    image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )
    with tarfile.open(path, "w:gz") as archive:
        archive.add(sample, arcname="README.txt")
        archive.add(image, arcname="images/calibration.png")


def create_onnx(
    path: Path,
    height: int,
    width: int,
    *,
    input_name: str,
    output_name: str,
    opset: int,
) -> str:
    input_info = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [1, 3, height, width]
    )
    output_info = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [1, 3, height, width]
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", [input_name], [output_name])],
        "rknode-e2e",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.save(model, str(path))  # pyright: ignore[reportUnknownMemberType]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_onnx(path: Path) -> tuple[str, list[int], int, list[str]]:
    model = onnx.load(  # pyright: ignore[reportUnknownMemberType]
        str(path), load_external_data=False
    )
    if len(model.graph.input) != 1:
        raise RuntimeError("E2E validation requires an ONNX model with exactly one input")
    model_input = model.graph.input[0]
    shape = [dimension.dim_value for dimension in model_input.type.tensor_type.shape.dim]
    if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
        raise RuntimeError(f"E2E validation requires a static NCHW input, received {shape}")
    opsets = [item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}]
    if len(opsets) != 1:
        raise RuntimeError(f"Expected one default-domain ONNX opset, received {opsets}")
    outputs = [item.name for item in model.graph.output]
    if not outputs:
        raise RuntimeError("ONNX model has no graph outputs")
    return model_input.name, shape, opsets[0], outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the central API to RK3588 conversion flow")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--admin-token", required=True)
    parser.add_argument("--worker-token", required=True)
    parser.add_argument(
        "--profile-id",
        choices=("yolo-detect", "deeplabv3plus", "ppocr-det", "ppocr-rec"),
        default="yolo-detect",
    )
    parser.add_argument("--variant")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--precision", choices=("fp16", "int8"), default="fp16")
    parser.add_argument(
        "--onnx-path",
        type=Path,
        help="Use a real static ONNX model instead of generating a contract fixture",
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/") + "/api/v1"
    admin_headers = {"Authorization": f"Bearer {args.admin_token}"}
    worker_headers = {"Authorization": f"Bearer {args.worker_token}"}
    with tempfile.TemporaryDirectory(prefix="rknode-e2e-") as raw_dir:
        work_dir = Path(raw_dir)
        dataset_path = work_dir / "dataset.tar.gz"
        onnx_path = args.onnx_path.resolve() if args.onnx_path else work_dir / "model.onnx"
        if not onnx_path.is_file() and args.onnx_path:
            raise FileNotFoundError(onnx_path)
        create_dataset_archive(dataset_path)
        with httpx.Client(timeout=30) as client:
            profile_document = require(
                client.get(f"{base_url}/model-profiles", headers=admin_headers)
            )
            raw_profiles: object = profile_document.get("profiles")
            if not isinstance(raw_profiles, list):
                raise RuntimeError("Model profile response has no profiles")
            profiles = cast(list[dict[str, Any]], raw_profiles)
            profile = next(
                (
                    item
                    for item in profiles
                    if item.get("id") == args.profile_id
                ),
                None,
            )
            if profile is None:
                raise RuntimeError(f"Model profile '{args.profile_id}' was not found")
            variants = cast(list[str], profile["variants"])
            variant = args.variant or variants[0]
            if variant not in variants:
                raise RuntimeError(f"Variant '{variant}' is not supported by {args.profile_id}")
            default_resolution = cast(dict[str, int], profile["defaultResolution"])
            width = args.width or default_resolution["width"]
            height = args.height or default_resolution["height"]
            input_profile = cast(dict[str, Any], profile["input"])
            variant_contracts = cast(dict[str, dict[str, Any]], profile["variantContracts"])
            variant_contract = variant_contracts.get(variant)
            export_profile = cast(dict[str, Any], profile["export"])
            opset = int(
                variant_contract["opset"]
                if variant_contract is not None
                else export_profile["opset"]
            )
            output_contract = str(
                variant_contract["outputContract"]
                if variant_contract is not None
                else profile["outputContract"]
            )
            if args.onnx_path:
                actual_input, actual_shape, actual_opset, output_names = inspect_onnx(onnx_path)
                expected_shape = [1, int(input_profile["channels"]), height, width]
                if actual_input != input_profile["name"]:
                    raise RuntimeError(
                        f"ONNX input '{actual_input}' does not match profile input "
                        f"'{input_profile['name']}'"
                    )
                if actual_shape != expected_shape:
                    raise RuntimeError(
                        f"ONNX input shape {actual_shape} does not match requested {expected_shape}"
                    )
                if actual_opset != opset:
                    raise RuntimeError(
                        f"ONNX opset {actual_opset} does not match profile opset {opset}"
                    )
                digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
            else:
                output_names = ["output0"]
                digest = create_onnx(
                    onnx_path,
                    height,
                    width,
                    input_name=str(input_profile["name"]),
                    output_name=output_names[0],
                    opset=opset,
                )
            metadata = {
                "name": f"E2E {args.profile_id} fixture",
                "description": "Generated conversion validation fixture",
                "version": "v1",
                "taskType": profile["taskType"],
                "classes": ["fixture"],
            }
            with dataset_path.open("rb") as dataset_file:
                dataset = require(
                    client.post(
                        f"{base_url}/datasets",
                        headers=admin_headers,
                        data={"metadata": json.dumps(metadata)},
                        files={"file": (dataset_path.name, dataset_file, "application/gzip")},
                    )
                )
            training = require(
                client.post(
                    f"{base_url}/training-jobs",
                    headers=admin_headers,
                    json={
                        "name": "E2E static ONNX",
                        "datasetId": dataset["id"],
                        "profileId": args.profile_id,
                        "variant": variant,
                        "resolution": {"width": width, "height": height},
                        "hyperparameters": {"epochs": 1, "batchSize": 1},
                        "accelerator": "cpu",
                    },
                )
            )
            trainer = require(
                client.post(
                    f"{base_url}/workers/register",
                    headers=worker_headers,
                    json={
                        "name": "e2e-fixture-trainer",
                        "kind": "trainer",
                        "capabilities": [args.profile_id],
                        "accelerator": "cpu",
                        "maxConcurrency": 1,
                        "version": "e2e",
                    },
                )
            )
            claim = require(
                client.post(
                    f"{base_url}/worker/jobs/claim",
                    headers=worker_headers,
                    json={"workerId": trainer["id"], "jobId": training["id"]},
                )
            )
            if claim["job"]["id"] != training["id"]:
                raise RuntimeError("Fixture trainer claimed an unexpected job")

            manifest = {
                "schemaVersion": 1,
                "modelFamily": profile["family"],
                "profileId": args.profile_id,
                "variant": variant,
                "taskType": profile["taskType"],
                "trainingJobId": training["id"],
                "onnxSha256": digest,
                "opset": opset,
                "resolution": {"width": width, "height": height},
                "input": {
                    "name": input_profile["name"],
                    "layout": input_profile["layout"],
                    "shape": [1, 3, height, width],
                    "dtype": input_profile["dtype"],
                    "colorSpace": input_profile["colorSpace"],
                },
                "preprocessing": profile["preprocessing"],
                "resizePolicy": input_profile["resizePolicy"],
                "outputContract": output_contract,
                "outputs": [
                    {"name": output_name, "semantic": output_contract}
                    for output_name in output_names
                ],
                "labels": ["fixture"],
                "supportedPrecisions": profile["precisions"],
                "rknn": profile["rknn"],
            }
            with onnx_path.open("rb") as model_file:
                artifact = require(
                    client.post(
                        f"{base_url}/worker/jobs/{training['id']}/artifacts",
                        headers=worker_headers,
                        data={
                            "lease_token": claim["leaseToken"],
                            "kind": "onnx",
                            "manifest": json.dumps(manifest),
                        },
                        files={"file": (onnx_path.name, model_file, "application/onnx")},
                    )
                )
            require(
                client.post(
                    f"{base_url}/worker/jobs/{training['id']}/complete",
                    headers=worker_headers,
                    json={"leaseToken": claim["leaseToken"], "result": {"e2e": True}},
                )
            )
            conversion_payload: dict[str, Any] = {
                "name": f"E2E RK3588 {args.precision.upper()}",
                "sourceArtifactId": artifact["id"],
                "precision": args.precision,
            }
            if args.precision == "int8":
                conversion_payload["calibrationDatasetId"] = dataset["id"]
            conversion = require(
                client.post(
                    f"{base_url}/conversion-jobs",
                    headers=admin_headers,
                    json=conversion_payload,
                )
            )

            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                job = require(
                    client.get(f"{base_url}/jobs/{conversion['id']}", headers=admin_headers)
                )
                if job["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(2)
            else:
                raise TimeoutError(f"Conversion {conversion['id']} did not finish")
            if job["status"] != "succeeded":
                raise RuntimeError(
                    f"Conversion failed: {job.get('errorCode')} {job.get('errorMessage')}"
                )
            artifacts = client.get(
                f"{base_url}/artifacts",
                headers=admin_headers,
                params={"jobId": conversion["id"], "kind": "rknn"},
            )
            artifacts.raise_for_status()
            raw_values: object = artifacts.json()
            if not isinstance(raw_values, list):
                raise RuntimeError("Conversion artifact response is not a list")
            values = cast(list[object], raw_values)
            if len(values) != 1:
                raise RuntimeError("Conversion did not publish exactly one RKNN artifact")
            rknn = values[0]
            if not isinstance(rknn, dict):
                raise RuntimeError("Conversion artifact response is not an object")
            download = client.get(
                f"{base_url}/artifacts/{rknn['id']}/download", headers=admin_headers
            )
            download.raise_for_status()
            if not download.content:
                raise RuntimeError("Downloaded RKNN artifact is empty")
            print(
                json.dumps(
                    {
                        "conversionJobId": conversion["id"],
                        "rknnArtifactId": rknn["id"],
                        "rknnBytes": len(download.content),
                        "deploymentReady": job.get("result", {}).get("deploymentReady"),
                    }
                )
            )


if __name__ == "__main__":
    main()
