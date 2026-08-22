from __future__ import annotations

from pathlib import Path

import pytest
from backend.platform_api.contracts import DeploymentManifest, Precision, Resolution
from backend.platform_api.errors import AppError
from backend.platform_api.profiles import ModelProfileRegistry


def registry() -> ModelProfileRegistry:
    return ModelProfileRegistry(Path(__file__).parents[1] / "config/model_profiles.json")


def test_profile_resolution_rules_are_enforced() -> None:
    profiles = registry()
    profiles.validate_resolution("yolo-detect", Resolution(width=640, height=384))

    with pytest.raises(AppError) as error:
        profiles.validate_resolution("yolo-detect", Resolution(width=650, height=640))

    assert error.value.code == "invalid_resolution"
    assert error.value.details["fields"]["width"] == "must be a multiple of 32"


def test_ppocr_recognition_is_fp16_only() -> None:
    profiles = registry()
    profile = profiles.get("ppocr-rec")

    assert profile.preprocessing.mean == [127.5, 127.5, 127.5]
    assert profile.preprocessing.std == [127.5, 127.5, 127.5]

    with pytest.raises(AppError) as error:
        profiles.validate_precision("ppocr-rec", Precision.INT8, "ds_any")

    assert error.value.code == "unsupported_precision"


def test_ppocr_detection_uses_board_qualified_quantization() -> None:
    profile = registry().get("ppocr-det")

    assert profile.rknn.quantized_algorithm == "normal"


def test_yolo_variant_contracts_keep_generation_specific_opsets() -> None:
    profiles = registry()

    assert profiles.variant_contract("yolo-detect", "yolov8n").opset == 12
    assert profiles.variant_contract("yolo-detect", "yolov10n").opset == 13
    assert (
        profiles.variant_contract("yolo-detect", "yolov5s").output_contract
        == "rknn_yolov5_anchored_heads_v1"
    )


def test_manifest_shape_must_match_resolution() -> None:
    profiles = registry()
    payload = {
        "schemaVersion": 1,
        "modelFamily": "YOLO",
        "profileId": "yolo-detect",
        "variant": "yolov8n",
        "taskType": "object_detection",
        "trainingJobId": "train_1",
        "onnxSha256": "a" * 64,
        "opset": 12,
        "resolution": {"width": 640, "height": 384},
        "input": {
            "name": "images",
            "layout": "NCHW",
            "shape": [1, 3, 640, 640],
            "dtype": "float32",
            "colorSpace": "RGB",
        },
        "preprocessing": {"mean": [0, 0, 0], "std": [255, 255, 255]},
        "resizePolicy": "letterbox",
        "outputContract": "rknn_yolo_dfl_split_heads_v1",
        "outputs": [{"name": "output0", "semantic": "detections"}],
        "labels": ["scratch"],
        "supportedPrecisions": ["int8", "fp16"],
        "rknn": {
            "targetPlatform": "rk3588",
            "quantizedAlgorithm": "normal",
            "optimizationLevel": 3,
            "requiresCalibrationFor": ["int8"],
        },
    }

    with pytest.raises(AppError) as error:
        profiles.validate_manifest(DeploymentManifest.model_validate(payload))

    assert error.value.code == "manifest_shape_mismatch"
