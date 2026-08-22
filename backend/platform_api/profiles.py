from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .contracts import (
    DeploymentManifest,
    ModelProfile,
    ModelProfileDocument,
    Precision,
    Resolution,
    VariantContract,
)
from .errors import AppError, NotFoundError


class ModelProfileRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._document = self._load(path)
        self._profiles = {profile.id: profile for profile in self._document.profiles}
        if len(self._profiles) != len(self._document.profiles):
            raise RuntimeError("model profile IDs must be unique")

    @staticmethod
    def _load(path: Path) -> ModelProfileDocument:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ModelProfileDocument.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise RuntimeError(f"invalid model profile document at {path}: {error}") from error

    @property
    def document(self) -> ModelProfileDocument:
        return self._document

    def get(self, profile_id: str) -> ModelProfile:
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise NotFoundError("model profile", profile_id)
        return profile

    def variant_contract(self, profile_id: str, variant: str) -> VariantContract:
        profile = self.get(profile_id)
        if variant not in profile.variants:
            raise AppError("unsupported_variant", f"Unsupported variant '{variant}'")
        return profile.variant_contracts.get(
            variant,
            VariantContract(
                exporter=profile.framework,
                opset=profile.export.opset,
                output_contract=profile.output_contract,
            ),
        )

    def validate_resolution(self, profile_id: str, resolution: Resolution) -> None:
        profile = self.get(profile_id)
        rule = profile.resolution_rule
        errors: dict[str, str] = {}
        if not rule.min_width <= resolution.width <= rule.max_width:
            errors["width"] = f"must be between {rule.min_width} and {rule.max_width}"
        elif resolution.width % rule.width_multiple:
            errors["width"] = f"must be a multiple of {rule.width_multiple}"
        if not rule.min_height <= resolution.height <= rule.max_height:
            errors["height"] = f"must be between {rule.min_height} and {rule.max_height}"
        elif resolution.height % rule.height_multiple:
            errors["height"] = f"must be a multiple of {rule.height_multiple}"
        if errors:
            raise AppError(
                "invalid_resolution",
                f"Resolution is not supported by profile '{profile_id}'",
                details={"fields": errors, "profileId": profile_id},
            )

    def validate_precision(
        self,
        profile_id: str,
        precision: Precision,
        calibration_dataset_id: str | None,
    ) -> None:
        profile = self.get(profile_id)
        if precision not in profile.precisions:
            raise AppError(
                "unsupported_precision",
                f"Profile '{profile_id}' does not support {precision.value}",
                details={"supported": [item.value for item in profile.precisions]},
            )
        if precision in profile.rknn.requires_calibration_for and not calibration_dataset_id:
            raise AppError(
                "calibration_dataset_required",
                f"{precision.value} conversion requires a calibration dataset",
            )

    def validate_manifest(self, manifest: DeploymentManifest) -> ModelProfile:
        profile = self.get(manifest.profile_id)
        self.validate_resolution(manifest.profile_id, manifest.resolution)
        if manifest.model_family != profile.family:
            raise AppError(
                "manifest_profile_mismatch", "Manifest model family does not match profile"
            )
        if manifest.task_type != profile.task_type:
            raise AppError("manifest_profile_mismatch", "Manifest task type does not match profile")
        variant_contract = self.variant_contract(manifest.profile_id, manifest.variant)
        if manifest.opset != variant_contract.opset:
            raise AppError(
                "unsupported_opset",
                f"Expected ONNX opset {variant_contract.opset}, got {manifest.opset}",
            )
        expected_shape = [
            1,
            profile.input.channels,
            manifest.resolution.height,
            manifest.resolution.width,
        ]
        if profile.input.layout == "NHWC":
            expected_shape = [
                1,
                manifest.resolution.height,
                manifest.resolution.width,
                profile.input.channels,
            ]
        if manifest.input.shape != expected_shape:
            raise AppError(
                "manifest_shape_mismatch",
                "Manifest input shape does not match requested resolution",
                details={"expected": expected_shape, "actual": manifest.input.shape},
            )
        if (
            manifest.input.layout != profile.input.layout
            or manifest.input.name != profile.input.name
        ):
            raise AppError(
                "manifest_input_mismatch", "Manifest input contract does not match profile"
            )
        if manifest.output_contract != variant_contract.output_contract:
            raise AppError(
                "manifest_output_mismatch", "Manifest output contract does not match profile"
            )
        if set(manifest.supported_precisions) - set(profile.precisions):
            raise AppError(
                "manifest_precision_mismatch", "Manifest declares unsupported precisions"
            )
        return profile
