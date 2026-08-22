from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

from backend.platform_api.contracts import TrainingJobCreate
from backend.platform_api.profiles import ModelProfileRegistry

from workers.common.artifacts import model_artifact_stem
from workers.common.client import PlatformClient
from workers.trainer.adapters import (
    AdapterEnvironment,
    AdapterRegistry,
    TrainingTask,
)
from workers.trainer.archive import extract_dataset
from workers.trainer.dataset import prepare_training_dataset
from workers.trainer.manifest import build_deployment_manifest
from workers.trainer.runner import CommandRunner
from workers.trainer.telemetry import TelemetryEntry


class TrainingExecutor:
    def __init__(
        self,
        profiles: ModelProfileRegistry,
        environment: AdapterEnvironment,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self.profiles = profiles
        self.adapters = AdapterRegistry(profiles, environment)
        self.runner = runner or CommandRunner()

    def execute(
        self,
        claim: dict[str, Any],
        client: PlatformClient,
        workspace: Path,
    ) -> dict[str, Any]:
        job, lease_token = self._claim_parts(claim)
        job_id = self._string(job, "id")
        raw_spec = job.get("spec")
        if not isinstance(raw_spec, dict):
            raise ValueError("Training job has no spec")
        spec = cast(dict[str, Any], raw_spec)
        request_keys = {
            "name",
            "datasetId",
            "profileId",
            "variant",
            "resolution",
            "hyperparameters",
            "accelerator",
        }
        request = TrainingJobCreate.model_validate(
            {key: value for key, value in spec.items() if key in request_keys}
        )
        raw_dataset = spec.get("dataset")
        if not isinstance(raw_dataset, dict):
            raise ValueError("Training job has no dataset snapshot")
        dataset = cast(dict[str, Any], raw_dataset)
        dataset_id = self._string(dataset, "id")
        expected_checksum = self._string(dataset, "sha256")
        labels = self._string_list(dataset.get("classes"), "dataset classes")
        dataset_format = dataset.get("datasetFormat", "auto")
        if not isinstance(dataset_format, str):
            raise ValueError("Dataset format must be a string")

        archive = workspace / "dataset" / self._string(dataset, "filename")
        client.progress(job_id, lease_token, 2, "download", "Downloading dataset")
        actual_checksum = client.download_dataset(dataset_id, archive)
        if actual_checksum != expected_checksum:
            raise ValueError(
                f"Dataset checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
            )
        dataset_dir = workspace / "dataset" / "extracted"
        extract_dataset(archive, dataset_dir)
        output_dir = workspace / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        client.progress(job_id, lease_token, 7, "validate_dataset", "Validating dataset layout")
        prepared = prepare_training_dataset(
            request.profile_id,
            dataset_dir,
            tuple(labels),
            dataset_format,
            workspace / "dataset" / "normalized",
        )
        labels = list(prepared.labels)
        if labels:
            client.update_dataset_classes(dataset_id, labels)
        task = TrainingTask(
            job_id=job_id,
            request=request,
            dataset_dir=prepared.root,
            output_dir=output_dir,
            labels=prepared.labels,
        )
        plan = self.adapters.get(request.profile_id).plan(task)
        log_path = workspace / "training.log"

        def report_progress(value: int, stage: str, message: str) -> None:
            client.progress(job_id, lease_token, value, stage, message)

        def report_telemetry(entries: tuple[TelemetryEntry, ...]) -> None:
            client.telemetry(
                job_id,
                lease_token,
                [entry.as_payload() for entry in entries],
            )

        self.runner.run(
            plan.steps,
            log_path,
            report_progress,
            report_telemetry,
        )
        if not plan.onnx_source.is_file():
            raise FileNotFoundError(f"Training adapter did not produce {plan.onnx_source}")
        artifact_stem = model_artifact_stem(
            request.variant,
            request.resolution.width,
            request.resolution.height,
        )
        model_path = output_dir / f"{artifact_stem}.onnx"
        if plan.onnx_source.resolve() != model_path.resolve():
            shutil.copy2(plan.onnx_source, model_path)

        checkpoint_path: Path | None = None
        if plan.checkpoint_source and plan.checkpoint_source.is_file():
            checkpoint_suffix = plan.checkpoint_source.suffix or ".bin"
            checkpoint_path = output_dir / f"{artifact_stem}{checkpoint_suffix}"
            if plan.checkpoint_source.resolve() != checkpoint_path.resolve():
                shutil.copy2(plan.checkpoint_source, checkpoint_path)

        client.progress(job_id, lease_token, 85, "validate", "Validating static ONNX contract")
        manifest = build_deployment_manifest(
            self.profiles,
            job_id=job_id,
            profile_id=request.profile_id,
            variant=request.variant,
            resolution=request.resolution,
            onnx_path=model_path,
            labels=labels,
        )
        manifest_payload = manifest.model_dump(mode="json", by_alias=True)
        manifest_path = output_dir / "deployment-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        uploaded: list[dict[str, Any]] = []
        uploaded.append(
            client.upload_artifact(
                job_id,
                lease_token,
                "onnx",
                model_path,
                manifest=manifest_payload,
            )
        )
        uploaded.append(client.upload_artifact(job_id, lease_token, "manifest", manifest_path))
        if log_path.is_file():
            uploaded.append(client.upload_artifact(job_id, lease_token, "training_log", log_path))
        if checkpoint_path is not None:
            uploaded.append(
                client.upload_artifact(job_id, lease_token, "training_checkpoint", checkpoint_path)
            )
        for path in plan.auxiliary_artifacts:
            if path.is_file():
                uploaded.append(client.upload_artifact(job_id, lease_token, "auxiliary", path))
        client.progress(job_id, lease_token, 98, "upload", "Uploaded training artifacts")
        return {
            "onnxArtifactId": uploaded[0].get("id"),
            "manifest": manifest_payload,
            "artifactIds": [item.get("id") for item in uploaded],
        }

    @staticmethod
    def _claim_parts(claim: dict[str, Any]) -> tuple[dict[str, Any], str]:
        raw_job = claim.get("job")
        lease_token = claim.get("leaseToken")
        if not isinstance(raw_job, dict) or not isinstance(lease_token, str):
            raise ValueError("Invalid training claim")
        return cast(dict[str, Any], raw_job), lease_token

    @staticmethod
    def _string(value: dict[str, Any], key: str) -> str:
        item = value.get(key)
        if not isinstance(item, str) or not item:
            raise ValueError(f"Expected non-empty string for {key}")
        return item

    @staticmethod
    def _string_list(value: object, label: str) -> list[str]:
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in cast(list[object], value)
        ):
            raise ValueError(f"Expected string list for {label}")
        return cast(list[str], value)
