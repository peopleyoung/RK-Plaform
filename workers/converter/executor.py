from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from workers.common.artifacts import model_artifact_stem
from workers.common.client import PlatformClient
from workers.converter.calibration import create_calibration_list
from workers.converter.engine import RknnConverter


class ConversionExecutor:
    def __init__(self, converter: RknnConverter | None = None) -> None:
        self.converter = converter or RknnConverter()

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
            raise ValueError("Conversion job has no spec")
        spec = cast(dict[str, Any], raw_spec)
        manifest = self._mapping(spec, "manifest")
        source = self._mapping(spec, "sourceArtifact")
        source_id = self._string(source, "id")
        expected_checksum = self._string(source, "sha256")
        precision = self._string(spec, "precision")

        source_path = workspace / "source" / self._string(source, "filename")
        client.progress(job_id, lease_token, 3, "download_onnx", "Downloading ONNX artifact")
        actual_checksum = client.download_artifact(source_id, source_path)
        if actual_checksum != expected_checksum:
            raise ValueError(
                f"ONNX checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
            )
        client.progress(job_id, lease_token, 8, "verify_onnx", "Verified ONNX artifact")

        calibration_list = None
        if precision == "int8":
            calibration = self._mapping(spec, "calibrationDataset")
            archive = workspace / "source" / self._string(calibration, "filename")
            client.progress(
                job_id,
                lease_token,
                10,
                "download_calibration",
                "Downloading calibration dataset",
            )
            calibration_checksum = client.download_dataset(self._string(calibration, "id"), archive)
            expected_calibration_checksum = self._string(calibration, "sha256")
            if calibration_checksum != expected_calibration_checksum:
                raise ValueError(
                    "Calibration dataset checksum mismatch: "
                    f"expected {expected_calibration_checksum}, got {calibration_checksum}"
                )
            client.progress(
                job_id,
                lease_token,
                15,
                "prepare_calibration",
                "Preparing calibration images",
            )
            calibration_list = create_calibration_list(archive, workspace)

        resolution = self._mapping(manifest, "resolution")
        artifact_stem = model_artifact_stem(
            self._string(manifest, "variant"),
            self._integer(resolution, "width"),
            self._integer(resolution, "height"),
        )
        rknn_path = workspace / "output" / f"{artifact_stem}.rknn"

        def report_progress(value: int, stage: str, message: str) -> None:
            client.progress(job_id, lease_token, value, stage, message)

        result = self.converter.convert(
            onnx_path=source_path,
            manifest=manifest,
            precision=precision,
            output_path=rknn_path,
            calibration_list=calibration_list,
            progress=report_progress,
        )
        report_path = workspace / "output" / "validation-report.json"
        report_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log_path = workspace / "output" / "conversion.log"
        log_path.write_text(result.log, encoding="utf-8")

        client.progress(job_id, lease_token, 92, "upload_rknn", "Uploading RKNN artifact")
        rknn_artifact = client.upload_artifact(job_id, lease_token, "rknn", rknn_path)
        client.progress(job_id, lease_token, 95, "upload_report", "Uploading validation report")
        report_artifact = client.upload_artifact(
            job_id, lease_token, "validation_report", report_path
        )
        client.progress(job_id, lease_token, 98, "upload_log", "Uploading conversion log")
        log_artifact = client.upload_artifact(job_id, lease_token, "conversion_log", log_path)
        return {
            "rknnArtifactId": rknn_artifact.get("id"),
            "validationReportArtifactId": report_artifact.get("id"),
            "conversionLogArtifactId": log_artifact.get("id"),
            "deploymentReady": result.report["deploymentReady"],
            "performanceReady": result.report["performanceReady"],
            "validation": result.report,
        }

    @staticmethod
    def _claim_parts(claim: dict[str, Any]) -> tuple[dict[str, Any], str]:
        raw_job = claim.get("job")
        lease_token = claim.get("leaseToken")
        if not isinstance(raw_job, dict) or not isinstance(lease_token, str):
            raise ValueError("Invalid conversion claim")
        return cast(dict[str, Any], raw_job), lease_token

    @staticmethod
    def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
        item = value.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"Conversion spec has no {key}")
        return cast(dict[str, Any], item)

    @staticmethod
    def _string(value: dict[str, Any], key: str) -> str:
        item = value.get(key)
        if not isinstance(item, str) or not item:
            raise ValueError(f"Expected non-empty string for {key}")
        return item

    @staticmethod
    def _integer(value: dict[str, Any], key: str) -> int:
        item = value.get(key)
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"Expected positive integer for {key}")
        return item
