from __future__ import annotations

import hashlib
import http.client
import json
import mimetypes
import secrets
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, cast


class WorkerApiError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"API request failed ({status}/{code}): {message}")
        self.status = status
        self.code = code


class PlatformClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._require_object(self._json("POST", "/workers/register", payload))

    def heartbeat(self, worker_id: str, active_jobs: int = 0) -> dict[str, Any]:
        return self._require_object(
            self._json(
                "POST",
                f"/workers/{urllib.parse.quote(worker_id)}/heartbeat",
                {"activeJobs": active_jobs},
            )
        )

    def claim(self, worker_id: str, job_id: str | None = None) -> dict[str, Any] | None:
        payload = {"workerId": worker_id}
        if job_id is not None:
            payload["jobId"] = job_id
        return self._json("POST", "/worker/jobs/claim", payload)

    def retained_job_ids(self) -> set[str]:
        payload = self._require_object(self._json("GET", "/worker/jobs/retained"))
        raw_job_ids: object = payload.get("jobIds")
        if not isinstance(raw_job_ids, list):
            raise WorkerApiError(200, "invalid_response", "Expected a jobIds string array")
        job_ids: set[str] = set()
        for raw_job_id in cast(list[object], raw_job_ids):
            if not isinstance(raw_job_id, str):
                raise WorkerApiError(200, "invalid_response", "Expected a jobIds string array")
            job_ids.add(raw_job_id)
        return job_ids

    def progress(
        self,
        job_id: str,
        lease_token: str,
        progress: int,
        stage: str,
        message: str = "",
        metrics: dict[str, float | int | str] | None = None,
    ) -> dict[str, Any]:
        return self._require_object(
            self._json(
                "POST",
                f"/worker/jobs/{urllib.parse.quote(job_id)}/progress",
                {
                    "leaseToken": lease_token,
                    "progress": progress,
                    "stage": stage,
                    "message": message,
                    "metrics": metrics or {},
                },
            )
        )

    def renew(self, job_id: str, lease_token: str) -> dict[str, Any]:
        return self._require_object(
            self._json(
                "POST",
                f"/worker/jobs/{urllib.parse.quote(job_id)}/renew",
                {"leaseToken": lease_token},
            )
        )

    def telemetry(
        self,
        job_id: str,
        lease_token: str,
        entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._require_object(
            self._json(
                "POST",
                f"/worker/jobs/{urllib.parse.quote(job_id)}/events",
                {"leaseToken": lease_token, "entries": entries},
            )
        )

    def complete(self, job_id: str, lease_token: str, result: dict[str, Any]) -> dict[str, Any]:
        return self._require_object(
            self._json(
                "POST",
                f"/worker/jobs/{urllib.parse.quote(job_id)}/complete",
                {"leaseToken": lease_token, "result": result},
            )
        )

    def fail(
        self,
        job_id: str,
        lease_token: str,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> dict[str, Any]:
        return self._require_object(
            self._json(
                "POST",
                f"/worker/jobs/{urllib.parse.quote(job_id)}/fail",
                {
                    "leaseToken": lease_token,
                    "code": code,
                    "message": message,
                    "retryable": retryable,
                },
            )
        )

    def download_dataset(self, dataset_id: str, target: Path) -> str:
        return self._download(f"/worker/datasets/{urllib.parse.quote(dataset_id)}/download", target)

    def update_dataset_classes(self, dataset_id: str, classes: list[str]) -> dict[str, Any]:
        return self._require_object(
            self._json(
                "PUT",
                f"/worker/datasets/{urllib.parse.quote(dataset_id)}/classes",
                {"classes": classes},
            )
        )

    def download_artifact(self, artifact_id: str, target: Path) -> str:
        return self._download(f"/artifacts/{urllib.parse.quote(artifact_id)}/download", target)

    def upload_artifact(
        self,
        job_id: str,
        lease_token: str,
        kind: str,
        path: Path,
        *,
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fields = {"lease_token": lease_token, "kind": kind}
        if manifest is not None:
            fields["manifest"] = json.dumps(manifest, separators=(",", ":"))
        return self._multipart(f"/worker/jobs/{urllib.parse.quote(job_id)}/artifacts", fields, path)

    def _json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                if not raw:
                    return None
                value: object = json.loads(raw)
                if value is None:
                    return None
                return self._require_object(value)
        except urllib.error.HTTPError as error:
            self._raise_http_error(error.code, error.read())
        except urllib.error.URLError as error:
            raise WorkerApiError(0, "connection_error", str(error.reason)) from error
        return None

    def _download(self, path: str, target: Path) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        request = urllib.request.Request(
            self.base_url + path,
            headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response, temporary.open(
                "wb"
            ) as output:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            temporary.replace(target)
        except urllib.error.HTTPError as error:
            temporary.unlink(missing_ok=True)
            self._raise_http_error(error.code, error.read())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return digest.hexdigest()

    def _multipart(self, path: str, fields: dict[str, str], file_path: Path) -> dict[str, Any]:
        boundary = f"rknode-{secrets.token_hex(16)}"
        field_parts: list[bytes] = []
        for name, value in fields.items():
            field_parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode()
            )
        filename = file_path.name.replace('"', "_")
        file_header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mimetypes.guess_type(filename)[0] or 'application/octet-stream'}\r\n"
            "\r\n"
        ).encode()
        closing = f"\r\n--{boundary}--\r\n".encode("ascii")
        content_length = (
            sum(map(len, field_parts)) + len(file_header) + file_path.stat().st_size + len(closing)
        )

        parsed = urllib.parse.urlsplit(self.base_url + path)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError(f"Invalid API URL: {self.base_url}")
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(parsed.hostname, parsed.port, timeout=self.timeout)
        request_path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
        try:
            connection.putrequest("POST", request_path)
            connection.putheader("Authorization", f"Bearer {self.token}")
            connection.putheader("Accept", "application/json")
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(content_length))
            connection.endheaders()
            for part in field_parts:
                connection.send(part)
            connection.send(file_header)
            with file_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    connection.send(chunk)
            connection.send(closing)
            response = connection.getresponse()
            raw = response.read()
            if not 200 <= response.status < 300:
                self._raise_http_error(response.status, raw)
            value: object = json.loads(raw)
            if not isinstance(value, dict):
                raise WorkerApiError(response.status, "invalid_response", "Expected JSON object")
            return cast(dict[str, Any], value)
        finally:
            connection.close()

    @staticmethod
    def _raise_http_error(status: int, body: bytes) -> None:
        try:
            payload = json.loads(body)
            error = payload.get("error", {})
            code = str(error.get("code", "http_error"))
            message = str(error.get("message", body.decode("utf-8", errors="replace")))
        except (json.JSONDecodeError, AttributeError):
            code = "http_error"
            message = body.decode("utf-8", errors="replace")
        raise WorkerApiError(status, code, message)

    @staticmethod
    def _require_object(value: object) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise WorkerApiError(200, "invalid_response", "Expected JSON object")
        return cast(dict[str, Any], value)
