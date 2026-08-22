from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, cast


class NodeClientError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class DirectNodeClient:
    def __init__(self, endpoint: str, token: str, *, timeout: float = 5.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        value = self._json("GET", "/health")
        if value.get("protocolVersion") != "1.0":
            raise NodeClientError(200, "node protocol version is not supported")
        return value

    def dispatch(self, job_id: str) -> dict[str, Any]:
        return self._json("POST", f"/api/v1/jobs/{job_id}/dispatch")

    def clean_job_cache(self, job_id: str) -> None:
        self._request("DELETE", f"/api/v1/jobs/{job_id}/cache")

    def apply_inference_revision(
        self,
        revision: int,
        *,
        node_id: str,
        central_api_url: str,
        access_token: str,
        desired: dict[str, Any],
    ) -> dict[str, Any]:
        return self._json(
            "PUT",
            f"/api/v1/inference/revisions/{revision}",
            {
                "nodeId": node_id,
                "centralApiUrl": central_api_url.rstrip("/"),
                "accessToken": access_token,
                "desired": desired,
            },
        )

    def inference_status(self) -> dict[str, Any]:
        return self._json("GET", "/api/v1/inference/status")

    def _json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        value = self._request(method, path, payload)
        if not isinstance(value, dict):
            raise NodeClientError(200, "node returned a non-object JSON response")
        return cast(dict[str, Any], value)

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> object:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + path,
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
                return json.loads(raw)
        except urllib.error.HTTPError as error:
            # Never include Authorization or request bodies in persisted errors.
            message = f"node returned HTTP {error.code}"
            raise NodeClientError(error.code, message) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise NodeClientError(0, "node connection failed") from error
