from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, cast


class AgentApiError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"API request failed ({status}/{code}): {message}")
        self.status = status
        self.code = code


class InferenceAgentClient:
    def __init__(self, base_url: str, *, access_token: str = "", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.timeout = timeout

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._object("POST", "/inference-agent/register", payload, authenticated=False)

    def heartbeat(self, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._object(
            "POST",
            f"/inference-agent/nodes/{urllib.parse.quote(node_id)}/heartbeat",
            payload,
        )

    def desired(self, node_id: str) -> dict[str, Any]:
        return self._object("GET", f"/inference-agent/nodes/{urllib.parse.quote(node_id)}/desired")

    def report_target(
        self, node_id: str, target_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._object(
            "POST",
            f"/inference-agent/nodes/{urllib.parse.quote(node_id)}/targets/"
            f"{urllib.parse.quote(target_id)}/status",
            payload,
        )

    def download_artifact(self, node_id: str, artifact_id: str, target: Path) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        offset = temporary.stat().st_size if temporary.is_file() else 0
        if offset:
            with temporary.open("rb") as existing:
                while chunk := existing.read(1024 * 1024):
                    digest.update(chunk)
        headers = self._headers()
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(
            self.base_url + f"/inference-agent/nodes/{urllib.parse.quote(node_id)}/artifacts/"
            f"{urllib.parse.quote(artifact_id)}/download",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_status = getattr(response, "status", response.getcode())
                if offset and response_status != 206:
                    offset = 0
                    digest = hashlib.sha256()
                with temporary.open("ab" if offset else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
            temporary.replace(target)
        except urllib.error.HTTPError as error:
            if error.code == 416 and temporary.is_file():
                temporary.replace(target)
                return digest.hexdigest()
            self._raise_http(error.code, error.read())
        except urllib.error.URLError as error:
            raise AgentApiError(0, "connection_error", str(error.reason)) from error
        return digest.hexdigest()

    def _object(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if authenticated:
            headers.update(self._headers())
        request = urllib.request.Request(
            self.base_url + path, data=body, method=method, headers=headers
        )
        value: object = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value: object = json.loads(response.read())
        except urllib.error.HTTPError as error:
            self._raise_http(error.code, error.read())
        except urllib.error.URLError as error:
            raise AgentApiError(0, "connection_error", str(error.reason)) from error
        if not isinstance(value, dict):
            raise AgentApiError(200, "invalid_response", "Expected a JSON object")
        return cast(dict[str, Any], value)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    @staticmethod
    def _raise_http(status: int, body: bytes) -> None:
        payload: dict[str, Any] = {}
        try:
            parsed: object = json.loads(body)
            if isinstance(parsed, dict):
                payload = cast(dict[str, Any], parsed)
        except (TypeError, ValueError):
            pass
        error_value = payload.get("error", {})
        error = cast(dict[str, Any], error_value) if isinstance(error_value, dict) else {}
        code = error.get("code", "http_error")
        message = error.get("message", "request failed")
        raise AgentApiError(status, str(code), str(message))
