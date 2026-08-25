from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import cast


class ZlmClientError(RuntimeError):
    pass


class ZlmClient:
    def __init__(self, host: str, port: int, secret: str, *, timeout: float = 5.0) -> None:
        self.origin = f"http://{self._format_host(host)}:{port}"
        self.secret = secret
        self.timeout = timeout

    def get_server_config(self) -> dict[str, object]:
        payload = self._request("/index/api/getServerConfig")
        data = payload.get("data", payload)
        if isinstance(data, list):
            items = cast(list[object], data)
            if len(items) == 1 and isinstance(items[0], dict):
                return cast(dict[str, object], items[0])
        if isinstance(data, dict):
            return cast(dict[str, object], data)
        raise ZlmClientError("ZLMediaKit returned an invalid server configuration")

    def get_media_info(self, app: str, stream: str) -> dict[str, object] | None:
        payload = self._request(
            "/index/api/getMediaList",
            {"vhost": "__defaultVhost__", "app": app, "stream": stream},
        )
        data = payload.get("data", [])
        if not isinstance(data, list):
            return None
        items = cast(list[object], data)
        if not items:
            return None
        item = items[0]
        return cast(dict[str, object], item) if isinstance(item, dict) else None

    def close_streams(self, app: str, stream: str) -> None:
        self._request(
            "/index/api/close_streams",
            {
                "vhost": "__defaultVhost__",
                "app": app,
                "stream": stream,
                "force": "1",
            },
        )

    def _request(
        self, path: str, parameters: dict[str, str] | None = None
    ) -> dict[str, object]:
        query = {"secret": self.secret, **(parameters or {})}
        url = f"{self.origin}{path}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_value: object = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise ZlmClientError(f"ZLMediaKit returned HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            raise ZlmClientError("ZLMediaKit control request failed") from error
        if not isinstance(raw_value, dict):
            raise ZlmClientError("ZLMediaKit rejected the control request")
        value = cast(dict[str, object], raw_value)
        if value.get("code") not in {0, None}:
            raise ZlmClientError("ZLMediaKit rejected the control request")
        return value

    @staticmethod
    def _format_host(host: str) -> str:
        return f"[{host}]" if ":" in host and not host.startswith("[") else host
