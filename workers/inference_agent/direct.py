from __future__ import annotations

import threading
from typing import Any, cast

from .agent import AgentSettings, InferenceAgent
from .client import InferenceAgentClient


class DirectInferenceController:
    """Owns the direct-mode agent configured by the first pushed revision."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agent: InferenceAgent | None = None
        self._identity: tuple[str, str] | None = None
        self._applying_revision: int | None = None
        self._last_error: str | None = None

    def preflight(self) -> bool:
        settings = AgentSettings.from_env(
            api_url_override="http://127.0.0.1/unused",
            node_id_override="direct-preflight",
        )
        agent = InferenceAgent(
            settings,
            InferenceAgentClient("http://127.0.0.1/unused", access_token="unused"),
        )
        passed = agent.run_self_test()
        with self._lock:
            self._last_error = None if passed else "runtime self-test failed"
        return passed

    def apply(self, revision: int, payload: dict[str, Any]) -> dict[str, Any]:
        node_id = str(payload.get("nodeId", "")).strip()
        central_api_url = str(payload.get("centralApiUrl", "")).strip().rstrip("/")
        access_token = str(payload.get("accessToken", "")).strip()
        desired_value = payload.get("desired")
        if not node_id or not central_api_url or not access_token:
            raise ValueError("nodeId, centralApiUrl, and accessToken are required")
        if not isinstance(desired_value, dict):
            raise ValueError("desired must be an object")
        desired = cast(dict[str, Any], desired_value)
        if int(desired.get("revision", -1)) != revision:
            raise ValueError("desired revision does not match the request path")

        with self._lock:
            identity = (node_id, central_api_url)
            if self._agent is None or self._identity != identity:
                settings = AgentSettings.from_env(
                    api_url_override=central_api_url,
                    node_id_override=node_id,
                )
                client = InferenceAgentClient(central_api_url, access_token=access_token)
                self._agent = InferenceAgent(settings, client)
                self._identity = identity
            else:
                self._agent.client.access_token = access_token
            if self._agent.last_revision == revision:
                return {"accepted": True, **self._status_unlocked()}
            if self._applying_revision is not None:
                return {
                    "accepted": self._applying_revision == revision,
                    **self._status_unlocked(),
                }
            self._applying_revision = revision
            self._last_error = None
            agent = self._agent
            threading.Thread(
                target=self._apply_in_background,
                args=(agent, revision, desired),
                name=f"apply-revision-{revision}",
                daemon=True,
            ).start()
            return {"accepted": True, **self._status_unlocked()}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def _apply_in_background(
        self, agent: InferenceAgent, revision: int, desired: dict[str, Any]
    ) -> None:
        error_message: str | None = None
        try:
            if not agent.apply_desired(desired):
                error_message = f"revision {revision} failed to apply"
        except Exception as error:
            error_message = str(error)
        finally:
            with self._lock:
                self._last_error = error_message
                self._applying_revision = None

    def _status_unlocked(self) -> dict[str, Any]:
        if self._agent is None:
            return {
                "configured": False,
                "actualRevision": 0,
                "applyingRevision": self._applying_revision,
                "lastError": self._last_error,
            }
        return {
            **self._agent.status(),
            "applyingRevision": self._applying_revision,
            "lastError": self._last_error,
        }
