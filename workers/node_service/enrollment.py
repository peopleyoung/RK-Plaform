from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import NodeServiceSettings

MINIMUM_TOKEN_LENGTH = 16


def resolve_node_token(settings: NodeServiceSettings) -> str:
    """Resolve file, legacy environment, or enrollment claim in that order."""
    if settings.node_token_file.exists():
        return _read_token(settings.node_token_file, "stored node Token")

    if settings.token is not None:
        return _validate_token(settings.token, "legacy node Token")

    if not settings.endpoint_id or not settings.platform_url:
        raise ValueError(
            "No node credential source is configured; provide a persistent Token, "
            "RKNODE_NODE_TOKEN, or endpoint enrollment settings"
        )
    if not settings.enrollment_token_file.exists():
        raise ValueError(
            "No node credential source is available; the enrollment credential file "
            f"does not exist: {settings.enrollment_token_file}"
        )

    enrollment_token = _read_token(
        settings.enrollment_token_file, "node enrollment credential"
    )
    token = claim_node_token(settings, enrollment_token)
    persist_node_token(settings.node_token_file, token)
    return token


def claim_node_token(settings: NodeServiceSettings, enrollment_token: str) -> str:
    """POST immutable node identity to the platform claim endpoint."""
    if not settings.endpoint_id or not settings.platform_url:
        raise ValueError("Node enrollment requires RKNODE_ENDPOINT_ID and RKNODE_PLATFORM_URL")
    enrollment_token = _validate_token(enrollment_token, "node enrollment credential")
    claim_url = (
        f"{settings.platform_url.rstrip('/')}/api/v1/"
        f"node-enrollments/{quote(settings.endpoint_id, safe='')}/claim"
    )
    body = json.dumps(
        {
            "enrollmentToken": enrollment_token,
            "name": settings.name,
            "kind": settings.kind,
            "accelerator": settings.accelerator,
            "capabilities": list(settings.capabilities),
            "version": settings.version,
            "maxConcurrency": settings.max_concurrency,
            "features": list(settings.features),
            "diagnostics": {},
        }
    ).encode("utf-8")
    request = Request(
        claim_url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.request_timeout_seconds) as response:
            content = response.read()
    except HTTPError as error:
        raise _claim_http_error(error, enrollment_token) from error
    except (OSError, URLError) as error:
        raise ValueError("node enrollment request failed; retry is safe") from error

    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("node enrollment response contained malformed JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("node enrollment response must be a JSON object")
    response_data = cast(dict[str, Any], payload)
    node_token = response_data.get("nodeToken")
    if not isinstance(node_token, str):
        raise ValueError("node enrollment response did not contain nodeToken")
    return _validate_token(node_token, "claimed node Token")


def persist_node_token(path: Path, token: str) -> None:
    """Write a newline-terminated Token atomically with mode 0600."""
    token = _validate_token(token, "node Token")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            file_descriptor = -1
            handle.write(f"{token}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)


def _read_token(path: Path, label: str) -> str:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"Unable to read {label} from {path}") from error
    return _validate_token(token, label)


def _validate_token(token: str, label: str) -> str:
    value = token.strip()
    if len(value) < MINIMUM_TOKEN_LENGTH:
        raise ValueError(f"{label} must contain at least {MINIMUM_TOKEN_LENGTH} characters")
    return value


def _claim_http_error(error: HTTPError, enrollment_token: str) -> ValueError:
    code = f"http_{error.code}"
    message = "the platform rejected node enrollment"
    try:
        payload = json.loads(error.read())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        payload = None
    if isinstance(payload, dict):
        payload_data = cast(dict[str, object], payload)
        detail = payload_data.get("detail")
        if isinstance(detail, dict):
            detail_data = cast(dict[str, object], detail)
            raw_code = detail_data.get("code")
            raw_message = detail_data.get("message")
            if isinstance(raw_code, str) and raw_code:
                code = raw_code
            if isinstance(raw_message, str) and raw_message:
                message = raw_message.replace(enrollment_token, "[redacted]")
    return ValueError(f"node enrollment failed ({code}): {message}")
