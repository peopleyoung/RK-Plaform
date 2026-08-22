from __future__ import annotations

import hmac
import os
from pathlib import Path


class NodeSecretStore:
    """Write-only-at-the-API-boundary storage for per-node bearer tokens."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    def write(self, endpoint_id: str, token: str, *, purpose: str = "node") -> None:
        path = self._path(endpoint_id, purpose)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(token, encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)

    def read(self, endpoint_id: str, *, purpose: str = "node") -> str | None:
        try:
            return self._path(endpoint_id, purpose).read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None

    def delete(self, endpoint_id: str, *, purpose: str = "node") -> None:
        self._path(endpoint_id, purpose).unlink(missing_ok=True)

    def matching_endpoint_id(self, candidate: str) -> str | None:
        matched_endpoint_id: str | None = None
        suffix = ".node.token"
        for path in self.root.glob("*.node.token"):
            try:
                expected = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if hmac.compare_digest(candidate, expected):
                matched_endpoint_id = path.name[: -len(suffix)]
        return matched_endpoint_id

    def _path(self, endpoint_id: str, purpose: str) -> Path:
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
        if not endpoint_id or any(
            character not in allowed for character in endpoint_id.lower()
        ):
            raise ValueError("invalid endpoint ID")
        if purpose not in {"node", "agent"}:
            raise ValueError("invalid node secret purpose")
        return self.root / f"{endpoint_id}.{purpose}.token"
