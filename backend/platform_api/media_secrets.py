from __future__ import annotations

import os
from pathlib import Path


class MediaSecretStore:
    """Restricted storage for gateway secrets and publication credentials."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    def write_api_secret(self, gateway_id: str, secret: str) -> None:
        self._write(self._gateway_path(gateway_id, "api"), secret)

    def write_hook_identity(self, gateway_id: str, identity: str) -> None:
        self._write(self._gateway_path(gateway_id, "hook"), identity)

    def api_secret(self, gateway_id: str) -> str | None:
        return self._read(self._gateway_path(gateway_id, "api"))

    def hook_identity(self, gateway_id: str) -> str | None:
        return self._read(self._gateway_path(gateway_id, "hook"))

    def write_publication_token(self, credential_id: str, token: str) -> None:
        self._write(self._credential_path(credential_id), token)

    def publication_token(self, credential_id: str) -> str | None:
        return self._read(self._credential_path(credential_id))

    def delete_publication_token(self, credential_id: str) -> None:
        self._credential_path(credential_id).unlink(missing_ok=True)

    def configured(self, gateway_id: str) -> tuple[bool, bool]:
        return (
            self.api_secret(gateway_id) is not None,
            self.hook_identity(gateway_id) is not None,
        )

    def delete(self, gateway_id: str) -> None:
        self._gateway_path(gateway_id, "api").unlink(missing_ok=True)
        self._gateway_path(gateway_id, "hook").unlink(missing_ok=True)

    def _write(self, path: Path, value: str) -> None:
        if not value:
            raise ValueError("media secret cannot be empty")
        temporary = path.with_suffix(path.suffix + ".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                target.write(value)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None

    def _gateway_path(self, gateway_id: str, purpose: str) -> Path:
        self._validate_id(gateway_id)
        if purpose not in {"api", "hook"}:
            raise ValueError("invalid gateway secret purpose")
        return self.root / f"{gateway_id}.{purpose}.secret"

    def _credential_path(self, credential_id: str) -> Path:
        self._validate_id(credential_id)
        return self.root / f"{credential_id}.publish.token"

    @staticmethod
    def _validate_id(value: str) -> None:
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
        if not value or any(character not in allowed for character in value.lower()):
            raise ValueError("invalid media secret ID")
