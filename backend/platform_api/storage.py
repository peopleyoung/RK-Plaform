from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from .errors import AppError, NotFoundError

SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class StoredFile:
    storage_key: str
    filename: str
    size_bytes: int
    sha256: str


class FileStorage:
    def __init__(self, root: Path, *, upload_limit_bytes: int) -> None:
        self.root = root.resolve()
        self.upload_limit_bytes = upload_limit_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".tmp").mkdir(exist_ok=True)

    @staticmethod
    def safe_filename(filename: str | None) -> str:
        value = Path(filename or "upload.bin").name
        value = SAFE_FILENAME.sub("_", value).strip("._")
        return value[:180] or "upload.bin"

    async def write_upload(self, upload: UploadFile, namespace: str) -> StoredFile:
        filename = self.safe_filename(upload.filename)
        identifier = secrets.token_hex(16)
        storage_key = f"{namespace}/{identifier}/{filename}"
        final_path = self.resolve(storage_key)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.root / ".tmp" / f"{identifier}.part"
        digest = hashlib.sha256()
        size = 0
        try:
            with temp_path.open("wb") as target:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.upload_limit_bytes:
                        raise AppError(
                            "upload_too_large",
                            f"Upload exceeds {self.upload_limit_bytes} bytes",
                            status_code=413,
                        )
                    digest.update(chunk)
                    target.write(chunk)
            temp_path.replace(final_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return StoredFile(storage_key, filename, size, digest.hexdigest())

    def resolve(self, storage_key: str) -> Path:
        if not storage_key or storage_key.startswith(("/", "\\")):
            raise AppError("invalid_storage_key", "Invalid storage key")
        candidate = (self.root / storage_key).resolve()
        if not candidate.is_relative_to(self.root):
            raise AppError("invalid_storage_key", "Storage key escapes the data directory")
        return candidate

    def require(self, storage_key: str) -> Path:
        path = self.resolve(storage_key)
        if not path.is_file():
            raise NotFoundError("stored file", storage_key)
        return path

    def remove(self, storage_key: str) -> None:
        self.resolve(storage_key).unlink(missing_ok=True)
