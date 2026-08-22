from __future__ import annotations

from typing import Any


class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class NotFoundError(AppError):
    def __init__(self, resource: str, resource_id: str) -> None:
        super().__init__(
            "not_found",
            f"{resource} '{resource_id}' was not found",
            status_code=404,
            details={"resource": resource, "id": resource_id},
        )


class ConflictError(AppError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(code, message, status_code=409, details=details)


class AuthenticationError(AppError):
    def __init__(self, message: str = "Invalid or missing bearer token") -> None:
        super().__init__("unauthorized", message, status_code=401)
