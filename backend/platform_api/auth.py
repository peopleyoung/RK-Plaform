from __future__ import annotations

import hmac
from typing import Annotated, Literal

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from .context import AppContext
from .contracts import ServiceEndpointEnrollmentStatus
from .db_models import ServiceEndpointRecord
from .errors import AuthenticationError

bearer = HTTPBearer(auto_error=False)


def get_context(request: Request) -> AppContext:
    return request.app.state.context


def _require_token(
    role: Literal["admin", "worker", "either"],
    credentials: HTTPAuthorizationCredentials | None,
    context: AppContext,
) -> str | None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError()
    token = credentials.credentials
    endpoint_id: str | None = None
    admin_token = context.settings.admin_token.get_secret_value()
    worker_token = context.settings.worker_token.get_secret_value()
    allowed = {
        "admin": (admin_token,),
        "worker": (worker_token,),
        "either": (admin_token, worker_token),
    }[role]
    token_matches = any(hmac.compare_digest(token, expected) for expected in allowed)
    if role in {"worker", "either"} and not token_matches:
        endpoint_id = context.node_secrets.matching_endpoint_id(token)
        if endpoint_id is not None:
            with context.database.session() as session:
                endpoint = session.scalar(
                    select(ServiceEndpointRecord).where(
                        ServiceEndpointRecord.id == endpoint_id,
                        ServiceEndpointRecord.mode == "direct",
                        ServiceEndpointRecord.kind.in_({"trainer", "converter"}),
                        ServiceEndpointRecord.enabled.is_(True),
                        ServiceEndpointRecord.enrollment_status
                        == ServiceEndpointEnrollmentStatus.ENROLLED.value,
                    )
                )
                token_matches = endpoint is not None
    if not token_matches:
        raise AuthenticationError()
    return endpoint_id


def require_admin(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    context: Annotated[AppContext, Depends(get_context)],
) -> None:
    _require_token("admin", credentials, context)


def require_worker(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    context: Annotated[AppContext, Depends(get_context)],
) -> str | None:
    return _require_token("worker", credentials, context)


def require_admin_or_worker(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    context: Annotated[AppContext, Depends(get_context)],
) -> str | None:
    return _require_token("either", credentials, context)
