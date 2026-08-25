from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response

from .context import AppContext
from .contracts import ApiErrorBody, ApiErrorResponse
from .database import Database
from .direct_dispatcher import DirectNodeDispatcher
from .errors import AppError
from .inference_routes import router as inference_router
from .media_routes import router as media_router
from .media_secrets import MediaSecretStore
from .media_service import MediaService
from .node_secrets import NodeSecretStore
from .profiles import ModelProfileRegistry
from .routes import router
from .settings import Settings
from .state_machine import JobStateMachine
from .storage import FileStorage

logger = logging.getLogger("rknode.api")


def _public_validation_errors(error: RequestValidationError) -> list[dict[str, object]]:
    return [
        {key: value for key, value in item.items() if key != "input"}
        for item in error.errors()
    ]


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    resolved_settings.data_dir.mkdir(parents=True, exist_ok=True)
    database = Database(resolved_settings.resolved_database_url)
    storage = FileStorage(
        resolved_settings.data_dir, upload_limit_bytes=resolved_settings.upload_limit_bytes
    )
    profiles = ModelProfileRegistry(resolved_settings.model_profiles_path)
    context = AppContext(
        settings=resolved_settings,
        database=database,
        storage=storage,
        profiles=profiles,
        jobs=JobStateMachine(lease_seconds=resolved_settings.worker_lease_seconds),
        node_secrets=NodeSecretStore(resolved_settings.node_secret_dir),
        media_secrets=MediaSecretStore(resolved_settings.media_secret_dir),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        database.create_schema()
        MediaService(context).bootstrap_builtin_gateway()
        stop_dispatcher = asyncio.Event()
        dispatcher_task: asyncio.Task[None] | None = None
        if resolved_settings.direct_dispatch_enabled:
            dispatcher_task = asyncio.create_task(
                DirectNodeDispatcher(context).run(stop_dispatcher),
                name="direct-node-dispatcher",
            )
        try:
            yield
        finally:
            stop_dispatcher.set()
            if dispatcher_task is not None:
                await dispatcher_task

    app = FastAPI(title=resolved_settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.context = context
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.middleware("http")(request_id_middleware)

    async def app_error_handler(request: Request, exception: Exception) -> JSONResponse:
        if not isinstance(exception, AppError):
            raise exception
        error = exception
        body = ApiErrorResponse(
            error=ApiErrorBody(
                code=error.code,
                message=error.message,
                request_id=getattr(request.state, "request_id", "unknown"),
                details=error.details,
            )
        )
        return JSONResponse(status_code=error.status_code, content=body.model_dump(by_alias=True))

    async def validation_error_handler(request: Request, exception: Exception) -> JSONResponse:
        if not isinstance(exception, RequestValidationError):
            raise exception
        error = exception
        body = ApiErrorResponse(
            error=ApiErrorBody(
                code="validation_error",
                message="Request validation failed",
                request_id=getattr(request.state, "request_id", "unknown"),
                details={"errors": _public_validation_errors(error)},
            )
        )
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(body.model_dump(by_alias=True)),
        )

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(router, prefix=resolved_settings.api_prefix)
    app.include_router(inference_router, prefix=resolved_settings.api_prefix)
    app.include_router(media_router, prefix=resolved_settings.api_prefix)
    return app


app = create_app()
