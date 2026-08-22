from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from .auth import get_context, require_admin
from .context import AppContext
from .contracts import InferencePlaybackSessionResponse
from .media_contracts import (
    MediaCredentialRole,
    MediaGatewayCreate,
    MediaGatewayResponse,
    MediaGatewayUpdate,
    ZlmHookRequest,
    ZlmHookResponse,
)
from .media_service import MediaService

router = APIRouter()
Admin = Annotated[None, Depends(require_admin)]
Context = Annotated[AppContext, Depends(get_context)]


@router.get("/media-gateways", response_model=list[MediaGatewayResponse])
def list_media_gateways(_: Admin, context: Context) -> list[MediaGatewayResponse]:
    return MediaService(context).list_gateways()


@router.post(
    "/media-gateways",
    response_model=MediaGatewayResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_media_gateway(
    payload: MediaGatewayCreate, _: Admin, context: Context
) -> MediaGatewayResponse:
    return MediaService(context).create_gateway(payload)


@router.put("/media-gateways/{gateway_id}", response_model=MediaGatewayResponse)
def update_media_gateway(
    gateway_id: str, payload: MediaGatewayUpdate, _: Admin, context: Context
) -> MediaGatewayResponse:
    return MediaService(context).update_gateway(gateway_id, payload)


@router.post("/media-gateways/{gateway_id}/probe", response_model=MediaGatewayResponse)
def probe_media_gateway(
    gateway_id: str, _: Admin, context: Context
) -> MediaGatewayResponse:
    return MediaService(context).probe_gateway(gateway_id)


@router.delete("/media-gateways/{gateway_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_media_gateway(gateway_id: str, _: Admin, context: Context) -> Response:
    MediaService(context).delete_gateway(gateway_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/inference-tasks/{task_id}/playback-session",
    response_model=InferencePlaybackSessionResponse,
)
def create_inference_playback_session(
    task_id: str, _: Admin, context: Context
) -> InferencePlaybackSessionResponse:
    return MediaService(context).create_playback_session(task_id, principal="admin")


def _hook_result(allowed: bool) -> ZlmHookResponse:
    return ZlmHookResponse(code=0, msg="success") if allowed else ZlmHookResponse(
        code=-1, msg="denied"
    )


@router.post(
    "/media-hooks/zlm/{gateway_id}/on-publish", response_model=ZlmHookResponse
)
def on_zlm_publish(
    gateway_id: str, payload: ZlmHookRequest, context: Context
) -> ZlmHookResponse:
    return _hook_result(
        MediaService(context).authorize_hook(
            gateway_id, payload, MediaCredentialRole.PUBLISH
        )
    )


@router.post("/media-hooks/zlm/{gateway_id}/on-play", response_model=ZlmHookResponse)
def on_zlm_play(
    gateway_id: str, payload: ZlmHookRequest, context: Context
) -> ZlmHookResponse:
    return _hook_result(
        MediaService(context).authorize_hook(gateway_id, payload, MediaCredentialRole.PLAY)
    )


@router.post(
    "/media-hooks/zlm/{gateway_id}/on-server-keepalive",
    response_model=ZlmHookResponse,
)
def on_zlm_server_keepalive(
    gateway_id: str, payload: ZlmHookRequest, context: Context
) -> ZlmHookResponse:
    return _hook_result(MediaService(context).receive_keepalive(gateway_id, payload))
