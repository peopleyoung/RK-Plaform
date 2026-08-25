from __future__ import annotations

import ipaddress
import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import ConfigDict, Field, SecretStr, field_validator, model_validator

from .api_models import ApiModel, to_camel

MEDIA_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
HOSTNAME_PATTERN = re.compile(
    r"^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?$"
)


class MediaGatewayStatus(StrEnum):
    DISABLED = "disabled"
    PROBING = "probing"
    ONLINE = "online"
    ERROR = "error"


class MediaCredentialRole(StrEnum):
    PUBLISH = "publish"
    PLAY = "play"


class PreviewCapabilityState(StrEnum):
    AVAILABLE = "available"
    UNSUPPORTED = "unsupported"
    MIGRATION_REQUIRED = "migration_required"
    GATEWAY_OFFLINE = "gateway_offline"


def validate_media_host(value: str) -> str:
    candidate = value.strip()
    if not candidate or any(character.isspace() for character in candidate):
        raise ValueError("media host is required")
    if any(marker in candidate for marker in ("://", "/", "?", "#", "@")):
        raise ValueError("media host must not contain a scheme, path, or credentials")
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        if HOSTNAME_PATTERN.fullmatch(candidate) is None or ".." in candidate:
            raise ValueError("media host must be an IP address or DNS hostname") from None
    return candidate


class MediaGatewayCreate(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    enabled: bool = True
    publish_host: str
    rtsp_port: int = Field(ge=1, le=65535)
    playback_host: str
    ws_port: int = Field(ge=1, le=65535)
    api_host: str
    api_port: int = Field(ge=1, le=65535)
    app: str = Field(default="live", pattern=MEDIA_IDENTIFIER_PATTERN.pattern)
    api_secret: SecretStr | None = None
    hook_identity: SecretStr | None = None

    @field_validator("publish_host", "playback_host", "api_host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        return validate_media_host(value)


class MediaGatewayUpdate(MediaGatewayCreate):
    pass


class MediaGatewayResponse(ApiModel):
    id: str
    name: str
    builtin: bool
    enabled: bool
    publish_host: str
    rtsp_port: int
    playback_host: str
    ws_port: int
    api_host: str
    api_port: int
    app: str
    status: MediaGatewayStatus
    api_secret_configured: bool
    hook_identity_configured: bool
    last_probe_at: datetime | None = None
    last_hook_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PreviewCapability(ApiModel):
    state: PreviewCapabilityState
    reason: str | None = None


class MediaTrackingConfig(ApiModel):
    enabled: bool = False
    track_buffer: int = Field(default=30, ge=1, le=10000)


class MediaKafkaConfig(ApiModel):
    enabled: bool = False
    brokers: str = ""
    topic: str = "sei_msg"
    key: str = ""
    queue_messages: int = Field(default=10000, ge=1, le=1000000)
    message_timeout_ms: int = Field(default=3000, ge=100, le=60000)

    @model_validator(mode="after")
    def validate_enabled_destination(self) -> MediaKafkaConfig:
        if self.enabled and (not self.brokers.strip() or not self.topic.strip()):
            raise ValueError("enabled Kafka requires brokers and topic")
        return self


class ZlmSeiTaskConfig(ApiModel):
    enabled: bool = False
    gateway_id: str | None = Field(default=None, max_length=48)
    stream_name: str | None = Field(
        default=None,
        pattern=MEDIA_IDENTIFIER_PATTERN.pattern,
    )
    reconnect_ms: int = Field(default=1000, ge=1000, le=4000)

    @model_validator(mode="after")
    def validate_binding(self) -> ZlmSeiTaskConfig:
        if self.enabled and (not self.gateway_id or not self.stream_name):
            raise ValueError("enabled ZLM SEI requires gatewayId and streamName")
        return self


class TaskMediaConfig(ApiModel):
    decoder: Literal["opencv", "rkmpp"] = "opencv"
    tracking: MediaTrackingConfig = Field(default_factory=MediaTrackingConfig)
    kafka: MediaKafkaConfig = Field(default_factory=MediaKafkaConfig)
    zlm_sei: ZlmSeiTaskConfig = Field(default_factory=ZlmSeiTaskConfig)


class ZlmHookRequest(ApiModel):
    # ZLM adds event-specific telemetry fields; authorization reads only this
    # deliberately bounded subset.
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )
    media_server_id: str = Field(min_length=1, max_length=256)
    app: str = ""
    stream: str = ""
    media_schema: str = Field(default="", alias="schema")
    params: str = ""


class ZlmHookResponse(ApiModel):
    code: int
    msg: str


def media_gateway_response(
    record: Any,
    *,
    api_secret_configured: bool,
    hook_identity_configured: bool,
) -> MediaGatewayResponse:
    return MediaGatewayResponse(
        id=record.id,
        name=record.name,
        builtin=record.builtin,
        enabled=record.enabled,
        publish_host=record.publish_host,
        rtsp_port=record.rtsp_port,
        playback_host=record.playback_host,
        ws_port=record.ws_port,
        api_host=record.api_host,
        api_port=record.api_port,
        app=record.app,
        status=MediaGatewayStatus(record.status),
        api_secret_configured=api_secret_configured,
        hook_identity_configured=hook_identity_configured,
        last_probe_at=record.last_probe_at,
        last_hook_at=record.last_hook_at,
        last_error=record.last_error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
