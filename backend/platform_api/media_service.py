from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Literal, cast
from urllib.parse import parse_qs, quote

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .context import AppContext
from .contracts import InferencePlaybackSessionResponse
from .db_models import (
    InferenceMediaBindingRecord,
    InferenceTaskRecord,
    MediaCredentialRecord,
    MediaGatewayRecord,
    utc_now,
)
from .errors import ConflictError, NotFoundError
from .media_contracts import (
    MediaCredentialRole,
    MediaGatewayCreate,
    MediaGatewayResponse,
    MediaGatewayStatus,
    MediaGatewayUpdate,
    PreviewCapability,
    PreviewCapabilityState,
    ZlmHookRequest,
    media_gateway_response,
)
from .zlm_client import ZlmClient, ZlmClientError


def _hash_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class MediaService:
    def __init__(self, context: AppContext) -> None:
        self.context = context

    def bootstrap_builtin_gateway(self) -> None:
        settings = self.context.settings
        with self.context.database.session() as session:
            record = session.get(MediaGatewayRecord, "gateway_builtin")
            if not settings.media_builtin_enabled:
                if record is not None:
                    record.enabled = False
                    record.status = MediaGatewayStatus.DISABLED.value
                    record.last_error = None
                return

            publish_host = (settings.media_publish_host or "").strip()
            playback_host = (settings.media_playback_host or "").strip()
            api_host = settings.media_api_host.strip()
            api_secret = settings.zlm_api_secret
            hook_identity = settings.zlm_hook_identity
            if not publish_host or not playback_host or not api_host:
                raise ValueError(
                    "built-in media gateway requires explicit publish, playback, and API hosts"
                )
            if api_secret is None or hook_identity is None:
                raise ValueError("built-in media gateway requires API and Hook secrets")

            if record is None:
                record = MediaGatewayRecord(
                    id="gateway_builtin",
                    name="Built-in ZLMediaKit",
                    builtin=True,
                    enabled=True,
                    publish_host=publish_host,
                    rtsp_port=settings.media_rtsp_port,
                    playback_host=playback_host,
                    ws_port=settings.media_ws_port,
                    api_host=api_host,
                    api_port=settings.media_api_port,
                    app="live",
                    status=MediaGatewayStatus.ERROR.value,
                    last_error="gateway is waiting for authenticated keepalive",
                )
                session.add(record)
            else:
                record.builtin = True
                record.enabled = True
                record.publish_host = publish_host
                record.rtsp_port = settings.media_rtsp_port
                record.playback_host = playback_host
                record.ws_port = settings.media_ws_port
                record.api_host = api_host
                record.api_port = settings.media_api_port
                record.status = MediaGatewayStatus.ERROR.value
                record.last_error = "gateway is waiting for authenticated keepalive"
                record.last_probe_at = None
            self.context.media_secrets.write_api_secret(
                record.id, api_secret.get_secret_value()
            )
            self.context.media_secrets.write_hook_identity(
                record.id, hook_identity.get_secret_value()
            )

    def create_gateway(
        self, payload: MediaGatewayCreate, *, builtin: bool = False
    ) -> MediaGatewayResponse:
        gateway_id = "gateway_" + uuid.uuid4().hex
        record = MediaGatewayRecord(
            id=gateway_id,
            name=payload.name,
            builtin=builtin,
            enabled=payload.enabled,
            publish_host=payload.publish_host,
            rtsp_port=payload.rtsp_port,
            playback_host=payload.playback_host,
            ws_port=payload.ws_port,
            api_host=payload.api_host,
            api_port=payload.api_port,
            app=payload.app,
            status=(
                MediaGatewayStatus.ERROR.value
                if payload.enabled
                else MediaGatewayStatus.DISABLED.value
            ),
            last_error="gateway has not been probed" if payload.enabled else None,
        )
        try:
            with self.context.database.session() as session:
                if session.scalar(
                    select(MediaGatewayRecord).where(MediaGatewayRecord.name == payload.name)
                ):
                    raise ConflictError(
                        "media_gateway_name_exists",
                        "A media gateway with this name already exists",
                    )
                session.add(record)
                session.flush()
                self._write_payload_secrets(gateway_id, payload)
        except Exception:
            self.context.media_secrets.delete(gateway_id)
            raise
        return self._response(record)

    def list_gateways(self) -> list[MediaGatewayResponse]:
        with self.context.database.session() as session:
            records = session.scalars(
                select(MediaGatewayRecord).order_by(MediaGatewayRecord.name, MediaGatewayRecord.id)
            ).all()
            return [self._response(record) for record in records]

    def update_gateway(
        self, gateway_id: str, payload: MediaGatewayUpdate
    ) -> MediaGatewayResponse:
        with self.context.database.session() as session:
            record = self._gateway(session, gateway_id)
            duplicate = session.scalar(
                select(MediaGatewayRecord).where(
                    MediaGatewayRecord.name == payload.name,
                    MediaGatewayRecord.id != gateway_id,
                )
            )
            if duplicate is not None:
                raise ConflictError(
                    "media_gateway_name_exists",
                    "A media gateway with this name already exists",
                )
            record.name = payload.name
            record.enabled = payload.enabled
            record.publish_host = payload.publish_host
            record.rtsp_port = payload.rtsp_port
            record.playback_host = payload.playback_host
            record.ws_port = payload.ws_port
            record.api_host = payload.api_host
            record.api_port = payload.api_port
            record.app = payload.app
            record.status = (
                MediaGatewayStatus.ERROR.value
                if payload.enabled
                else MediaGatewayStatus.DISABLED.value
            )
            record.last_error = "gateway requires a new probe" if payload.enabled else None
            record.last_probe_at = None
            self._write_payload_secrets(gateway_id, payload)
            session.flush()
            return self._response(record)

    def delete_gateway(self, gateway_id: str) -> None:
        with self.context.database.session() as session:
            record = self._gateway(session, gateway_id)
            if record.builtin:
                raise ConflictError(
                    "builtin_media_gateway_delete_forbidden",
                    "The built-in media gateway cannot be deleted",
                )
            binding = session.scalar(
                select(InferenceMediaBindingRecord).where(
                    InferenceMediaBindingRecord.gateway_id == gateway_id
                )
            )
            if binding is not None:
                raise ConflictError(
                    "media_gateway_in_use",
                    "The media gateway is referenced by an inference task",
                )
            session.execute(
                delete(MediaCredentialRecord).where(
                    MediaCredentialRecord.gateway_id == gateway_id
                )
            )
            session.delete(record)
        self.context.media_secrets.delete(gateway_id)

    def probe_gateway(self, gateway_id: str) -> MediaGatewayResponse:
        with self.context.database.session() as session:
            record = self._gateway(session, gateway_id)
            if not record.enabled:
                record.status = MediaGatewayStatus.DISABLED.value
                record.last_error = None
                return self._response(record)
            record.status = MediaGatewayStatus.PROBING.value
            api_secret = self.context.media_secrets.api_secret(gateway_id)
            if api_secret is None:
                record.status = MediaGatewayStatus.ERROR.value
                record.last_error = "ZLMediaKit API secret is not configured"
                return self._response(record)
            try:
                config = ZlmClient(
                    record.api_host, record.api_port, api_secret
                ).get_server_config()
                self._validate_hook_config(record, config)
                if (
                    record.last_hook_at is None
                    or (utc_now() - _as_utc(record.last_hook_at)).total_seconds() > 30
                ):
                    raise ZlmClientError("authenticated ZLMediaKit keepalive is stale")
            except ZlmClientError as error:
                record.status = MediaGatewayStatus.ERROR.value
                record.last_error = str(error)[:500]
            else:
                record.status = MediaGatewayStatus.ONLINE.value
                record.last_error = None
            record.last_probe_at = utc_now()
            session.flush()
            return self._response(record)

    def receive_keepalive(self, gateway_id: str, payload: ZlmHookRequest) -> bool:
        with self.context.database.session() as session:
            record = session.get(MediaGatewayRecord, gateway_id)
            if record is None or not record.enabled or not self._hook_identity_matches(
                gateway_id, payload.media_server_id
            ):
                return False
            record.last_hook_at = utc_now()
            return True

    def issue_publish_credential(
        self,
        *,
        gateway_id: str,
        task_id: str,
        revision: int,
        app: str,
        stream_name: str,
        principal: str,
    ) -> str:
        token, record = self._new_credential(
            role=MediaCredentialRole.PUBLISH,
            gateway_id=gateway_id,
            task_id=task_id,
            revision=revision,
            app=app,
            stream_name=stream_name,
            principal=principal,
            expires_at=None,
        )
        self.context.media_secrets.write_publication_token(record.id, token)
        return token

    def issue_play_credential(
        self,
        *,
        gateway_id: str,
        task_id: str,
        revision: int,
        app: str,
        stream_name: str,
        principal: str,
    ) -> tuple[str, datetime]:
        expires_at = utc_now() + timedelta(seconds=60)
        token, _ = self._new_credential(
            role=MediaCredentialRole.PLAY,
            gateway_id=gateway_id,
            task_id=task_id,
            revision=revision,
            app=app,
            stream_name=stream_name,
            principal=principal,
            expires_at=expires_at,
        )
        return token, expires_at

    def validate_task_media(
        self,
        session: Session,
        media: dict[str, object],
        *,
        task_id: str | None = None,
    ) -> tuple[MediaGatewayRecord, str] | None:
        zlm = self._zlm_config(media)
        if zlm.get("enabled") is not True:
            return None
        gateway_id = str(zlm.get("gatewayId", ""))
        stream_name = str(zlm.get("streamName", ""))
        gateway = self._gateway(session, gateway_id)
        if not gateway.enabled or gateway.status != MediaGatewayStatus.ONLINE.value:
            raise ConflictError(
                "media_gateway_offline",
                "The selected media gateway is not online",
            )
        active_statuses = {"deploying", "running", "degraded"}
        conflict = session.scalar(
            select(InferenceMediaBindingRecord)
            .join(
                InferenceTaskRecord,
                InferenceTaskRecord.id == InferenceMediaBindingRecord.task_id,
            )
            .where(
                InferenceMediaBindingRecord.gateway_id == gateway.id,
                InferenceMediaBindingRecord.app == gateway.app,
                InferenceMediaBindingRecord.stream_name == stream_name,
                InferenceTaskRecord.status.in_(active_statuses),
                *(
                    (InferenceMediaBindingRecord.task_id != task_id,)
                    if task_id is not None
                    else ()
                ),
            )
        )
        if conflict is not None:
            raise ConflictError(
                "media_stream_in_use",
                "The selected gateway stream is already used by an active task",
                gatewayId=gateway.id,
                streamName=stream_name,
            )
        return gateway, stream_name

    def bind_task(
        self, session: Session, task: InferenceTaskRecord
    ) -> InferenceMediaBindingRecord | None:
        validated = self.validate_task_media(
            session, task.media_json, task_id=task.id
        )
        binding = session.get(InferenceMediaBindingRecord, task.id)
        if validated is None:
            if binding is not None:
                session.delete(binding)
            return None
        gateway, stream_name = validated
        if binding is None:
            binding = InferenceMediaBindingRecord(
                task_id=task.id,
                gateway_id=gateway.id,
                app=gateway.app,
                stream_name=stream_name,
            )
            session.add(binding)
        else:
            binding.gateway_id = gateway.id
            binding.app = gateway.app
            binding.stream_name = stream_name
        return binding

    def preview_capability(
        self, session: Session, task: InferenceTaskRecord
    ) -> PreviewCapability:
        if task.media_migration_required:
            return PreviewCapability(
                state=PreviewCapabilityState.MIGRATION_REQUIRED,
                reason="media_migration_required",
            )
        zlm = self._zlm_config(task.media_json)
        if zlm.get("enabled") is not True:
            return PreviewCapability(
                state=PreviewCapabilityState.UNSUPPORTED,
                reason="zlm_sei_disabled",
            )
        if not task.input_uri.startswith("rtsp://"):
            return PreviewCapability(
                state=PreviewCapabilityState.UNSUPPORTED,
                reason="input_not_rtsp",
            )
        if task.media_json.get("decoder") != "rkmpp":
            return PreviewCapability(
                state=PreviewCapabilityState.UNSUPPORTED,
                reason="decoder_not_rkmpp",
            )
        gateway_id = str(zlm.get("gatewayId", ""))
        gateway = session.get(MediaGatewayRecord, gateway_id)
        if (
            gateway is None
            or not gateway.enabled
            or gateway.status != MediaGatewayStatus.ONLINE.value
        ):
            return PreviewCapability(
                state=PreviewCapabilityState.GATEWAY_OFFLINE,
                reason="media_gateway_offline",
            )
        return PreviewCapability(state=PreviewCapabilityState.AVAILABLE, reason=None)

    def ensure_publish_credential(
        self, session: Session, task: InferenceTaskRecord
    ) -> MediaCredentialRecord | None:
        binding = session.get(InferenceMediaBindingRecord, task.id)
        if binding is None:
            return None
        existing = session.scalar(
            select(MediaCredentialRecord).where(
                MediaCredentialRecord.task_id == task.id,
                MediaCredentialRecord.role == MediaCredentialRole.PUBLISH.value,
                MediaCredentialRecord.revision == task.config_revision,
                MediaCredentialRecord.revoked_at.is_(None),
            )
        )
        if existing is not None and self.context.media_secrets.publication_token(
            existing.id
        ):
            return existing
        self.revoke_task_publication(session, task.id)
        token = secrets.token_urlsafe(32)
        credential = MediaCredentialRecord(
            id="media_credential_" + uuid.uuid4().hex,
            token_hash=_hash_token(token),
            role=MediaCredentialRole.PUBLISH.value,
            gateway_id=binding.gateway_id,
            task_id=task.id,
            revision=task.config_revision,
            app=binding.app,
            stream_name=binding.stream_name,
            principal=task.node_id,
        )
        session.add(credential)
        session.flush()
        self.context.media_secrets.write_publication_token(credential.id, token)
        return credential

    def node_media(
        self, session: Session, task: InferenceTaskRecord
    ) -> dict[str, object]:
        media = deepcopy(task.media_json)
        zlm = self._zlm_config(media)
        if zlm.get("enabled") is not True:
            return media
        binding = session.get(InferenceMediaBindingRecord, task.id)
        credential = self.ensure_publish_credential(session, task)
        if binding is None or credential is None:
            raise ConflictError(
                "media_binding_missing",
                "The task media binding is missing",
                taskId=task.id,
            )
        token = self.context.media_secrets.publication_token(credential.id)
        gateway = self._gateway(session, binding.gateway_id)
        if token is None:
            raise ConflictError(
                "media_publication_credential_missing",
                "The task publication credential is missing",
                taskId=task.id,
            )
        media["zlmSei"] = {
            "enabled": True,
            "publishUri": self._publish_uri(
                gateway, binding.app, binding.stream_name, token
            ),
            "reconnectMs": self._reconnect_ms(zlm),
        }
        return media

    def create_playback_session(
        self, task_id: str, *, principal: str
    ) -> InferencePlaybackSessionResponse:
        with self.context.database.session() as session:
            task = session.get(InferenceTaskRecord, task_id)
            if task is None:
                raise NotFoundError("inference task", task_id)
            capability = self.preview_capability(session, task)
            if capability.state != PreviewCapabilityState.AVAILABLE:
                raise ConflictError(
                    capability.reason or "preview_unsupported",
                    "Realtime preview is not available for this task",
                    state=capability.state.value,
                )
            binding = session.get(InferenceMediaBindingRecord, task.id)
            if binding is None:
                raise ConflictError(
                    "media_binding_missing",
                    "The task media binding is missing",
                    taskId=task.id,
                )
            gateway = self._gateway(session, binding.gateway_id)
            token = secrets.token_urlsafe(32)
            expires_at = utc_now() + timedelta(seconds=60)
            session.add(
                MediaCredentialRecord(
                    id="media_credential_" + uuid.uuid4().hex,
                    token_hash=_hash_token(token),
                    role=MediaCredentialRole.PLAY.value,
                    gateway_id=gateway.id,
                    task_id=task.id,
                    revision=task.config_revision,
                    app=binding.app,
                    stream_name=binding.stream_name,
                    principal=principal,
                    expires_at=expires_at,
                )
            )
            codec = self._stream_codec(gateway, binding.app, binding.stream_name)
            return InferencePlaybackSessionResponse(
                stream_url=self._play_uri(
                    gateway, binding.app, binding.stream_name, token
                ),
                expires_at=expires_at,
                task_id=task.id,
                revision=task.config_revision,
                gateway_id=gateway.id,
                app=binding.app,
                stream_name=binding.stream_name,
                codec=codec,
                reconnect_ms=self._reconnect_ms(self._zlm_config(task.media_json)),
            )

    def revoke_task_publication(self, session: Session, task_id: str) -> None:
        credentials = session.scalars(
            select(MediaCredentialRecord).where(
                MediaCredentialRecord.task_id == task_id,
                MediaCredentialRecord.role == MediaCredentialRole.PUBLISH.value,
                MediaCredentialRecord.revoked_at.is_(None),
            )
        ).all()
        now = utc_now()
        for credential in credentials:
            credential.revoked_at = now
            self.context.media_secrets.delete_publication_token(credential.id)

    def close_task_stream(self, task_id: str) -> None:
        with self.context.database.session() as session:
            binding = session.get(InferenceMediaBindingRecord, task_id)
            if binding is None:
                return
            gateway = session.get(MediaGatewayRecord, binding.gateway_id)
            if gateway is None:
                return
            secret = self.context.media_secrets.api_secret(gateway.id)
            if secret is None:
                return
            host, port, app, stream = (
                gateway.api_host,
                gateway.api_port,
                binding.app,
                binding.stream_name,
            )
        try:
            ZlmClient(host, port, secret).close_streams(app, stream)
        except ZlmClientError:
            return

    def authorize_hook(
        self,
        gateway_id: str,
        payload: ZlmHookRequest,
        role: MediaCredentialRole,
    ) -> bool:
        token_name = "publishToken" if role == MediaCredentialRole.PUBLISH else "playToken"
        token = self._single_query_value(payload.params, token_name)
        if token is None:
            return False
        with self.context.database.session() as session:
            gateway = session.get(MediaGatewayRecord, gateway_id)
            if (
                gateway is None
                or not gateway.enabled
                or gateway.status != MediaGatewayStatus.ONLINE.value
                or not self._hook_identity_matches(gateway_id, payload.media_server_id)
                or payload.app != gateway.app
                or (role == MediaCredentialRole.PUBLISH and payload.media_schema != "rtsp")
                or (
                    role == MediaCredentialRole.PLAY
                    and payload.media_schema not in {"http", "rtmp"}
                )
            ):
                return False
            credential = session.scalar(
                select(MediaCredentialRecord).where(
                    MediaCredentialRecord.token_hash == _hash_token(token),
                    MediaCredentialRecord.gateway_id == gateway_id,
                    MediaCredentialRecord.role == role.value,
                )
            )
            now = utc_now()
            if (
                credential is None
                or credential.revoked_at is not None
                or credential.app != payload.app
                or credential.stream_name != payload.stream
                or (
                    credential.expires_at is not None
                    and _as_utc(credential.expires_at) <= now
                )
                or (role == MediaCredentialRole.PLAY and credential.used_at is not None)
            ):
                return False
            task = session.get(InferenceTaskRecord, credential.task_id)
            if (
                task is not None
                and (
                    task.config_revision != credential.revision
                    or task.status not in {"deploying", "running", "degraded"}
                )
            ):
                return False
            if role == MediaCredentialRole.PLAY:
                credential.used_at = now
            return True

    def _new_credential(
        self,
        *,
        role: MediaCredentialRole,
        gateway_id: str,
        task_id: str,
        revision: int,
        app: str,
        stream_name: str,
        principal: str,
        expires_at: datetime | None,
    ) -> tuple[str, MediaCredentialRecord]:
        token = secrets.token_urlsafe(32)
        record = MediaCredentialRecord(
            id="media_credential_" + uuid.uuid4().hex,
            token_hash=_hash_token(token),
            role=role.value,
            gateway_id=gateway_id,
            task_id=task_id,
            revision=revision,
            app=app,
            stream_name=stream_name,
            principal=principal,
            expires_at=expires_at,
        )
        with self.context.database.session() as session:
            gateway = self._gateway(session, gateway_id)
            if not gateway.enabled or gateway.status != MediaGatewayStatus.ONLINE.value:
                raise ConflictError(
                    "media_gateway_offline",
                    "The selected media gateway is not online",
                )
            session.add(record)
        return token, record

    def _write_payload_secrets(
        self, gateway_id: str, payload: MediaGatewayCreate | MediaGatewayUpdate
    ) -> None:
        if payload.api_secret is not None:
            self.context.media_secrets.write_api_secret(
                gateway_id, payload.api_secret.get_secret_value()
            )
        if payload.hook_identity is not None:
            self.context.media_secrets.write_hook_identity(
                gateway_id, payload.hook_identity.get_secret_value()
            )

    def _response(self, record: MediaGatewayRecord) -> MediaGatewayResponse:
        api_configured, hook_configured = self.context.media_secrets.configured(record.id)
        return media_gateway_response(
            record,
            api_secret_configured=api_configured,
            hook_identity_configured=hook_configured,
        )

    @staticmethod
    def _gateway(session: Session, gateway_id: str) -> MediaGatewayRecord:
        record = session.get(MediaGatewayRecord, gateway_id)
        if record is None:
            raise NotFoundError("media gateway", gateway_id)
        return record

    def _hook_identity_matches(self, gateway_id: str, candidate: str) -> bool:
        expected = self.context.media_secrets.hook_identity(gateway_id)
        return expected is not None and hmac.compare_digest(candidate, expected)

    @staticmethod
    def _single_query_value(parameters: str, name: str) -> str | None:
        values = parse_qs(parameters.lstrip("?"), keep_blank_values=True).get(name, [])
        return values[0] if len(values) == 1 and values[0] else None

    @staticmethod
    def _validate_hook_config(
        record: MediaGatewayRecord, config: dict[str, object]
    ) -> None:
        expected = {
            "hook.on_publish": f"/media-hooks/zlm/{record.id}/on-publish",
            "hook.on_play": f"/media-hooks/zlm/{record.id}/on-play",
        }
        for key, path in expected.items():
            value = config.get(key)
            if not isinstance(value, str) or path not in value:
                raise ZlmClientError(f"ZLMediaKit {key} does not target this platform")

    @staticmethod
    def _zlm_config(media: dict[str, object]) -> dict[str, object]:
        value = media.get("zlmSei", {})
        return cast(dict[str, object], value) if isinstance(value, dict) else {}

    @staticmethod
    def _reconnect_ms(zlm: dict[str, object]) -> int:
        value = zlm.get("reconnectMs", 1000)
        if isinstance(value, int) and not isinstance(value, bool) and 1000 <= value <= 4000:
            return value
        return 1000

    @staticmethod
    def _format_host(host: str) -> str:
        return f"[{host}]" if ":" in host and not host.startswith("[") else host

    def _publish_uri(
        self,
        gateway: MediaGatewayRecord,
        app: str,
        stream: str,
        token: str,
    ) -> str:
        host = self._format_host(gateway.publish_host)
        return (
            f"rtsp://{host}:{gateway.rtsp_port}/{app}/{stream}"
            f"?publishToken={quote(token, safe='')}"
        )

    def _play_uri(
        self,
        gateway: MediaGatewayRecord,
        app: str,
        stream: str,
        token: str,
    ) -> str:
        host = self._format_host(gateway.playback_host)
        return (
            f"ws://{host}:{gateway.ws_port}/{app}/{stream}.live.flv"
            f"?playToken={quote(token, safe='')}"
        )

    def _stream_codec(
        self, gateway: MediaGatewayRecord, app: str, stream: str
    ) -> Literal["h264", "h265", "unknown"]:
        secret = self.context.media_secrets.api_secret(gateway.id)
        if secret is None:
            return "unknown"
        try:
            media = ZlmClient(gateway.api_host, gateway.api_port, secret).get_media_info(
                app, stream
            )
        except ZlmClientError:
            return "unknown"
        raw_tracks: object = media.get("tracks") if media else None
        if not isinstance(raw_tracks, list):
            return "unknown"
        for raw_track in cast(list[object], raw_tracks):
            if not isinstance(raw_track, dict):
                continue
            track = cast(dict[str, object], raw_track)
            codec = str(track.get("codec_id_name", track.get("codec_id", ""))).lower()
            if codec in {"h264", "avc"}:
                return "h264"
            if codec in {"h265", "hevc"}:
                return "h265"
        return "unknown"
