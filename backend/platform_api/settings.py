from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RKNODE_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "RKNode Platform API"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    data_dir: Path = Path("var")
    database_url: str | None = None
    model_profiles_path: Path = Path("config/model_profiles.json")
    admin_token: SecretStr = Field(default=SecretStr("dev-admin-token"))
    worker_token: SecretStr = Field(default=SecretStr("dev-worker-token"))
    upload_limit_bytes: int = 50 * 1024 * 1024 * 1024
    worker_lease_seconds: int = 120
    worker_offline_seconds: int = 30
    worker_max_retries: int = 2
    inference_node_offline_seconds: int = 30
    public_api_url: str = "http://127.0.0.1:8000/api/v1"
    node_secret_dir_name: str = "node-secrets"
    media_secret_dir_name: str = "media-secrets"
    media_builtin_enabled: bool = False
    media_publish_host: str | None = None
    media_playback_host: str | None = None
    media_rtsp_port: int = Field(default=8554, ge=1, le=65535)
    media_ws_port: int = Field(default=8081, ge=1, le=65535)
    media_api_host: str = "media"
    media_api_port: int = Field(default=80, ge=1, le=65535)
    zlm_api_secret: SecretStr | None = None
    zlm_hook_identity: SecretStr | None = None
    direct_dispatch_enabled: bool = True
    direct_dispatch_interval_seconds: float = 3.0
    direct_node_timeout_seconds: float = 5.0
    node_enrollment_ttl_seconds: int = Field(default=900, ge=60, le=86400)
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @model_validator(mode="after")
    def validate_production_tokens(self) -> Settings:
        if self.environment.lower() not in {"production", "prod"}:
            return self
        admin = self.admin_token.get_secret_value()
        worker = self.worker_token.get_secret_value()
        if admin == "dev-admin-token" or worker == "dev-worker-token":
            raise ValueError("production requires explicit admin and worker tokens")
        if admin == worker:
            raise ValueError("admin and worker tokens must be different")
        return self

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        database_path = (self.data_dir / "platform.db").resolve()
        return f"sqlite:///{database_path}"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def node_secret_dir(self) -> Path:
        return self.data_dir / self.node_secret_dir_name

    @property
    def media_secret_dir(self) -> Path:
        return self.data_dir / self.media_secret_dir_name

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
