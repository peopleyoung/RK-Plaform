from __future__ import annotations

import pytest
from backend.platform_api.settings import Settings
from pydantic import SecretStr, ValidationError


def test_production_rejects_default_tokens() -> None:
    with pytest.raises(ValidationError, match="production requires explicit"):
        Settings(environment="production")


def test_production_requires_distinct_tokens() -> None:
    with pytest.raises(ValidationError, match="must be different"):
        Settings(
            environment="production",
            admin_token=SecretStr("shared"),
            worker_token=SecretStr("shared"),
        )


def test_production_accepts_explicit_distinct_tokens() -> None:
    settings = Settings(
        environment="production",
        admin_token=SecretStr("production-admin"),
        worker_token=SecretStr("production-worker"),
    )

    assert settings.environment == "production"


def test_node_enrollment_ttl_defaults_to_fifteen_minutes() -> None:
    assert Settings().node_enrollment_ttl_seconds == 900


def test_node_enrollment_ttl_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKNODE_NODE_ENROLLMENT_TTL_SECONDS", "60")

    assert Settings().node_enrollment_ttl_seconds == 60
