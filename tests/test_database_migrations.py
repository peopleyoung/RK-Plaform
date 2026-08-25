from __future__ import annotations

from pathlib import Path

from backend.platform_api.database import Database
from backend.platform_api.db_models import ServiceEndpointRecord
from backend.platform_api.service import service_endpoint_response
from sqlalchemy import text


def _seed_legacy_service_endpoint(
    database_path: Path, *, token_configured: bool, mode: str = "direct"
) -> Database:
    database = Database(f"sqlite:///{database_path}")
    with database.engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE service_endpoints (
                    id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(120) NOT NULL UNIQUE,
                    kind VARCHAR(30) NOT NULL,
                    endpoint VARCHAR(500) NOT NULL,
                    mode VARCHAR(20) NOT NULL DEFAULT 'pull',
                    scheme VARCHAR(10) NOT NULL DEFAULT 'http',
                    host VARCHAR(255) NOT NULL DEFAULT '',
                    port INTEGER NOT NULL DEFAULT 10081,
                    accelerator VARCHAR(30) NOT NULL,
                    capabilities_json JSON NOT NULL,
                    enabled BOOLEAN NOT NULL,
                    token_configured BOOLEAN NOT NULL DEFAULT 0,
                    probe_status VARCHAR(30) NOT NULL DEFAULT 'unprobed',
                    last_probe_at DATETIME,
                    last_error TEXT,
                    remote_metadata_json JSON,
                    inference_node_id VARCHAR(48),
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO service_endpoints (
                    id, name, kind, endpoint, mode, scheme, host, port,
                    accelerator, capabilities_json, enabled, token_configured,
                    probe_status, remote_metadata_json, created_at, updated_at
                ) VALUES (
                    'service_legacy', 'legacy-trainer', 'trainer',
                    'http://192.0.2.10:10081', :mode, 'http', '192.0.2.10', 10081,
                    'cpu', '["yolo-detect"]', 1, :token_configured, 'online', '{}',
                    '2026-08-15 00:00:00', '2026-08-15 00:00:00'
                )
                """
            ),
            {"mode": mode, "token_configured": token_configured},
        )
    return database


def _seed_legacy_inference_task(database_path: Path) -> Database:
    database = Database(f"sqlite:///{database_path}")
    with database.engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE inference_tasks (
                    id VARCHAR(48) PRIMARY KEY,
                    name VARCHAR(120) NOT NULL,
                    status VARCHAR(30) NOT NULL,
                    release_id VARCHAR(48) NOT NULL,
                    node_id VARCHAR(48) NOT NULL,
                    group_id VARCHAR(48),
                    input_uri VARCHAR(2000) NOT NULL,
                    interval INTEGER NOT NULL DEFAULT 1,
                    thresholds_json JSON NOT NULL DEFAULT '{}',
                    output_json JSON NOT NULL DEFAULT '{}',
                    media_json JSON NOT NULL DEFAULT '{}',
                    analytics_json JSON NOT NULL DEFAULT '{}',
                    npu_core_mask VARCHAR(30) NOT NULL DEFAULT 'auto',
                    npu_core_policy VARCHAR(30) NOT NULL DEFAULT 'shared',
                    config_revision INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO inference_tasks (
                    id, name, status, release_id, node_id, input_uri,
                    created_at, updated_at
                ) VALUES (
                    'itask_legacy', 'legacy-inference', 'stopped',
                    'release_legacy', 'inode_legacy', 'rtsp://camera/legacy',
                    '2026-08-19 00:00:00', '2026-08-19 00:00:00'
                )
                """
            )
        )
    return database


def test_legacy_inference_task_gains_single_context_worker_defaults(tmp_path: Path) -> None:
    database = _seed_legacy_inference_task(tmp_path / "legacy-inference.db")

    database.create_schema()

    with database.engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT context_count, worker_count "
                "FROM inference_tasks WHERE id = 'itask_legacy'"
            )
        ).one()
    assert row.context_count == 1
    assert row.worker_count == 1


def test_legacy_output_uri_task_is_marked_for_media_migration(tmp_path: Path) -> None:
    database = _seed_legacy_inference_task(tmp_path / "legacy-media.db")
    with database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE inference_tasks SET media_json = :media "
                "WHERE id = 'itask_legacy'"
            ),
            {
                "media": (
                    '{"decoder":"rkmpp","zlmSei":{"enabled":true,'
                    '"outputUri":"rtsp://legacy/live/task"}}'
                )
            },
        )

    database.create_schema()
    database.create_schema()

    with database.engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT media_migration_required FROM inference_tasks "
                "WHERE id = 'itask_legacy'"
            )
        ).one()
    assert row.media_migration_required == 1


def test_existing_direct_endpoint_with_token_is_migrated_as_enrolled(
    tmp_path: Path,
) -> None:
    database = _seed_legacy_service_endpoint(
        tmp_path / "legacy.db", token_configured=True
    )

    database.create_schema()

    with database.session() as session:
        record = session.get(ServiceEndpointRecord, "service_legacy")
        assert record is not None
        assert record.enrollment_status == "enrolled"
        assert record.enrollment_token_hash is None


def test_existing_direct_endpoint_without_token_is_migrated_as_pending(
    tmp_path: Path,
) -> None:
    database = _seed_legacy_service_endpoint(
        tmp_path / "legacy.db", token_configured=False
    )

    database.create_schema()

    with database.session() as session:
        record = session.get(ServiceEndpointRecord, "service_legacy")
        assert record is not None
        assert record.enrollment_status == "pending"


def test_pull_endpoint_remains_outside_pending_enrollment(tmp_path: Path) -> None:
    database = _seed_legacy_service_endpoint(
        tmp_path / "legacy.db", token_configured=False, mode="pull"
    )

    database.create_schema()

    with database.session() as session:
        record = session.get(ServiceEndpointRecord, "service_legacy")
        assert record is not None
        assert record.enrollment_status == "enrolled"


def test_endpoint_response_exposes_state_without_secret_material(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'response.db'}")
    database.create_schema()
    with database.session() as session:
        record = ServiceEndpointRecord(
            id="service_redaction",
            name="redaction-trainer",
            kind="trainer",
            endpoint="http://192.0.2.20:10081",
            mode="direct",
            scheme="http",
            host="192.0.2.20",
            port=10081,
            accelerator="cpu",
            capabilities_json=["yolo-detect"],
            enabled=True,
            token_configured=False,
            enrollment_status="pending",
            enrollment_token_hash="a" * 64,
            probe_status="unprobed",
        )
        session.add(record)
        session.flush()

        body = service_endpoint_response(record).model_dump(mode="json", by_alias=True)

    assert body["enrollmentStatus"] == "pending"
    assert body["enrollmentExpiresAt"] is None
    assert "enrollmentToken" not in body
    assert "enrollmentTokenHash" not in body
