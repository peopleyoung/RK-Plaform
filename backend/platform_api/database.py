from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .db_models import Base, InferenceTaskRecord


class Database:
    def __init__(self, url: str) -> None:
        connect_args: dict[str, object] = {}
        engine_options: dict[str, object] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        if url in {"sqlite://", "sqlite:///:memory:"}:
            engine_options["poolclass"] = StaticPool
        self.engine: Engine = create_engine(
            url,
            connect_args=connect_args,
            **engine_options,
        )
        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection: Any, _: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)
        columns = {column["name"] for column in inspect(self.engine).get_columns("datasets")}
        if "dataset_format" not in columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE datasets ADD COLUMN dataset_format "
                        "VARCHAR(40) NOT NULL DEFAULT 'auto'"
                    )
                )
        inference_columns = {
            column["name"] for column in inspect(self.engine).get_columns("inference_tasks")
        }
        missing_inference_columns = {
            "npu_core_mask": "VARCHAR(30) NOT NULL DEFAULT 'auto'",
            "npu_core_policy": "VARCHAR(30) NOT NULL DEFAULT 'shared'",
            "media_json": "JSON NOT NULL DEFAULT '{}'",
            "analytics_json": "JSON NOT NULL DEFAULT '{}'",
            "context_count": "INTEGER NOT NULL DEFAULT 1",
            "worker_count": "INTEGER NOT NULL DEFAULT 1",
            "media_migration_required": "BOOLEAN NOT NULL DEFAULT 0",
        }
        with self.engine.begin() as connection:
            for name, definition in missing_inference_columns.items():
                if name not in inference_columns:
                    connection.execute(
                        text(f"ALTER TABLE inference_tasks ADD COLUMN {name} {definition}")
                    )
        refreshed_inference_columns = {
            column["name"] for column in inspect(self.engine).get_columns("inference_tasks")
        }
        model_inference_columns = {column.name for column in InferenceTaskRecord.__table__.columns}
        if model_inference_columns.issubset(refreshed_inference_columns):
            with self.session() as session:
                tasks = session.scalars(select(InferenceTaskRecord)).all()
                for task in tasks:
                    media = task.media_json if isinstance(task.media_json, dict) else {}
                    zlm = media.get("zlmSei")
                    task.media_migration_required = bool(
                        isinstance(zlm, dict)
                        and zlm.get("outputUri")
                        and not zlm.get("gatewayId")
                    )
        endpoint_columns = {
            column["name"] for column in inspect(self.engine).get_columns("service_endpoints")
        }
        missing_endpoint_columns = {
            "mode": "VARCHAR(20) NOT NULL DEFAULT 'pull'",
            "scheme": "VARCHAR(10) NOT NULL DEFAULT 'http'",
            "host": "VARCHAR(255) NOT NULL DEFAULT ''",
            "port": "INTEGER NOT NULL DEFAULT 10081",
            "token_configured": "BOOLEAN NOT NULL DEFAULT 0",
            "enrollment_status": "VARCHAR(20) NOT NULL DEFAULT 'enrolled'",
            "enrollment_token_hash": "VARCHAR(64)",
            "enrollment_expires_at": "DATETIME",
            "enrollment_claimed_at": "DATETIME",
            "enrolled_at": "DATETIME",
            "probe_status": "VARCHAR(30) NOT NULL DEFAULT 'unprobed'",
            "last_probe_at": "DATETIME",
            "last_error": "TEXT",
            "remote_metadata_json": "JSON",
            "inference_node_id": "VARCHAR(48)",
        }
        with self.engine.begin() as connection:
            for name, definition in missing_endpoint_columns.items():
                if name not in endpoint_columns:
                    connection.execute(
                        text(f"ALTER TABLE service_endpoints ADD COLUMN {name} {definition}")
                    )
            connection.execute(
                text(
                    "UPDATE service_endpoints "
                    "SET enrollment_status = CASE "
                    "WHEN mode = 'direct' AND token_configured = 0 THEN 'pending' "
                    "ELSE 'enrolled' END "
                    "WHERE enrollment_status IS NULL "
                    "OR enrollment_status NOT IN ('pending', 'claimed', 'enrolled') "
                    "OR (mode = 'direct' AND token_configured = 0 "
                    "AND enrollment_token_hash IS NULL AND enrollment_status = 'enrolled')"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_service_endpoints_enrollment_status "
                    "ON service_endpoints (enrollment_status)"
                )
            )

    def drop_schema(self) -> None:
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
