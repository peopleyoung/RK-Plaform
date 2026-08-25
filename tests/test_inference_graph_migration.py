from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from scripts.migrate_inference_graph_v1 import (
    backup_database,
    clear_inference_state,
    inspect_database,
    sqlite_path,
)


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE inference_nodes (
            id TEXT PRIMARY KEY,
            desired_revision INTEGER NOT NULL,
            deployment_status TEXT NOT NULL
        );
        CREATE TABLE inference_tasks (id TEXT PRIMARY KEY, node_id TEXT NOT NULL);
        CREATE TABLE deployments (id TEXT PRIMARY KEY);
        CREATE TABLE deployment_targets (id TEXT PRIMARY KEY);
        CREATE TABLE deployment_events (id INTEGER PRIMARY KEY);
        CREATE TABLE inference_graph_revisions (id TEXT PRIMARY KEY);
        CREATE TABLE media_credentials (id TEXT PRIMARY KEY);
        CREATE TABLE inference_media_bindings (task_id TEXT PRIMARY KEY);
        INSERT INTO inference_nodes VALUES ('node-1', 4, 'idle');
        INSERT INTO inference_tasks VALUES ('task-1', 'node-1');
        INSERT INTO deployments VALUES ('deployment-1');
        INSERT INTO deployment_targets VALUES ('target-1');
        INSERT INTO deployment_events VALUES (1);
        INSERT INTO inference_graph_revisions VALUES ('revision-1');
        INSERT INTO media_credentials VALUES ('credential-1');
        INSERT INTO inference_media_bindings VALUES ('task-1');
        """
    )
    connection.commit()
    return connection


def test_graph_migration_backs_up_then_clears_inference_state(tmp_path: Path) -> None:
    database = tmp_path / "platform.db"
    backup = tmp_path / "platform.before-graph-v1.db"
    with _database(database) as connection:
        before = inspect_database(connection)
        assert before["affectedNodeIds"] == ["node-1"]

        backup_database(connection, backup)
        clear_inference_state(connection, ["node-1"])

        after = inspect_database(connection)
        assert set(after["counts"].values()) == {0}
        node = connection.execute(
            "SELECT desired_revision, deployment_status FROM inference_nodes WHERE id = 'node-1'"
        ).fetchone()
        assert node == (5, "deploying")

    with sqlite3.connect(backup) as restored:
        assert restored.execute("SELECT COUNT(*) FROM inference_tasks").fetchone()[0] == 1


def test_graph_migration_rejects_unsafe_database_and_backup_targets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="in-memory"):
        sqlite_path("sqlite:///:memory:")
    with pytest.raises(ValueError, match="only filesystem-backed SQLite"):
        sqlite_path("postgresql://localhost/platform")

    database = tmp_path / "platform.db"
    backup = tmp_path / "existing.db"
    backup.touch()
    with _database(database) as connection, pytest.raises(ValueError, match="already exists"):
        backup_database(connection, backup)
