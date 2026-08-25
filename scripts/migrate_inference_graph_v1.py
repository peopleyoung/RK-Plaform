#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

INFERENCE_TABLES = (
    "deployment_events",
    "deployment_targets",
    "deployments",
    "inference_graph_revisions",
    "media_credentials",
    "inference_media_bindings",
    "inference_tasks",
)


def default_database() -> str:
    configured = os.getenv("RKNODE_DATABASE_URL")
    if configured:
        return configured
    data_dir = Path(os.getenv("RKNODE_DATA_DIR", "var"))
    return str(data_dir / "platform.db")


def sqlite_path(value: str) -> Path:
    if value in {"sqlite://", "sqlite:///:memory:", ":memory:"}:
        raise ValueError("in-memory SQLite databases cannot be migrated")
    if value.startswith("sqlite:///"):
        value = value.removeprefix("sqlite:///")
    elif "://" in value:
        raise ValueError("only filesystem-backed SQLite databases are supported")
    return Path(value).expanduser().resolve()


def existing_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def inspect_database(connection: sqlite3.Connection) -> dict[str, object]:
    tables = existing_tables(connection)
    counts = {
        table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table in INFERENCE_TABLES
        if table in tables
    }
    affected_nodes: list[str] = []
    if "inference_tasks" in tables:
        affected_nodes = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT node_id FROM inference_tasks ORDER BY node_id"
            )
        ]
    return {"counts": counts, "affectedNodeIds": affected_nodes}


def backup_database(connection: sqlite3.Connection, destination: Path) -> None:
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"backup destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as backup:
        connection.backup(backup)


def clear_inference_state(connection: sqlite3.Connection, affected_nodes: Sequence[str]) -> None:
    tables = existing_tables(connection)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("BEGIN IMMEDIATE")
    try:
        for table in INFERENCE_TABLES:
            if table in tables:
                connection.execute(f'DELETE FROM "{table}"')
        if affected_nodes and "inference_nodes" in tables:
            placeholders = ",".join("?" for _ in affected_nodes)
            connection.execute(
                "UPDATE inference_nodes "
                "SET desired_revision = desired_revision + 1, deployment_status = 'deploying' "
                f"WHERE id IN ({placeholders})",
                tuple(affected_nodes),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Back up and remove legacy inference state before graph-contract rollout."
    )
    command.add_argument("--database", default=default_database())
    command.add_argument("--backup", type=Path)
    command.add_argument("--execute", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    database = sqlite_path(args.database)
    if not database.is_file():
        raise SystemExit(f"database does not exist: {database}")
    if args.execute and args.backup is None:
        raise SystemExit("--execute requires --backup PATH")

    with sqlite3.connect(database) as connection:
        before = inspect_database(connection)
        result: dict[str, object] = {
            "database": str(database),
            "mode": "execute" if args.execute else "dry-run",
            "before": before,
        }
        if args.execute:
            assert args.backup is not None
            backup_database(connection, args.backup)
            clear_inference_state(connection, before["affectedNodeIds"])
            result["backup"] = str(args.backup.expanduser().resolve())
            result["after"] = inspect_database(connection)
            result["completedAt"] = datetime.now(UTC).isoformat()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
