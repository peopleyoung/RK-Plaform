from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

JOB_WORKSPACE_NAME = re.compile(r"(?:train|convert)_[0-9a-f]{32}")


@dataclass(frozen=True)
class WorkspaceCleanup:
    job_id: str
    bytes_removed: int


def prune_orphan_workspaces(
    work_dir: Path,
    retained_job_ids: set[str],
) -> list[WorkspaceCleanup]:
    root = work_dir.resolve()
    if not root.is_dir():
        return []

    removed: list[WorkspaceCleanup] = []
    for candidate in root.iterdir():
        if (
            JOB_WORKSPACE_NAME.fullmatch(candidate.name) is None
            or candidate.name in retained_job_ids
            or candidate.is_symlink()
            or not candidate.is_dir()
            or candidate.resolve().parent != root
        ):
            continue
        bytes_removed = _directory_bytes(candidate)
        shutil.rmtree(candidate)
        removed.append(WorkspaceCleanup(candidate.name, bytes_removed))
    return removed


def _directory_bytes(directory: Path) -> int:
    total = 0
    for candidate in directory.rglob("*"):
        try:
            if not candidate.is_symlink() and candidate.is_file():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total
