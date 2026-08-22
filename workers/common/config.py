from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class WorkerConfig:
    api_url: str
    token: str
    name: str
    kind: str
    accelerator: str
    capabilities: tuple[str, ...]
    work_dir: Path
    version: str = "0.1.0"
    max_concurrency: int = 1
    poll_seconds: float = 3.0
    request_timeout_seconds: float = 30.0
    lease_renew_seconds: float = 30.0
    workspace_reconcile_seconds: float = 30.0

    @classmethod
    def from_env(cls, *, token_override: str | None = None) -> WorkerConfig:
        api_url = os.getenv("RKNODE_API_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")
        token = token_override or os.getenv("RKNODE_WORKER_TOKEN", "")
        name = os.getenv("RKNODE_WORKER_NAME", "")
        kind = os.getenv("RKNODE_WORKER_KIND", "")
        accelerator = os.getenv("RKNODE_WORKER_ACCELERATOR", "")
        missing = [
            env_name
            for env_name, value in (
                ("RKNODE_WORKER_TOKEN", token),
                ("RKNODE_WORKER_NAME", name),
                ("RKNODE_WORKER_KIND", kind),
                ("RKNODE_WORKER_ACCELERATOR", accelerator),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required worker settings: {', '.join(missing)}")
        capabilities = _csv_env("RKNODE_WORKER_CAPABILITIES", "")
        if not capabilities:
            raise ValueError("RKNODE_WORKER_CAPABILITIES must contain at least one profile ID")
        return cls(
            api_url=api_url,
            token=token,
            name=name,
            kind=kind,
            accelerator=accelerator,
            capabilities=capabilities,
            work_dir=Path(os.getenv("RKNODE_WORK_DIR", "/data/jobs")),
            version=os.getenv("RKNODE_WORKER_VERSION", "0.1.0"),
            max_concurrency=int(os.getenv("RKNODE_WORKER_MAX_CONCURRENCY", "1")),
            poll_seconds=float(os.getenv("RKNODE_WORKER_POLL_SECONDS", "3")),
            request_timeout_seconds=float(os.getenv("RKNODE_REQUEST_TIMEOUT_SECONDS", "30")),
            lease_renew_seconds=float(os.getenv("RKNODE_LEASE_RENEW_SECONDS", "30")),
            workspace_reconcile_seconds=max(
                1.0, float(os.getenv("RKNODE_WORKSPACE_RECONCILE_SECONDS", "30"))
            ),
        )
