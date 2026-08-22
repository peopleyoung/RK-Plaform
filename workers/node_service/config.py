from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in os.environ.get(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class NodeServiceSettings:
    token: str | None
    name: str
    kind: str
    accelerator: str
    capabilities: tuple[str, ...]
    work_dir: Path
    endpoint_id: str | None = None
    platform_url: str | None = None
    enrollment_token_file: Path = Path("/run/secrets/rknode-enrollment-token")
    node_token_file: Path = Path("/data/state/node-token")
    request_timeout_seconds: float = 30.0
    host: str = "0.0.0.0"
    port: int = 10081
    version: str = "0.1.0"
    max_concurrency: int = 1
    require_accelerator_device: bool = True
    features: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> NodeServiceSettings:
        raw_token = os.environ.get("RKNODE_NODE_TOKEN", "").strip()
        token = raw_token or None
        name = os.environ.get("RKNODE_NODE_NAME", "").strip()
        kind = os.environ.get("RKNODE_NODE_KIND", "").strip().lower()
        accelerator = os.environ.get("RKNODE_NODE_ACCELERATOR", "").strip().lower()
        capabilities = _csv("RKNODE_NODE_CAPABILITIES")
        missing = [
            env_name
            for env_name, value in (
                ("RKNODE_NODE_NAME", name),
                ("RKNODE_NODE_KIND", kind),
                ("RKNODE_NODE_ACCELERATOR", accelerator),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required node settings: {', '.join(missing)}")
        if token is not None and len(token) < 16:
            raise ValueError("RKNODE_NODE_TOKEN must contain at least 16 characters")
        if kind not in {"trainer", "converter", "inference"}:
            raise ValueError("RKNODE_NODE_KIND must be trainer, converter, or inference")
        expected_accelerators = {
            "trainer": {"cpu", "cuda"},
            "converter": {"rk3588"},
            "inference": {"rk3588"},
        }
        if accelerator not in expected_accelerators[kind]:
            raise ValueError(f"Invalid accelerator {accelerator!r} for {kind} node")
        if not capabilities:
            raise ValueError("RKNODE_NODE_CAPABILITIES must contain at least one capability")
        request_timeout_seconds = float(
            os.environ.get("RKNODE_REQUEST_TIMEOUT_SECONDS", "30")
        )
        if request_timeout_seconds <= 0:
            raise ValueError("RKNODE_REQUEST_TIMEOUT_SECONDS must be greater than zero")
        return cls(
            token=token,
            name=name,
            kind=kind,
            accelerator=accelerator,
            capabilities=capabilities,
            work_dir=Path(os.environ.get("RKNODE_NODE_WORK_DIR", "/data/jobs")),
            endpoint_id=os.environ.get("RKNODE_ENDPOINT_ID", "").strip() or None,
            platform_url=os.environ.get("RKNODE_PLATFORM_URL", "").strip() or None,
            enrollment_token_file=Path(
                os.environ.get(
                    "RKNODE_ENROLLMENT_TOKEN_FILE",
                    "/run/secrets/rknode-enrollment-token",
                )
            ),
            node_token_file=Path(
                os.environ.get("RKNODE_NODE_TOKEN_FILE", "/data/state/node-token")
            ),
            request_timeout_seconds=request_timeout_seconds,
            host=os.environ.get("RKNODE_NODE_HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=int(os.environ.get("RKNODE_NODE_PORT", "10081")),
            version=os.environ.get("RKNODE_NODE_VERSION", "0.1.0").strip() or "0.1.0",
            max_concurrency=max(1, int(os.environ.get("RKNODE_NODE_MAX_CONCURRENCY", "1"))),
            require_accelerator_device=os.environ.get(
                "RKNODE_REQUIRE_NPU_DEVICE", "true"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            features=_csv("RKNODE_NODE_FEATURES"),
        )
