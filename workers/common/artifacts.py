from __future__ import annotations

import re

UNSAFE_ARTIFACT_STEM = re.compile(r"[^A-Za-z0-9._-]+")


def model_artifact_stem(model_name: str, width: int, height: int) -> str:
    """Build a storage-safe '<model>-<width>x<height>' artifact stem."""
    if width <= 0 or height <= 0:
        raise ValueError("Artifact dimensions must be positive")
    safe_name = UNSAFE_ARTIFACT_STEM.sub("_", model_name).strip("._-")[:120]
    if not safe_name:
        safe_name = "model"
    return f"{safe_name}-{width}x{height}"
