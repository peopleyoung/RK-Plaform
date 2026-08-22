#!/usr/bin/env python3
"""Lock a ZLMediaKit RepoDigest after the real media gate has passed."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


REPO_DIGEST_PATTERN = re.compile(
    r"^zlmediakit/zlmediakit@sha256:[a-f0-9]{64}$"
)
REQUIRED_CHECKS = frozenset(
    {
        "directWsFlv",
        "h264VideoNonblank",
        "playAuthorized",
        "publishAuthorized",
        "seiArrived",
    }
)


def require_repo_digest(image: str) -> str:
    candidate = image.strip()
    if REPO_DIGEST_PATTERN.fullmatch(candidate) is None:
        raise ValueError(
            "image must be an immutable ZLMediaKit RepoDigest "
            "(zlmediakit/zlmediakit@sha256:<64 lowercase hex>)"
        )
    return candidate


def validate_verification(image: str, record: Path) -> dict[str, Any]:
    candidate = require_repo_digest(image)
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("candidate verification record is missing or invalid") from error
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise ValueError("candidate verification record has an unsupported schema")
    if payload.get("image") != candidate:
        raise ValueError("candidate verification record does not match the image")
    if payload.get("passed") is not True:
        raise ValueError("candidate verification did not pass")
    fixture_hash = payload.get("fixtureHash")
    if not isinstance(fixture_hash, str) or re.fullmatch(r"[a-f0-9]{64}", fixture_hash) is None:
        raise ValueError("candidate verification fixture hash is missing or invalid")
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("candidate verification checks are missing")
    failed = sorted(name for name in REQUIRED_CHECKS if checks.get(name) is not True)
    if failed:
        raise ValueError("candidate verification checks failed: " + ", ".join(failed))
    return payload


def lock_verified_image(image: str, verification: Path, output: Path) -> None:
    candidate = require_repo_digest(image)
    validate_verification(candidate, verification)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(candidate + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock_verified_image(args.image, args.verification, args.output)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
