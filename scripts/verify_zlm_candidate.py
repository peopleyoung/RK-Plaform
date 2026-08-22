#!/usr/bin/env python3
"""Run the real browser media gate for an immutable ZLMediaKit candidate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lock_zlm_base_image import require_repo_digest, validate_verification


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    try:
        image = require_repo_digest(args.image)
    except ValueError as error:
        parser.error(str(error))

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_media_e2e.py"),
            "--image",
            image,
            "--record",
            str(args.record),
            "--candidate-gate",
        ],
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    try:
        validate_verification(image, args.record)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
