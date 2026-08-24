#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import secrets
from pathlib import Path

ANCHORS = {
    "zlm-api-secret": "replace-with-zlm-api-secret",
    "zlm-hook-identity": "replace-with-zlm-hook-identity",
}


def configure_compose(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for anchor, placeholder in ANCHORS.items():
        pattern = re.compile(
            rf"^(\s*{re.escape(anchor)}:\s*&{re.escape(anchor)}\s+)(\S+)(\s*)$",
            re.MULTILINE,
        )
        match = pattern.search(text)
        if match is None:
            raise SystemExit(f"ERROR: missing {anchor} anchor in {path}")
        if match.group(2) == placeholder:
            text = pattern.sub(
                lambda current: current.group(1) + secrets.token_hex(32) + current.group(3),
                text,
                count=1,
            )

    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-file", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    configure_compose(args.compose_file.resolve())


if __name__ == "__main__":
    main()
