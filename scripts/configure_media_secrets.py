#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path


NAMES = ("RKNODE_ZLM_API_SECRET", "RKNODE_ZLM_HOOK_IDENTITY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    path = args.env_file.resolve()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    configured = {
        line.split("=", 1)[0]
        for line in lines
        if "=" in line and line.split("=", 1)[0] in NAMES and line.split("=", 1)[1]
    }
    for name in NAMES:
        if name not in configured:
            lines.append(f"{name}={secrets.token_hex(32)}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write("\n".join(lines).rstrip() + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
