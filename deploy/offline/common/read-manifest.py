#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path("manifest.json").read_text(encoding="utf-8"))
    value = manifest[args.field]
    if args.list:
        if not isinstance(value, list):
            raise SystemExit(f"ERROR: {args.field} is not a list")
        for item in value:
            if isinstance(item, dict):
                print(item["tag"])
            else:
                print(item)
        return
    if isinstance(value, (dict, list)):
        raise SystemExit(f"ERROR: {args.field} is not a scalar")
    print(value)


if __name__ == "__main__":
    main()
