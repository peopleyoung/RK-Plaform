#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from string import Template


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if len(value) < 32 or any(character.isspace() for character in value):
        raise SystemExit(f"ERROR: {name} is missing or invalid")
    return value


template_path = Path(
    os.environ.get(
        "RKNODE_ZLM_CONFIG_TEMPLATE", "/opt/rknode-media/config.ini.template"
    )
)
output_path = Path(
    os.environ.get("RKNODE_ZLM_CONFIG_OUTPUT", "/opt/media/conf/config.ini")
)
values = {
    "RKNODE_ZLM_API_SECRET": required("RKNODE_ZLM_API_SECRET"),
    "RKNODE_ZLM_HOOK_IDENTITY": required("RKNODE_ZLM_HOOK_IDENTITY"),
}
rendered = Template(template_path.read_text(encoding="utf-8")).substitute(values)
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        target.write(rendered)
        target.flush()
        os.fsync(target.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, output_path)
finally:
    temporary.unlink(missing_ok=True)

if os.environ.get("RKNODE_ZLM_RENDER_ONLY") == "1":
    raise SystemExit(0)

os.chdir("/opt/media/bin")
os.execv(
    "./MediaServer",
    ["./MediaServer", "-s", "default.pem", "-c", "../conf/config.ini", "-l", "0"],
)
