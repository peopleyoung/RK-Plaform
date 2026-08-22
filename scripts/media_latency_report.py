#!/usr/bin/env python3
"""Summarize and gate synchronized browser media latency samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _numbers(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if isinstance(item, (int, float)) and math.isfinite(item)]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def build_report(payload: dict[str, Any]) -> dict[str, Any]:
    details = payload.get("details")
    if not isinstance(details, dict):
        details = payload
    latency = _numbers(details.get("latencySamplesMs"))
    skew = _numbers(details.get("overlaySkewFrames"))
    reconnect_ms = float(details.get("reconnectDurationMs", 0))
    queue_depth = float(details.get("maxQueueDepth", 0))
    queue_age_ms = float(details.get("maxQueueAgeMs", 0))
    p95_latency = _percentile(latency, 0.95)
    p95_skew = _percentile(skew, 0.95)
    gates = {
        "hasSamples": bool(latency),
        "latencyP95Ms": p95_latency <= 1000,
        "overlaySkewP95Frames": p95_skew <= 1,
        "reconnectMs": reconnect_ms <= 5000,
        "queueDepth": queue_depth <= 120,
        "queueAgeMs": queue_age_ms <= 2000,
    }
    return {
        "schemaVersion": 1,
        "sampleCount": len(latency),
        "latencyMs": {
            "p50": round(_percentile(latency, 0.50), 3),
            "p95": round(p95_latency, 3),
            "max": round(max(latency, default=0.0), 3),
        },
        "overlaySkewFrames": {
            "p95": round(p95_skew, 3),
            "max": round(max(skew, default=0.0), 3),
        },
        "reconnectDurationMs": round(reconnect_ms, 3),
        "maxQueueDepth": int(queue_depth),
        "maxQueueAgeMs": round(queue_age_ms, 3),
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="media E2E JSON record")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        parser.error("media E2E record must be a JSON object")
    report = build_report(payload)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
