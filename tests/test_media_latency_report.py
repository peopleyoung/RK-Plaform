from __future__ import annotations

from scripts.media_latency_report import build_report


def test_latency_report_calculates_percentiles_and_gates() -> None:
    report = build_report(
        {
            "details": {
                "latencySamplesMs": [10, 20, 30, 40],
                "overlaySkewFrames": [0, 0.5, 1],
                "reconnectDurationMs": 2500,
                "maxQueueDepth": 4,
                "maxQueueAgeMs": 200,
            }
        }
    )

    assert report["sampleCount"] == 4
    assert report["latencyMs"]["p50"] == 25
    assert report["latencyMs"]["p95"] == 38.5
    assert report["passed"] is True


def test_latency_report_rejects_missing_samples_and_unbounded_queue() -> None:
    report = build_report(
        {"details": {"latencySamplesMs": [], "maxQueueDepth": 121}}
    )

    assert report["gates"]["hasSamples"] is False
    assert report["gates"]["queueDepth"] is False
    assert report["passed"] is False
