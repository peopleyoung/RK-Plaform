from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from scripts.lock_zlm_base_image import lock_verified_image
from scripts.run_media_e2e import ffmpeg_filter_value

DIGEST = "zlmediakit/zlmediakit@sha256:" + "a" * 64
REQUIRED_CHECKS = {
    "directWsFlv": True,
    "h264VideoNonblank": True,
    "playAuthorized": True,
    "publishAuthorized": True,
    "seiArrived": True,
}


def verification(image: str = DIGEST, *, passed: bool = True) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "image": image,
        "verifiedAt": "2026-08-21T00:00:00Z",
        "fixtureHash": "b" * 64,
        "passed": passed,
        "checks": REQUIRED_CHECKS,
    }


@pytest.mark.parametrize(
    "image",
    ["zlmediakit/zlmediakit:master", "sha256:" + "a" * 64, "", "example@sha256:123"],
)
def test_lock_rejects_non_repository_digests(tmp_path: Path, image: str) -> None:
    record = tmp_path / "verification.json"
    record.write_text(json.dumps(verification(image)), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable ZLMediaKit RepoDigest"):
        lock_verified_image(image, record, tmp_path / "lock")


def test_lock_requires_matching_successful_complete_verification(tmp_path: Path) -> None:
    output = tmp_path / "lock"
    for payload in (
        verification("zlmediakit/zlmediakit@sha256:" + "c" * 64),
        verification(passed=False),
        {**verification(), "checks": {**REQUIRED_CHECKS, "seiArrived": False}},
    ):
        record = tmp_path / "verification.json"
        record.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            lock_verified_image(DIGEST, record, output)
        assert not output.exists()


def test_lock_atomically_writes_only_the_verified_digest(tmp_path: Path) -> None:
    record = tmp_path / "verification.json"
    output = tmp_path / "zlm-base-image.lock"
    record.write_text(json.dumps(verification()), encoding="utf-8")

    lock_verified_image(DIGEST, record, output)

    assert output.read_text(encoding="utf-8") == DIGEST + "\n"
    assert not output.with_suffix(".lock.tmp").exists()


def test_candidate_verifier_requires_an_immutable_digest() -> None:
    result = subprocess.run(
        [
            "python3",
            "scripts/verify_zlm_candidate.py",
            "--image",
            "zlmediakit/zlmediakit:master",
            "--record",
            "/tmp/unused-zlm-verification.json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "immutable ZLMediaKit RepoDigest" in result.stderr


def test_ffmpeg_sei_filter_value_escapes_avoption_separators() -> None:
    assert ffmpeg_filter_value('{"schema_version":2}') == ("'{\"schema_version\"\\:2}'")
    assert ffmpeg_filter_value(r'{"path":"a\b"}') == '\'{"path"\\:"a\\\\b"}\''


def test_ffmpeg_sei_filter_value_rejects_unrepresentable_quotes() -> None:
    with pytest.raises(ValueError, match="single quotes"):
        ffmpeg_filter_value('{"label":"it\'s"}')
