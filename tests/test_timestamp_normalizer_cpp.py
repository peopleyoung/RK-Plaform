from __future__ import annotations

import subprocess
from pathlib import Path


def test_rtsp_timestamp_normalizer_cpp_contract(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    pipeline = project / "third_party" / "nv_video_pipeline"
    binary = tmp_path / "timestamp-normalizer-test"

    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(pipeline / "src" / "utils"),
            str(pipeline / "tests" / "TimestampNormalizerTest.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=project,
    )
    subprocess.run([str(binary)], check=True, timeout=10)
