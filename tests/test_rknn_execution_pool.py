from __future__ import annotations

import subprocess
from pathlib import Path


def test_rknn_execution_pool_cpp_contract(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    pipeline = project / "third_party" / "nv_video_pipeline"
    binary = tmp_path / "rknn-execution-pool-test"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            "-I",
            str(pipeline / "3rdparty" / "rknpu2" / "include"),
            "-I",
            str(pipeline / "src" / "rknn_instance"),
            str(pipeline / "tests" / "RknnExecutionPoolTest.cpp"),
            str(pipeline / "src" / "rknn_instance" / "RknnExecutionPool.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=project,
    )
    subprocess.run([str(binary)], check=True, timeout=10)
