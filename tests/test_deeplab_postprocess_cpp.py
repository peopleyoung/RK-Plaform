from __future__ import annotations

import subprocess
from pathlib import Path


def test_deeplab_postprocess_cpp_contract(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    pipeline = project / "third_party" / "nv_video_pipeline"
    image = "rknode-rk3588-node:2026.08.24-business"
    command = " ".join(
        (
            "c++ -std=c++17 -Wall -Wextra -Werror",
            "-I/workspace/third_party/nv_video_pipeline/src/rknn_instance",
            "-I/usr/include/opencv4",
            "/workspace/third_party/nv_video_pipeline/tests/DeepLabPostprocessTest.cpp",
            "/workspace/third_party/nv_video_pipeline/src/rknn_instance/DeepLabPostprocess.cpp",
            "-lopencv_imgproc -lopencv_core",
            "-o /build/deeplab-postprocess-test",
            "&& /build/deeplab-postprocess-test",
        )
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform=linux/arm64",
            "--entrypoint",
            "sh",
            "-v",
            f"{project}:/workspace:ro",
            "-v",
            f"{tmp_path}:/build",
            image,
            "-c",
            command,
        ],
        check=True,
        cwd=pipeline,
        timeout=120,
    )
