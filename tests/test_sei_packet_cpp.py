from __future__ import annotations

import subprocess
from pathlib import Path


def test_sei_packet_and_media_url_cpp_contract(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    pipeline = project / "third_party" / "nv_video_pipeline"
    sei_binary = tmp_path / "sei-packet-test"
    url_binary = tmp_path / "media-url-test"

    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(pipeline / "src" / "objects"),
            "-I",
            str(pipeline / "src" / "utils"),
            str(pipeline / "tests" / "SeiPacketTest.cpp"),
            str(pipeline / "src" / "utils" / "SeiPacket.cpp"),
            "-o",
            str(sei_binary),
        ],
        check=True,
        cwd=project,
    )
    subprocess.run([str(sei_binary)], check=True, timeout=10)

    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(pipeline / "src" / "utils"),
            str(pipeline / "tests" / "MediaUrlTest.cpp"),
            str(pipeline / "src" / "utils" / "MediaUrl.cpp"),
            "-o",
            str(url_binary),
        ],
        check=True,
        cwd=project,
    )
    subprocess.run([str(url_binary)], check=True, timeout=10)
