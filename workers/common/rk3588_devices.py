from __future__ import annotations

from pathlib import Path

RK3588_INFERENCE_PATHS = (
    Path("/dev/dri/card0"),
    Path("/dev/dri/renderD128"),
    Path("/dev/mpp_service"),
    Path("/dev/rga"),
    Path("/dev/dma_heap"),
    # Compose mounts the device-tree base read-only; /proc/device-tree is
    # masked by Docker and is not a reliable path inside an unprivileged node.
    Path("/sys/firmware/devicetree/base/compatible"),
)


def missing_rk3588_inference_paths(paths: tuple[Path, ...] = RK3588_INFERENCE_PATHS) -> list[Path]:
    return [path for path in paths if not path.exists()]
