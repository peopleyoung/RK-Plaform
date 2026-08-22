from __future__ import annotations

from pathlib import Path

from workers.trainer.archive import extract_dataset

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def create_calibration_list(
    archive: Path,
    workspace: Path,
    *,
    max_samples: int = 200,
) -> Path:
    extracted = workspace / "calibration"
    extract_dataset(archive, extracted)
    images = sorted(
        path.resolve()
        for path in extracted.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError("Calibration dataset contains no supported image files")
    selected = images[:max_samples]
    list_path = workspace / "calibration-dataset.txt"
    list_path.write_text("\n".join(map(str, selected)) + "\n", encoding="utf-8")
    return list_path
