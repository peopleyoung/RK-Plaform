from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from scripts.normalize_voc_segmentation_archive import normalize_archive


def write_fixture(path: Path, sample_ids: tuple[str, ...]) -> None:
    root = "source dataset"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{root}/ImageSets/Segmentation/default.txt",
            "\n".join(sample_ids) + "\n",
        )
        archive.writestr(f"{root}/labelmap.txt", "background:0,0,0::\nng:1,2,3::\n")
        for sample_id in sample_ids:
            archive.writestr(f"{root}/JPEGImages/{sample_id}.png", b"image")
            archive.writestr(f"{root}/SegmentationClass/{sample_id}.png", b"mask")


def test_archive_normalizes_ids_and_all_references(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    output = tmp_path / "fixed.zip"
    write_fixture(source, ("plain_id", "image with spaces"))

    summary = normalize_archive(source, output, "fixed_root")

    assert summary.samples == 2
    assert summary.renamed_ids == 1
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        split = archive.read(
            "fixed_root/ImageSets/Segmentation/default.txt"
        ).decode()
    assert "fixed_root/JPEGImages/image_with_spaces.png" in names
    assert "fixed_root/SegmentationClass/image_with_spaces.png" in names
    assert split == "plain_id\nimage_with_spaces\n"


def test_archive_rejects_normalization_collision(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    write_fixture(source, ("image with spaces", "image_with_spaces"))

    with pytest.raises(ValueError, match="normalization collision"):
        normalize_archive(source, tmp_path / "fixed.zip", "fixed_root")


def test_archive_rejects_missing_mask(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("root/ImageSets/Segmentation/default.txt", "sample_a\nsample_b\n")
        archive.writestr("root/JPEGImages/sample_a.png", b"image")
        archive.writestr("root/JPEGImages/sample_b.png", b"image")
        archive.writestr("root/SegmentationClass/sample_a.png", b"mask")

    with pytest.raises(ValueError, match="images=1, masks=0"):
        normalize_archive(source, tmp_path / "fixed.zip", "fixed_root")
