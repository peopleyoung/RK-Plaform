from __future__ import annotations

import argparse
import re
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SPLIT_FILES = (
    "ImageSets/Segmentation/default.txt",
    "ImageSets/Segmentation/train.txt",
    "ImageSets/Segmentation/val.txt",
)
ID_DIRECTORIES = {"Annotations", "JPEGImages", "SegmentationClass", "SegmentationObject"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MASK_SUFFIXES = {".bmp", ".png", ".tif", ".tiff"}
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ArchiveSummary:
    source_root: str
    output_root: str
    samples: int
    renamed_ids: int
    output_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize whitespace in Pascal VOC segmentation sample IDs."
    )
    parser.add_argument("source", type=Path, help="Source VOC segmentation ZIP archive.")
    parser.add_argument("output", type=Path, help="New normalized ZIP archive.")
    parser.add_argument(
        "--root-name",
        default="voc_segmentation_fixed",
        help="ASCII root directory name inside the new archive.",
    )
    return parser.parse_args()


def normalize_sample_id(value: str) -> str:
    normalized = re.sub(r"\s+", "_", value.strip())
    if normalized in {"", ".", ".."} or not SAFE_NAME.fullmatch(normalized):
        raise ValueError(f"Sample ID cannot be normalized safely: {value!r}")
    return normalized


def normalize_archive(source: Path, output: Path, root_name: str) -> ArchiveSummary:
    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError(f"Source is not a ZIP archive: {source}")
    if output == source:
        raise ValueError("Output must not overwrite the source archive")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not SAFE_NAME.fullmatch(root_name) or root_name in {".", ".."}:
        raise ValueError("--root-name must contain only ASCII letters, digits, '.', '_' or '-'")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(source) as input_zip:
            members = input_zip.infolist()
            _validate_members(members)
            source_root = _source_root(members)
            split_ids = _read_split_ids(input_zip, source_root)
            all_ids = sorted({sample_id for ids in split_ids.values() for sample_id in ids})
            id_mapping = _id_mapping(all_ids)
            _validate_pairs(members, source_root, all_ids)

            with zipfile.ZipFile(temporary, "w", allowZip64=True) as output_zip:
                for member in members:
                    target_name = _target_member_name(
                        member.filename, source_root, root_name, id_mapping
                    )
                    target_info = _copy_info(member, target_name)
                    relative = _relative_member(member.filename, source_root)
                    if relative in split_ids:
                        content = "\n".join(id_mapping[item] for item in split_ids[relative]) + "\n"
                        output_zip.writestr(target_info, content.encode("utf-8"))
                    elif member.is_dir():
                        output_zip.writestr(target_info, b"")
                    else:
                        with input_zip.open(member) as reader, output_zip.open(
                            target_info, "w", force_zip64=True
                        ) as writer:
                            shutil.copyfileobj(reader, writer, length=1024 * 1024)

        with zipfile.ZipFile(temporary) as result_zip:
            corrupt_member = result_zip.testzip()
            if corrupt_member is not None:
                raise ValueError(f"Generated ZIP failed CRC validation: {corrupt_member}")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return ArchiveSummary(
        source_root=source_root,
        output_root=root_name,
        samples=len(all_ids),
        renamed_ids=sum(original != normalized for original, normalized in id_mapping.items()),
        output_bytes=output.stat().st_size,
    )


def _validate_members(members: list[zipfile.ZipInfo]) -> None:
    names: set[str] = set()
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member: {name}")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"Archive symlinks are not supported: {name}")
        if name in names:
            raise ValueError(f"Archive contains duplicate member names: {name}")
        names.add(name)


def _source_root(members: list[zipfile.ZipInfo]) -> str:
    roots: set[str] = set()
    for member in members:
        for split_file in SPLIT_FILES:
            suffix = f"/{split_file}"
            if member.filename.endswith(suffix):
                roots.add(member.filename.removesuffix(suffix))
    if len(roots) != 1:
        raise ValueError(f"Expected one VOC segmentation root, found {len(roots)}")
    return roots.pop()


def _read_split_ids(
    archive: zipfile.ZipFile, source_root: str
) -> dict[str, tuple[str, ...]]:
    available = {member.filename for member in archive.infolist() if not member.is_dir()}
    result: dict[str, tuple[str, ...]] = {}
    for split_file in SPLIT_FILES:
        member_name = f"{source_root}/{split_file}"
        if member_name not in available:
            continue
        raw = archive.read(member_name).decode("utf-8-sig")
        ids = tuple(line.strip() for line in raw.splitlines() if line.strip())
        if not ids:
            raise ValueError(f"{split_file} contains no sample IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split_file} contains true duplicate sample IDs")
        result[split_file] = ids
    if "ImageSets/Segmentation/default.txt" not in result and not {
        "ImageSets/Segmentation/train.txt",
        "ImageSets/Segmentation/val.txt",
    }.issubset(result):
        raise ValueError("VOC segmentation requires default.txt or both train.txt and val.txt")
    return result


def _id_mapping(sample_ids: list[str]) -> dict[str, str]:
    mapping = {sample_id: normalize_sample_id(sample_id) for sample_id in sample_ids}
    reverse: dict[str, str] = {}
    for original, normalized in mapping.items():
        previous = reverse.get(normalized)
        if previous is not None and previous != original:
            raise ValueError(
                f"Sample ID normalization collision: {previous!r} and {original!r} "
                f"both become {normalized!r}"
            )
        reverse[normalized] = original
    return mapping


def _validate_pairs(
    members: list[zipfile.ZipInfo], source_root: str, sample_ids: list[str]
) -> None:
    images = _sample_files(members, source_root, "JPEGImages", IMAGE_SUFFIXES)
    masks = _sample_files(members, source_root, "SegmentationClass", MASK_SUFFIXES)
    for sample_id in sample_ids:
        image_count = len(images.get(sample_id, ()))
        mask_count = len(masks.get(sample_id, ()))
        if image_count != 1 or mask_count != 1:
            raise ValueError(
                f"Sample {sample_id!r} must have exactly one image and one mask; "
                f"found images={image_count}, masks={mask_count}"
            )


def _sample_files(
    members: list[zipfile.ZipInfo],
    source_root: str,
    directory: str,
    suffixes: set[str],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    prefix = f"{source_root}/{directory}/"
    for member in members:
        if member.is_dir() or not member.filename.startswith(prefix):
            continue
        filename = member.filename.removeprefix(prefix)
        path = PurePosixPath(filename)
        if len(path.parts) != 1 or path.suffix.lower() not in suffixes:
            continue
        result.setdefault(path.stem, []).append(member.filename)
    return result


def _relative_member(member_name: str, source_root: str) -> str:
    if member_name == f"{source_root}/":
        return ""
    prefix = f"{source_root}/"
    if not member_name.startswith(prefix):
        raise ValueError(f"Archive member is outside the VOC root: {member_name}")
    return member_name.removeprefix(prefix)


def _target_member_name(
    member_name: str,
    source_root: str,
    output_root: str,
    id_mapping: dict[str, str],
) -> str:
    relative = _relative_member(member_name, source_root)
    if not relative:
        return f"{output_root}/"
    path = PurePosixPath(relative)
    if len(path.parts) == 2 and path.parts[0] in ID_DIRECTORIES:
        normalized_stem = id_mapping.get(path.stem)
        if normalized_stem is not None:
            relative = str(path.with_name(f"{normalized_stem}{path.suffix}"))
    return f"{output_root}/{relative}"


def _copy_info(source: zipfile.ZipInfo, target_name: str) -> zipfile.ZipInfo:
    target = zipfile.ZipInfo(target_name, date_time=source.date_time)
    target.compress_type = source.compress_type
    target.comment = source.comment
    target.extra = source.extra
    target.internal_attr = source.internal_attr
    target.external_attr = source.external_attr
    target.create_system = source.create_system
    return target


def main() -> None:
    args = parse_args()
    summary = normalize_archive(args.source, args.output, args.root_name)
    print(f"Source root: {summary.source_root}")
    print(f"Output root: {summary.output_root}")
    print(f"Samples: {summary.samples}")
    print(f"Renamed IDs: {summary.renamed_ids}")
    print(f"Output bytes: {summary.output_bytes}")
    print(f"Output archive: {args.output.resolve()}")


if __name__ == "__main__":
    main()
