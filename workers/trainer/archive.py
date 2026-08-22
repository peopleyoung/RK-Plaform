from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO


class UnsafeArchiveError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractionLimits:
    max_members: int = 500_000
    max_total_bytes: int = 100 * 1024**3
    max_file_bytes: int = 50 * 1024**3
    reserved_disk_bytes: int = 2 * 1024**3

    @classmethod
    def from_env(cls) -> ExtractionLimits:
        return cls(
            max_members=int(os.getenv("RKNODE_DATASET_MAX_MEMBERS", "500000")),
            max_total_bytes=int(
                os.getenv("RKNODE_DATASET_MAX_EXTRACTED_BYTES", str(100 * 1024**3))
            ),
            max_file_bytes=int(os.getenv("RKNODE_DATASET_MAX_FILE_BYTES", str(50 * 1024**3))),
            reserved_disk_bytes=int(
                os.getenv("RKNODE_DATASET_RESERVED_DISK_BYTES", str(2 * 1024**3))
            ),
        )


def extract_dataset(
    archive: Path,
    target: Path,
    limits: ExtractionLimits | None = None,
) -> None:
    resolved_limits = limits or ExtractionLimits.from_env()
    target.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target).free
    available_bytes = max(0, free_bytes - resolved_limits.reserved_disk_bytes)
    total_limit = min(resolved_limits.max_total_bytes, available_bytes)
    if total_limit <= 0:
        raise UnsafeArchiveError("Not enough free disk space to extract the dataset")
    if zipfile.is_zipfile(archive):
        _extract_zip(archive, target, resolved_limits, total_limit)
        return
    if tarfile.is_tarfile(archive):
        _extract_tar(archive, target, resolved_limits, total_limit)
        return
    raise UnsafeArchiveError("Dataset is not a supported ZIP or TAR archive")


def _safe_destination(target: Path, member_name: str) -> Path:
    member = PurePosixPath(member_name.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts:
        raise UnsafeArchiveError(f"Archive member escapes target directory: {member_name}")
    destination = (target / Path(*member.parts)).resolve()
    if not destination.is_relative_to(target.resolve()):
        raise UnsafeArchiveError(f"Archive member escapes target directory: {member_name}")
    return destination


def _extract_zip(
    archive: Path,
    target: Path,
    limits: ExtractionLimits,
    total_limit: int,
) -> None:
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        _check_member_count(len(members), limits)
        declared_total = sum(member.file_size for member in members if not member.is_dir())
        _check_total_size(declared_total, total_limit)
        for member in members:
            destination = _safe_destination(target, member.filename)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            _check_file_size(member.file_size, member.filename, limits)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_file, destination.open("wb") as output_file:
                _copy_limited(input_file, output_file, member.file_size, member.filename)


def _extract_tar(
    archive: Path,
    target: Path,
    limits: ExtractionLimits,
    total_limit: int,
) -> None:
    with tarfile.open(archive) as source:
        count = 0
        total = 0
        for member in source:
            count += 1
            _check_member_count(count, limits)
            if member.issym() or member.islnk() or member.isdev():
                raise UnsafeArchiveError(f"Archive links/devices are not allowed: {member.name}")
            destination = _safe_destination(target, member.name)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            _check_file_size(member.size, member.name, limits)
            total += member.size
            _check_total_size(total, total_limit)
            input_file = source.extractfile(member)
            if input_file is None:
                raise UnsafeArchiveError(f"Could not read archive member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with input_file, destination.open("wb") as output_file:
                _copy_limited(input_file, output_file, member.size, member.name)


def _check_member_count(count: int, limits: ExtractionLimits) -> None:
    if count > limits.max_members:
        raise UnsafeArchiveError(f"Dataset archive contains more than {limits.max_members} members")


def _check_total_size(size: int, total_limit: int) -> None:
    if size > total_limit:
        raise UnsafeArchiveError(
            f"Dataset expands to {size} bytes, exceeding the {total_limit}-byte limit"
        )


def _check_file_size(size: int, name: str, limits: ExtractionLimits) -> None:
    if size > limits.max_file_bytes:
        raise UnsafeArchiveError(
            f"Archive member '{name}' is {size} bytes, exceeding the per-file limit"
        )


def _copy_limited(source: IO[bytes], target: IO[bytes], expected_size: int, name: str) -> None:
    copied = 0
    while chunk := source.read(1024 * 1024):
        copied += len(chunk)
        if copied > expected_size:
            raise UnsafeArchiveError(f"Archive member '{name}' exceeds its declared size")
        target.write(chunk)
    if copied != expected_size:
        raise UnsafeArchiveError(
            f"Archive member '{name}' size mismatch: expected {expected_size}, got {copied}"
        )
