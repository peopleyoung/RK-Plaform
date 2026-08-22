from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from backend.platform_api.contracts import DatasetFormat


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedDataset:
    root: Path
    labels: tuple[str, ...]


@dataclass(frozen=True)
class VocLabelMap:
    labels: tuple[str, ...]
    color_indexes: dict[tuple[int, int, int], int]


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MASK_SUFFIXES = {".bmp", ".png", ".tif", ".tiff"}
DETECTION_FORMATS = {
    DatasetFormat.YOLO,
    DatasetFormat.COCO_DETECTION,
    DatasetFormat.VOC_DETECTION,
}
SEGMENTATION_FORMATS = {
    DatasetFormat.MASK_PAIRS,
    DatasetFormat.COCO_SEGMENTATION,
    DatasetFormat.VOC_SEGMENTATION,
}


def validate_training_dataset(
    profile_id: str,
    dataset_dir: Path,
    labels: tuple[str, ...],
) -> None:
    if profile_id == "yolo-detect":
        validate_yolo_dataset(dataset_dir, labels)
    elif profile_id == "deeplabv3plus":
        validate_deeplab_dataset(dataset_dir, labels)
    elif profile_id in {"ppocr-det", "ppocr-rec"}:
        validate_ppocr_dataset(dataset_dir, profile_id)
    else:
        raise DatasetValidationError(f"Unsupported training profile: {profile_id}")


def prepare_training_dataset(
    profile_id: str,
    dataset_dir: Path,
    labels: tuple[str, ...],
    dataset_format: str,
    normalized_dir: Path,
) -> PreparedDataset:
    try:
        requested = DatasetFormat(dataset_format)
    except ValueError as error:
        raise DatasetValidationError(f"Unsupported dataset format: {dataset_format}") from error

    if profile_id == "yolo-detect":
        resolved = _resolve_detection_format(dataset_dir, requested)
        if resolved == DatasetFormat.YOLO:
            discovered = _yolo_labels(_unique_file(dataset_dir, "data.yaml"))
            effective_labels = _select_labels(labels, discovered, "YOLO data.yaml")
            validate_yolo_dataset(dataset_dir, effective_labels)
            return PreparedDataset(dataset_dir, effective_labels)
        if resolved == DatasetFormat.VOC_DETECTION:
            discovered = _voc_detection_labels(dataset_dir)
            effective_labels = _select_labels(
                labels, discovered, "Pascal VOC annotations", ordered=False
            )
            _normalize_voc_detection(dataset_dir, normalized_dir, effective_labels)
        else:
            discovered = _coco_dataset_labels(dataset_dir, segmentation=False)
            effective_labels = _select_labels(labels, discovered, "COCO categories")
            _normalize_coco_detection(dataset_dir, normalized_dir, effective_labels)
        validate_yolo_dataset(normalized_dir, effective_labels)
        return PreparedDataset(normalized_dir, effective_labels)

    if profile_id == "deeplabv3plus":
        resolved = _resolve_segmentation_format(dataset_dir, requested)
        if resolved == DatasetFormat.MASK_PAIRS:
            root = _mask_pairs_root(dataset_dir)
            effective_labels = _segmentation_labels(root, _mask_pair_mask_paths(root), labels)
            validate_deeplab_dataset(root, effective_labels)
            return PreparedDataset(root, effective_labels)
        if resolved == DatasetFormat.VOC_SEGMENTATION:
            voc_root = _unique_layout_root(
                _voc_segmentation_roots(dataset_dir), "Pascal VOC segmentation layout"
            )
            splits = _voc_segmentation_splits(voc_root)
            mask_paths = _voc_segmentation_mask_paths(voc_root, splits)
            effective_labels, color_indexes = _voc_segmentation_labels(voc_root, mask_paths, labels)
            _normalize_voc_segmentation(dataset_dir, normalized_dir, splits, color_indexes)
        else:
            discovered = _coco_dataset_labels(dataset_dir, segmentation=True)
            effective_labels = _select_labels(labels, discovered, "COCO categories")
            _normalize_coco_segmentation(dataset_dir, normalized_dir, effective_labels)
        validate_deeplab_dataset(normalized_dir, effective_labels)
        return PreparedDataset(normalized_dir, effective_labels)

    expected = {
        "ppocr-det": DatasetFormat.PPOCR_DETECTION,
        "ppocr-rec": DatasetFormat.PPOCR_RECOGNITION,
    }.get(profile_id)
    if expected is not None:
        if requested not in {DatasetFormat.AUTO, expected}:
            raise DatasetValidationError(
                f"Dataset format '{requested}' is not compatible with profile '{profile_id}'"
            )
        validate_ppocr_dataset(dataset_dir, profile_id)
        return PreparedDataset(dataset_dir, labels)
    raise DatasetValidationError(f"Unsupported training profile: {profile_id}")


def validate_yolo_dataset(dataset_dir: Path, labels: tuple[str, ...]) -> Path:
    data_yaml = _unique_file(dataset_dir, "data.yaml")
    config = _yaml_mapping(data_yaml)
    yaml_labels = list(_yolo_labels(data_yaml, config))
    if labels and yaml_labels != list(labels):
        raise DatasetValidationError(
            "YOLO data.yaml class names must exactly match the dataset metadata classes"
        )

    base = _yolo_base(config, data_yaml, dataset_dir)
    for split in ("train", "val"):
        raw_paths: object = config.get(split)
        paths: list[object] = (
            cast(list[object], raw_paths) if isinstance(raw_paths, list) else [raw_paths]
        )
        if not paths or any(not isinstance(item, str) or not item.strip() for item in paths):
            raise DatasetValidationError(f"YOLO data.yaml must define a non-empty '{split}' path")
        image_count = sum(
            _count_yolo_images(base, item, dataset_dir) for item in cast(list[str], paths)
        )
        if image_count == 0:
            raise DatasetValidationError(f"YOLO '{split}' split contains no supported images")
    return data_yaml


def _yolo_labels(data_yaml: Path, config: dict[str, Any] | None = None) -> tuple[str, ...]:
    config = config or _yaml_mapping(data_yaml)
    names: object = config.get("names")
    yaml_labels: list[str]
    if isinstance(names, list) and all(isinstance(item, str) for item in cast(list[object], names)):
        yaml_labels = cast(list[str], cast(list[object], names))
    elif isinstance(names, dict) and all(
        isinstance(key, (str, int)) and isinstance(value, str)
        for key, value in cast(dict[object, object], names).items()
    ):
        names_mapping = cast(dict[str | int, str], cast(dict[object, object], names))
        try:
            indexed_names = sorted(
                ((int(key), value) for key, value in names_mapping.items()),
                key=lambda item: item[0],
            )
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(
                "YOLO class keys must be consecutive integer indexes"
            ) from error
        if [index for index, _ in indexed_names] != list(range(len(indexed_names))):
            raise DatasetValidationError("YOLO class keys must be consecutive integer indexes")
        yaml_labels = [value for _, value in indexed_names]
    else:
        raise DatasetValidationError("YOLO data.yaml must define class names as a list or map")
    cleaned = [name.strip() for name in yaml_labels]
    if not cleaned or any(not name for name in cleaned):
        raise DatasetValidationError("YOLO data.yaml class names must not be empty")
    if len(cleaned) != len(set(cleaned)):
        raise DatasetValidationError("YOLO data.yaml class names must be unique")
    return tuple(cleaned)


def write_resolved_yolo_config(data_yaml: Path, dataset_dir: Path, output: Path) -> Path:
    config = _yaml_mapping(data_yaml)
    config["path"] = str(_yolo_base(config, data_yaml, dataset_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output


def validate_deeplab_dataset(dataset_dir: Path, labels: tuple[str, ...]) -> None:
    if len(labels) < 2:
        raise DatasetValidationError("DeepLabV3+ requires at least two dataset classes")
    _validate_segmentation_split(dataset_dir, "train", len(labels), required=True)
    has_val_images = (dataset_dir / "images" / "val").is_dir()
    has_val_masks = (dataset_dir / "masks" / "val").is_dir()
    if has_val_images != has_val_masks:
        raise DatasetValidationError(
            "DeepLabV3+ validation images and masks must either both exist or both be absent"
        )
    if has_val_images:
        _validate_segmentation_split(dataset_dir, "val", len(labels), required=True)


def validate_ppocr_dataset(dataset_dir: Path, profile_id: str) -> None:
    for name in ("train.txt", "val.txt"):
        label_file = _unique_file(dataset_dir, name)
        valid_lines = 0
        for line_number, raw_line in enumerate(
            label_file.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            image_name, separator, annotation = raw_line.partition("\t")
            if not separator or not image_name.strip() or not annotation:
                raise DatasetValidationError(
                    f"{name}:{line_number} must contain an image path and tab-separated label"
                )
            image = _contained_path(dataset_dir, image_name.strip(), f"{name}:{line_number}")
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                raise DatasetValidationError(
                    f"{name}:{line_number} image does not exist: {image_name}"
                )
            if profile_id == "ppocr-det":
                try:
                    boxes = json.loads(annotation)
                except json.JSONDecodeError as error:
                    raise DatasetValidationError(
                        f"{name}:{line_number} detection label is not valid JSON"
                    ) from error
                if not isinstance(boxes, list):
                    raise DatasetValidationError(
                        f"{name}:{line_number} detection label must be a JSON list"
                    )
            valid_lines += 1
        if valid_lines == 0:
            raise DatasetValidationError(f"{name} contains no usable samples")


def _validate_segmentation_split(
    dataset_dir: Path, split: str, class_count: int, *, required: bool
) -> None:
    images_dir = dataset_dir / "images" / split
    masks_dir = dataset_dir / "masks" / split
    if required and (not images_dir.is_dir() or not masks_dir.is_dir()):
        raise DatasetValidationError(
            f"DeepLabV3+ requires images/{split} and masks/{split} directories"
        )
    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise DatasetValidationError(f"DeepLabV3+ images/{split} contains no supported images")
    for image in images:
        relative = image.relative_to(images_dir)
        masks = [
            (masks_dir / relative).with_suffix(suffix)
            for suffix in sorted(MASK_SUFFIXES)
            if (masks_dir / relative).with_suffix(suffix).is_file()
        ]
        if len(masks) != 1:
            raise DatasetValidationError(f"No matching mask for images/{split}/{relative}")
        _validate_segmentation_pair(image, masks[0], class_count)


def _validate_segmentation_pair(image: Path, mask: Path, class_count: int) -> None:
    image_module, _ = _pillow_modules()
    try:
        with image_module.open(image) as source, image_module.open(mask) as target:
            if source.size != target.size:
                raise DatasetValidationError(
                    f"Segmentation image and mask dimensions differ: {image} vs {mask}"
                )
            if target.mode not in {"L", "P", "I", "I;16", "I;16B", "I;16L"}:
                raise DatasetValidationError(
                    "Segmentation mask must contain class indexes, "
                    f"got mode '{target.mode}': {mask}"
                )
    except OSError as error:
        raise DatasetValidationError(
            f"Cannot decode segmentation image or mask: {image}"
        ) from error
    lower, upper = _mask_class_range(mask)
    if lower < 0 or upper >= class_count:
        raise DatasetValidationError(
            f"Segmentation mask class ids must be in 0..{class_count - 1}, "
            f"found {lower}..{upper}: {mask}"
        )


def _mask_class_range(mask: Path) -> tuple[int, int]:
    image_module, _ = _pillow_modules()
    try:
        with image_module.open(mask) as target:
            if target.mode not in {"L", "P", "I", "I;16", "I;16B", "I;16L"}:
                raise DatasetValidationError(
                    "Segmentation mask must contain class indexes, "
                    f"got mode '{target.mode}': {mask}"
                )
            extrema: object = target.getextrema()
    except OSError as error:
        raise DatasetValidationError(f"Cannot decode segmentation mask: {mask}") from error
    if not isinstance(extrema, tuple):
        raise DatasetValidationError(f"Cannot determine segmentation class range: {mask}")
    extrema_values = cast(tuple[object, ...], extrema)
    if len(extrema_values) != 2 or not all(
        isinstance(value, (int, float)) for value in extrema_values
    ):
        raise DatasetValidationError(f"Cannot determine segmentation class range: {mask}")
    lower, upper = cast(tuple[int, int], extrema_values)
    return lower, upper


def _count_yolo_images(base: Path, raw_path: str, dataset_dir: Path) -> int:
    path = _contained_path(base, raw_path, "YOLO split", containment_root=dataset_dir)
    if path.is_dir():
        return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return 1
    if path.is_file() and path.suffix.lower() == ".txt":
        count = 0
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            if not raw_line.strip():
                continue
            image = _contained_path(
                path.parent,
                raw_line.strip(),
                f"{path.name}:{line_number}",
                containment_root=dataset_dir,
            )
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                raise DatasetValidationError(
                    f"{path.name}:{line_number} image does not exist: {raw_line.strip()}"
                )
            count += 1
        return count
    raise DatasetValidationError(f"YOLO split path does not exist: {raw_path}")


def _yolo_base(config: dict[str, Any], data_yaml: Path, dataset_dir: Path) -> Path:
    raw_path: object = config.get("path", ".")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise DatasetValidationError("YOLO data.yaml 'path' must be a non-empty string")
    return _contained_path(data_yaml.parent, raw_path, "YOLO dataset path", dataset_dir)


def _contained_path(
    base: Path,
    value: str,
    label: str,
    containment_root: Path | None = None,
) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    root = (containment_root or base).resolve()
    if not resolved.is_relative_to(root):
        raise DatasetValidationError(f"{label} escapes the extracted dataset: {value}")
    return resolved


def _unique_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise DatasetValidationError(
            f"Dataset must contain exactly one '{name}', found {len(matches)}"
        )
    return matches[0]


def _yaml_mapping(path: Path) -> dict[str, Any]:
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"{path.name} root must be an object")
    return cast(dict[str, Any], raw)


def _resolve_detection_format(root: Path, requested: DatasetFormat) -> DatasetFormat:
    if requested != DatasetFormat.AUTO:
        if requested not in DETECTION_FORMATS:
            raise DatasetValidationError(
                f"Dataset format '{requested}' is not valid for object detection"
            )
        return requested
    matches: list[DatasetFormat] = []
    if list(root.rglob("data.yaml")):
        matches.append(DatasetFormat.YOLO)
    if _voc_detection_roots(root):
        matches.append(DatasetFormat.VOC_DETECTION)
    if _coco_annotation_candidates(root, "train") and _coco_annotation_candidates(root, "val"):
        matches.append(DatasetFormat.COCO_DETECTION)
    return _single_detected_format(matches, "object detection")


def _resolve_segmentation_format(root: Path, requested: DatasetFormat) -> DatasetFormat:
    if requested != DatasetFormat.AUTO:
        if requested not in SEGMENTATION_FORMATS:
            raise DatasetValidationError(
                f"Dataset format '{requested}' is not valid for semantic segmentation"
            )
        return requested
    matches: list[DatasetFormat] = []
    if _mask_pairs_roots(root):
        matches.append(DatasetFormat.MASK_PAIRS)
    if _voc_segmentation_roots(root):
        matches.append(DatasetFormat.VOC_SEGMENTATION)
    if _coco_annotation_candidates(root, "train") and _coco_annotation_candidates(root, "val"):
        matches.append(DatasetFormat.COCO_SEGMENTATION)
    return _single_detected_format(matches, "semantic segmentation")


def _single_detected_format(matches: list[DatasetFormat], task: str) -> DatasetFormat:
    if len(matches) != 1:
        found = ", ".join(item.value for item in matches) or "none"
        raise DatasetValidationError(
            f"Could not uniquely detect {task} dataset format; found: {found}. "
            "Select the dataset format explicitly when uploading."
        )
    return matches[0]


def _select_labels(
    existing: tuple[str, ...],
    discovered: tuple[str, ...],
    source: str,
    *,
    ordered: bool = True,
) -> tuple[str, ...]:
    if not discovered:
        raise DatasetValidationError(f"No class names were found in {source}")
    if not existing:
        return discovered
    matches = existing == discovered if ordered else set(existing) == set(discovered)
    if not matches:
        raise DatasetValidationError(
            f"Classes discovered from {source} do not match the stored dataset classes"
        )
    return existing


def _mask_pairs_roots(root: Path) -> list[Path]:
    candidates = [root]
    candidates.extend(path.parent.parent for path in root.rglob("images/train"))
    unique = {path.resolve() for path in candidates}
    return sorted(
        path
        for path in unique
        if (path / "images" / "train").is_dir() and (path / "masks" / "train").is_dir()
    )


def _mask_pairs_root(root: Path) -> Path:
    matches = _mask_pairs_roots(root)
    if len(matches) != 1:
        raise DatasetValidationError(
            "Dataset must contain exactly one images/train + masks/train layout, "
            f"found {len(matches)}"
        )
    return matches[0]


def _mask_pair_mask_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for split in ("train", "val"):
        directory = root / "masks" / split
        if directory.is_dir():
            paths.extend(
                path for path in directory.rglob("*") if path.suffix.lower() in MASK_SUFFIXES
            )
    return sorted(paths)


def _voc_detection_roots(root: Path) -> list[Path]:
    candidates = {path.parent.resolve() for path in root.rglob("Annotations") if path.is_dir()}
    return sorted(
        path
        for path in candidates
        if _voc_detection_image_dirs(path) and (path / "ImageSets" / "Main").is_dir()
    )


def _voc_segmentation_roots(root: Path) -> list[Path]:
    candidates = {
        path.parent.resolve() for path in root.rglob("SegmentationClass") if path.is_dir()
    }
    return sorted(
        path
        for path in candidates
        if (path / "JPEGImages").is_dir() and (path / "ImageSets" / "Segmentation").is_dir()
    )


def _voc_detection_labels(root: Path) -> tuple[str, ...]:
    voc_root = _unique_layout_root(_voc_detection_roots(root), "Pascal VOC detection layout")
    names: set[str] = set()
    splits = _voc_detection_splits(voc_root)
    for split in ("train", "val"):
        for image_id in splits[split]:
            xml_path = voc_root / "Annotations" / f"{image_id}.xml"
            if not xml_path.is_file():
                raise DatasetValidationError(f"VOC annotation does not exist: {xml_path}")
            try:
                annotation = ET.parse(xml_path).getroot()
            except ET.ParseError as error:
                raise DatasetValidationError(f"Invalid VOC XML: {xml_path}") from error
            for item in annotation.findall("object"):
                name = (item.findtext("name") or "").strip()
                if not name:
                    raise DatasetValidationError(
                        f"VOC annotation {xml_path.name} contains an object without a class name"
                    )
                names.add(name)
    if not names:
        raise DatasetValidationError("Pascal VOC annotations contain no object classes")
    return tuple(sorted(names))


def _voc_detection_image_dirs(voc_root: Path) -> list[Path]:
    return [path for path in (voc_root / "JPEGImages", voc_root / "images") if path.is_dir()]


def _voc_detection_splits(voc_root: Path) -> dict[str, list[str]]:
    split_root = voc_root / "ImageSets" / "Main"
    train_path = split_root / "train.txt"
    val_path = split_root / "val.txt"
    if train_path.is_file() and val_path.is_file():
        splits = {"train": _split_ids(train_path), "val": _split_ids(val_path)}
    elif train_path.is_file() or val_path.is_file():
        raise DatasetValidationError(
            "VOC detection must provide both train.txt and val.txt, or an all.txt/default.txt "
            "file for automatic splitting"
        )
    else:
        aggregate_path = next(
            (
                path
                for path in (split_root / "all.txt", split_root / "default.txt")
                if path.is_file()
            ),
            None,
        )
        ids = (
            _split_ids(aggregate_path)
            if aggregate_path is not None
            else _voc_manifest_image_ids(voc_root)
        )
        if len(ids) < 2:
            raise DatasetValidationError(
                "VOC detection requires at least two samples for automatic train/val splitting"
            )
        splits = _deterministic_train_val_split(ids)
    overlap = set(splits["train"]) & set(splits["val"])
    if overlap:
        raise DatasetValidationError(
            f"VOC train and val splits overlap: {', '.join(sorted(overlap)[:5])}"
        )
    return splits


def _voc_manifest_image_ids(voc_root: Path) -> list[str]:
    manifest_path = voc_root / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetValidationError(
            "VOC detection requires train.txt + val.txt, all.txt/default.txt, or manifest.json"
        )
    try:
        payload: object = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"Invalid VOC manifest: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise DatasetValidationError("VOC manifest.json must contain an images array")
    payload_mapping = cast(dict[str, object], payload)
    raw_images = payload_mapping.get("images")
    if not isinstance(raw_images, list):
        raise DatasetValidationError("VOC manifest.json must contain an images array")
    ids: list[str] = []
    for index, item in enumerate(cast(list[object], raw_images), start=1):
        if not isinstance(item, dict):
            raise DatasetValidationError(
                f"VOC manifest.json images[{index}] must contain a file name"
            )
        raw_file = cast(dict[str, object], item).get("file")
        if not isinstance(raw_file, str):
            raise DatasetValidationError(
                f"VOC manifest.json images[{index}] must contain a file name"
            )
        image_id = Path(raw_file).stem
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", image_id):
            raise DatasetValidationError(
                f"VOC manifest.json images[{index}] has an invalid image id"
            )
        ids.append(image_id)
    if not ids:
        raise DatasetValidationError("VOC manifest.json contains no images")
    if len(ids) != len(set(ids)):
        raise DatasetValidationError("VOC manifest.json contains duplicate image ids")
    return ids


def _deterministic_train_val_split(ids: list[str]) -> dict[str, list[str]]:
    ranked = sorted(
        ids,
        key=lambda image_id: (hashlib.sha256(image_id.encode()).digest(), image_id),
    )
    val_count = max(1, math.ceil(len(ids) * 0.2))
    val_ids = set(ranked[:val_count])
    return {
        "train": [image_id for image_id in ids if image_id not in val_ids],
        "val": [image_id for image_id in ids if image_id in val_ids],
    }


def _voc_segmentation_splits(voc_root: Path) -> dict[str, list[str]]:
    split_root = voc_root / "ImageSets" / "Segmentation"
    train_path = split_root / "train.txt"
    val_path = split_root / "val.txt"
    if train_path.is_file() and val_path.is_file():
        splits = {"train": _split_ids(train_path), "val": _split_ids(val_path)}
    elif train_path.is_file() or val_path.is_file():
        raise DatasetValidationError(
            "VOC segmentation must provide both train.txt and val.txt, or only default.txt"
        )
    else:
        default_path = split_root / "default.txt"
        if not default_path.is_file():
            raise DatasetValidationError(
                "VOC segmentation requires train.txt + val.txt, or a CVAT default.txt split"
            )
        ids = _split_ids(default_path)
        if len(ids) < 2:
            raise DatasetValidationError(
                "VOC default.txt requires at least two samples for automatic train/val splitting"
            )
        splits = _deterministic_train_val_split(ids)
    overlap = set(splits["train"]) & set(splits["val"])
    if overlap:
        raise DatasetValidationError(
            f"VOC train and val splits overlap: {', '.join(sorted(overlap)[:5])}"
        )
    return splits


def _voc_segmentation_mask(voc_root: Path, image_id: str) -> Path:
    matches = [
        voc_root / "SegmentationClass" / f"{image_id}{suffix}"
        for suffix in sorted(MASK_SUFFIXES)
        if (voc_root / "SegmentationClass" / f"{image_id}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise DatasetValidationError(
            f"VOC segmentation mask '{image_id}' must resolve to exactly one file, "
            f"found {len(matches)}"
        )
    return matches[0]


def _voc_segmentation_mask_paths(voc_root: Path, splits: dict[str, list[str]]) -> list[Path]:
    return [
        _voc_segmentation_mask(voc_root, image_id)
        for split in ("train", "val")
        for image_id in splits[split]
    ]


def _voc_segmentation_labels(
    voc_root: Path,
    mask_paths: list[Path],
    existing: tuple[str, ...],
) -> tuple[tuple[str, ...], dict[tuple[int, int, int], int] | None]:
    label_map = _voc_label_map(voc_root)
    if label_map is None:
        image_module, _ = _pillow_modules()
        for mask_path in mask_paths:
            try:
                with image_module.open(mask_path) as mask:
                    if mask.mode == "RGB":
                        raise DatasetValidationError(
                            f"VOC RGB masks require a root-level labelmap.txt: {mask_path}"
                        )
            except OSError as error:
                raise DatasetValidationError(
                    f"Cannot decode segmentation mask: {mask_path}"
                ) from error
        return _segmentation_labels(voc_root, mask_paths, existing), None

    declared = _optional_class_labels(voc_root)
    if declared and declared != label_map.labels:
        raise DatasetValidationError(
            "VOC labelmap.txt classes do not match classes.txt or labels.txt"
        )
    effective = _select_labels(existing, label_map.labels, "VOC labelmap.txt")
    if not 2 <= len(effective) <= 256:
        raise DatasetValidationError("VOC labelmap.txt must define between 2 and 256 classes")
    for mask_path in mask_paths:
        _validate_voc_segmentation_mask(mask_path, len(effective), label_map.color_indexes)
    return effective, label_map.color_indexes


def _voc_label_map(voc_root: Path) -> VocLabelMap | None:
    path = voc_root / "labelmap.txt"
    if not path.is_file():
        return None
    labels: list[str] = []
    colors: list[tuple[int, int, int]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(":", maxsplit=3)
        if len(fields) < 2 or not fields[0].strip():
            raise DatasetValidationError(f"labelmap.txt:{line_number} has an invalid class row")
        color_parts = [part.strip() for part in fields[1].split(",")]
        try:
            color = tuple(int(part) for part in color_parts)
        except ValueError as error:
            raise DatasetValidationError(
                f"labelmap.txt:{line_number} has an invalid RGB color"
            ) from error
        if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
            raise DatasetValidationError(
                f"labelmap.txt:{line_number} RGB color must contain three values in 0..255"
            )
        labels.append(fields[0].strip())
        colors.append(color)
    if not labels:
        raise DatasetValidationError("labelmap.txt contains no class rows")
    if len(labels) != len(set(labels)):
        raise DatasetValidationError("labelmap.txt class names must be unique")
    if len(colors) != len(set(colors)):
        raise DatasetValidationError("labelmap.txt RGB colors must be unique")
    return VocLabelMap(
        labels=tuple(labels),
        color_indexes={color: index for index, color in enumerate(colors)},
    )


def _validate_voc_segmentation_mask(
    mask_path: Path,
    class_count: int,
    color_indexes: dict[tuple[int, int, int], int],
) -> None:
    image_module, _ = _pillow_modules()
    try:
        with image_module.open(mask_path) as mask:
            if mask.mode == "RGB":
                colors: object = mask.getcolors(maxcolors=257)
                if not isinstance(colors, list):
                    raise DatasetValidationError(
                        f"VOC RGB mask contains more than 256 colors: {mask_path}"
                    )
                unknown = sorted(
                    cast(tuple[int, int, int], color)
                    for _, color in cast(list[tuple[int, object]], colors)
                    if color not in color_indexes
                )
                if unknown:
                    raise DatasetValidationError(
                        f"VOC RGB mask uses colors missing from labelmap.txt: "
                        f"{unknown[:5]} in {mask_path}"
                    )
                return
    except OSError as error:
        raise DatasetValidationError(f"Cannot decode segmentation mask: {mask_path}") from error
    lower, upper = _mask_class_range(mask_path)
    if lower < 0 or upper >= class_count:
        raise DatasetValidationError(
            f"Segmentation mask class ids must be in 0..{class_count - 1}, "
            f"found {lower}..{upper}: {mask_path}"
        )


def _segmentation_labels(
    label_root: Path,
    mask_paths: list[Path],
    existing: tuple[str, ...],
) -> tuple[str, ...]:
    if not mask_paths:
        raise DatasetValidationError("Semantic segmentation dataset contains no masks")
    lower = 0
    upper = 0
    for mask_path in mask_paths:
        mask_lower, mask_upper = _mask_class_range(mask_path)
        lower = min(lower, mask_lower)
        upper = max(upper, mask_upper)
    if lower < 0 or upper > 255:
        raise DatasetValidationError(
            f"Segmentation mask class ids must be in 0..255, found {lower}..{upper}"
        )
    if upper < 1:
        raise DatasetValidationError(
            "Semantic segmentation masks must contain at least one foreground class id"
        )

    declared = _optional_class_labels(label_root)
    if declared:
        effective = _select_labels(existing, declared, "class label file")
    elif existing:
        effective = existing
    else:
        effective = ("background", *(f"class_{index}" for index in range(1, upper + 1)))
    if not 2 <= len(effective) <= 256:
        raise DatasetValidationError(
            "Semantic segmentation datasets require between 2 and 256 classes"
        )
    if upper >= len(effective):
        raise DatasetValidationError(
            f"Segmentation mask class id {upper} has no corresponding class name"
        )
    return effective


def _optional_class_labels(root: Path) -> tuple[str, ...]:
    matches = [path for name in ("classes.txt", "labels.txt") if (path := root / name).is_file()]
    if len(matches) > 1:
        raise DatasetValidationError(
            "Dataset must contain at most one root-level classes.txt or labels.txt"
        )
    if not matches:
        return ()
    labels = tuple(
        line.strip()
        for line in matches[0].read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if not labels:
        raise DatasetValidationError(f"{matches[0].name} contains no class names")
    if len(labels) != len(set(labels)):
        raise DatasetValidationError(f"{matches[0].name} class names must be unique")
    return labels


def _unique_layout_root(matches: list[Path], label: str) -> Path:
    if len(matches) != 1:
        raise DatasetValidationError(
            f"Dataset must contain exactly one {label}, found {len(matches)}"
        )
    return matches[0]


def _split_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise DatasetValidationError(f"Required split file does not exist: {path}")
    values: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        value = raw_line.strip().split(maxsplit=1)[0] if raw_line.strip() else ""
        if not value:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise DatasetValidationError(f"{path.name}:{line_number} has an invalid image id")
        values.append(value)
    if not values:
        raise DatasetValidationError(f"{path} contains no image ids")
    if len(values) != len(set(values)):
        raise DatasetValidationError(f"{path} contains duplicate image ids")
    return values


def _voc_image(voc_root: Path, image_id: str, filename: str | None = None) -> Path:
    candidates: list[Path] = []
    for images in _voc_detection_image_dirs(voc_root):
        if filename:
            candidates.append(_contained_path(images, filename, f"VOC image {image_id}"))
        candidates.extend(images / f"{image_id}{suffix}" for suffix in sorted(IMAGE_SUFFIXES))
    matches = {path.resolve() for path in candidates if path.is_file()}
    if len(matches) != 1:
        raise DatasetValidationError(
            f"VOC image '{image_id}' must resolve to exactly one image, found {len(matches)}"
        )
    return next(iter(matches))


def _normalize_voc_detection(root: Path, output: Path, labels: tuple[str, ...]) -> None:
    voc_root = _unique_layout_root(_voc_detection_roots(root), "Pascal VOC detection layout")
    class_indexes = {name: index for index, name in enumerate(labels)}
    splits = _voc_detection_splits(voc_root)
    for split in ("train", "val"):
        for image_id in splits[split]:
            xml_path = voc_root / "Annotations" / f"{image_id}.xml"
            if not xml_path.is_file():
                raise DatasetValidationError(f"VOC annotation does not exist: {xml_path}")
            try:
                annotation = ET.parse(xml_path).getroot()
            except ET.ParseError as error:
                raise DatasetValidationError(f"Invalid VOC XML: {xml_path}") from error
            width = _positive_xml_number(annotation, "size/width", xml_path)
            height = _positive_xml_number(annotation, "size/height", xml_path)
            filename = annotation.findtext("filename")
            image = _voc_image(voc_root, image_id, filename.strip() if filename else None)
            target_image = output / "images" / split / f"{image_id}{image.suffix.lower()}"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target_image)
            lines: list[str] = []
            for item in annotation.findall("object"):
                name = (item.findtext("name") or "").strip()
                if name not in class_indexes:
                    raise DatasetValidationError(
                        f"VOC annotation {xml_path.name} uses unknown class '{name}'"
                    )
                box = item.find("bndbox")
                if box is None:
                    raise DatasetValidationError(f"VOC object has no bndbox: {xml_path}")
                xmin = _xml_number(box, "xmin", xml_path)
                ymin = _xml_number(box, "ymin", xml_path)
                xmax = _xml_number(box, "xmax", xml_path)
                ymax = _xml_number(box, "ymax", xml_path)
                if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                    raise DatasetValidationError(f"VOC bbox is outside the image: {xml_path}")
                if xmax <= xmin or ymax <= ymin:
                    raise DatasetValidationError(f"VOC bbox has non-positive size: {xml_path}")
                lines.append(
                    _yolo_line(
                        class_indexes[name], xmin, ymin, xmax - xmin, ymax - ymin, width, height
                    )
                )
            target_label = output / "labels" / split / f"{image_id}.txt"
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    _write_yolo_config(output, labels)


def _xml_number(root: ET.Element, path: str, source: Path) -> float:
    raw = root.findtext(path)
    try:
        value = float(raw) if raw is not None else float("nan")
    except ValueError as error:
        raise DatasetValidationError(f"VOC XML {source} has invalid '{path}'") from error
    if not math.isfinite(value):
        raise DatasetValidationError(f"VOC XML {source} has invalid '{path}'")
    return value


def _positive_xml_number(root: ET.Element, path: str, source: Path) -> float:
    value = _xml_number(root, path, source)
    if value <= 0:
        raise DatasetValidationError(f"VOC XML {source} requires positive '{path}'")
    return value


def _yolo_line(
    class_index: int,
    left: float,
    top: float,
    width: float,
    height: float,
    image_width: float,
    image_height: float,
) -> str:
    center_x = (left + width / 2) / image_width
    center_y = (top + height / 2) / image_height
    return (
        f"{class_index} {center_x:.8f} {center_y:.8f} "
        f"{width / image_width:.8f} {height / image_height:.8f}"
    )


def _write_yolo_config(output: Path, labels: tuple[str, ...]) -> None:
    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(labels)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _coco_annotation_candidates(root: Path, split: str) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*.json"):
        if split not in path.stem.lower():
            continue
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            mapping = cast(dict[object, object], payload)
            if all(
                isinstance(mapping.get(key), list)
                for key in ("images", "annotations", "categories")
            ):
                matches.append(path)
    return sorted(matches)


def _coco_payload(root: Path, split: str) -> tuple[Path, dict[str, Any]]:
    matches = _coco_annotation_candidates(root, split)
    if len(matches) != 1:
        raise DatasetValidationError(
            f"COCO dataset must contain exactly one '{split}' annotation JSON, found {len(matches)}"
        )
    raw: object = json.loads(matches[0].read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"COCO annotation root must be an object: {matches[0]}")
    return matches[0], cast(dict[str, Any], raw)


def _coco_category_rows(payload: dict[str, Any]) -> list[tuple[int, str]]:
    categories = _mapping_list(payload.get("categories"), "COCO categories")
    category_rows: list[tuple[int, str]] = []
    for category in categories:
        category_id = _integer(category.get("id"), "COCO category id")
        name = category.get("name")
        if not isinstance(name, str) or not name.strip():
            raise DatasetValidationError("COCO category name must be a non-empty string")
        category_rows.append((category_id, name.strip()))
    category_rows.sort()
    if len({item[0] for item in category_rows}) != len(category_rows):
        raise DatasetValidationError("COCO category ids must be unique")
    if len({item[1] for item in category_rows}) != len(category_rows):
        raise DatasetValidationError("COCO category names must be unique")
    if not category_rows:
        raise DatasetValidationError("COCO categories must not be empty")
    return category_rows


def _coco_dataset_labels(root: Path, *, segmentation: bool) -> tuple[str, ...]:
    expected: list[tuple[int, str]] | None = None
    for split in ("train", "val"):
        _, payload = _coco_payload(root, split)
        rows = _coco_category_rows(payload)
        if expected is not None and rows != expected:
            raise DatasetValidationError(
                "COCO train and val category ids and names must exactly match"
            )
        expected = rows
    names = tuple(name for _, name in expected or [])
    return ("background", *names) if segmentation else names


def _coco_contract(
    payload: dict[str, Any], labels: tuple[str, ...], *, segmentation: bool
) -> tuple[dict[int, int], list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    category_rows = _coco_category_rows(payload)
    expected = list(labels[1:] if segmentation else labels)
    if [name for _, name in category_rows] != expected:
        detail = "excluding the background class" if segmentation else ""
        raise DatasetValidationError(
            f"COCO category order must exactly match dataset metadata classes {detail}".strip()
        )
    offset = 1 if segmentation else 0
    category_indexes = {
        category_id: index + offset for index, (category_id, _) in enumerate(category_rows)
    }
    images = _mapping_list(payload.get("images"), "COCO images")
    image_ids = [_integer(image.get("id"), "COCO image id") for image in images]
    if len(image_ids) != len(set(image_ids)):
        raise DatasetValidationError("COCO image ids must be unique")
    known_images = set(image_ids)
    annotations = _mapping_list(payload.get("annotations"), "COCO annotations")
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        image_id = _integer(annotation.get("image_id"), "COCO annotation image_id")
        if image_id not in known_images:
            raise DatasetValidationError(f"COCO annotation references unknown image id {image_id}")
        category_id = _integer(annotation.get("category_id"), "COCO annotation category_id")
        if category_id not in category_indexes:
            raise DatasetValidationError(
                f"COCO annotation references unknown category id {category_id}"
            )
        by_image[image_id].append(annotation)
    return category_indexes, images, by_image


def _mapping_list(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise DatasetValidationError(f"{label} must be a list of objects")
    values = cast(list[object], value)
    if not all(isinstance(item, dict) for item in values):
        raise DatasetValidationError(f"{label} must be a list of objects")
    return cast(list[dict[str, Any]], values)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetValidationError(f"{label} must be an integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise DatasetValidationError(f"{label} must be positive")
    return result


def _resolve_coco_image(root: Path, annotation_path: Path, file_name: object) -> Path:
    if not isinstance(file_name, str) or not file_name.strip():
        raise DatasetValidationError("COCO image file_name must be a non-empty string")
    relative = Path(file_name.strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise DatasetValidationError(f"COCO image path escapes the dataset: {file_name}")
    bases = (root, annotation_path.parent, annotation_path.parent.parent)
    direct = {
        (_contained_path(base, str(relative), "COCO image", root)).resolve() for base in bases
    }
    matches = {path for path in direct if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    if not matches:
        matches = {
            path.resolve()
            for path in root.rglob(relative.name)
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path.as_posix().endswith(relative.as_posix())
        }
    if len(matches) != 1:
        raise DatasetValidationError(
            f"COCO image '{file_name}' must resolve to exactly one file, found {len(matches)}"
        )
    return next(iter(matches))


def _normalize_coco_detection(root: Path, output: Path, labels: tuple[str, ...]) -> None:
    for split in ("train", "val"):
        annotation_path, payload = _coco_payload(root, split)
        category_indexes, images, annotations = _coco_contract(payload, labels, segmentation=False)
        if not images:
            raise DatasetValidationError(f"COCO '{split}' split contains no images")
        for image_row in images:
            image_id = _integer(image_row.get("id"), "COCO image id")
            width = _positive_integer(image_row.get("width"), "COCO image width")
            height = _positive_integer(image_row.get("height"), "COCO image height")
            source = _resolve_coco_image(root, annotation_path, image_row.get("file_name"))
            target = output / "images" / split / f"{image_id}{source.suffix.lower()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            lines: list[str] = []
            for annotation in annotations.get(image_id, []):
                bbox = annotation.get("bbox")
                if not isinstance(bbox, list):
                    raise DatasetValidationError("COCO bbox must contain four numbers")
                bbox_values = cast(list[object], bbox)
                if len(bbox_values) != 4 or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in bbox_values
                ):
                    raise DatasetValidationError("COCO bbox must contain four numbers")
                left, top, box_width, box_height = cast(list[float], bbox_values)
                right = min(float(width), left + box_width)
                bottom = min(float(height), top + box_height)
                left = max(0.0, left)
                top = max(0.0, top)
                if right <= left or bottom <= top:
                    raise DatasetValidationError(
                        f"COCO annotation has an invalid bbox for image id {image_id}"
                    )
                category_id = _integer(annotation.get("category_id"), "COCO category_id")
                lines.append(
                    _yolo_line(
                        category_indexes[category_id],
                        left,
                        top,
                        right - left,
                        bottom - top,
                        width,
                        height,
                    )
                )
            label = output / "labels" / split / f"{image_id}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    _write_yolo_config(output, labels)


def _normalize_voc_segmentation(
    root: Path,
    output: Path,
    splits: dict[str, list[str]],
    color_indexes: dict[tuple[int, int, int], int] | None,
) -> None:
    voc_root = _unique_layout_root(_voc_segmentation_roots(root), "Pascal VOC segmentation layout")
    for split in ("train", "val"):
        for image_id in splits[split]:
            image = _voc_image(voc_root, image_id)
            mask = _voc_segmentation_mask(voc_root, image_id)
            image_target = output / "images" / split / f"{image_id}{image.suffix.lower()}"
            mask_target = output / "masks" / split / f"{image_id}.png"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_target)
            _copy_voc_segmentation_mask(mask, mask_target, color_indexes)


def _copy_voc_segmentation_mask(
    source: Path,
    target: Path,
    color_indexes: dict[tuple[int, int, int], int] | None,
) -> None:
    image_module, _ = _pillow_modules()
    try:
        with image_module.open(source) as mask:
            if mask.mode != "RGB":
                if source.suffix.lower() == ".png":
                    shutil.copy2(source, target)
                else:
                    mask.save(target, format="PNG")
                return
            if color_indexes is None:
                raise DatasetValidationError(
                    f"VOC RGB mask requires a root-level labelmap.txt: {source}"
                )
            numpy = cast(Any, importlib.import_module("numpy"))
            pixels = numpy.asarray(mask)
            indexes = numpy.full(pixels.shape[:2], -1, dtype=numpy.int16)
            for color, class_index in color_indexes.items():
                indexes[numpy.all(pixels == color, axis=2)] = class_index
            if bool(numpy.any(indexes < 0)):
                raise DatasetValidationError(
                    f"VOC RGB mask uses colors missing from labelmap.txt: {source}"
                )
            image_module.fromarray(indexes.astype(numpy.uint8)).save(target, format="PNG")
    except OSError as error:
        raise DatasetValidationError(f"Cannot decode segmentation mask: {source}") from error


def _normalize_coco_segmentation(root: Path, output: Path, labels: tuple[str, ...]) -> None:
    image_module, draw_module = _pillow_modules()
    for split in ("train", "val"):
        annotation_path, payload = _coco_payload(root, split)
        category_indexes, images, annotations = _coco_contract(payload, labels, segmentation=True)
        if not images:
            raise DatasetValidationError(f"COCO '{split}' split contains no images")
        for image_row in images:
            image_id = _integer(image_row.get("id"), "COCO image id")
            width = _positive_integer(image_row.get("width"), "COCO image width")
            height = _positive_integer(image_row.get("height"), "COCO image height")
            source = _resolve_coco_image(root, annotation_path, image_row.get("file_name"))
            image_target = output / "images" / split / f"{image_id}{source.suffix.lower()}"
            mask_target = output / "masks" / split / f"{image_id}.png"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, image_target)
            mask = image_module.new("L", (width, height), color=0)
            draw = draw_module.Draw(mask)
            for annotation in sorted(
                annotations.get(image_id, []),
                key=lambda item: _integer(item.get("id", 0), "COCO annotation id"),
            ):
                category_id = _integer(annotation.get("category_id"), "COCO category_id")
                class_index = category_indexes[category_id]
                segmentation = annotation.get("segmentation")
                if isinstance(segmentation, list):
                    _draw_coco_polygons(
                        draw, cast(list[object], segmentation), class_index, image_id
                    )
                elif isinstance(segmentation, dict):
                    instance_mask = _decode_coco_rle(
                        cast(dict[object, object], segmentation),
                        width,
                        height,
                        image_id,
                        image_module,
                    )
                    mask.paste(class_index, mask=instance_mask)
                else:
                    raise DatasetValidationError(
                        f"COCO segmentation for image id {image_id} must be polygon or RLE"
                    )
            mask.save(mask_target, format="PNG")


def _draw_coco_polygons(draw: Any, polygons: list[object], class_index: int, image_id: int) -> None:
    if not polygons:
        raise DatasetValidationError(f"COCO image id {image_id} has an empty polygon list")
    for polygon in polygons:
        if not isinstance(polygon, list):
            raise DatasetValidationError(
                f"COCO image id {image_id} has an invalid segmentation polygon"
            )
        polygon_values = cast(list[object], polygon)
        if (
            len(polygon_values) < 6
            or len(polygon_values) % 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in polygon_values
            )
        ):
            raise DatasetValidationError(
                f"COCO image id {image_id} has an invalid segmentation polygon"
            )
        coordinates = cast(list[float], polygon_values)
        draw.polygon(
            [
                (coordinates[index], coordinates[index + 1])
                for index in range(0, len(coordinates), 2)
            ],
            fill=class_index,
        )


def _decode_coco_rle(
    segmentation: dict[object, object],
    width: int,
    height: int,
    image_id: int,
    image_module: Any,
) -> Any:
    size = segmentation.get("size")
    if size != [height, width]:
        raise DatasetValidationError(
            f"COCO RLE size for image id {image_id} must be [{height}, {width}]"
        )
    raw_counts = segmentation.get("counts")
    if isinstance(raw_counts, list):
        counts = [_integer(value, "COCO RLE count") for value in cast(list[object], raw_counts)]
    elif isinstance(raw_counts, str):
        counts = _decode_compressed_rle_counts(raw_counts, image_id)
    else:
        raise DatasetValidationError(f"COCO RLE counts for image id {image_id} are invalid")
    total = width * height
    column_major = bytearray(total)
    position = 0
    foreground = False
    for count in counts:
        if count < 0 or position + count > total:
            raise DatasetValidationError(f"COCO RLE for image id {image_id} is out of bounds")
        if foreground:
            column_major[position : position + count] = b"\xff" * count
        position += count
        foreground = not foreground
    if position != total:
        raise DatasetValidationError(f"COCO RLE for image id {image_id} has the wrong size")
    row_major = bytearray(total)
    for index, value in enumerate(column_major):
        if value:
            x, y = divmod(index, height)
            row_major[y * width + x] = value
    return image_module.frombytes("L", (width, height), bytes(row_major))


def _pillow_modules() -> tuple[Any, Any]:
    try:
        return (
            cast(Any, importlib.import_module("PIL.Image")),
            cast(Any, importlib.import_module("PIL.ImageDraw")),
        )
    except ImportError as error:
        raise DatasetValidationError(
            "COCO segmentation normalization requires Pillow on the training worker"
        ) from error


def _decode_compressed_rle_counts(value: str, image_id: int) -> list[int]:
    counts: list[int] = []
    position = 0
    while position < len(value):
        result = 0
        shift = 0
        more = True
        byte = 0
        while more:
            if position >= len(value):
                raise DatasetValidationError(
                    f"COCO compressed RLE for image id {image_id} is truncated"
                )
            byte = ord(value[position]) - 48
            if byte < 0 or byte > 63:
                raise DatasetValidationError(
                    f"COCO compressed RLE for image id {image_id} is invalid"
                )
            position += 1
            result |= (byte & 0x1F) << (5 * shift)
            more = bool(byte & 0x20)
            shift += 1
        if byte & 0x10:
            result |= -1 << (5 * shift)
        if len(counts) > 2:
            result += counts[-2]
        counts.append(result)
    return counts
