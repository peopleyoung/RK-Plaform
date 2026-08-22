from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import onnx
import pytest
from backend.platform_api.contracts import Resolution
from backend.platform_api.profiles import ModelProfileRegistry
from onnx import TensorProto, helper
from PIL import Image
from workers.trainer.archive import ExtractionLimits, UnsafeArchiveError, extract_dataset
from workers.trainer.dataset import (
    DatasetValidationError,
    prepare_training_dataset,
    validate_ppocr_dataset,
    validate_yolo_dataset,
)
from workers.trainer.manifest import build_deployment_manifest


def registry() -> ModelProfileRegistry:
    return ModelProfileRegistry(Path(__file__).parents[1] / "config/model_profiles.json")


def write_model(
    path: Path,
    shape: list[int | str],
    *,
    opset: int = 12,
    input_name: str = "images",
    output_name: str = "output0",
) -> None:
    source = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)
    target = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)
    graph = helper.make_graph(
        [helper.make_node("Identity", [input_name], [output_name])],
        "deployment",
        [source],
        [target],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.save(model, path)


def test_manifest_is_derived_from_actual_onnx_shape(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    write_model(model_path, [1, 3, 384, 640])

    manifest = build_deployment_manifest(
        registry(),
        job_id="train_test",
        profile_id="yolo-detect",
        variant="yolov8n",
        resolution=Resolution(width=640, height=384),
        onnx_path=model_path,
        labels=["scratch"],
    )

    assert manifest.input.shape == [1, 3, 384, 640]
    assert manifest.output_contract == "rknn_yolo_dfl_split_heads_v1"
    assert manifest.onnx_sha256


@pytest.mark.parametrize(
    ("profile_id", "variant", "input_name", "output_name", "width", "height"),
    [
        ("yolo-detect", "yolov8n", "images", "output0", 640, 384),
        ("deeplabv3plus", "mobilenet_v2", "images", "logits", 768, 512),
        ("ppocr-det", "ppocrv4_det", "x", "maps", 640, 384),
        ("ppocr-rec", "ppocrv4_rec", "x", "ctc_logits", 640, 64),
    ],
)
def test_every_profile_emits_requested_static_shape(
    tmp_path: Path,
    profile_id: str,
    variant: str,
    input_name: str,
    output_name: str,
    width: int,
    height: int,
) -> None:
    model_path = tmp_path / f"{profile_id}.onnx"
    write_model(
        model_path,
        [1, 3, height, width],
        opset=14 if profile_id.startswith("ppocr-") else 12,
        input_name=input_name,
        output_name=output_name,
    )

    deployment = build_deployment_manifest(
        registry(),
        job_id="train_test",
        profile_id=profile_id,
        variant=variant,
        resolution=Resolution(width=width, height=height),
        onnx_path=model_path,
        labels=["fixture"],
    )

    assert deployment.resolution == Resolution(width=width, height=height)
    assert deployment.input.shape == [1, 3, height, width]
    assert deployment.input.name == input_name
    assert deployment.outputs[0].name == output_name


def test_manifest_rejects_dynamic_export(tmp_path: Path) -> None:
    model_path = tmp_path / "dynamic.onnx"
    write_model(model_path, [1, 3, "height", "width"])

    with pytest.raises(ValueError, match="positive static dimensions"):
        build_deployment_manifest(
            registry(),
            job_id="train_test",
            profile_id="yolo-detect",
            variant="yolov8n",
            resolution=Resolution(width=640, height=384),
            onnx_path=model_path,
            labels=[],
        )


def test_dataset_extraction_rejects_zip_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("../../outside.txt", "unsafe")

    with pytest.raises(UnsafeArchiveError, match="escapes"):
        extract_dataset(archive, tmp_path / "dataset")


def test_dataset_extraction_rejects_tar_links(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        target.addfile(info, io.BytesIO())

    with pytest.raises(UnsafeArchiveError, match="not allowed"):
        extract_dataset(archive, tmp_path / "dataset")


def test_dataset_extraction_enforces_expanded_size_limit(tmp_path: Path) -> None:
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        target.writestr("large.bin", b"0" * 1024)

    with pytest.raises(UnsafeArchiveError, match="exceeding"):
        extract_dataset(
            archive,
            tmp_path / "dataset",
            ExtractionLimits(
                max_members=10,
                max_total_bytes=512,
                max_file_bytes=2048,
                reserved_disk_bytes=0,
            ),
        )


def test_yolo_dataset_validation_checks_paths_and_class_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        image = dataset / "images" / split / "sample.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"fixture")
    (dataset / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: [target]\n",
        encoding="utf-8",
    )

    validate_yolo_dataset(dataset, ("target",))
    prepared = prepare_training_dataset(
        "yolo-detect", dataset, (), "yolo", tmp_path / "normalized"
    )
    assert prepared.root == dataset
    assert prepared.labels == ("target",)

    with pytest.raises(DatasetValidationError, match="exactly match"):
        validate_yolo_dataset(dataset, ("different",))

    (dataset / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: {0: target, 2: gap}\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="consecutive"):
        validate_yolo_dataset(dataset, ("target", "gap"))


def test_ppocr_dataset_validation_rejects_paths_outside_archive(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.txt").write_text("../outside.jpg\ttext\n", encoding="utf-8")
    (dataset / "val.txt").write_text("../outside.jpg\ttext\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="escapes"):
        validate_ppocr_dataset(dataset, "ppocr-rec")


def test_voc_detection_is_normalized_to_yolo(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset" / "VOCdevkit" / "VOC2007"
    (dataset / "Annotations").mkdir(parents=True)
    (dataset / "JPEGImages").mkdir()
    (dataset / "ImageSets" / "Main").mkdir(parents=True)
    for split in ("train", "val"):
        image_id = f"{split}_sample"
        (dataset / "JPEGImages" / f"{image_id}.jpg").write_bytes(b"fixture")
        (dataset / "ImageSets" / "Main" / f"{split}.txt").write_text(
            f"{image_id}\n", encoding="utf-8"
        )
        (dataset / "Annotations" / f"{image_id}.xml").write_text(
            f"""
<annotation>
  <filename>{image_id}.jpg</filename>
  <size><width>100</width><height>50</height></size>
  <object><name>target</name><bndbox>
    <xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>25</ymax>
  </bndbox></object>
</annotation>
""".strip(),
            encoding="utf-8",
        )

    prepared = prepare_training_dataset(
        "yolo-detect",
        tmp_path / "dataset",
        (),
        "voc_detection",
        tmp_path / "normalized",
    )

    assert prepared.root == tmp_path / "normalized"
    assert prepared.labels == ("target",)
    assert (prepared.root / "data.yaml").is_file()
    assert (prepared.root / "images" / "train" / "train_sample.jpg").is_file()
    assert (prepared.root / "labels" / "train" / "train_sample.txt").read_text() == (
        "0 0.20000000 0.30000000 0.20000000 0.40000000\n"
    )


def test_exporter_voc_detection_images_and_all_split_are_normalized(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "Annotations").mkdir(parents=True)
    (dataset / "images").mkdir()
    (dataset / "ImageSets" / "Main").mkdir(parents=True)
    image_ids = [f"sample_{index}" for index in range(5)]
    for image_id in image_ids:
        (dataset / "images" / f"{image_id}.png").write_bytes(b"fixture")
        (dataset / "Annotations" / f"{image_id}.xml").write_text(
            f"""
<annotation>
  <filename>{image_id}.png</filename>
  <size><width>100</width><height>50</height></size>
  <object><name>ng</name><bndbox>
    <xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>25</ymax>
  </bndbox></object>
</annotation>
""".strip(),
            encoding="utf-8",
        )
    (dataset / "ImageSets" / "Main" / "all.txt").write_text(
        "\n".join(image_ids) + "\n", encoding="utf-8"
    )

    prepared = prepare_training_dataset(
        "yolo-detect",
        dataset,
        (),
        "voc_detection",
        tmp_path / "normalized",
    )

    train_images = sorted((prepared.root / "images" / "train").glob("*.png"))
    val_images = sorted((prepared.root / "images" / "val").glob("*.png"))
    assert prepared.labels == ("ng",)
    assert len(train_images) == 4
    assert len(val_images) == 1
    assert {path.stem for path in train_images + val_images} == set(image_ids)
    assert not ({path.stem for path in train_images} & {path.stem for path in val_images})
    validate_yolo_dataset(prepared.root, prepared.labels)


def test_exporter_voc_detection_manifest_can_supply_image_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "Annotations").mkdir(parents=True)
    (dataset / "images").mkdir()
    (dataset / "ImageSets" / "Main").mkdir(parents=True)
    image_ids = ["sample_a", "sample_b"]
    for image_id in image_ids:
        (dataset / "images" / f"{image_id}.png").write_bytes(b"fixture")
        (dataset / "Annotations" / f"{image_id}.xml").write_text(
            f"""
<annotation>
  <filename>{image_id}.png</filename>
  <size><width>10</width><height>10</height></size>
  <object><name>target</name><bndbox>
    <xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax>
  </bndbox></object>
</annotation>
""".strip(),
            encoding="utf-8",
        )
    (dataset / "manifest.json").write_text(
        json.dumps({"images": [{"file": f"images/{image_id}.png"} for image_id in image_ids]}),
        encoding="utf-8",
    )

    prepared = prepare_training_dataset(
        "yolo-detect",
        dataset,
        (),
        "voc_detection",
        tmp_path / "normalized",
    )

    train_ids = {path.stem for path in (prepared.root / "images" / "train").iterdir()}
    val_ids = {path.stem for path in (prepared.root / "images" / "val").iterdir()}
    assert train_ids | val_ids == set(image_ids)
    assert len(train_ids) == len(val_ids) == 1


def test_coco_detection_is_normalized_to_yolo(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "annotations").mkdir(parents=True)
    (dataset / "images").mkdir()
    for index, split in enumerate(("train", "val"), start=1):
        (dataset / "images" / f"{split}.jpg").write_bytes(b"fixture")
        payload = {
            "images": [
                {"id": index, "file_name": f"images/{split}.jpg", "width": 100, "height": 50}
            ],
            "categories": [{"id": 4, "name": "target"}],
            "annotations": [
                {"id": index, "image_id": index, "category_id": 4, "bbox": [10, 5, 20, 20]}
            ],
        }
        (dataset / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    prepared = prepare_training_dataset(
        "yolo-detect",
        dataset,
        (),
        "coco_detection",
        tmp_path / "normalized",
    )

    assert prepared.labels == ("target",)
    assert (prepared.root / "images" / "val" / "2.jpg").is_file()
    assert (prepared.root / "labels" / "train" / "1.txt").read_text() == (
        "0 0.20000000 0.30000000 0.20000000 0.40000000\n"
    )


def test_voc_segmentation_is_normalized_to_mask_pairs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset" / "VOC2012"
    (dataset / "JPEGImages").mkdir(parents=True)
    (dataset / "SegmentationClass").mkdir()
    (dataset / "ImageSets" / "Segmentation").mkdir(parents=True)
    for split in ("train", "val"):
        image_id = f"{split}_sample"
        Image.new("RGB", (4, 4), color="white").save(
            dataset / "JPEGImages" / f"{image_id}.jpg"
        )
        Image.new("P", (4, 4), color=1).save(
            dataset / "SegmentationClass" / f"{image_id}.png"
        )
        (dataset / "ImageSets" / "Segmentation" / f"{split}.txt").write_text(
            f"{image_id}\n", encoding="utf-8"
        )

    prepared = prepare_training_dataset(
        "deeplabv3plus",
        tmp_path / "dataset",
        (),
        "voc_segmentation",
        tmp_path / "normalized",
    )

    assert prepared.labels == ("background", "class_1")
    assert (prepared.root / "images" / "train" / "train_sample.jpg").is_file()
    assert (prepared.root / "masks" / "val" / "val_sample.png").is_file()


def test_cvat_voc_segmentation_uses_default_split_and_rgb_labelmap(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "JPEGImages").mkdir(parents=True)
    (dataset / "SegmentationClass").mkdir()
    (dataset / "ImageSets" / "Segmentation").mkdir(parents=True)
    image_ids = [f"sample_{index}" for index in range(5)]
    for image_id in image_ids:
        Image.new("RGB", (4, 4), color="white").save(
            dataset / "JPEGImages" / f"{image_id}.jpg"
        )
        mask = Image.new("RGB", (4, 4), color=(0, 0, 0))
        mask.putpixel((1, 1), (48, 112, 32))
        mask.save(dataset / "SegmentationClass" / f"{image_id}.png")
    (dataset / "ImageSets" / "Segmentation" / "default.txt").write_text(
        "\n".join(image_ids) + "\n", encoding="utf-8"
    )
    label_map_path = dataset / "labelmap.txt"
    label_map_content = (
        "# label:color_rgb:parts:actions\n"
        "background:0,0,0::\n"
        "ng:48,112,32::\n"
    )
    label_map_path.write_text(label_map_content, encoding="utf-8")

    prepared = prepare_training_dataset(
        "deeplabv3plus",
        dataset,
        (),
        "voc_segmentation",
        tmp_path / "normalized",
    )

    assert prepared.labels == ("background", "ng")
    train_masks = sorted((prepared.root / "masks" / "train").glob("*.png"))
    val_masks = sorted((prepared.root / "masks" / "val").glob("*.png"))
    assert len(train_masks) == 4
    assert len(val_masks) == 1
    assert {path.stem for path in train_masks + val_masks} == set(image_ids)
    for path in train_masks + val_masks:
        with Image.open(path) as mask:
            assert mask.mode == "L"
            colors = mask.getcolors(maxcolors=3)
            assert colors is not None
            assert {color for _, color in colors} == {0, 1}

    label_map_path.unlink()
    with pytest.raises(DatasetValidationError, match=r"RGB masks require.*labelmap.txt"):
        prepare_training_dataset(
            "deeplabv3plus",
            dataset,
            (),
            "voc_segmentation",
            tmp_path / "normalized-missing-labelmap",
        )
    label_map_path.write_text(label_map_content, encoding="utf-8")

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(
        dataset / "SegmentationClass" / "sample_0.png"
    )
    with pytest.raises(DatasetValidationError, match="missing from labelmap"):
        prepare_training_dataset(
            "deeplabv3plus",
            dataset,
            (),
            "voc_segmentation",
            tmp_path / "normalized-invalid",
        )


def test_mask_pair_segmentation_preserves_palette_class_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset" / "nested"
    for split in ("train", "val"):
        image = dataset / "images" / split / "sample.jpg"
        mask = dataset / "masks" / split / "sample.png"
        image.parent.mkdir(parents=True)
        mask.parent.mkdir(parents=True)
        Image.new("RGB", (4, 4), color="white").save(image)
        Image.new("P", (4, 4), color=1).save(mask)

    prepared = prepare_training_dataset(
        "deeplabv3plus",
        tmp_path / "dataset",
        (),
        "mask_pairs",
        tmp_path / "normalized",
    )

    assert prepared.root == dataset
    assert prepared.labels == ("background", "class_1")

    (dataset / "classes.txt").write_text("background\ntarget\n", encoding="utf-8")
    named = prepare_training_dataset(
        "deeplabv3plus",
        tmp_path / "dataset",
        (),
        "mask_pairs",
        tmp_path / "normalized",
    )
    assert named.labels == ("background", "target")

    Image.new("L", (4, 4), color=2).save(dataset / "masks" / "train" / "sample.png")
    with pytest.raises(DatasetValidationError, match="class id"):
        prepare_training_dataset(
            "deeplabv3plus",
            tmp_path / "dataset",
            ("background", "target"),
            "mask_pairs",
            tmp_path / "normalized",
        )


def test_coco_segmentation_polygon_and_rle_are_normalized(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "annotations").mkdir(parents=True)
    (dataset / "images").mkdir()
    for index, split in enumerate(("train", "val"), start=1):
        Image.new("RGB", (4, 4), color="white").save(dataset / "images" / f"{split}.png")
        segmentation: object = (
            [[0, 0, 3, 0, 3, 3, 0, 3]]
            if split == "train"
            else {"size": [4, 4], "counts": [0, 2, 14]}
        )
        payload = {
            "images": [
                {"id": index, "file_name": f"images/{split}.png", "width": 4, "height": 4}
            ],
            "categories": [{"id": 7, "name": "target"}],
            "annotations": [
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 7,
                    "segmentation": segmentation,
                }
            ],
        }
        (dataset / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    prepared = prepare_training_dataset(
        "deeplabv3plus",
        dataset,
        (),
        "coco_segmentation",
        tmp_path / "normalized",
    )

    assert prepared.labels == ("background", "target")
    with Image.open(prepared.root / "masks" / "train" / "1.png") as train_mask:
        assert train_mask.getpixel((1, 1)) == 1
    with Image.open(prepared.root / "masks" / "val" / "2.png") as val_mask:
        assert val_mask.getpixel((0, 0)) == 1
        assert val_mask.getpixel((1, 0)) == 0


def test_explicit_dataset_format_must_match_training_profile(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="not valid for object detection"):
        prepare_training_dataset(
            "yolo-detect",
            tmp_path,
            ("target",),
            "mask_pairs",
            tmp_path / "normalized",
        )
