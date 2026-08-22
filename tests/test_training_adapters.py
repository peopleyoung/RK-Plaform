from __future__ import annotations

from pathlib import Path

import pytest
from backend.platform_api.contracts import TrainingJobCreate
from backend.platform_api.profiles import ModelProfileRegistry
from workers.trainer.adapters import (
    AdapterEnvironment,
    DeepLabAdapter,
    PpocrAdapter,
    TrainingTask,
    YoloAdapter,
)
from workers.trainer.scripts.prepare_ppocr_config import resolve_config
from workers.trainer.scripts.select_ppocr_checkpoint import select_checkpoint
from workers.trainer.scripts.train_deeplab import _use_rknn_friendly_decoder_upsampling
from workers.trainer.scripts.yolo_ultralytics import resolve_model_source


def profiles() -> ModelProfileRegistry:
    return ModelProfileRegistry(Path(__file__).parents[1] / "config/model_profiles.json")


def make_task(
    tmp_path: Path,
    *,
    profile_id: str,
    variant: str,
    width: int,
    height: int,
    accelerator: str = "cpu",
    optimizer: str = "auto",
    learning_rate: float | None = None,
    seed: int = 42,
) -> TrainingTask:
    return TrainingTask(
        job_id="train_test",
        request=TrainingJobCreate.model_validate(
            {
                "name": "test",
                "datasetId": "ds_test",
                "profileId": profile_id,
                "variant": variant,
                "resolution": {"width": width, "height": height},
                "hyperparameters": {
                    "epochs": 2,
                    "batchSize": 1,
                    "pretrained": False,
                    "optimizer": optimizer,
                    "learningRate": learning_rate,
                    "seed": seed,
                },
                "accelerator": accelerator,
            }
        ),
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        labels=("background", "target"),
    )


def framework_environment(tmp_path: Path, names: dict[str, str]) -> AdapterEnvironment:
    roots: dict[str, Path] = {}
    for name, entrypoint in names.items():
        root = tmp_path / name
        path = root / entrypoint
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
        roots[name] = root
    return AdapterEnvironment(
        python="/usr/bin/python3",
        project_root=Path(__file__).parents[1],
        framework_roots=roots,
    )


def test_ultralytics_pretrained_weights_must_be_installed_locally(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    weight = weights / "yolov8n.pt"
    weight.write_bytes(b"fixture")

    assert resolve_model_source("yolov8n", False, weights) == str(weight)
    assert resolve_model_source("yolov8n", True, weights) == "yolov8n.yaml"

    weight.unlink()
    with pytest.raises(FileNotFoundError, match="Pretrained weights are not installed"):
        resolve_model_source("yolov8n", False, weights)


def test_yolov8_plan_uses_rockchip_checkout_and_static_dimensions(tmp_path: Path) -> None:
    data_yaml = tmp_path / "dataset" / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("path: .\n", encoding="utf-8")
    environment = framework_environment(tmp_path, {"yolov8": "ultralytics/__init__.py"})
    task = make_task(
        tmp_path,
        profile_id="yolo-detect",
        variant="yolov8n",
        width=640,
        height=384,
    )

    plan = YoloAdapter(profiles(), environment).plan(task)

    command = plan.steps[0].argv
    assert command[0] == "/usr/bin/python3"
    assert command[command.index("--width") + 1] == "640"
    assert command[command.index("--height") + 1] == "384"
    assert command[command.index("--opset") + 1] == "12"
    assert command[command.index("--export-format") + 1] == "rknn"
    assert "--no-pretrained" in command
    assert plan.onnx_source == tmp_path / "output/model.onnx"


def test_yolov10_plan_uses_generation_specific_opset(tmp_path: Path) -> None:
    data_yaml = tmp_path / "dataset" / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("path: .\n", encoding="utf-8")
    environment = framework_environment(tmp_path, {"yolov10": "ultralytics/__init__.py"})
    task = make_task(
        tmp_path,
        profile_id="yolo-detect",
        variant="yolov10n",
        width=960,
        height=544,
    )

    command = YoloAdapter(profiles(), environment).plan(task).steps[0].argv

    assert command[command.index("--opset") + 1] == "13"
    assert command[command.index("--export-format") + 1] == "rknn"


def test_deeplab_plan_keeps_width_and_height_independent(tmp_path: Path) -> None:
    task = make_task(
        tmp_path,
        profile_id="deeplabv3plus",
        variant="resnet50",
        width=768,
        height=512,
        accelerator="cuda",
        optimizer="SGD",
        learning_rate=0.02,
        seed=7,
    )
    environment = framework_environment(tmp_path, {})

    command = DeepLabAdapter(profiles(), environment).plan(task).steps[0].argv

    assert command[command.index("--width") + 1] == "768"
    assert command[command.index("--height") + 1] == "512"
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--optimizer") + 1] == "SGD"
    assert command[command.index("--learning-rate") + 1] == "0.02"
    assert command[command.index("--seed") + 1] == "7"
    assert "--no-pretrained" in command


def test_deeplab_rknn_variant_is_forwarded_to_training_script(tmp_path: Path) -> None:
    task = make_task(
        tmp_path,
        profile_id="deeplabv3plus",
        variant="mobilenet_v2_rknn",
        width=512,
        height=512,
    )
    environment = framework_environment(tmp_path, {})

    command = DeepLabAdapter(profiles(), environment).plan(task).steps[0].argv

    assert command[command.index("--variant") + 1] == "mobilenet_v2_rknn"


def test_deeplab_replaces_legacy_align_corners_decoder_resize() -> None:
    class Decoder:
        up = type("LegacyUpsample", (), {"scale_factor": 4.0})()

    class Model:
        decoder = Decoder()

    class NN:
        @staticmethod
        def Upsample(**values: object) -> dict[str, object]:
            return values

    class Torch:
        nn = NN()

    model = Model()
    _use_rknn_friendly_decoder_upsampling(Torch(), model)

    assert model.decoder.up == {
        "scale_factor": 4.0,
        "mode": "bilinear",
        "align_corners": False,
    }


def test_yolov5_scratch_plan_includes_model_config(tmp_path: Path) -> None:
    data_yaml = tmp_path / "dataset" / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("path: .\n", encoding="utf-8")
    environment = framework_environment(tmp_path, {"yolov5": "train.py"})
    (environment.framework_roots["yolov5"] / "export.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    task = make_task(
        tmp_path,
        profile_id="yolo-detect",
        variant="yolov5s",
        width=640,
        height=384,
    )

    command = YoloAdapter(profiles(), environment).plan(task).steps[0].argv

    assert "--weights=" in command
    assert command[command.index("--cfg") + 1].endswith("models/yolov5s.yaml")


def test_ppocr_plan_forces_static_paddle_onnx_shape(tmp_path: Path, monkeypatch: object) -> None:
    root_entrypoints = {
        "paddleocr": "tools/train.py",
    }
    environment = framework_environment(tmp_path, root_entrypoints)
    paddle_root = environment.framework_roots["paddleocr"]
    export_script = paddle_root / "tools/export_model.py"
    export_script.write_text("# fixture\n", encoding="utf-8")
    config = paddle_root / "det.yml"
    config.write_text("Architecture: {}\n", encoding="utf-8")
    monkeypatch.setenv("RKNODE_PPOCRV4_DET_CONFIG", str(config))  # type: ignore[attr-defined]
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    (dataset / "train.txt").write_text("image.jpg\t[]\n", encoding="utf-8")
    (dataset / "val.txt").write_text("image.jpg\t[]\n", encoding="utf-8")
    task = make_task(
        tmp_path,
        profile_id="ppocr-det",
        variant="ppocrv4_det",
        width=640,
        height=384,
    )

    plan = PpocrAdapter(profiles(), environment, "ppocr-det").plan(task)

    optimize = plan.steps[-1].argv
    convert = plan.steps[-2].argv
    assert convert[convert.index("--opset_version") + 1] == "14"
    assert optimize[:3] == ("/usr/bin/python3", "-m", "paddle2onnx.optimize")
    assert optimize[optimize.index("--input_shape_dict") + 1] == '{"x":[1,3,384,640]}'


def test_ppocr_config_uses_custom_shape_for_training_and_export(tmp_path: Path) -> None:
    source = tmp_path / "ppocr.yml"
    source.write_text(
        """
Global:
  pretrained_model: https://example.test/model.pdparams
Train:
  dataset:
    transforms:
      - EastRandomCropData:
          size: [640, 640]
      - RecConAug:
          image_shape: [48, 320, 3]
  sampler:
    scales: [[320, 48], [320, 64]]
  loader:
    batch_size_per_card: 8
Eval:
  dataset:
    transforms:
      - RecResizeImg:
          image_shape: [3, 48, 320]
  loader:
    batch_size_per_card: 8
""",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    config = resolve_config(
        source,
        profile_id="ppocr-rec",
        dataset=dataset,
        train_label=dataset / "train.txt",
        val_label=dataset / "val.txt",
        save_dir=tmp_path / "output",
        width=640,
        height=64,
        epochs=3,
        batch_size=2,
        device="cpu",
        pretrained=False,
        optimizer="SGD",
        learning_rate=0.005,
        seed=17,
    )

    assert config["Global"]["d2s_train_image_shape"] == [3, 64, 640]
    assert config["Global"]["pretrained_model"] is None
    assert config["Global"]["seed"] == 17
    assert config["Optimizer"]["name"] == "Momentum"
    assert config["Optimizer"]["lr"]["learning_rate"] == 0.005
    assert config["Train"]["sampler"]["scales"] == [[640, 64]]
    transforms = config["Train"]["dataset"]["transforms"]
    assert transforms[0]["EastRandomCropData"]["size"] == [640, 64]
    assert transforms[1]["RecConAug"]["image_shape"] == [64, 640, 3]
    eval_resize = config["Eval"]["dataset"]["transforms"][0]["RecResizeImg"]
    assert eval_resize["image_shape"] == [3, 64, 640]


def test_ppocr_config_does_not_create_an_empty_sampler(tmp_path: Path) -> None:
    source = tmp_path / "ppocr.yml"
    source.write_text(
        "Global: {}\nOptimizer:\n  name: Adam\n  lr:\n    name: Const\n"
        "Train:\n  dataset: {}\n  loader: {}\n"
        "Eval:\n  dataset: {}\n  loader: {}\n",
        encoding="utf-8",
    )

    config = resolve_config(
        source,
        profile_id="ppocr-rec",
        dataset=tmp_path / "dataset",
        train_label=tmp_path / "dataset/train.txt",
        val_label=tmp_path / "dataset/val.txt",
        save_dir=tmp_path / "output",
        width=128,
        height=32,
        epochs=1,
        batch_size=1,
        device="cpu",
        pretrained=False,
    )

    assert "sampler" not in config["Train"]


def test_ppocr_checkpoint_selection_falls_back_to_latest(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "latest.pdparams").write_bytes(b"latest")
    output_prefix = tmp_path / "selected"

    selected = select_checkpoint(train_dir, output_prefix)

    assert selected == tmp_path / "selected.pdparams"
    assert selected.read_bytes() == b"latest"
