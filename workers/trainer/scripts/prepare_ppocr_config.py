from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve a PPOCR training config")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-id", choices=("ppocr-det", "ppocr-rec"), required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--train-label", required=True, type=Path)
    parser.add_argument("--val-label", required=True, type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--optimizer", choices=("auto", "AdamW", "SGD"), default="auto")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def resolve_config(
    source: Path,
    *,
    profile_id: str,
    dataset: Path,
    train_label: Path,
    val_label: Path,
    save_dir: Path,
    width: int,
    height: int,
    epochs: int,
    batch_size: int,
    device: str,
    pretrained: bool,
    optimizer: str = "auto",
    learning_rate: float | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("PPOCR config root must be an object")
    config = cast(dict[str, Any], raw)
    global_config = _section(config, "Global")
    global_config.update(
        {
            "use_gpu": device == "cuda",
            "distributed": False,
            "epoch_num": epochs,
            "save_model_dir": str(save_dir),
            "d2s_train_image_shape": [3, height, width],
            "seed": seed,
        }
    )
    if not pretrained:
        global_config["pretrained_model"] = None

    for split, label in (("Train", train_label), ("Eval", val_label)):
        section = _section(config, split)
        dataset_config = _section(section, "dataset")
        dataset_config["data_dir"] = str(dataset)
        dataset_config["label_file_list"] = [str(label)]
        loader = _section(section, "loader")
        loader["batch_size_per_card"] = batch_size

    if profile_id == "ppocr-rec":
        train_config = _section(config, "Train")
        raw_sampler: object = train_config.get("sampler")
        if raw_sampler is not None:
            if not isinstance(raw_sampler, dict):
                raise ValueError("PPOCR Train.sampler must be an object")
            sampler = cast(dict[str, Any], raw_sampler)
            sampler["scales"] = [[width, height]]
            sampler["first_bs"] = batch_size

    optimizer_config = _section(config, "Optimizer")
    if optimizer == "AdamW":
        optimizer_config["name"] = "AdamW"
    elif optimizer == "SGD":
        optimizer_config["name"] = "Momentum"
        optimizer_config.setdefault("momentum", 0.9)
    if learning_rate is not None:
        learning_rate_config = _section(optimizer_config, "lr")
        learning_rate_config["learning_rate"] = learning_rate

    _patch_shapes(config, profile_id, width, height)
    return config


def _patch_shapes(value: object, profile_id: str, width: int, height: int) -> None:
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        for key, nested in mapping.items():
            if key == "EastRandomCropData" and isinstance(nested, dict):
                cast(dict[str, Any], nested)["size"] = [width, height]
            elif key == "DetResizeForTest":
                if nested is None:
                    mapping[key] = {"image_shape": [height, width]}
                elif isinstance(nested, dict):
                    nested_mapping = cast(dict[str, Any], nested)
                    nested_mapping["image_shape"] = [height, width]
                    nested_mapping.pop("limit_side_len", None)
                    nested_mapping.pop("limit_type", None)
            elif profile_id == "ppocr-rec" and key == "RecConAug" and isinstance(nested, dict):
                cast(dict[str, Any], nested)["image_shape"] = [height, width, 3]
            elif profile_id == "ppocr-rec" and key.endswith("RecResizeImg"):
                if nested is None:
                    mapping[key] = {"image_shape": [3, height, width]}
                elif isinstance(nested, dict):
                    cast(dict[str, Any], nested)["image_shape"] = [3, height, width]
            _patch_shapes(cast(object, nested), profile_id, width, height)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _patch_shapes(nested, profile_id, width, height)


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        value = {}
        config[key] = value
    if not isinstance(value, dict):
        raise ValueError(f"PPOCR config section '{key}' must be an object")
    return cast(dict[str, Any], value)


def main() -> None:
    args = parse_args()
    config = resolve_config(
        args.source,
        profile_id=args.profile_id,
        dataset=args.dataset,
        train_label=args.train_label,
        val_label=args.val_label,
        save_dir=args.save_dir,
        width=args.width,
        height=args.height,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        pretrained=not args.no_pretrained,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
