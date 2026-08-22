from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable


@runtime_checkable
class SupportsFloat(Protocol):
    def __float__(self) -> float: ...


def finite_float(value: object) -> float | None:
    if not isinstance(value, SupportsFloat):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--opset", required=True, type=int)
    parser.add_argument("--export-format", choices=("onnx", "rknn"), required=True)
    parser.add_argument("--optimizer", choices=("auto", "AdamW", "SGD"), default="auto")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=Path(os.getenv("RKNODE_PRETRAINED_WEIGHTS_ROOT", "/opt/weights")),
    )
    return parser.parse_args()


def resolve_model_source(variant: str, no_pretrained: bool, weights_root: Path) -> str:
    if no_pretrained:
        return f"{variant}.yaml"
    local_weight = weights_root / f"{variant}.pt"
    if local_weight.is_file():
        return str(local_weight)
    raise FileNotFoundError(
        f"Pretrained weights are not installed for '{variant}': {local_weight}. "
        "Install the weight in the trainer image or disable pretrained training."
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    ultralytics = cast(Any, importlib.import_module("ultralytics"))
    yolo_class: Any = ultralytics.YOLO

    model_source = resolve_model_source(args.variant, args.no_pretrained, args.weights_root)
    model: Any = yolo_class(model_source)

    def emit_epoch_metrics(trainer: Any) -> None:
        payload: dict[str, float | int] = {
            "epoch": int(trainer.epoch) + 1,
            "epochs": int(trainer.epochs),
        }
        raw_metrics = getattr(trainer, "metrics", {})
        if isinstance(raw_metrics, dict):
            for name, raw_value in cast(dict[object, object], raw_metrics).items():
                if (value := finite_float(raw_value)) is not None:
                    payload[str(name)] = value
        loss_names = getattr(trainer, "loss_names", ())
        loss_values = getattr(trainer, "tloss", ())
        try:
            for name, raw_value in zip(loss_names, loss_values, strict=False):
                if (value := finite_float(raw_value)) is not None:
                    payload[f"train/{name}"] = value
        except TypeError:
            pass
        print(f"RKNODE_METRIC {json.dumps(payload, sort_keys=True)}", flush=True)

    if hasattr(model, "add_callback"):
        model.add_callback("on_fit_epoch_end", emit_epoch_metrics)
    run_root = args.output_checkpoint.parent / "ultralytics"
    train_arguments: dict[str, Any] = {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": max(args.width, args.height),
        "device": args.device,
        "project": str(run_root),
        "name": "train",
        "exist_ok": True,
        "optimizer": args.optimizer,
        "seed": args.seed,
    }
    if args.learning_rate is not None:
        train_arguments["lr0"] = args.learning_rate
    model.train(
        **train_arguments,
    )
    trainer: Any = model.trainer
    best = Path(cast(str, trainer.best))
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output_checkpoint)

    trained: Any = yolo_class(str(args.output_checkpoint))
    torch = cast(Any, importlib.import_module("torch"))
    original_onnx_export = torch.onnx.export

    def legacy_onnx_export(*export_args: Any, **export_kwargs: Any) -> Any:
        export_kwargs.setdefault("dynamo", False)
        return original_onnx_export(*export_args, **export_kwargs)

    torch.onnx.export = legacy_onnx_export
    try:
        exported: Any = trained.export(
            format=args.export_format,
            imgsz=[args.height, args.width],
            batch=1,
            dynamic=False,
            opset=args.opset,
            simplify=False,
            device=args.device,
        )
    finally:
        torch.onnx.export = original_onnx_export
    exported_path = Path(str(exported))
    if exported_path.is_dir():
        matches = list(exported_path.rglob("*.onnx"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one ONNX export in {exported_path}, found {len(matches)}")
        exported_path = matches[0]
    if exported_path.suffix != ".onnx" or not exported_path.is_file():
        raise RuntimeError(f"Rockchip exporter did not produce an ONNX file: {exported_path}")
    shutil.copy2(exported_path, args.output_model)


if __name__ == "__main__":
    main()
