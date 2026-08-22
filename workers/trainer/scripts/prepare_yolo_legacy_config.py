from __future__ import annotations

import argparse
import pprint
import runpy
from pathlib import Path
from typing import Any, cast

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve legacy YOLO hyperparameters")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("yaml", "python"), required=True)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--optimizer", choices=("auto", "SGD"), default="auto")
    return parser.parse_args()


def resolve_yaml(source: Path, learning_rate: float | None) -> dict[str, Any]:
    raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("YOLO hyperparameter config root must be an object")
    config = cast(dict[str, Any], raw)
    if learning_rate is not None:
        config["lr0"] = learning_rate
    return config


def resolve_python(
    source: Path,
    learning_rate: float | None,
    optimizer: str,
) -> str:
    namespace = runpy.run_path(str(source))
    sections: dict[str, dict[str, Any]] = {}
    for name in ("model", "solver", "data_aug"):
        value = namespace.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"YOLOv6 config must define a '{name}' dictionary")
        sections[name] = cast(dict[str, Any], value).copy()
    if learning_rate is not None:
        sections["solver"]["lr0"] = learning_rate
    if optimizer != "auto":
        sections["solver"]["optim"] = optimizer
    return (
        "\n\n".join(
            f"{name} = {pprint.pformat(value, sort_dicts=False)}"
            for name, value in sections.items()
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "yaml":
        config = resolve_yaml(args.source, args.learning_rate)
        args.output.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    else:
        args.output.write_text(
            resolve_python(args.source, args.learning_rate, args.optimizer), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
