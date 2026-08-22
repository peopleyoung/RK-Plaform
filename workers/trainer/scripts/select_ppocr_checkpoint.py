from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a PPOCR export checkpoint")
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    return parser.parse_args()


def select_checkpoint(train_dir: Path, output_prefix: Path) -> Path:
    source_prefix = next(
        (
            candidate
            for candidate in (train_dir / "best_accuracy", train_dir / "latest")
            if candidate.with_suffix(".pdparams").is_file()
        ),
        None,
    )
    if source_prefix is None:
        raise FileNotFoundError(
            f"PPOCR training produced neither best_accuracy nor latest in {train_dir}"
        )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdparams", ".pdopt", ".states"):
        source = source_prefix.with_suffix(suffix)
        if source.is_file():
            shutil.copy2(source, output_prefix.with_suffix(suffix))
    return output_prefix.with_suffix(".pdparams")


def main() -> None:
    args = parse_args()
    selected = select_checkpoint(args.train_dir, args.output_prefix)
    print(f"selected_checkpoint={selected}", flush=True)


if __name__ == "__main__":
    main()
