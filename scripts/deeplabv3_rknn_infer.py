from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

DEFAULT_LABELS = ("background", "ng")
OutputLayout = Literal["auto", "nchw", "nhwc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a DeepLabV3+ RKNN model on an RK3588 board."
    )
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=Path,
        required=True,
        help="Path to the .rknn model file.",
    )
    parser.add_argument(
        "--image-path",
        "--image_path",
        dest="image_path",
        type=Path,
        default=Path("../model/11.jpg"),
        help="Input image path. Default: ../model/11.jpg",
    )
    parser.add_argument("--target", default="rk3588", help="RKNPU target platform.")
    parser.add_argument(
        "--device-id",
        "--device_id",
        dest="device_id",
        default=None,
        help="Optional RKNN device ID.",
    )
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated labels in model output order.",
    )
    parser.add_argument(
        "--output-layout",
        choices=("auto", "nchw", "nhwc"),
        default="auto",
        help="RKNN output layout. Auto detects it from the label count.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Optional expected SHA-256 for the RKNN file.",
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        default=Path("output-mask.png"),
        help="Class-index mask output path.",
    )
    parser.add_argument(
        "--color-output",
        type=Path,
        default=Path("output-color.png"),
        help="Colored segmentation mask output path.",
    )
    parser.add_argument(
        "--overlay-output",
        type=Path,
        default=Path("output.png"),
        help="Image and segmentation overlay output path.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Segmentation opacity in the overlay, from 0 to 1.",
    )
    parser.add_argument("--show", action="store_true", help="Display results with matplotlib.")
    parser.add_argument("--verbose-rknn", action="store_true")
    return parser.parse_args()


def parse_labels(raw: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(labels) < 2:
        raise ValueError("DeepLabV3+ requires at least two labels")
    if len(labels) != len(set(labels)):
        raise ValueError("Labels must be unique")
    if len(labels) > 256:
        raise ValueError("This demo supports at most 256 labels")
    return labels


def require_file(path: Path, description: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{description} does not exist: {path} (working directory: {Path.cwd()})"
        ) from error
    if not resolved.is_file():
        raise ValueError(f"{description} is not a regular file: {resolved}")
    if resolved.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def logits_to_chw(
    output: np.ndarray,
    class_count: int,
    layout: OutputLayout = "auto",
) -> np.ndarray:
    value = np.asarray(output)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"Expected output batch size 1, got shape {list(value.shape)}")
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"Expected a 3D/4D logits tensor, got shape {list(value.shape)}")

    if layout == "nchw":
        if value.shape[0] != class_count:
            raise ValueError(
                f"NCHW channel count {value.shape[0]} does not match {class_count} labels"
            )
        return value
    if layout == "nhwc":
        if value.shape[-1] != class_count:
            raise ValueError(
                f"NHWC channel count {value.shape[-1]} does not match {class_count} labels"
            )
        return np.moveaxis(value, -1, 0)

    first_matches = value.shape[0] == class_count
    last_matches = value.shape[-1] == class_count
    if first_matches and not last_matches:
        return value
    if last_matches and not first_matches:
        return np.moveaxis(value, -1, 0)
    if first_matches and last_matches:
        raise ValueError(
            f"Output layout is ambiguous for shape {list(value.shape)}; "
            "set --output-layout explicitly"
        )
    raise ValueError(
        f"Cannot find a {class_count}-channel class axis in output shape {list(value.shape)}; "
        "check --labels and --output-layout"
    )


def resize_logits(logits: np.ndarray, height: int, width: int, cv2: Any) -> np.ndarray:
    if logits.shape[1:] == (height, width):
        return logits
    return np.stack(
        [
            cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
            for channel in logits
        ],
        axis=0,
    )


def classify_logits(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 3:
        raise ValueError(f"Expected CHW logits, got shape {list(logits.shape)}")
    return np.argmax(logits, axis=0).astype(np.uint8)


def pascal_colormap(size: int = 256) -> np.ndarray:
    colors = np.zeros((size, 3), dtype=np.uint8)
    for index in range(size):
        value = index
        for shift in range(8):
            colors[index, 0] |= ((value >> 0) & 1) << (7 - shift)
            colors[index, 1] |= ((value >> 1) & 1) << (7 - shift)
            colors[index, 2] |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
    return colors


def check_return_code(code: int, stage: str) -> None:
    if code != 0:
        raise RuntimeError(f"RKNN {stage} failed with return code {code}")


def write_rgb(path: Path, image: np.ndarray, cv2: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write image: {path}")


def print_class_summary(mask: np.ndarray, labels: Sequence[str]) -> None:
    values, counts = np.unique(mask, return_counts=True)
    total = int(mask.size)
    print("Detected classes:")
    for value, count in zip(values.tolist(), counts.tolist(), strict=True):
        label = labels[value] if value < len(labels) else f"class_{value}"
        print(f"  {value}: {label} - {count} pixels ({count / total:.2%})")


def show_results(source: np.ndarray, color_mask: np.ndarray, overlay: np.ndarray) -> None:
    try:
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise RuntimeError("--show requires matplotlib") from error
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, image, title in zip(
        axes,
        (source, color_mask, overlay),
        ("Input image", "Segmentation mask", "Overlay"),
        strict=True,
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def run(args: argparse.Namespace) -> None:
    try:
        import cv2
        from rknn.api import RKNN
    except ImportError as error:
        raise RuntimeError("This script requires opencv-python and rknn-toolkit2") from error

    if args.input_width <= 0 or args.input_height <= 0:
        raise ValueError("Input width and height must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    labels = parse_labels(args.labels)
    model_path = require_file(args.model_path, "RKNN model")
    image_path = require_file(args.image_path, "Input image")
    checksum = sha256_file(model_path)
    if args.expected_sha256 and checksum.lower() != args.expected_sha256.lower():
        raise ValueError(
            f"RKNN SHA-256 mismatch: expected {args.expected_sha256}, got {checksum}"
        )

    source_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise ValueError(f"OpenCV cannot decode input image: {image_path}")
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    original_height, original_width = source_rgb.shape[:2]
    model_input = cv2.resize(
        source_rgb,
        (args.input_width, args.input_height),
        interpolation=cv2.INTER_LINEAR,
    )

    print(f"Model: {model_path}")
    print(f"Model size: {model_path.stat().st_size} bytes")
    print(f"Model SHA-256: {checksum}")
    print(f"Image: {image_path}")
    print(f"Labels: {list(labels)}")

    rknn = RKNN(verbose=args.verbose_rknn)
    try:
        check_return_code(rknn.load_rknn(str(model_path)), "load_rknn")
        runtime_options: dict[str, str] = {"target": args.target}
        if args.device_id:
            runtime_options["device_id"] = args.device_id
        check_return_code(rknn.init_runtime(**runtime_options), "init_runtime")
        raw_outputs = rknn.inference(inputs=[model_input], data_format="nhwc")
        if not raw_outputs:
            raise RuntimeError("RKNN inference returned no outputs")
        output = np.asarray(raw_outputs[0])
    finally:
        rknn.release()

    print(f"Raw output shape: {list(output.shape)}")
    logits = logits_to_chw(output, len(labels), cast(OutputLayout, args.output_layout))
    logits = resize_logits(logits, original_height, original_width, cv2)
    mask = classify_logits(logits)
    color_mask = pascal_colormap()[mask]
    overlay = np.clip(
        source_rgb.astype(np.float32) * (1.0 - args.alpha)
        + color_mask.astype(np.float32) * args.alpha,
        0,
        255,
    ).astype(np.uint8)

    args.mask_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.mask_output), mask):
        raise RuntimeError(f"Failed to write mask: {args.mask_output}")
    write_rgb(args.color_output, color_mask, cv2)
    write_rgb(args.overlay_output, overlay, cv2)
    print_class_summary(mask, labels)
    print(f"Class-index mask: {args.mask_output.resolve()}")
    print(f"Color mask: {args.color_output.resolve()}")
    print(f"Overlay: {args.overlay_output.resolve()}")

    if args.show:
        show_results(source_rgb, color_mask, overlay)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
