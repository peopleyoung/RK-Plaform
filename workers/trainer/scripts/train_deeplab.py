from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, cast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--variant",
        choices=("mobilenet_v2_rknn", "mobilenet_v2", "resnet50"),
        required=True,
    )
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("auto", "AdamW", "SGD"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


class SegmentationDataset:
    def __init__(
        self,
        root: Path,
        split: str,
        width: int,
        height: int,
        *,
        image_module: Any,
        numpy_module: Any,
        torch_module: Any,
    ) -> None:
        self.images_dir = root / "images" / split
        self.masks_dir = root / "masks" / split
        self.width = width
        self.height = height
        self.image_module = image_module
        self.numpy = numpy_module
        self.torch = torch_module
        self.images = sorted(
            path
            for path in self.images_dir.rglob("*")
            if path.suffix.lower()
            in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
        )
        if not self.images:
            raise ValueError(f"No images found in {self.images_dir}")
        for image in self.images:
            if self._mask_path(image) is None:
                raise ValueError(f"No mask with matching stem for {image}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image_path = self.images[index]
        mask_path = self._mask_path(image_path)
        if mask_path is None:
            raise ValueError(f"No mask for {image_path}")
        image = self.image_module.open(image_path).convert("RGB")
        mask = self.image_module.open(mask_path)
        if mask.mode not in {"L", "P", "I", "I;16", "I;16B", "I;16L"}:
            raise ValueError(f"Segmentation mask must contain class indexes: {mask_path}")
        resampling = self.image_module.Resampling
        image = image.resize((self.width, self.height), resampling.BILINEAR)
        mask = mask.resize((self.width, self.height), resampling.NEAREST)
        image_array = self.numpy.asarray(image, dtype=self.numpy.float32)
        image_array = (image_array - 127.5) / 127.5
        image_tensor = self.torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_array = self.numpy.asarray(mask, dtype=self.numpy.int64)
        mask_tensor = self.torch.from_numpy(mask_array.copy())
        return image_tensor, mask_tensor

    def _mask_path(self, image: Path) -> Path | None:
        relative = image.relative_to(self.images_dir)
        for suffix in (".png", ".bmp", ".tif", ".tiff"):
            candidate = (self.masks_dir / relative).with_suffix(suffix)
            if candidate.is_file():
                return candidate
        return None


def main() -> None:
    args = parse_args()
    if args.classes < 2:
        raise ValueError("DeepLabV3+ requires at least two segmentation classes")
    try:
        torch = cast(Any, importlib.import_module("torch"))
        numpy = cast(Any, importlib.import_module("numpy"))
        image_module = cast(Any, importlib.import_module("PIL.Image"))
        smp = cast(Any, importlib.import_module("segmentation_models_pytorch"))
    except ImportError as error:
        raise RuntimeError(
            "DeepLabV3+ training requires torch, Pillow, numpy and segmentation-models-pytorch"
        ) from error

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    train_dataset = SegmentationDataset(
        args.dataset,
        "train",
        args.width,
        args.height,
        image_module=image_module,
        numpy_module=numpy,
        torch_module=torch,
    )
    val_root = args.dataset / "images" / "val"
    val_dataset = (
        SegmentationDataset(
            args.dataset,
            "val",
            args.width,
            args.height,
            image_module=image_module,
            numpy_module=numpy,
            torch_module=torch,
        )
        if val_root.is_dir()
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = (
        torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
        if val_dataset is not None
        else None
    )
    model = _build_model(
        torch,
        smp,
        variant=args.variant,
        classes=args.classes,
        width=args.width,
        height=args.height,
        pretrained=not args.no_pretrained,
    ).to(device)
    freeze_batch_norm = min(args.batch_size, len(train_dataset)) == 1
    optimizer = (
        torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        if args.optimizer == "SGD"
        else torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = float("inf")
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        if freeze_batch_norm:
            _freeze_batch_norm(torch, model)
        train_loss, _, _ = _run_epoch(
            torch, model, train_loader, criterion, optimizer, device, args.classes
        )
        model.eval()
        with torch.no_grad():
            validation = (
                _run_epoch(
                    torch,
                    model,
                    val_loader,
                    criterion,
                    None,
                    device,
                    args.classes,
                    collect_scores=True,
                )
                if val_loader is not None
                else (train_loss, None, None)
            )
        validation_loss, pixel_accuracy, mean_iou = validation
        score_text = (
            f" pixel_accuracy={pixel_accuracy:.6f} mean_iou={mean_iou:.6f}"
            if pixel_accuracy is not None and mean_iou is not None
            else ""
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f}{score_text}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model.state_dict(), args.output_checkpoint)

    model.load_state_dict(
        torch.load(args.output_checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    model.to("cpu")
    sample = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        args.output_model,
        input_names=["images"],
        output_names=["logits"],
        opset_version=12,
        dynamic_axes=None,
        do_constant_folding=True,
        dynamo=False,
    )


def _run_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any | None,
    device: Any,
    classes: int,
    *,
    collect_scores: bool = False,
) -> tuple[float, float | None, float | None]:
    total = 0.0
    batches = 0
    correct_pixels = 0
    total_pixels = 0
    intersections = [0] * classes
    unions = [0] * classes
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        deployment_logits = model(images)
        logits = torch.nn.functional.interpolate(
            deployment_logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = criterion(logits, masks)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu())
        batches += 1
        if collect_scores:
            predictions = logits.argmax(dim=1)
            correct_pixels += int((predictions == masks).sum().detach().cpu())
            total_pixels += int(masks.numel())
            for class_id in range(classes):
                predicted = predictions == class_id
                expected = masks == class_id
                intersections[class_id] += int((predicted & expected).sum().detach().cpu())
                unions[class_id] += int((predicted | expected).sum().detach().cpu())
    if batches == 0:
        raise ValueError("Dataset loader produced no batches")
    if not collect_scores:
        return total / batches, None, None
    valid_ious = [
        intersections[index] / unions[index] for index in range(classes) if unions[index]
    ]
    pixel_accuracy = correct_pixels / total_pixels if total_pixels else 0.0
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    return total / batches, pixel_accuracy, mean_iou


def _freeze_batch_norm(torch: Any, model: Any) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _build_model(
    torch: Any,
    smp: Any,
    *,
    variant: str,
    classes: int,
    width: int,
    height: int,
    pretrained: bool,
) -> Any:
    if variant == "mobilenet_v2_rknn":
        return _build_official_rknn_mobilenet_v2(
            torch,
            smp,
            classes=classes,
            width=width,
            height=height,
            pretrained=pretrained,
        )

    model = smp.DeepLabV3Plus(
        encoder_name=variant,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=classes,
        # Keep deployment logits at 1/4 resolution. Full-size interpolation is
        # training/post-processing work and must not become a CPU RKNN operator.
        upsampling=1,
    )
    _use_rknn_friendly_decoder_upsampling(torch, model)
    return model


def _build_official_rknn_mobilenet_v2(
    torch: Any,
    smp: Any,
    *,
    classes: int,
    width: int,
    height: int,
    pretrained: bool,
) -> Any:
    """Build the lightweight graph exported by Rockchip's TensorFlow example."""

    feature_height = (height + 7) // 8
    feature_width = (width + 7) // 8
    model = smp.DeepLabV3(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        encoder_output_stride=8,
        decoder_channels=256,
        in_channels=3,
        classes=classes,
        upsampling=1,
    )
    _remove_mobilenet_v2_classification_projection(model)

    class OfficialRknnDecoder(torch.nn.Module):
        def __init__(self, input_channels: int) -> None:
            torch.nn.Module.__init__(self)
            self.image_pool = torch.nn.AvgPool2d(
                kernel_size=(feature_height, feature_width),
                stride=(feature_height, feature_width),
            )
            self.image_projection = self._projection(input_channels, 256)
            self.aspp_projection = self._projection(input_channels, 256)
            self.concat_projection = self._projection(512, 256)

        @staticmethod
        def _projection(input_channels: int, output_channels: int) -> Any:
            return torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU(),
            )

        def forward(self, features: list[Any]) -> Any:
            encoder_output = features[-1]
            image_feature = self.image_projection(self.image_pool(encoder_output))
            image_feature = torch.nn.functional.interpolate(
                image_feature,
                scale_factor=(feature_height, feature_width),
                mode="bilinear",
                align_corners=True,
            )
            aspp_feature = self.aspp_projection(encoder_output)
            return self.concat_projection(torch.cat((image_feature, aspp_feature), dim=1))

    decoder = OfficialRknnDecoder(model.encoder.out_channels[-1])
    smp.base.initialization.initialize_decoder(decoder)
    model.decoder = decoder
    return model


def _remove_mobilenet_v2_classification_projection(model: Any) -> None:
    encoder = model.encoder
    features = getattr(encoder, "features", None)
    out_indexes = getattr(encoder, "_out_indexes", None)
    out_channels = getattr(encoder, "_out_channels", None)
    if (
        features is None
        or len(features) != 19
        or not isinstance(out_indexes, list)
        or out_indexes[-1] != 18
        or not isinstance(out_channels, list)
        or out_channels[-1] != 1280
    ):
        raise RuntimeError("Unsupported segmentation-models-pytorch MobileNetV2 encoder layout")

    final_conv = features[-1][0]
    if getattr(final_conv, "in_channels", None) != 320:
        raise RuntimeError("MobileNetV2 classification projection does not consume 320 channels")

    encoder.features = features[:-1]
    encoder._out_indexes = [*out_indexes[:-1], len(encoder.features) - 1]
    encoder._out_channels = [*out_channels[:-1], 320]


def _use_rknn_friendly_decoder_upsampling(torch: Any, model: Any) -> None:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise RuntimeError("DeepLabV3+ model does not expose a decoder")
    current = getattr(decoder, "up", None)
    scale_factor = getattr(current, "scale_factor", None)
    if scale_factor is None:
        raise RuntimeError("DeepLabV3+ decoder does not expose a fixed upsampling scale")
    decoder.up = torch.nn.Upsample(
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
    )


if __name__ == "__main__":
    main()
