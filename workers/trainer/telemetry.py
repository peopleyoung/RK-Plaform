from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Literal, cast

ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER_PATTERN = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
KEY_VALUE_PATTERN = re.compile(
    rf"(?P<key>[A-Za-z][A-Za-z0-9_./()@-]{{0,79}})\s*[:=]\s*(?P<value>{NUMBER_PATTERN})"
)
EPOCH_PATTERN = re.compile(
    r"epoch\s*[:=]?\s*\[?\s*(?P<current>\d+)\s*/\s*(?P<total>\d+)",
    re.IGNORECASE,
)
FRACTION_PATTERN = re.compile(r"^(?P<current>\d+)\s*/\s*(?P<total>\d+)$")
STRUCTURED_PREFIX = "RKNODE_METRIC "

METRIC_ALIASES = {
    "p": "precision",
    "r": "recall",
    "acc": "accuracy",
    "avg_acc": "accuracy",
    "pix_acc": "pixel_accuracy",
    "miou": "mean_iou",
    "map_50": "map50",
    "map_50_95": "map50_95",
    "map50_95": "map50_95",
    "map50-95": "map50_95",
}
METRIC_MARKERS = (
    "loss",
    "map",
    "precision",
    "recall",
    "acc",
    "iou",
    "hmean",
    "edit",
    "f1",
    "lr",
)


@dataclass(frozen=True)
class ParsedMetrics:
    metrics: dict[str, float]
    step: int | None = None
    epoch: int | None = None
    total_epochs: int | None = None


@dataclass(frozen=True)
class TelemetryEntry:
    type: Literal["log", "metric"]
    stage: str
    message: str
    metrics: dict[str, float] = field(default_factory=lambda: {})
    step: int | None = None
    epoch: int | None = None
    total_epochs: int | None = None
    level: Literal["debug", "info", "warning", "error"] = "info"

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": self.type,
            "level": self.level,
            "message": self.message,
            "stage": self.stage,
            "metrics": self.metrics,
        }
        if self.step is not None:
            payload["step"] = self.step
        if self.epoch is not None:
            payload["epoch"] = self.epoch
        if self.total_epochs is not None:
            payload["totalEpochs"] = self.total_epochs
        return payload


class TrainingMetricParser:
    def __init__(self) -> None:
        self._training_headers: tuple[str, ...] = ()
        self._validation_headers: tuple[str, ...] = ()
        self._last_epoch: int | None = None
        self._last_total_epochs: int | None = None
        self._zero_based_epochs: bool | None = None

    def parse(self, line: str) -> ParsedMetrics | None:
        cleaned = ANSI_PATTERN.sub("", line).replace("\r", "").strip()
        if not cleaned:
            return None
        structured = self._structured(cleaned)
        if structured is not None:
            self._remember_epoch(structured)
            return structured

        tokens = cleaned.replace(",", " ").split()
        if self._capture_headers(tokens):
            return None

        tabular = self._tabular(tokens)
        keyed = self._keyed(cleaned)
        parsed = _merge_metrics(tabular, keyed)
        if parsed is not None:
            self._remember_epoch(parsed)
        return parsed

    def _structured(self, line: str) -> ParsedMetrics | None:
        marker = line.find(STRUCTURED_PREFIX)
        if marker < 0:
            return None
        try:
            raw = json.loads(line[marker + len(STRUCTURED_PREFIX) :])
        except json.JSONDecodeError:
            return None
        if not isinstance(raw, dict):
            return None
        return _from_mapping(cast(dict[str, object], raw))

    def _capture_headers(self, tokens: list[str]) -> bool:
        normalized = [_normalize_token(token) for token in tokens]
        metric_count = sum(_canonical_metric_name(token) is not None for token in tokens)
        keyed_line = any("=" in token or ":" in token for token in tokens)
        if "epoch" in normalized and metric_count >= 2 and not keyed_line:
            index = normalized.index("epoch")
            self._training_headers = tuple(tokens[index:])
            return True
        if "class" in normalized and metric_count >= 2:
            index = normalized.index("class")
            self._validation_headers = tuple(tokens[index:])
            return True
        return False

    def _tabular(self, tokens: list[str]) -> ParsedMetrics | None:
        for index, token in enumerate(tokens):
            match = FRACTION_PATTERN.fullmatch(token)
            if match and self._training_headers:
                current = int(match.group("current"))
                total = int(match.group("total"))
                if self._zero_based_epochs is None:
                    self._zero_based_epochs = current == 0
                epoch = current + 1 if self._zero_based_epochs else current
                total_epochs = total + 1 if self._zero_based_epochs else total
                metrics = _metrics_from_columns(self._training_headers, tokens[index:])
                return ParsedMetrics(metrics, epoch=epoch, total_epochs=total_epochs)

        normalized = [_normalize_token(token) for token in tokens]
        if self._validation_headers and "all" in normalized and self._last_epoch is not None:
            index = normalized.index("all")
            metrics = _metrics_from_columns(self._validation_headers, tokens[index:])
            if metrics:
                return ParsedMetrics(
                    metrics,
                    epoch=self._last_epoch,
                    total_epochs=self._last_total_epochs,
                )
        return None

    def _keyed(self, line: str) -> ParsedMetrics | None:
        values: dict[str, object] = {
            match.group("key"): match.group("value") for match in KEY_VALUE_PATTERN.finditer(line)
        }
        epoch_match = EPOCH_PATTERN.search(line)
        if epoch_match:
            values["epoch"] = int(epoch_match.group("current"))
            values["epochs"] = int(epoch_match.group("total"))
        return _from_mapping(values)

    def _remember_epoch(self, parsed: ParsedMetrics) -> None:
        if parsed.epoch is not None:
            self._last_epoch = parsed.epoch
        if parsed.total_epochs is not None:
            self._last_total_epochs = parsed.total_epochs


def _merge_metrics(
    first: ParsedMetrics | None, second: ParsedMetrics | None
) -> ParsedMetrics | None:
    if first is None:
        return second
    if second is None:
        return first
    return ParsedMetrics(
        metrics={**first.metrics, **second.metrics},
        step=second.step if second.step is not None else first.step,
        epoch=second.epoch if second.epoch is not None else first.epoch,
        total_epochs=(
            second.total_epochs if second.total_epochs is not None else first.total_epochs
        ),
    )


def _from_mapping(values: dict[str, object]) -> ParsedMetrics | None:
    metrics: dict[str, float] = {}
    step = _integer(values.get("global_step", values.get("step")), minimum=0)
    epoch = _integer(values.get("epoch"), minimum=1)
    total_epochs = _integer(
        values.get("epochs", values.get("total_epochs", values.get("totalEpochs"))),
        minimum=1,
    )
    for name, raw_value in values.items():
        canonical = _canonical_metric_name(name)
        value = _number(raw_value)
        if canonical is not None and value is not None:
            metrics[canonical] = value
    if not metrics:
        return None
    return ParsedMetrics(metrics, step=step, epoch=epoch, total_epochs=total_epochs)


def _metrics_from_columns(headers: tuple[str, ...], values: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for header, raw_value in zip(headers, values, strict=False):
        canonical = _canonical_metric_name(header)
        value = _number(raw_value)
        if canonical is not None and value is not None:
            metrics[canonical] = value
    return metrics


def _canonical_metric_name(name: str) -> str | None:
    normalized = name.strip().lower()
    normalized = re.sub(r"\([a-z]+\)$", "", normalized)
    normalized = normalized.replace("metrics/", "")
    normalized = normalized.replace("train/", "train_").replace("val/", "val_")
    normalized = normalized.replace("mAP", "map")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    normalized = METRIC_ALIASES.get(normalized, normalized)
    if not normalized or normalized in {"epoch", "epochs", "step", "global_step"}:
        return None
    return normalized if any(marker in normalized for marker in METRIC_MARKERS) else None


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _integer(value: object, *, minimum: int) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer() or number < minimum:
        return None
    return int(number)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value) if isinstance(value, (int, float, str)) else math.nan
    except ValueError:
        return None
    return number if math.isfinite(number) else None
