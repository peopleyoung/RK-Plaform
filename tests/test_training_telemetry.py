from __future__ import annotations

import sys
from pathlib import Path

from workers.trainer.adapters import CommandStep
from workers.trainer.runner import CommandRunner
from workers.trainer.telemetry import TelemetryEntry, TrainingMetricParser


def test_metric_parser_supports_structured_paddle_and_legacy_yolo_logs() -> None:
    parser = TrainingMetricParser()
    structured = parser.parse(
        'RKNODE_METRIC {"epoch": 1, "epochs": 3, "train/box_loss": 0.4, '
        '"metrics/mAP50(B)": 0.8}'
    )
    assert structured is not None
    assert structured.epoch == 1
    assert structured.total_epochs == 3
    assert structured.metrics == {"train_box_loss": 0.4, "map50": 0.8}

    paddle = parser.parse(
        "epoch: [2/3], global_step: 8, lr: 0.001, loss: 0.25, acc: 0.9"
    )
    assert paddle is not None
    assert paddle.step == 8
    assert paddle.epoch == 2
    assert paddle.metrics == {"lr": 0.001, "loss": 0.25, "accuracy": 0.9}

    assert parser.parse("Epoch GPU_mem box_loss obj_loss cls_loss Instances Size") is None
    yolo = parser.parse("0/2 0G 0.31 0.22 0.11 10 640")
    assert yolo is not None
    assert yolo.epoch == 1
    assert yolo.total_epochs == 3
    assert yolo.metrics == {"box_loss": 0.31, "obj_loss": 0.22, "cls_loss": 0.11}


def test_command_runner_streams_log_batches_and_metric_entries(tmp_path: Path) -> None:
    script = (
        "print('initializing trainer')\n"
        "print('epoch=1/2 train_loss=0.7 val_loss=0.6')\n"
        "print('epoch: [2/2], global_step: 4, loss: 0.4, acc: 0.8')\n"
    )
    step = CommandStep("train", (sys.executable, "-c", script), tmp_path, {})
    progress: list[tuple[int, str, str]] = []
    telemetry: list[TelemetryEntry] = []

    CommandRunner().run(
        (step,),
        tmp_path / "training.log",
        lambda value, stage, message: progress.append((value, stage, message)),
        lambda entries: telemetry.extend(entries),
    )

    assert [item[0] for item in progress] == [10, 45, 79, 80]
    assert progress[1][2] == "Epoch 1/2"
    assert progress[2][2] == "Epoch 2/2"
    assert any(
        entry.type == "log" and "initializing trainer" in entry.message
        for entry in telemetry
    )
    metric_entries = [entry for entry in telemetry if entry.type == "metric"]
    assert [entry.epoch for entry in metric_entries] == [1, 2]
    assert metric_entries[0].metrics == {"train_loss": 0.7, "val_loss": 0.6}
    assert metric_entries[1].metrics == {"loss": 0.4, "accuracy": 0.8}
    assert "initializing trainer" in (tmp_path / "training.log").read_text(encoding="utf-8")
