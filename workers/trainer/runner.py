from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from .adapters import CommandStep
from .telemetry import ParsedMetrics, TelemetryEntry, TrainingMetricParser

TelemetryReporter = Callable[[tuple[TelemetryEntry, ...]], None]
LOG_BATCH_LINES = 20
LOG_BATCH_CHARS = 12000
LOG_FLUSH_SECONDS = 1.0
MAX_LOG_LINE_CHARS = 12000


class CommandExecutionError(RuntimeError):
    pass


class CommandRunner:
    def run(
        self,
        steps: tuple[CommandStep, ...],
        log_path: Path,
        progress: Callable[[int, str, str], None],
        telemetry: TelemetryReporter | None = None,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        total = len(steps)
        with log_path.open("a", encoding="utf-8") as log:
            for index, step in enumerate(steps):
                start_progress = 10 + int(index * 70 / max(total, 1))
                end_progress = 10 + int((index + 1) * 70 / max(total, 1))
                progress(start_progress, step.name, f"Starting {step.name}")
                last_progress = start_progress
                environment = os.environ.copy()
                environment.update(step.env)
                environment["PYTHONUNBUFFERED"] = "1"
                command_line = f"[{step.name}] {' '.join(step.argv)}"
                log.write(f"\n{command_line}\n")
                log.flush()
                if telemetry is not None:
                    telemetry((TelemetryEntry("log", step.name, command_line),))
                process = subprocess.Popen(
                    step.argv,
                    cwd=step.cwd,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert process.stdout is not None
                parser = TrainingMetricParser()
                pending_lines: list[str] = []
                pending_chars = 0
                last_flush = time.monotonic()

                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        message = line.rstrip("\r\n")
                        if len(message) > MAX_LOG_LINE_CHARS:
                            message = f"{message[:MAX_LOG_LINE_CHARS]} [truncated]"
                        parsed = parser.parse(message)
                        if parsed is not None:
                            epoch_progress = _epoch_progress(
                                parsed,
                                start_progress=start_progress,
                                end_progress=end_progress,
                            )
                            if epoch_progress is not None and epoch_progress > last_progress:
                                progress(
                                    epoch_progress,
                                    step.name,
                                    f"Epoch {parsed.epoch}/{parsed.total_epochs}",
                                )
                                last_progress = epoch_progress
                            _report_log_batch(telemetry, step.name, pending_lines)
                            pending_lines.clear()
                            pending_chars = 0
                            last_flush = time.monotonic()
                            if telemetry is not None:
                                telemetry(
                                    (
                                        TelemetryEntry(
                                            "metric",
                                            step.name,
                                            message,
                                            metrics=parsed.metrics,
                                            step=parsed.step,
                                            epoch=parsed.epoch,
                                            total_epochs=parsed.total_epochs,
                                        ),
                                    )
                                )
                        elif message:
                            pending_lines.append(message)
                            pending_chars += len(message)
                            if (
                                len(pending_lines) >= LOG_BATCH_LINES
                                or pending_chars >= LOG_BATCH_CHARS
                                or time.monotonic() - last_flush >= LOG_FLUSH_SECONDS
                            ):
                                _report_log_batch(telemetry, step.name, pending_lines)
                                pending_lines.clear()
                                pending_chars = 0
                                last_flush = time.monotonic()
                    _report_log_batch(telemetry, step.name, pending_lines)
                    pending_lines.clear()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                return_code = process.wait()
                if return_code:
                    raise CommandExecutionError(
                        f"Training step '{step.name}' exited with status {return_code}"
                    )
                progress(end_progress, step.name, f"Finished {step.name}")


def _report_log_batch(
    reporter: TelemetryReporter | None,
    stage: str,
    lines: list[str],
) -> None:
    if reporter is not None and lines:
        reporter((TelemetryEntry("log", stage, "\n".join(lines)),))


def _epoch_progress(
    parsed: ParsedMetrics,
    *,
    start_progress: int,
    end_progress: int,
) -> int | None:
    epoch = parsed.epoch
    total_epochs = parsed.total_epochs
    if epoch is None or total_epochs is None or total_epochs <= 0:
        return None
    bounded_epoch = min(max(epoch, 0), total_epochs)
    span = max(end_progress - start_progress, 1)
    mapped = start_progress + int(span * bounded_epoch / total_epochs)
    return min(mapped, max(start_progress, end_progress - 1))
