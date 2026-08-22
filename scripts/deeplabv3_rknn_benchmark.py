from __future__ import annotations

import argparse
import importlib
import queue
import statistics
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

if __package__:
    from scripts.deeplabv3_rknn_infer import (
        classify_logits,
        logits_to_chw,
        parse_labels,
        require_file,
        resize_logits,
    )
else:
    from deeplabv3_rknn_infer import (  # type: ignore[import-not-found]
        classify_logits,
        logits_to_chw,
        parse_labels,
        require_file,
        resize_logits,
    )

BenchmarkMode = Literal["single", "parallel"]
OutputLayout = Literal["auto", "nchw", "nhwc"]
CORE_ATTRIBUTE_MAP = {
    "auto": "NPU_CORE_AUTO",
    "0": "NPU_CORE_0",
    "1": "NPU_CORE_1",
    "2": "NPU_CORE_2",
    "01": "NPU_CORE_0_1",
    "012": "NPU_CORE_0_1_2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DeepLabV3+ on RK3588 NPU")
    parser.add_argument("--model-path", "--model_path", dest="model_path", type=Path, required=True)
    parser.add_argument(
        "--image-path",
        "--image_path",
        dest="image_path",
        type=Path,
        default=Path("../model/11.png"),
    )
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--device-id", "--device_id", dest="device_id", default=None)
    parser.add_argument("--mode", choices=("single", "parallel"), default="single")
    parser.add_argument(
        "--core-mask",
        "--core_mask",
        dest="core_mask",
        choices=tuple(CORE_ATTRIBUTE_MAP),
        default="012",
        help="single mode core selection; 012 gives minimum latency",
    )
    parser.add_argument(
        "--cores",
        default="012",
        help="parallel mode streams pinned to unique cores, e.g. 012",
    )
    parser.add_argument(
        "--allow-unstable-parallel",
        action="store_true",
        help=(
            "allow multiple concurrent RKNN runtime instances; Toolkit2 2.3.2 with "
            "RKNPU driver 0.9.8 can time out or crash the kernel under this load"
        ),
    )
    parser.add_argument("--loops", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--labels", default="background,ng,scratch")
    parser.add_argument(
        "--output-layout",
        choices=("auto", "nchw", "nhwc"),
        default="auto",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help=(
            "include in-memory color conversion, resize and logits post-processing; "
            "excludes disk I/O"
        ),
    )
    return parser.parse_args()


def parse_parallel_cores(raw: str) -> tuple[str, ...]:
    core_ids = tuple(raw)
    if not core_ids or any(item not in {"0", "1", "2"} for item in core_ids):
        raise ValueError("--cores must contain only 0, 1 and 2")
    if len(core_ids) != len(set(core_ids)):
        raise ValueError("--cores must not contain duplicate core IDs")
    return core_ids


def validate_parallel_safety(
    core_ids: tuple[str, ...],
    *,
    allow_unstable_parallel: bool,
) -> None:
    if len(core_ids) > 1 and not allow_unstable_parallel:
        raise ValueError(
            "Multiple concurrent RKNN runtimes are disabled because Toolkit2 2.3.2 with "
            "RKNPU driver 0.9.8 has produced NPU job timeouts and kernel crashes on this "
            "RK3588 host. Use --mode single --core-mask 012 for single-stream latency. "
            "Upgrade and qualify the board runtime/driver before using "
            "--allow-unstable-parallel."
        )


def resolve_core_mask(rknn_type: Any, selection: str) -> int:
    attribute = CORE_ATTRIBUTE_MAP[selection]
    value = getattr(rknn_type, attribute, None)
    if not isinstance(value, int):
        raise RuntimeError(f"RKNN runtime does not provide {attribute}")
    return value


def preprocess(source_bgr: np.ndarray, width: int, height: int, cv2: Any) -> np.ndarray:
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(source_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def postprocess(
    output: Any,
    class_count: int,
    output_layout: OutputLayout,
    height: int,
    width: int,
    cv2: Any,
) -> np.ndarray:
    logits = logits_to_chw(np.asarray(output), class_count, output_layout)
    logits = resize_logits(logits, height, width, cv2)
    return classify_logits(logits)


def inference_action(
    runtime: Any,
    prepared_input: np.ndarray,
    source_bgr: np.ndarray,
    *,
    width: int,
    height: int,
    class_count: int,
    output_layout: OutputLayout,
    e2e: bool,
    cv2: Any,
) -> Callable[[], None]:
    def invoke() -> None:
        model_input = preprocess(source_bgr, width, height, cv2) if e2e else prepared_input
        outputs = runtime.inference(inputs=[model_input], data_format="nhwc")
        if not outputs:
            raise RuntimeError("RKNN inference returned no outputs")
        if e2e:
            postprocess(
                outputs[0],
                class_count,
                output_layout,
                source_bgr.shape[0],
                source_bgr.shape[1],
                cv2,
            )

    return invoke


def measure(action: Callable[[], None], warmup: int, loops: int) -> list[float]:
    for _ in range(warmup):
        action()
    latencies: list[float] = []
    for _ in range(loops):
        started = time.perf_counter()
        action()
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def run_parallel(
    actions: dict[str, Callable[[], None]],
    warmup: int,
    loops: int,
) -> tuple[dict[str, list[float]], float]:
    warmup_actions: dict[str, Callable[[], None]] = {
        core_id: partial(_repeat, action, warmup)
        for core_id, action in actions.items()
    }
    _run_threads(warmup_actions)

    results: dict[str, list[float]] = {}
    result_lock = threading.Lock()
    started_at = 0.0

    def mark_start() -> None:
        nonlocal started_at
        started_at = time.perf_counter()

    barrier = threading.Barrier(len(actions) + 1, action=mark_start)

    def measured_worker(core_id: str, action: Callable[[], None]) -> None:
        barrier.wait()
        values = measure(action, 0, loops)
        with result_lock:
            results[core_id] = values

    workers: dict[str, Callable[[], None]] = {
        core_id: partial(measured_worker, core_id, action)
        for core_id, action in actions.items()
    }
    threads, failures = _start_threads(workers)
    barrier.wait()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    _raise_thread_failure(failures)
    return results, elapsed_ms


def _repeat(action: Callable[[], None], count: int) -> None:
    for _ in range(count):
        action()


def _run_threads(actions: dict[str, Callable[[], None]]) -> None:
    threads, failures = _start_threads(actions)
    for thread in threads:
        thread.join()
    _raise_thread_failure(failures)


def _start_threads(
    actions: dict[str, Callable[[], None]],
) -> tuple[list[threading.Thread], queue.Queue[BaseException]]:
    failures: queue.Queue[BaseException] = queue.Queue()

    def guarded(action: Callable[[], None]) -> None:
        try:
            action()
        except BaseException as error:
            failures.put(error)

    threads = [
        threading.Thread(target=guarded, args=(action,), name=f"rknn-core-{core_id}")
        for core_id, action in actions.items()
    ]
    for thread in threads:
        thread.start()
    return threads, failures


def _raise_thread_failure(failures: queue.Queue[BaseException]) -> None:
    if not failures.empty():
        raise RuntimeError("Parallel RKNN inference failed") from failures.get()


def print_latency_report(latencies: list[float], core_mask: str, e2e: bool) -> None:
    values = np.asarray(latencies)
    average = statistics.fmean(latencies)
    print("\nSingle-stream latency")
    print(f"  Core mask: {core_mask}")
    print(f"  Timing: {'in-memory end-to-end' if e2e else 'inference only'}")
    print(f"  Average: {average:.2f} ms")
    percentiles = np.percentile(values, [50, 90, 99])
    print(
        f"  P50/P90/P99: {percentiles[0]:.2f} / "
        f"{percentiles[1]:.2f} / {percentiles[2]:.2f} ms"
    )
    print(f"  Min/Max/Std: {values.min():.2f} / {values.max():.2f} / {values.std():.2f} ms")
    print(f"  FPS: {1000.0 / average:.2f}")


def print_throughput_report(
    results: dict[str, list[float]],
    elapsed_ms: float,
    loops: int,
    e2e: bool,
) -> None:
    total_inferences = len(results) * loops
    print("\nParallel-stream throughput")
    print(f"  Timing: {'in-memory end-to-end' if e2e else 'inference only'}")
    for core_id, latencies in sorted(results.items()):
        average = statistics.fmean(latencies)
        print(f"  Core {core_id}: {average:.2f} ms, {1000.0 / average:.2f} FPS")
    print(f"  Measured wall time (warmup excluded): {elapsed_ms:.2f} ms")
    print(f"  Aggregate throughput: {total_inferences / (elapsed_ms / 1000.0):.2f} FPS")


def create_runtime(rknn_type: Any, model_path: Path, args: argparse.Namespace, core: str) -> Any:
    runtime = rknn_type(verbose=False)
    if runtime.load_rknn(str(model_path)) != 0:
        raise RuntimeError(f"load_rknn failed for core selection {core}")
    options: dict[str, Any] = {
        "target": args.target,
        "core_mask": resolve_core_mask(rknn_type, core),
    }
    if args.device_id:
        options["device_id"] = args.device_id
    if runtime.init_runtime(**options) != 0:
        runtime.release()
        raise RuntimeError(f"init_runtime failed for core selection {core}")
    return runtime


def run(args: argparse.Namespace) -> None:
    if args.loops <= 0 or args.warmup < 0:
        raise ValueError("--loops must be positive and --warmup must be non-negative")
    if args.input_width <= 0 or args.input_height <= 0:
        raise ValueError("input width and height must be positive")

    try:
        cv2 = cast(Any, importlib.import_module("cv2"))
        rknn_module = cast(Any, importlib.import_module("rknn.api"))
    except ImportError as error:
        raise RuntimeError("This script requires OpenCV and RKNN Toolkit2") from error
    rknn_type = rknn_module.RKNN

    model_path = require_file(args.model_path, "RKNN model")
    image_path = require_file(args.image_path, "input image")
    source_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise ValueError(f"OpenCV cannot decode input image: {image_path}")
    prepared_input = preprocess(source_bgr, args.input_width, args.input_height, cv2)
    labels = parse_labels(args.labels)
    output_layout = cast(OutputLayout, args.output_layout)

    if args.mode == "single":
        runtime = create_runtime(rknn_type, model_path, args, args.core_mask)
        try:
            action = inference_action(
                runtime,
                prepared_input,
                source_bgr,
                width=args.input_width,
                height=args.input_height,
                class_count=len(labels),
                output_layout=output_layout,
                e2e=args.e2e,
                cv2=cv2,
            )
            latencies = measure(action, args.warmup, args.loops)
        finally:
            runtime.release()
        print_latency_report(latencies, args.core_mask, args.e2e)
        return

    core_ids = parse_parallel_cores(args.cores)
    validate_parallel_safety(
        core_ids,
        allow_unstable_parallel=args.allow_unstable_parallel,
    )
    runtimes: dict[str, Any] = {}
    try:
        for core_id in core_ids:
            runtimes[core_id] = create_runtime(rknn_type, model_path, args, core_id)
        actions = {
            core_id: inference_action(
                runtime,
                prepared_input,
                source_bgr,
                width=args.input_width,
                height=args.input_height,
                class_count=len(labels),
                output_layout=output_layout,
                e2e=args.e2e,
                cv2=cv2,
            )
            for core_id, runtime in runtimes.items()
        }
        results, elapsed_ms = run_parallel(actions, args.warmup, args.loops)
    finally:
        for runtime in runtimes.values():
            runtime.release()
    print_throughput_report(results, elapsed_ms, args.loops, args.e2e)


def main() -> None:
    try:
        run(parse_args())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
