from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from rknn.api import RKNN


def create_probe_model(path: Path, height: int, width: int) -> None:
    input_info = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, height, width]
    )
    output_info = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 3, height, width]
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "rknode-board-probe",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save(model, path)


def check(code: int, stage: str) -> None:
    if code != 0:
        raise RuntimeError(f"{stage} failed with return code {code}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe RKNN conversion and on-board inference")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/rknode-probe"))
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument(
        "--runtime-target",
        choices=("rk3588", "local"),
        default="rk3588",
        help="Use 'local' to call init_runtime without a target.",
    )
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.work_dir / "probe.onnx"
    rknn_path = args.work_dir / "probe.rknn"
    create_probe_model(onnx_path, args.height, args.width)

    converter = RKNN(verbose=True)
    try:
        check(
            converter.config(
                mean_values=[[0.0, 0.0, 0.0]],
                std_values=[[255.0, 255.0, 255.0]],
                target_platform="rk3588",
            ),
            "config",
        )
        check(converter.load_onnx(model=str(onnx_path)), "load_onnx")
        check(converter.build(do_quantization=False), "build")
        check(converter.export_rknn(str(rknn_path)), "export_rknn")
        if args.runtime_target == "local":
            check(converter.init_runtime(), "init_runtime(local)")
        else:
            check(converter.init_runtime(target="rk3588"), "init_runtime(rk3588)")
        sample = np.zeros((1, args.height, args.width, 3), dtype=np.uint8)
        outputs = converter.inference(inputs=[sample], data_format="nhwc")
        if not outputs:
            raise RuntimeError("inference returned no outputs")
        print(f"PROBE_OK output_shapes={[list(value.shape) for value in outputs]}")
    finally:
        converter.release()


if __name__ == "__main__":
    main()
