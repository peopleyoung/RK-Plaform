from __future__ import annotations

import numpy as np
import pytest
from scripts.deeplabv3_rknn_benchmark import (
    parse_parallel_cores,
    postprocess,
    resolve_core_mask,
    validate_parallel_safety,
)


class FakeRknn:
    NPU_CORE_AUTO = 0
    NPU_CORE_0 = 1
    NPU_CORE_1 = 2
    NPU_CORE_2 = 4
    NPU_CORE_0_1 = 3
    NPU_CORE_0_1_2 = 7


class FakeCv2:
    INTER_LINEAR = 1

    @staticmethod
    def resize(channel: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
        assert interpolation == FakeCv2.INTER_LINEAR
        width, height = size
        return np.resize(channel, (height, width))


def test_core_masks_use_runtime_constants() -> None:
    assert resolve_core_mask(FakeRknn, "auto") == 0
    assert resolve_core_mask(FakeRknn, "012") == 7


def test_parallel_cores_must_be_unique_and_valid() -> None:
    assert parse_parallel_cores("012") == ("0", "1", "2")
    with pytest.raises(ValueError, match="duplicate"):
        parse_parallel_cores("001")
    with pytest.raises(ValueError, match="only 0, 1 and 2"):
        parse_parallel_cores("03")


def test_multi_runtime_parallel_requires_explicit_unstable_opt_in() -> None:
    with pytest.raises(ValueError, match="kernel crashes"):
        validate_parallel_safety(("0", "1", "2"), allow_unstable_parallel=False)

    validate_parallel_safety(("0", "1", "2"), allow_unstable_parallel=True)
    validate_parallel_safety(("1",), allow_unstable_parallel=False)


def test_postprocess_accepts_optimized_nchw_logits() -> None:
    output = np.zeros((1, 3, 2, 2), dtype=np.float32)
    output[:, 2, :, :] = 1.0

    mask = postprocess(output, 3, "auto", 4, 4, FakeCv2())

    assert mask.shape == (4, 4)
    assert np.all(mask == 2)


def test_missing_runtime_core_constant_is_actionable() -> None:
    with pytest.raises(RuntimeError, match="NPU_CORE_0_1_2"):
        resolve_core_mask(object(), "012")
