from __future__ import annotations

import numpy as np
import pytest
from scripts.deeplabv3_rknn_infer import (
    classify_logits,
    logits_to_chw,
    parse_labels,
    pascal_colormap,
)


def test_nchw_deeplab_output_is_preserved() -> None:
    output = np.zeros((1, 2, 4, 5), dtype=np.float32)

    logits = logits_to_chw(output, class_count=2)

    assert logits.shape == (2, 4, 5)


def test_nhwc_deeplab_output_is_transposed() -> None:
    output = np.zeros((1, 4, 5, 3), dtype=np.float32)

    logits = logits_to_chw(output, class_count=3)

    assert logits.shape == (3, 4, 5)


def test_label_count_must_match_output_channels() -> None:
    output = np.zeros((1, 2, 4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="Cannot find a 3-channel class axis"):
        logits_to_chw(output, class_count=3)


def test_classification_and_pascal_colors_are_stable() -> None:
    logits = np.array(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 2.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )

    mask = classify_logits(logits)
    colors = pascal_colormap(3)

    assert mask.tolist() == [[0, 1], [1, 0]]
    assert colors.tolist() == [[0, 0, 0], [128, 0, 0], [0, 128, 0]]


def test_labels_are_trimmed_and_unique() -> None:
    assert parse_labels(" background, ng ") == ("background", "ng")
    with pytest.raises(ValueError, match="unique"):
        parse_labels("background,ng,ng")
