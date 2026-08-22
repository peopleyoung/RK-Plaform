from __future__ import annotations

import pytest
from workers.common.artifacts import model_artifact_stem


def test_model_artifact_stem_includes_input_scale() -> None:
    assert model_artifact_stem("mobilenet_v2", 512, 384) == "mobilenet_v2-512x384"


def test_model_artifact_stem_removes_unsafe_path_characters() -> None:
    assert model_artifact_stem("../custom model", 640, 640) == "custom_model-640x640"


@pytest.mark.parametrize(("width", "height"), [(0, 640), (640, -1)])
def test_model_artifact_stem_rejects_invalid_scale(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        model_artifact_stem("model", width, height)
