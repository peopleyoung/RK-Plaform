from __future__ import annotations

import pytest
from workers.trainer.scripts.train_deeplab import _build_model

torch = pytest.importorskip("torch")
smp = pytest.importorskip("segmentation_models_pytorch")


def test_official_rknn_variant_exports_stride_eight_logits_without_plus_decoder() -> None:
    model = _build_model(
        torch,
        smp,
        variant="mobilenet_v2_rknn",
        classes=3,
        width=128,
        height=128,
        pretrained=False,
    )
    model.eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 128, 128))

    assert tuple(output.shape) == (1, 3, 16, 16)
    assert model.encoder.output_stride == 8
    assert model.encoder.out_channels[-1] == 320
    assert len(model.encoder.features) == 18
    assert model.decoder.image_pool.kernel_size == (16, 16)
    assert model.decoder.aspp_projection[0].in_channels == 320
    assert not hasattr(model.decoder, "up")
    assert not hasattr(model.decoder, "block1")
    assert not hasattr(model.decoder, "block2")
