"""Tests for model export and numerical parity with the runtime."""

import pytest
import torch

from vit_trainer.config import ExportConfig
from vit_trainer.models import load_model

onnxruntime = pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def small_model():
    """A tiny ViT keeps export tests fast while exercising the real graph."""
    from torchvision.models import VisionTransformer

    model = VisionTransformer(
        image_size=64,
        patch_size=16,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=10,
    )
    model.eval()
    return model


class TestExportConfig:
    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="format must be one of"):
            ExportConfig(format="tflite")

    def test_readme_constructor_signature(self, small_model, tmp_path):
        """The README advertised this call before it existed."""
        config = ExportConfig(
            output_path=str(tmp_path / "model.onnx"),
            opset_version=14,
            image_size=64,
        )
        assert config.export(small_model).exists()


class TestOnnxParity:
    def test_logits_match_pytorch(self, small_model, tmp_path):
        """`onnx.checker` validates the graph, not the numbers it produces."""
        config = ExportConfig(output_path=str(tmp_path / "model.onnx"), image_size=64)
        config.export(small_model)

        results = config.verify(small_model, batch_sizes=(1, 4))

        assert results["batches"][1]["predictions_agree"]
        assert results["batches"][1]["max_logit_delta"] < 1e-4
        assert results["batches"][4]["max_logit_delta"] < 1e-4
        assert results["size_mb"] > 0

    def test_dynamic_batch_disabled_skips_larger_batches(self, small_model, tmp_path):
        config = ExportConfig(
            output_path=str(tmp_path / "static.onnx"),
            image_size=64,
            dynamic_batch=False,
        )
        config.export(small_model)

        results = config.verify(small_model, batch_sizes=(1, 4))
        assert set(results["batches"]) == {1}

    def test_verify_fails_on_a_mismatched_model(self, small_model, tmp_path):
        """A parity check that never fails is not a check."""
        config = ExportConfig(output_path=str(tmp_path / "model.onnx"), image_size=64)
        config.export(small_model)

        import copy

        drifted = copy.deepcopy(small_model)
        with torch.no_grad():
            drifted.heads.head.bias.add_(5.0)

        with pytest.raises(AssertionError):
            config.verify(drifted, batch_sizes=(1,))


class TestTorchScriptExport:
    def test_torchscript_roundtrip(self, small_model, tmp_path):
        config = ExportConfig(
            output_path=str(tmp_path / "model.pt"),
            format="torchscript",
            image_size=64,
        )
        config.export(small_model)

        # Scripting generalizes across batch sizes; tracing would not.
        results = config.verify(small_model, batch_sizes=(1, 4))
        assert results["batches"][1]["predictions_agree"]
        assert results["batches"][4]["predictions_agree"]


@pytest.mark.slow
def test_full_vit_b_16_parity(tmp_path):
    """The real export path, at the size users actually deploy."""
    model = load_model("vit_b_16", num_classes=10, pretrained=False)
    config = ExportConfig(output_path=str(tmp_path / "vit_b_16.onnx"))
    config.export(model)

    results = config.verify(model, batch_sizes=(1,), atol=1e-3)
    assert results["batches"][1]["predictions_agree"]
