"""Tests for attention extraction and overlay rendering."""

import numpy as np
import pytest
import torch
from PIL import Image

from vit_trainer.models import load_model
from vit_trainer.visualization import (
    forward_with_attention,
    show_attention_on_image,
    visualize_attention,
)


@pytest.fixture(scope="module")
def model():
    """Untrained ViT-B/16 — weights don't matter for forward-path equivalence."""
    m = load_model("vit_b_16", num_classes=10, pretrained=False)
    m.eval()
    return m


class TestForwardPath:
    """The attention path must be the path the model actually takes."""

    def test_logits_match_model_forward(self, model):
        """Reproduces the bug where pos_embedding was never added.

        The manual forward used for attention capture must agree with
        model(x). Dropping the positional embedding still yields plausible
        attention maps, so only an equivalence check catches it.
        """
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            _, manual_logits = forward_with_attention(model, x)
            reference_logits = model(x)

        assert torch.allclose(manual_logits, reference_logits, atol=1e-4)

    def test_positional_embedding_changes_attention(self, model):
        """Guard the specific regression: pos_embedding must affect the result."""
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            attn_with, _ = forward_with_attention(model, x)

            saved = model.encoder.pos_embedding.detach().clone()
            model.encoder.pos_embedding.data.zero_()
            attn_without, _ = forward_with_attention(model, x)
            model.encoder.pos_embedding.data.copy_(saved)

        assert not torch.allclose(attn_with, attn_without, atol=1e-6)

    def test_attention_rows_are_distributions(self, model):
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            attn, _ = forward_with_attention(model, x)

        # [batch, heads, tokens, tokens]; 197 = 196 patches + CLS
        assert attn.shape == (1, 12, 197, 197)
        assert torch.allclose(attn.sum(dim=-1), torch.ones(1, 12, 197), atol=1e-4)

    def test_invalid_layer_index_raises(self, model):
        with pytest.raises(ValueError, match="out of range"):
            forward_with_attention(model, torch.randn(1, 3, 224, 224), layer_idx=99)


class TestAttentionMap:
    def test_map_shape_matches_patch_grid(self, model):
        attn_map = visualize_attention(model, torch.randn(3, 224, 224))
        assert attn_map.shape == (14, 14)

    def test_batched_input_uses_first_image(self, model):
        """Previously reshaped a [batch, patches] tensor as if it were 2D."""
        batched = torch.randn(2, 3, 224, 224)
        assert visualize_attention(model, batched).shape == (14, 14)

        single = visualize_attention(model, batched[0])
        assert np.allclose(single, visualize_attention(model, batched), atol=1e-5)

    def test_single_head_selection(self, model):
        avg = visualize_attention(model, torch.randn(3, 224, 224))
        head0 = visualize_attention(model, torch.randn(3, 224, 224), head_idx=0)
        assert head0.shape == avg.shape

    def test_no_grad_graph_retained(self, model):
        """The map is returned as numpy, so nothing should hold the graph."""
        attn_map = visualize_attention(model, torch.randn(3, 224, 224))
        assert isinstance(attn_map, np.ndarray)


class TestOverlay:
    @staticmethod
    def _realistic_map(peak_patches=10):
        """A CLS attention row: a softmax over 196 patches, so values ~0.005."""
        rng = np.random.default_rng(0)
        logits = rng.normal(0, 1, 196)
        logits[50 : 50 + peak_patches] += 3
        attn = np.exp(logits)
        return (attn / attn.sum()).reshape(14, 14)

    def test_overlay_preserves_dynamic_range(self):
        """Reproduces the bug where (attn * 255).astype(uint8) came first.

        Without normalizing to [0, 1] first, ~70% of a real attention map
        truncates to exactly 0 and only ~10 levels survive.
        """
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        attn_map = self._realistic_map()

        # alpha=1.0 isolates the heatmap from the base image
        overlay = show_attention_on_image(image, attn_map, alpha=1.0)

        assert len(np.unique(overlay.reshape(-1, 3), axis=0)) > 50

    def test_overlay_peak_lands_on_the_peak_patches(self):
        image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        attn_map = self._realistic_map()
        overlay = show_attention_on_image(image, attn_map, alpha=1.0)

        # jet maps the maximum to red; the argmax patch is row 3 of the 14x14 grid
        peak_row = np.unravel_index(attn_map.argmax(), attn_map.shape)[0]
        brightest_row = overlay[:, :, 0].mean(axis=1).argmax() // 16
        assert abs(int(brightest_row) - int(peak_row)) <= 1

    def test_overlay_shape_and_dtype(self):
        image = Image.new("RGB", (64, 48), color=(10, 20, 30))
        overlay = show_attention_on_image(image, self._realistic_map())

        assert overlay.shape == (48, 64, 3)
        assert overlay.dtype == np.uint8

    def test_rgba_input_accepted(self):
        """Gradio hands back RGBA for PNG uploads."""
        image = Image.new("RGBA", (64, 64), color=(10, 20, 30, 255))
        assert show_attention_on_image(image, self._realistic_map()).shape == (
            64,
            64,
            3,
        )
