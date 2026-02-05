"""Tests for model loading and configuration."""

import pytest
import torch
from torch import nn

from vit_trainer.models import load_model, VIT_VARIANTS, get_model_info, count_parameters


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_vit_b_16(self):
        """Test loading vit_b_16 model."""
        model = load_model("vit_b_16", num_classes=10)
        assert isinstance(model, nn.Module)
        assert model.heads.head.out_features == 10

    def test_load_vit_b_32(self):
        """Test loading vit_b_32 model."""
        model = load_model("vit_b_32", num_classes=10)
        assert isinstance(model, nn.Module)
        assert model.heads.head.out_features == 10

    def test_load_with_custom_classes(self):
        """Test loading model with custom number of classes."""
        for num_classes in [2, 100, 1000]:
            model = load_model("vit_b_16", num_classes=num_classes)
            assert model.heads.head.out_features == num_classes

    def test_load_invalid_variant(self):
        """Test that invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            load_model("vit_invalid", num_classes=10)

    def test_model_forward_pass(self):
        """Test model forward pass with dummy input."""
        model = load_model("vit_b_16", num_classes=10)
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, 10)

    def test_load_without_pretrained(self):
        """Test loading model without pretrained weights."""
        model = load_model("vit_b_16", num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)


class TestVITVariants:
    """Tests for VIT_VARIANTS registry."""

    def test_all_variants_exist(self):
        """Test that all expected variants are in registry."""
        expected = ["vit_b_16", "vit_b_32", "vit_l_16"]
        for variant in expected:
            assert variant in VIT_VARIANTS

    def test_variant_info_structure(self):
        """Test that variant info has expected fields."""
        for variant_name, (model_fn, weights, info) in VIT_VARIANTS.items():
            assert "patch_size" in info
            assert "hidden_dim" in info
            assert "num_heads" in info
            assert "params" in info


class TestModelInfo:
    """Tests for get_model_info function."""

    def test_get_info_vit_b_16(self):
        """Test getting info for vit_b_16."""
        info = get_model_info("vit_b_16")
        assert info["patch_size"] == 16
        assert info["hidden_dim"] == 768
        assert info["num_heads"] == 12

    def test_get_info_invalid(self):
        """Test that invalid variant raises error."""
        with pytest.raises(ValueError):
            get_model_info("invalid_variant")


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_all_parameters(self):
        """Test counting all parameters."""
        model = load_model("vit_b_16", num_classes=10)
        total = count_parameters(model, trainable_only=False)
        assert total > 0

    def test_count_trainable_only(self):
        """Test counting trainable parameters."""
        model = load_model("vit_b_16", num_classes=10)

        # Freeze some layers
        for param in model.conv_proj.parameters():
            param.requires_grad = False

        trainable = count_parameters(model, trainable_only=True)
        total = count_parameters(model, trainable_only=False)

        assert trainable < total
