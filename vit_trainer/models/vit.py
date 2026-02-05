"""Vision Transformer model registry and factory functions."""

from typing import Dict, Tuple, Optional, Any
import torch
from torch import nn
from torchvision.models import (
    vit_b_16,
    vit_b_32,
    vit_l_16,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
)


# Model registry: variant -> (model_fn, weights, info)
VIT_VARIANTS: Dict[str, Tuple[Any, Any, Dict[str, Any]]] = {
    "vit_b_16": (
        vit_b_16,
        ViT_B_16_Weights.IMAGENET1K_V1,
        {
            "patch_size": 16,
            "hidden_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "params": "86M",
            "imagenet_acc": "81.1%",
        },
    ),
    "vit_b_32": (
        vit_b_32,
        ViT_B_32_Weights.IMAGENET1K_V1,
        {
            "patch_size": 32,
            "hidden_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "params": "88M",
            "imagenet_acc": "75.9%",
        },
    ),
    "vit_l_16": (
        vit_l_16,
        ViT_L_16_Weights.IMAGENET1K_V1,
        {
            "patch_size": 16,
            "hidden_dim": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "params": "304M",
            "imagenet_acc": "79.7%",
        },
    ),
}


def get_model_info(variant: str) -> Dict[str, Any]:
    """Get information about a ViT variant.

    Args:
        variant: Model variant name

    Returns:
        Dictionary with model information

    Raises:
        ValueError: If variant is not supported
    """
    if variant not in VIT_VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Choose from {list(VIT_VARIANTS.keys())}"
        )
    return VIT_VARIANTS[variant][2]


def load_model(
    variant: str = "vit_b_16",
    num_classes: int = 10,
    pretrained: bool = True,
    compile_model: bool = False,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a Vision Transformer model.

    Args:
        variant: ViT variant ('vit_b_16', 'vit_b_32', 'vit_l_16')
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        compile_model: Use torch.compile for optimization (PyTorch 2.x)
        checkpoint_path: Path to load trained weights from
        device: Device to load model on

    Returns:
        Configured ViT model

    Raises:
        ValueError: If variant is not supported

    Example:
        >>> model = load_model("vit_b_16", num_classes=10)
        >>> model = load_model("vit_b_16", checkpoint_path="best_model.pt")
    """
    if variant not in VIT_VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Choose from {list(VIT_VARIANTS.keys())}"
        )

    model_fn, weights, _ = VIT_VARIANTS[variant]

    # Load model with or without pretrained weights
    if pretrained and checkpoint_path is None:
        model = model_fn(weights=weights)
    else:
        model = model_fn(weights=None)

    # Modify classification head for target number of classes
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    # Optional: Compile model for faster inference (PyTorch 2.x)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the classification head.

    Useful for linear probing before fine-tuning.

    Args:
        model: ViT model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classification head
    for param in model.heads.parameters():
        param.requires_grad = True


def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze all model parameters.

    Args:
        model: ViT model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


def get_layer_groups(model: nn.Module) -> Dict[str, nn.Module]:
    """Get named layer groups for discriminative learning rates.

    Args:
        model: ViT model

    Returns:
        Dictionary mapping group names to modules
    """
    return {
        "embeddings": model.conv_proj,
        "encoder": model.encoder,
        "head": model.heads,
    }
