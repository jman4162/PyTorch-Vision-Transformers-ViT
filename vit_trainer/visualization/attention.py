"""Attention map visualization for Vision Transformers.

Attention weights are a diagnostic, not an explanation: they show which tokens
a head pooled from, which is not the same as why the model made a prediction.
See Jain & Wallace, "Attention is not Explanation" (arXiv:1902.10186).
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from ..data.transforms import denormalize


def _layer_attention(
    layer: nn.Module, hidden_dim: int, x: torch.Tensor
) -> torch.Tensor:
    """Recompute one encoder layer's self-attention weights.

    ``nn.MultiheadAttention`` is called with ``need_weights=False`` inside
    torchvision's ``EncoderBlock``, so the weights are never materialized. We
    redo the projection from the layer's own parameters instead.

    Args:
        layer: A torchvision ``EncoderBlock``
        hidden_dim: Model embedding dimension
        x: Layer input [B, tokens, hidden_dim]

    Returns:
        Attention weights [B, heads, tokens, tokens]
    """
    ln_x = layer.ln_1(x)

    qkv = F.linear(
        ln_x,
        layer.self_attention.in_proj_weight,
        layer.self_attention.in_proj_bias,
    )

    batch = x.shape[0]
    num_heads = layer.self_attention.num_heads
    head_dim = hidden_dim // num_heads

    qkv = qkv.reshape(batch, -1, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]

    scores = torch.matmul(q, k.transpose(-2, -1)) * head_dim**-0.5
    return torch.softmax(scores, dim=-1)


def forward_with_attention(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the ViT forward pass, capturing attention weights at one layer.

    This reproduces ``torchvision.models.VisionTransformer.forward`` step by
    step. The positional embedding and encoder dropout matter: without them the
    captured attention comes from a representation the model never sees, and
    the resulting map still looks plausible. ``test_attention.py`` asserts the
    logits returned here match ``model(input_tensor)``.

    Args:
        model: torchvision ViT model
        input_tensor: Batched image tensor [B, C, H, W]
        layer_idx: Encoder layer to capture (-1 for last)

    Returns:
        Tuple of (attention weights [B, heads, tokens, tokens], logits [B, classes])

    Raises:
        ValueError: If layer_idx is out of range
    """
    num_layers = len(model.encoder.layers)
    resolved_idx = layer_idx + num_layers if layer_idx < 0 else layer_idx
    if not 0 <= resolved_idx < num_layers:
        raise ValueError(
            f"layer_idx {layer_idx} out of range for a model with {num_layers} layers"
        )

    x = model._process_input(input_tensor)
    batch = x.shape[0]

    # Prepend the class token, then add position information and apply dropout,
    # matching torchvision's Encoder.forward.
    x = torch.cat([model.class_token.expand(batch, -1, -1), x], dim=1)
    x = x + model.encoder.pos_embedding
    x = model.encoder.dropout(x)

    attn_weights = None
    for i, layer in enumerate(model.encoder.layers):
        if i == resolved_idx:
            attn_weights = _layer_attention(layer, model.hidden_dim, x)
        x = layer(x)

    # Finish the forward pass so the logits can be checked against the model.
    logits = model.heads(model.encoder.ln(x)[:, 0])

    assert attn_weights is not None  # resolved_idx is range-checked above
    return attn_weights, logits


def visualize_attention(
    model: nn.Module,
    image: torch.Tensor,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract the CLS token's attention over image patches.

    Args:
        model: ViT model
        image: Image tensor [C, H, W] or [1, C, H, W]
        layer_idx: Which transformer layer to visualize (-1 for last)
        head_idx: Which attention head (None for average across heads)
        device: Device to run inference on

    Returns:
        Attention map as a 2D array (14x14 for vit_b_16 at 224px)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    input_tensor = image.unsqueeze(0) if image.dim() == 3 else image
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        attn_weights, _ = forward_with_attention(model, input_tensor, layer_idx)

    # Row 0 is the CLS token; column 0 is its attention to itself, so drop it.
    cls_attn = attn_weights[:, :, 0, 1:]  # [B, heads, num_patches]

    if head_idx is not None:
        cls_attn = cls_attn[:, head_idx]
    else:
        cls_attn = cls_attn.mean(dim=1)

    cls_attn = cls_attn[0]  # first image of the batch
    grid = int(round(cls_attn.shape[-1] ** 0.5))

    return cls_attn.reshape(grid, grid).cpu().numpy()


def show_attention_on_image(
    image: Image.Image,
    attention_map: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Overlay an attention map on the original image.

    Args:
        image: PIL Image
        attention_map: 2D array (e.g. 14x14)
        alpha: Transparency for overlay (0=original, 1=heatmap only)

    Returns:
        Blended image as a uint8 array
    """
    image = image.convert("RGB")

    # Stretch to [0, 1] *before* the 8-bit conversion. A CLS attention row is a
    # distribution over ~200 patches, so its values average around 0.005 —
    # quantizing first collapses most of the map to zero.
    attn = attention_map.astype(np.float32)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    attn_resized = (
        np.array(
            Image.fromarray((attn * 255).astype(np.uint8)).resize(
                image.size, Image.BILINEAR
            )
        )
        / 255.0
    )

    heatmap = (plt.cm.jet(attn_resized)[:, :, :3] * 255).astype(np.uint8)

    img_array = np.array(image)
    blended = (1 - alpha) * img_array + alpha * heatmap

    return blended.astype(np.uint8)


def visualize_samples_with_attention(
    model: nn.Module,
    dataset,
    class_names: List[str],
    num_samples: int = 4,
    layer_idx: int = -1,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """Visualize predictions with attention maps for random samples.

    Args:
        model: ViT model
        dataset: PyTorch dataset
        class_names: List of class names
        num_samples: Number of samples to visualize
        layer_idx: Transformer layer for attention (-1 for last)
        device: Device to run inference on
        save_path: Path to save figure (optional)
        seed: Seed for sample selection, for reproducible figures
    """
    from torchvision import transforms

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]

        # Denormalize for visualization
        img_pil = transforms.ToPILImage()(denormalize(img_tensor))

        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        attn_map = visualize_attention(
            model, img_tensor, layer_idx=layer_idx, device=device
        )

        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_title(f"True: {class_names[label]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(attn_map, cmap="hot")
        axes[i, 1].set_title(f"CLS attention (layer {layer_idx})")
        axes[i, 1].axis("off")

        overlay = show_attention_on_image(img_pil.resize((224, 224)), attn_map)
        axes[i, 2].imshow(overlay)

        pred_label = class_names[pred_idx]
        color = "green" if pred_idx == label else "red"
        axes[i, 2].set_title(f"Pred: {pred_label} ({confidence:.1%})", color=color)
        axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
