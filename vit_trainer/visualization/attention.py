"""Attention map visualization for Vision Transformers."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from ..data.transforms import denormalize


def visualize_attention(
    model: nn.Module,
    image: torch.Tensor,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    """Extract attention map from a ViT model for the CLS token.

    Args:
        model: ViT model
        image: Image tensor [C, H, W] or [1, C, H, W]
        layer_idx: Which transformer layer to visualize (-1 for last)
        head_idx: Which attention head (None for average across heads)
        device: Device to run inference on

    Returns:
        Attention map as 2D numpy array (14x14 for vit_b_16)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Prepare input
    if image.dim() == 3:
        input_tensor = image.unsqueeze(0)
    else:
        input_tensor = image
    input_tensor = input_tensor.to(device)

    # Get intermediate representations
    x = model._process_input(input_tensor)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    # Handle negative layer index
    num_layers = len(model.encoder.layers)
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx

    # Pass through encoder layers
    for i, layer in enumerate(model.encoder.layers):
        if i == layer_idx:
            # Get attention weights from this layer
            ln_x = layer.ln_1(x)

            # Compute attention manually to get weights
            qkv = layer.self_attention.in_proj_weight
            qkv_bias = layer.self_attention.in_proj_bias

            # Linear projection
            qkv_out = torch.nn.functional.linear(ln_x, qkv, qkv_bias)

            # Split into Q, K, V
            embed_dim = model.hidden_dim
            num_heads = layer.self_attention.num_heads
            head_dim = embed_dim // num_heads

            qkv_out = qkv_out.reshape(n, -1, 3, num_heads, head_dim).permute(
                2, 0, 3, 1, 4
            )
            q, k, _v = qkv_out[0], qkv_out[1], qkv_out[2]

            # Compute attention weights
            scale = head_dim**-0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_weights, dim=-1)

            # Get CLS token attention (first token attending to all others)
            cls_attn = attn_weights[:, :, 0, 1:]  # [batch, heads, num_patches]

            if head_idx is not None:
                cls_attn = cls_attn[:, head_idx]  # [batch, num_patches]
            else:
                cls_attn = cls_attn.mean(dim=1)  # Average over heads

            # Reshape to 2D
            num_patches = int(cls_attn.shape[-1] ** 0.5)
            attn_map = cls_attn.reshape(num_patches, num_patches)

            return attn_map.cpu().detach().numpy()

        # Continue forward pass
        x = layer(x)

    return None


def show_attention_on_image(
    image: Image.Image,
    attention_map: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Overlay attention map on the original image.

    Args:
        image: PIL Image
        attention_map: 2D numpy array (e.g., 14x14)
        alpha: Transparency for overlay (0=original, 1=heatmap only)

    Returns:
        Blended image as numpy array
    """
    # Resize attention map to image size
    attn_resized = np.array(
        Image.fromarray((attention_map * 255).astype(np.uint8)).resize(
            image.size, Image.BILINEAR
        )
    )

    # Normalize
    attn_resized = attn_resized / (attn_resized.max() + 1e-8)

    # Create heatmap
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend with original image
    img_array = np.array(image)
    blended = (1 - alpha) * img_array + alpha * heatmap
    blended = blended.astype(np.uint8)

    return blended


def visualize_samples_with_attention(
    model: nn.Module,
    dataset,
    class_names: List[str],
    num_samples: int = 4,
    layer_idx: int = -1,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
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
    """
    from torchvision import transforms

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i, idx in enumerate(indices):
        # Get image and label
        img_tensor, label = dataset[idx]

        # Denormalize for visualization
        img_denorm = denormalize(img_tensor)
        img_pil = transforms.ToPILImage()(img_denorm)

        # Get prediction
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        # Get attention map
        attn_map = visualize_attention(model, img_tensor, layer_idx=layer_idx, device=device)

        # Plot original image
        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_title(f"True: {class_names[label]}")
        axes[i, 0].axis("off")

        # Plot attention map
        if attn_map is not None:
            axes[i, 1].imshow(attn_map, cmap="hot")
            axes[i, 1].set_title("Attention Map (Last Layer)")
            axes[i, 1].axis("off")

            # Plot overlay
            overlay = show_attention_on_image(img_pil.resize((224, 224)), attn_map)
            axes[i, 2].imshow(overlay)
        else:
            axes[i, 1].text(
                0.5, 0.5, "Attention extraction failed", ha="center", va="center"
            )
            axes[i, 1].axis("off")
            axes[i, 2].imshow(img_pil)

        pred_label = class_names[pred_idx]
        color = "green" if pred_idx == label else "red"
        axes[i, 2].set_title(f"Pred: {pred_label} ({confidence:.1%})", color=color)
        axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def get_attention_rollout(
    model: nn.Module,
    image: torch.Tensor,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Compute attention rollout across all layers.

    Attention rollout multiplies attention matrices across layers to show
    cumulative attention flow from input to CLS token.

    Args:
        model: ViT model
        image: Image tensor [C, H, W] or [1, C, H, W]
        device: Device to run inference on

    Returns:
        Rolled-out attention map as 2D numpy array
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Collect attention from all layers
    attention_maps = []

    for layer_idx in range(len(model.encoder.layers)):
        attn = visualize_attention(model, image, layer_idx=layer_idx, device=device)
        if attn is not None:
            attn_flat = attn.flatten()
            # Normalize
            attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
            attention_maps.append(attn_flat)

    if not attention_maps:
        return None

    # Multiply attention matrices (simplified rollout)
    rollout = attention_maps[0]
    for attn in attention_maps[1:]:
        rollout = rollout * attn
        rollout = rollout / (rollout.sum() + 1e-8)

    # Reshape to 2D
    side = int(np.sqrt(len(rollout)))
    return rollout.reshape(side, side)
