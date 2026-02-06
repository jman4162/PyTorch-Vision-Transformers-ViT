"""Image transforms for ViT training and inference."""

from torchvision import transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(
    image_size: int = 224,
    horizontal_flip: bool = True,
    rotation: int = 10,
    color_jitter: bool = True,
) -> transforms.Compose:
    """Get training transform with data augmentation.

    Args:
        image_size: Target image size
        horizontal_flip: Apply random horizontal flip
        rotation: Max rotation degrees (0 to disable)
        color_jitter: Apply color jitter augmentation

    Returns:
        Composed transform for training data
    """
    transform_list = [transforms.Resize((image_size, image_size))]

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))

    if color_jitter:
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transforms.Compose(transform_list)


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Get validation/test transform without augmentation.

    Args:
        image_size: Target image size

    Returns:
        Composed transform for validation/test data
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Get transform for single image inference.

    Alias for get_val_transform for clarity.

    Args:
        image_size: Target image size

    Returns:
        Composed transform for inference
    """
    return get_val_transform(image_size)


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Denormalize a tensor for visualization.

    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized tensor
    """
    import torch

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return torch.clamp(tensor * std + mean, 0, 1)
