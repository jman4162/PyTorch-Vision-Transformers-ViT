"""CIFAR-10/100 data loading utilities."""

from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .transforms import get_train_transform, get_val_transform

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# CIFAR-100 superclass names
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


def get_cifar10_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    train_split: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: Optional[int] = None,
    image_size: int = 224,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get CIFAR-10 data loaders with proper train/val/test splits.

    Uses Subset instead of random_split to ensure validation data
    uses the correct transforms (without augmentation).

    Args:
        batch_size: Batch size for all loaders
        data_dir: Directory to store/load dataset
        train_split: Fraction of training data for training (rest for validation)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for reproducible splits
        image_size: Target image size
        augment_train: Apply data augmentation to training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)
        >>> for images, labels in train_loader:
        ...     # Training loop
        ...     pass
    """
    # Set seed for reproducible splits
    if seed is not None:
        np.random.seed(seed)

    # Get transforms
    train_transform = get_train_transform(image_size) if augment_train else get_val_transform(image_size)
    val_transform = get_val_transform(image_size)

    # Download dataset once
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)

    # Create split indices
    full_train_size = 50000  # CIFAR-10 training set size
    indices = list(range(full_train_size))
    np.random.shuffle(indices)

    train_size = int(train_split * full_train_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create datasets with PROPER transforms using Subset
    # This fixes the common bug where random_split doesn't change transforms
    train_dataset = Subset(
        datasets.CIFAR10(root=data_dir, train=True, transform=train_transform),
        train_indices,
    )
    val_dataset = Subset(
        datasets.CIFAR10(root=data_dir, train=True, transform=val_transform),
        val_indices,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_cifar100_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    train_split: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: Optional[int] = None,
    image_size: int = 224,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get CIFAR-100 data loaders with proper train/val/test splits.

    Args:
        batch_size: Batch size for all loaders
        data_dir: Directory to store/load dataset
        train_split: Fraction of training data for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for reproducible splits
        image_size: Target image size
        augment_train: Apply data augmentation to training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if seed is not None:
        np.random.seed(seed)

    train_transform = get_train_transform(image_size) if augment_train else get_val_transform(image_size)
    val_transform = get_val_transform(image_size)

    datasets.CIFAR100(root=data_dir, train=True, download=True)
    datasets.CIFAR100(root=data_dir, train=False, download=True)

    full_train_size = 50000
    indices = list(range(full_train_size))
    np.random.shuffle(indices)

    train_size = int(train_split * full_train_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(
        datasets.CIFAR100(root=data_dir, train=True, transform=train_transform),
        train_indices,
    )
    val_dataset = Subset(
        datasets.CIFAR100(root=data_dir, train=True, transform=val_transform),
        val_indices,
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_class_names(dataset: str = "cifar10") -> list:
    """Get class names for a dataset.

    Args:
        dataset: Dataset name ('cifar10' or 'cifar100')

    Returns:
        List of class names
    """
    if dataset == "cifar10":
        return CIFAR10_CLASSES
    elif dataset == "cifar100":
        return CIFAR100_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
