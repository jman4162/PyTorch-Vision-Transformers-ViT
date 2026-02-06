"""Configuration management for ViT training."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import yaml


@dataclass
class TrainingConfig:
    """Configuration for ViT training.

    Attributes:
        model_variant: ViT variant ('vit_b_16', 'vit_b_32', 'vit_l_16')
        num_classes: Number of output classes
        batch_size: Training batch size
        epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        warmup_epochs: Number of warmup epochs
        patience: Early stopping patience
        use_amp: Enable mixed precision training
        gradient_clip: Max gradient norm for clipping
        seed: Random seed for reproducibility
        compile_model: Use torch.compile (PyTorch 2.x)
        data_dir: Directory for dataset storage
        model_dir: Directory for saving models
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
    """

    # Model settings
    model_variant: str = "vit_b_16"
    num_classes: int = 10
    compile_model: bool = False

    # Training settings
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    patience: int = 3
    use_amp: bool = True
    gradient_clip: float = 1.0

    # Reproducibility
    seed: int = 42

    # Data settings
    dataset: str = "cifar10"
    data_dir: str = "./data"
    train_split: float = 0.8

    # System settings
    model_dir: str = "./models"
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "auto"

    # Augmentation settings
    augment_train: bool = True
    image_size: int = 224

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_variants = ["vit_b_16", "vit_b_32", "vit_l_16"]
        if self.model_variant not in valid_variants:
            raise ValueError(f"model_variant must be one of {valid_variants}")

        valid_datasets = ["cifar10", "cifar100", "imagefolder"]
        if self.dataset not in valid_datasets:
            raise ValueError(f"dataset must be one of {valid_datasets}")

        if not 0 < self.train_split < 1:
            raise ValueError("train_split must be between 0 and 1")

        if self.lr <= 0:
            raise ValueError("lr must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_args(cls, args) -> "TrainingConfig":
        """Create config from argparse namespace."""
        return cls(**{k: v for k, v in vars(args).items() if hasattr(cls, k)})


@dataclass
class ExportConfig:
    """Configuration for model export.

    Attributes:
        format: Export format ('onnx', 'torchscript')
        opset_version: ONNX opset version
        dynamic_batch: Enable dynamic batch size
        optimize: Apply optimizations
    """

    format: str = "onnx"
    opset_version: int = 14
    dynamic_batch: bool = True
    optimize: bool = True

    def __post_init__(self):
        valid_formats = ["onnx", "torchscript"]
        if self.format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}")
