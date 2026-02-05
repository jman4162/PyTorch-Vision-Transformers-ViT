"""vit-trainer: A simple, educational package for fine-tuning Vision Transformers.

This package provides a clean, well-documented implementation of ViT fine-tuning
with modern training techniques including mixed precision, warmup scheduling,
and attention visualization.

Example:
    >>> from vit_trainer import Trainer, load_model, get_cifar10_loaders
    >>>
    >>> # Load data and model
    >>> train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)
    >>> model = load_model("vit_b_16", num_classes=10)
    >>>
    >>> # Train
    >>> trainer = Trainer(model, lr=1e-4, use_amp=True)
    >>> history = trainer.fit(train_loader, val_loader, epochs=10)
    >>>
    >>> # Evaluate
    >>> loss, accuracy = trainer.evaluate(test_loader)
"""

__version__ = "0.1.0"
__author__ = "John Hodge"

# Core imports
from .config import ExportConfig, TrainingConfig
from .data import (
    CIFAR10_CLASSES,
    CIFAR100_CLASSES,
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_train_transform,
    get_val_transform,
)
from .evaluation import (
    compute_metrics,
    evaluate_model,
    get_predictions,
    plot_confusion_matrix,
)
from .models import VIT_VARIANTS, get_model_info, load_model
from .training import EarlyStopping, ModelCheckpoint, Trainer
from .visualization import (
    show_attention_on_image,
    visualize_attention,
    visualize_samples_with_attention,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "TrainingConfig",
    "ExportConfig",
    # Models
    "load_model",
    "VIT_VARIANTS",
    "get_model_info",
    # Data
    "get_cifar10_loaders",
    "get_cifar100_loaders",
    "CIFAR10_CLASSES",
    "CIFAR100_CLASSES",
    "get_train_transform",
    "get_val_transform",
    # Training
    "Trainer",
    "EarlyStopping",
    "ModelCheckpoint",
    # Evaluation
    "evaluate_model",
    "get_predictions",
    "compute_metrics",
    "plot_confusion_matrix",
    # Visualization
    "visualize_attention",
    "show_attention_on_image",
    "visualize_samples_with_attention",
]
