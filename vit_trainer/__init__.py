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
    get_class_names,
    get_train_transform,
    get_val_transform,
    make_split_indices,
)
from .evaluation import (
    compute_metrics,
    evaluate_model,
    get_predictions,
    plot_confusion_matrix,
    plot_training_history,
)
from .models import (
    VIT_VARIANTS,
    freeze_backbone,
    get_model_info,
    load_model,
    load_state_dict,
    read_checkpoint_metadata,
    unfreeze_model,
)
from .provenance import collect_run_metadata, hash_indices, save_run_metadata
from .training import EarlyStopping, ModelCheckpoint, Trainer
from .visualization import (
    forward_with_attention,
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
    "load_state_dict",
    "read_checkpoint_metadata",
    "freeze_backbone",
    "unfreeze_model",
    "VIT_VARIANTS",
    "get_model_info",
    # Data
    "get_cifar10_loaders",
    "get_cifar100_loaders",
    "get_class_names",
    "make_split_indices",
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
    "plot_training_history",
    # Provenance
    "collect_run_metadata",
    "save_run_metadata",
    "hash_indices",
    # Visualization
    "visualize_attention",
    "forward_with_attention",
    "show_attention_on_image",
    "visualize_samples_with_attention",
]
