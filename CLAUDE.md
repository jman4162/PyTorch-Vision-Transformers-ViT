# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`vit-trainer` is a pip-installable Python package for fine-tuning Vision Transformer (ViT) models. It provides a clean, educational implementation with modern training techniques, achieving 97.65% accuracy on CIFAR-10.

## Package Structure

```
vit-trainer/
├── vit_trainer/              # Main package
│   ├── __init__.py           # Public API exports
│   ├── config.py             # TrainingConfig dataclass
│   ├── cli.py                # Command-line interface
│   ├── data/
│   │   ├── cifar.py          # CIFAR-10/100 loaders
│   │   └── transforms.py     # Image transforms
│   ├── models/
│   │   └── vit.py            # Model registry + factory
│   ├── training/
│   │   ├── trainer.py        # Trainer class (AMP, warmup)
│   │   └── callbacks.py      # EarlyStopping, Checkpointing
│   ├── evaluation/
│   │   └── metrics.py        # Accuracy, confusion matrix
│   └── visualization/
│       └── attention.py      # Attention map extraction
├── tests/                    # Unit tests (pytest)
├── configs/                  # YAML configurations
├── notebooks/                # Tutorial notebooks
├── app.py                    # Gradio demo
└── pyproject.toml            # Package configuration
```

## Running the Project

### Install Package
```bash
pip install -e .                  # Basic install
pip install -e ".[all]"           # With all extras
```

### CLI Commands
```bash
vit-train train --model vit_b_16 --dataset cifar10 --epochs 10
vit-train eval --checkpoint best_model.pt --dataset cifar10
vit-train predict --checkpoint best_model.pt --image cat.jpg
vit-train export --checkpoint best_model.pt --output model.onnx
```

### Python API
```python
from vit_trainer import Trainer, load_model, get_cifar10_loaders

train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)
model = load_model("vit_b_16", num_classes=10)
trainer = Trainer(model, lr=1e-4, use_amp=True)
trainer.fit(train_loader, val_loader, epochs=10)
```

### Gradio Demo
```bash
python app.py  # Opens at http://localhost:7860
```

### Run Tests
```bash
pytest tests/
```

## Key Implementation Details

### Data Pipeline
- CIFAR-10/100 with `Subset` (not `random_split`) for proper validation transforms
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Default augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter

### Model Variants
| Variant | Parameters | Use Case |
|---------|------------|----------|
| `vit_b_16` | 86M | Default - best accuracy/speed |
| `vit_b_32` | 88M | Faster inference |
| `vit_l_16` | 304M | Higher accuracy |

### Training Configuration
- Optimizer: AdamW (lr=1e-4, weight_decay=0.05)
- Scheduler: Cosine annealing with linear warmup (2 epochs)
- Mixed Precision: AMP with GradScaler
- Gradient Clipping: max_norm=1.0
- Early stopping: patience=3

### Public API (from `__init__.py`)
```python
# Config
TrainingConfig, ExportConfig

# Models
load_model, VIT_VARIANTS, get_model_info

# Data
get_cifar10_loaders, get_cifar100_loaders
CIFAR10_CLASSES, CIFAR100_CLASSES
get_train_transform, get_val_transform

# Training
Trainer, EarlyStopping, ModelCheckpoint

# Evaluation
evaluate_model, get_predictions, compute_metrics, plot_confusion_matrix

# Visualization
visualize_attention, show_attention_on_image, visualize_samples_with_attention
```

## Development Commands

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black vit_trainer/
ruff check vit_trainer/ --fix

# Type check
mypy vit_trainer/
```

## Environment Notes

- Python 3.8+ required
- GPU (CUDA) strongly recommended for training
- Auto-detects device (CUDA/CPU)
- Models saved to `./models/` by default
