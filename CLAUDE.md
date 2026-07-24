# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`vit-trainer` is a pip-installable Python package for fine-tuning Vision Transformer (ViT) models. It is an educational, production-minded implementation: short enough to read end to end, with run provenance and verified export.

One run of the predecessor notebook (`notebooks/Fine_tuning_Vision_Transformers_ViT_with_PyTorch.ipynb`, torch 2.2.1+cu121, Colab GPU, seed 42, 10 epochs) reached 97.65% on the CIFAR-10 test set. That result has not been reproduced through the package API — do not describe it as a package benchmark.

## Package Structure

```
vit-trainer/
├── vit_trainer/              # Main package
│   ├── __init__.py           # Public API exports
│   ├── config.py             # TrainingConfig, ExportConfig (export + verify)
│   ├── cli.py                # Command-line interface
│   ├── provenance.py         # Run manifests (commit, env, hardware, split hash)
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
ImageNet top-1 below is torchvision's figure for the `IMAGENET1K_V1` weights this
package loads. L/16 scores *below* B/16 because torchvision trained it from
scratch on ImageNet-1k; do not describe it as the higher-accuracy option.

| Variant | Parameters | ImageNet top-1 | Use Case |
|---------|------------|----------------|----------|
| `vit_b_16` | 86M | 81.1% | Default |
| `vit_b_32` | 88M | 75.9% | Faster inference (4x fewer tokens) |
| `vit_l_16` | 304M | 79.7% | Larger, not better on these weights |

### Training Configuration
- Optimizer: AdamW (lr=1e-4, weight_decay=0.05)
- Scheduler: Cosine annealing with linear warmup (2 epochs)
- Mixed Precision: AMP with GradScaler
- Gradient Clipping: max_norm=1.0
- Early stopping: patience=3

### Public API (from `__init__.py`)
```python
# Config
TrainingConfig, ExportConfig          # ExportConfig has .export() and .verify()

# Models
load_model, load_state_dict, read_checkpoint_metadata
freeze_backbone, unfreeze_model, VIT_VARIANTS, get_model_info

# Data
get_cifar10_loaders, get_cifar100_loaders, get_class_names, make_split_indices
CIFAR10_CLASSES, CIFAR100_CLASSES
get_train_transform, get_val_transform

# Training
Trainer, EarlyStopping, ModelCheckpoint

# Evaluation
evaluate_model, get_predictions, compute_metrics
plot_confusion_matrix, plot_training_history

# Provenance
collect_run_metadata, save_run_metadata, hash_indices

# Visualization
visualize_attention, forward_with_attention
show_attention_on_image, visualize_samples_with_attention
```

### Invariants worth preserving
- `forward_with_attention` must reproduce `model(x)` exactly; `tests/test_attention.py`
  asserts it. Attention capture that skips the positional embedding still yields
  plausible-looking maps, so only the equivalence check catches a regression.
- Normalize an attention map to [0, 1] before any uint8 conversion. A CLS
  attention row averages ~0.005, so quantizing first zeroes most of it.
- Every `TrainingConfig` field must reach the component that consumes it;
  `tests/test_cli.py::TestTrainForwarding` asserts this.
- Export is not done until `ExportConfig.verify` compares logits against the
  runtime. `onnx.checker` only validates the graph.
- Describe attention output as a diagnostic, never as interpretability or
  explanation.

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
