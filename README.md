# vit-trainer

[![PyPI](https://img.shields.io/pypi/v/vit-trainer.svg)](https://pypi.org/project/vit-trainer/)
[![CI](https://github.com/jman4162/PyTorch-Vision-Transformers-ViT/actions/workflows/ci.yml/badge.svg)](https://github.com/jman4162/PyTorch-Vision-Transformers-ViT/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/PyTorch-Vision-Transformers-ViT/blob/main/notebooks/tutorial.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A simple, educational package for fine-tuning Vision Transformer (ViT) models using PyTorch. Achieves **97.65% accuracy** on CIFAR-10 with modern training techniques.

![ViT](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg)

## Why vit-trainer?

| vs. timm/transformers | vit-trainer |
|-----------------------|-------------|
| 1000+ model architectures | Focused on ViT fine-tuning |
| Complex APIs | Simple, readable code |
| Research-oriented | Educational + Production ready |

**Features:**
- Mixed precision training (AMP) for 2-3x speedup
- AdamW optimizer with cosine annealing + warmup
- Attention visualization for interpretability
- ONNX export for deployment
- CLI and Python API

## Installation

```bash
pip install vit-trainer
```

### Optional Dependencies

```bash
# Gradio web demo
pip install "vit-trainer[demo]"

# ONNX export
pip install "vit-trainer[export]"

# Everything
pip install "vit-trainer[all]"
```

### Install from Source

```bash
git clone https://github.com/jman4162/PyTorch-Vision-Transformers-ViT.git
cd PyTorch-Vision-Transformers-ViT
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from vit_trainer import Trainer, load_model, get_cifar10_loaders

# Load data and model
train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)
model = load_model("vit_b_16", num_classes=10)

# Train
trainer = Trainer(model, lr=1e-4, use_amp=True)
history = trainer.fit(train_loader, val_loader, epochs=10)

# Evaluate
loss, accuracy = trainer.evaluate(test_loader)
print(f"Test Accuracy: {accuracy:.2f}%")
```

### Command Line Interface

```bash
# Train a model
vit-train train --model vit_b_16 --dataset cifar10 --epochs 10

# Evaluate a trained model
vit-train eval --checkpoint best_model.pt --dataset cifar10 --plot-confusion

# Predict on a single image
vit-train predict --checkpoint best_model.pt --image cat.jpg --show-attention

# Export to ONNX
vit-train export --checkpoint best_model.pt --output model.onnx
```

### Configuration Files

```bash
# Use YAML config
vit-train train --config configs/default.yaml
```

## Usage Examples

### Training with Custom Settings

```python
from vit_trainer import Trainer, load_model, get_cifar10_loaders, TrainingConfig

# Create config
config = TrainingConfig(
    model_variant="vit_b_16",
    batch_size=64,
    epochs=10,
    lr=1e-4,
    weight_decay=0.05,
    warmup_epochs=2,
    patience=3,
    use_amp=True,
)

# Train
train_loader, val_loader, _ = get_cifar10_loaders(batch_size=config.batch_size)
model = load_model(config.model_variant, num_classes=10)
trainer = Trainer(
    model,
    lr=config.lr,
    weight_decay=config.weight_decay,
    warmup_epochs=config.warmup_epochs,
    use_amp=config.use_amp,
)
trainer.fit(train_loader, val_loader, epochs=config.epochs, patience=config.patience)
```

### Attention Visualization

```python
from vit_trainer import visualize_samples_with_attention, CIFAR10_CLASSES

visualize_samples_with_attention(
    model,
    test_loader.dataset,
    CIFAR10_CLASSES,
    num_samples=4,
)
```

### Evaluation Metrics

```python
from vit_trainer import get_predictions, compute_metrics, plot_confusion_matrix

y_pred, y_true, probs = get_predictions(model, test_loader)
metrics = compute_metrics(y_true, y_pred, CIFAR10_CLASSES)

print(metrics["classification_report"])
plot_confusion_matrix(y_true, y_pred, CIFAR10_CLASSES)
```

### Loading Trained Models

```python
from vit_trainer import load_model

# Load from checkpoint
model = load_model(
    "vit_b_16",
    num_classes=10,
    checkpoint_path="best_model.pt",
)
```

### ONNX Export

```python
from vit_trainer import load_model, ExportConfig

# Load trained model
model = load_model("vit_b_16", num_classes=10, checkpoint_path="best_model.pt")

# Export to ONNX
config = ExportConfig(output_path="model.onnx", opset_version=14)
config.export(model)

# Or use CLI
# vit-train export --checkpoint best_model.pt --output model.onnx
```

## API Reference

```python
from vit_trainer import (
    # Configuration
    TrainingConfig,           # Training hyperparameters
    ExportConfig,             # ONNX export settings

    # Models
    load_model,               # Load ViT with pretrained weights
    VIT_VARIANTS,             # Available model variants

    # Data
    get_cifar10_loaders,      # CIFAR-10 data loaders
    get_cifar100_loaders,     # CIFAR-100 data loaders
    CIFAR10_CLASSES,          # Class names

    # Training
    Trainer,                  # Training loop with AMP
    EarlyStopping,            # Early stopping callback
    ModelCheckpoint,          # Save best model

    # Evaluation
    evaluate_model,           # Loss and accuracy
    compute_metrics,          # Precision, recall, F1
    plot_confusion_matrix,    # Visualization

    # Visualization
    visualize_attention,      # Attention heatmaps
)
```

## Project Structure

```
vit-trainer/
├── vit_trainer/
│   ├── __init__.py         # Public API
│   ├── config.py           # TrainingConfig dataclass
│   ├── cli.py              # Command-line interface
│   ├── data/               # Data loaders and transforms
│   ├── models/             # Model registry and factory
│   ├── training/           # Trainer and callbacks
│   ├── evaluation/         # Metrics and plotting
│   └── visualization/      # Attention maps
├── tests/                  # Unit tests (44 tests)
├── configs/                # YAML configurations
├── notebooks/              # Tutorial notebooks
├── app.py                  # Gradio demo
└── pyproject.toml          # Package configuration
```

## ViT Variants

| Variant | Patch Size | Parameters | ImageNet Acc | Use Case |
|---------|------------|------------|--------------|----------|
| `vit_b_16` | 16x16 | 86M | 81.1% | Best accuracy/speed |
| `vit_b_32` | 32x32 | 88M | 75.9% | Faster inference |
| `vit_l_16` | 16x16 | 304M | 79.7% | Higher accuracy |

## Training Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.65% |
| **Model** | vit_b_16 |
| **Training Time** | ~11 min/epoch (GPU) |

## Gradio Demo

```bash
# Launch interactive web interface
python app.py
# Opens at http://localhost:7860
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black vit_trainer/
ruff check vit_trainer/

# Type check
mypy vit_trainer/
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 32` or `16`
- AMP is enabled by default

### Slow Training on CPU
- Use Google Colab (free GPU)
- Training on CPU is very slow (~60 min/epoch)

### Import Errors
- Make sure to install the package: `pip install vit-trainer`

## Resources

- [Original ViT Paper](https://arxiv.org/abs/2010.11929)
- [PyTorch ViT Documentation](https://pytorch.org/vision/main/models/vision_transformer.html)
- [Hugging Face ViT](https://huggingface.co/docs/transformers/en/model_doc/vit)
- [CIFAR-10 SOTA](https://paperswithcode.com/sota/image-classification-on-cifar-10)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
