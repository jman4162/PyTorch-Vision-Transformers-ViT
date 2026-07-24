# vit-trainer

[![PyPI](https://img.shields.io/pypi/v/vit-trainer.svg)](https://pypi.org/project/vit-trainer/)
[![CI](https://github.com/jman4162/PyTorch-Vision-Transformers-ViT/actions/workflows/ci.yml/badge.svg)](https://github.com/jman4162/PyTorch-Vision-Transformers-ViT/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/PyTorch-Vision-Transformers-ViT/blob/main/notebooks/tutorial.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An educational, production-minded package for fine-tuning Vision Transformer (ViT)
models in PyTorch. The training loop is short enough to read in one sitting.

![ViT](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg)

## Why vit-trainer?

Not capability — `timm` and `transformers` cover far more. The difference is
legibility: the model factory, training loop, scheduler, data split, evaluation,
and export path are each one small file you can read end to end.

| vs. timm/transformers | vit-trainer |
|-----------------------|-------------|
| 1000+ model architectures | Three ViT variants |
| Deep, configurable APIs | One `Trainer`, one `TrainingConfig` |
| Built for research throughput | Built to be read |

**Features:**
- Mixed precision training (AMP)
- AdamW with linear warmup and cosine decay
- Attention visualization
- ONNX and TorchScript export, verified against PyTorch logits
- Run manifests recording commit, environment, hardware, seed, and data split
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

# Load data and model. Pass a seed: without one the train/val split is drawn
# fresh each run and your results are not comparable to each other.
train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64, seed=42)
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

# Evaluate a trained model (variant and dataset are read from the checkpoint)
vit-train eval --checkpoint models/best_model_vit_b_16_cifar10.pt --plot-confusion

# Predict on a single image
vit-train predict --checkpoint best_model.pt --image cat.jpg --show-attention

# Export to ONNX and check the runtime reproduces PyTorch's logits
vit-train export --checkpoint best_model.pt --output model.onnx
```

### Configuration Files

```bash
# Use YAML config
vit-train train --config configs/default.yaml

# Flags you type override the YAML; everything else comes from the file
vit-train train --config configs/default.yaml --batch-size 32 --no-amp
```

Every field in `TrainingConfig` is passed to the component that uses it. If you
set `num_workers` in a config, it reaches the DataLoader.

### Run Manifests

Each training run writes `{model_dir}/{model_name}.run.json` next to the
checkpoint, recording the git commit, package version, Python/PyTorch/CUDA
versions, GPU name, seed, resolved config, a hash of the validation split, best
epoch, wall-clock time, and peak GPU memory. Quote results with this file — an
accuracy on its own is not reproducible.

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
train_loader, val_loader, _ = get_cifar10_loaders(
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    train_split=config.train_split,
    seed=config.seed,
)
model = load_model(config.model_variant, num_classes=config.num_classes)
trainer = Trainer(
    model,
    lr=config.lr,
    weight_decay=config.weight_decay,
    warmup_epochs=config.warmup_epochs,
    use_amp=config.use_amp,
    gradient_clip=config.gradient_clip,
    metadata={"config": config.to_dict()},
)
trainer.fit(train_loader, val_loader, epochs=config.epochs, patience=config.patience)
```

`num_classes` follows the dataset unless you set it: `TrainingConfig(dataset="cifar100")`
resolves to 100 classes, and an explicit value that contradicts the dataset raises.

### Attention Visualization

```python
from vit_trainer import visualize_samples_with_attention, CIFAR10_CLASSES

visualize_samples_with_attention(
    model,
    test_loader.dataset,
    CIFAR10_CLASSES,
    num_samples=4,
    seed=0,
)
```

This shows which patches the final-layer CLS token pooled from. Treat it as a
diagnostic, not an explanation of the prediction — attention weights and feature
importance are [not the same thing](https://arxiv.org/abs/1902.10186).

`forward_with_attention` exposes the underlying pass and returns logits as well
as attention, so you can assert the captured attention comes from the same
forward pass the model uses for prediction:

```python
from vit_trainer import forward_with_attention

attn, logits = forward_with_attention(model, images)
assert torch.allclose(logits, model(images), atol=1e-4)
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

# A valid graph is not a correct one. Run both and compare the numbers.
results = config.verify(model)          # raises if logits or argmax disagree
print(results["batches"][1]["max_logit_delta"])

# Or use CLI (verification runs by default)
# vit-train export --checkpoint best_model.pt --output model.onnx
```

`verify` runs PyTorch and ONNX Runtime on identical inputs at batch sizes 1 and
4, then asserts the predicted classes match and the logits agree within `atol`.
On a pretrained ViT-B/16 the observed maximum logit deviation is around 3e-06.
`format="torchscript"` exports a scripted module and checks it the same way.

## API Reference

```python
from vit_trainer import (
    # Configuration
    TrainingConfig,           # Training hyperparameters
    ExportConfig,             # Export settings, export(), verify()

    # Models
    load_model,               # Load ViT with pretrained weights
    load_state_dict,          # Read weights from either checkpoint layout
    read_checkpoint_metadata, # Config and metrics stored with the weights
    freeze_backbone,          # Linear probing
    VIT_VARIANTS,             # Available model variants

    # Data
    get_cifar10_loaders,      # CIFAR-10 data loaders
    get_cifar100_loaders,     # CIFAR-100 data loaders
    make_split_indices,       # Deterministic train/val split
    CIFAR10_CLASSES,          # Class names

    # Training
    Trainer,                  # Training loop with AMP
    EarlyStopping,            # Early stopping callback
    ModelCheckpoint,          # Save best model

    # Evaluation
    evaluate_model,           # Loss and accuracy
    compute_metrics,          # Precision, recall, F1
    plot_confusion_matrix,    # Visualization
    plot_training_history,    # Loss and LR curves

    # Provenance
    collect_run_metadata,     # Commit, environment, hardware, config
    save_run_metadata,        # Write run.json

    # Visualization
    visualize_attention,      # CLS attention map
    forward_with_attention,   # Attention plus logits, for verification
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
├── tests/                  # Unit tests (pytest)
├── configs/                # YAML configurations
├── notebooks/              # Tutorial notebooks
├── app.py                  # Gradio demo
└── pyproject.toml          # Package configuration
```

## ViT Variants

ImageNet top-1 is torchvision's published figure for the `IMAGENET1K_V1` weights
this package loads.

| Variant | Patch Size | Parameters | ImageNet top-1 | Notes |
|---------|------------|------------|----------------|-------|
| `vit_b_16` | 16x16 | 86M | 81.1% | Default; best accuracy per unit of compute |
| `vit_b_32` | 32x32 | 88M | 75.9% | 4x fewer tokens, so much faster |
| `vit_l_16` | 16x16 | 304M | 79.7% | 3.5x the parameters and *lower* top-1 |

ViT-L/16 scoring below ViT-B/16 is not a typo. Torchvision trained its L/16
`IMAGENET1K_V1` weights from scratch on ImageNet-1k, where a model that large
overfits without the larger pretraining corpus the original paper used. If you
want a stronger starting point than B/16, use the SWAG weights
(`ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1`, 88.1% at 512px) rather than the V1
weights loaded here.

## Training Results

One run, before the package refactor:

| | |
|--------|-------|
| **Test accuracy** | 97.65% (9,765 / 10,000 CIFAR-10 test images) |
| **Model** | `vit_b_16`, ImageNet-1k V1 weights, all layers fine-tuned |
| **Recipe** | 10 epochs max, batch 64, AdamW lr 1e-4, wd 0.05, 2-epoch warmup, seed 42 |
| **Environment** | torch 2.2.1+cu121, Colab GPU (model not recorded) |
| **Time** | ~11 min/epoch |
| **Source** | `notebooks/Fine_tuning_Vision_Transformers_ViT_with_PyTorch.ipynb`, cell output retained |

Two caveats worth stating plainly:

1. That number came from the standalone notebook this package was extracted
   from, not from `vit_trainer` itself. It has not been reproduced through the
   package API.
2. It is a single run. There is no seed-to-seed variance estimate, and the GPU
   model was never recorded — which is exactly why runs now write a manifest.

CIFAR-10 images are 32x32 and get resized to 224x224 to match the pretrained
model's input. Upsampling satisfies the interface; it does not add detail. This
makes CIFAR-10 a convenient transfer-learning demo rather than a searching test
of what a ViT can do.

## Gradio Demo

```bash
# Launch interactive web interface
python app.py
# Opens at http://localhost:7860
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,export]"

# Run tests
pytest tests/
pytest tests/ -m "not slow"    # skip full-size ViT load/export

# Format code
black vit_trainer/
ruff check vit_trainer/

# Type check
mypy vit_trainer/
```

## What this is not

- Not a replacement for `timm`, Lightning, or Accelerate. Those are better at
  almost everything except being short.
- Not resumable. Checkpoints store weights, config, epoch, and metrics, but not
  optimizer or AMP scaler state — AdamW moments for 86M parameters would put
  every checkpoint over 1 GB. An interrupted run restarts.
- Not dataset-agnostic. CIFAR-10 and CIFAR-100 only. `imagefolder` used to be
  accepted by config validation and silently loaded CIFAR-100; it is now
  rejected until it is actually implemented.
- No distributed training, structured experiment logging, or hyperparameter
  search.
- Not evidence that a vanilla ViT is the right model for your small image
  dataset. Compare against a CNN baseline before assuming it is.

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 32` or `16`
- AMP is enabled by default

### AMP is not making training faster
AMP's speedup depends on Tensor Core hardware, tensor shapes that keep them fed,
and enough work per step to hide the overhead. On CPU it is disabled outright
(`GradScaler` is only created for CUDA devices), and on small models or
input-bound loops the gain can be negligible. Measure it on your setup rather
than assuming a multiplier.

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
- [Attention is not Explanation](https://arxiv.org/abs/1902.10186)
- [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
