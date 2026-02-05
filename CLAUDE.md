# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-ready tutorial demonstrating how to fine-tune Vision Transformer (ViT) models for image classification using PyTorch. Achieves 97.65% accuracy on CIFAR-10 with modern training techniques.

## Running the Project

### Main Tutorial (Jupyter Notebook)
```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook Fine_tuning_Vision_Transformers_ViT_with_PyTorch.ipynb
```

### Gradio Demo (Standalone)
```bash
python app.py
# Opens at http://localhost:7860
```

**Note:** GPU strongly recommended (~11 minutes per epoch on GPU).

## Architecture

### Data Pipeline
- CIFAR-10 dataset (60,000 images, 10 classes)
- Images resized to 224x224 (ViT input size), normalized with ImageNet statistics
- Data augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter
- Train/Val/Test split: 40,000 / 10,000 / 10,000
- **Fixed**: Uses `Subset` instead of `random_split` for proper validation transforms

### Model Variants
| Variant | Parameters | Use Case |
|---------|------------|----------|
| `vit_b_16` | 86M | Default - best accuracy/speed |
| `vit_b_32` | 88M | Faster inference |
| `vit_l_16` | 304M | Higher accuracy, more memory |

### Training Configuration
- Optimizer: AdamW (lr=1e-4, weight_decay=0.05)
- Scheduler: Cosine annealing with 2-epoch warmup
- Mixed Precision: AMP with GradScaler
- Gradient Clipping: max_norm=1.0
- Loss: CrossEntropyLoss
- Batch size: 64, Epochs: 10 (max)
- Early stopping: patience=3 epochs

### Evaluation & Visualization
- Accuracy, precision, recall, F1 per class
- Confusion matrix
- Attention map visualization
- Misclassified examples analysis

### Deployment
- ONNX export with dynamic batch size
- ONNX Runtime inference testing
- Inference benchmarking (PyTorch vs ONNX)
- Gradio web interface

## Key Files

| File | Description |
|------|-------------|
| `Fine_tuning_Vision_Transformers_ViT_with_PyTorch.ipynb` | Complete tutorial |
| `app.py` | Standalone Gradio demo |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore patterns |

## Environment Notes

- Auto-detects Colab vs local environment
- In Colab: mounts Google Drive, saves to `/content/drive/MyDrive/ViT_models/`
- Locally: saves to `./models/` directory
- Python 3.8+ required with CUDA recommended

## Key Implementation Details

### Validation Transform Fix
The tutorial uses `Subset` instead of `random_split` to ensure validation data doesn't get augmentation:
```python
train_dataset = Subset(
    datasets.CIFAR10(root='./data', train=True, transform=train_transform),
    train_indices
)
val_dataset = Subset(
    datasets.CIFAR10(root='./data', train=True, transform=val_transform),
    val_indices
)
```

### Mixed Precision Training
```python
scaler = GradScaler()
with autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Attention Visualization
The notebook includes functions to extract and visualize attention maps from ViT layers, showing which image patches the model focuses on for predictions.
