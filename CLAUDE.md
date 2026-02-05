# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational tutorial demonstrating how to fine-tune the Vision Transformer (ViT) model (`vit_b_16`) for image classification using PyTorch. The project achieves 97.65% accuracy on CIFAR-10.

## Running the Project

This is a Jupyter notebook tutorial that works in both Google Colab and local environments:

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch torchvision torchsummary tqdm scikit-learn seaborn matplotlib numpy Pillow

# Then open the notebook in Jupyter/JupyterLab or Google Colab and execute cells sequentially
```

**Note:** GPU strongly recommended (~11 minutes per epoch on GPU).

## Architecture

**Data Pipeline:**
- CIFAR-10 dataset (60,000 images, 10 classes)
- Images resized to 224x224 (ViT input size), normalized with ImageNet statistics
- Data augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter
- Train/Val/Test split: 40,000 / 10,000 / 10,000

**Model:**
- Pre-trained `vit_b_16` from torchvision using `ViT_B_16_Weights.IMAGENET1K_V1`
- Classification head replaced: 1000 classes → 10 classes for CIFAR-10

**Training Configuration:**
- Optimizer: Adam (lr=3e-5)
- Scheduler: StepLR (gamma=0.7 per epoch)
- Loss: CrossEntropyLoss
- Batch size: 64, Epochs: 10 (max)
- Early stopping: patience=3 epochs

**Evaluation:**
- Accuracy metrics
- Classification report (precision, recall, F1 per class)
- Confusion matrix visualization

## Key Files

- `Fine_tuning_Vision_Transformers_ViT_with_PyTorch.ipynb` - Complete tutorial with code, explanations, and outputs
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore patterns

## Environment Notes

- The notebook auto-detects Colab vs local environment
- In Colab: mounts Google Drive, saves to `/content/drive/MyDrive/ViT_models/`
- Locally: saves to `./models/` directory
- Python 3.8+ required with CUDA recommended
