# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational tutorial demonstrating how to fine-tune the Vision Transformer (ViT) model (`vit_b_16`) for object recognition using PyTorch. The project achieves 97.65% accuracy on CIFAR-10.

## Running the Project

This is a Jupyter notebook tutorial designed for Google Colab:

```bash
# Install dependencies
pip install torch torchvision torchsummary tqdm

# Then open the notebook in Jupyter/JupyterLab or Google Colab and execute cells sequentially
```

**Note:** GPU strongly recommended (~11 minutes per epoch on GPU).

## Architecture

**Data Pipeline:**
- CIFAR-10 dataset (60,000 images, 10 classes)
- Images resized to 224x224 (ViT input size), normalized with ImageNet statistics
- Train/Val/Test split: 40,000 / 10,000 / 10,000

**Model:**
- Pre-trained `vit_b_16` from torchvision (ImageNet weights)
- Classification head replaced: 1000 classes → 10 classes for CIFAR-10

**Training Configuration:**
- Optimizer: Adam (lr=3e-5)
- Scheduler: StepLR (gamma=0.7 per epoch)
- Loss: CrossEntropyLoss
- Batch size: 64, Epochs: 10

## Key File

- `Introduction_to_Fine_tuning_Vision_Transformers_(ViT)_for_Robotics_Applications_with_PyTorch.ipynb` - Complete tutorial with code, explanations, and outputs

## Environment Notes

- The notebook contains Colab-specific code (Google Drive mounting for model checkpoints)
- Model saves to `/content/drive/MyDrive/ViT_models/` (Colab path)
- Python 3.8+ required with CUDA recommended
