# Fine-tuning Vision Transformers (ViT) with PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/PyTorch-Vision-Transformers-ViT/blob/main/Fine_tuning_Vision_Transformers_ViT_with_PyTorch.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive tutorial demonstrating how to fine-tune Vision Transformer (ViT) models for image classification using PyTorch. This tutorial achieves **97.65% accuracy** on CIFAR-10.

![ViT](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Tutorial Structure](#tutorial-structure)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)
- [Contributing](#contributing)
- [License](#license)

## Overview

Vision Transformers have emerged as a powerful class of models in computer vision, rivaling traditional CNNs in various tasks. This tutorial demonstrates how to leverage the pretrained `vit_b_16` model from torchvision and fine-tune it for the CIFAR-10 dataset.

## Features

- **Modern PyTorch API**: Uses the current `weights` parameter instead of deprecated `pretrained=True`
- **Data Augmentation**: Includes random flips, rotations, and color jitter for better generalization
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Progress Tracking**: Uses tqdm for batch-level progress bars
- **Comprehensive Evaluation**: Includes classification report and confusion matrix visualization
- **Environment Agnostic**: Works in both Google Colab and local environments

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

## Installation

### Option 1: Google Colab (Recommended for beginners)

Click the "Open in Colab" badge above and run the notebook directly. No local setup required!

### Option 2: Local Installation

1. Clone the repository:
```bash
git clone https://github.com/jman4162/PyTorch-Vision-Transformers-ViT.git
cd PyTorch-Vision-Transformers-ViT
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter and open the notebook:
```bash
jupyter notebook Fine_tuning_Vision_Transformers_ViT_with_PyTorch.ipynb
```

## Usage

Simply run the notebook cells sequentially. The notebook will:

1. Install required packages
2. Download the CIFAR-10 dataset automatically
3. Load the pretrained ViT model
4. Fine-tune on CIFAR-10
5. Evaluate and visualize results

## Tutorial Structure

| Section | Description |
|---------|-------------|
| **Setup** | Install dependencies, import libraries, set random seeds |
| **Data Preparation** | Load CIFAR-10, apply transforms, create DataLoaders |
| **Model Setup** | Load pretrained ViT, modify classification head |
| **Training** | Train with early stopping, save best model |
| **Evaluation** | Accuracy, classification report, confusion matrix |

## Results

Our fine-tuned model achieves:

- **Test Accuracy**: 97.65%
- **Training Time**: ~11 minutes per epoch on GPU

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Learning Rate | 3e-5 |
| LR Scheduler | StepLR (gamma=0.7) |
| Optimizer | Adam |
| Early Stopping | patience=3 |

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size from 64 to 32 or 16
- Use mixed precision training with `torch.cuda.amp`

**Slow Training on CPU**
- Training on CPU is very slow. Use Google Colab (free GPU) or a local GPU

**ModuleNotFoundError**
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**Model Not Saving**
- Check that the `models/` directory exists or is created automatically
- In Colab, ensure Google Drive is mounted if saving to Drive

## Additional Resources

- [Original ViT Paper](https://arxiv.org/abs/2010.11929)
- [Hugging Face ViT](https://huggingface.co/docs/transformers/en/model_doc/vit)
- [PyTorch ViT Documentation](https://pytorch.org/vision/main/models/vision_transformer.html)
- [D2L AI - Attention Mechanisms](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
- [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
