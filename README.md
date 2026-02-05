# Fine-tuning Vision Transformers (ViT) with PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/PyTorch-Vision-Transformers-ViT/blob/main/Fine_tuning_Vision_Transformers_ViT_with_PyTorch.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready tutorial for fine-tuning Vision Transformer (ViT) models using PyTorch. Achieves **97.65% accuracy** on CIFAR-10 with modern training techniques.

![ViT](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg)

## Features

| Feature | Description |
|---------|-------------|
| **Mixed Precision Training (AMP)** | 2-3x speedup with FP16 |
| **AdamW + Cosine Annealing** | Modern optimizer with warmup |
| **Multiple ViT Variants** | vit_b_16, vit_b_32, vit_l_16 |
| **Attention Visualization** | See what the model focuses on |
| **ONNX Export** | Production deployment ready |
| **Gradio Demo** | Interactive web interface |
| **Proper Validation** | Fixed common random_split bug |

## Quick Start

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge above - no setup required!

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/jman4162/PyTorch-Vision-Transformers-ViT.git
cd PyTorch-Vision-Transformers-ViT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Fine_tuning_Vision_Transformers_ViT_with_PyTorch.ipynb
```

### Option 3: Run the Gradio Demo

```bash
# After training or with pretrained weights
python app.py
# Opens at http://localhost:7860
```

## Tutorial Structure

| Section | Description |
|---------|-------------|
| **Setup** | Install dependencies, configure environment |
| **Data Preparation** | CIFAR-10 with proper train/val splits |
| **Model Setup** | Load ViT variants, configure for 10 classes |
| **Training** | AMP, warmup, cosine annealing, early stopping |
| **Evaluation** | Accuracy, confusion matrix, classification report |
| **Visualization** | Attention maps, misclassified examples |
| **Deployment** | ONNX export, benchmarking, Gradio demo |

## Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.65% |
| **Model** | vit_b_16 |
| **Parameters** | 86M |
| **Training** | ~11 min/epoch (GPU) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 0.05 |
| Scheduler | Cosine + 2-epoch warmup |
| Batch Size | 64 |
| Mixed Precision | Enabled |
| Early Stopping | patience=3 |

## ViT Variants

| Variant | Patch Size | Parameters | Use Case |
|---------|------------|------------|----------|
| `vit_b_16` | 16x16 | 86M | Best accuracy/speed balance |
| `vit_b_32` | 32x32 | 88M | Faster, lower accuracy |
| `vit_l_16` | 16x16 | 304M | Higher accuracy, more memory |

## Key Files

| File | Description |
|------|-------------|
| `Fine_tuning_Vision_Transformers_ViT_with_PyTorch.ipynb` | Main tutorial notebook |
| `app.py` | Standalone Gradio web demo |
| `requirements.txt` | Python dependencies |
| `CLAUDE.md` | AI assistant guidance |

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `batch_size = 32` or `16`
- Mixed precision is enabled by default

### Slow Training on CPU
- Use Google Colab (free GPU)
- Training on CPU is very slow

### Model Not Saving
- Check `models/` directory exists
- In Colab, ensure Drive is mounted

### ONNX Export Fails
- Ensure model is on CPU before export
- Use opset_version >= 14

## Additional Resources

- [Original ViT Paper](https://arxiv.org/abs/2010.11929)
- [Hugging Face ViT](https://huggingface.co/docs/transformers/en/model_doc/vit)
- [PyTorch ViT Documentation](https://pytorch.org/vision/main/models/vision_transformer.html)
- [D2L AI - Attention Mechanisms](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
- [CIFAR-10 SOTA](https://paperswithcode.com/sota/image-classification-on-cifar-10)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
