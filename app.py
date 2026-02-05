"""
Gradio Web Interface for ViT Image Classifier

Run with: python app.py

This creates an interactive web interface for classifying images
using a fine-tuned Vision Transformer model.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import gradio as gr
from PIL import Image
import numpy as np

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform (same as validation transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path="best_model_vit_b_16_cifar10.pt"):
    """Load the fine-tuned ViT model."""
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, 10)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Using pretrained weights only.")

    model.to(device)
    model.eval()
    return model


# Load model globally
model = load_model()


def predict(image):
    """
    Predict class for an uploaded image.

    Args:
        image: PIL Image from Gradio

    Returns:
        Dictionary of class probabilities
    """
    if image is None:
        return {cls: 0.0 for cls in CIFAR10_CLASSES}

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Return as dictionary
    return {CIFAR10_CLASSES[i]: float(probs[i]) for i in range(10)}


def predict_with_attention(image):
    """
    Predict class and show attention visualization.

    Args:
        image: PIL Image from Gradio

    Returns:
        Tuple of (predictions dict, attention overlay image)
    """
    if image is None:
        return {cls: 0.0 for cls in CIFAR10_CLASSES}, None

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    predictions = {CIFAR10_CLASSES[i]: float(probs[i]) for i in range(10)}

    # Get attention (simplified - just return resized image for now)
    # Full attention extraction requires more complex hook setup
    return predictions, image.resize((224, 224))


# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="Vision Transformer Image Classifier",
    description="""
    Upload an image to classify it using a fine-tuned Vision Transformer (ViT).

    **Model**: vit_b_16 fine-tuned on CIFAR-10

    **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

    **Note**: This model was trained on 32x32 CIFAR-10 images. For best results,
    use images of single objects similar to the training data.
    """,
    examples=[],
    theme="default",
    allow_flagging="never"
)


if __name__ == "__main__":
    print(f"Running on device: {device}")
    print("Starting Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
