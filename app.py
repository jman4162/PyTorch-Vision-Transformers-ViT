"""Gradio Web Interface for ViT Image Classifier.

Run with: python app.py
Opens at: http://localhost:7860

This creates an interactive web interface for classifying images
using a fine-tuned Vision Transformer model.
"""

import torch
import gradio as gr
from PIL import Image

from vit_trainer import (
    load_model,
    CIFAR10_CLASSES,
    get_val_transform,
    visualize_attention,
    show_attention_on_image,
)

# Configuration
MODEL_VARIANT = "vit_b_16"
MODEL_PATH = "best_model_vit_b_16_cifar10.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
try:
    model = load_model(
        MODEL_VARIANT,
        num_classes=10,
        checkpoint_path=MODEL_PATH,
        device=DEVICE,
    )
    print(f"Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    model = load_model(MODEL_VARIANT, num_classes=10, device=DEVICE)
    print(f"Warning: {MODEL_PATH} not found. Using pretrained weights only.")

model.eval()

# Image transform
transform = get_val_transform(image_size=224)


def predict(image: Image.Image) -> dict:
    """Predict class for an uploaded image.

    Args:
        image: PIL Image from Gradio

    Returns:
        Dictionary of class probabilities
    """
    if image is None:
        return {cls: 0.0 for cls in CIFAR10_CLASSES}

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Return as dictionary
    return {CIFAR10_CLASSES[i]: float(probs[i]) for i in range(10)}


def predict_with_attention(image: Image.Image):
    """Predict class and show attention visualization.

    Args:
        image: PIL Image from Gradio

    Returns:
        Tuple of (predictions dict, attention overlay image)
    """
    if image is None:
        return {cls: 0.0 for cls in CIFAR10_CLASSES}, None

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    predictions = {CIFAR10_CLASSES[i]: float(probs[i]) for i in range(10)}

    # Get attention map
    attn_map = visualize_attention(model, input_tensor[0], device=DEVICE)
    overlay = show_attention_on_image(image.resize((224, 224)), attn_map)

    return predictions, Image.fromarray(overlay)


# Create Gradio interface
with gr.Blocks(title="ViT Image Classifier") as demo:
    gr.Markdown(
        """
        # Vision Transformer Image Classifier

        Upload an image to classify it using a fine-tuned Vision Transformer (ViT).

        **Model**: vit_b_16 fine-tuned on CIFAR-10

        **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

        **Note**: This model was trained on 32x32 CIFAR-10 images upscaled to 224x224.
        For best results, use images of single objects similar to the training data.

        The attention overlay shows which patches the final-layer CLS token pooled
        from. That is a diagnostic, not an explanation of the prediction.
        """
    )

    with gr.Tab("Simple Classification"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload an Image")
                classify_btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                output_label = gr.Label(num_top_classes=5, label="Predictions")

        classify_btn.click(predict, inputs=input_image, outputs=output_label)
        input_image.change(predict, inputs=input_image, outputs=output_label)

    with gr.Tab("With Attention Visualization"):
        with gr.Row():
            with gr.Column():
                input_image_attn = gr.Image(type="pil", label="Upload an Image")
                classify_attn_btn = gr.Button(
                    "Classify with Attention", variant="primary"
                )
            with gr.Column():
                output_label_attn = gr.Label(num_top_classes=5, label="Predictions")
                output_attention = gr.Image(type="pil", label="Attention Overlay")

        classify_attn_btn.click(
            predict_with_attention,
            inputs=input_image_attn,
            outputs=[output_label_attn, output_attention],
        )

    gr.Markdown(
        """
        ---
        Built with [vit-trainer](https://github.com/jman4162/PyTorch-Vision-Transformers-ViT)
        """
    )


if __name__ == "__main__":
    print(f"Running on device: {DEVICE}")
    print("Starting Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
