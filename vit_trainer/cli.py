"""Command-line interface for vit-trainer."""

import argparse
import sys


def get_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="vit-trainer",
        description="Train and evaluate Vision Transformer models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a ViT model")
    train_parser.add_argument(
        "--model",
        type=str,
        default="vit_b_16",
        choices=["vit_b_16", "vit_b_32", "vit_l_16"],
        help="ViT model variant",
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset to use",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size"
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    train_parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="Weight decay"
    )
    train_parser.add_argument(
        "--warmup-epochs", type=int, default=2, help="Warmup epochs"
    )
    train_parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience"
    )
    train_parser.add_argument(
        "--no-amp", action="store_true", help="Disable mixed precision training"
    )
    train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    train_parser.add_argument(
        "--data-dir", type=str, default="./data", help="Data directory"
    )
    train_parser.add_argument(
        "--model-dir", type=str, default="./models", help="Model save directory"
    )
    train_parser.add_argument(
        "--config", type=str, help="Path to YAML config file"
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        default="vit_b_16",
        choices=["vit_b_16", "vit_b_32", "vit_l_16"],
        help="ViT model variant",
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset to use",
    )
    eval_parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size"
    )
    eval_parser.add_argument(
        "--data-dir", type=str, default="./data", help="Data directory"
    )
    eval_parser.add_argument(
        "--plot-confusion", action="store_true", help="Plot confusion matrix"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict on single image")
    predict_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    predict_parser.add_argument(
        "--image", type=str, required=True, help="Path to image file"
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        default="vit_b_16",
        choices=["vit_b_16", "vit_b_32", "vit_l_16"],
        help="ViT model variant",
    )
    predict_parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset (for class names)",
    )
    predict_parser.add_argument(
        "--top-k", type=int, default=5, help="Show top-k predictions"
    )
    predict_parser.add_argument(
        "--show-attention", action="store_true", help="Show attention visualization"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    export_parser.add_argument(
        "--model",
        type=str,
        default="vit_b_16",
        choices=["vit_b_16", "vit_b_32", "vit_l_16"],
        help="ViT model variant",
    )
    export_parser.add_argument(
        "--output", type=str, default="model.onnx", help="Output ONNX file"
    )
    export_parser.add_argument(
        "--num-classes", type=int, default=10, help="Number of classes"
    )

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    """Run training command."""
    import random

    import numpy as np
    import torch

    from .config import TrainingConfig
    from .data import get_cifar10_loaders, get_cifar100_loaders
    from .models import load_model
    from .training import Trainer

    # Load config from file or args
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig(
            model_variant=args.model,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            use_amp=not args.no_amp,
            seed=args.seed,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
        )

    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("Training Configuration:")
    print(f"  Model: {config.model_variant}")
    print(f"  Dataset: {config.dataset}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  AMP: {config.use_amp}")
    print()

    # Get data loaders
    num_classes = 10 if config.dataset == "cifar10" else 100
    if config.dataset == "cifar10":
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size=config.batch_size,
            data_dir=config.data_dir,
            seed=config.seed,
        )
    else:
        train_loader, val_loader, test_loader = get_cifar100_loaders(
            batch_size=config.batch_size,
            data_dir=config.data_dir,
            seed=config.seed,
        )

    print(f"Data loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    # Load model
    model = load_model(config.model_variant, num_classes=num_classes)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = Trainer(
        model=model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        use_amp=config.use_amp,
        model_dir=config.model_dir,
    )

    # Train
    trainer.fit(
        train_loader,
        val_loader,
        epochs=config.epochs,
        patience=config.patience,
        model_name=f"best_model_{config.model_variant}_{config.dataset}",
    )

    # Evaluate on test set
    print("\nFinal Evaluation:")
    trainer.evaluate(test_loader)

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run evaluation command."""
    from .data import (
        CIFAR10_CLASSES,
        CIFAR100_CLASSES,
        get_cifar10_loaders,
        get_cifar100_loaders,
    )
    from .evaluation import (
        evaluate_model,
        get_predictions,
        plot_confusion_matrix,
        print_classification_report,
    )
    from .models import load_model

    # Get data
    num_classes = 10 if args.dataset == "cifar10" else 100
    class_names = CIFAR10_CLASSES if args.dataset == "cifar10" else CIFAR100_CLASSES

    if args.dataset == "cifar10":
        _, _, test_loader = get_cifar10_loaders(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
        )
    else:
        _, _, test_loader = get_cifar100_loaders(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
        )

    # Load model
    model = load_model(
        args.model,
        num_classes=num_classes,
        checkpoint_path=args.checkpoint,
    )

    # Evaluate
    loss, accuracy = evaluate_model(model, test_loader)
    print(f"\nTest Loss: {loss:.6f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Classification report
    y_pred, y_true, _ = get_predictions(model, test_loader)
    print()
    print_classification_report(y_true, y_pred, class_names)

    # Confusion matrix
    if args.plot_confusion:
        plot_confusion_matrix(y_true, y_pred, class_names)

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Run prediction command."""
    import torch
    from PIL import Image

    from .data import CIFAR10_CLASSES, CIFAR100_CLASSES
    from .data.transforms import get_val_transform
    from .models import load_model

    # Get class names
    num_classes = 10 if args.dataset == "cifar10" else 100
    class_names = CIFAR10_CLASSES if args.dataset == "cifar10" else CIFAR100_CLASSES

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        args.model,
        num_classes=num_classes,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    model.eval()

    # Load and preprocess image
    image = Image.open(args.image).convert("RGB")
    transform = get_val_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Show top-k predictions
    top_probs, top_indices = torch.topk(probs, args.top_k)

    print(f"\nPredictions for: {args.image}")
    print("-" * 40)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        print(f"{i+1}. {class_names[idx]}: {prob.item()*100:.2f}%")

    # Show attention if requested
    if args.show_attention:
        import matplotlib.pyplot as plt

        from .visualization import show_attention_on_image, visualize_attention

        attn_map = visualize_attention(model, input_tensor[0], device=device)
        if attn_map is not None:
            overlay = show_attention_on_image(image.resize((224, 224)), attn_map)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(image)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(attn_map, cmap="hot")
            axes[1].set_title("Attention Map")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title(f"Prediction: {class_names[top_indices[0]]}")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Run export command."""
    import torch
    import torch.onnx

    from .models import load_model

    # Load model
    model = load_model(
        args.model,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint,
    )
    model.eval()
    model.cpu()

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"Model exported to {args.output}")

    # Verify
    import onnx
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")

    # File size
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    return 0


def main() -> int:
    """Main entry point for CLI."""
    parser = get_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "train": cmd_train,
        "eval": cmd_eval,
        "predict": cmd_predict,
        "export": cmd_export,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
