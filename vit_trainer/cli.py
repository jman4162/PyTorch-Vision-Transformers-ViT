"""Command-line interface for vit-trainer."""

import argparse
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import DATASET_CLASSES

MODEL_CHOICES = ["vit_b_16", "vit_b_32", "vit_l_16"]
DATASET_CHOICES = sorted(DATASET_CLASSES)

# Argument spec for `train`. The dest of every option matches a TrainingConfig
# field so the parsed values can be layered straight onto a YAML config.
TrainArg = Tuple[List[str], Dict[str, Any]]
TRAIN_ARGUMENTS: List[TrainArg] = [
    (
        ["--model"],
        {
            "dest": "model_variant",
            "choices": MODEL_CHOICES,
            "default": "vit_b_16",
            "help": "ViT model variant",
        },
    ),
    (
        ["--dataset"],
        {
            "choices": DATASET_CHOICES,
            "default": "cifar10",
            "help": "Dataset to use",
        },
    ),
    (
        ["--num-classes"],
        {"type": int, "help": "Number of classes (default: per dataset)"},
    ),
    (
        ["--epochs"],
        {"type": int, "default": 10, "help": "Number of training epochs"},
    ),
    (["--batch-size"], {"type": int, "default": 64, "help": "Batch size"}),
    (["--lr"], {"type": float, "default": 1e-4, "help": "Learning rate"}),
    (["--weight-decay"], {"type": float, "default": 0.05, "help": "Weight decay"}),
    (["--warmup-epochs"], {"type": int, "default": 2, "help": "Warmup epochs"}),
    (
        ["--patience"],
        {"type": int, "default": 3, "help": "Early stopping patience"},
    ),
    (
        ["--gradient-clip"],
        {"type": float, "default": 1.0, "help": "Max gradient norm"},
    ),
    (
        ["--train-split"],
        {"type": float, "default": 0.8, "help": "Train fraction of the split"},
    ),
    (["--image-size"], {"type": int, "default": 224, "help": "Input image size"}),
    (["--num-workers"], {"type": int, "default": 4, "help": "DataLoader workers"}),
    (["--seed"], {"type": int, "default": 42, "help": "Random seed"}),
    (["--data-dir"], {"default": "./data", "help": "Data directory"}),
    (["--model-dir"], {"default": "./models", "help": "Model save directory"}),
    (["--device"], {"default": "auto", "help": "Device: auto, cuda, cpu, mps"}),
    (
        ["--no-amp"],
        {
            "dest": "use_amp",
            "action": "store_false",
            "help": "Disable mixed precision",
        },
    ),
    (
        ["--no-augment"],
        {
            "dest": "augment_train",
            "action": "store_false",
            "help": "Disable training augmentation",
        },
    ),
    (
        ["--no-pin-memory"],
        {
            "dest": "pin_memory",
            "action": "store_false",
            "help": "Disable pinned memory",
        },
    ),
    (
        ["--compile"],
        {
            "dest": "compile_model",
            "action": "store_true",
            "help": "Use torch.compile",
        },
    ),
]


def _add_train_arguments(parser: argparse.ArgumentParser, use_defaults: bool) -> None:
    """Attach the training options.

    With `use_defaults=False` every option defaults to `argparse.SUPPRESS`, so
    parsing yields only the flags the user actually typed. That is what makes
    `--config file.yaml --batch-size 32` behave as expected instead of one
    silently discarding the other.
    """
    for flags, kwargs in TRAIN_ARGUMENTS:
        options = dict(kwargs)
        if not use_defaults:
            options.pop("default", None)
        parser.add_argument(*flags, **options)


def get_parser(use_defaults: bool = True) -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="vit-trainer",
        description="Train and evaluate Vision Transformer models",
        argument_default=None if use_defaults else argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a ViT model",
        argument_default=None if use_defaults else argparse.SUPPRESS,
    )
    _add_train_arguments(train_parser, use_defaults)
    train_parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_CHOICES,
        help="ViT variant (default: from the checkpoint, else vit_b_16)",
    )
    eval_parser.add_argument(
        "--dataset", type=str, choices=DATASET_CHOICES, help="Dataset to use"
    )
    eval_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    eval_parser.add_argument(
        "--data-dir", type=str, default="./data", help="Data directory"
    )
    eval_parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers"
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
        choices=MODEL_CHOICES,
        help="ViT variant (default: from the checkpoint, else vit_b_16)",
    )
    predict_parser.add_argument(
        "--dataset", type=str, choices=DATASET_CHOICES, help="Dataset (for class names)"
    )
    predict_parser.add_argument(
        "--top-k", type=int, default=5, help="Show top-k predictions"
    )
    predict_parser.add_argument(
        "--show-attention", action="store_true", help="Show attention visualization"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model for deployment")
    export_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    export_parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_CHOICES,
        help="ViT variant (default: from the checkpoint, else vit_b_16)",
    )
    export_parser.add_argument(
        "--output", type=str, default="model.onnx", help="Output file"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript"],
        help="Export format",
    )
    export_parser.add_argument(
        "--opset-version", type=int, default=14, help="ONNX opset version"
    )
    export_parser.add_argument(
        "--num-classes", type=int, help="Number of classes (default: from checkpoint)"
    )
    export_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the PyTorch vs runtime numerical check",
    )

    return parser


def resolve_train_config(argv: Optional[Sequence[str]] = None):
    """Build a TrainingConfig from dataclass defaults, YAML, and CLI flags.

    Precedence, lowest first: TrainingConfig defaults, then `--config` YAML,
    then options the user typed. Every resulting field is passed to the
    components in `cmd_train` — a config value that goes nowhere is a bug.

    Args:
        argv: Argument list; defaults to sys.argv[1:]

    Returns:
        The resolved TrainingConfig
    """
    from .config import TrainingConfig

    argv = list(sys.argv[1:] if argv is None else argv)

    args = get_parser().parse_args(argv)
    explicit = vars(get_parser(use_defaults=False).parse_args(argv))
    explicit.pop("config", None)
    explicit.pop("command", None)

    if getattr(args, "config", None):
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig.from_args(args)

    return config.merge(explicit)


def _resolve_device(name: str):
    """Turn a device string into a torch.device, resolving 'auto'."""
    import torch

    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_defaults(checkpoint_path: str) -> Dict[str, Any]:
    """Read model variant / dataset / classes recorded in a checkpoint."""
    from .models import read_checkpoint_metadata

    try:
        stored = read_checkpoint_metadata(checkpoint_path).get("metadata", {}) or {}
    except (OSError, RuntimeError):
        return {}

    return stored.get("config", {}) or {}


def _resolve_inference_args(args: argparse.Namespace) -> Tuple[str, str, int]:
    """Pick model variant, dataset, and class count for eval/predict/export.

    Explicit flags win; otherwise fall back to what the checkpoint recorded,
    then to the package defaults.
    """
    stored = _checkpoint_defaults(args.checkpoint)

    variant = getattr(args, "model", None) or stored.get("model_variant") or "vit_b_16"
    dataset = getattr(args, "dataset", None) or stored.get("dataset") or "cifar10"
    num_classes = getattr(args, "num_classes", None) or stored.get("num_classes")
    if num_classes is None:
        num_classes = DATASET_CLASSES[dataset]

    return variant, dataset, num_classes


def cmd_train(args: argparse.Namespace, argv: Optional[Sequence[str]] = None) -> int:
    """Run training command."""
    import random
    import time

    import numpy as np
    import torch

    from .data import get_cifar10_loaders, get_cifar100_loaders, get_class_names
    from .data.cifar import make_split_indices
    from .models import load_model
    from .provenance import collect_run_metadata, hash_indices, save_run_metadata
    from .training import Trainer

    config = resolve_train_config(argv)

    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = _resolve_device(config.device)

    print("Training Configuration:")
    for key, value in sorted(config.to_dict().items()):
        print(f"  {key}: {value}")
    print(f"  resolved device: {device}")
    print()

    loader_fn = (
        get_cifar10_loaders if config.dataset == "cifar10" else get_cifar100_loaders
    )
    train_loader, val_loader, test_loader = loader_fn(
        batch_size=config.batch_size,
        data_dir=config.data_dir,
        train_split=config.train_split,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
        image_size=config.image_size,
        augment_train=config.augment_train,
    )

    print(
        f"Data loaded: {len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test"
    )

    # Same call the loaders make, so the manifest records the actual split.
    _, val_indices = make_split_indices(
        50000, train_split=config.train_split, seed=config.seed
    )
    split_hash = hash_indices(val_indices)
    print(f"Validation split hash: {split_hash}")

    model = load_model(
        config.model_variant,
        num_classes=config.num_classes,
        compile_model=config.compile_model,
        device=device,
    )
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    metadata = {"config": config.to_dict(), "split_hash": split_hash}

    trainer = Trainer(
        model=model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        use_amp=config.use_amp,
        gradient_clip=config.gradient_clip,
        device=device,
        model_dir=config.model_dir,
        metadata=metadata,
    )

    model_name = f"best_model_{config.model_variant}_{config.dataset}"
    started = time.time()

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=config.epochs,
        patience=config.patience,
        model_name=model_name,
    )

    print("\nFinal Evaluation:")
    test_loss, test_accuracy = trainer.evaluate(test_loader)

    best_epoch = (
        int(np.argmin(history["val_loss"])) + 1 if history["val_loss"] else None
    )
    manifest = collect_run_metadata(
        config,
        split_hash=split_hash,
        results={
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "best_epoch": best_epoch,
            "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
            "epochs_run": len(history["val_loss"]),
            "stopped_early": len(history["val_loss"]) < config.epochs,
            "wall_clock_seconds": round(time.time() - started, 1),
            "mean_epoch_seconds": (
                round(float(np.mean(history["epoch_time"])), 1)
                if history["epoch_time"]
                else None
            ),
            "peak_gpu_memory_mb": trainer.peak_memory_mb(),
            "class_names": get_class_names(config.dataset),
            "history": history,
        },
    )

    manifest_path = save_run_metadata(
        manifest, f"{config.model_dir}/{model_name}.run.json"
    )
    print(f"\nRun manifest written to {manifest_path}")
    print("Quote results with this file; the accuracy alone is not reproducible.")

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

    variant, dataset, num_classes = _resolve_inference_args(args)
    class_names = CIFAR10_CLASSES if dataset == "cifar10" else CIFAR100_CLASSES
    print(f"Model: {variant}  Dataset: {dataset}  Classes: {num_classes}")

    loader_fn = get_cifar10_loaders if dataset == "cifar10" else get_cifar100_loaders
    _, _, test_loader = loader_fn(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )

    model = load_model(
        variant,
        num_classes=num_classes,
        checkpoint_path=args.checkpoint,
    )

    loss, accuracy = evaluate_model(model, test_loader)
    print(f"\nTest Loss: {loss:.6f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    y_pred, y_true, _ = get_predictions(model, test_loader)
    print()
    print_classification_report(y_true, y_pred, class_names)

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

    variant, dataset, num_classes = _resolve_inference_args(args)
    class_names = CIFAR10_CLASSES if dataset == "cifar10" else CIFAR100_CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        variant,
        num_classes=num_classes,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    model.eval()

    image = Image.open(args.image).convert("RGB")
    transform = get_val_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, min(args.top_k, num_classes))

    print(f"\nPredictions for: {args.image}")
    print("-" * 40)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        print(f"{i + 1}. {class_names[idx]}: {prob.item() * 100:.2f}%")

    if args.show_attention:
        import matplotlib.pyplot as plt

        from .visualization import show_attention_on_image, visualize_attention

        attn_map = visualize_attention(model, input_tensor[0], device=device)
        overlay = show_attention_on_image(image.resize((224, 224)), attn_map)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(attn_map, cmap="hot")
        axes[1].set_title("CLS attention (last layer)")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Prediction: {class_names[top_indices[0]]}")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Run export command."""
    from .config import ExportConfig
    from .models import load_model

    variant, _, num_classes = _resolve_inference_args(args)

    model = load_model(
        variant,
        num_classes=num_classes,
        checkpoint_path=args.checkpoint,
    )

    config = ExportConfig(
        output_path=args.output,
        format=args.format,
        opset_version=args.opset_version,
    )
    path = config.export(model)
    print(f"Exported {variant} ({num_classes} classes) to {path}")

    if args.no_verify:
        print("Skipped numerical verification (--no-verify).")
        return 0

    try:
        results = config.verify(model)
    except ImportError:
        print("Install 'vit-trainer[export]' to verify the exported model.")
        return 0

    # A valid graph is not a correct one; check the numbers the runtime produces.
    for batch_size, stats in results["batches"].items():
        print(
            f"  batch {batch_size}: max logit delta "
            f"{stats['max_logit_delta']:.2e}, predictions match"
        )
    print(f"Model size: {results['size_mb']:.2f} MB")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = get_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        return cmd_train(args, argv)

    commands = {
        "eval": cmd_eval,
        "predict": cmd_predict,
        "export": cmd_export,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
