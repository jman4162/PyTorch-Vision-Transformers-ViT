"""Configuration management for ViT training."""

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import yaml

if TYPE_CHECKING:
    from torch import nn

# Datasets the training path can actually load, and their class counts.
DATASET_CLASSES = {"cifar10": 10, "cifar100": 100}


@dataclass
class TrainingConfig:
    """Configuration for ViT training.

    Attributes:
        model_variant: ViT variant ('vit_b_16', 'vit_b_32', 'vit_l_16')
        num_classes: Number of output classes (None follows the dataset)
        batch_size: Training batch size
        epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        warmup_epochs: Number of warmup epochs
        patience: Early stopping patience
        use_amp: Enable mixed precision training
        gradient_clip: Max gradient norm for clipping
        seed: Random seed for reproducibility
        compile_model: Use torch.compile (PyTorch 2.x)
        data_dir: Directory for dataset storage
        model_dir: Directory for saving models
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
    """

    # Model settings
    model_variant: str = "vit_b_16"
    num_classes: Optional[int] = None
    compile_model: bool = False

    # Training settings
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    patience: int = 3
    use_amp: bool = True
    gradient_clip: float = 1.0

    # Reproducibility
    seed: int = 42

    # Data settings
    dataset: str = "cifar10"
    data_dir: str = "./data"
    train_split: float = 0.8

    # System settings
    model_dir: str = "./models"
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "auto"

    # Augmentation settings
    augment_train: bool = True
    image_size: int = 224

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_variants = ["vit_b_16", "vit_b_32", "vit_l_16"]
        if self.model_variant not in valid_variants:
            raise ValueError(f"model_variant must be one of {valid_variants}")

        # "imagefolder" was listed here before it was implemented, and the CLI
        # quietly routed it to CIFAR-100. Keep the accepted values to what the
        # training path can actually load.
        if self.dataset not in DATASET_CLASSES:
            raise ValueError(f"dataset must be one of {sorted(DATASET_CLASSES)}")

        expected_classes = DATASET_CLASSES[self.dataset]
        if self.num_classes is None:
            self.num_classes = expected_classes
        elif self.num_classes != expected_classes:
            raise ValueError(
                f"num_classes={self.num_classes} does not match dataset "
                f"'{self.dataset}' ({expected_classes} classes)"
            )

        if not 0 < self.train_split < 1:
            raise ValueError("train_split must be between 0 and 1")

        if self.lr <= 0:
            raise ValueError("lr must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_args(cls, args) -> "TrainingConfig":
        """Create config from argparse namespace."""
        return cls(**{k: v for k, v in vars(args).items() if k in cls.field_names()})

    @classmethod
    def field_names(cls) -> set:
        """Names of the configurable fields."""
        return {f.name for f in fields(cls)}

    def merge(self, overrides: dict) -> "TrainingConfig":
        """Return a copy with `overrides` applied.

        Used to layer explicitly-passed CLI flags over a YAML config; unknown
        keys are ignored so an argparse namespace can be passed straight in.

        Args:
            overrides: Field values to replace

        Returns:
            New config with the overrides applied and re-validated

        Raises:
            ValueError: If the merged result is invalid
        """
        known = {k: v for k, v in overrides.items() if k in self.field_names()}

        # Switching dataset without also naming a class count re-derives it,
        # so `--dataset cifar100` alone stays valid.
        if "dataset" in known and "num_classes" not in known:
            known["num_classes"] = None

        return replace(self, **known)


@dataclass
class ExportConfig:
    """Configuration for model export.

    Attributes:
        output_path: Destination file
        format: Export format ('onnx', 'torchscript')
        opset_version: ONNX opset version
        dynamic_batch: Enable dynamic batch size
        optimize: Apply constant folding
        image_size: Spatial size of the traced dummy input

    Example:
        >>> config = ExportConfig(output_path="model.onnx")
        >>> config.export(model)
        >>> config.verify(model)  # PyTorch vs ONNX Runtime logits
    """

    output_path: str = "model.onnx"
    format: str = "onnx"
    opset_version: int = 14
    dynamic_batch: bool = True
    optimize: bool = True
    image_size: int = 224

    def __post_init__(self):
        valid_formats = ["onnx", "torchscript"]
        if self.format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}")

    def _dummy_input(self, batch_size: int = 1) -> Any:
        import torch

        return torch.randn(batch_size, 3, self.image_size, self.image_size)

    def export(
        self, model: "nn.Module", output_path: Union[str, Path, None] = None
    ) -> Path:
        """Serialize a model to disk.

        Args:
            model: Model to export (moved to CPU and set to eval mode)
            output_path: Override for `self.output_path`

        Returns:
            Path to the written file
        """
        import torch

        path = Path(output_path or self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        model.cpu()
        dummy_input = self._dummy_input()

        if self.format == "onnx":
            dynamic_axes = (
                {"image": {0: "batch_size"}, "logits": {0: "batch_size"}}
                if self.dynamic_batch
                else None
            )
            export_kwargs = {
                "export_params": True,
                "opset_version": self.opset_version,
                "do_constant_folding": self.optimize,
                "input_names": ["image"],
                "output_names": ["logits"],
                "dynamic_axes": dynamic_axes,
            }
            # torch >= 2.9 routes export through the dynamo exporter by default,
            # which pulls in onnxscript. Force the classic TorchScript exporter so
            # the base install needs no extra dependency; older torch (< 2.5) has
            # no `dynamo` kwarg, so fall back to the default there.
            try:
                torch.onnx.export(
                    model, dummy_input, str(path), dynamo=False, **export_kwargs
                )
            except TypeError:
                torch.onnx.export(model, dummy_input, str(path), **export_kwargs)
        else:
            # Scripted rather than traced: tracing a ViT bakes in the batch and
            # spatial dimensions of the example input.
            torch.jit.save(torch.jit.script(model), str(path))

        return path

    def verify(
        self,
        model: "nn.Module",
        output_path: Union[str, Path, None] = None,
        batch_sizes: tuple = (1, 4),
        atol: float = 1e-4,
    ) -> Dict[str, Any]:
        """Check the exported file reproduces PyTorch's logits.

        `onnx.checker.check_model` only validates that the graph is well formed.
        It says nothing about whether the runtime computes the same numbers, so
        run both on identical inputs and compare.

        Args:
            model: The source PyTorch model
            output_path: Override for `self.output_path`
            batch_sizes: Batch sizes to check (needs `dynamic_batch` for >1)
            atol: Absolute tolerance on logits

        Returns:
            Dict with per-batch max logit deviation, argmax agreement, and size

        Raises:
            AssertionError: If predictions disagree or logits exceed `atol`
        """
        import numpy as np
        import torch

        path = Path(output_path or self.output_path)
        model.eval()
        model.cpu()

        if self.format == "onnx":
            import onnx
            import onnxruntime as ort

            onnx.checker.check_model(onnx.load(str(path)))
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name

            def run(x):
                return session.run(None, {input_name: x.numpy()})[0]

        else:
            traced = torch.jit.load(str(path))

            def run(x):
                with torch.no_grad():
                    return traced(x).numpy()

        results: Dict[str, Any] = {
            "size_mb": path.stat().st_size / (1024 * 1024),
            "batches": {},
        }

        for batch_size in batch_sizes:
            if batch_size != 1 and not self.dynamic_batch:
                continue

            x = self._dummy_input(batch_size)
            with torch.no_grad():
                torch_logits = model(x).numpy()
            exported_logits = run(x)

            max_delta = float(np.abs(torch_logits - exported_logits).max())
            agree = bool(
                (torch_logits.argmax(axis=1) == exported_logits.argmax(axis=1)).all()
            )

            assert agree, f"batch {batch_size}: predicted classes differ"
            assert (
                max_delta <= atol
            ), f"batch {batch_size}: max logit delta {max_delta:.2e} exceeds {atol:.0e}"

            results["batches"][batch_size] = {
                "max_logit_delta": max_delta,
                "predictions_agree": agree,
            }

        return results
