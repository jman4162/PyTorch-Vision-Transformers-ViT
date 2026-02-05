"""Trainer class for Vision Transformer models."""

import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    """Trainer for Vision Transformer models.

    Features:
    - Mixed precision training (AMP) for 2-3x speedup
    - Gradient clipping to prevent exploding gradients
    - Learning rate warmup with cosine decay
    - Early stopping and best model checkpointing
    - Callback support for extensibility

    Example:
        >>> from vit_trainer import Trainer, load_model, get_cifar10_loaders
        >>> model = load_model("vit_b_16", num_classes=10)
        >>> train_loader, val_loader, _ = get_cifar10_loaders(batch_size=64)
        >>> trainer = Trainer(model, lr=1e-4, use_amp=True)
        >>> history = trainer.fit(train_loader, val_loader, epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 2,
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        device: Optional[torch.device] = None,
        model_dir: str = "./models",
        callbacks: Optional[List] = None,
    ):
        """Initialize trainer.

        Args:
            model: ViT model to train
            optimizer: Custom optimizer (default: AdamW)
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_epochs: Number of warmup epochs
            use_amp: Enable mixed precision training
            gradient_clip: Max gradient norm (0 to disable)
            device: Training device (auto-detected if None)
            model_dir: Directory to save checkpoints
            callbacks: List of callback objects
        """
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision scaler
        self.scaler = (
            GradScaler()
            if use_amp and self.device.type == "cuda"
            else None
        )

        # Callbacks
        self.callbacks = callbacks or []

        # Training state
        self.current_epoch = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

    def _get_warmup_scheduler(
        self, total_epochs: int
    ) -> LambdaLR:
        """Create learning rate scheduler with warmup and cosine decay."""
        def lr_lambda(epoch: int) -> float:
            if epoch < self.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - self.warmup_epochs) / (
                    total_epochs - self.warmup_epochs
                )
                return 0.5 * (1 + np.cos(np.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        current_lr = self.optimizer.param_groups[0]["lr"]

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False,
        )

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.scaler is not None:
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()

                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        return running_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Run validation epoch."""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.scaler is not None:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()

        return running_loss / len(val_loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 3,
        save_best: bool = True,
        model_name: str = "best_model",
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience (0 to disable)
            save_best: Save best model checkpoint
            model_name: Name for saved model file

        Returns:
            Training history dictionary
        """
        # Setup scheduler
        scheduler = self._get_warmup_scheduler(epochs)

        # Setup callbacks
        callbacks = list(self.callbacks)
        if patience > 0:
            callbacks.append(EarlyStopping(patience=patience))
        if save_best:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.model_dir / f"{model_name}.pt",
                    monitor="val_loss",
                    save_best_only=True,
                )
            )

        # Initialize callbacks
        for cb in callbacks:
            cb.on_train_begin(self)

        print(f"Training on {self.device}")
        if self.scaler is not None:
            print("Mixed precision training enabled (AMP)")
        print(f"Epochs: {epochs}, Patience: {patience}")
        print("-" * 50)

        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Callbacks: epoch begin
            for cb in callbacks:
                cb.on_epoch_begin(epoch, self)

            # Training
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validation
            val_loss = self._validate_epoch(val_loader)
            self.history["val_loss"].append(val_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)
            scheduler.step()

            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.2e}, "
                f"Time: {epoch_time:.1f}s"
            )

            # Callbacks: epoch end
            logs = {"train_loss": train_loss, "val_loss": val_loss, "lr": current_lr}
            stop_training = False
            for cb in callbacks:
                result = cb.on_epoch_end(epoch, logs, self)
                if result is False:
                    stop_training = True

            if stop_training:
                print(f"\nTraining stopped at epoch {epoch + 1}")
                break

        # Callbacks: training end
        for cb in callbacks:
            cb.on_train_end(self)

        return self.history

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total

        print(f"Test Loss: {avg_loss:.6f}")
        print(f"Test Accuracy: {accuracy:.2f}%")

        return avg_loss, accuracy

    def predict(
        self,
        images: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """Make predictions on images.

        Args:
            images: Input tensor [B, C, H, W]
            return_probs: Return probabilities instead of class indices

        Returns:
            Predictions tensor
        """
        self.model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            if return_probs:
                return torch.softmax(outputs, dim=1)
            return torch.argmax(outputs, dim=1)

    def save(self, filepath: str) -> None:
        """Save model weights.

        Args:
            filepath: Path to save model
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model weights.

        Args:
            filepath: Path to model file
        """
        state_dict = torch.load(
            filepath, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {filepath}")
