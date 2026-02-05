"""Training callbacks for monitoring and checkpointing."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: Any) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self, epoch: int, logs: Dict[str, float], trainer: Any
    ) -> Optional[bool]:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary with training metrics
            trainer: Trainer instance

        Returns:
            False to stop training, None to continue
        """
        pass


class EarlyStopping(Callback):
    """Stop training when a metric stops improving.

    Args:
        monitor: Metric to monitor (default: 'val_loss')
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' (auto-detected from monitor name)
        verbose: Print status messages

    Example:
        >>> early_stop = EarlyStopping(patience=3, monitor='val_loss')
        >>> trainer = Trainer(model, callbacks=[early_stop])
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 3,
        min_delta: float = 0.0,
        mode: Optional[str] = None,
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        # Auto-detect mode from metric name
        if mode is None:
            self.mode = "min" if "loss" in monitor else "max"
        else:
            self.mode = mode

        self.best_value: Optional[float] = None
        self.counter = 0
        self.stopped_epoch = 0

    def on_train_begin(self, trainer: Any) -> None:
        self.best_value = None
        self.counter = 0

    def on_epoch_end(
        self, epoch: int, logs: Dict[str, float], trainer: Any
    ) -> Optional[bool]:
        current = logs.get(self.monitor)
        if current is None:
            return None

        if self.best_value is None:
            self.best_value = current
            return None

        if self.mode == "min":
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  No improvement for {self.counter} epoch(s)")

            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print("  Early stopping triggered!")
                return False

        return None


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    Args:
        filepath: Path to save model (can include {epoch}, {val_loss}, etc.)
        monitor: Metric to monitor for best model
        save_best_only: Only save when metric improves
        mode: 'min' or 'max' (auto-detected)
        verbose: Print status messages

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath='models/best_model.pt',
        ...     monitor='val_loss',
        ...     save_best_only=True
        ... )
        >>> trainer = Trainer(model, callbacks=[checkpoint])
    """

    def __init__(
        self,
        filepath: str = "checkpoint.pt",
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: Optional[str] = None,
        verbose: bool = True,
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose

        if mode is None:
            self.mode = "min" if "loss" in monitor else "max"
        else:
            self.mode = mode

        self.best_value: Optional[float] = None

    def on_train_begin(self, trainer: Any) -> None:
        self.best_value = None
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self, epoch: int, logs: Dict[str, float], trainer: Any
    ) -> Optional[bool]:
        current = logs.get(self.monitor)
        if current is None:
            return None

        if self.save_best_only:
            if self.best_value is None:
                is_best = True
            elif self.mode == "min":
                is_best = current < self.best_value
            else:
                is_best = current > self.best_value

            if is_best:
                self.best_value = current
                self._save_model(trainer, epoch, logs)
        else:
            self._save_model(trainer, epoch, logs)

        return None

    def _save_model(
        self, trainer: Any, epoch: int, logs: Dict[str, float]
    ) -> None:
        # Format filepath with epoch and metrics
        filepath_str = str(self.filepath)
        filepath_str = filepath_str.format(
            epoch=epoch,
            **{k: f"{v:.4f}" for k, v in logs.items()},
        )
        filepath = Path(filepath_str)

        torch.save(trainer.model.state_dict(), filepath)

        if self.verbose:
            metric_value = logs.get(self.monitor, 0)
            print(f"  Model saved to {filepath} ({self.monitor}: {metric_value:.6f})")


class LearningRateLogger(Callback):
    """Log learning rate at each epoch."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.lrs = []

    def on_epoch_end(
        self, epoch: int, logs: Dict[str, float], trainer: Any
    ) -> Optional[bool]:
        lr = trainer.optimizer.param_groups[0]["lr"]
        self.lrs.append(lr)
        return None
