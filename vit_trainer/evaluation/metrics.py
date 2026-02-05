"""Evaluation metrics and utilities."""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Evaluate model accuracy on a dataset.

    Args:
        model: PyTorch model
        test_loader: Data loader for evaluation
        device: Device to run evaluation on

    Returns:
        Tuple of (loss, accuracy percentage)

    Example:
        >>> loss, accuracy = evaluate_model(model, test_loader)
        >>> print(f"Accuracy: {accuracy:.2f}%")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get all predictions, labels, and probabilities from a dataset.

    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run inference on

    Returns:
        Tuple of (predictions, true_labels, probabilities)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Getting predictions"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Dictionary with accuracy, per-class metrics, and classification report
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix,
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    metrics = {
        "accuracy": accuracy * 100,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if class_names is not None:
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=class_names
        )
        metrics["per_class"] = {
            name: {
                "precision": p,
                "recall": r,
                "f1": f,
                "support": int(s),
            }
            for name, p, r, f, s in zip(
                class_names, precision, recall, f1, support
            )
        }

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
        cmap: Colormap name
        title: Plot title
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> None:
    """Plot training history (loss and learning rate).

    Args:
        history: Training history dictionary with 'train_loss', 'val_loss', 'lr'
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate plot
    ax2.plot(epochs, history["lr"], "g-", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> None:
    """Print a formatted classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    from sklearn.metrics import classification_report

    print("Classification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))
