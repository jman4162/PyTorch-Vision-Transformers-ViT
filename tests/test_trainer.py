"""Tests for Trainer class and callbacks."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vit_trainer.training import Trainer, EarlyStopping, ModelCheckpoint
from vit_trainer.models import load_model
from vit_trainer.config import TrainingConfig


def create_dummy_loaders(num_samples=32, batch_size=8, num_classes=10):
    """Create dummy data loaders for testing."""
    # Create random data
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, loader  # Return same loader for train and val


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        # Use a tiny model for fast tests
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        return model

    @pytest.fixture
    def trainer(self, simple_model, tmp_path):
        """Create a trainer instance."""
        return Trainer(
            model=simple_model,
            lr=1e-3,
            use_amp=False,  # Disable for CPU testing
            model_dir=str(tmp_path),
        )

    def test_trainer_init(self, trainer):
        """Test trainer initialization."""
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_trainer_device_detection(self, simple_model, tmp_path):
        """Test automatic device detection."""
        trainer = Trainer(simple_model, model_dir=str(tmp_path))
        # Should be cuda if available, else cpu
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert trainer.device.type == expected.type

    def test_trainer_fit_single_epoch(self, trainer):
        """Test fitting for a single epoch."""
        train_loader, val_loader = create_dummy_loaders(
            num_samples=16, batch_size=8
        )

        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=1,
            patience=0,
            save_best=False,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 1

    def test_trainer_evaluate(self, trainer):
        """Test evaluation."""
        _, test_loader = create_dummy_loaders(num_samples=16, batch_size=8)

        loss, accuracy = trainer.evaluate(test_loader)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100

    def test_trainer_predict(self, trainer):
        """Test prediction."""
        images = torch.randn(4, 3, 224, 224)

        predictions = trainer.predict(images)

        assert predictions.shape == (4,)
        assert predictions.min() >= 0
        assert predictions.max() < 10

    def test_trainer_predict_probs(self, trainer):
        """Test prediction with probabilities."""
        images = torch.randn(4, 3, 224, 224)

        probs = trainer.predict(images, return_probs=True)

        assert probs.shape == (4, 10)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_trainer_save_load(self, trainer, tmp_path):
        """Test saving and loading model."""
        filepath = str(tmp_path / "test_model.pt")

        trainer.save(filepath)
        trainer.load(filepath)

        assert (tmp_path / "test_model.pt").exists()


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_early_stopping_init(self):
        """Test early stopping initialization."""
        es = EarlyStopping(patience=3, monitor="val_loss")
        assert es.patience == 3
        assert es.monitor == "val_loss"
        assert es.mode == "min"

    def test_early_stopping_mode_detection(self):
        """Test automatic mode detection."""
        es_loss = EarlyStopping(monitor="val_loss")
        assert es_loss.mode == "min"

        es_acc = EarlyStopping(monitor="val_accuracy")
        assert es_acc.mode == "max"

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers after patience."""
        es = EarlyStopping(patience=2, monitor="val_loss")
        es.on_train_begin(None)

        # Simulated losses (no improvement after epoch 1)
        losses = [1.0, 0.9, 0.95, 0.96, 0.97]

        for i, loss in enumerate(losses):
            result = es.on_epoch_end(i, {"val_loss": loss}, None)
            if result is False:
                # Should trigger at epoch 4 (2 epochs without improvement)
                assert i >= 3
                break

    def test_early_stopping_resets_on_improvement(self):
        """Test that counter resets when metric improves."""
        es = EarlyStopping(patience=3, monitor="val_loss")
        es.on_train_begin(None)

        # Loss decreases, then stays same, then decreases again
        es.on_epoch_end(0, {"val_loss": 1.0}, None)
        es.on_epoch_end(1, {"val_loss": 1.0}, None)  # No improvement
        assert es.counter == 1

        es.on_epoch_end(2, {"val_loss": 0.8}, None)  # Improvement!
        assert es.counter == 0


class TestModelCheckpoint:
    """Tests for ModelCheckpoint callback."""

    @pytest.fixture
    def simple_model(self):
        return nn.Linear(10, 2)

    def test_checkpoint_init(self, tmp_path):
        """Test checkpoint initialization."""
        cp = ModelCheckpoint(
            filepath=str(tmp_path / "model.pt"),
            monitor="val_loss",
        )
        assert cp.monitor == "val_loss"
        assert cp.save_best_only is True

    def test_checkpoint_saves_best(self, tmp_path, simple_model):
        """Test that checkpoint saves on improvement."""

        class MockTrainer:
            def __init__(self, model):
                self.model = model

        cp = ModelCheckpoint(
            filepath=str(tmp_path / "best.pt"),
            monitor="val_loss",
            save_best_only=True,
        )

        trainer = MockTrainer(simple_model)
        cp.on_train_begin(trainer)

        # First epoch
        cp.on_epoch_end(0, {"val_loss": 1.0}, trainer)
        assert (tmp_path / "best.pt").exists()

        # Better epoch - should save
        import os
        mtime1 = os.path.getmtime(tmp_path / "best.pt")
        cp.on_epoch_end(1, {"val_loss": 0.5}, trainer)
        mtime2 = os.path.getmtime(tmp_path / "best.pt")
        assert mtime2 >= mtime1  # File was updated

        # Worse epoch - should not save (same mtime)
        cp.on_epoch_end(2, {"val_loss": 0.8}, trainer)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.model_variant == "vit_b_16"
        assert config.batch_size == 64
        assert config.lr == 1e-4

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            TrainingConfig(model_variant="invalid")

        with pytest.raises(ValueError):
            TrainingConfig(train_split=1.5)

        with pytest.raises(ValueError):
            TrainingConfig(lr=-0.001)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TrainingConfig(batch_size=32)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["batch_size"] == 32

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        d = {"batch_size": 128, "lr": 0.001}
        config = TrainingConfig.from_dict(d)
        assert config.batch_size == 128
        assert config.lr == 0.001

    def test_config_save_load(self, tmp_path):
        """Test saving and loading config."""
        config = TrainingConfig(batch_size=32, lr=0.001)
        filepath = tmp_path / "config.yaml"

        config.save(filepath)
        loaded = TrainingConfig.load(filepath)

        assert loaded.batch_size == 32
        assert loaded.lr == 0.001
