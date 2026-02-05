"""Tests for data loading utilities."""

import pytest
import torch
import numpy as np

from vit_trainer.data import (
    get_cifar10_loaders,
    CIFAR10_CLASSES,
    CIFAR100_CLASSES,
    get_train_transform,
    get_val_transform,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from vit_trainer.data.transforms import denormalize


class TestTransforms:
    """Tests for image transforms."""

    def test_train_transform_output_shape(self):
        """Test that train transform produces correct shape."""
        from PIL import Image

        transform = get_train_transform(image_size=224)
        dummy_image = Image.new("RGB", (32, 32), color="red")

        tensor = transform(dummy_image)

        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_val_transform_output_shape(self):
        """Test that val transform produces correct shape."""
        from PIL import Image

        transform = get_val_transform(image_size=224)
        dummy_image = Image.new("RGB", (32, 32), color="blue")

        tensor = transform(dummy_image)

        assert tensor.shape == (3, 224, 224)

    def test_transform_custom_size(self):
        """Test transforms with custom image size."""
        from PIL import Image

        for size in [128, 224, 384]:
            transform = get_val_transform(image_size=size)
            dummy_image = Image.new("RGB", (50, 50))
            tensor = transform(dummy_image)
            assert tensor.shape == (3, size, size)

    def test_denormalize(self):
        """Test denormalization returns values in [0, 1]."""
        # Create normalized tensor
        tensor = torch.randn(3, 224, 224)
        denorm = denormalize(tensor)

        assert denorm.min() >= 0
        assert denorm.max() <= 1

    def test_denormalize_batch(self):
        """Test denormalization works with batch."""
        tensor = torch.randn(4, 3, 224, 224)
        denorm = denormalize(tensor)

        assert denorm.shape == (4, 3, 224, 224)


class TestClassNames:
    """Tests for class name constants."""

    def test_cifar10_classes_count(self):
        """Test CIFAR-10 has 10 classes."""
        assert len(CIFAR10_CLASSES) == 10

    def test_cifar100_classes_count(self):
        """Test CIFAR-100 has 100 classes."""
        assert len(CIFAR100_CLASSES) == 100

    def test_cifar10_classes_content(self):
        """Test CIFAR-10 class names."""
        expected = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]
        assert CIFAR10_CLASSES == expected


class TestImageNetStats:
    """Tests for ImageNet normalization statistics."""

    def test_imagenet_mean(self):
        """Test ImageNet mean values."""
        assert len(IMAGENET_MEAN) == 3
        assert all(0 <= m <= 1 for m in IMAGENET_MEAN)

    def test_imagenet_std(self):
        """Test ImageNet std values."""
        assert len(IMAGENET_STD) == 3
        assert all(0 < s <= 1 for s in IMAGENET_STD)


# Skip actual data loading tests in CI to avoid downloading datasets
@pytest.mark.skipif(
    not pytest.importorskip("torchvision", reason="torchvision required"),
    reason="Skip data loading tests"
)
class TestDataLoaders:
    """Tests for data loader creation (requires downloading data)."""

    @pytest.fixture(scope="class")
    def cifar10_loaders(self, tmp_path_factory):
        """Create CIFAR-10 loaders once for all tests."""
        data_dir = tmp_path_factory.mktemp("data")
        return get_cifar10_loaders(
            batch_size=4,
            data_dir=str(data_dir),
            num_workers=0,
            seed=42,
        )

    def test_loader_returns_tuple(self, cifar10_loaders):
        """Test that loader returns (train, val, test) tuple."""
        train_loader, val_loader, test_loader = cifar10_loaders
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_batch_shape(self, cifar10_loaders):
        """Test that batches have correct shape."""
        train_loader, _, _ = cifar10_loaders
        images, labels = next(iter(train_loader))

        assert images.shape[0] <= 4  # batch size
        assert images.shape[1] == 3  # channels
        assert images.shape[2] == 224  # height
        assert images.shape[3] == 224  # width
        assert labels.shape[0] == images.shape[0]

    def test_labels_in_range(self, cifar10_loaders):
        """Test that labels are in valid range."""
        train_loader, _, _ = cifar10_loaders
        _, labels = next(iter(train_loader))

        assert labels.min() >= 0
        assert labels.max() <= 9

    def test_reproducible_split(self):
        """Test that same seed produces same split."""
        import tempfile
        with tempfile.TemporaryDirectory() as data_dir:
            loader1 = get_cifar10_loaders(
                batch_size=4, data_dir=data_dir, seed=42, num_workers=0
            )
            loader2 = get_cifar10_loaders(
                batch_size=4, data_dir=data_dir, seed=42, num_workers=0
            )

            # Compare underlying dataset indices (not shuffled batches)
            # The Subset objects should have the same indices with same seed
            train_indices_1 = loader1[0].dataset.indices
            train_indices_2 = loader2[0].dataset.indices

            assert train_indices_1 == train_indices_2
