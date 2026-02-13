"""Tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np

from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.loader import (
    CheXpertDataset,
    get_data_loaders,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    normalize_uncertainty_labels,
    create_soft_labels,
)


class TestDataPreprocessing:
    """Test data preprocessing functions."""

    def test_train_transforms(self, image_size: int) -> None:
        """Test training transforms."""
        transform = get_train_transforms(image_size, augmentation_strength=0.5)
        assert transform is not None

        # Test transform on sample image
        sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = transform(image=sample_image)

        assert "image" in result
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (3, image_size, image_size)

    def test_val_transforms(self, image_size: int) -> None:
        """Test validation transforms."""
        transform = get_val_transforms(image_size)
        assert transform is not None

        sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = transform(image=sample_image)

        assert "image" in result
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (3, image_size, image_size)

    def test_normalize_uncertainty_labels(self) -> None:
        """Test uncertainty label normalization."""
        # Test with uncertain labels (-1)
        labels = np.array([[-1, 0, 1], [0, -1, 1]])

        # U-zeros policy
        normalized_zeros = normalize_uncertainty_labels(labels, u_zeros=True)
        assert normalized_zeros[0, 0] == 0.0
        assert normalized_zeros[1, 1] == 0.0

        # U-ones policy
        normalized_ones = normalize_uncertainty_labels(labels, u_zeros=False)
        assert normalized_ones[0, 0] == 1.0
        assert normalized_ones[1, 1] == 1.0

        # Test with NaN
        labels_nan = np.array([[np.nan, 0, 1]])
        normalized_nan = normalize_uncertainty_labels(labels_nan, u_zeros=True)
        assert normalized_nan[0, 0] == 0.0

    def test_create_soft_labels(self) -> None:
        """Test soft label creation."""
        labels = np.array([[0, 1, 1], [1, 0, 1]])
        smoothing = 0.1

        soft_labels = create_soft_labels(labels, smoothing)

        # Check that labels are smoothed
        assert soft_labels[0, 0] > 0  # Was 0, now > 0
        assert soft_labels[0, 1] < 1  # Was 1, now < 1
        assert soft_labels.shape == labels.shape

        # Test with uncertainty mask
        uncertainty_mask = np.array([[True, False, False], [False, True, False]])
        soft_labels_unc = create_soft_labels(labels, smoothing, uncertainty_mask)
        assert soft_labels_unc.shape == labels.shape


class TestCheXpertDataset:
    """Test CheXpert dataset loader."""

    def test_synthetic_dataset_creation(self, num_classes: int) -> None:
        """Test creation of synthetic dataset."""
        dataset = CheXpertDataset(
            split="train",
            use_synthetic=True,
            num_samples=100,
        )

        assert len(dataset) == 100
        assert dataset.num_labels == num_classes

    def test_dataset_getitem(self, num_classes: int) -> None:
        """Test dataset __getitem__ method."""
        dataset = CheXpertDataset(
            split="train",
            transform=get_val_transforms(224),
            use_synthetic=True,
            num_samples=10,
        )

        # Get single item
        image, labels, uncertainty_mask = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (num_classes,)
        assert isinstance(uncertainty_mask, torch.Tensor)
        assert uncertainty_mask.shape == (num_classes,)

    def test_dataset_iteration(self) -> None:
        """Test iterating over dataset."""
        dataset = CheXpertDataset(
            split="train",
            transform=get_val_transforms(224),
            use_synthetic=True,
            num_samples=5,
        )

        count = 0
        for image, labels, uncertainty_mask in dataset:
            count += 1
            assert isinstance(image, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert isinstance(uncertainty_mask, torch.Tensor)

        assert count == 5


class TestDataLoaders:
    """Test data loader creation."""

    def test_get_data_loaders(self, sample_config: dict) -> None:
        """Test data loader creation."""
        train_loader, val_loader, test_loader = get_data_loaders(
            sample_config, use_synthetic=True
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check batch from train loader
        images, labels, uncertainty_masks = next(iter(train_loader))
        batch_size = sample_config["data"]["batch_size"]

        assert images.shape[0] == batch_size
        assert labels.shape[0] == batch_size
        assert uncertainty_masks.shape[0] == batch_size

    def test_data_loader_iteration(self, sample_config: dict) -> None:
        """Test iterating through data loader."""
        train_loader, _, _ = get_data_loaders(sample_config, use_synthetic=True)

        batch_count = 0
        for images, labels, uncertainty_masks in train_loader:
            batch_count += 1
            assert images.shape[0] <= sample_config["data"]["batch_size"]
            if batch_count >= 3:  # Only test a few batches
                break

        assert batch_count >= 3
