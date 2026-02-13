"""Pytest fixtures for testing."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device() -> torch.device:
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def num_classes() -> int:
    """Number of classes for testing."""
    return 14


@pytest.fixture
def batch_size() -> int:
    """Batch size for testing."""
    return 4


@pytest.fixture
def image_size() -> int:
    """Image size for testing."""
    return 224


@pytest.fixture
def sample_images(batch_size: int, image_size: int) -> torch.Tensor:
    """Generate sample chest X-ray-like images."""
    return torch.randn(batch_size, 3, image_size, image_size)


@pytest.fixture
def sample_labels(batch_size: int, num_classes: int) -> torch.Tensor:
    """Generate sample multi-label targets."""
    return torch.randint(0, 2, (batch_size, num_classes)).float()


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration for testing."""
    return {
        "model": {
            "backbone": "resnet18",  # Use smaller model for testing
            "pretrained": False,
            "num_classes": 14,
            "use_credal_sets": True,
            "dropout_rate": 0.3,
            "use_hierarchical_structure": True,
        },
        "credal": {
            "enabled": True,
            "concentration_prior": 1.0,
            "uncertainty_budget": 0.1,
            "adaptive_temperature": True,
        },
        "data": {
            "image_size": 224,
            "batch_size": 4,
            "num_workers": 0,
            "augmentation_strength": 0.5,
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "optimizer": "adam",
            "scheduler": "cosine",
            "warmup_epochs": 1,
            "early_stopping_patience": 5,
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
        },
        "loss": {
            "type": "evidential_credal",
            "evidential_weight": 0.1,
            "label_smoothing": 0.1,
            "hierarchical_weight": 0.2,
        },
        "paths": {
            "checkpoint_dir": "./test_models",
            "results_dir": "./test_results",
        },
        "seed": 42,
    }


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def temp_results_dir(tmp_path: Path) -> Path:
    """Create temporary results directory."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
