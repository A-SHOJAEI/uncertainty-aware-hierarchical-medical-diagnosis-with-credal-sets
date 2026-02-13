"""Tests for model architecture and components."""

import pytest
import torch

from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.model import (
    CredalSetClassifier,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.components import (
    CredalSetLayer,
    EvidentialLoss,
    HierarchicalConsistencyLoss,
    AdaptiveTemperatureScaling,
    compute_prediction_sets,
)


class TestCredalSetLayer:
    """Test credal set layer."""

    def test_credal_set_layer_creation(self, num_classes: int) -> None:
        """Test creating credal set layer."""
        layer = CredalSetLayer(
            in_features=512,
            num_classes=num_classes,
            concentration_prior=1.0,
            dropout=0.3,
        )
        assert layer is not None

    def test_credal_set_layer_forward(self, num_classes: int, batch_size: int) -> None:
        """Test forward pass through credal set layer."""
        layer = CredalSetLayer(in_features=512, num_classes=num_classes)

        # Create sample input
        x = torch.randn(batch_size, 512)

        # Forward pass
        evidence, alpha, uncertainty = layer(x)

        assert evidence.shape == (batch_size, num_classes)
        assert alpha.shape == (batch_size, num_classes)
        assert uncertainty.shape == (batch_size, 1)

        # Check that evidence is positive
        assert torch.all(evidence >= 0)

        # Check that alpha > prior
        assert torch.all(alpha >= layer.concentration_prior)

    def test_get_credal_bounds(self, num_classes: int, batch_size: int) -> None:
        """Test computing credal bounds."""
        layer = CredalSetLayer(in_features=512, num_classes=num_classes)

        alpha = torch.rand(batch_size, num_classes) * 10 + 1
        lower, upper = layer.get_credal_bounds(alpha, confidence=0.9)

        assert lower.shape == (batch_size, num_classes)
        assert upper.shape == (batch_size, num_classes)

        # Check bounds are valid
        assert torch.all(lower >= 0)
        assert torch.all(upper <= 1)
        assert torch.all(lower <= upper)


class TestEvidentialLoss:
    """Test evidential loss function."""

    def test_evidential_loss_creation(self, num_classes: int) -> None:
        """Test creating evidential loss."""
        loss_fn = EvidentialLoss(
            num_classes=num_classes,
            lambda_reg=0.1,
            annealing_step=10,
        )
        assert loss_fn is not None

    def test_evidential_loss_forward(
        self, num_classes: int, batch_size: int
    ) -> None:
        """Test forward pass through evidential loss."""
        loss_fn = EvidentialLoss(num_classes=num_classes)

        alpha = torch.rand(batch_size, num_classes) * 10 + 1
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()

        loss, loss_dict = loss_fn(alpha, targets, epoch=5)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0

        assert "classification" in loss_dict
        assert "kl_divergence" in loss_dict
        assert "annealing_coef" in loss_dict


class TestHierarchicalConsistencyLoss:
    """Test hierarchical consistency loss."""

    def test_hierarchical_loss_creation(self) -> None:
        """Test creating hierarchical loss."""
        loss_fn = HierarchicalConsistencyLoss()
        assert loss_fn is not None

    def test_hierarchical_loss_forward(
        self, num_classes: int, batch_size: int
    ) -> None:
        """Test forward pass through hierarchical loss."""
        loss_fn = HierarchicalConsistencyLoss()

        predictions = torch.rand(batch_size, num_classes)
        loss = loss_fn(predictions)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss >= 0


class TestAdaptiveTemperatureScaling:
    """Test adaptive temperature scaling."""

    def test_temperature_scaling_creation(self, num_classes: int) -> None:
        """Test creating temperature scaling."""
        scaler = AdaptiveTemperatureScaling(num_classes=num_classes)
        assert scaler is not None
        assert scaler.temperatures.shape == (num_classes,)

    def test_temperature_scaling_forward(
        self, num_classes: int, batch_size: int
    ) -> None:
        """Test forward pass through temperature scaling."""
        scaler = AdaptiveTemperatureScaling(num_classes=num_classes)

        logits = torch.randn(batch_size, num_classes)
        scaled_logits = scaler(logits)

        assert scaled_logits.shape == logits.shape


class TestCredalSetClassifier:
    """Test main classifier model."""

    def test_model_creation_with_credal_sets(self, num_classes: int) -> None:
        """Test creating model with credal sets."""
        model = CredalSetClassifier(
            num_classes=num_classes,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=True,
        )
        assert model is not None
        assert model.use_credal_sets is True

    def test_model_creation_without_credal_sets(self, num_classes: int) -> None:
        """Test creating model without credal sets."""
        model = CredalSetClassifier(
            num_classes=num_classes,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=False,
        )
        assert model is not None
        assert model.use_credal_sets is False

    def test_model_forward_with_credal(
        self, num_classes: int, sample_images: torch.Tensor
    ) -> None:
        """Test forward pass with credal sets."""
        model = CredalSetClassifier(
            num_classes=num_classes,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=True,
        )

        evidence, alpha, uncertainty = model(sample_images)

        batch_size = sample_images.shape[0]
        assert evidence.shape == (batch_size, num_classes)
        assert alpha.shape == (batch_size, num_classes)
        assert uncertainty.shape == (batch_size, 1)

    def test_model_forward_without_credal(
        self, num_classes: int, sample_images: torch.Tensor
    ) -> None:
        """Test forward pass without credal sets."""
        model = CredalSetClassifier(
            num_classes=num_classes,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=False,
        )

        logits, _, _ = model(sample_images)

        batch_size = sample_images.shape[0]
        assert logits.shape == (batch_size, num_classes)

    def test_model_predict(
        self, num_classes: int, sample_images: torch.Tensor
    ) -> None:
        """Test prediction method."""
        model = CredalSetClassifier(
            num_classes=num_classes,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=True,
        )

        predictions = model.predict(sample_images, return_uncertainty=True)

        assert "probabilities" in predictions
        assert "uncertainty" in predictions
        assert "lower_bounds" in predictions
        assert "upper_bounds" in predictions

        batch_size = sample_images.shape[0]
        assert predictions["probabilities"].shape == (batch_size, num_classes)


class TestPredictionSets:
    """Test prediction set computation."""

    def test_compute_prediction_sets(self, num_classes: int, batch_size: int) -> None:
        """Test computing prediction sets from alpha."""
        alpha = torch.rand(batch_size, num_classes) * 10 + 1

        set_mask, set_size = compute_prediction_sets(alpha, confidence=0.9)

        assert set_mask.shape == (batch_size, num_classes)
        assert set_size.shape == (batch_size,)
        assert torch.all(set_size >= 0)
        assert torch.all(set_size <= num_classes)
