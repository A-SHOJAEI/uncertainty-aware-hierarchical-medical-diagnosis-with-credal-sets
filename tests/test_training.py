"""Tests for training loop and trainer."""

import pytest
import torch
from pathlib import Path

from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.model import (
    CredalSetClassifier,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.training.trainer import (
    CredalTrainer,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.loader import (
    get_data_loaders,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.evaluation.metrics import (
    compute_auroc,
    compute_auprc,
    compute_expected_calibration_error,
    compute_coverage_at_confidence,
    evaluate_model,
)


class TestTrainer:
    """Test training loop."""

    def test_trainer_creation(
        self, sample_config: dict, device: torch.device
    ) -> None:
        """Test creating trainer."""
        # Create small dataset for testing
        train_loader, val_loader, _ = get_data_loaders(
            sample_config, use_synthetic=True
        )

        model = CredalSetClassifier(
            num_classes=14,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=True,
        )

        trainer = CredalTrainer(
            model=model,
            config=sample_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_train_epoch(self, sample_config: dict, device: torch.device) -> None:
        """Test training for one epoch."""
        train_loader, val_loader, _ = get_data_loaders(
            sample_config, use_synthetic=True
        )

        model = CredalSetClassifier(
            num_classes=14,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=True,
        )

        trainer = CredalTrainer(
            model=model,
            config=sample_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        # Train for one epoch
        metrics = trainer.train_epoch(epoch=1)

        assert "loss" in metrics
        assert metrics["loss"] >= 0

    def test_validate(self, sample_config: dict, device: torch.device) -> None:
        """Test validation."""
        train_loader, val_loader, _ = get_data_loaders(
            sample_config, use_synthetic=True
        )

        model = CredalSetClassifier(
            num_classes=14,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=True,
        )

        trainer = CredalTrainer(
            model=model,
            config=sample_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        # Validate
        metrics = trainer.validate(epoch=1)

        assert "val_loss" in metrics
        assert metrics["val_loss"] >= 0

    def test_checkpoint_save_load(
        self, sample_config: dict, device: torch.device, temp_checkpoint_dir: Path
    ) -> None:
        """Test saving and loading checkpoints."""
        train_loader, val_loader, _ = get_data_loaders(
            sample_config, use_synthetic=True
        )

        model = CredalSetClassifier(
            num_classes=14,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=True,
        )

        # Update config with temp directory
        sample_config["paths"]["checkpoint_dir"] = str(temp_checkpoint_dir)

        trainer = CredalTrainer(
            model=model,
            config=sample_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.pt"
        metrics = {"val_loss": 0.5}
        trainer.save_checkpoint(checkpoint_path, epoch=1, metrics=metrics)

        assert checkpoint_path.exists()

        # Load checkpoint
        checkpoint = trainer.load_checkpoint(checkpoint_path)

        assert checkpoint["epoch"] == 1
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint


class TestMetrics:
    """Test evaluation metrics."""

    def test_compute_auroc(self) -> None:
        """Test AUROC computation."""
        y_true = torch.randint(0, 2, (100, 5)).numpy().astype(float)
        y_pred = torch.rand(100, 5).numpy()

        auroc = compute_auroc(y_true, y_pred)

        assert isinstance(auroc, float)
        assert 0.0 <= auroc <= 1.0

    def test_compute_auprc(self) -> None:
        """Test AUPRC computation."""
        y_true = torch.randint(0, 2, (100, 5)).numpy().astype(float)
        y_pred = torch.rand(100, 5).numpy()

        auprc = compute_auprc(y_true, y_pred)

        assert isinstance(auprc, float)
        assert 0.0 <= auprc <= 1.0

    def test_compute_ece(self) -> None:
        """Test Expected Calibration Error."""
        y_true = torch.randint(0, 2, (100, 5)).numpy().astype(float)
        y_pred = torch.rand(100, 5).numpy()

        ece = compute_expected_calibration_error(y_true, y_pred, n_bins=10)

        assert isinstance(ece, float)
        assert ece >= 0.0

    def test_compute_coverage(self) -> None:
        """Test coverage computation."""
        y_true = torch.randint(0, 2, (100, 5)).numpy().astype(float)
        y_pred = torch.rand(100, 5).numpy()
        uncertainty = torch.rand(100, 5).numpy() * 0.1

        coverage = compute_coverage_at_confidence(
            y_true, y_pred, uncertainty, confidence_level=0.9
        )

        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0

    def test_evaluate_model(
        self, sample_config: dict, device: torch.device
    ) -> None:
        """Test full model evaluation."""
        _, _, test_loader = get_data_loaders(sample_config, use_synthetic=True)

        model = CredalSetClassifier(
            num_classes=14,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=True,
        )
        model = model.to(device)

        metrics = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            confidence_levels=[0.8, 0.9],
        )

        assert "auroc_mean" in metrics
        assert "auprc_mean" in metrics
        assert "expected_calibration_error" in metrics
        assert isinstance(metrics["auroc_mean"], float)
