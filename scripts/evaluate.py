#!/usr/bin/env python
"""Evaluation script for trained credal set medical diagnosis model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.utils.config import (
    load_config,
    set_random_seeds,
    setup_logging,
    get_device,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.loader import (
    get_data_loaders,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.model import (
    CredalSetClassifier,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.evaluation.metrics import (
    evaluate_model,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.evaluation.analysis import (
    plot_calibration_curve,
    plot_uncertainty_distribution,
    create_results_summary,
    analyze_prediction_confidence,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty-aware medical diagnosis model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (optional, will use config from checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Tuple of (model, config)
    """
    logging.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Create model
    model_config = config.get("model", {})
    credal_config = config.get("credal", {})

    model = CredalSetClassifier(
        num_classes=model_config.get("num_classes", 14),
        backbone=model_config.get("backbone", "densenet121"),
        pretrained=False,  # Don't need pretrained weights when loading
        use_credal_sets=model_config.get("use_credal_sets", True),
        concentration_prior=credal_config.get("concentration_prior", 1.0),
        dropout_rate=model_config.get("dropout_rate", 0.3),
        use_hierarchical_structure=model_config.get("use_hierarchical_structure", True),
        adaptive_temperature=credal_config.get("adaptive_temperature", True),
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logging.info(
        f"Loaded model from epoch {checkpoint['epoch']} "
        f"with validation loss {checkpoint['metrics']['val_loss']:.4f}"
    )

    return model, config


def collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Collect all predictions and ground truth.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to run on

    Returns:
        Dictionary with predictions, labels, and uncertainties
    """
    all_preds = []
    all_labels = []
    all_uncertainties = []
    all_alphas = []

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device)

            # Get predictions
            outputs = model.predict(images, return_uncertainty=True)

            all_preds.append(outputs["probabilities"].cpu().numpy())
            all_labels.append(labels.numpy())

            if "uncertainty" in outputs:
                all_uncertainties.append(outputs["uncertainty"].cpu().numpy())

            if "alpha" in outputs:
                all_alphas.append(outputs["alpha"].cpu().numpy())

    return {
        "predictions": np.concatenate(all_preds, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "uncertainties": np.concatenate(all_uncertainties, axis=0) if all_uncertainties else None,
        "alphas": np.concatenate(all_alphas, axis=0) if all_alphas else None,
    }


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO")
    logging.info("Starting evaluation script")

    # Get device
    device = get_device(args.device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model, config = load_model(checkpoint_path, device)

        # Set random seeds
        seed = config.get("seed", 42)
        set_random_seeds(seed)

        # Create data loaders
        logging.info("Creating data loaders...")
        train_loader, val_loader, test_loader = get_data_loaders(
            config, use_synthetic=True
        )

        # Select data loader
        data_loader = test_loader if args.split == "test" else val_loader
        logging.info(f"Evaluating on {args.split} split")

        # Evaluate model
        logging.info("Running evaluation...")
        eval_config = config.get("evaluation", {})
        confidence_levels = eval_config.get("confidence_levels", [0.8, 0.9, 0.95])

        metrics = evaluate_model(
            model=model,
            data_loader=data_loader,
            device=device,
            confidence_levels=confidence_levels,
        )

        # Save metrics
        metrics_file = output_dir / f"metrics_{args.split}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Saved metrics to {metrics_file}")

        # Print summary table
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"{'Metric':<40} {'Value':>15}")
        print("-" * 60)

        key_metrics = [
            "auroc_mean",
            "auprc_mean",
            "expected_calibration_error",
            "coverage_at_90",
            "prediction_set_efficiency",
        ]

        for metric in key_metrics:
            if metric in metrics:
                print(f"{metric:<40} {metrics[metric]:>15.4f}")

        print("=" * 60 + "\n")

        # Collect predictions for visualization
        logging.info("Collecting predictions for visualization...")
        predictions_data = collect_predictions(model, data_loader, device)

        # Create visualizations
        logging.info("Creating visualizations...")

        # Calibration curve
        plot_calibration_curve(
            predictions_data["labels"],
            predictions_data["predictions"],
            n_bins=eval_config.get("calibration_bins", 15),
            save_path=output_dir / "calibration_curve.png",
        )

        # Uncertainty distribution
        if predictions_data["uncertainties"] is not None:
            plot_uncertainty_distribution(
                predictions_data["uncertainties"],
                save_path=output_dir / "uncertainty_distribution.png",
            )

        # Confidence analysis
        analyze_prediction_confidence(
            predictions_data["predictions"],
            predictions_data["labels"],
            save_path=output_dir / "confidence_analysis.png",
        )

        # Create results summary
        create_results_summary(
            metrics,
            save_path=output_dir / "results_summary.md",
        )

        # Save predictions if requested
        if args.save_predictions:
            pred_file = output_dir / f"predictions_{args.split}.npz"
            np.savez(
                pred_file,
                predictions=predictions_data["predictions"],
                labels=predictions_data["labels"],
                uncertainties=predictions_data["uncertainties"],
                alphas=predictions_data["alphas"],
            )
            logging.info(f"Saved predictions to {pred_file}")

        logging.info("Evaluation completed successfully!")

    except Exception as e:
        logging.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
