#!/usr/bin/env python
"""Inference script for credal set medical diagnosis model."""

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
from PIL import Image

from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.utils.config import (
    setup_logging,
    get_device,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.preprocessing import (
    get_val_transforms,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.loader import (
    CHEXPERT_LABELS,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.model import (
    CredalSetClassifier,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on chest X-ray images"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input chest X-ray image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for positive class",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions JSON (optional)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to display",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Loaded model
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
        pretrained=False,
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

    logging.info("Model loaded successfully")
    return model


def load_and_preprocess_image(
    image_path: Path, image_size: int = 320
) -> torch.Tensor:
    """
    Load and preprocess a single image.

    Args:
        image_path: Path to image file
        image_size: Target image size

    Returns:
        Preprocessed image tensor [1, 3, H, W]
    """
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")

    # Convert to numpy
    image_np = np.array(image)

    # Apply transforms
    transform = get_val_transforms(image_size)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"]

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def format_predictions(
    predictions: dict,
    threshold: float = 0.5,
    top_k: int = 5,
) -> dict:
    """
    Format predictions for display.

    Args:
        predictions: Raw predictions from model
        threshold: Threshold for positive predictions
        top_k: Number of top predictions to include

    Returns:
        Formatted predictions dictionary
    """
    probs = predictions["probabilities"].squeeze().cpu().numpy()

    # Get top-k predictions
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = {
        "top_predictions": [],
        "positive_findings": [],
        "all_probabilities": {},
    }

    # Top-k predictions
    for idx in top_indices:
        label = CHEXPERT_LABELS[idx]
        prob = float(probs[idx])
        uncertainty = None

        if "uncertainty" in predictions:
            unc = predictions["uncertainty"].squeeze().cpu().numpy()
            uncertainty = float(unc[idx])

        pred_dict = {
            "pathology": label,
            "probability": prob,
            "predicted_class": "positive" if prob > threshold else "negative",
        }

        if uncertainty is not None:
            pred_dict["uncertainty"] = uncertainty

        # Add credal bounds if available
        if "lower_bounds" in predictions and "upper_bounds" in predictions:
            lower = predictions["lower_bounds"].squeeze().cpu().numpy()[idx]
            upper = predictions["upper_bounds"].squeeze().cpu().numpy()[idx]
            pred_dict["credal_interval"] = [float(lower), float(upper)]

        results["top_predictions"].append(pred_dict)

    # All positive findings
    positive_indices = np.where(probs > threshold)[0]
    for idx in positive_indices:
        label = CHEXPERT_LABELS[idx]
        prob = float(probs[idx])
        results["positive_findings"].append({
            "pathology": label,
            "probability": prob,
        })

    # All probabilities
    for idx, label in enumerate(CHEXPERT_LABELS):
        results["all_probabilities"][label] = float(probs[idx])

    return results


def print_predictions(results: dict, threshold: float) -> None:
    """
    Print predictions to console in a formatted way.

    Args:
        results: Formatted predictions
        threshold: Threshold used for classification
    """
    print("\n" + "=" * 80)
    print("CHEST X-RAY DIAGNOSIS PREDICTIONS")
    print("=" * 80)

    print(f"\nTop Predictions (threshold={threshold}):")
    print("-" * 80)
    print(f"{'Pathology':<30} {'Probability':>12} {'Prediction':>12} {'Uncertainty':>12}")
    print("-" * 80)

    for pred in results["top_predictions"]:
        pathology = pred["pathology"]
        prob = pred["probability"]
        pred_class = pred["predicted_class"]
        unc = pred.get("uncertainty", "N/A")

        if isinstance(unc, float):
            unc_str = f"{unc:.4f}"
        else:
            unc_str = str(unc)

        print(f"{pathology:<30} {prob:>12.4f} {pred_class:>12} {unc_str:>12}")

        # Print credal interval if available
        if "credal_interval" in pred:
            lower, upper = pred["credal_interval"]
            print(f"  └─ Credal interval: [{lower:.4f}, {upper:.4f}]")

    print("-" * 80)

    if results["positive_findings"]:
        print(f"\nPositive Findings ({len(results['positive_findings'])} total):")
        for finding in results["positive_findings"]:
            print(f"  - {finding['pathology']}: {finding['probability']:.4f}")
    else:
        print("\nNo positive findings detected.")

    print("=" * 80 + "\n")


def main() -> None:
    """Main inference function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO")
    logging.info("Starting inference script")

    # Get device
    device = get_device(args.device)

    try:
        # Load model
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = load_model(checkpoint_path, device)

        # Load and preprocess image
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logging.info(f"Loading image from {image_path}")
        image_tensor = load_and_preprocess_image(image_path)
        image_tensor = image_tensor.to(device)

        # Run inference
        logging.info("Running inference...")
        with torch.no_grad():
            predictions = model.predict(image_tensor, return_uncertainty=True)

        # Format results
        results = format_predictions(
            predictions,
            threshold=args.threshold,
            top_k=args.top_k,
        )

        # Print to console
        print_predictions(results, args.threshold)

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            logging.info(f"Saved predictions to {output_path}")

        logging.info("Inference completed successfully!")

    except Exception as e:
        logging.error(f"Inference failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
