"""Evaluation metrics for uncertainty-aware medical diagnosis."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import softmax


def compute_auroc(
    y_true: np.ndarray, y_pred: np.ndarray, per_class: bool = False
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        y_true: Ground truth labels [n_samples, n_classes]
        y_pred: Predicted probabilities [n_samples, n_classes]
        per_class: If True, return per-class AUROC

    Returns:
        Mean AUROC across classes, or array of per-class AUROCs
    """
    try:
        if per_class:
            aurocs = []
            for i in range(y_true.shape[1]):
                # Skip if class has no positive samples
                if y_true[:, i].sum() == 0 or y_true[:, i].sum() == len(y_true):
                    aurocs.append(np.nan)
                else:
                    aurocs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            return np.array(aurocs)
        else:
            # Macro average
            aurocs = []
            for i in range(y_true.shape[1]):
                if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
                    aurocs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            return np.mean(aurocs) if aurocs else 0.0
    except Exception as e:
        logging.warning(f"Error computing AUROC: {e}")
        return 0.0


def compute_auprc(
    y_true: np.ndarray, y_pred: np.ndarray, per_class: bool = False
) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    Args:
        y_true: Ground truth labels [n_samples, n_classes]
        y_pred: Predicted probabilities [n_samples, n_classes]
        per_class: If True, return per-class AUPRC

    Returns:
        Mean AUPRC across classes, or array of per-class AUPRCs
    """
    try:
        if per_class:
            auprcs = []
            for i in range(y_true.shape[1]):
                if y_true[:, i].sum() == 0:
                    auprcs.append(np.nan)
                else:
                    auprcs.append(average_precision_score(y_true[:, i], y_pred[:, i]))
            return np.array(auprcs)
        else:
            auprcs = []
            for i in range(y_true.shape[1]):
                if y_true[:, i].sum() > 0:
                    auprcs.append(average_precision_score(y_true[:, i], y_pred[:, i]))
            return np.mean(auprcs) if auprcs else 0.0
    except Exception as e:
        logging.warning(f"Error computing AUPRC: {e}")
        return 0.0


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures the difference between predicted confidence and actual accuracy.

    Args:
        y_true: Ground truth labels [n_samples, n_classes]
        y_pred: Predicted probabilities [n_samples, n_classes]
        n_bins: Number of bins for calibration

    Returns:
        Expected calibration error (lower is better)
    """
    # For multi-label, compute ECE per class and average
    eces = []

    for i in range(y_true.shape[1]):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_p, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for j in range(n_bins):
            mask = bin_indices == j
            if mask.sum() > 0:
                # Average predicted confidence in bin
                bin_confidence = y_p[mask].mean()
                # Actual accuracy in bin
                bin_accuracy = y_t[mask].mean()
                # Weighted difference
                ece += (mask.sum() / len(y_p)) * abs(bin_confidence - bin_accuracy)

        eces.append(ece)

    return np.mean(eces)


def compute_coverage_at_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    confidence_level: float = 0.9,
) -> float:
    """
    Compute coverage at a given confidence level.

    For credal sets: percentage of times true label falls within prediction set.

    Args:
        y_true: Ground truth labels [n_samples, n_classes]
        y_pred: Predicted probabilities [n_samples, n_classes]
        uncertainty: Uncertainty estimates [n_samples, n_classes]
        confidence_level: Confidence level (e.g., 0.9 for 90%)

    Returns:
        Coverage (should be close to confidence_level for well-calibrated model)
    """
    # For binary per-class prediction:
    # Coverage = fraction of times |y_pred - y_true| < threshold based on uncertainty

    if uncertainty is None:
        # Use prediction probability directly
        # Consider prediction "covers" ground truth if confidence is appropriate
        threshold = 1.0 - confidence_level
        coverage = np.mean(
            np.abs(y_pred - y_true) < (threshold + y_pred * (1 - y_pred) * 2)
        )
    else:
        # Use uncertainty to define interval
        # Wider intervals for higher uncertainty
        # z-score for confidence level
        z = 1.645 if confidence_level == 0.9 else 1.96

        # Approximate standard deviation from uncertainty
        std = np.sqrt(uncertainty)

        # Check if true label falls within [y_pred - z*std, y_pred + z*std]
        lower = np.clip(y_pred - z * std, 0, 1)
        upper = np.clip(y_pred + z * std, 0, 1)

        # Coverage: fraction where y_true is in [lower, upper]
        in_interval = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(in_interval)

    return coverage


def compute_prediction_set_efficiency(
    prediction_sets: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Compute prediction set efficiency.

    Efficiency = average size of prediction sets (smaller is better,
    while maintaining coverage).

    Args:
        prediction_sets: Binary masks of prediction sets [n_samples, n_classes]
        y_true: Ground truth labels [n_samples, n_classes]

    Returns:
        Average prediction set size
    """
    set_sizes = prediction_sets.sum(axis=1)
    return np.mean(set_sizes)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    confidence_levels: List[float] = [0.8, 0.9, 0.95],
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.

    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        confidence_levels: List of confidence levels for coverage computation

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_uncertainties = []

    with torch.no_grad():
        for images, labels, uncertainty_masks in data_loader:
            images = images.to(device)

            # Get predictions
            outputs = model.predict(images, return_uncertainty=True)
            probs = outputs["probabilities"].cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy())

            if "uncertainty" in outputs:
                unc = outputs["uncertainty"].cpu().numpy()
                all_uncertainties.append(unc)

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # Compute metrics
    metrics = {}

    # AUROC
    metrics["auroc_mean"] = compute_auroc(y_true, y_pred, per_class=False)
    auroc_per_class = compute_auroc(y_true, y_pred, per_class=True)
    for i, auroc in enumerate(auroc_per_class):
        if not np.isnan(auroc):
            metrics[f"auroc_class_{i}"] = auroc

    # AUPRC
    metrics["auprc_mean"] = compute_auprc(y_true, y_pred, per_class=False)

    # Expected Calibration Error
    metrics["expected_calibration_error"] = compute_expected_calibration_error(
        y_true, y_pred, n_bins=15
    )

    # Coverage at different confidence levels
    if all_uncertainties:
        uncertainty = np.concatenate(all_uncertainties, axis=0)
        for conf_level in confidence_levels:
            coverage = compute_coverage_at_confidence(
                y_true, y_pred, uncertainty, confidence_level=conf_level
            )
            metrics[f"coverage_at_{int(conf_level * 100)}"] = coverage

    # Prediction set efficiency (using threshold on predictions)
    threshold = 0.5
    prediction_sets = (y_pred > threshold).astype(float)
    metrics["prediction_set_efficiency"] = compute_prediction_set_efficiency(
        prediction_sets, y_true
    )

    logging.info("Evaluation complete")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logging.info(f"{key}: {value:.4f}")

    return metrics
