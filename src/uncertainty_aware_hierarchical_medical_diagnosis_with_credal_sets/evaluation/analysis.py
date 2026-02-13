"""Results analysis and visualization utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot calibration curve (reliability diagram).

    Args:
        y_true: Ground truth labels [n_samples, n_classes]
        y_pred: Predicted probabilities [n_samples, n_classes]
        n_bins: Number of bins
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot first 4 classes
    for i in range(min(4, y_true.shape[1])):
        ax = axes[i]

        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_p, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_confidences = []
        bin_accuracies = []

        for j in range(n_bins):
            mask = bin_indices == j
            if mask.sum() > 0:
                bin_confidences.append(y_p[mask].mean())
                bin_accuracies.append(y_t[mask].mean())

        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(bin_confidences, bin_accuracies, "o-", label="Model")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("True Frequency")
        ax.set_title(f"Class {i} Calibration")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved calibration curve to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_uncertainty_distribution(
    uncertainty: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot distribution of uncertainty values.

    Args:
        uncertainty: Uncertainty estimates [n_samples, n_classes]
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of uncertainties
    axes[0].hist(uncertainty.flatten(), bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Uncertainty")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Uncertainty Distribution")
    axes[0].grid(alpha=0.3)

    # Box plot per class
    if uncertainty.shape[1] <= 14:
        axes[1].boxplot(uncertainty, labels=[f"C{i}" for i in range(uncertainty.shape[1])])
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("Uncertainty")
        axes[1].set_title("Per-Class Uncertainty")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved uncertainty distribution to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_auroc_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot AUROC comparison across different model variants.

    Args:
        results: Dictionary of {model_name: {metric_name: value}}
        save_path: Optional path to save figure
    """
    model_names = list(results.keys())
    auroc_values = [results[name].get("auroc_mean", 0.0) for name in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, auroc_values, alpha=0.7, edgecolor="black")

    # Color bars
    colors = ["red" if val < 0.7 else "yellow" if val < 0.8 else "green" for val in auroc_values]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xlabel("Model Variant")
    plt.ylabel("Mean AUROC")
    plt.title("AUROC Comparison Across Model Variants")
    plt.ylim([0, 1])
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for i, (name, val) in enumerate(zip(model_names, auroc_values)):
        plt.text(i, val + 0.02, f"{val:.3f}", ha="center", va="bottom")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved AUROC comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def create_results_summary(
    metrics: Dict[str, float],
    save_path: Path,
) -> None:
    """
    Create a summary table of results.

    Args:
        metrics: Dictionary of metric names and values
        save_path: Path to save summary
    """
    # Filter main metrics
    main_metrics = [
        "auroc_mean",
        "auprc_mean",
        "expected_calibration_error",
        "coverage_at_90",
        "prediction_set_efficiency",
    ]

    summary_lines = ["# Evaluation Results\n\n"]
    summary_lines.append("## Main Metrics\n\n")
    summary_lines.append("| Metric | Value |\n")
    summary_lines.append("|--------|-------|\n")

    for metric in main_metrics:
        if metric in metrics:
            summary_lines.append(f"| {metric} | {metrics[metric]:.4f} |\n")

    # Per-class AUROC
    summary_lines.append("\n## Per-Class AUROC\n\n")
    summary_lines.append("| Class | AUROC |\n")
    summary_lines.append("|-------|-------|\n")

    for key, value in sorted(metrics.items()):
        if key.startswith("auroc_class_"):
            class_idx = key.split("_")[-1]
            summary_lines.append(f"| Class {class_idx} | {value:.4f} |\n")

    # Write to file
    with open(save_path, "w") as f:
        f.writelines(summary_lines)

    logging.info(f"Saved results summary to {save_path}")


def analyze_prediction_confidence(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    Analyze prediction confidence for correct vs incorrect predictions.

    Args:
        y_pred: Predicted probabilities [n_samples, n_classes]
        y_true: Ground truth labels [n_samples, n_classes]
        save_path: Optional path to save figure
    """
    # Compute correctness (using threshold 0.5)
    predictions_binary = (y_pred > 0.5).astype(int)
    correct = (predictions_binary == y_true).astype(int)

    # Get confidence (max probability)
    confidence_correct = y_pred[correct == 1]
    confidence_incorrect = y_pred[correct == 0]

    plt.figure(figsize=(10, 6))

    plt.hist(
        confidence_correct.flatten(),
        bins=50,
        alpha=0.5,
        label="Correct",
        color="green",
        edgecolor="black",
    )
    plt.hist(
        confidence_incorrect.flatten(),
        bins=50,
        alpha=0.5,
        label="Incorrect",
        color="red",
        edgecolor="black",
    )

    plt.xlabel("Prediction Confidence")
    plt.ylabel("Frequency")
    plt.title("Prediction Confidence Distribution")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved confidence analysis to {save_path}")
    else:
        plt.show()

    plt.close()
