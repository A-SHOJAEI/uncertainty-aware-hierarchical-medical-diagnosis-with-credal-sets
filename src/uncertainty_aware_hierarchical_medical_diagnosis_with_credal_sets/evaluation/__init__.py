"""Evaluation metrics and analysis tools."""

from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.evaluation.metrics import (
    compute_auroc,
    compute_auprc,
    compute_expected_calibration_error,
    compute_coverage_at_confidence,
    compute_prediction_set_efficiency,
    evaluate_model,
)

__all__ = [
    "compute_auroc",
    "compute_auprc",
    "compute_expected_calibration_error",
    "compute_coverage_at_confidence",
    "compute_prediction_set_efficiency",
    "evaluate_model",
]
