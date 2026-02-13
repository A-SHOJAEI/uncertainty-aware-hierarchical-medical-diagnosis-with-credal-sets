"""
Uncertainty-Aware Hierarchical Medical Diagnosis with Credal Sets.

A novel framework for chest X-ray diagnosis that explicitly models aleatoric
uncertainty through credal set theory.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"
__license__ = "MIT"

from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.model import (
    CredalSetClassifier,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.training.trainer import (
    CredalTrainer,
)

__all__ = [
    "CredalSetClassifier",
    "CredalTrainer",
]
