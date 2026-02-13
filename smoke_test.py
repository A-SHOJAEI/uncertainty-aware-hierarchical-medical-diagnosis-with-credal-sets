#!/usr/bin/env python3
"""Quick smoke test to verify core functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.model import CredalSetClassifier
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.components import EvidentialLoss
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.loader import CheXpertDataset
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.preprocessing import get_val_transforms

print("Creating model...")
model = CredalSetClassifier(
    num_classes=14,
    backbone="resnet18",
    pretrained=False,
    use_credal_sets=True,
)

print("Creating dataset...")
dataset = CheXpertDataset(
    split="train",
    transform=get_val_transforms(224),
    use_synthetic=True,
    num_samples=10,
)

print("Testing forward pass...")
image, labels, uncertainty_mask = dataset[0]
image = image.unsqueeze(0)  # Add batch dim

evidence, alpha, uncertainty = model(image)
print(f"  Evidence shape: {evidence.shape}")
print(f"  Alpha shape: {alpha.shape}")
print(f"  Uncertainty shape: {uncertainty.shape}")

print("Testing loss computation...")
loss_fn = EvidentialLoss(num_classes=14)
labels = labels.unsqueeze(0)
loss, loss_dict = loss_fn(alpha, labels, epoch=1)
print(f"  Loss: {loss.item():.4f}")
print(f"  Components: {loss_dict}")

print("\n✓ All smoke tests passed!")
