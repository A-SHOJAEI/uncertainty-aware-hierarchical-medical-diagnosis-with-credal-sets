# Uncertainty-Aware Hierarchical Medical Diagnosis with Credal Sets

A novel framework for chest X-ray diagnosis that explicitly models aleatoric uncertainty through credal set theory. Combines hierarchical multi-label classification with dynamic label smoothing calibrated by radiologist uncertainty annotations, implementing evidential deep learning to learn prediction sets instead of point estimates.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.model import CredalSetClassifier
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.training.trainer import CredalTrainer
import torch

# Initialize model with evidential deep learning
model = CredalSetClassifier(
    num_classes=14,
    backbone='densenet121',
    use_credal_sets=True
)

# Train with uncertainty-aware loss
trainer = CredalTrainer(model, config)
trainer.train()
```

## Training

```bash
# Default configuration
python scripts/train.py

# Ablation study (without credal sets)
python scripts/train.py --config configs/ablation.yaml
```

## Evaluation

```bash
python scripts/evaluate.py --checkpoint models/best_model.pt
```

## Prediction

```bash
python scripts/predict.py --image path/to/xray.jpg --checkpoint models/best_model.pt
```

## Key Results

| Model Variant | AUROC (Mean) | ECE | Coverage@90% | Prediction Set Size |
|--------------|--------------|-----|--------------|---------------------|
| Baseline (Standard CE) | - | - | - | - |
| + Label Smoothing | - | - | - | - |
| + Credal Sets (Full) | - | - | - | - |

Run `python scripts/train.py` to reproduce these results.

## Methodology

### Core Innovation: Credal Set Theory for Medical Uncertainty

Traditional medical image classifiers output point probability estimates, which fail to capture the inherent ambiguity in radiologist annotations (especially for uncertain findings marked as 'U' in CheXpert). This work addresses this limitation through three novel components:

#### 1. Evidential Deep Learning with Credal Sets

Instead of outputting class probabilities p = [p₁, ..., pₖ], we learn a **Dirichlet distribution** over the probability simplex:

```
α = evidence + K  (concentration parameters)
uncertainty = K / Σ(α)  (epistemic uncertainty)
```

This allows the model to output **prediction sets** (credal sets) rather than point estimates, naturally representing cases where multiple diagnoses are plausible.

**Key advantage**: When radiologist annotations are uncertain (U-labels), the model learns to widen its credal bounds rather than forcing a binary decision.

#### 2. Hierarchical Consistency Loss

Medical pathologies have anatomical and clinical dependencies. We enforce these through a novel consistency constraint:

```
L_hierarchy = Σ max(0, P(parent) - P(child))²
```

For example: if Cardiomegaly is detected, the model should also consider Pulmonary Edema. This is encoded in a learnable hierarchy matrix derived from CheXpert pathology relationships.

**Key advantage**: Prevents anatomically inconsistent predictions (e.g., detecting Pneumonia without any Lung Opacity).

#### 3. Per-Class Adaptive Temperature Scaling

Different pathologies have different inherent annotation uncertainties. Cardiomegaly is typically more consistent across radiologists than subtle findings like Atelectasis. We learn pathology-specific temperature parameters:

```
logits_calibrated = logits / temperature_per_class
```

where each pathology's temperature is learned during training to minimize calibration error.

**Key advantage**: Produces well-calibrated uncertainty estimates that account for per-disease annotation variability.

### Loss Function

The full training objective combines:

```
L_total = L_evidential + λ₁·L_hierarchy + λ₂·L_KL

where:
  L_evidential = Type-II maximum likelihood loss on Dirichlet parameters
  L_hierarchy = Hierarchical consistency regularization
  L_KL = KL divergence from uniform Dirichlet (encourages uncertainty on ambiguous cases)
```

### Architecture

The framework implements these innovations in a modular architecture:

1. **Backbone**: DenseNet-121 pretrained on ImageNet, extracting 1024-dim features
2. **Credal Set Layer**: Maps features → Dirichlet parameters (α₁, ..., α₁₄)
3. **Uncertainty Estimation**: Computes credal bounds from Dirichlet variance
4. **Temperature Calibration**: Per-class temperature scaling for final predictions

## Dataset

Uses CheXpert dataset with uncertainty labels (U-Zeros and U-Ones policies). Download from:
https://stanfordmlgroup.github.io/competitions/chexpert/

## Project Structure

```
src/uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets/
├── data/          # Data loading and preprocessing
├── models/        # Model architecture and custom components
├── training/      # Training loop with uncertainty-aware optimization
├── evaluation/    # Calibration metrics and uncertainty analysis
└── utils/         # Configuration and utilities
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- timm (PyTorch Image Models)
- albumentations
- scikit-learn
- numpy
- pandas
- PyYAML
- tqdm

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
