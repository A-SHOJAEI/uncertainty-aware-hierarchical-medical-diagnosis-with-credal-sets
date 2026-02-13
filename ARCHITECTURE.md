# Architecture Overview

## Novel Contributions

This project implements three key innovations for uncertainty-aware medical diagnosis:

### 1. Evidential Deep Learning with Credal Sets

Instead of outputting point probabilities, the model learns Dirichlet distribution parameters that represent credal sets - regions of probability distributions that capture epistemic uncertainty.

**Implementation**: `src/models/components.py:CredalSetLayer`

The credal set layer outputs:
- Evidence values (positive reals)
- Dirichlet concentration parameters α = evidence + prior
- Uncertainty estimates: u = K / Σα_i

### 2. Custom Evidential Loss Function

A novel loss combining Type-II maximum likelihood with KL divergence regularization, specifically designed for multi-label medical diagnosis.

**Implementation**: `src/models/components.py:EvidentialLoss`

Components:
- Classification loss on Dirichlet parameters
- KL divergence regularization to prevent overconfidence
- Annealing schedule for progressive uncertainty learning

### 3. Hierarchical Consistency Constraints

Enforces anatomical and clinical relationships between pathologies (e.g., Cardiomegaly → Edema).

**Implementation**: `src/models/components.py:HierarchicalConsistencyLoss`

Uses a learned hierarchy matrix to enforce probabilistic implications between related conditions.

## Model Architecture

```
Input (Chest X-ray)
    ↓
DenseNet-121 Backbone (pretrained)
    ↓
Global Average Pooling
    ↓
Credal Set Layer
    ↓
Evidence → Dirichlet Parameters (α)
    ↓
Uncertainty Quantification
    ↓
Prediction Sets (not just point estimates)
```

## Training Pipeline

1. **Data Augmentation**: Medical imaging-specific transforms (rotation, brightness, Gaussian noise)
2. **Loss Computation**:
   - Evidential loss on Dirichlet parameters
   - Hierarchical consistency loss
   - Combined with adaptive weighting
3. **Uncertainty Calibration**: Per-class temperature scaling
4. **Early Stopping**: Based on validation loss with patience
5. **Mixed Precision**: AMP for faster training

## Evaluation Metrics

- **AUROC**: Standard classification performance
- **Expected Calibration Error (ECE)**: Calibration quality
- **Coverage@Confidence**: Prediction set validity
- **Prediction Set Efficiency**: Average set size (smaller is better)

## Key Differences from Standard Approaches

| Aspect | Standard CNN | This Framework |
|--------|--------------|----------------|
| Output | Point probabilities | Credal sets (probability regions) |
| Uncertainty | None or dropout-based | Explicit Dirichlet modeling |
| Loss | BCE | Evidential + hierarchical constraints |
| Calibration | Post-hoc temperature scaling | Adaptive per-class learning |
| Predictions | Single label | Prediction sets with bounds |

## File Organization

- `models/model.py`: Main classifier with credal output
- `models/components.py`: Custom loss functions and layers
- `training/trainer.py`: Training loop with evidential optimization
- `evaluation/metrics.py`: Uncertainty-aware evaluation metrics
- `data/loader.py`: CheXpert dataset with uncertainty labels
