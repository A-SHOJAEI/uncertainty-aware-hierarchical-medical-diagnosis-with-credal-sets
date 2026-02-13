# Quick Start Guide

## Installation

```bash
# Clone or navigate to project directory
cd uncertainty-aware-hierarchical-medical-diagnosis-with-credal-sets

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 validate_project.py
```

## Running Training

### Default Configuration (Full Model with Credal Sets)

```bash
python scripts/train.py
```

This will:
- Use synthetic CheXpert data for demonstration
- Train with evidential deep learning loss
- Save best model to `models/best_model.pt`
- Log training progress and metrics
- Generate results in `results/training_results.json`

### Ablation Study (Baseline without Credal Sets)

```bash
python scripts/train.py --config configs/ablation.yaml
```

This trains a baseline model without credal sets for comparison.

## Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint models/best_model.pt --split test

# Results will be saved to results/ directory including:
# - metrics_test.json: Numerical metrics
# - calibration_curve.png: Reliability diagram
# - uncertainty_distribution.png: Uncertainty analysis
# - results_summary.md: Human-readable summary
```

## Inference on New Images

```bash
python scripts/predict.py \
    --checkpoint models/best_model.pt \
    --image path/to/chest_xray.jpg \
    --output predictions.json
```

This outputs:
- Top-k predictions with probabilities
- Uncertainty estimates per pathology
- Credal interval bounds (prediction sets)
- List of positive findings

## Running Tests

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run specific test module
PYTHONPATH=src pytest tests/test_model.py -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
.
├── configs/
│   ├── default.yaml          # Full model configuration
│   └── ablation.yaml         # Baseline configuration
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── predict.py            # Inference script
├── src/uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets/
│   ├── models/
│   │   ├── model.py          # Main classifier
│   │   └── components.py     # Custom loss functions and layers
│   ├── training/
│   │   └── trainer.py        # Training loop
│   ├── data/
│   │   ├── loader.py         # Data loading
│   │   └── preprocessing.py  # Augmentation
│   ├── evaluation/
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── analysis.py       # Visualization
│   └── utils/
│       └── config.py         # Configuration utilities
├── tests/                    # Unit tests
├── models/                   # Saved checkpoints
├── results/                  # Evaluation results
└── data/                     # Dataset directory

```

## Key Features

1. **Evidential Deep Learning**: Models epistemic uncertainty via Dirichlet distributions
2. **Credal Set Output**: Prediction sets instead of point estimates
3. **Hierarchical Constraints**: Enforces anatomical relationships between pathologies
4. **Adaptive Calibration**: Per-class temperature scaling
5. **Comprehensive Metrics**: AUROC, ECE, coverage, prediction set efficiency

## Customization

### Modify Model Architecture

Edit `configs/default.yaml`:
```yaml
model:
  backbone: densenet121  # or resnet50, efficientnet_b0, etc.
  use_credal_sets: true  # set to false for standard classifier
  dropout_rate: 0.3
```

### Adjust Training Hyperparameters

```yaml
training:
  epochs: 50
  learning_rate: 0.0001
  batch_size: 16
  scheduler: cosine  # or step, plateau
```

### Configure Loss Weights

```yaml
loss:
  evidential_weight: 0.1      # Weight for uncertainty regularization
  hierarchical_weight: 0.2    # Weight for anatomical constraints
  label_smoothing: 0.1        # Calibration smoothing
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Use smaller backbone (resnet18 instead of densenet121)
- Disable mixed precision: `mixed_precision: false`

### Slow Training
- Increase batch size
- Reduce image size: `image_size: 224`
- Use fewer workers: `num_workers: 2`

### Poor Calibration
- Increase label smoothing
- Adjust evidential_weight
- Train for more epochs with early stopping

## Next Steps

1. Download real CheXpert dataset from https://stanfordmlgroup.github.io/competitions/chexpert/
2. Update `data_root` in config to point to dataset
3. Set `use_synthetic: False` in data loader
4. Train on full dataset
5. Compare full model vs ablation study results
6. Analyze uncertainty on ambiguous cases
