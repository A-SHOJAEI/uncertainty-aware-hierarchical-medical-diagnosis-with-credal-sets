#!/usr/bin/env python3
"""Validation script to check project completeness and correctness."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_file_exists(filepath: str, required: bool = True) -> bool:
    """Check if a file exists."""
    path = Path(filepath)
    exists = path.exists()
    status = "✓" if exists else ("✗ REQUIRED" if required else "- optional")
    print(f"  {status} {filepath}")
    return exists or not required


def check_imports() -> bool:
    """Check that all core modules can be imported."""
    print("\nChecking imports...")
    try:
        from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.model import CredalSetClassifier
        from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.components import (
            EvidentialLoss,
            CredalSetLayer,
            HierarchicalConsistencyLoss,
        )
        from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.training.trainer import CredalTrainer
        from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.loader import get_data_loaders
        from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.evaluation.metrics import evaluate_model
        from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.utils.config import load_config
        print("  ✓ All core modules import successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def check_configs() -> bool:
    """Check that config files are valid YAML."""
    print("\nChecking configurations...")
    try:
        import yaml

        configs = ["configs/default.yaml", "configs/ablation.yaml"]
        all_valid = True

        for config_path in configs:
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                # Check for scientific notation (should not be present)
                config_str = open(config_path).read()
                if "e-" in config_str.lower() or "1e" in config_str.lower():
                    print(f"  ✗ {config_path} contains scientific notation")
                    all_valid = False
                else:
                    print(f"  ✓ {config_path} is valid")
            except Exception as e:
                print(f"  ✗ {config_path} error: {e}")
                all_valid = False

        return all_valid
    except Exception as e:
        print(f"  ✗ Config check failed: {e}")
        return False


def check_model_creation() -> bool:
    """Check that model can be created."""
    print("\nChecking model creation...")
    try:
        import torch
        from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.model import CredalSetClassifier

        model = CredalSetClassifier(
            num_classes=14,
            backbone="resnet18",
            pretrained=False,
            use_credal_sets=True,
        )

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        evidence, alpha, uncertainty = model(x)

        assert evidence.shape == (2, 14)
        assert alpha.shape == (2, 14)
        assert uncertainty.shape == (2, 1)

        print("  ✓ Model creation and forward pass successful")
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def check_custom_components() -> bool:
    """Check custom components exist and work."""
    print("\nChecking custom components...")
    try:
        import torch
        from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.components import (
            EvidentialLoss,
            CredalSetLayer,
            HierarchicalConsistencyLoss,
        )

        # Test CredalSetLayer
        layer = CredalSetLayer(512, 14)
        x = torch.randn(4, 512)
        evidence, alpha, uncertainty = layer(x)
        assert evidence.shape == (4, 14)
        print("  ✓ CredalSetLayer works")

        # Test EvidentialLoss
        loss_fn = EvidentialLoss(14)
        targets = torch.randint(0, 2, (4, 14)).float()
        loss, loss_dict = loss_fn(alpha, targets, epoch=1)
        assert loss.ndim == 0
        print("  ✓ EvidentialLoss works")

        # Test HierarchicalConsistencyLoss
        h_loss_fn = HierarchicalConsistencyLoss()
        preds = torch.rand(4, 14)
        h_loss = h_loss_fn(preds)
        assert h_loss.ndim == 0
        print("  ✓ HierarchicalConsistencyLoss works")

        return True
    except Exception as e:
        print(f"  ✗ Custom components check failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("PROJECT VALIDATION")
    print("=" * 70)

    print("\nChecking required files...")

    required_files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "pyproject.toml",
        ".gitignore",
        "configs/default.yaml",
        "configs/ablation.yaml",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/predict.py",
        "src/uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets/__init__.py",
        "src/uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets/models/model.py",
        "src/uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets/models/components.py",
        "src/uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets/training/trainer.py",
        "src/uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets/data/loader.py",
        "src/uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets/evaluation/metrics.py",
        "tests/test_data.py",
        "tests/test_model.py",
        "tests/test_training.py",
    ]

    files_ok = all(check_file_exists(f) for f in required_files)

    imports_ok = check_imports()
    configs_ok = check_configs()
    model_ok = check_model_creation()
    components_ok = check_custom_components()

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    checks = {
        "Required files": files_ok,
        "Module imports": imports_ok,
        "Configuration files": configs_ok,
        "Model creation": model_ok,
        "Custom components": components_ok,
    }

    for check, status in checks.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{check:<30} {status_str}")

    all_passed = all(checks.values())

    print("=" * 70)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Project is ready!")
    else:
        print("✗ SOME CHECKS FAILED - Please review errors above")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
