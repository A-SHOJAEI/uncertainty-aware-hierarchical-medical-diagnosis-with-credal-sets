#!/usr/bin/env python
"""Training script for credal set medical diagnosis model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.utils.config import (
    load_config,
    set_random_seeds,
    setup_logging,
    get_device,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.loader import (
    get_data_loaders,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.model import (
    CredalSetClassifier,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.training.trainer import (
    CredalTrainer,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train uncertainty-aware medical diagnosis model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def initialize_mlflow(config: dict) -> None:
    """
    Initialize MLflow tracking (optional).

    Args:
        config: Configuration dictionary
    """
    try:
        import mlflow

        mlflow_config = config.get("mlflow", {})
        experiment_name = mlflow_config.get("experiment_name", "credal_medical_diagnosis")
        tracking_uri = mlflow_config.get("tracking_uri", "./mlruns")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

        # Log parameters
        mlflow.log_params(
            {
                "model_backbone": config.get("model", {}).get("backbone"),
                "learning_rate": config.get("training", {}).get("learning_rate"),
                "batch_size": config.get("data", {}).get("batch_size"),
                "use_credal_sets": config.get("model", {}).get("use_credal_sets"),
            }
        )

        logging.info("MLflow tracking initialized")
        return True
    except Exception as e:
        logging.warning(f"MLflow initialization failed: {e}. Continuing without MLflow.")
        return False


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)
    logging.info("Starting training script")

    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Set random seeds for reproducibility
    seed = config.get("seed", 42)
    set_random_seeds(seed)

    # Get device
    device = get_device(args.device)

    # Create directories
    checkpoint_dir = Path(config.get("paths", {}).get("checkpoint_dir", "./models"))
    results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize MLflow (wrapped in try/except)
    mlflow_active = initialize_mlflow(config)

    try:
        # Create data loaders
        logging.info("Creating data loaders...")
        train_loader, val_loader, test_loader = get_data_loaders(
            config, use_synthetic=True
        )

        # Create model
        logging.info("Creating model...")
        model_config = config.get("model", {})
        credal_config = config.get("credal", {})

        model = CredalSetClassifier(
            num_classes=model_config.get("num_classes", 14),
            backbone=model_config.get("backbone", "densenet121"),
            pretrained=model_config.get("pretrained", True),
            use_credal_sets=model_config.get("use_credal_sets", True),
            concentration_prior=credal_config.get("concentration_prior", 1.0),
            dropout_rate=model_config.get("dropout_rate", 0.3),
            use_hierarchical_structure=model_config.get("use_hierarchical_structure", True),
            adaptive_temperature=credal_config.get("adaptive_temperature", True),
        )

        logging.info(
            f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
        )

        # Create trainer
        logging.info("Initializing trainer...")
        trainer = CredalTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        # Load checkpoint if provided
        if args.checkpoint:
            logging.info(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(Path(args.checkpoint))

        # Train model
        logging.info("Starting training...")
        final_metrics = trainer.train()

        # Save final results
        results_file = results_dir / "training_results.json"
        with open(results_file, "w") as f:
            json.dump(final_metrics, f, indent=2)
        logging.info(f"Saved training results to {results_file}")

        # Log to MLflow
        if mlflow_active:
            try:
                import mlflow
                mlflow.log_metrics(final_metrics)
                mlflow.log_artifact(str(results_file))
                mlflow.end_run()
            except Exception as e:
                logging.warning(f"MLflow logging failed: {e}")

        logging.info("Training completed successfully!")

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        if mlflow_active:
            try:
                import mlflow
                mlflow.end_run(status="KILLED")
            except:
                pass
        sys.exit(0)

    except Exception as e:
        logging.error(f"Training failed with error: {e}", exc_info=True)
        if mlflow_active:
            try:
                import mlflow
                mlflow.end_run(status="FAILED")
            except:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
