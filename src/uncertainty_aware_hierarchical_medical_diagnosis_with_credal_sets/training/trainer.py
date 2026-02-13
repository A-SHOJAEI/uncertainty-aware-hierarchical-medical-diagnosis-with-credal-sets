"""Training loop with evidential loss, early stopping, and learning rate scheduling."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from tqdm import tqdm

from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.components import (
    EvidentialLoss,
    HierarchicalConsistencyLoss,
)
from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.data.preprocessing import (
    create_soft_labels,
)


class CredalTrainer:
    """
    Trainer for credal set medical diagnosis model.

    Supports:
    - Evidential deep learning loss
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - MLflow logging (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Configuration dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Extract config
        training_config = config.get("training", {})
        loss_config = config.get("loss", {})
        self.use_credal = config.get("model", {}).get("use_credal_sets", True)

        self.epochs = training_config.get("epochs", 50)
        self.learning_rate = training_config.get("learning_rate", 0.0001)
        self.weight_decay = training_config.get("weight_decay", 0.00001)
        self.gradient_clip = training_config.get("gradient_clip_norm", 1.0)
        self.early_stopping_patience = training_config.get("early_stopping_patience", 10)
        self.use_amp = training_config.get("mixed_precision", True)

        # Setup optimizer
        optimizer_name = training_config.get("optimizer", "adam").lower()
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )

        # Setup learning rate scheduler
        scheduler_name = training_config.get("scheduler", "cosine").lower()
        warmup_epochs = training_config.get("warmup_epochs", 5)

        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs - warmup_epochs
            )
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )

        self.warmup_epochs = warmup_epochs

        # Setup loss functions
        if self.use_credal:
            self.evidential_loss = EvidentialLoss(
                num_classes=model.num_classes,
                lambda_reg=loss_config.get("evidential_weight", 0.1),
                annealing_step=10,
            )
        else:
            self.base_criterion = nn.BCEWithLogitsLoss()

        # Hierarchical consistency loss
        use_hierarchical = config.get("model", {}).get("use_hierarchical_structure", True)
        if use_hierarchical:
            self.hierarchical_loss = HierarchicalConsistencyLoss()
            self.hierarchical_weight = loss_config.get("hierarchical_weight", 0.2)
        else:
            self.hierarchical_loss = None
            self.hierarchical_weight = 0.0

        # Label smoothing
        self.label_smoothing = loss_config.get("label_smoothing", 0.1)

        # Mixed precision
        self.scaler = GradScaler('cuda') if self.use_amp and torch.cuda.is_available() else None

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get("paths", {}).get("checkpoint_dir", "./models"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Trainer initialized: optimizer={optimizer_name}, scheduler={scheduler_name}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        loss_components = {"classification": 0.0, "hierarchical": 0.0, "kl_div": 0.0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        for batch_idx, (images, labels, uncertainty_masks) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            uncertainty_masks = uncertainty_masks.to(self.device)

            # Apply label smoothing
            if self.label_smoothing > 0:
                soft_labels = create_soft_labels(
                    labels.cpu().numpy(),
                    self.label_smoothing,
                    uncertainty_masks.cpu().numpy().astype(bool),
                )
                labels = torch.from_numpy(soft_labels).to(self.device)

            # Forward pass with mixed precision
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp and torch.cuda.is_available()):
                if self.use_credal:
                    evidence, alpha, uncertainty = self.model(images)

                    # Evidential loss
                    loss, loss_dict = self.evidential_loss(alpha, labels, epoch)
                    loss_components["classification"] += loss_dict["classification"]
                    loss_components["kl_div"] += loss_dict["kl_divergence"]

                    # Hierarchical consistency loss
                    if self.hierarchical_loss is not None:
                        S = torch.sum(alpha, dim=1, keepdim=True)
                        probs = alpha / S
                        h_loss = self.hierarchical_loss(probs)
                        loss = loss + self.hierarchical_weight * h_loss
                        loss_components["hierarchical"] += h_loss.item()
                else:
                    logits, _, _ = self.model(images)
                    loss = self.base_criterion(logits, labels)
                    loss_components["classification"] += loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            # Update metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        # Average metrics
        avg_loss = total_loss / total_samples
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)

        metrics = {"loss": avg_loss, **loss_components}
        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for images, labels, uncertainty_masks in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            if self.use_credal:
                evidence, alpha, uncertainty = self.model(images)
                loss, _ = self.evidential_loss(alpha, labels, epoch)

                # Add hierarchical loss
                if self.hierarchical_loss is not None:
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    probs = alpha / S
                    h_loss = self.hierarchical_loss(probs)
                    loss = loss + self.hierarchical_weight * h_loss
            else:
                logits, _, _ = self.model(images)
                loss = self.base_criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        return {"val_loss": avg_loss}

    def train(self) -> Dict[str, float]:
        """
        Full training loop with early stopping.

        Returns:
            Dictionary of final metrics
        """
        logging.info("Starting training...")

        for epoch in range(1, self.epochs + 1):
            # Warmup learning rate
            if epoch <= self.warmup_epochs:
                warmup_factor = epoch / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate * warmup_factor

            # Train and validate
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Learning rate scheduling
            if epoch > self.warmup_epochs:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Logging
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch {epoch}/{self.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0

                # Save best model
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                logging.info(f"Saved best model to {checkpoint_path}")
            else:
                self.patience_counter += 1
                logging.info(f"Early stopping patience: {self.patience_counter}/{self.early_stopping_patience}")

                if self.patience_counter >= self.early_stopping_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch}")
                    break

        logging.info(f"Training completed. Best val loss: {self.best_val_loss:.4f}")
        return {"best_val_loss": self.best_val_loss}

    def save_checkpoint(
        self, path: Path, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Metrics dictionary
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> Dict:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logging.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint
