"""Custom model components: evidential loss, credal set layer, and hierarchical constraints."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CredalSetLayer(nn.Module):
    """
    Credal set layer that outputs Dirichlet distribution parameters.

    Instead of outputting point probabilities, this layer outputs concentration
    parameters (alpha) for a Dirichlet distribution, representing epistemic
    uncertainty through credal sets.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        concentration_prior: float = 1.0,
        dropout: float = 0.3,
    ):
        """
        Initialize credal set layer.

        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes
            concentration_prior: Prior concentration for Dirichlet (K in EDL literature)
            dropout: Dropout rate
        """
        super().__init__()
        self.num_classes = num_classes
        self.concentration_prior = concentration_prior

        # Evidence network: outputs unnormalized evidence per class
        self.evidence_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute Dirichlet parameters.

        Args:
            x: Input features [batch_size, in_features]

        Returns:
            Tuple of:
                - evidence: Raw evidence values [batch_size, num_classes]
                - alpha: Dirichlet concentration parameters [batch_size, num_classes]
                - uncertainty: Total uncertainty (inverse of Dirichlet strength)
        """
        # Compute evidence (must be positive)
        evidence = self.evidence_net(x)
        evidence = F.softplus(evidence)  # Ensures positivity

        # Dirichlet parameters: alpha = evidence + prior
        alpha = evidence + self.concentration_prior

        # Total uncertainty: inverse of Dirichlet strength
        # S = sum(alpha_k) represents total evidence strength
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.num_classes / S

        # Expected probabilities under Dirichlet: p_k = alpha_k / S
        probs = alpha / S

        return evidence, alpha, uncertainty

    def get_credal_bounds(
        self, alpha: torch.Tensor, confidence: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute credal set bounds (prediction intervals) for each class.

        Args:
            alpha: Dirichlet concentration parameters [batch_size, num_classes]
            confidence: Confidence level for bounds (e.g., 0.9 for 90%)

        Returns:
            Tuple of (lower_bounds, upper_bounds) for each class probability
        """
        # Use Dirichlet variance to compute approximate confidence intervals
        # Var(p_k) = (alpha_k * (S - alpha_k)) / (S^2 * (S + 1))
        S = torch.sum(alpha, dim=1, keepdim=True)
        mean = alpha / S
        variance = (alpha * (S - alpha)) / (S ** 2 * (S + 1))
        std = torch.sqrt(variance)

        # Approximate with normal distribution (valid for large alpha)
        z_score = 1.645 if confidence == 0.9 else 1.96  # 90% or 95%
        lower = torch.clamp(mean - z_score * std, min=0.0, max=1.0)
        upper = torch.clamp(mean + z_score * std, min=0.0, max=1.0)

        return lower, upper


class EvidentialLoss(nn.Module):
    """
    Evidential Deep Learning loss for uncertainty-aware classification.

    Combines classification loss with uncertainty regularization to learn
    both accurate predictions and calibrated uncertainty estimates.

    Reference: Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty"
    """

    def __init__(
        self,
        num_classes: int,
        lambda_reg: float = 0.1,
        annealing_step: int = 10,
    ):
        """
        Initialize evidential loss.

        Args:
            num_classes: Number of classes
            lambda_reg: Weight for KL divergence regularization
            annealing_step: Steps for annealing the regularization term
        """
        super().__init__()
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg
        self.annealing_step = annealing_step
        self.current_step = 0

    def forward(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute evidential loss.

        Args:
            alpha: Dirichlet concentration parameters [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes] (can be soft)
            epoch: Current training epoch for annealing

        Returns:
            Tuple of (total_loss, loss_dict with components)
        """
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S

        # Type II maximum likelihood loss (sum of squared error on Dirichlet parameters)
        # This encourages concentrating evidence on correct classes
        err = (targets - probs) ** 2
        var = probs * (1 - probs) / (S + 1)
        classification_loss = torch.sum(err + var, dim=1)

        # KL divergence regularization: KL(Dir(alpha) || Dir(uniform))
        # Penalizes overconfident predictions on wrong classes
        alpha_tilde = targets + (1 - targets) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)

        # Compute KL divergence
        kl_div = torch.lgamma(S_tilde).squeeze(-1) - torch.lgamma(torch.tensor(self.num_classes, dtype=torch.float32, device=alpha.device))
        kl_div = kl_div - torch.sum(torch.lgamma(alpha_tilde), dim=1)
        kl_div = kl_div + torch.sum(
            (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
            dim=1,
        )

        # Annealing coefficient
        annealing_coef = min(1.0, epoch / self.annealing_step) if self.annealing_step > 0 else 1.0

        # Total loss
        total_loss = torch.mean(classification_loss + annealing_coef * self.lambda_reg * kl_div)

        loss_dict = {
            "classification": torch.mean(classification_loss).item(),
            "kl_divergence": torch.mean(kl_div).item(),
            "annealing_coef": annealing_coef,
        }

        return total_loss, loss_dict


class HierarchicalConsistencyLoss(nn.Module):
    """
    Hierarchical consistency loss for medical diagnosis.

    Enforces anatomical and clinical relationships between pathologies.
    For example: Cardiomegaly -> increased risk of Edema.
    """

    def __init__(self, hierarchy_matrix: Optional[torch.Tensor] = None):
        """
        Initialize hierarchical consistency loss.

        Args:
            hierarchy_matrix: Binary matrix [num_classes, num_classes] where
                             hierarchy_matrix[i, j] = 1 if class i -> class j
        """
        super().__init__()

        # Default CheXpert pathology hierarchy (simplified)
        if hierarchy_matrix is None:
            # 14 CheXpert classes
            hierarchy_matrix = torch.zeros(14, 14)
            # Example relationships:
            # Cardiomegaly (idx 2) -> Edema (idx 5)
            hierarchy_matrix[2, 5] = 1.0
            # Edema -> Consolidation (idx 6)
            hierarchy_matrix[5, 6] = 1.0
            # Consolidation -> Pneumonia (idx 7)
            hierarchy_matrix[6, 7] = 1.0
            # Atelectasis (idx 8) -> Lung Opacity (idx 3)
            hierarchy_matrix[8, 3] = 1.0

        self.register_buffer("hierarchy_matrix", hierarchy_matrix)

    def forward(self, predictions: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute hierarchical consistency loss.

        Args:
            predictions: Predicted probabilities [batch_size, num_classes]
            temperature: Temperature for softening constraints

        Returns:
            Consistency loss (scalar)
        """
        # For each hierarchical relationship i -> j:
        # If P(i) is high, P(j) should also be reasonably high
        # Loss = sum_{i->j} max(0, P(i) - P(j))^2

        batch_size = predictions.size(0)
        loss = torch.tensor(0.0, device=predictions.device)
        num_constraints = 0

        for i in range(self.hierarchy_matrix.size(0)):
            for j in range(self.hierarchy_matrix.size(1)):
                if self.hierarchy_matrix[i, j] > 0:
                    # i should imply j
                    violation = F.relu(predictions[:, i] - predictions[:, j])
                    loss = loss + torch.mean(violation ** 2) / temperature
                    num_constraints += 1

        if num_constraints > 0:
            loss = loss / num_constraints

        return loss


class AdaptiveTemperatureScaling(nn.Module):
    """
    Per-class adaptive temperature scaling for calibration.

    Learns different temperature parameters for each pathology based on
    the inherent uncertainty in that pathology's annotations.
    """

    def __init__(self, num_classes: int, init_temperature: float = 1.0):
        """
        Initialize adaptive temperature scaling.

        Args:
            num_classes: Number of classes
            init_temperature: Initial temperature value
        """
        super().__init__()
        self.temperatures = nn.Parameter(
            torch.ones(num_classes) * init_temperature
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply per-class temperature scaling.

        Args:
            logits: Model logits [batch_size, num_classes]

        Returns:
            Temperature-scaled logits
        """
        # Ensure temperatures are positive
        temps = F.softplus(self.temperatures).unsqueeze(0)
        return logits / temps


def compute_prediction_sets(
    alpha: torch.Tensor, confidence: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute prediction sets from Dirichlet parameters.

    A prediction set contains all classes whose credal bounds overlap
    with high-confidence regions.

    Args:
        alpha: Dirichlet concentration parameters [batch_size, num_classes]
        confidence: Confidence level for prediction sets

    Returns:
        Tuple of:
            - set_mask: Binary mask indicating classes in prediction set
            - set_size: Size of prediction set for each sample
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S

    # Compute uncertainty per class
    variance = (alpha * (S - alpha)) / (S ** 2 * (S + 1))
    std = torch.sqrt(variance)

    # Include class in set if: mean - threshold * std < some_confidence_bound
    # Simplified: include if probability > threshold or uncertainty is high
    threshold = 1.0 - confidence

    # Classes with high probability or high uncertainty
    set_mask = (probs > threshold) | (std > 0.2)

    set_size = torch.sum(set_mask.float(), dim=1)

    return set_mask, set_size
