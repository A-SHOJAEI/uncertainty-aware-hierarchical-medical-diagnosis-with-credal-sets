"""Main model architecture for credal set medical diagnosis."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import timm

from uncertainty_aware_hierarchical_medical_diagnosis_with_credal_sets.models.components import (
    CredalSetLayer,
    AdaptiveTemperatureScaling,
)


class CredalSetClassifier(nn.Module):
    """
    Credal set classifier for uncertainty-aware medical diagnosis.

    Combines a CNN backbone with credal set output layer for epistemic
    uncertainty quantification and adaptive temperature scaling for calibration.
    """

    def __init__(
        self,
        num_classes: int = 14,
        backbone: str = "densenet121",
        pretrained: bool = True,
        use_credal_sets: bool = True,
        concentration_prior: float = 1.0,
        dropout_rate: float = 0.3,
        use_hierarchical_structure: bool = True,
        adaptive_temperature: bool = True,
    ):
        """
        Initialize credal set classifier.

        Args:
            num_classes: Number of pathology classes
            backbone: Name of backbone architecture (from timm)
            pretrained: Whether to use ImageNet pretrained weights
            use_credal_sets: If True, use evidential output; else standard sigmoid
            concentration_prior: Prior for Dirichlet concentration
            dropout_rate: Dropout rate in final layers
            use_hierarchical_structure: Whether to use hierarchical constraints
            adaptive_temperature: Whether to use per-class temperature scaling
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_credal_sets = use_credal_sets
        self.use_hierarchical_structure = use_hierarchical_structure
        self.adaptive_temperature = adaptive_temperature

        # Load backbone from timm
        try:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool="avg",
            )
            self.feature_dim = self.backbone.num_features
            logging.info(
                f"Loaded backbone {backbone} with {self.feature_dim} features "
                f"(pretrained={pretrained})"
            )
        except Exception as e:
            logging.error(f"Failed to load backbone {backbone}: {e}")
            raise

        # Output layer
        if use_credal_sets:
            self.output_layer = CredalSetLayer(
                in_features=self.feature_dim,
                num_classes=num_classes,
                concentration_prior=concentration_prior,
                dropout=dropout_rate,
            )
            logging.info("Using credal set output layer (evidential)")
        else:
            # Standard multi-label classification
            self.output_layer = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim, num_classes),
            )
            logging.info("Using standard sigmoid output layer")

        # Optional: Adaptive temperature scaling for calibration
        if adaptive_temperature and not use_credal_sets:
            self.temperature_scaler = AdaptiveTemperatureScaling(num_classes)
        else:
            self.temperature_scaler = None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input images [batch_size, 3, height, width]

        Returns:
            If use_credal_sets:
                - evidence: Raw evidence values
                - alpha: Dirichlet parameters
                - uncertainty: Uncertainty estimates
            Else:
                - logits: Raw logits
                - None, None
        """
        # Extract features
        features = self.backbone(x)

        # Output prediction
        if self.use_credal_sets:
            evidence, alpha, uncertainty = self.output_layer(features)
            return evidence, alpha, uncertainty
        else:
            logits = self.output_layer(features)

            # Apply temperature scaling if available
            if self.temperature_scaler is not None:
                logits = self.temperature_scaler(logits)

            return logits, None, None

    def predict(
        self, x: torch.Tensor, return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions with uncertainty estimates.

        Args:
            x: Input images [batch_size, 3, height, width]
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary containing:
                - probabilities: Class probabilities
                - uncertainty: Uncertainty estimates (if available)
                - lower_bounds: Lower credal bounds (if available)
                - upper_bounds: Upper credal bounds (if available)
        """
        self.eval()
        with torch.no_grad():
            if self.use_credal_sets:
                evidence, alpha, uncertainty = self.forward(x)

                # Expected probabilities
                S = torch.sum(alpha, dim=1, keepdim=True)
                probabilities = alpha / S

                result = {
                    "probabilities": probabilities,
                    "uncertainty": uncertainty,
                    "alpha": alpha,
                    "evidence": evidence,
                }

                if return_uncertainty:
                    # Compute credal bounds
                    lower, upper = self.output_layer.get_credal_bounds(alpha, confidence=0.9)
                    result["lower_bounds"] = lower
                    result["upper_bounds"] = upper

                return result
            else:
                logits, _, _ = self.forward(x)
                probabilities = torch.sigmoid(logits)

                result = {"probabilities": probabilities}

                # Estimate uncertainty from prediction entropy
                if return_uncertainty:
                    # Binary cross entropy as uncertainty proxy
                    p_clipped = torch.clamp(probabilities, 1e-7, 1 - 1e-7)
                    entropy = -(
                        p_clipped * torch.log(p_clipped)
                        + (1 - p_clipped) * torch.log(1 - p_clipped)
                    )
                    result["uncertainty"] = entropy

                return result

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate feature maps for visualization.

        Args:
            x: Input images

        Returns:
            Feature maps from backbone
        """
        return self.backbone(x)
