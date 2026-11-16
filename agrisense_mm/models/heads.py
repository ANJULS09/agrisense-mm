"""
AgriSense-MM: Multi-Task Output Heads & Uncertainty Estimation.
Tasks:
1. 6-class stress classification
2. Severity regression (0 - 100 continuous score)
3. Forecasting heads (t+3 days, t+7 days stress class)
4. Monte Carlo (MC) Dropout uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MultiTaskOutputHeads(nn.Module):
    def __init__(
        self,
        in_dim: int = 512,
        num_classes: int = 6,
        mc_dropout_p: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mc_dropout_p = mc_dropout_p

        # 1. 6-Class Stress Classifier
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mc_dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mc_dropout_p),
            nn.Linear(128, num_classes),
        )

        # 2. Continuous Severity Regressor (0 - 100)
        self.sev_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mc_dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Bound to [0, 1], then multiplied by 100.0
        )

        # 3. Forecast Head: t + 3 Days Ahead
        self.forecast_3d_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mc_dropout_p),
            nn.Linear(128, num_classes),
        )

        # 4. Forecast Head: t + 7 Days Ahead
        self.forecast_7d_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mc_dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Single deterministic or stochastic forward pass through all task heads.
        Args:
            h: Fused or GNN-refined representation [B, 512]

        Returns:
            Dict containing:
                'cls_logits':       [B, 6]
                'severity':         [B, 1] in [0, 100]
                'forecast_3d_logits': [B, 6]
                'forecast_7d_logits': [B, 6]
        """
        cls_logits = self.cls_head(h)
        severity = self.sev_head(h) * 100.0
        fc_3d_logits = self.forecast_3d_head(h)
        fc_7d_logits = self.forecast_7d_head(h)

        return {
            "cls_logits": cls_logits,
            "severity": severity,
            "forecast_3d_logits": fc_3d_logits,
            "forecast_7d_logits": fc_7d_logits,
        }

    def predict_with_mc_dropout(
        self,
        h: torch.Tensor,
        num_mc_samples: int = 30,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs Monte Carlo Dropout inference by forcing dropout layers to remain active.
        Computes predictive mean, predictive variance, entropy, and uncertainty scores.

        Args:
            h: Fused or GNN-refined representation [B, 512]
            num_mc_samples: Number of stochastic forward passes (default 30)

        Returns:
            Dict with:
                'pred_class':           [B] predicted class index
                'class_probabilities':  [B, 6] mean posterior probability distribution
                'pred_severity':        [B, 1] mean predicted severity [0, 100]
                'severity_variance':    [B, 1] variance across MC samples
                'predictive_entropy':   [B, 1] normalized epistemic uncertainty in [0, 1]
                'forecast_3d_class':    [B]
                'forecast_7d_class':    [B]
        """
        # Set module to train mode temporarily to activate dropout stochastics
        was_training = self.training
        self.train()

        mc_cls_probs = []
        mc_severities = []
        mc_fc3_probs = []
        mc_fc7_probs = []

        with torch.no_grad():
            for _ in range(num_mc_samples):
                outputs = self.forward(h)
                probs = F.softmax(outputs["cls_logits"], dim=-1)
                mc_cls_probs.append(probs)
                mc_severities.append(outputs["severity"])
                mc_fc3_probs.append(F.softmax(outputs["forecast_3d_logits"], dim=-1))
                mc_fc7_probs.append(F.softmax(outputs["forecast_7d_logits"], dim=-1))

        # Restore prior training mode
        self.train(was_training)

        # Stack across MC passes: [T, B, C]
        stack_cls = torch.stack(mc_cls_probs, dim=0)       # [T, B, 6]
        stack_sev = torch.stack(mc_severities, dim=0)      # [T, B, 1]
        stack_fc3 = torch.stack(mc_fc3_probs, dim=0)       # [T, B, 6]
        stack_fc7 = torch.stack(mc_fc7_probs, dim=0)       # [T, B, 6]

        # Mean predictive distributions
        mean_cls_probs = torch.mean(stack_cls, dim=0)      # [B, 6]
        mean_severity = torch.mean(stack_sev, dim=0)       # [B, 1]
        var_severity = torch.var(stack_sev, dim=0)         # [B, 1]
        mean_fc3_probs = torch.mean(stack_fc3, dim=0)      # [B, 6]
        mean_fc7_probs = torch.mean(stack_fc7, dim=0)      # [B, 6]

        # Normalized Shannon Entropy: H = - sum(p * log(p)) / log(K)
        # Bounded strictly in [0, 1]
        eps = 1e-8
        entropy = -torch.sum(mean_cls_probs * torch.log(mean_cls_probs + eps), dim=-1, keepdim=True)
        max_entropy = torch.log(torch.tensor(float(self.num_classes)))
        normalized_uncertainty = torch.clamp(entropy / max_entropy, 0.0, 1.0)

        pred_class = torch.argmax(mean_cls_probs, dim=-1)
        pred_fc3 = torch.argmax(mean_fc3_probs, dim=-1)
        pred_fc7 = torch.argmax(mean_fc7_probs, dim=-1)

        return {
            "pred_class": pred_class,
            "class_probabilities": mean_cls_probs,
            "pred_severity": mean_severity,
            "severity_variance": var_severity,
            "predictive_entropy": normalized_uncertainty,
            "forecast_3d_class": pred_fc3,
            "forecast_3d_probabilities": mean_fc3_probs,
            "forecast_7d_class": pred_fc7,
            "forecast_7d_probabilities": mean_fc7_probs,
        }

