"""
AgriSense-MM: Multi-Task Loss Functions.
Implements:
1. Focal Loss / Cross-Entropy for Stress Classification
2. Huber Loss (Smooth L1) for Severity Regression (0 - 100)
3. Forecast Losses at t+3 and t+7 days
4. Motion Quality & Platform State Losses
5. Combined Weighted Loss:
   L = L_cls + 0.5 * L_sev + 0.7 * (L_fc3 + L_fc7) + 0.2 * L_quality + 0.1 * L_state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
    FL(p_t) = - alpha * (1 - p_t)^gamma * log(p_t)
    Down-weights well-classified examples and focuses on hard edge-of-field cases.
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        p_t = torch.exp(-ce_loss)  # probability of true class
        focal_term = (1.0 - p_t) ** self.gamma
        loss = focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AgriSenseMultiTaskLoss(nn.Module):
    def __init__(
        self,
        weight_cls: float = 1.0,
        weight_sev: float = 0.5,
        weight_forecast: float = 0.7,
        weight_quality: float = 0.2,
        weight_state: float = 0.1,
        use_focal: bool = True,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.w_cls = weight_cls
        self.w_sev = weight_sev
        self.w_fc = weight_forecast
        self.w_qual = weight_quality
        self.w_state = weight_state

        if use_focal:
            self.cls_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Huber loss (delta=5.0) for severity regression
        self.sev_criterion = nn.HuberLoss(delta=5.0)
        self.fc_criterion = nn.CrossEntropyLoss()
        self.qual_criterion = nn.MSELoss()
        self.state_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the weighted multi-task loss and logs individual components.
        """
        # 1. Stress Classification Loss
        l_cls = self.cls_criterion(predictions["cls_logits"], targets["targets_cls"])

        # 2. Severity Regression Loss (scaled by 1/100 for gradient magnitude normalization)
        l_sev = self.sev_criterion(predictions["severity"], targets["targets_sev"]) / 100.0

        # 3. Forecasting Losses (t+3 and t+7 days)
        l_fc3 = self.fc_criterion(predictions["forecast_3d_logits"], targets["targets_fc3"])
        l_fc7 = self.fc_criterion(predictions["forecast_7d_logits"], targets["targets_fc7"])
        l_fc = 0.5 * (l_fc3 + l_fc7)

        # 4. Motion Quality Regressor Loss
        l_qual = self.qual_criterion(predictions["pred_quality"], targets["targets_qual"])

        # 5. Motion State Classification Loss
        l_state = self.state_criterion(predictions["motion_state_logits"], targets["targets_state"])

        # Total weighted loss
        total_loss = (
            self.w_cls * l_cls
            + self.w_sev * l_sev
            + self.w_fc * l_fc
            + self.w_qual * l_qual
            + self.w_state * l_state
        )

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_cls": float(l_cls.detach().item()),
            "loss_sev": float(l_sev.detach().item()),
            "loss_fc3": float(l_fc3.detach().item()),
            "loss_fc7": float(l_fc7.detach().item()),
            "loss_qual": float(l_qual.detach().item()),
            "loss_state": float(l_state.detach().item()),
        }

        return total_loss, loss_dict

