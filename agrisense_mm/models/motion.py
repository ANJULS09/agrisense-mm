"""
AgriSense-MM: Motion / IMU Branch.
1D-CNN on accelerometer + gyroscope windows (MPU6050, 100 Hz, 1-sec = [B, 6, 100]).
Outputs:
1. 64-d motion embedding
2. Image quality / blur score q in [0, 1] (Laplacian variance proxy)
3. Platform motion state logits (4 states: cruising, turning, hovering, high_vibration)
4. Motion-aware visual gating module
"""

import torch
import torch.nn as nn
from typing import Tuple


class MotionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        motion_dim: int = 64,
        num_states: int = 4,
        dropout_p: float = 0.15,
    ):
        super().__init__()
        self.motion_dim = motion_dim

        # 3-Stage 1D-CNN Feature Extractor
        self.conv_net = nn.Sequential(
            # Stage 1: Conv -> BN -> ReLU -> Pool
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # 100 -> 50

            # Stage 2: Conv -> BN -> ReLU -> Pool
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # 50 -> 25

            # Stage 3: Conv -> BN -> ReLU -> Adaptive Pool
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),       # -> 128 x 1
        )

        # Dense projection to motion embedding
        self.fc_motion = nn.Sequential(
            nn.Linear(128, motion_dim),
            nn.LayerNorm(motion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

        # Head 1: Image Quality / Blur Predictor q in [0, 1]
        self.quality_head = nn.Sequential(
            nn.Linear(motion_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Bound quality strictly in [0, 1]
        )

        # Head 2: Platform Motion State Classifier (cruising, turning, hovering, vibration)
        self.state_head = nn.Sequential(
            nn.Linear(motion_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(32, num_states),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: IMU tensor of shape [B, 6, 100] (ax, ay, az, gx, gy, gz)

        Returns:
            motion_emb:    [B, 64]
            quality_score: [B, 1] in [0, 1]
            state_logits:  [B, 4]
        """
        feats = self.conv_net(x).flatten(1)     # [B, 128]
        motion_emb = self.fc_motion(feats)       # [B, 64]

        quality_score = self.quality_head(motion_emb)  # [B, 1]
        state_logits = self.state_head(motion_emb)    # [B, 4]

        return motion_emb, quality_score, state_logits


class MotionQualityGating(nn.Module):
    """
    Applies motion-aware quality gating to the visual branch:
    z_vis_gated = z_vis * (alpha_min + (1 - alpha_min) * q)
    When vehicle vibration causes severe blur (q -> 0), image branch is attenuated,
    forcing the network to rely on environmental time series and spatial neighbors.
    """
    def __init__(self, min_retention: float = 0.15):
        super().__init__()
        self.min_retention = min_retention

    def forward(self, visual_emb: torch.Tensor, quality_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_emb:    [B, D] or [B, N, D]
            quality_score: [B, 1] in [0, 1]

        Returns:
            gated_visual_emb with same shape as visual_emb
        """
        effective_q = self.min_retention + (1.0 - self.min_retention) * quality_score
        if visual_emb.dim() == 3:
            effective_q = effective_q.unsqueeze(1)  # [B, 1, 1]
        return visual_emb * effective_q

