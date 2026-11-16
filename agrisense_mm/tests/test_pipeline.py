"""
Unit Tests for Data Pipeline and Losses.
"""

import os
import torch
import numpy as np
import pytest

from agrisense_mm.data.generator import (
    generate_synthetic_leaf_image,
    generate_environmental_sequence,
    generate_imu_window,
    generate_static_features,
    compute_laplacian_blur_score,
)
from agrisense_mm.training.dataset import MultiModalCropDataset, collate_multimodal_batch
from agrisense_mm.training.losses import AgriSenseMultiTaskLoss, FocalLoss


def test_procedural_leaf_generation():
    img_rgb, q = generate_synthetic_leaf_image(stress_idx=1, severity=75.0, platform_state="cruising")
    assert img_rgb.shape == (224, 224, 3)
    assert 0.0 <= q <= 1.0

    # Blur test
    img_blur, q_blur = generate_synthetic_leaf_image(stress_idx=4, severity=80.0, platform_state="high_vibration")
    assert q_blur <= q  # blurred should have lower quality


def test_sensor_data_shapes():
    env = generate_environmental_sequence(stress_idx=2, severity=50.0, days=14)
    assert env.shape == (14, 7)

    imu = generate_imu_window(platform_state="turning", points=100)
    assert imu.shape == (6, 100)

    static = generate_static_features(row=1, col=2)
    assert static.shape == (5,)


def test_multitask_loss():
    loss_fn = AgriSenseMultiTaskLoss(use_focal=True)
    B = 4
    preds = {
        "cls_logits": torch.randn(B, 6),
        "severity": torch.rand(B, 1) * 100.0,
        "forecast_3d_logits": torch.randn(B, 6),
        "forecast_7d_logits": torch.randn(B, 6),
        "pred_quality": torch.rand(B, 1),
        "motion_state_logits": torch.randn(B, 4),
    }
    targets = {
        "targets_cls": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "targets_sev": torch.tensor([[10.0], [50.0], [80.0], [30.0]], dtype=torch.float32),
        "targets_fc3": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "targets_fc7": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "targets_qual": torch.tensor([[0.9], [0.8], [0.4], [0.7]], dtype=torch.float32),
        "targets_state": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }

    total_loss, loss_dict = loss_fn(preds, targets)
    assert total_loss.item() > 0.0
    assert "loss_cls" in loss_dict
    assert "loss_sev" in loss_dict
    assert "loss_fc3" in loss_dict

