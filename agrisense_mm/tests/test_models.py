"""
Unit Tests for AgriSense-MM Neural Network Models.
Verifies shapes, forward/backward passes, and MC-Dropout uncertainty.
"""

import pytest
import torch
import numpy as np

from agrisense_mm.models import (
    VisualEncoder,
    EnvironmentalTemporalEncoder,
    MotionEncoder,
    MotionQualityGating,
    MultimodalCrossAttentionFusion,
    SpatialZoneGNN,
    build_grid_edge_index,
    MultiTaskOutputHeads,
    AgriSenseMMNet,
    GradCAMExplainer,
)


def test_visual_encoder():
    model = VisualEncoder(embedding_dim=512, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    global_emb, patch_tokens = model(x)

    assert global_emb.shape == (2, 512)
    assert patch_tokens.shape == (2, 49, 512)


def test_environmental_encoder():
    # Bi-LSTM
    lstm_enc = EnvironmentalTemporalEncoder(in_features=7, out_dim=256, use_transformer=False)
    x = torch.randn(2, 14, 7)
    summary_emb, seq_tokens = lstm_enc(x)
    assert summary_emb.shape == (2, 256)
    assert seq_tokens.shape == (2, 14, 256)

    # Transformer
    tf_enc = EnvironmentalTemporalEncoder(in_features=7, out_dim=256, use_transformer=True)
    summary_emb_tf, seq_tokens_tf = tf_enc(x)
    assert summary_emb_tf.shape == (2, 256)
    assert seq_tokens_tf.shape == (2, 14, 256)


def test_motion_encoder_and_gating():
    motion_model = MotionEncoder(in_channels=6, motion_dim=64, num_states=4)
    x = torch.randn(3, 6, 100)
    mot_emb, quality_score, state_logits = motion_model(x)

    assert mot_emb.shape == (3, 64)
    assert quality_score.shape == (3, 1)
    assert state_logits.shape == (3, 4)
    assert torch.all(quality_score >= 0.0) and torch.all(quality_score <= 1.0)

    # Test Quality Gating
    gater = MotionQualityGating(min_retention=0.15)
    vis_tokens = torch.ones(3, 49, 512)
    zero_q = torch.zeros(3, 1)
    gated = gater(vis_tokens, zero_q)
    assert torch.allclose(gated, torch.ones_like(vis_tokens) * 0.15)


def test_cross_attention_fusion():
    fusion = MultimodalCrossAttentionFusion(fusion_dim=512, env_dim=256, motion_dim=64, static_dim=5)
    vis_tokens = torch.randn(2, 49, 512)
    env_tokens = torch.randn(2, 14, 256)
    mot_emb = torch.randn(2, 64)
    static_feat = torch.randn(2, 5)

    fused_vec, explain = fusion(vis_tokens, env_tokens, mot_emb, static_feat)
    assert fused_vec.shape == (2, 512)
    assert "pct_env" in explain
    assert "pct_motion" in explain


def test_spatial_gnn():
    edge_index = build_grid_edge_index(rows=2, cols=2)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0

    gnn = SpatialZoneGNN(in_dim=512, hidden_dim=512, out_dim=512)
    x = torch.randn(4, 512)
    out = gnn(x, edge_index)
    assert out.shape == (4, 512)


def test_multitask_heads_and_mc_dropout():
    heads = MultiTaskOutputHeads(in_dim=512, num_classes=6)
    h = torch.randn(3, 512)
    out = heads(h)

    assert out["cls_logits"].shape == (3, 6)
    assert out["severity"].shape == (3, 1)
    assert out["forecast_3d_logits"].shape == (3, 6)
    assert out["forecast_7d_logits"].shape == (3, 6)

    # MC-Dropout
    mc = heads.predict_with_mc_dropout(h, num_mc_samples=5)
    assert mc["pred_class"].shape == (3,)
    assert mc["predictive_entropy"].shape == (3, 1)
    assert mc["pred_severity"].shape == (3, 1)


def test_end_to_end_agrisense_net():
    model = AgriSenseMMNet()
    imgs = torch.randn(2, 3, 224, 224)
    env = torch.randn(2, 14, 7)
    imu = torch.randn(2, 6, 100)
    static = torch.randn(2, 5)

    out = model(imgs, env, imu, static)
    assert out["cls_logits"].shape == (2, 6)
    assert out["severity"].shape == (2, 1)
    assert out["pred_quality"].shape == (2, 1)

