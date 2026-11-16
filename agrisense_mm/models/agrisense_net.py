"""
AgriSense-MM: Unified Multimodal Neural Network Architecture.
Integrates:
1. Visual Branch (EfficientNet-B0 -> 512-d + patch tokens)
2. Environmental Branch (Bi-LSTM / Temporal Transformer -> 256-d + tokens)
3. Motion / IMU Branch (1D-CNN -> quality score q + platform state + 64-d)
4. Motion Quality Gating (scales visual tokens by q)
5. Cross-Attention Fusion (visual queries attend to sensor context)
6. Spatial Zone GNN (GraphSAGE / GAT over field grid zones)
7. Multi-Task Heads (6-class Cls, Severity Reg, Forecast t+3/t+7, MC-Dropout)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from agrisense_mm.models.visual import VisualEncoder
from agrisense_mm.models.temporal import EnvironmentalTemporalEncoder
from agrisense_mm.models.motion import MotionEncoder, MotionQualityGating
from agrisense_mm.models.fusion import MultimodalCrossAttentionFusion
from agrisense_mm.models.gnn import SpatialZoneGNN
from agrisense_mm.models.heads import MultiTaskOutputHeads
from agrisense_mm.data.constants import NUM_CLASSES, NUM_PLATFORM_STATES


class AgriSenseMMNet(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        visual_dim: int = 512,
        env_dim: int = 256,
        motion_dim: int = 64,
        static_dim: int = 5,
        fusion_dim: int = 512,
        use_transformer_env: bool = False,
        enable_env: bool = True,
        enable_motion: bool = True,
        enable_gnn: bool = True,
        modality_dropout_p: float = 0.15,
        dropout_p: float = 0.15,
    ):
        super().__init__()
        self.enable_env = enable_env
        self.enable_motion = enable_motion
        self.enable_gnn = enable_gnn

        # Branch 1: Visual Encoder
        self.visual_branch = VisualEncoder(
            embedding_dim=visual_dim,
            pretrained=True,
            dropout_p=dropout_p,
        )

        # Branch 2: Environmental Temporal Encoder
        self.env_branch = EnvironmentalTemporalEncoder(
            in_features=7,
            hidden_dim=env_dim // 2,
            out_dim=env_dim,
            use_transformer=use_transformer_env,
            dropout_p=dropout_p,
        )

        # Branch 3: Motion / IMU Encoder
        self.motion_branch = MotionEncoder(
            in_channels=6,
            motion_dim=motion_dim,
            num_states=NUM_PLATFORM_STATES,
            dropout_p=dropout_p,
        )
        self.quality_gater = MotionQualityGating(min_retention=0.15)

        # Branch 4: Multimodal Cross-Attention Fusion
        self.fusion = MultimodalCrossAttentionFusion(
            fusion_dim=fusion_dim,
            num_heads=4,
            num_layers=2,
            env_dim=env_dim,
            motion_dim=motion_dim,
            static_dim=static_dim,
            modality_dropout_p=modality_dropout_p,
            dropout_p=dropout_p,
        )

        # Fallback projection for ablation modes (e.g. vision-only)
        self.vision_only_proj = nn.Linear(visual_dim, fusion_dim)

        # Branch 5: Spatial Zone GNN
        self.gnn = SpatialZoneGNN(
            in_dim=fusion_dim,
            hidden_dim=fusion_dim,
            out_dim=fusion_dim,
            dropout_p=dropout_p,
        )

        # Branch 6: Multi-Task Output Heads
        self.heads = MultiTaskOutputHeads(
            in_dim=fusion_dim,
            num_classes=num_classes,
            mc_dropout_p=dropout_p,
        )

    def forward(
        self,
        images: torch.Tensor,                               # [B, 3, 224, 224]
        env_seqs: Optional[torch.Tensor] = None,             # [B, 14, 7]
        imu_windows: Optional[torch.Tensor] = None,          # [B, 6, 100]
        static_feats: Optional[torch.Tensor] = None,         # [B, 5]
        edge_index: Optional[torch.Tensor] = None,           # [2, E]
    ) -> Dict[str, torch.Tensor]:
        """
        Unified Forward Pass.
        Returns:
            Dict containing:
                'cls_logits':           [B, 6]
                'severity':             [B, 1]
                'forecast_3d_logits':   [B, 6]
                'forecast_7d_logits':   [B, 6]
                'pred_quality':         [B, 1]
                'motion_state_logits':  [B, 4]
                'fused_vector':         [B, 512]
                'explain_dict':         Modality attribution dict
        """
        B = images.size(0)
        device = images.device

        # 1. Visual Branch
        vis_emb, vis_patch_tokens = self.visual_branch(images)  # [B, 512], [B, 49, 512]

        # 2. Motion Branch
        if self.enable_motion and imu_windows is not None:
            mot_emb, quality_score, state_logits = self.motion_branch(imu_windows)
            # Apply Motion-Aware Blur Gating to visual tokens
            vis_patch_tokens = self.quality_gater(vis_patch_tokens, quality_score)
        else:
            mot_emb = torch.zeros(B, 64, device=device)
            quality_score = torch.ones(B, 1, device=device)
            state_logits = torch.zeros(B, NUM_PLATFORM_STATES, device=device)

        # 3. Environmental Branch
        if self.enable_env and env_seqs is not None:
            _, env_tokens = self.env_branch(env_seqs)  # [B, 14, 256]
        else:
            env_tokens = torch.zeros(B, 14, 256, device=device)

        if static_feats is None:
            static_feats = torch.zeros(B, 5, device=device)

        # 4. Fusion
        if self.enable_env or self.enable_motion:
            fused_emb, explain_dict = self.fusion(
                vis_patch_tokens=vis_patch_tokens,
                env_tokens=env_tokens,
                motion_emb=mot_emb,
                static_feat=static_feats,
            )
        else:
            # Vision-Only Ablation baseline
            fused_emb = self.vision_only_proj(vis_emb)
            explain_dict = {
                "pct_env": torch.zeros(B, device=device),
                "pct_motion": torch.zeros(B, device=device),
                "pct_soil_gps": torch.zeros(B, device=device),
            }

        # 5. Spatial Zone GNN
        if self.enable_gnn and edge_index is not None:
            zone_emb = self.gnn(fused_emb, edge_index)
        else:
            zone_emb = fused_emb

        # 6. Multi-Task Output Heads
        head_outputs = self.heads(zone_emb)

        # Assemble full output dictionary
        return {
            "cls_logits": head_outputs["cls_logits"],
            "severity": head_outputs["severity"],
            "forecast_3d_logits": head_outputs["forecast_3d_logits"],
            "forecast_7d_logits": head_outputs["forecast_7d_logits"],
            "pred_quality": quality_score,
            "motion_state_logits": state_logits,
            "fused_vector": zone_emb,
            "explain_dict": explain_dict,
        }

    def predict_uncertainty(
        self,
        images: torch.Tensor,
        env_seqs: Optional[torch.Tensor] = None,
        imu_windows: Optional[torch.Tensor] = None,
        static_feats: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        num_mc_samples: int = 30,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs MC-Dropout uncertainty inference across all heads.
        """
        # Run forward pass up to zone embedding
        with torch.no_grad():
            B = images.size(0)
            vis_emb, vis_patch_tokens = self.visual_branch(images)
            if self.enable_motion and imu_windows is not None:
                mot_emb, quality_score, state_logits = self.motion_branch(imu_windows)
                vis_patch_tokens = self.quality_gater(vis_patch_tokens, quality_score)
            else:
                mot_emb = torch.zeros(B, 64, device=images.device)
                quality_score = torch.ones(B, 1, device=images.device)
                state_logits = torch.zeros(B, NUM_PLATFORM_STATES, device=images.device)

            if self.enable_env and env_seqs is not None:
                _, env_tokens = self.env_branch(env_seqs)
            else:
                env_tokens = torch.zeros(B, 14, 256, device=images.device)

            if static_feats is None:
                static_feats = torch.zeros(B, 5, device=images.device)

            if self.enable_env or self.enable_motion:
                fused_emb, explain_dict = self.fusion(
                    vis_patch_tokens=vis_patch_tokens,
                    env_tokens=env_tokens,
                    motion_emb=mot_emb,
                    static_feat=static_feats,
                )
            else:
                fused_emb = self.vision_only_proj(vis_emb)
                explain_dict = {}

            if self.enable_gnn and edge_index is not None:
                zone_emb = self.gnn(fused_emb, edge_index)
            else:
                zone_emb = fused_emb

        # Run MC-Dropout over heads
        mc_results = self.heads.predict_with_mc_dropout(
            zone_emb, num_mc_samples=num_mc_samples
        )
        mc_results["pred_quality"] = quality_score
        mc_results["motion_state"] = torch.argmax(state_logits, dim=-1)
        mc_results["explain_dict"] = explain_dict

        return mc_results

