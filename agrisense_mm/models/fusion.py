"""
AgriSense-MM: Cross-Attention Multimodal Fusion.
Fuses visual tokens (queries) with environmental, motion, and GPS/soil tokens (keys/values).
Includes:
- Multi-head cross-attention (4 heads, 2 layers, residual + LayerNorm)
- Modality dropout (p=0.15) for missing-sensor robustness
- Attention weight extraction for per-modality attribution explainability
- Produces fused representation [B, 512]
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class ModalityDropout(nn.Module):
    """
    Randomly drops individual modalities during training with probability p_drop.
    Guarantees missing-modality robustness when a physical sensor drops out in the field.
    """
    def __init__(self, p_drop: float = 0.15):
        super().__init__()
        self.p_drop = p_drop

    def forward(
        self,
        vis_tokens: torch.Tensor,
        env_tokens: torch.Tensor,
        motion_tokens: torch.Tensor,
        static_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.p_drop <= 0.0:
            return vis_tokens, env_tokens, motion_tokens, static_tokens

        B = vis_tokens.size(0)
        device = vis_tokens.device

        # Mask per sample in batch: keep at least one modality alive
        drop_vis = (torch.rand(B, 1, 1, device=device) < self.p_drop)
        drop_env = (torch.rand(B, 1, 1, device=device) < self.p_drop)
        drop_mot = (torch.rand(B, 1, 1, device=device) < self.p_drop)

        # Ensure we never drop all three modalities simultaneously
        all_dropped = drop_vis & drop_env & drop_mot
        drop_vis = drop_vis & (~all_dropped)

        vis_tokens = vis_tokens.masked_fill(drop_vis, 0.0)
        env_tokens = env_tokens.masked_fill(drop_env, 0.0)
        motion_tokens = motion_tokens.masked_fill(drop_mot, 0.0)

        return vis_tokens, env_tokens, motion_tokens, static_tokens


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout_p),
        )

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention: Query = Visual tokens, Key/Value = Environmental + Motion + Soil tokens
        q_norm = self.norm1(query)
        kv_norm = self.norm1(key_value)

        attn_out, attn_weights = self.mha(
            q_norm, kv_norm, kv_norm, need_weights=True, average_attn_weights=False
        )
        x = query + attn_out

        # FFN with residual connection
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class MultimodalCrossAttentionFusion(nn.Module):
    def __init__(
        self,
        fusion_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        env_dim: int = 256,
        motion_dim: int = 64,
        static_dim: int = 5,
        modality_dropout_p: float = 0.15,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.modality_dropout = ModalityDropout(p_drop=modality_dropout_p)

        # Projections to align all modalities into fusion_dim (512)
        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Learned modality type embeddings to distinguish token sources
        self.token_type_vis = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        self.token_type_env = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        self.token_type_mot = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        self.token_type_sta = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)

        # Stacked Cross-Attention Layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                ffn_dim=fusion_dim * 2,
                dropout_p=dropout_p,
            )
            for _ in range(num_layers)
        ])

        # Final aggregation and output normalization
        self.output_norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        vis_patch_tokens: torch.Tensor,   # [B, 49, 512]
        env_tokens: torch.Tensor,         # [B, 14, 256] or [B, 1, 256]
        motion_emb: torch.Tensor,         # [B, 64]
        static_feat: torch.Tensor,        # [B, 5]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            vis_patch_tokens: [B, N_vis, 512]
            env_tokens:       [B, N_env, 256]
            motion_emb:       [B, 64]
            static_feat:      [B, 5]

        Returns:
            fused_vector: [B, 512]
            explain_dict: Dict with attention attribution maps
        """
        B = vis_patch_tokens.size(0)

        # Project modalities to common fusion_dim (512)
        p_env = self.env_proj(env_tokens)                      # [B, N_env, 512]
        p_mot = self.motion_proj(motion_emb).unsqueeze(1)      # [B, 1, 512]
        p_sta = self.static_proj(static_feat).unsqueeze(1)     # [B, 1, 512]

        # Add modality identity embeddings
        vis_tokens = vis_patch_tokens + self.token_type_vis
        p_env = p_env + self.token_type_env
        p_mot = p_mot + self.token_type_mot
        p_sta = p_sta + self.token_type_sta

        # Apply Modality Dropout
        vis_tokens, p_env, p_mot, p_sta = self.modality_dropout(
            vis_tokens, p_env, p_mot, p_sta
        )

        # Concatenate multi-sensor context into Key/Value sequence
        # Shape: [B, N_env + 1 + 1, 512] = [B, 16, 512]
        kv_context = torch.cat([p_env, p_mot, p_sta], dim=1)

        # Pass through cross-attention layers
        curr_q = vis_tokens
        last_attn_weights = None

        for layer in self.layers:
            curr_q, attn_weights = layer(curr_q, kv_context)
            last_attn_weights = attn_weights  # [B, num_heads, N_vis, N_ctx]

        # Global average pool over visual tokens that attended to sensor context
        fused_tokens = self.output_norm(curr_q)       # [B, N_vis, 512]
        fused_vector = torch.mean(fused_tokens, dim=1) # [B, 512]

        # Calculate per-modality attribution weights for explainability
        # last_attn_weights: [B, heads, N_vis, N_ctx]
        # Average across heads and query tokens to get 1D weight over context tokens
        mean_attn = torch.mean(last_attn_weights, dim=(1, 2))  # [B, N_ctx]
        n_env = p_env.size(1)

        attn_env = torch.sum(mean_attn[:, :n_env], dim=1, keepdim=True)
        attn_mot = mean_attn[:, n_env : n_env + 1]
        attn_sta = mean_attn[:, n_env + 1 : n_env + 2]

        # Normalize across modalities for clean percentage attribution
        modality_total = attn_env + attn_mot + attn_sta + 1e-6
        explain_dict = {
            "attn_weights_raw": last_attn_weights,
            "pct_env": (attn_env / modality_total).squeeze(1),
            "pct_motion": (attn_mot / modality_total).squeeze(1),
            "pct_soil_gps": (attn_sta / modality_total).squeeze(1),
        }

        return fused_vector, explain_dict

