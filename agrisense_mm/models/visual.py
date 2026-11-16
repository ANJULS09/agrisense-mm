"""
AgriSense-MM: Visual Branch.
EfficientNet-B0 / ViT image encoder that outputs:
1. 512-d global visual embedding
2. 49 patch/spatial tokens (7x7 grid projected to 512-d) for cross-attention
3. Hooks for Grad-CAM explainability
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class VisualEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load EfficientNet-B0 backbone
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            efficientnet = models.efficientnet_b0(weights=weights)
        except Exception:
            # Offline / cache fallback
            efficientnet = models.efficientnet_b0(weights=None)

        # Feature extractor up to last conv layer (yields [B, 1280, 7, 7] for 224x224 input)
        self.features = efficientnet.features
        in_channels = 1280  # EfficientNet-B0 final stage channels

        # Spatial patch token projection: [B, 1280, H, W] -> [B, embedding_dim, H, W]
        self.token_proj = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Global pooling and embedding projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout_p),
        )

        # Grad-CAM hook variables
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_gradcam_hook()

    def _register_gradcam_hook(self):
        """Registers forward and backward hooks on the last conv stage for Grad-CAM."""
        target_layer = self.features[-1]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: RGB image tensor of shape [B, 3, 224, 224] (normalized ImageNet style)

        Returns:
            global_embedding: [B, embedding_dim] (512-d)
            patch_tokens: [B, num_patches, embedding_dim] (e.g. [B, 49, 512])
        """
        # Feature extraction: [B, 1280, 7, 7]
        feats = self.features(x)

        # Project to embedding dim: [B, 512, 7, 7]
        proj_feats = self.token_proj(feats)
        B, C, H, W = proj_feats.shape

        # Patch tokens for cross-attention: [B, 49, 512]
        patch_tokens = proj_feats.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Global pooled vector
        pooled = self.global_pool(proj_feats).flatten(1)  # [B, 512]
        global_embedding = self.fc_proj(pooled)           # [B, 512]

        return global_embedding, patch_tokens

