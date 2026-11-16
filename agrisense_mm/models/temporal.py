"""
AgriSense-MM: Environmental Temporal Branch.
Bi-LSTM / Temporal Transformer over 14-day sequences of meteorological
and soil variables (temp, humidity, soil moisture, rain, solar, wind, NDVI).
Outputs:
1. 256-d environmental embedding
2. Temporal sequence tokens [B, 14, 256] for cross-attention
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class EnvironmentalTemporalEncoder(nn.Module):
    def __init__(
        self,
        in_features: int = 7,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 2,
        use_transformer: bool = False,
        dropout_p: float = 0.15,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.out_dim = out_dim

        # Input linear projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        if use_transformer:
            self.pos_enc = PositionalEncoding(hidden_dim, max_len=32)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout_p,
                activation="relu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.out_proj = nn.Linear(hidden_dim, out_dim)
        else:
            # 2-layer Bidirectional LSTM: hidden_dim * 2 = 128 * 2 = 256
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_p if num_layers > 1 else 0.0,
            )
            self.out_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
            )

        # Temporal attention pooling layer (learns which days matter most)
        self.attn_pool = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Environmental tensor of shape [B, 14, 7]

        Returns:
            summary_emb: [B, 256] global temporal embedding
            seq_tokens:  [B, 14, 256] per-day sequence tokens
        """
        # [B, 14, 7] -> [B, 14, hidden_dim]
        h = self.input_proj(x)

        if self.use_transformer:
            h = self.pos_enc(h)
            h = self.transformer(h)  # [B, 14, hidden_dim]
            seq_tokens = self.out_proj(h)  # [B, 14, out_dim]
        else:
            lstm_out, _ = self.lstm(h)  # [B, 14, hidden_dim * 2]
            seq_tokens = self.out_proj(lstm_out)  # [B, 14, out_dim]

        # Temporal attention pooling: compute weight per day
        attn_weights = torch.softmax(self.attn_pool(seq_tokens), dim=1)  # [B, 14, 1]
        summary_emb = torch.sum(seq_tokens * attn_weights, dim=1)        # [B, out_dim]

        return summary_emb, seq_tokens

