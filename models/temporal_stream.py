import torch
import torch.nn as nn
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class FourierEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, scale=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scale = scale if scale is not None else config.FOURIER_SCALE
        B_mat = torch.randn(input_dim, hidden_dim // 2) * self.scale
        self.register_buffer('B', B_mat)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.B
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.proj(out)

class TemporalStream(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = config.HIDDEN_DIM # 128
        dropout_rate = config.DROPOUT

        self.local_proj = nn.Sequential(
            nn.Linear(config.CLIP_EMBED_DIM, config.BOTTLENECK_DIM),
            nn.LayerNorm(config.BOTTLENECK_DIM),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.global_proj = nn.Sequential(
            nn.Linear(config.CLIP_EMBED_DIM, config.BOTTLENECK_DIM),
            nn.LayerNorm(config.BOTTLENECK_DIM),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.physio_mapper = FourierEmbedding(config.PHYSIO_INPUT_DIM, config.BOTTLENECK_DIM)
        self.physio_dropout = nn.Dropout(dropout_rate)

        self.fusion_proj = nn.Sequential(
            nn.Linear(config.BOTTLENECK_DIM * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=config.NUM_HEADS, dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, config.NUM_CLASSES)
        )

    def extract_features(self, local, global_v, physio, mask=None):
        x_l = self.local_proj(local)
        x_g = self.global_proj(global_v)
        x_p = self.physio_dropout(self.physio_mapper(physio))
        x_fused = self.fusion_proj(torch.cat([x_l, x_g, x_p], dim=-1))
        out = self.transformer(x_fused, src_key_padding_mask=mask)
        if mask is not None:
            mask_v = (~mask).unsqueeze(-1).float()
            return (out * mask_v).sum(dim=1) / mask_v.sum(dim=1).clamp(min=1e-9)
        return out.mean(dim=1)

    def forward(self, local, global_v, physio, mask=None):
        feat = self.extract_features(local, global_v, physio, mask)
        logits = self.classifier(feat)
        return logits, feat