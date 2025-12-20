import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MotionBlock(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x): return x + self.net(x)


class MotionStream(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = config.HIDDEN_DIM  # 128
        b_dim = config.BOTTLENECK_DIM_MOTION  # 16

        self.input_proj = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(MotionBlock(hidden_dim), MotionBlock(hidden_dim))
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, b_dim), nn.LayerNorm(b_dim), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(b_dim, b_dim // 2), nn.ReLU(), nn.Linear(b_dim // 2, config.NUM_CLASSES)
        )

    def forward(self, x, mask=None):
        x = self.bottleneck(self.res_blocks(self.input_proj(x)))
        if mask is not None:
            mask_v = (~mask).unsqueeze(-1).float()
            pooled = (x * mask_v).sum(dim=1) / mask_v.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        return self.classifier(pooled), pooled