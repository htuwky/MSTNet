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
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)


class MotionStream(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 6
        hidden_dim = config.HIDDEN_DIM  # 128
        # --- 修改点：bottleneck 压低至 16 ---
        self.bottleneck_dim = config.BOTTLENECK_DIM_MOTION
        dropout_rate = config.DROPOUT

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            self.input_norm,
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.res_block1 = MotionBlock(hidden_dim, dropout_rate)
        self.res_block2 = MotionBlock(hidden_dim, dropout_rate)

        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.bottleneck(x)

        if mask is not None:
            mask_valid = (~mask).unsqueeze(-1).float()
            pooled = (x * mask_valid).sum(dim=1) / mask_valid.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        # 返回特征用于主模型融合
        return pooled