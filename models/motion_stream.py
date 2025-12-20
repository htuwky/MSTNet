import torch
import torch.nn as nn
import sys
import os

# 路径修复：确保能从上一级目录导入 config
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
        hidden_dim = config.HIDDEN_DIM  # 128
        b_dim = config.BOTTLENECK_DIM_MOTION  # 16

        # --- 优化后的输入层逻辑 ---
        # 移除 nn.LayerNorm(6)，改为先 Linear 扩维至 128 再进行 LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 先将 6 维输入映射到 128 维
            nn.LayerNorm(hidden_dim),  # 在 128 维特征空间进行归一化
            nn.ReLU()
        )

        # 时序残差块
        self.res_blocks = nn.Sequential(
            MotionBlock(hidden_dim),
            MotionBlock(hidden_dim)
        )

        # 瓶颈压缩层
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, b_dim),
            nn.LayerNorm(b_dim),
            nn.ReLU()
        )

        # 独立分类头 (用于深度监督)
        self.classifier = nn.Sequential(
            nn.Linear(b_dim, b_dim // 2),
            nn.ReLU(),
            nn.Linear(b_dim // 2, config.NUM_CLASSES)
        )

    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: [Batch, Seq, 6] 输入特征 (包含局部光流、全局光流、眼动速度)
            mask: [Batch, Seq] 填充掩码
        """
        # 特征提取
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.bottleneck(x)

        # 掩码均值池化 (Masked Mean Pooling)
        if mask is not None:
            mask_v = (~mask).unsqueeze(-1).float()
            pooled = (x * mask_v).sum(dim=1) / mask_v.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        # 返回诊断结果和特征向量
        return self.classifier(pooled), pooled