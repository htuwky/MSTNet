import torch
import torch.nn as nn
import math
import sys
import os

# 路径修复：确保能从上一级目录导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FourierEmbedding(nn.Module):
    """
    傅里叶特征映射 (Fourier Feature Mapping)
    作用：将低维坐标 (x, y, t) 映射到高维空间。
    修正：引入 LayerNorm 并规范化初始化过程，增强小样本下的收敛稳定性。
    """

    def __init__(self, input_dim, hidden_dim, scale=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 从 config 读取缩放比例
        self.scale = scale if scale is not None else config.FOURIER_SCALE

        # 方案：使用固定的随机初始化，确保编码空间的稳定性
        # 维度是 hidden_dim // 2，后面拼接后翻倍
        B_mat = torch.randn(input_dim, hidden_dim // 2) * self.scale
        self.register_buffer('B', B_mat)

        # 最后的线性投影 + LayerNorm：防止正弦波输出量级爆炸，确保与视觉特征对齐
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        # x: [Batch, Seq, 3] -> (x, y, t_norm)
        # 1. 投影到频率空间: 2 * pi * x * B
        x_proj = 2 * math.pi * x @ self.B

        # 2. 计算 sin 和 cos 并拼接: [sin(v), cos(v)]
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # 3. 线性融合与归一化
        return self.proj(out)


class TemporalStream(nn.Module):
    def __init__(self):
        super().__init__()

        # ================= 配置读取 (主干 128 维) =================
        clip_dim = config.CLIP_EMBED_DIM       # 512
        bottleneck_dim = config.BOTTLENECK_DIM # 64
        physio_in_dim = config.PHYSIO_INPUT_DIM # 3
        hidden_dim = config.HIDDEN_DIM         # 128
        num_heads = config.NUM_HEADS           # 4
        num_layers = config.NUM_LAYERS         # 2
        dropout_rate = config.DROPOUT          # 0.5
        num_classes = config.NUM_CLASSES       # 2

        # ================= 1. 瓶颈压缩层 =================
        self.local_proj = nn.Sequential(
            nn.Linear(clip_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.global_proj = nn.Sequential(
            nn.Linear(clip_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.physio_mapper = FourierEmbedding(
            input_dim=physio_in_dim,
            hidden_dim=bottleneck_dim,
            scale=config.FOURIER_SCALE
        )
        self.physio_dropout = nn.Dropout(dropout_rate)

        # ================= 2. 拼接融合层 =================
        concat_dim = bottleneck_dim * 3
        self.fusion_proj = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # ================= 3. 时序建模 (Transformer) =================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ================= 4. 分类头 (用于辅助监督) =================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def extract_features(self, local, global_v, physio, mask=None):
        """提取池化后的 128 维特征供主模型融合"""
        x_local = self.local_proj(local)
        x_global = self.global_proj(global_v)
        x_physio = self.physio_mapper(physio)
        x_physio = self.physio_dropout(x_physio)

        x_concat = torch.cat([x_local, x_global, x_physio], dim=-1)
        x_fused = self.fusion_proj(x_concat)

        out = self.transformer(x_fused, src_key_padding_mask=mask)

        if mask is not None:
            mask_valid = (~mask).unsqueeze(-1).float()
            out = (out * mask_valid).sum(dim=1) / mask_valid.sum(dim=1).clamp(min=1e-9)
        else:
            out = out.mean(dim=1)

        return out

    def forward(self, local, global_v, physio, mask=None):
        feat = self.extract_features(local, global_v, physio, mask)
        logits = self.classifier(feat)
        return logits