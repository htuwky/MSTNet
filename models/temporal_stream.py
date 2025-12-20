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
    稳定性：引入 LayerNorm 防止正弦波输出量级爆炸。
    """

    def __init__(self, input_dim, hidden_dim, scale=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scale = scale if scale is not None else config.FOURIER_SCALE

        # ==================== 修改部分 ====================
        # 创建一个独立的随机数生成器并固定种子（例如 42）
        # 这样无论在哪个服务器运行，B 矩阵的内容永远一致
        gen = torch.Generator().manual_seed(42)

        # 使用该生成器创建随机矩阵 B
        B_mat = torch.randn(
            input_dim,
            hidden_dim // 2,
            generator=gen
        ) * self.scale
        # =================================================

        self.register_buffer('B', B_mat)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        # x_proj: [Batch, Seq, hidden_dim//2]
        x_proj = 2 * math.pi * x @ self.B
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.proj(out)


class TemporalStream(nn.Module):
    """
    时序主干流：处理 CLIP 视觉特征与生理位置信息。
    """

    def __init__(self):
        super().__init__()

        # ================= 配置读取 (主干 128 维) =================
        hidden_dim = config.HIDDEN_DIM  # 128
        # 规范化修改：明确使用主干流专用的瓶颈维度变量
        bottleneck_dim = config.BOTTLENECK_DIM_TEMPORAL  # 64
        dropout_rate = config.DROPOUT  # 0.5

        # 1. 瓶颈压缩层：统一使用规范化的 bottleneck_dim
        self.local_proj = nn.Sequential(
            nn.Linear(config.CLIP_EMBED_DIM, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.global_proj = nn.Sequential(
            nn.Linear(config.CLIP_EMBED_DIM, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.physio_mapper = FourierEmbedding(config.PHYSIO_INPUT_DIM, bottleneck_dim)
        self.physio_dropout = nn.Dropout(dropout_rate)

        # 2. 拼接融合层
        self.fusion_proj = nn.Sequential(
            nn.Linear(bottleneck_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 3. 时序建模 (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.NUM_HEADS,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)

        # 4. 独立分类头 (用于深度监督)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, config.NUM_CLASSES)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def extract_features(self, local, global_v, physio, mask=None):
        """提取池化后的 128 维特征供主模型合体使用"""
        x_l = self.local_proj(local)
        x_g = self.global_proj(global_v)
        x_p = self.physio_dropout(self.physio_mapper(physio))

        # 融合三方信息并映射至 hidden_dim
        x_fused = self.fusion_proj(torch.cat([x_l, x_g, x_p], dim=-1))

        # Transformer 时序编码
        out = self.transformer(x_fused, src_key_padding_mask=mask)

        # 掩码均值池化 (Masked Mean Pooling)
        if mask is not None:
            mask_v = (~mask).unsqueeze(-1).float()
            return (out * mask_v).sum(dim=1) / mask_v.sum(dim=1).clamp(min=1e-9)
        return out.mean(dim=1)

    def forward(self, local, global_v, physio, mask=None):
        """
        前向传播：同时返回诊断结果和特征。
        """
        feat = self.extract_features(local, global_v, physio, mask)
        logits = self.classifier(feat)
        # 统一返回两个结果，适配深度监督架构
        return logits, feat