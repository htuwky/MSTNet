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
    原理：通过引入高频的正弦/余弦波，捕捉微小的位置抖动。
    """

    def __init__(self, input_dim, hidden_dim, scale=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # B 矩阵：随机高斯分布频率矩阵，维度是 hidden_dim // 2
        self.register_buffer('B', torch.randn(input_dim, hidden_dim // 2) * scale)

        # 最后的线性投影
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # 1. 投影到频率空间
        x_proj = 2 * math.pi * x @ self.B

        # 2. 计算 sin 和 cos 并拼接
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # 3. 线性融合
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
        # 局部视觉分支 [B, S, 512] -> [B, S, 64]
        self.local_proj = nn.Sequential(
            nn.Linear(clip_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 全局视觉分支 [B, S, 512] -> [B, S, 64]
        self.global_proj = nn.Sequential(
            nn.Linear(clip_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 生理坐标分支 (x, y, t) -> [B, S, 64]
        self.physio_mapper = FourierEmbedding(
            input_dim=physio_in_dim,
            hidden_dim=bottleneck_dim,
            scale=config.FOURIER_SCALE
        )
        self.physio_dropout = nn.Dropout(dropout_rate)

        # ================= 2. 拼接融合层 =================
        # 维度 = 64*3 = 192 -> 128 (主干维度)
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

        # ================= 4. 分类头 (用于独立流预训练) =================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def extract_features(self, local, global_v, physio, mask=None):
        """
        核心方法：提取池化后的 128 维特征，供主模型 MSTNet 门控注入使用。
        """
        # Step 1: 压缩
        x_local = self.local_proj(local)   # [B, S, 64]
        x_global = self.global_proj(global_v) # [B, S, 64]
        x_physio = self.physio_mapper(physio) # [B, S, 64]
        x_physio = self.physio_dropout(x_physio)

        # Step 2: 融合
        x_concat = torch.cat([x_local, x_global, x_physio], dim=-1) # [B, S, 192]
        x_fused = self.fusion_proj(x_concat) # [B, S, 128]

        # Step 3: Transformer 编码
        out = self.transformer(x_fused, src_key_padding_mask=mask) # [B, S, 128]

        # Step 4: 掩码均值池化 (Masked Mean Pooling)
        if mask is not None:
            mask_valid = (~mask).unsqueeze(-1).float() # [B, S, 1]
            out = (out * mask_valid).sum(dim=1) / mask_valid.sum(dim=1).clamp(min=1e-9)
        else:
            out = out.mean(dim=1) # [B, 128]

        return out

    def forward(self, local, global_v, physio, mask=None):
        """
        前向传播逻辑：返回分类 Logits。
        """
        feat = self.extract_features(local, global_v, physio, mask)
        logits = self.classifier(feat) # [B, 2]
        return logits


# ================= 单元测试 (Unit Test) =================
if __name__ == "__main__":
    model = TemporalStream()
    print("✅ MSTNet Temporal Stream (主干版) 构建成功!")

    B, S = 2, config.MAX_SEQ_LEN
    fake_local = torch.randn(B, S, config.CLIP_EMBED_DIM)
    fake_global = torch.randn(B, S, config.CLIP_EMBED_DIM)
    fake_physio = torch.randn(B, S, config.PHYSIO_INPUT_DIM)

    fake_mask = torch.zeros(B, S, dtype=torch.bool)
    fake_mask[1, S // 2:] = True

    # 测试特征提取
    feat = model.extract_features(fake_local, fake_global, fake_physio, fake_mask)
    print(f"特征提取形状: {feat.shape} (预期 [2, 128])")

    # 测试完整前向
    output = model(fake_local, fake_global, fake_physio, fake_mask)
    print(f"分类输出形状: {output.shape} (预期 [2, 2])")