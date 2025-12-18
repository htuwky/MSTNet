import torch
import torch.nn as nn
import math
import sys
import os

# 路径黑魔法：确保能从上一级目录导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FourierEmbedding(nn.Module):
    """
    傅里叶特征映射 (Fourier Feature Mapping)
    作用：将低维坐标 (x, y, t) 映射到高维空间。
    原理：神经网络倾向于学习低频函数(Spectral Bias)，通过人为引入高频的正弦/余弦波，
    可以让模型敏锐地捕捉到"微小的位置抖动"和"精确的时间节点"。
    """

    def __init__(self, input_dim, hidden_dim, scale=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # B 矩阵：随机的高斯分布频率矩阵
        # register_buffer 意味着它不是可训练参数(固定死的)，但在保存模型时会被保存下来
        # 维度是 hidden_dim // 2，因为后面 cat([sin, cos]) 会把维度翻倍
        self.register_buffer('B', torch.randn(input_dim, hidden_dim // 2) * scale)

        # 最后的线性投影，用来融合频率特征，调整到目标维度
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [Batch, Seq, input_dim] -> (x, y, t_norm)

        # 1. 投影到频率空间: 2 * pi * x * B
        x_proj = 2 * math.pi * x @ self.B

        # 2. 计算 sin 和 cos，并拼接: [sin(v), cos(v)]
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # 3. 线性融合
        return self.proj(out)


class TemporalStream(nn.Module):
    def __init__(self):
        super().__init__()

        # ================= 配置读取 (解耦) =================
        clip_dim = config.CLIP_EMBED_DIM  # 512
        bottleneck_dim = config.BOTTLENECK_DIM  # 64 (小样本核心优化)
        physio_in_dim = config.PHYSIO_INPUT_DIM  # 3 (x, y, t)
        hidden_dim = config.HIDDEN_DIM  # 128
        num_heads = config.NUM_HEADS  # 4
        num_layers = config.NUM_LAYERS  # 2
        dropout_rate = config.DROPOUT  # 0.5 (强力Dropout)
        num_classes = config.NUM_CLASSES  # 2

        # ================= 1. 瓶颈压缩层 (The Bottlenecks) =================
        # 策略：先降维，再Dropout，防止过拟合

        # 分支 A: 局部视觉 (Local) [B, S, 512] -> [B, S, 64]
        self.local_proj = nn.Sequential(
            nn.Linear(clip_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),  # 加个 LN 统一分布
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 分支 B: 全局视觉 (Global) [B, S, 512] -> [B, S, 64]
        # 注意：这里处理的是序列(Sequence)，每一帧背景都在变
        self.global_proj = nn.Sequential(
            nn.Linear(clip_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 分支 C: 生理/坐标 (Physio) [B, S, 3] -> [B, S, 64]
        self.physio_mapper = FourierEmbedding(
            input_dim=physio_in_dim,
            hidden_dim=bottleneck_dim,
            scale=config.FOURIER_SCALE
        )
        self.physio_dropout = nn.Dropout(dropout_rate)

        # ================= 2. 拼接融合层 (Fusion) =================
        # 拼接维度 = 64(Local) + 64(Global) + 64(Physio) = 192
        # 这个 192 是 64 的倍数，GPU 计算非常快
        concat_dim = bottleneck_dim * 3

        self.fusion_proj = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),  # 192 -> 128
            nn.LayerNorm(hidden_dim),  # 融合后必须归一化
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # ================= 3. 时序建模 (Transformer) =================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # 128
            nhead=num_heads,  # 4
            dim_feedforward=hidden_dim * 4,  # 512
            dropout=dropout_rate,  # 0.5
            activation='gelu',  # GELU 通常比 ReLU 平滑
            batch_first=True,
            norm_first=True  # Pre-Norm 结构，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ================= 4. 分类头 (Classifier) =================
        # 简单的两层 MLP 进行分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 初始化权重 (对小样本训练至关重要)
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化，帮助模型在早期快速收敛"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, local, global_v, physio, mask=None):
        """
        前向传播逻辑
        Args:
            local:    [Batch, Seq, 512] - 局部视觉特征
            global_v: [Batch, Seq, 512] - 动态全局背景特征
            physio:   [Batch, Seq, 3]   - (x, y, t_norm)
            mask:     [Batch, Seq]      - Padding Mask (True代表是填充值)
        Returns:
            logits:   [Batch, 2]
        """

        # --- Step 1: 独立压缩 (Independent Compression) ---
        x_local = self.local_proj(local)  # [B, S, 64]
        x_global = self.global_proj(global_v)  # [B, S, 64]

        x_physio = self.physio_mapper(physio)  # [B, S, 64]
        x_physio = self.physio_dropout(x_physio)

        # --- Step 2: 拼接 (Concatenation) ---
        # 核心升级点：在特征维度拼接，保留"前景 vs 背景"的对比信息
        # x_concat: [B, S, 192]
        x_concat = torch.cat([x_local, x_global, x_physio], dim=-1)

        # --- Step 3: 融合 (Fusion) ---
        # 压缩回 Transformer 喜欢的维度
        # x_fused: [B, S, 128]
        x_fused = self.fusion_proj(x_concat)

        # --- Step 4: Transformer 编码 ---
        # src_key_padding_mask: True 的位置会被 Attention 忽略
        out = self.transformer(x_fused, src_key_padding_mask=mask)

        # --- Step 5: 时序池化 (Masked Mean Pooling) ---
        if mask is not None:
            # mask 是 Bool 类型 (True=Padding)，取反变成 (True=Valid)
            mask_valid = (~mask).unsqueeze(-1).float()  # [B, S, 1]

            # 只对非 Padding 的部分求平均
            # clamp(min=1e-9) 防止除以0
            out = (out * mask_valid).sum(dim=1) / mask_valid.sum(dim=1).clamp(min=1e-9)
        else:
            out = out.mean(dim=1)  # [B, 128]

        # --- Step 6: 分类 ---
        logits = self.classifier(out)  # [B, 2]

        return logits


# ================= 单元测试 (Unit Test) =================
if __name__ == "__main__":
    # 简单的冒烟测试
    model = TemporalStream()
    print("✅ MSTNet Temporal Stream 构建成功!")

    # 模拟输入 (Batch=2, Seq=512)
    B, S = 2, config.MAX_SEQ_LEN
    fake_local = torch.randn(B, S, config.CLIP_EMBED_DIM)
    fake_global = torch.randn(B, S, config.CLIP_EMBED_DIM)
    fake_physio = torch.randn(B, S, config.PHYSIO_INPUT_DIM)  # (x,y,t)

    # 模拟 Mask (假设第2个样本只有一半是真实的)
    fake_mask = torch.zeros(B, S, dtype=torch.bool)
    fake_mask[1, S // 2:] = True

    output = model(fake_local, fake_global, fake_physio, fake_mask)
    print(f"输入形状: Local{fake_local.shape}, Physio{fake_physio.shape}")
    print(f"输出形状: {output.shape} (预期 [2, 2])")