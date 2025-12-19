import torch
import torch.nn as nn
import sys
import os

# 路径修复：确保能从上一级目录导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MotionBlock(nn.Module):
    """
    残差运动块：通过残差连接防止深层网络梯度消失，
    并能够同时捕捉到细微的速度变化和大尺度的运动偏差。
    """

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
        return x + self.net(x)  # 残差连接


class MotionStream(nn.Module):
    """
    Motion Stream (运动流)：
    输入维度：[Batch, Seq, 6]
    (6维包含：局部光流u/v, 全局光流u/v, 眼动速度x/y)
    作用：判断眼动速度与物体/背景速度是否匹配。
    """

    def __init__(self):
        super().__init__()

        # 从 config 读取配置
        # 输入维度固定为 6 (光流4维 + 眼动速度2维)
        input_dim = 6
        hidden_dim = config.HIDDEN_DIM  # 128
        bottleneck_dim = config.BOTTLENECK_DIM  # 64
        dropout_rate = config.DROPOUT  # 0.5
        num_classes = config.NUM_CLASSES  # 2

        # 1. 初始投影：将 6 维物理特征投影到高维空间
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 2. 特征提取层：使用残差块深度挖掘运动模式
        self.res_block1 = MotionBlock(hidden_dim, dropout_rate)
        self.res_block2 = MotionBlock(hidden_dim, dropout_rate)

        # 3. 瓶颈压缩层 (MSTNet 核心策略：压缩以防止过拟合)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 4. 独立分类头 (用于流的预训练或深度监督)
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim // 2, num_classes)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [Batch, Seq, 6] - 6维运动特征
            mask: [Batch, Seq] - Padding Mask (True表示填充部分)
        """
        # --- Step 1: 投影与残差特征提取 ---
        x = self.input_proj(x)  # [B, S, 128]
        x = self.res_block1(x)
        x = self.res_block2(x)

        # --- Step 2: 压缩到瓶颈维度 ---
        # x: [B, S, 64]
        x = self.bottleneck(x)

        # --- Step 3: 时序池化 (处理异常占位和填充) ---
        # 即使中间有异常值点，我们也对整个序列求平均，
        # 但要通过 mask 排除掉末尾的 Padding 部分。
        if mask is not None:
            mask_valid = (~mask).unsqueeze(-1).float()  # [B, S, 1]
            # 只对真实有效的时间戳点进行均值池化
            pooled = (x * mask_valid).sum(dim=1) / mask_valid.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)  # [B, 64]

        # --- Step 4: 输出分类 Logits ---
        logits = self.classifier(pooled)  # [B, 2]

        return logits, pooled  # 返回 logits 用于预训练，pooled 用于最终的三流融合


# ================= 单元测试 =================
if __name__ == "__main__":
    model = MotionStream()
    print("✅ MSTNet Motion Stream 构建成功!")

    # 模拟输入 (Batch=4, Seq=512, Dim=6)
    B, S, D = 4, config.MAX_SEQ_LEN, 6
    fake_input = torch.randn(B, S, D)

    # 模拟 Mask
    fake_mask = torch.zeros(B, S, dtype=torch.bool)
    fake_mask[2:, S // 2:] = True  # 后两个样本有一半是填充的

    logits, features = model(fake_input, fake_mask)
    print(f"输入形状: {fake_input.shape}")
    print(f"分类输出 (Logits): {logits.shape} (预期 [4, 2])")
    print(f"特征输出 (Bottleneck): {features.shape} (预期 [4, 64])")