import torch
import torch.nn as nn
import sys
import os

# 路径修复：确保能从上一级目录导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MotionBlock(nn.Module):
    """
    残差运动块：通过残差连接防止深层网络梯度消失。
    作用：捕捉细微的速度变化，并保持训练稳定性。
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
    Motion Stream (运动流) - 完整版：
    输入维度：[Batch, Seq, 6] (包含：u_loc, v_loc, u_glob, v_glob, vx, vy)
    核心补丁：增加了入口 LayerNorm，用于将巨大的原始物理量级（±40）压缩到稳定区间。
    """

    def __init__(self):
        super().__init__()

        # 从 config 读取统一配置
        input_dim = 6
        hidden_dim = config.HIDDEN_DIM  # 128
        bottleneck_dim = config.BOTTLENECK_DIM  # 64
        dropout_rate = config.DROPOUT  # 0.5
        num_classes = config.NUM_CLASSES  # 2

        # 1. 入口标准化与初步投影
        # 这里的 input_norm 是解决量级冲突的关键
        self.input_norm = nn.LayerNorm(input_dim)

        self.input_proj = nn.Sequential(
            self.input_norm,  # 先标准化输入
            nn.Linear(input_dim, hidden_dim),  # 再投影到隐藏层
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 2. 特征提取层：使用两个残差块深度挖掘运动模式
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
            x: [Batch, Seq, 6] - 6维原始运动特征
            mask: [Batch, Seq] - Padding Mask (True表示填充部分)
        """
        # --- Step 1: 标准化、投影与残差特征提取 ---
        x = self.input_proj(x)  # [B, S, 128]
        x = self.res_block1(x)
        x = self.res_block2(x)

        # --- Step 2: 压缩到瓶颈维度 (64维) ---
        x = self.bottleneck(x)  # [B, S, 64]

        # --- Step 3: 时序池化 (排除填充位) ---
        if mask is not None:
            # mask 是 True 的地方代表 Padding，需要取反
            mask_valid = (~mask).unsqueeze(-1).float()  # [B, S, 1]
            # 加权求和并除以有效长度
            pooled = (x * mask_valid).sum(dim=1) / mask_valid.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)  # [B, 64]

        # --- Step 4: 输出 ---
        logits = self.classifier(pooled)  # [B, 2]

        return logits, pooled  # 返回 logits 用于预训练，pooled 用于主模型门控注入


# ================= 单元测试 =================
if __name__ == "__main__":
    # 模拟 config 环境进行测试
    model = MotionStream()
    print("✅ MSTNet Motion Stream (完整补丁版) 构建成功!")

    # 模拟原始巨大数值输入 (Batch=2, Seq=512, Dim=6)
    fake_input = torch.randn(2, 512, 6) * 40.0
    fake_mask = torch.zeros(2, 512, dtype=torch.bool)
    fake_mask[1, 256:] = True  # 第二个样本一半是填充

    logits, features = model(fake_input, fake_mask)
    print(f"输入量级: {fake_input.abs().mean().item():.2f}")
    print(f"特征量级: {features.abs().mean().item():.2f} (已成功平滑)")
    print(f"输出维度: Logits {logits.shape}, Features {features.shape}")