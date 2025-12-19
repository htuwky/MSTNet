import torch
import torch.nn as nn
import sys
import os

# 路径修复：确保能从上一级目录导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from .temporal_stream import TemporalStream
from .motion_stream import MotionStream
from .gnn_stream import GNNStream


class MSTNet(nn.Module):
    """
    MSTNet 主模型 (完整合体版)
    架构：1个主干流 (Temporal) + 2个辅助流 (Motion, GNN)
    机制：门控注入 (Gated Injection) -> Feature_final = T_feat * (1 + Sigmoid(Linear(M_feat + G_feat)))
    """

    def __init__(self):
        super().__init__()

        # 1. 初始化三流模块
        self.temporal_stream = TemporalStream()  # 主干：PFC (128维)
        self.motion_stream = MotionStream()  # 辅助：SC (16维)
        self.gnn_stream = GNNStream()  # 辅助：Parietal (32维)

        # 2. 门控生成器 (Gate Generator)
        # 维度对齐：Motion(16) + GNN(32) = 48维
        # 注意：这些维度应确保与各自子流内部的 bottleneck_dim 一致
        aux_dim = config.BOTTLENECK_DIM_MOTION + config.BOTTLENECK_DIM_GNN
        main_dim = config.HIDDEN_DIM  # 128

        self.gate_generator = nn.Sequential(
            nn.Linear(aux_dim, main_dim),
            nn.LayerNorm(main_dim),  # 保证门控分布稳定
            nn.Sigmoid()  # 将输出锁定在 0~1 之间
        )

        # 3. 最终诊断分类器 (主头)
        self.classifier = nn.Sequential(
            nn.Linear(main_dim, main_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(main_dim // 2, config.NUM_CLASSES)
        )

    def forward(self, batch_data):
        """
        前向传播
        Args:
            batch_data: 包含各流输入和 mask 的字典
        Returns:
            main_logits: 最终融合诊断结果
            m_logits:    运动流独立诊断结果 (用于深度监督 Loss)
            g_logits:    结构流独立诊断结果 (用于深度监督 Loss)
        """
        mask = batch_data['mask']

        # --- A. 提取主干流特征 (128维) ---
        # 使用专用的 extract_features 方法获取池化后的特征
        t_feat = self.temporal_stream.extract_features(
            batch_data['temporal_input']['local'],
            batch_data['temporal_input']['global'],
            batch_data['temporal_input']['physio'],
            mask
        )  # [B, 128]

        # --- B. 提取辅助流特征与 Logits ---
        # 修改后的子流 forward 需返回 (logits, pooled_feat)
        m_logits, m_feat = self.motion_stream(batch_data['motion_input'], mask)  # m_feat: [B, 16]
        g_logits, g_feat = self.gnn_stream(
            batch_data['gnn_input']['local'],
            batch_data['gnn_input']['coords'],
            mask
        )  # g_feat: [B, 32]

        # --- C. 门控注入 (Gated Injection) ---
        # 1. 拼接辅助流特征 [B, 48]
        aux_combined = torch.cat([m_feat, g_feat], dim=-1)

        # 2. 生成 128 维门控向量 G
        gate = self.gate_generator(aux_combined)

        # 3. 核心注入公式：增强主干特征
        # 当辅助流发现“跟丢物体”或“逻辑混乱”时，gate 趋近 1，主干特征被放大
        final_feat = t_feat * (1 + gate)

        # --- D. 输出结果 ---
        main_logits = self.classifier(final_feat)

        # 返回三组 Logits 供多任务 Loss 计算
        return main_logits, m_logits, g_logits


# ================= 单元测试 =================
if __name__ == "__main__":
    # 模拟输入数据字典
    B, S = 2, config.MAX_SEQ_LEN
    fake_batch = {
        'temporal_input': {
            'local': torch.randn(B, S, 512),
            'global': torch.randn(B, S, 512),
            'physio': torch.randn(B, S, 3)
        },
        'motion_input': torch.randn(B, S, 6),
        'gnn_input': {
            'local': torch.randn(B, S, 512),
            'coords': torch.randn(B, S, 3)
        },
        'mask': torch.zeros(B, S, dtype=torch.bool),
        'label': torch.ones(B, dtype=torch.long)
    }

    model = MSTNet()
    main_out, m_out, g_out = model(fake_batch)

    print("✅ MSTNet 合体模型测试通过!")
    print(f"主分类输出: {main_out.shape}")
    print(f"运动流辅助输出: {m_out.shape}")
    print(f"结构流辅助输出: {g_out.shape}")