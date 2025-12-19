import torch
import torch.nn as nn
import sys
import os

# 路径修复
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .temporal_stream import TemporalStream
from .motion_stream import MotionStream
from .gnn_stream import GNNStream


class MSTNet(nn.Module):
    """
    MSTNet 主模型：实现 1主 + 2辅 的门控注入架构
    """

    def __init__(self):
        super().__init__()

        # 1. 实例化三个流模块
        self.temporal_stream = TemporalStream()  # 主干 (128维)
        self.motion_stream = MotionStream()  # 辅助 (16维)
        self.gnn_stream = GNNStream()  # 辅助 (32维)

        # 2. 门控生成器 (Gate Generator)
        # 辅助特征总维度 = 16 (Motion) + 32 (GNN) = 48 维
        aux_dim = 16 + 32
        main_dim = 128  # 对应 TemporalStream 的 hidden_dim

        self.gate_generator = nn.Sequential(
            nn.Linear(aux_dim, main_dim),
            nn.Sigmoid()  # 生成 0~1 的权重向量 G
        )

        # 3. 最终诊断分类器
        self.classifier = nn.Sequential(
            nn.Linear(main_dim, main_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(main_dim // 2, config.NUM_CLASSES)
        )

    def forward(self, batch_data):
        """
        Args:
            batch_data: DataLoader 返回的字典
        """
        # --- A. 提取主干流特征 ---
        # 调用 extract_features 获取池化后的 128 维特征
        t_feat = self.temporal_stream.extract_features(
            batch_data['temporal_input']['local'],
            batch_data['temporal_input']['global'],
            batch_data['temporal_input']['physio'],
            batch_data['mask']
        )  # [Batch, 128]

        # --- B. 提取辅助流特征 ---
        # m_feat 输出为 16 维
        m_feat = self.motion_stream(batch_data['motion_input'], batch_data['mask'])
        # g_feat 输出为 32 维
        g_feat = self.gnn_stream(
            batch_data['gnn_input']['local'],
            batch_data['gnn_input']['coords'],
            batch_data['mask']
        )

        # --- C. 门控注入 (Gated Injection) ---
        # 1. 拼接辅助流特征: [Batch, 48]
        aux_combined = torch.cat([m_feat, g_feat], dim=-1)

        # 2. 生成门控向量 G: [Batch, 128]
        gate = self.gate_generator(aux_combined)

        # 3. 注入逻辑: Feature_final = Feature_temporal * (1 + G)
        # 辅助流发现异常时 G 趋近 1，放大特征；正常时 G 趋近 0，保持原特征。
        final_feat = t_feat * (1 + gate)

        # --- D. 最终分类 ---
        logits = self.classifier(final_feat)

        return logits