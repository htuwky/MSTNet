import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .temporal_stream import TemporalStream
from .motion_stream import MotionStream
from .gnn_stream import GNNStream


class MSTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_stream = TemporalStream()
        self.motion_stream = MotionStream()
        self.gnn_stream = GNNStream()

        # 可学习的 Loss 权重 (log_var)
        # 初始化为 0，对应的精度权重为 exp(0)=1
        self.loss_log_vars = nn.Parameter(torch.zeros(4))

        aux_dim = config.BOTTLENECK_DIM_MOTION + config.BOTTLENECK_DIM_GNN
        self.gate_generator = nn.Sequential(
            nn.Linear(aux_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        )

    def forward(self, batch_data):
        mask = batch_data['mask']

        # 1. 获取所有流的 Logits 和 Features
        t_logits, t_feat = self.temporal_stream(
            batch_data['temporal_input']['local'],
            batch_data['temporal_input']['global'],
            batch_data['temporal_input']['physio'],
            mask
        )
        m_logits, m_feat = self.motion_stream(batch_data['motion_input'], mask)
        g_logits, g_feat = self.gnn_stream(
            batch_data['gnn_input']['local'],
            batch_data['gnn_input']['coords'],
            mask
        )

        # 2. 门控注入
        gate = self.gate_generator(torch.cat([m_feat, g_feat], dim=-1))
        final_feat = t_feat * (1 + gate)
        main_logits = self.classifier(final_feat)

        # 返回字典，包含所有诊断结果和 Loss 权重参数
        return {
            "logits": {
                "main": main_logits,
                "temporal": t_logits,
                "motion": m_logits,
                "gnn": g_logits
            },
            "log_vars": self.loss_log_vars
        }