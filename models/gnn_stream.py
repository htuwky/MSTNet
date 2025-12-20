import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 路径修复：确保能从上一级目录导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GNNStream(nn.Module):
    """
    GNNStream 模块：利用图注意力机制捕捉空间拓扑逻辑。
    修正说明：将 Mask 逻辑移至 Softmax 之前，并使用 -1e9 填充以增强数值稳定性。
    """

    def __init__(self):
        super().__init__()
        node_dim = config.GNN_NODE_DIM  # 节点特征维度 (12)
        b_dim = config.BOTTLENECK_DIM_GNN  # 瓶颈层维度 (32)
        num_classes = config.NUM_CLASSES  # 分类数 (2)

        # 1. 节点特征编码：压缩视觉特征
        self.node_encoder = nn.Sequential(
            nn.Linear(config.CLIP_EMBED_DIM, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU()
        )

        # 2. 空间邻近参数：可学习的衰减系数
        self.spatial_gamma = nn.Parameter(torch.tensor(5.0))

        # 3. 注意力投影层
        self.w_q = nn.Linear(node_dim, node_dim)
        self.w_k = nn.Linear(node_dim, node_dim)
        self.w_v = nn.Linear(node_dim, node_dim)

        # 4. 瓶颈层与特征聚合
        self.bottleneck = nn.Sequential(
            nn.Linear(node_dim, b_dim),
            nn.LayerNorm(b_dim),
            nn.ReLU()
        )

        # 5. 独立分类头 (用于深度监督)
        self.classifier = nn.Sequential(
            nn.Linear(b_dim, b_dim // 2),
            nn.ReLU(),
            nn.Linear(b_dim // 2, num_classes)
        )

    def forward(self, local_feat, coords, mask=None):
        """
        Args:
            local_feat: [Batch, Seq, 512] 局部视觉特征
            coords: [Batch, Seq, 3] 坐标信息 (x, y, t)
            mask: [Batch, Seq] 填充掩码 (True 代表填充位)
        """
        h = self.node_encoder(local_feat)  # [Batch, Seq, 12]

        # 显存优化：计算距离矩阵
        with torch.no_grad():
            dist = torch.cdist(coords[:, :, :2], coords[:, :, :2], p=2)  # [B, S, S]

        # 空间衰减系数：控制图连接紧密度
        spatial_adj = torch.exp(-dist * torch.abs(self.spatial_gamma))

        # 计算 Q, K, V
        q, k, v = self.w_q(h), self.w_k(h), self.w_v(h)

        # 计算原始注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (h.shape[-1] ** 0.5)

        # --- 掩码逻辑修正：在 Softmax 之前填充 -inf ---
        if mask is not None:
            # 扩展 mask 维度以匹配 [B, S, S] 的注意力矩阵
            # 将填充位置（True）设为极小值
            m = mask.unsqueeze(1).expand(-1, h.size(1), -1)
            attn_scores = attn_scores.masked_fill(m, float('-1e9'))

        # 计算最终注意力权重并结合空间先验
        attn_weights = F.softmax(attn_scores, dim=-1) * spatial_adj

        # 特征聚合与瓶颈层压缩
        out = self.bottleneck(torch.matmul(attn_weights, v))  # [B, S, 32]

        # 掩码均值池化 (Masked Mean Pooling)
        if mask is not None:
            mask_v = (~mask).unsqueeze(-1).float()  # [B, S, 1]
            pooled = (out * mask_v).sum(dim=1) / mask_v.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)

        # 返回独立分类 Logits 和 特征向量
        logits = self.classifier(pooled)
        return logits, pooled