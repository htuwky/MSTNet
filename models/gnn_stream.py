import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GNNStream(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = config.HIDDEN_DIM  # 128
        self.node_dim = 12
        self.bottleneck_dim = 32
        dropout_rate = config.DROPOUT
        num_classes = config.NUM_CLASSES

        # 1. 节点特征编码：强行压缩至 12 维，捕捉模糊语义
        self.node_encoder = nn.Sequential(
            nn.Linear(config.CLIP_EMBED_DIM, self.node_dim),
            nn.LayerNorm(self.node_dim),
            nn.ReLU()
        )

        # 2. 空间邻近参数：可学习的衰减系数，控制图连接的紧密度
        self.spatial_gamma = nn.Parameter(torch.tensor(5.0))

        self.w_query = nn.Linear(self.node_dim, self.node_dim)
        self.w_key = nn.Linear(self.node_dim, self.node_dim)
        self.w_value = nn.Linear(self.node_dim, self.node_dim)

        # 3. 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Linear(self.node_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 4. 独立分类头 (辅助监督)
        self.classifier = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim // 2),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim // 2, num_classes)
        )

    def forward(self, local_feat, coords, mask=None):
        batch_size, seq_len, _ = local_feat.shape
        h = self.node_encoder(local_feat)

        # 空间辅助注意力
        pos = coords[:, :, :2]  # (x, y)
        dist_matrix = torch.cdist(pos, pos, p=2)  # [B, S, S]

        # 核心修正：使用可学习参数并取绝对值防止数值溢出
        spatial_adj = torch.exp(-dist_matrix * torch.abs(self.spatial_gamma))

        q, k, v = self.w_query(h), self.w_key(h), self.w_value(h)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (h.shape[-1] ** 0.5)

        # 将视觉相似度与空间邻近性融合
        attn_weights = F.softmax(attn_scores, dim=-1) * spatial_adj

        if mask is not None:
            m = mask.unsqueeze(1).expand(-1, seq_len, -1)
            attn_weights = attn_weights.masked_fill(m, 0)

        h_graph = torch.matmul(attn_weights, v)
        out = self.bottleneck(h_graph)

        if mask is not None:
            mask_valid = (~mask).unsqueeze(-1).float()
            pooled = (out * mask_valid).sum(dim=1) / mask_valid.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)

        # 为了适配主模型合体，返回 (logits, pooled_feat)
        logits = self.classifier(pooled)
        return logits, pooled