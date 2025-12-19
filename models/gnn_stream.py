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
        # --- 修改点：视觉节点压缩至 12，最终输出压缩至 32 ---
        self.node_dim = 12
        self.bottleneck_dim = 32
        dropout_rate = config.DROPOUT

        self.node_encoder = nn.Sequential(
            nn.Linear(config.CLIP_EMBED_DIM, self.node_dim),
            nn.LayerNorm(self.node_dim),
            nn.ReLU()
        )

        self.w_query = nn.Linear(self.node_dim, self.node_dim)
        self.w_key = nn.Linear(self.node_dim, self.node_dim)
        self.w_value = nn.Linear(self.node_dim, self.node_dim)

        self.bottleneck = nn.Sequential(
            nn.Linear(self.node_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, local_feat, coords, mask=None):
        h = self.node_encoder(local_feat)

        # 空间辅助注意力
        pos = coords[:, :, :2]
        dist_matrix = torch.cdist(pos, pos, p=2)
        spatial_adj = torch.exp(-dist_matrix * 5.0)

        q, k, v = self.w_query(h), self.w_key(h), self.w_value(h)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (h.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1) * spatial_adj

        if mask is not None:
            m = mask.unsqueeze(1).expand(-1, local_feat.size(1), -1)
            attn_weights = attn_weights.masked_fill(m, 0)

        h_graph = torch.matmul(attn_weights, v)
        out = self.bottleneck(h_graph)

        if mask is not None:
            mask_valid = (~mask).unsqueeze(-1).float()
            pooled = (out * mask_valid).sum(dim=1) / mask_valid.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)

        return pooled