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
        node_dim = config.GNN_NODE_DIM  # 12
        b_dim = config.BOTTLENECK_DIM_GNN  # 32

        self.node_encoder = nn.Sequential(
            nn.Linear(config.CLIP_EMBED_DIM, node_dim), nn.LayerNorm(node_dim), nn.ReLU()
        )
        self.spatial_gamma = nn.Parameter(torch.tensor(5.0))
        self.w_q = nn.Linear(node_dim, node_dim)
        self.w_k = nn.Linear(node_dim, node_dim)
        self.w_v = nn.Linear(node_dim, node_dim)

        self.bottleneck = nn.Sequential(
            nn.Linear(node_dim, b_dim), nn.LayerNorm(b_dim), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(b_dim, b_dim // 2), nn.ReLU(), nn.Linear(b_dim // 2, config.NUM_CLASSES)
        )

    def forward(self, local_feat, coords, mask=None):
        h = self.node_encoder(local_feat)
        with torch.no_grad():
            dist = torch.cdist(coords[:, :, :2], coords[:, :, :2], p=2)
        spatial_adj = torch.exp(-dist * torch.abs(self.spatial_gamma))

        q, k, v = self.w_q(h), self.w_k(h), self.w_v(h)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (h.shape[-1] ** 0.5), dim=-1) * spatial_adj
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).expand(-1, h.size(1), -1), 0)

        out = self.bottleneck(torch.matmul(attn, v))
        if mask is not None:
            mask_v = (~mask).unsqueeze(-1).float()
            pooled = (out * mask_v).sum(dim=1) / mask_v.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)
        return self.classifier(pooled), pooled