import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 路径修复
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GNNStream(nn.Module):
    """
    GNN Stream: 将眼动轨迹建模为动态图
    输入:
        local_feat: [B, S, 512] (视觉特征)
        coords: [B, S, 3] (x, y, t)
    """

    def __init__(self):
        super().__init__()

        # 配置读取
        hidden_dim = config.HIDDEN_DIM  # 128
        bottleneck_dim = config.BOTTLENECK_DIM  # 64
        dropout_rate = config.DROPOUT
        num_classes = config.NUM_CLASSES

        # 1. 节点特征压缩：将 512 维 CLIP 特征压缩，减小图计算压力
        self.node_encoder = nn.Sequential(
            nn.Linear(config.CLIP_EMBED_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 2. 图注意力层 (Simplified GAT)
        # 我们在这里模拟图卷积逻辑：利用空间距离计算注意力权重
        self.w_query = nn.Linear(hidden_dim, hidden_dim)
        self.w_key = nn.Linear(hidden_dim, hidden_dim)
        self.w_value = nn.Linear(hidden_dim, hidden_dim)

        # 3. 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 4. 独立分类头
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim // 2, num_classes)
        )

    def forward(self, local_feat, coords, mask=None):
        """
        local_feat: [B, S, 512]
        coords: [B, S, 3] (x, y, t)
        mask: [B, S] (Padding Mask)
        """
        batch_size, seq_len, _ = local_feat.shape

        # --- Step 1: 构造节点特征 ---
        # h: [B, S, 128]
        h = self.node_encoder(local_feat)

        # --- Step 2: 构造空间图 (Spatial-Aware Attention) ---
        # 我们利用坐标 coords 算一个距离矩阵，辅助注意力机制
        # dist_mask: [B, S, S] 记录点与点之间的空间欧几里得距离
        pos = coords[:, :, :2]  # 只取 (x, y)
        dist_matrix = torch.cdist(pos, pos, p=2)  # [B, S, S]

        # 空间衰减：距离越远，权重越小 (类似 RBF 核)
        spatial_adj = torch.exp(-dist_matrix * 5.0)

        # --- Step 3: 图卷积操作 (基于注意力的消息传递) ---
        q = self.w_query(h)
        k = self.w_key(h)
        v = self.w_value(h)

        # 经典的 Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (h.shape[-1] ** 0.5)

        # 核心融合：将视觉注意力与空间邻近性(spatial_adj)相乘
        # 这意味着：只有空间上离得近且视觉特征相关的点，才会进行强信息交换
        attn_weights = F.softmax(attn_scores, dim=-1) * spatial_adj

        if mask is not None:
            # 屏蔽填充位，防止 Padding 节点参与构图
            m = mask.unsqueeze(1).expand(-1, seq_len, -1)
            attn_weights = attn_weights.masked_fill(m, 0)

        # 聚合邻居信息: [B, S, 128]
        h_graph = torch.matmul(attn_weights, v)

        # --- Step 4: 瓶颈压缩与池化 ---
        out = self.bottleneck(h_graph)  # [B, S, 64]

        if mask is not None:
            mask_valid = (~mask).unsqueeze(-1).float()
            pooled = (out * mask_valid).sum(dim=1) / mask_valid.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)  # [B, 64]

        # --- Step 5: 分类 ---
        logits = self.classifier(pooled)

        return logits, pooled


# ================= 单元测试 =================
if __name__ == "__main__":
    model = GNNStream()
    print("✅ MSTNet GNN Stream (Spatial-Aware) 构建成功!")

    B, S = 2, config.MAX_SEQ_LEN
    f_local = torch.randn(B, S, 512)
    f_coords = torch.rand(B, S, 3)  # (x, y, t)
    f_mask = torch.zeros(B, S, dtype=torch.bool)

    logits, features = model(f_local, f_coords, f_mask)
    print(f"输出 Logits 形状: {logits.shape}")  # [2, 2]
    print(f"输出 Bottleneck 形状: {features.shape}")  # [2, 64]