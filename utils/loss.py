import torch
import torch.nn as nn

# 1. 定义基础损失函数
# 考虑到您有 400 个样本且类别平衡（健康/患病各 200），使用标准交叉熵
criterion = nn.CrossEntropyLoss()


def compute_mstnet_loss(main_logits, m_logits, g_logits, labels, alpha=0.2, beta=0.2):
    """
    计算 MSTNet 的复合损失函数
    Args:
        main_logits: 最终门控融合后的输出 (来自 MSTNet)
        m_logits: 运动流独立分类头的输出 (来自 MotionStream)
        g_logits: 结构流独立分类头的输出 (来自 GNNStream)
        labels: 真实标签
        alpha: 运动流 Loss 权重
        beta: 结构流 Loss 权重
    """
    # 主任务损失：衡量最终诊断的准确性
    loss_main = criterion(main_logits, labels)

    # 辅助任务 1：强迫运动流学习“视网膜滑移”等物理特征
    loss_motion = criterion(m_logits, labels)

    # 辅助任务 2：强迫 GNN 流学习空间拓扑异常
    loss_gnn = criterion(g_logits, labels)

    # 总损失叠加
    total_loss = loss_main + alpha * loss_motion + beta * loss_gnn

    return total_loss, loss_main, loss_motion, loss_gnn