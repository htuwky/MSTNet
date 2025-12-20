import torch
import torch.nn as nn


class MSTNetLoss(nn.Module):
    def __init__(self):
        super(MSTNetLoss, self).__init__()
        # 使用标准交叉熵作为基础损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output_dict, labels):
        """
        计算 MSTNet 的自动加权复合损失
        Args:
            output_dict: MSTNet.forward 返回的字典，包含：
                        - 'logits': 字典 {"main": ..., "temporal": ..., "motion": ..., "gnn": ...}
                        - 'log_vars': 长度为 4 的 nn.Parameter (来自模型内部)
            labels: 真实标签
        Returns:
            total_loss: 自动加权后的总损失
            loss_main: 最终诊断原始损失
            loss_t: 时序流原始损失
            loss_m: 运动流原始损失
            loss_g: 结构流原始损失
        """
        logits_dict = output_dict['logits']
        log_vars = output_dict['log_vars']

        # 1. 计算四个任务的原始交叉熵损失
        loss_main = self.criterion(logits_dict['main'], labels)
        loss_t = self.criterion(logits_dict['temporal'], labels)
        loss_m = self.criterion(logits_dict['motion'], labels)
        loss_g = self.criterion(logits_dict['gnn'], labels)

        # 将损失组合为 Tensor 方便批量计算
        losses = torch.stack([loss_main, loss_t, loss_m, loss_g])

        # 2. 自动加权逻辑 (Uncertainty Weighting)
        # 公式: Loss = exp(-log_var) * Loss_task + log_var
        # 这种设计可以在辅助任务收敛后自动降低其权重，聚焦于主任务
        weighted_losses = torch.exp(-log_vars) * losses + log_vars

        # 3. 求和得到最终梯度下降的 Loss
        total_loss = weighted_losses.sum()

        return total_loss, loss_main, loss_t, loss_m, loss_g


# 为了保持与之前脚本调用习惯一致，提供一个封装函数
def compute_mstnet_loss(output_dict, labels):
    """
    便捷调用函数
    """
    loss_module = MSTNetLoss()
    return loss_module(output_dict, labels)