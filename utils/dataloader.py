import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys

# 确保能导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MSTNetDataset(Dataset):
    def __init__(self, mode='train', split_ratio=0.8):
        """
        MSTNet 数据集加载器
        Args:
            mode: 'train' 或 'val'
            split_ratio: 训练集比例
        """
        self.temp_dir = config.TEMP_FEATURE_DIR
        self.max_len = config.MAX_SEQ_LEN  # 512

        # 1. 扫描所有主特征文件 (例如 001.npy, 002.npy)
        # 排除掉带 _motion 的，我们通过主文件关联运动文件
        all_files = sorted([
            f for f in os.listdir(self.temp_dir)
            if f.endswith('.npy') and '_motion' not in f
        ])

        if not all_files:
            raise FileNotFoundError(f"❌ 在 {self.temp_dir} 没找到特征文件，请先运行提取脚本！")

        # 2. 划分训练/验证集
        split_idx = int(len(all_files) * split_ratio)
        if mode == 'train':
            self.file_list = all_files[:split_idx]
        else:
            self.file_list = all_files[split_idx:]

        print(f"📦 MSTNet Dataset [{mode}]: 加载了 {len(self.file_list)} 个样本")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # --- 1. 定位文件路径 ---
        visual_file = self.file_list[idx]
        subject_id = visual_file.replace('.npy', '')
        motion_file = f"{subject_id}_motion.npy"

        visual_path = os.path.join(self.temp_dir, visual_file)
        motion_path = os.path.join(self.temp_dir, motion_file)

        # --- 2. 加载数据字典 ---
        # v_data 包含: 'local', 'global', 'timestamp'
        v_data = np.load(visual_path, allow_pickle=True).item()
        # m_data 包含: 'motion', 'physio'
        m_data = np.load(motion_path, allow_pickle=True).item()

        # --- 3. 提取各个流所需的原始数据 ---
        local_feat = v_data['local']  # [Seq, 512] -> 用于 Temporal 和 GNN
        global_feat = v_data['global']  # [Seq, 512] -> 用于 Temporal
        motion_feat = m_data['motion']  # [Seq, 6]   -> 用于 Motion 流
        physio_feat = m_data['physio']  # [Seq, 3] (x, y, t) -> 用于 Temporal 位置和 GNN 构图
        # 将 physio_feat 的第 3 列 (Index 2) 从原始秒数转换为 0-1 比例
        # 这样 (x, y, t) 全部锁死在 [0, 1] 之间
        physio_feat[:, 2] /= config.VIDEO_DURATION
        # 标签逻辑：假设前一半序号为健康(0)，后一半为患病(1)
        label = 1 if int(subject_id) > (config.NUM_SIMULATED_PEOPLE // 2) else 0

        # --- 4. 统一时序处理 (截断或填充) ---
        # 按照你要求的：即使有异常值点，我们也保留时间戳占位，统一到 512
        curr_len = local_feat.shape[0]

        if curr_len >= self.max_len:
            # 截断到 512
            local_feat = local_feat[:self.max_len]
            global_feat = global_feat[:self.max_len]
            motion_feat = motion_feat[:self.max_len]
            physio_feat = physio_feat[:self.max_len]
            # Padding Mask: 全部为 False (表示全是真实数据)
            mask = torch.zeros(self.max_len, dtype=torch.bool)
        else:
            # 填充 (Zero Padding) 到 512
            pad_len = self.max_len - curr_len

            local_feat = np.pad(local_feat, ((0, pad_len), (0, 0)))
            global_feat = np.pad(global_feat, ((0, pad_len), (0, 0)))
            motion_feat = np.pad(motion_feat, ((0, pad_len), (0, 0)))
            physio_feat = np.pad(physio_feat, ((0, pad_len), (0, 0)))

            # Mask: 前面有效位是 False，后面填充位是 True
            mask = torch.zeros(self.max_len, dtype=torch.bool)
            mask[curr_len:] = True

        # --- 5. 最终打包返回 ---
        # 返回一个大字典，方便主模型分发给三个流
        return {
            "temporal_input": {
                "local": torch.from_numpy(local_feat).float(),
                "global": torch.from_numpy(global_feat).float(),
                "physio": torch.from_numpy(physio_feat).float()
            },
            "motion_input": torch.from_numpy(motion_feat).float(),
            "gnn_input": {
                "local": torch.from_numpy(local_feat).float(),
                "coords": torch.from_numpy(physio_feat).float()
            },
            "mask": mask,  # 形状 [512]
            "label": torch.tensor(label, dtype=torch.long)
        }


def get_mstnet_loaders(batch_size=None):
    """
    便捷获取训练和验证 DataLoader 的接口
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE  # 使用 config.py 里的 64

    train_ds = MSTNetDataset(mode='train')
    val_ds = MSTNetDataset(mode='val')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS  #
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS  #
    )

    return train_loader, val_loader


# ================= 单元测试 (Unit Test) =================
if __name__ == "__main__":
    try:
        train_l, val_l = get_mstnet_loaders(batch_size=4)
        sample_batch = next(iter(train_l))

        print("\n✅ DataLoader 测试成功!")
        print(f"Batch Label Shape: {sample_batch['label'].shape}")
        print(f"Temporal Local Shape: {sample_batch['temporal_input']['local'].shape}")
        print(f"Motion Input Shape: {sample_batch['motion_input'].shape}")
        print(f"GNN Coords Shape: {sample_batch['gnn_input']['coords'].shape}")
        print(f"Mask Sample: {sample_batch['mask'][0][:10]}... (False代表有效)")
    except Exception as e:
        print(f"❌ 测试失败: {e}")