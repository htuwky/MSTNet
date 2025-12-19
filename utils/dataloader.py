import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import random

# 确保能导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MSTNetDataset(Dataset):
    def __init__(self, mode='train', split_ratio=0.8):
        """
        MSTNet 数据集加载器 (增加随机打乱逻辑)
        Args:
            mode: 'train' 或 'val'
            split_ratio: 训练集比例
        """
        self.temp_dir = config.TEMP_FEATURE_DIR
        self.max_len = config.MAX_SEQ_LEN

        # 1. 扫描所有主特征文件
        all_files = sorted([
            f for f in os.listdir(self.temp_dir)
            if f.endswith('.npy') and '_motion' not in f
        ])

        if not all_files:
            raise FileNotFoundError(f"❌ 在 {self.temp_dir} 没找到特征文件！")

        # --- 【关键修改：全局打乱】 ---
        # 使用 config 中的种子确保每次划分的结果一致
        random.seed(config.SEED)
        random.shuffle(all_files)

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
        visual_file = self.file_list[idx]
        subject_id = visual_file.replace('.npy', '')
        motion_file = f"{subject_id}_motion.npy"

        visual_path = os.path.join(self.temp_dir, visual_file)
        motion_path = os.path.join(self.temp_dir, motion_file)

        v_data = np.load(visual_path, allow_pickle=True).item()
        m_data = np.load(motion_path, allow_pickle=True).item()

        local_feat = v_data['local']
        global_feat = v_data['global']
        motion_feat = m_data['motion']
        physio_feat = m_data['physio']

        # 时间戳归一化
        physio_feat[:, 2] /= config.VIDEO_DURATION

        # 标签逻辑
        label = 1 if int(subject_id) > (config.NUM_SIMULATED_PEOPLE // 2) else 0

        # 统一时序处理 (截断或填充)
        curr_len = local_feat.shape[0]
        if curr_len >= self.max_len:
            local_feat = local_feat[:self.max_len]
            global_feat = global_feat[:self.max_len]
            motion_feat = motion_feat[:self.max_len]
            physio_feat = physio_feat[:self.max_len]
            mask = torch.zeros(self.max_len, dtype=torch.bool)
        else:
            pad_len = self.max_len - curr_len
            local_feat = np.pad(local_feat, ((0, pad_len), (0, 0)))
            global_feat = np.pad(global_feat, ((0, pad_len), (0, 0)))
            motion_feat = np.pad(motion_feat, ((0, pad_len), (0, 0)))
            physio_feat = np.pad(physio_feat, ((0, pad_len), (0, 0)))
            mask = torch.zeros(self.max_len, dtype=torch.bool)
            mask[curr_len:] = True

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
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long)
        }


def get_mstnet_loaders(batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # 初始化时会自动执行 shuffle 逻辑
    train_ds = MSTNetDataset(mode='train')
    val_ds = MSTNetDataset(mode='val')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # 训练过程中的 batch 打乱
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader