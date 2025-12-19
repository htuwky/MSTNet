import sys
import os
import glob
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# === 完全从 Config 读取配置 ===
BATCH_SIZE = config.EXTRACT_BATCH_SIZE  # 使用提取专用的 Batch
NUM_WORKERS = config.NUM_WORKERS
TEMP_SAVE_DIR = config.TEMP_FEATURE_DIR


# ==============================

def get_frame_idx(timestamp):
    return int(timestamp * config.VIDEO_FPS) + 1


class GazeDataset(Dataset):
    def __init__(self, df, frame_dir, preprocess, crop_size, video_w, video_h):
        # 过滤空值
        self.data = df.dropna(subset=['Gaze_X', 'Gaze_Y', 'Timestamp']).to_dict('records')
        self.frame_dir = frame_dir
        self.preprocess = preprocess
        self.crop_size = crop_size
        self.video_w = video_w
        self.video_h = video_h
        self.cache = {'idx': -1, 'img': None}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ts = row['Timestamp']
        target_frame_idx = get_frame_idx(ts)

        # 1. 读图
        current_img = self.cache['img']
        if self.cache['idx'] != target_frame_idx:
            frame_path = os.path.join(self.frame_dir, f"frame_{target_frame_idx:05d}.png")
            if os.path.exists(frame_path):
                try:
                    current_img = Image.open(frame_path).convert("RGB")
                    self.cache['idx'] = target_frame_idx
                    self.cache['img'] = current_img
                except:
                    if current_img is None: current_img = Image.new("RGB", (self.video_w, self.video_h), (0, 0, 0))
            else:
                if current_img is None: current_img = Image.new("RGB", (self.video_w, self.video_h), (0, 0, 0))

        # 2. 坐标
        w, h = current_img.size
        try:
            raw_x, raw_y = float(row['Gaze_X']), float(row['Gaze_Y'])
        except:
            raw_x, raw_y = 0.5, 0.5

        gx = int(raw_x * self.video_w)
        gy = int(raw_y * self.video_h)

        # 3. 裁剪 (Zero Padding)
        half = self.crop_size // 2
        if gx - half >= 0 and gy - half >= 0 and gx + half <= w and gy + half <= h:
            patch = current_img.crop((gx - half, gy - half, gx + half, gy + half))
        else:
            patch = Image.new("RGB", (self.crop_size, self.crop_size), (0, 0, 0))
            src_left = max(0, gx - half)
            src_top = max(0, gy - half)
            src_right = min(w, gx + half)
            src_bottom = min(h, gy + half)
            dst_left = max(0, -(gx - half))
            dst_top = max(0, -(gy - half))
            if src_right > src_left and src_bottom > src_top:
                crop_part = current_img.crop((src_left, src_top, src_right, src_bottom))
                patch.paste(crop_part, (dst_left, dst_top))

        return self.preprocess(patch), self.preprocess(current_img), ts


def main():
    print("=" * 60)
    print(f"🚀 MSTNet Feature Extraction (Fully Decoupled)")
    print("=" * 60)
    config.print_config()

    os.makedirs(TEMP_SAVE_DIR, exist_ok=True)

    device = config.DEVICE
    print(f"⏳ Loading CLIP ({config.CLIP_MODEL_NAME})...")
    model, preprocess = clip.load(config.CLIP_MODEL_NAME, device=device)
    model.eval()

    csv_files = glob.glob(os.path.join(config.CSV_DIR, '*.csv'))
    print(f"📂 Found {len(csv_files)} CSV files.")

    for csv_path in tqdm(csv_files, desc="Processing"):
        subject_id = os.path.basename(csv_path).split('.')[0]
        save_path = os.path.join(TEMP_SAVE_DIR, f"{subject_id}.npy")

        if os.path.exists(save_path): continue

        try:
            df = pd.read_csv(csv_path)
            dataset = GazeDataset(df, config.FRAME_DIR, preprocess, config.CROP_SIZE, config.VIDEO_W, config.VIDEO_H)
            if len(dataset) == 0: continue

            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                    pin_memory=False)
            local_list, global_list, timestamp_list = [], [], []

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    for b_local, b_global, b_ts in dataloader:
                        b_local = b_local.to(device)
                        b_global = b_global.to(device)

                        l = model.encode_image(b_local)
                        g = model.encode_image(b_global)

                        l /= l.norm(dim=-1, keepdim=True)
                        g /= g.norm(dim=-1, keepdim=True)

                        local_list.append(l.cpu().numpy().astype(np.float16))
                        global_list.append(g.cpu().numpy().astype(np.float16))
                        timestamp_list.append(b_ts.numpy().astype(np.float64))

            if local_list:
                data_dict = {
                    'local': np.vstack(local_list),
                    'global': np.vstack(global_list),
                    'timestamp': np.concatenate(timestamp_list)
                }
                np.save(save_path, data_dict)

            del dataset, dataloader, local_list, global_list, timestamp_list, data_dict, df
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error {subject_id}: {e}")
            continue

    print("✅ All features extracted to temp folder!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()