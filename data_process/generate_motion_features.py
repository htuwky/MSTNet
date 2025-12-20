import sys
import os
import glob
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import gc

# 引入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ================= 配置 =================
# 光流计算分辨率：320x180 足够捕捉运动趋势，且速度快、省内存
FLOW_W = 320
FLOW_H = 180
TEMP_SAVE_DIR = config.TEMP_FEATURE_DIR


# ========================================

def get_frame_idx(timestamp):
    """
    针对 23 FPS 且文件名从 1 开始的精确对齐逻辑
    """

    return int(timestamp * config.VIDEO_FPS) + 1



def precompute_frames_flow(frame_dir):
    """
    核心逻辑：只读一次帧文件夹！算出所有帧的光流存内存。
    """
    print(f"🎬 正在读取帧文件夹: {frame_dir}")

    # 1. 获取所有帧文件并按文件名排序 (确保 frame_01, frame_02 顺序正确)
    # 支持 png, jpg, jpeg
    frame_files = sorted(
        glob.glob(os.path.join(frame_dir, "*.png")) +
        glob.glob(os.path.join(frame_dir, "*.jpg"))
    )

    if not frame_files:
        raise ValueError(f"❌ 文件夹里没找到图片！请检查 config.FRAME_DIR: {frame_dir}")

    print(f"   发现 {len(frame_files)} 帧，开始预计算光流...")

    dense_flows = []  # 存每一帧的全图光流
    global_flows = []  # 存每一帧的背景光流

    prev_gray = None

    # 遍历每一帧图片
    for fpath in tqdm(frame_files, desc="Pre-computing Flow"):
        # 读取图片 (OpenCV 读取快)
        img = cv2.imread(fpath)
        if img is None: continue

        # 缩放 (加速+省内存) 并转灰度
        img_small = cv2.resize(img, (FLOW_W, FLOW_H))
        curr_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            # 第一帧没有前一帧，光流补0
            flow = np.zeros((FLOW_H, FLOW_W, 2), dtype=np.float32)
        else:
            # 计算光流 (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

        # 存入内存列表
        dense_flows.append(flow)  # [H, W, 2]
        global_flows.append(np.mean(flow, axis=(0, 1)))  # [2]

        prev_gray = curr_gray

    print(f"✅ 光流库构建完毕！内存中已有 {len(dense_flows)} 帧的数据。")
    return dense_flows, global_flows


def main():
    print("=" * 60)
    print(f"🚀 MSTNet 极速光流提取 (Frame Sequence Mode)")
    print("=" * 60)

    os.makedirs(TEMP_SAVE_DIR, exist_ok=True)

    # --- STEP 1: 只算一遍帧文件夹 ---
    try:
        # 内存里现在有了整个视频的光流数据
        video_flows, bg_flows = precompute_frames_flow(config.FRAME_DIR)
        total_frames = len(video_flows)
    except Exception as e:
        print(f"❌ 预处理失败: {e}")
        return

    # --- STEP 2: 400人排队查表 (极速) ---
    csv_files = glob.glob(os.path.join(config.CSV_DIR, '*.csv'))
    print(f"\n⚡ 开始为 {len(csv_files)} 个受试者生成数据...")

    for csv_path in tqdm(csv_files, desc="Matching Subjects"):
        subject_id = os.path.basename(csv_path).split('.')[0]
        save_path = os.path.join(TEMP_SAVE_DIR, f"{subject_id}_motion.npy")

        # 如果已有，跳过
        # if os.path.exists(save_path): continue
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['Gaze_X', 'Gaze_Y', 'Timestamp'])

            motion_list = []
            coords_list = []

            prev_x, prev_y = 0.5, 0.5

            for idx, row in df.iterrows():
                ts = row['Timestamp']
                gx, gy = row['Gaze_X'], row['Gaze_Y']

                # 1. 找对应帧号
                frame_idx = get_frame_idx(ts)

                # 越界保护 (防止 csv 时间比帧总数长)
                if frame_idx >= total_frames: frame_idx = total_frames - 1
                if frame_idx < 0: frame_idx = 0

                # 2. 查表 (Lookup) - 这一步是瞬间完成的
                # A. 拿背景光流
                u_glob, v_glob = bg_flows[frame_idx]

                # B. 拿局部光流 (根据注视点坐标去挖)
                ix = int(gx * FLOW_W)
                iy = int(gy * FLOW_H)
                # 坐标限制在 [0, W-1]
                ix = np.clip(ix, 0, FLOW_W - 1)
                iy = np.clip(iy, 0, FLOW_H - 1)

                u_loc, v_loc = video_flows[frame_idx][iy, ix]

                # 3. 算眼动速度 (像素级)
                if idx == 0:
                    vx, vy = 0.0, 0.0
                else:
                    vx = (gx - prev_x)
                    vy = (gy - prev_y)

                prev_x, prev_y = gx, gy

                # 4. 打包
                # Motion流需要这6个数
                motion_vec = np.array([u_loc, v_loc, u_glob, v_glob, vx, vy], dtype=np.float32)
                # GNN/Temporal流备份用的坐标
                coord_vec = np.array([gx, gy, ts], dtype=np.float32)

                motion_list.append(motion_vec)
                coords_list.append(coord_vec)

            # 保存文件
            if motion_list:
                data_dict = {
                    'motion': np.vstack(motion_list),
                    'physio': np.vstack(coords_list)
                }
                np.save(save_path, data_dict)

        except Exception as e:
            print(f"Error {subject_id}: {e}")
            continue

    print("\n✅ 搞定！400个 motion.npy 文件已生成。")


if __name__ == "__main__":
    main()